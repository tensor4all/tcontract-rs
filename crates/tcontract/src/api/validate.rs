//! The one lowering of operand metadata into a validated role description.
//!
//! Metadata validation happens here, when a [`Problem`](crate::api::Problem)
//! is built, before any scatter allocation, kernel selection or zero-size
//! shortcut:
//!
//! 1. label counts, repeated-label extent agreement and checked stride sums
//!    (a repeated label selects a diagonal);
//! 2. classification of the unique labels into M, N, K and batch roles
//!    (a label isolated in A or B is a K axis whose stride is zero in the
//!    other input; an output-only label is unsupported);
//! 3. checked address ranges (byte spans) and extent products;
//! 4. output injectivity on the reduced logical D domain (conservative);
//! 5. whether a separate C addresses D identically.
//!
//! Pointer-dependent checks (layout match, slice bounds, overlap of the actual
//! addresses) happen on every execution; see `crate::api::preflight`.

use crate::api::error::{AliasError, ConfigError, OperandId, Result, ShapeError, Unsupported};
use crate::api::labels::Labels;
use crate::api::problem::{CSpec, DType, LayoutSpec, OperandSpec, RoleAxis, Roles, Span};

/// What lowering concluded; read through `Problem`'s accessors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Lowered {
    pub(crate) roles: Roles,
    /// A, B, C, D (C is D's for the modes without a separate C).
    pub(crate) spans: [Option<Span>; 4],
    pub(crate) c_matches_d: bool,
    pub(crate) k_empty: bool,
    pub(crate) out_empty: bool,
}

/// One label of one operand after repeated labels collapsed onto a diagonal.
#[derive(Clone, Copy, Debug)]
struct Mode {
    label: i64,
    extent: usize,
    stride: isize,
}

fn overflow(what: &'static str) -> crate::api::error::Error {
    ShapeError::Overflow { what }.into()
}

/// Collapse repeated labels of one operand onto their diagonal.
fn reduce(operand: OperandId, layout: &LayoutSpec, labels: &[i64]) -> Result<Vec<Mode>> {
    if layout.rank() != labels.len() {
        return Err(ShapeError::LabelCount {
            operand,
            modes: layout.rank(),
            labels: labels.len(),
        }
        .into());
    }
    let mut out: Vec<Mode> = Vec::with_capacity(labels.len());
    let mut product: usize = 1;
    for (k, &label) in labels.iter().enumerate() {
        let (extent, stride) = (layout.dims()[k], layout.strides()[k]);
        match out.iter_mut().find(|m| m.label == label) {
            Some(m) => {
                if m.extent != extent {
                    return Err(ShapeError::ExtentMismatch {
                        label,
                        expected: m.extent,
                        found: extent,
                    }
                    .into());
                }
                m.stride = m
                    .stride
                    .checked_add(stride)
                    .ok_or_else(|| overflow("repeated-label stride sum"))?;
            }
            None => {
                // Every scatter vector the plan builds is as long as the
                // product of the extents of some subset of one operand's axes,
                // so bounding each operand's whole product bounds them all.
                product = product
                    .checked_mul(extent)
                    .filter(|&p| i64::try_from(p).is_ok())
                    .ok_or_else(|| overflow("extent product"))?;
                out.push(Mode {
                    label,
                    extent,
                    stride,
                });
            }
        }
    }
    Ok(out)
}

/// The addressed element range of an operand over its *original* axes, checked
/// so that every byte offset fits `isize`.
fn span_of(dtype: DType, layout: &LayoutSpec) -> Result<Option<Span>> {
    let elem = dtype.size() as i128;
    let (mut lo, mut hi) = (layout.offset() as i128, layout.offset() as i128);
    for (&e, &s) in layout.dims().iter().zip(layout.strides()) {
        if e == 0 {
            return Ok(None);
        }
        // INVARIANT: |e - 1| < 2^64 and |s| <= 2^63, so each reach and the
        // running sums over at most 2^31 axes stay inside i128.
        let reach = (e as i128 - 1) * s as i128;
        if reach < 0 {
            lo += reach;
        } else {
            hi += reach;
        }
    }
    let limit = isize::MAX as i128;
    if lo.abs() * elem > limit || (hi + 1) * elem > limit {
        return Err(overflow("address range"));
    }
    Ok(Some(Span { lo, hi }))
}

/// Whether distinct index tuples of the reduced D address distinct elements:
/// after sorting by `|stride|`, each stride must clear the reach of the smaller
/// ones. Conservative (a rejection is not a proof of aliasing).
fn injective(modes: &[Mode]) -> bool {
    if modes.iter().any(|m| m.extent == 0) {
        return true;
    }
    let axes = modes
        .iter()
        .filter(|m| m.extent > 1)
        .map(|m| (m.extent as i128, (m.stride as i128).abs()));
    // A stack buffer for the usual ranks: planning allocates nothing here.
    const INLINE: usize = 16;
    if modes.len() <= INLINE {
        let mut buf = [(0i128, 0i128); INLINE];
        let mut n = 0;
        for a in axes {
            buf[n] = a;
            n += 1;
        }
        chained(&mut buf[..n])
    } else {
        chained(&mut axes.collect::<Vec<_>>())
    }
}

/// Whether `(extent, |stride|)` axes, sorted by stride, each start at or past
/// the reach of the ones below them.
fn chained(axes: &mut [(i128, i128)]) -> bool {
    axes.sort_by_key(|a| a.1);
    let mut reach = 1i128;
    for &(e, s) in &*axes {
        if s < reach {
            return false;
        }
        reach = s.saturating_mul(e);
    }
    true
}

/// Whether distinct indices of a layout address distinct elements: the
/// conservative sorted-stride test the output check uses. An empty layout is
/// injective.
pub fn is_injective_layout(dims: &[usize], strides: &[isize]) -> bool {
    let modes: Vec<Mode> = dims
        .iter()
        .zip(strides)
        .map(|(&extent, &stride)| Mode {
            label: 0,
            extent,
            stride,
        })
        .collect();
    injective(&modes)
}

#[derive(Clone, Copy, Debug)]
struct Label {
    id: i64,
    extent: usize,
    strides: [isize; 4],
    in_a: bool,
    in_b: bool,
    in_c: bool,
    in_d: bool,
}

fn merge(table: &mut Vec<Label>, modes: &[Mode], slot: OperandId) -> Result<()> {
    for m in modes {
        let l = match table.iter_mut().find(|l| l.id == m.label) {
            Some(l) => {
                if l.extent != m.extent {
                    return Err(ShapeError::ExtentMismatch {
                        label: m.label,
                        expected: l.extent,
                        found: m.extent,
                    }
                    .into());
                }
                l
            }
            None => {
                table.push(Label {
                    id: m.label,
                    extent: m.extent,
                    strides: [0; 4],
                    in_a: false,
                    in_b: false,
                    in_c: false,
                    in_d: false,
                });
                table.last_mut().expect("just pushed")
            }
        };
        l.strides[slot as usize] = m.stride;
        match slot {
            OperandId::A => l.in_a = true,
            OperandId::B => l.in_b = true,
            OperandId::C => l.in_c = true,
            OperandId::D => l.in_d = true,
        }
    }
    Ok(())
}

pub(crate) fn lower(
    dtype: DType,
    a: &OperandSpec,
    b: &OperandSpec,
    c: &CSpec,
    d: &OperandSpec,
    labels: &Labels,
) -> Result<Lowered> {
    // The C mode and the C labels go together.
    if matches!(c, CSpec::Separate(_)) != labels.c().is_some() {
        return Err(ConfigError::CLabels.into());
    }
    let ra = reduce(OperandId::A, a.layout(), labels.a())?;
    let rb = reduce(OperandId::B, b.layout(), labels.b())?;
    let rd = reduce(OperandId::D, d.layout(), labels.d())?;
    let rc_own;
    let (rc, c_layout) = match (c, labels.c()) {
        (CSpec::Separate(spec), Some(lc)) => {
            rc_own = reduce(OperandId::C, spec.layout(), lc)?;
            (&rc_own[..], None)
        }
        // No separate C: mirror D so the C strides are well formed; they are
        // never read at beta zero and equal D's otherwise.
        _ => (&rd[..], Some(d.layout())),
    };

    let mut table: Vec<Label> = Vec::with_capacity(ra.len() + rb.len() + rd.len());
    merge(&mut table, &ra, OperandId::A)?;
    merge(&mut table, &rb, OperandId::B)?;
    merge(&mut table, rc, OperandId::C)?;
    merge(&mut table, &rd, OperandId::D)?;

    // C and D must describe the same label set.
    if table.iter().any(|l| l.in_c != l.in_d) {
        return Err(ShapeError::OutputLabelMismatch.into());
    }

    let mut roles = Roles::default();
    for l in &table {
        let axis = RoleAxis {
            label: l.id,
            extent: l.extent,
            strides: l.strides,
            in_a: l.in_a,
            in_b: l.in_b,
        };
        match (l.in_a, l.in_b, l.in_d) {
            (true, true, true) => roles.h.push(axis),
            (true, false, true) => roles.m.push(axis),
            (false, true, true) => roles.n.push(axis),
            // Contracted, or isolated in A or in B (a reduction).
            (true, _, false) | (false, true, false) => roles.k.push(axis),
            (false, false, true) => {
                return Err(Unsupported::OutputOnlyLabel { label: l.id }.into());
            }
            // A label in C alone was rejected above as an output mismatch, and
            // every label is created by merging some operand.
            (false, false, false) => unreachable!("label present in no operand"),
        }
    }

    let spans = [
        span_of(dtype, a.layout())?,
        span_of(dtype, b.layout())?,
        span_of(
            dtype,
            c_layout.unwrap_or_else(|| match c {
                CSpec::Separate(s) => s.layout(),
                _ => d.layout(),
            }),
        )?,
        span_of(dtype, d.layout())?,
    ];

    if !injective(&rd) {
        return Err(AliasError::OutputNotInjective.into());
    }

    // C addresses D identically when every label has the same summed stride and
    // the logical offsets agree. Plan-time proof for `C == D`.
    let c_matches_d = match c {
        CSpec::Separate(spec) => {
            spec.layout().offset() == d.layout().offset()
                && rd.iter().all(|m| {
                    rc.iter()
                        .any(|x| x.label == m.label && x.stride == m.stride)
                })
        }
        _ => true,
    };

    let k_empty = roles.k.iter().any(|x| x.extent == 0);
    let out_empty = roles
        .m
        .iter()
        .chain(&roles.n)
        .chain(&roles.h)
        .any(|x| x.extent == 0);
    Ok(Lowered {
        roles,
        spans,
        c_matches_d,
        k_empty,
        out_empty,
    })
}
