//! Index analysis: a validated [`Problem`] seen as a (batched) matrix product
//! over scatter vectors.
//!
//! Given the normalized roles of a [`Problem`], this module works out how to
//! view the contraction
//!
//! ```text
//! D[idx_D] = alpha * op_A(A[idx_A]) * op_B(B[idx_B]) + beta * op_C(C[idx_C])
//! ```
//!
//! as a (batched) matrix multiplication over scatter matrices.
//!
//! # Index classes
//!
//! Every distinct label is assigned to exactly one class:
//!
//! | class | in A | in B | in D | role |
//! |-------|------|------|------|------|
//! | `M`   | yes  | no   | yes  | free index of A, rows of the GEMM |
//! | `N`   | no   | yes  | yes  | free index of B, columns of the GEMM |
//! | `K`   | yes  | yes  | no   | contracted index |
//! | `H`   | yes  | yes  | yes  | Hadamard / batch index |
//!
//! Two further cases fold into `K` rather than needing special machinery:
//!
//! * a label present **only in A** (TAPP "isolated" index, i.e. a reduction
//!   `sum_i A[...,i,...]`) becomes a contraction index with `stride_B = 0`;
//! * likewise a label present only in B becomes one with `stride_A = 0`.
//!
//! Because the block-scatter machinery treats a zero stride as a perfectly
//! regular access pattern, this costs nothing and — unlike a pre-reduction
//! pass — needs no workspace. A label present only in the output (TAPP
//! "case 5", broadcasting) is rejected; TAPP does not require it.
//!
//! # Repeated labels
//!
//! A label repeated within a single tensor selects that tensor's diagonal.
//! Extents must agree and the strides are summed, after which the label is
//! treated as a single mode. This is applied per tensor before classification.
//!
//! # Ordering and folding
//!
//! Within a class the labels are ordered fastest-varying first, then adjacent
//! labels are merged whenever their extents and strides are compatible in
//! *every* operand (`s_next == s_prev * extent_prev`). Folding is what makes
//! an ordinary matrix multiply collapse to a single index per class — and
//! therefore to a plain GEMM with fully regular block scatter — and it
//! substantially raises the regular-block fraction for real contractions.
//!
//! The ordering heuristic is: class `M`, `N` and `H` are ordered by increasing
//! |stride| in `D`, class `K` by increasing |stride| in `A`. Rationale: the
//! output update is the one access that packing cannot hide, so `D` gets first
//! claim on contiguity; `K` only affects the packing of `A` and `B`. This is a
//! heuristic and a Phase 4 tuning knob.

use tcontract_kernel::scatter::{build_scatter, run_structure};

use super::config::PlanConfig;
use super::orientation::{Orient, RowBlock};
use crate::api::{OperandId, Problem, Result, RoleAxis, ShapeError};

///
/// Carrying all four strides together is what makes folding checkable: two
/// adjacent axes may be merged only when their strides are compatible in
/// *every* operand at once, so the test has to see them side by side. An axis
/// absent from an operand has stride 0 there, which is not a special case —
/// see the module docs on isolated indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Axis {
    /// Length of the axis after folding, i.e. the product of the extents that
    /// were merged into it.
    pub extent: i64,
    /// Stride in `A`, in elements; 0 if the axis does not appear in `A`.
    pub sa: i64,
    /// Stride in `B`.
    pub sb: i64,
    /// Stride in `C`.
    pub sc: i64,
    /// Stride in `D`.
    pub sd: i64,
}

/// Diagnostics about a plan, useful for benchmarking write-ups and for
/// dispatch heuristics.
///
/// The four dimensions are the *matrix* shape the contraction was reduced to,
/// after diagonals are collapsed, extent-1 axes are dropped and compatible
/// axes are folded — so they are what the engine works on rather than what the
/// caller wrote. Reading them is the cheapest way to see whether folding did
/// what you expected.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PlanStats {
    /// Rows of the matrix view: the product of the `M` extents.
    pub m: usize,
    /// Columns: the product of the `N` extents.
    pub n: usize,
    /// Contraction depth: the product of the `K` extents, including any
    /// isolated (reduction) indices folded in.
    pub k: usize,
    /// Number of independent matrix products, i.e. the product of the Hadamard
    /// extents. 1 when there are none.
    pub batch: usize,
    /// The folded `M` axes, fastest-varying in `D` first.
    pub m_axes: Vec<Axis>,
    /// The folded `N` axes, fastest-varying in `D` first.
    pub n_axes: Vec<Axis>,
    /// The folded `K` axes, fastest-varying in `A` first.
    pub k_axes: Vec<Axis>,
    /// The folded Hadamard axes.
    pub h_axes: Vec<Axis>,
    /// True when `M`, `N` and `K` each folded to at most one axis **and there
    /// are no Hadamard axes** — i.e. the contraction is exactly *one* GEMM on
    /// strided matrices.
    ///
    /// A batch index therefore makes this false even when every other class
    /// folded perfectly: the work is then a sequence of GEMMs rather than one,
    /// which is the distinction the flag exists to draw. Check `batch == 1`
    /// alongside it if what you want is "no gather anywhere".
    pub is_pure_gemm: bool,
}

impl PlanStats {
    /// Multiply-accumulate count (batch * m * n * k).
    ///
    /// Multiply by `Element::FLOPS_PER_MAC` for a flop count. This
    /// counts *useful* work — the padding the engine does on edge blocks is
    /// deliberately not included, so throughput computed from it is comparable
    /// against another library's.
    ///
    /// # Panics
    ///
    /// In a debug build, if the product exceeds `u64::MAX`; in release it wraps.
    /// Reaching that needs all four dimensions near `10^5`, which is only a few
    /// megabytes of scatter vectors and therefore a plan that builds fine even
    /// though nothing could execute it — so it is a real edge, not an
    /// unreachable one.
    pub fn macs(&self) -> u64 {
        (self.batch as u64) * (self.m as u64) * (self.n as u64) * (self.k as u64)
    }
}

/// The packed plan: everything that depends only on shapes, strides and roles,
/// and not on the element type or the data pointers.
///
/// Scatter vectors are element-type independent. The kernel family, the
/// blocking and the partition live in the resolved configuration the owning
/// [`Plan`](super::Plan) freezes next to this record; the orientation and
/// row-block requests are read from [`PlanConfig`] once, here.
#[derive(Clone, Debug)]
pub struct PackedPlan {
    pub(crate) a_m: Vec<i64>,
    pub(crate) a_k: Vec<i64>,
    pub(crate) b_k: Vec<i64>,
    pub(crate) b_n: Vec<i64>,
    pub(crate) c_m: Vec<i64>,
    pub(crate) c_n: Vec<i64>,
    pub(crate) d_m: Vec<i64>,
    pub(crate) d_n: Vec<i64>,
    pub(crate) h_a: Vec<i64>,
    pub(crate) h_b: Vec<i64>,
    pub(crate) h_c: Vec<i64>,
    pub(crate) h_d: Vec<i64>,
    /// Run structure of the output's `M` and `N` scatters, `(len, stride)`, or
    /// `(len, 0)` when there is none. Cached because
    /// [`PackedPlan::transposes_gemm`] and [`PackedPlan::row_block`] both turn
    /// on it and are called several times per execution, while the scatters
    /// themselves never change.
    pub(crate) d_m_run: (usize, i64),
    pub(crate) d_n_run: (usize, i64),
    pub(crate) conj_a: bool,
    pub(crate) conj_b: bool,
    pub(crate) conj_c: bool,
    pub(crate) conj_d: bool,
    /// Orientation request (see [`PlanConfig::orientation`]).
    pub(crate) orient: Orient,
    /// Row-block request (see [`PlanConfig::row_block`]).
    pub(crate) row_block_mode: RowBlock,
    /// Forced L3 domain count (see [`CacheModel::l3_domains`](super::CacheModel)).
    pub(crate) l3_domains: Option<usize>,
    /// The matrix shape and folded axes this plan reduced to: the answer to
    /// "what did the index analysis actually decide".
    pub stats: PlanStats,
}

fn axis_of(r: &RoleAxis) -> Axis {
    Axis {
        extent: r.extent() as i64,
        sa: r.stride(OperandId::A) as i64,
        sb: r.stride(OperandId::B) as i64,
        sc: r.stride(OperandId::C) as i64,
        sd: r.stride(OperandId::D) as i64,
    }
}

/// The axes of one role: extent-1 axes dropped (their strides are
/// unconstrained), the rest sorted by `key` and folded.
fn role_axes(role: &[RoleAxis], key: impl Fn(&Axis) -> (u64, u64)) -> Vec<Axis> {
    let mut axes: Vec<Axis> = role.iter().map(axis_of).filter(|a| a.extent != 1).collect();
    axes.sort_by_key(|x| key(x));
    fold_axes(axes)
}

impl PackedPlan {
    /// Analyse a validated problem.
    ///
    /// This is where every shape-dependent decision of the packed route is
    /// made -- ordering, folding and the scatter vectors -- so it is the work
    /// a [`Plan`](super::Plan) hoists out of a loop. The [`stats`](Self::stats)
    /// field reports what it concluded.
    pub(crate) fn from_problem(p: &Problem, cfg: &PlanConfig) -> Result<PackedPlan> {
        let roles = p.roles();
        let m_ax = role_axes(roles.m(), |x| (x.sd.unsigned_abs(), x.sa.unsigned_abs()));
        let n_ax = role_axes(roles.n(), |x| (x.sd.unsigned_abs(), x.sb.unsigned_abs()));
        let k_ax = role_axes(roles.k(), |x| (x.sa.unsigned_abs(), x.sb.unsigned_abs()));
        let h_ax = role_axes(roles.h(), |x| (x.sd.unsigned_abs(), x.sa.unsigned_abs()));

        for axes in [&m_ax, &n_ax, &k_ax, &h_ax] {
            // A scatter vector has one entry per element of its role, so the
            // product must be addressable before anything is allocated.
            let len = axes
                .iter()
                .try_fold(1usize, |acc, x| acc.checked_mul(x.extent as usize))
                .filter(|&l| {
                    l.checked_mul(core::mem::size_of::<i64>())
                        .is_some_and(|b| b <= isize::MAX as usize)
                });
            if len.is_none() {
                return Err(ShapeError::Overflow {
                    what: "role extent product",
                }
                .into());
            }
        }

        // An operand with a zero extent holds no element, so no numerical access
        // can reach it, and its per-role scatter vectors are unreachable values.
        // A caller may legitimately leave an empty operand's strides at an
        // unreachable magnitude (a view over an empty slice), which would
        // overflow the scatter odometer below. Zero that operand's strides
        // instead; the vector lengths, and so the reported stats, are unchanged.
        let empty = |groups: [&[Axis]; 3]| {
            groups
                .iter()
                .any(|group| group.iter().any(|axis| axis.extent == 0))
        };
        let a_empty = empty([&m_ax, &k_ax, &h_ax]);
        let b_empty = empty([&k_ax, &n_ax, &h_ax]);
        let c_empty = empty([&m_ax, &n_ax, &h_ax]);
        let d_empty = empty([&m_ax, &n_ax, &h_ax]);
        let stride = |is_empty: bool, value: i64| if is_empty { 0 } else { value };

        let a_m = build_scatter_for(&m_ax, |x| stride(a_empty, x.sa));
        let a_k = build_scatter_for(&k_ax, |x| stride(a_empty, x.sa));
        let b_k = build_scatter_for(&k_ax, |x| stride(b_empty, x.sb));
        let b_n = build_scatter_for(&n_ax, |x| stride(b_empty, x.sb));
        let c_m = build_scatter_for(&m_ax, |x| stride(c_empty, x.sc));
        let c_n = build_scatter_for(&n_ax, |x| stride(c_empty, x.sc));
        let d_m = build_scatter_for(&m_ax, |x| stride(d_empty, x.sd));
        let d_n = build_scatter_for(&n_ax, |x| stride(d_empty, x.sd));
        let h_a = build_scatter_for(&h_ax, |x| stride(a_empty, x.sa));
        let h_b = build_scatter_for(&h_ax, |x| stride(b_empty, x.sb));
        let h_c = build_scatter_for(&h_ax, |x| stride(c_empty, x.sc));
        let h_d = build_scatter_for(&h_ax, |x| stride(d_empty, x.sd));

        let stats = PlanStats {
            m: d_m.len(),
            n: d_n.len(),
            k: a_k.len(),
            batch: h_d.len(),
            is_pure_gemm: m_ax.len() <= 1 && n_ax.len() <= 1 && k_ax.len() <= 1 && h_ax.is_empty(),
            m_axes: m_ax,
            n_axes: n_ax,
            k_axes: k_ax,
            h_axes: h_ax,
        };
        let d_m_run = run_structure(&d_m).unwrap_or((d_m.len(), 0));
        let d_n_run = run_structure(&d_n).unwrap_or((d_n.len(), 0));
        Ok(PackedPlan {
            a_m,
            a_k,
            b_k,
            b_n,
            c_m,
            c_n,
            d_m,
            d_n,
            h_a,
            h_b,
            h_c,
            h_d,
            d_m_run,
            d_n_run,
            conj_a: p.a().op().is_conj(),
            conj_b: p.b().op().is_conj(),
            conj_c: p.op_c().is_conj(),
            conj_d: p.d().op().is_conj(),
            orient: cfg.orientation,
            row_block_mode: cfg.row_block,
            l3_domains: cfg.cache_model.l3_domains,
            stats,
        })
    }

    /// `true` when the contraction produces no output elements.
    pub fn is_empty(&self) -> bool {
        self.stats.m == 0 || self.stats.n == 0 || self.stats.batch == 0
    }

    /// `true` when the contraction dimension is empty, so `D = beta * C`.
    pub fn has_empty_contraction(&self) -> bool {
        self.stats.k == 0
    }
}

/// Merge adjacent axes whose strides are compatible in every operand.
///
/// The compatibility test multiplies a stride by the preceding extent, and the
/// merged extent multiplies two extents. An empty operand may legitimately carry
/// an unreachable-magnitude stride on its empty axis, so both are checked: an
/// overflow means the axes are simply not foldable, which leaves them separate
/// and keeps the stride normalization for empty operands in charge of the
/// scatter vectors. The planner's own role-size guards bound a real layout's
/// role products, so refusing to fold on overflow loses no real fold.
fn fold_axes(axes: Vec<Axis>) -> Vec<Axis> {
    let mut out: Vec<Axis> = Vec::with_capacity(axes.len());
    for ax in axes {
        if let Some(p) = out.last_mut() {
            let foldable =
                p.sa.checked_mul(p.extent)
                    .is_some_and(|expected| expected == ax.sa)
                    && p.sb
                        .checked_mul(p.extent)
                        .is_some_and(|expected| expected == ax.sb)
                    && p.sc
                        .checked_mul(p.extent)
                        .is_some_and(|expected| expected == ax.sc)
                    && p.sd
                        .checked_mul(p.extent)
                        .is_some_and(|expected| expected == ax.sd);
            if foldable {
                if let Some(extent) = p.extent.checked_mul(ax.extent) {
                    p.extent = extent;
                    continue;
                }
            }
        }
        out.push(ax);
    }
    out
}

fn build_scatter_for(axes: &[Axis], pick: impl Fn(&Axis) -> i64) -> Vec<i64> {
    let extents: Vec<i64> = axes.iter().map(|a| a.extent).collect();
    let strides: Vec<i64> = axes.iter().map(pick).collect();
    build_scatter(&extents, &strides)
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{build, lay, laying};

    #[test]
    fn empty_operand_with_unreachable_strides_still_plans() {
        // A[i,k] over an empty slice with a stride a caller may leave
        // unreachable, B[k,j] empty, D[i,j]: an empty contraction is a valid
        // problem, and planning must not overflow while building the scatter
        // vectors of the empty operands.
        let a = laying(&[2, 0], &[isize::MAX as i64, 1]);
        let b = lay(&[0, 3]);
        let d = lay(&[2, 3]);
        let p = build(&a, &[0, 1], &b, &[1, 2], &d, &[0, 2]);
        assert_eq!((p.stats.m, p.stats.n, p.stats.k), (2, 3, 0));
    }

    #[test]
    fn multi_axis_empty_view_with_unreachable_stride_still_plans() {
        // A's empty axis is last, so folding would test `isize::MAX * 2` on the
        // first M axis before the empty-operand stride normalization runs.
        let a = laying(&[2, 2, 0], &[isize::MAX as i64, 1, 1]);
        let b = lay(&[0, 3]);
        let d = lay(&[2, 2, 3]);
        let p = build(&a, &[0, 1, 3], &b, &[3, 2], &d, &[0, 1, 2]);
        assert_eq!((p.stats.m, p.stats.n, p.stats.k), (4, 3, 0));
    }

    #[test]
    fn empty_output_with_unreachable_strides_still_plans() {
        // The output's empty axis carries the unreachable stride, so the write
        // side has to survive the same preparation.
        let a = lay(&[2, 0]);
        let b = lay(&[0, 0]);
        let d = laying(&[2, 0], &[1, isize::MAX as i64]);
        let p = build(&a, &[0, 1], &b, &[1, 2], &d, &[0, 2]);
        assert_eq!((p.stats.m, p.stats.n, p.stats.k), (2, 0, 0));
    }

    #[test]
    fn plain_matmul_folds_to_pure_gemm() {
        // C[i,j] = A[i,k] B[k,j], all column-major.
        let a = lay(&[4, 5]);
        let b = lay(&[5, 6]);
        let d = lay(&[4, 6]);
        let p = build(&a, &[0, 2], &b, &[2, 1], &d, &[0, 1]);
        assert_eq!(
            (p.stats.m, p.stats.n, p.stats.k, p.stats.batch),
            (4, 6, 5, 1)
        );
        assert!(p.stats.is_pure_gemm);
    }

    #[test]
    fn adjacent_free_indices_fold() {
        // D[a,b,j] = A[a,b,k] B[k,j] with a,b adjacent in both A and D.
        let a = lay(&[4, 3, 5]);
        let b = lay(&[5, 6]);
        let d = lay(&[4, 3, 6]);
        let p = build(&a, &[0, 1, 3], &b, &[3, 2], &d, &[0, 1, 2]);
        assert_eq!(p.stats.m_axes.len(), 1, "a and b should fold into one axis");
        assert_eq!(p.stats.m, 12);
        assert!(p.stats.is_pure_gemm);
    }

    #[test]
    fn hadamard_index_recognised() {
        // D[b,i,j] = A[b,i,k] B[b,k,j]
        let a = lay(&[2, 4, 5]);
        let b = lay(&[2, 5, 6]);
        let d = lay(&[2, 4, 6]);
        let p = build(&a, &[9, 0, 3], &b, &[9, 3, 1], &d, &[9, 0, 1]);
        assert_eq!(p.stats.batch, 2);
        assert_eq!((p.stats.m, p.stats.n, p.stats.k), (4, 6, 5));
    }

    #[test]
    fn isolated_index_becomes_zero_stride_contraction() {
        // D[i,j] = sum_{k,r} A[i,k,r] B[k,j]: r is isolated in A.
        let a = lay(&[4, 5, 3]);
        let b = lay(&[5, 6]);
        let d = lay(&[4, 6]);
        let p = build(&a, &[0, 2, 7], &b, &[2, 1], &d, &[0, 1]);
        assert_eq!(p.stats.k, 15, "contraction extent is 5*3");
        // B must re-read the same column for each r.
        assert_eq!(p.b_k.len(), 15);
        assert_eq!(&p.b_k[..6], &[0, 1, 2, 3, 4, 0]);
    }

    #[test]
    fn repeated_label_takes_diagonal() {
        // D[j] = sum_i A[i,i] B[i,j] -- A's trace-like diagonal.
        let a = lay(&[4, 4]);
        let b = lay(&[4, 6]);
        let d = lay(&[6]);
        let p = build(&a, &[0, 0], &b, &[0, 1], &d, &[1]);
        assert_eq!(p.stats.k, 4);
        // stride 1 + stride 4 = 5
        assert_eq!(p.a_k, vec![0, 5, 10, 15]);
    }
}
