//! The validated metadata description of one contraction.
//!
//! [`Problem`] describes
//! `D = op_D(alpha * sum_K(op_A(A) * op_B(B)) + beta * op_C(C))`: operand
//! layouts, the C mode, and the axes grouped by role. It holds no tensor
//! payload, pointer, executor, alpha or beta. It is built through
//! [`Problem::from_labels`] or [`Problem::from_dot_general`], which lower to
//! the same validated representation; fields are private, so a validated
//! problem cannot be forged.

use crate::api::dot_general::DotGeneral;
use crate::api::error::{Error, OperandId, Result, ShapeError};
use crate::api::labels::Labels;
use crate::api::validate::{self, Lowered};

/// The storage and accumulation type of every operand of a problem.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    /// `f32`.
    F32,
    /// `f64`.
    F64,
    /// Interleaved single-precision complex.
    C32,
    /// Interleaved double-precision complex.
    C64,
}

impl DType {
    /// Bytes per element.
    pub const fn size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F64 | DType::C32 => 8,
            DType::C64 => 16,
        }
    }

    /// Whether the type is complex.
    pub const fn is_complex(self) -> bool {
        matches!(self, DType::C32 | DType::C64)
    }

    /// The type's name (`"f32"`, `"f64"`, `"c32"`, `"c64"`).
    pub const fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::C32 => "c32",
            DType::C64 => "c64",
        }
    }
}

/// Extents, signed element strides and a logical offset of one operand.
///
/// Strides may be negative or zero (a broadcast axis). The offset is the
/// element index, relative to the start of the backing storage, of the index
/// tuple `(0, .., 0)`.
///
/// # Examples
///
/// ```
/// use tcontract::api::LayoutSpec;
/// let l = LayoutSpec::new(&[2, 3], &[1, 2], 0).unwrap();
/// assert_eq!(l.rank(), 2);
/// assert!(LayoutSpec::new(&[2, 3], &[1], 0).is_err());
/// assert!(LayoutSpec::from_signed(&[2, -3], &[1, 2], 0).is_err());
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LayoutSpec {
    dims: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
}

impl LayoutSpec {
    /// A layout from extents, strides and an offset.
    ///
    /// # Errors
    ///
    /// [`ShapeError::LayoutRank`] when the lengths differ.
    pub fn new(dims: &[usize], strides: &[isize], offset: isize) -> Result<Self> {
        if dims.len() != strides.len() {
            return Err(ShapeError::LayoutRank {
                extents: dims.len(),
                strides: strides.len(),
            }
            .into());
        }
        Ok(Self {
            dims: dims.to_vec(),
            strides: strides.to_vec(),
            offset,
        })
    }

    /// A layout from signed 64-bit values, as a C caller supplies them.
    ///
    /// # Errors
    ///
    /// [`ShapeError::LayoutRank`], [`ShapeError::NegativeExtent`], or
    /// [`ShapeError::Overflow`] when a value does not fit this platform's
    /// `usize`/`isize`.
    pub fn from_signed(dims: &[i64], strides: &[i64], offset: i64) -> Result<Self> {
        if dims.len() != strides.len() {
            return Err(ShapeError::LayoutRank {
                extents: dims.len(),
                strides: strides.len(),
            }
            .into());
        }
        let mut d = Vec::with_capacity(dims.len());
        for (axis, &e) in dims.iter().enumerate() {
            if e < 0 {
                return Err(ShapeError::NegativeExtent { axis, extent: e }.into());
            }
            d.push(usize::try_from(e).map_err(|_| ShapeError::Overflow { what: "extent" })?);
        }
        let s = strides
            .iter()
            .map(|&x| isize::try_from(x).map_err(|_| ShapeError::Overflow { what: "stride" }))
            .collect::<core::result::Result<Vec<_>, _>>()?;
        let offset =
            isize::try_from(offset).map_err(|_| ShapeError::Overflow { what: "offset" })?;
        Ok(Self {
            dims: d,
            strides: s,
            offset,
        })
    }

    /// Extent of each axis.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Signed element stride of each axis.
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// The logical offset.
    pub fn offset(&self) -> isize {
        self.offset
    }

    /// Number of axes.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Whether a view with these extents, strides and offset has exactly this
    /// layout.
    pub fn matches(&self, dims: &[usize], strides: &[isize], offset: isize) -> bool {
        self.dims == dims && self.strides == strides && self.offset == offset
    }
}

/// The element-wise operation applied to an operand.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Op {
    /// Use the elements as stored.
    #[default]
    Identity,
    /// Use their complex conjugates (no effect for real types).
    Conjugate,
}

impl Op {
    /// Whether the operation conjugates.
    pub fn is_conj(self) -> bool {
        matches!(self, Op::Conjugate)
    }
}

/// A layout plus an element-wise operation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct OperandSpec {
    layout: LayoutSpec,
    op: Op,
}

impl OperandSpec {
    /// An operand with [`Op::Identity`].
    pub fn new(layout: LayoutSpec) -> Self {
        Self {
            layout,
            op: Op::Identity,
        }
    }

    /// The same operand with `op`.
    #[must_use]
    pub fn with_op(mut self, op: Op) -> Self {
        self.op = op;
        self
    }

    /// The same operand, conjugated.
    #[must_use]
    pub fn conj(self) -> Self {
        self.with_op(Op::Conjugate)
    }

    /// The layout.
    pub fn layout(&self) -> &LayoutSpec {
        &self.layout
    }

    /// The element-wise operation.
    pub fn op(&self) -> Op {
        self.op
    }
}

/// How the `beta * op_C(C)` term is supplied.
///
/// Omitting C never implicitly turns an overwrite into an accumulation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum CSpec {
    /// No C term: D is overwritten (beta is zero).
    Absent,
    /// Accumulate from D itself, with `op_C` applied to the old D.
    Output(Op),
    /// Accumulate from a separately described C of the same labels as D.
    Separate(OperandSpec),
}

/// One reduced logical axis: an extent and its signed strides.
///
/// Strides are `[A, B, C, D]`. An operand that does not carry the axis has
/// stride zero (so an isolated reduction is a `K` axis with stride zero in
/// the other input); [`in_a`](Self::in_a)/[`in_b`](Self::in_b) tell a missing
/// operand from a genuine zero stride.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoleAxis {
    pub(crate) label: i64,
    pub(crate) extent: usize,
    pub(crate) strides: [isize; 4],
    pub(crate) in_a: bool,
    pub(crate) in_b: bool,
}

impl RoleAxis {
    /// The label the axis came from.
    pub fn label(&self) -> i64 {
        self.label
    }

    /// The extent.
    pub fn extent(&self) -> usize {
        self.extent
    }

    /// The (diagonal-summed) stride in `operand`; zero when absent.
    pub fn stride(&self, operand: OperandId) -> isize {
        self.strides[operand as usize]
    }

    /// Whether A carries the axis.
    pub fn in_a(&self) -> bool {
        self.in_a
    }

    /// Whether B carries the axis.
    pub fn in_b(&self) -> bool {
        self.in_b
    }
}

/// The reduced axes grouped by role.
///
/// | Role | Present in |
/// |---|---|
/// | batch (`h`) | A, B, D (and C) |
/// | M | A, C, D |
/// | N | B, C, D |
/// | K | A and/or B, absent from C/D; a missing operand has stride zero |
///
/// Axes are in first-appearance order; ordering, folding and orientation are
/// decisions of plan construction, not of the problem.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Roles {
    pub(crate) m: Vec<RoleAxis>,
    pub(crate) n: Vec<RoleAxis>,
    pub(crate) k: Vec<RoleAxis>,
    pub(crate) h: Vec<RoleAxis>,
}

impl Roles {
    /// Axes of A, C and D.
    pub fn m(&self) -> &[RoleAxis] {
        &self.m
    }

    /// Axes of B, C and D.
    pub fn n(&self) -> &[RoleAxis] {
        &self.n
    }

    /// Contracted (and isolated, reduced) axes.
    pub fn k(&self) -> &[RoleAxis] {
        &self.k
    }

    /// Batch axes.
    pub fn h(&self) -> &[RoleAxis] {
        &self.h
    }
}

/// The element offsets an operand can address, inclusive, relative to the
/// start of its storage (the logical offset included).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub(crate) lo: i128,
    pub(crate) hi: i128,
}

impl Span {
    /// Lowest addressed element offset.
    pub fn lo(&self) -> i128 {
        self.lo
    }

    /// Highest addressed element offset.
    pub fn hi(&self) -> i128 {
        self.hi
    }
}

/// A validated contraction problem.
///
/// # Examples
///
/// ```
/// use tcontract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
/// let l = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
/// // D[i,j] = sum_k A[i,k] B[k,j]
/// let p = Problem::from_labels(
///     DType::F64,
///     l(&[2, 3], &[1, 2]),
///     l(&[3, 4], &[1, 3]),
///     CSpec::Absent,
///     l(&[2, 4], &[1, 2]),
///     &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
/// ).unwrap();
/// assert_eq!((p.roles().m().len(), p.roles().n().len(), p.roles().k().len()), (1, 1, 1));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Problem {
    dtype: DType,
    a: OperandSpec,
    b: OperandSpec,
    c: CSpec,
    d: OperandSpec,
    lowered: Lowered,
}

impl Problem {
    /// Lower a label description.
    ///
    /// # Errors
    ///
    /// [`Error::Config`] (C labels vs C mode), [`Error::Shape`] (label counts,
    /// extent agreement, overflow), [`Error::Unsupported`] (an output-only
    /// label) and [`Error::Alias`] (a non-injective D), all before any scatter
    /// allocation, selection or zero-size shortcut.
    pub fn from_labels(
        dtype: DType,
        a: OperandSpec,
        b: OperandSpec,
        c: CSpec,
        d: OperandSpec,
        labels: &Labels,
    ) -> Result<Self> {
        let lowered = validate::lower(dtype, &a, &b, &c, &d, labels)?;
        Ok(Self {
            dtype,
            a,
            b,
            c,
            d,
            lowered,
        })
    }

    /// Lower a `dot_general` description. D must have the extents
    /// `[lhs_free, rhs_free, batch]`; C is D with identity `op_C`.
    ///
    /// # Errors
    ///
    /// As [`from_labels`](Self::from_labels), plus the axis-list errors of
    /// [`DotGeneral::validate`] and [`ShapeError::OutputExtents`].
    pub fn from_dot_general(
        dtype: DType,
        a: OperandSpec,
        b: OperandSpec,
        d: OperandSpec,
        dot: &DotGeneral,
    ) -> Result<Self> {
        let shape = dot.validate(a.layout().dims(), b.layout().dims())?;
        if shape.out_dims != d.layout().dims() {
            return Err(Error::Shape(ShapeError::OutputExtents {
                expected: shape.out_dims,
                actual: d.layout().dims().to_vec(),
            }));
        }
        // One lowering: axes get synthetic labels and take the label path.
        let (nb, nc) = (
            dot.lhs_batch().len() as i64,
            dot.lhs_contract().len() as i64,
        );
        let (mut la, mut lb) = (vec![0i64; a.layout().rank()], vec![0i64; b.layout().rank()]);
        let mut ld = Vec::with_capacity(d.layout().rank());
        let mut next = nb + nc;
        for &x in &shape.lhs_free {
            la[x] = next;
            ld.push(next);
            next += 1;
        }
        for &x in &shape.rhs_free {
            lb[x] = next;
            ld.push(next);
            next += 1;
        }
        for (p, (&l, &r)) in dot.lhs_batch().iter().zip(dot.rhs_batch()).enumerate() {
            la[l] = p as i64;
            lb[r] = p as i64;
            ld.push(p as i64);
        }
        for (p, (&l, &r)) in dot
            .lhs_contract()
            .iter()
            .zip(dot.rhs_contract())
            .enumerate()
        {
            la[l] = nb + p as i64;
            lb[r] = nb + p as i64;
        }
        Self::from_labels(
            dtype,
            a,
            b,
            CSpec::Output(Op::Identity),
            d,
            &Labels::new(&la, &lb, &ld),
        )
    }

    /// The storage type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Operand A as given.
    pub fn a(&self) -> &OperandSpec {
        &self.a
    }

    /// Operand B as given.
    pub fn b(&self) -> &OperandSpec {
        &self.b
    }

    /// The output D as given.
    pub fn d(&self) -> &OperandSpec {
        &self.d
    }

    /// The C mode.
    pub fn c_spec(&self) -> &CSpec {
        &self.c
    }

    /// The reduced axes by role.
    pub fn roles(&self) -> &Roles {
        &self.lowered.roles
    }

    /// The `op_C` applied to the accumulation source (identity for an
    /// overwrite).
    pub fn op_c(&self) -> Op {
        match &self.c {
            CSpec::Absent => Op::Identity,
            CSpec::Output(op) => *op,
            CSpec::Separate(c) => c.op(),
        }
    }

    /// The element offsets operand `which` can address; `None` when it has no
    /// elements. For C in `Absent`/`Output` mode this is D's.
    pub fn span(&self, which: OperandId) -> Option<Span> {
        self.lowered.spans[which as usize]
    }

    /// Whether the separate C maps every element exactly as D does (so
    /// `C == D` is an in-place update). Always true for `Absent`/`Output`.
    pub fn c_matches_d(&self) -> bool {
        self.lowered.c_matches_d
    }

    /// Some contracted extent is zero: the product has no terms.
    pub fn k_empty(&self) -> bool {
        self.lowered.k_empty
    }

    /// The output has no elements.
    pub fn out_empty(&self) -> bool {
        self.lowered.out_empty
    }

    /// Every reduced axis is a batch axis (a Hadamard product, or a scalar).
    pub fn all_batch(&self) -> bool {
        let r = &self.lowered.roles;
        r.m.is_empty() && r.n.is_empty() && r.k.is_empty()
    }

    /// Multiply-accumulates of one execution: the product of every reduced
    /// extent (an empty role set contributes one).
    pub fn macs(&self) -> u128 {
        let r = &self.lowered.roles;
        r.m.iter()
            .chain(&r.n)
            .chain(&r.k)
            .chain(&r.h)
            .map(|x| x.extent as u128)
            .product()
    }
}
