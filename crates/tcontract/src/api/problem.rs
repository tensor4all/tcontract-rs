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
use tinyvec::TinyVec;

/// Inline capacity for one operand's metadata, matching
/// [`strided_view::INLINE_RANK`]. Rank at or below this allocates nothing.
pub(crate) const INLINE_RANK: usize = 8;
/// Inline capacity for the combined role set: two rank-[`INLINE_RANK`] operands
/// can contribute up to `2 * INLINE_RANK` distinct labels. Above it the vector
/// spills, so arbitrary rank still works.
pub(crate) const COMBINED_RANK: usize = 2 * INLINE_RANK;

/// Extents (`usize`) with inline storage for [`INLINE_RANK`] axes.
pub(crate) type DimVec = TinyVec<[usize; INLINE_RANK]>;
/// Signed strides (`isize`) with inline storage for [`INLINE_RANK`] axes.
pub(crate) type StrideVec = TinyVec<[isize; INLINE_RANK]>;
/// Labels (`i64`) with inline storage for [`INLINE_RANK`] axes.
pub(crate) type LabelVec = TinyVec<[i64; INLINE_RANK]>;
/// One role's axes with inline storage for [`COMBINED_RANK`].
pub(crate) type RoleVec = TinyVec<[RoleAxis; COMBINED_RANK]>;
/// An index list (`usize`) with inline storage for [`INLINE_RANK`] axes.
pub(crate) type IndexVec = TinyVec<[usize; INLINE_RANK]>;
/// An index list that can reach `2 * INLINE_RANK` (an output's rank).
pub(crate) type OutputVec = TinyVec<[usize; COMBINED_RANK]>;

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
    dims: DimVec,
    strides: StrideVec,
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
            dims: dims.iter().copied().collect(),
            strides: strides.iter().copied().collect(),
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
        let mut d = DimVec::new();
        for (axis, &e) in dims.iter().enumerate() {
            if e < 0 {
                return Err(ShapeError::NegativeExtent { axis, extent: e }.into());
            }
            d.push(usize::try_from(e).map_err(|_| ShapeError::Overflow { what: "extent" })?);
        }
        let s = strides
            .iter()
            .map(|&x| isize::try_from(x).map_err(|_| ShapeError::Overflow { what: "stride" }))
            .collect::<core::result::Result<StrideVec, _>>()?;
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
        self.dims.as_slice()
    }

    /// Signed element stride of each axis.
    pub fn strides(&self) -> &[isize] {
        self.strides.as_slice()
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
        self.dims.as_slice() == dims && self.strides.as_slice() == strides && self.offset == offset
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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
    pub(crate) m: RoleVec,
    pub(crate) n: RoleVec,
    pub(crate) k: RoleVec,
    pub(crate) h: RoleVec,
}

impl Roles {
    /// Axes of A, C and D.
    pub fn m(&self) -> &[RoleAxis] {
        self.m.as_slice()
    }

    /// Axes of B, C and D.
    pub fn n(&self) -> &[RoleAxis] {
        self.n.as_slice()
    }

    /// Contracted (and isolated, reduced) axes.
    pub fn k(&self) -> &[RoleAxis] {
        self.k.as_slice()
    }

    /// Batch axes.
    pub fn h(&self) -> &[RoleAxis] {
        self.h.as_slice()
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

/// A borrowed view of one operand's layout, for [`ProblemRef`].
///
/// # Examples
///
/// ```
/// use tcontract::api::LayoutRef;
/// let l = LayoutRef { dims: &[2, 3], strides: &[1, 2], offset: 0 };
/// assert_eq!(l.rank(), 2);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LayoutRef<'a> {
    /// Extents, one per axis.
    pub dims: &'a [usize],
    /// Signed element strides, one per axis.
    pub strides: &'a [isize],
    /// The logical offset.
    pub offset: isize,
}

impl LayoutRef<'_> {
    /// Number of axes.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }
}

/// The borrowed C mode of a [`ProblemRef`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CRef<'a> {
    /// No C term.
    Absent,
    /// Accumulate from D itself, with this `op_C`.
    Output(Op),
    /// Accumulate from a separately described C layout and `op_C`.
    Separate(LayoutRef<'a>, Op),
}

/// The borrowed contraction description of a [`ProblemRef`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ContractionRef<'a> {
    /// Label semantics, as [`Labels`].
    Labels {
        /// Labels of A.
        a: &'a [i64],
        /// Labels of B.
        b: &'a [i64],
        /// Labels of a separate C, if any.
        c: Option<&'a [i64]>,
        /// Labels of D.
        d: &'a [i64],
    },
    /// Axis-list semantics, as [`DotGeneral`].
    DotGeneral {
        /// Contracted axes of A.
        lhs_contract: &'a [usize],
        /// Contracted axes of B.
        rhs_contract: &'a [usize],
        /// Batch axes of A.
        lhs_batch: &'a [usize],
        /// Batch axes of B.
        rhs_batch: &'a [usize],
    },
}

/// A borrowed, inline description of a contraction, for
/// [`Problem::from_ref`] and `Plan::compile`.
///
/// It borrows every layout, label and axis list, so a caller that keeps those in
/// its own storage can compile a plan without handing ownership over. The
/// element-wise operations are carried inline. A `DotGeneral` contraction ignores
/// `c`, exactly as [`Problem::from_dot_general`] does (its C is D with the
/// identity op).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ProblemRef<'a> {
    /// The storage type.
    pub dtype: DType,
    /// Operand A and its element-wise op.
    pub a: (LayoutRef<'a>, Op),
    /// Operand B and its element-wise op.
    pub b: (LayoutRef<'a>, Op),
    /// The C mode.
    pub c: CRef<'a>,
    /// The output D and its element-wise op.
    pub d: (LayoutRef<'a>, Op),
    /// The contraction description.
    pub contraction: ContractionRef<'a>,
}

impl Problem {
    /// The heap this problem owns: operand layouts and role tables that spilled
    /// their inline storage.
    pub(crate) fn heap_bytes(&self) -> usize {
        use crate::heap::tiny;
        let layout = |o: &OperandSpec| tiny(&o.layout.dims) + tiny(&o.layout.strides);
        let Self {
            dtype: _,
            a,
            b,
            c,
            d,
            lowered,
        } = self;
        let Lowered {
            roles: Roles { m, n, k, h },
            spans: _,
            c_matches_d: _,
            k_empty: _,
            out_empty: _,
        } = lowered;
        let c = match c {
            CSpec::Separate(c) => layout(c),
            CSpec::Absent | CSpec::Output(_) => 0,
        };
        layout(a) + layout(b) + c + layout(d) + tiny(m) + tiny(n) + tiny(k) + tiny(h)
    }

    /// Lower a borrowed [`ProblemRef`].
    ///
    /// # Errors
    ///
    /// As [`Problem::from_labels`] or [`Problem::from_dot_general`], whichever
    /// `p.contraction` selects, plus [`ShapeError::LayoutRank`] for a layout whose
    /// extents and strides differ in length.
    ///
    /// # Examples
    ///
    /// ```
    /// use tcontract::api::{CRef, ContractionRef, DType, LayoutRef, Op, Problem, ProblemRef};
    /// let a = LayoutRef { dims: &[2, 3], strides: &[1, 2], offset: 0 };
    /// let b = LayoutRef { dims: &[3, 4], strides: &[1, 3], offset: 0 };
    /// let d = LayoutRef { dims: &[2, 4], strides: &[1, 2], offset: 0 };
    /// let p = ProblemRef {
    ///     dtype: DType::F64,
    ///     a: (a, Op::Identity),
    ///     b: (b, Op::Identity),
    ///     c: CRef::Absent,
    ///     d: (d, Op::Identity),
    ///     contraction: ContractionRef::Labels {
    ///         a: &[0, 2], b: &[2, 1], c: None, d: &[0, 1],
    ///     },
    /// };
    /// let problem = Problem::from_ref(&p).unwrap();
    /// assert_eq!(problem.roles().k().len(), 1);
    /// ```
    pub fn from_ref(p: &ProblemRef<'_>) -> Result<Self> {
        let spec = |(l, op): (LayoutRef<'_>, Op)| -> Result<OperandSpec> {
            Ok(OperandSpec::new(LayoutSpec::new(l.dims, l.strides, l.offset)?).with_op(op))
        };
        let (a, b, d) = (spec(p.a)?, spec(p.b)?, spec(p.d)?);
        match p.contraction {
            ContractionRef::Labels {
                a: la,
                b: lb,
                c: lc,
                d: ld,
            } => {
                let c = match p.c {
                    CRef::Absent => CSpec::Absent,
                    CRef::Output(op) => CSpec::Output(op),
                    CRef::Separate(l, op) => CSpec::Separate(spec((l, op))?),
                };
                let mut labels = Labels::new(la, lb, ld);
                if let Some(lc) = lc {
                    labels = labels.with_c(lc);
                }
                Self::from_labels(p.dtype, a, b, c, d, &labels)
            }
            ContractionRef::DotGeneral {
                lhs_contract,
                rhs_contract,
                lhs_batch,
                rhs_batch,
            } => {
                let dot = DotGeneral::new(lhs_contract, rhs_contract, lhs_batch, rhs_batch);
                Self::from_dot_general(p.dtype, a, b, d, &dot)
            }
        }
    }

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
        if shape.out_dims.as_slice() != d.layout().dims() {
            return Err(Error::Shape(ShapeError::OutputExtents {
                expected: shape.out_dims.to_vec(),
                actual: d.layout().dims().to_vec(),
            }));
        }
        // One lowering: axes get synthetic labels and take the label path.
        let (nb, nc) = (
            dot.lhs_batch().len() as i64,
            dot.lhs_contract().len() as i64,
        );
        // INVARIANT: ranks above `INLINE_RANK` spill rather than allocate only
        // when they fit; the vectors below are construction scratch.
        let mut la: LabelVec = LabelVec::new();
        la.resize(a.layout().rank(), 0);
        let mut lb: LabelVec = LabelVec::new();
        lb.resize(b.layout().rank(), 0);
        let mut ld: LabelVec = LabelVec::new();
        let mut next = nb + nc;
        for &x in shape.lhs_free.iter() {
            la[x] = next;
            ld.push(next);
            next += 1;
        }
        for &x in shape.rhs_free.iter() {
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
            .chain(r.n.as_slice())
            .chain(r.k.as_slice())
            .chain(r.h.as_slice())
            .map(|x| x.extent as u128)
            .product()
    }
}
