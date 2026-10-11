//! The one error type of `tcontract`.
//!
//! Every variant carries typed context (operand, axis or label, expected and
//! actual values); nothing is stringified. Preparation and input-validation
//! failures write nothing; a failure inside a foreign backend after compute
//! started may leave a partial output, and no rollback is promised.

use tcontract_exec::ExecError;
use tcontract_kernel::SelectError;

/// Which operand a diagnostic is about.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OperandId {
    /// The left input.
    A,
    /// The right input.
    B,
    /// The accumulation source when it is described separately from `D`.
    C,
    /// The output.
    D,
}

impl core::fmt::Display for OperandId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            OperandId::A => "A",
            OperandId::B => "B",
            OperandId::C => "C",
            OperandId::D => "D",
        })
    }
}

/// An invalid configuration or problem description (nothing about the data).
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ConfigError {
    /// A plan was built for a storage type other than the problem's.
    #[error("the plan's scalar type is {plan} but the problem is described for {problem}")]
    DtypeMismatch {
        /// The plan's storage type.
        plan: &'static str,
        /// The problem's storage type.
        problem: &'static str,
    },
    /// `CSpec` and `Labels::c` disagree: a separate C needs its own labels and
    /// no other mode may carry them.
    #[error("a separate C needs C labels, and only a separate C may carry them")]
    CLabels,
    /// A contracting/batch list of A and B differ in length.
    #[error("the {what} axis lists of A ({lhs}) and B ({rhs}) differ in length")]
    AxisListLength {
        /// `"contracting"` or `"batch"`.
        what: &'static str,
        /// Length of A's list.
        lhs: usize,
        /// Length of B's list.
        rhs: usize,
    },
    /// An axis index is outside the operand's rank.
    #[error("{operand} axis {axis} is out of range for rank {rank}")]
    AxisOutOfRange {
        /// The operand.
        operand: OperandId,
        /// The axis.
        axis: usize,
        /// The operand's rank.
        rank: usize,
    },
    /// An axis index is listed twice in a `DotGeneral`.
    #[error("{operand} axis {axis} is listed twice")]
    AxisRepeated {
        /// The operand.
        operand: OperandId,
        /// The axis.
        axis: usize,
    },
    /// An axis list that is not a permutation.
    #[error("the axis list is not a permutation of 0..{rank}")]
    NotAPermutation {
        /// The rank the permutation had to cover.
        rank: usize,
    },
    /// A plan-configuration option that cannot be honoured or combined.
    #[error("invalid plan configuration: {0}")]
    Option(&'static str),
    /// A blocking dimension has both an absolute and a percentage override.
    #[error("blocking {dim}: an absolute size and a percentage override are mutually exclusive")]
    BlockingExclusive {
        /// `"mc"`, `"kc"` or `"nc"`.
        dim: &'static str,
    },
    /// A numeric option that must be positive (or finite) is not.
    #[error("{what} must be positive")]
    NotPositive {
        /// The option.
        what: &'static str,
    },
}

/// A shape, label or arithmetic problem in the metadata.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ShapeError {
    /// A layout has different numbers of extents and strides.
    #[error("{extents} extents but {strides} strides")]
    LayoutRank {
        /// Number of extents.
        extents: usize,
        /// Number of strides.
        strides: usize,
    },
    /// A signed extent is negative.
    #[error("axis {axis} has the negative extent {extent}")]
    NegativeExtent {
        /// The axis.
        axis: usize,
        /// The extent.
        extent: i64,
    },
    /// An operand has a different number of labels than modes.
    #[error("{operand} has {modes} modes but {labels} labels")]
    LabelCount {
        /// The operand.
        operand: OperandId,
        /// Number of modes.
        modes: usize,
        /// Number of labels.
        labels: usize,
    },
    /// C and D are labelled with different label sets.
    #[error("C and D carry different label sets")]
    OutputLabelMismatch,
    /// One label has two different extents.
    #[error("label {label} has extent {found} where {expected} was already seen")]
    ExtentMismatch {
        /// The label.
        label: i64,
        /// The extent seen first.
        expected: usize,
        /// The conflicting extent.
        found: usize,
    },
    /// Paired axes of A and B (or an output axis) have different extents.
    #[error("A axis {lhs_axis} has extent {lhs} but B axis {rhs_axis} has {rhs}")]
    PairedExtent {
        /// A's axis.
        lhs_axis: usize,
        /// B's axis.
        rhs_axis: usize,
        /// A's extent.
        lhs: usize,
        /// B's extent.
        rhs: usize,
    },
    /// The output's extents are not the contraction's.
    #[error("D has extents {actual:?}, expected {expected:?}")]
    OutputExtents {
        /// The extents the contraction produces.
        expected: Vec<usize>,
        /// The extents of D.
        actual: Vec<usize>,
    },
    /// A product, stride sum, scatter size or address range overflows.
    #[error("{what} overflows")]
    Overflow {
        /// What overflowed.
        what: &'static str,
    },
}

/// An execution-time layout problem.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum LayoutError {
    /// A view's extents, strides or offset differ from the planned ones.
    #[error("{operand}: the view's layout differs from the planned layout")]
    Mismatch {
        /// The operand.
        operand: OperandId,
    },
    /// The backing slice of a view is too short for its layout.
    #[error("{operand}: the backing slice does not cover the addressed range")]
    Bounds {
        /// The operand.
        operand: OperandId,
    },
    /// A separate C was needed and none was given, or the reverse.
    #[error("the accumulation source does not match the planned C mode")]
    CMode,
}

/// An aliasing problem.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum AliasError {
    /// Distinct output index tuples may address one element (zero or
    /// overlapping strides). The test is conservative.
    #[error("D addresses an element more than once (zero or overlapping strides)")]
    OutputNotInjective,
    /// The output overlaps an input.
    #[error("D overlaps {operand}")]
    OutputOverlapsInput {
        /// The input.
        operand: OperandId,
    },
    /// C and D overlap other than as the same mapping.
    #[error("C and D overlap without being the same mapping")]
    CDOverlap,
}

/// A well-formed request this implementation declines.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Unsupported {
    /// A label appears only in the output (TAPP case 5, a broadcast).
    #[error("label {label} appears only in the output (a broadcast)")]
    OutputOnlyLabel {
        /// The label.
        label: i64,
    },
    /// The plan would copy a full operand, and the configuration forbids it.
    #[error("the contraction needs materialized operands {operands:?} (A, B, C)")]
    WouldMaterialize {
        /// Which of A, B, C would be copied.
        operands: [bool; 3],
    },
    /// Another declined request, described by a fixed reason.
    #[error("unsupported: {0}")]
    Reason(&'static str),
}

/// The one error of `tcontract`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// An invalid configuration or problem description.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// A shape, label or arithmetic problem.
    #[error(transparent)]
    Shape(#[from] ShapeError),
    /// An execution-time layout problem.
    #[error(transparent)]
    Layout(#[from] LayoutError),
    /// An aliasing problem.
    #[error(transparent)]
    Alias(#[from] AliasError),
    /// A request this implementation declines (including an explicit
    /// would-materialize reason).
    #[error(transparent)]
    Unsupported(#[from] Unsupported),
    /// Kernel selection failed.
    #[error("kernel selection: {0}")]
    Select(#[from] SelectError),
    /// The execution context refused the request.
    #[error("execution context: {0}")]
    Exec(#[from] ExecError),
    /// A failure of a foreign backend, source preserved.
    #[error("backend: {0}")]
    Backend(Box<dyn std::error::Error + Send + Sync + 'static>),
    /// An impossible invariant; never an input-validation catch-all.
    #[error("internal error: {0}")]
    Internal(&'static str),
}

impl Error {
    /// Wrap a foreign backend's error.
    pub fn backend(e: impl Into<Box<dyn std::error::Error + Send + Sync + 'static>>) -> Self {
        Error::Backend(e.into())
    }

    /// Whether the request was declined before anything was written (#31's
    /// decline contract): a different backend may be able to take it.
    pub fn is_unsupported(&self) -> bool {
        matches!(self, Error::Unsupported(_) | Error::Exec(_))
    }
}

/// `Result` with the crate's [`Error`].
pub type Result<T> = core::result::Result<T, Error>;
