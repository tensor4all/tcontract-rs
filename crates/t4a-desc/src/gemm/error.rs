//! The one error type of the GEMM descriptor.

use core::fmt;

use super::Operand;

/// Why a job list or a buffer was rejected.
///
/// Every `job` field is an index into the caller's original job list. When
/// several jobs fail, the error names the **lowest** failing id, so the report
/// does not depend on any internal sort order.
///
/// The enum is `#[non_exhaustive]`: a later descriptor rule may add a variant.
///
/// # Examples
///
/// ```
/// use t4a_desc::gemm::{GemmBatch, GemmError, GemmJob, GemmOps, Operand, Trans};
/// use t4a_desc::ScalarKind;
///
/// let ops = GemmOps { a: Trans::N, b: Trans::N };
/// let mut job = GemmJob::compact(ops, [0; 3], 4, 1, 1);
/// job.lda = 3; // below m = 4
/// let err = GemmBatch::new(ScalarKind::F32, ops, vec![job]).unwrap_err();
/// assert_eq!(err, GemmError::LeadingDim { job: 0, operand: Operand::A, ld: 3, min: 4 });
/// ```
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GemmError {
    /// Checked arithmetic on a job overflowed, or a value does not fit in
    /// `isize` (elements, or bytes for an end offset).
    Overflow {
        /// Original job id.
        job: usize,
        /// The quantity: an extent or leading dimension (`"m"`, …, `"ldc"`) that
        /// exceeds `isize::MAX`, or `"A span"`, `"A end"`, `"A byte end"` and
        /// their `B`/`C` forms for a footprint, its end and its end in bytes.
        what: &'static str,
    },
    /// A leading dimension is below BLAS's minimum `max(1, stored rows)`.
    LeadingDim {
        /// Original job id.
        job: usize,
        /// The operand whose leading dimension is too small.
        operand: Operand,
        /// The given leading dimension.
        ld: usize,
        /// The smallest leading dimension the job allows.
        min: usize,
    },
    /// A buffer is shorter than a job's footprint in it.
    Bounds {
        /// Original job id: the lowest whose footprint ends past `available`.
        job: usize,
        /// The buffer.
        operand: Operand,
        /// Exclusive end of the job's footprint, in elements.
        end: usize,
        /// The buffer's length, in elements.
        available: usize,
    },
    /// Two non-empty output footprints overlap.
    ///
    /// The test compares the intervals
    /// `[c_offset, c_offset + (n-1)·ldc + m)`, so it is conservative: two
    /// interleaved outputs with `ldc > m` are rejected even when no element is
    /// shared.
    Overlap {
        /// The lowest original id whose output overlaps another.
        job: usize,
        /// One job whose output it overlaps.
        other: usize,
    },
    /// An uninitialized-output entry needs `ldc == m` on every non-empty job.
    FreshLayout {
        /// The lowest original id with `ldc != m`.
        job: usize,
    },
    /// The non-empty outputs do not tile the fresh output exactly.
    FreshCoverage {
        /// The lowest original id that does not start where the previous
        /// output (in offset order) ends, or at 0 for the first; `None` when
        /// the outputs tile a prefix whose length differs from the buffer's.
        job: Option<usize>,
    },
}

impl fmt::Display for GemmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Overflow { job, what } => write!(f, "GEMM job {job}: {what} overflows"),
            Self::LeadingDim {
                job,
                operand,
                ld,
                min,
            } => write!(
                f,
                "GEMM job {job}: ld{} = {ld} is below the minimum {min}",
                lower(*operand)
            ),
            Self::Bounds {
                job,
                operand,
                end,
                available,
            } => write!(
                f,
                "GEMM job {job}: {operand} footprint ends at {end}, buffer has {available} elements"
            ),
            Self::Overlap { job, other } => {
                write!(f, "GEMM jobs {job} and {other} have overlapping outputs")
            }
            Self::FreshLayout { job } => {
                write!(f, "GEMM job {job}: an uninitialized output needs ldc == m")
            }
            Self::FreshCoverage { job: Some(job) } => write!(
                f,
                "GEMM job {job}: outputs leave a gap before it in the uninitialized output"
            ),
            Self::FreshCoverage { job: None } => write!(
                f,
                "GEMM outputs do not cover the uninitialized output exactly"
            ),
        }
    }
}

impl std::error::Error for GemmError {}

const fn lower(operand: Operand) -> char {
    match operand {
        Operand::A => 'a',
        Operand::B => 'b',
        Operand::C => 'c',
    }
}
