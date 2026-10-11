//! [`PlanReport`]: what a plan decided, fixed when it is built.

use tcontract_kernel::{ComplexScheme, Origin, PartitionPolicy};

use super::PlanStats;
use crate::driver::DynamicReport;

/// Which implementation a plan runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Algorithm {
    /// The packed BLIS/TBLIS-style driver: general strides are packed straight
    /// into micro-kernel panels, so no operand is transposed in memory.
    Packed,
    /// faer on a copy-free fusion to one strided batched GEMM.
    Faer,
    /// One strided elementwise pass over an all-batch (Hadamard) problem.
    Elementwise,
}

impl Algorithm {
    /// The stable algorithm name.
    pub const fn name(self) -> &'static str {
        match self {
            Algorithm::Packed => "packed",
            Algorithm::Faer => "faer",
            Algorithm::Elementwise => "elementwise",
        }
    }
}

/// Why the planner chose a plan's [`Algorithm`]: the rule of
/// [`Plan`](super::Plan) that applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Reason {
    /// Rule 1: the configuration or a selector requires the packed driver.
    Forced,
    /// Rule 2: an all-batch problem.
    AllBatch,
    /// Rule 3: a copy-free fusion to one strided batched GEMM within the
    /// dtype's [`FaerLimit`](super::FaerLimit).
    Fused,
    /// A copy-free fusion whose GEMM volume exceeds the dtype's
    /// [`FaerLimit`](super::FaerLimit), so the packed driver runs.
    AboveFaerLimit {
        /// The fused `m * n * k` per batch item.
        volume: u64,
        /// The bound it exceeds.
        limit: u64,
    },
    /// Rule 4: no copy-free fusion exists.
    NotFusable,
}

/// What the packed strategy resolved to: the family, its geometry and the grid.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct PackedReport {
    /// The resolved kernel family's stable identifier.
    pub family_id: &'static str,
    /// Provenance of the family.
    pub origin: Origin,
    /// The family's complex scheme, or none for real storage.
    pub complex: Option<ComplexScheme>,
    /// Logical register tile rows.
    pub mr: usize,
    /// Logical register tile columns.
    pub nr: usize,
    /// Cache block rows at the serial width.
    pub mc: usize,
    /// Cache block columns at the serial width.
    pub nc: usize,
    /// Cache block depth.
    pub kc: usize,
    /// The grid policy the plan froze.
    pub partition: PartitionPolicy,
    /// Round strip boundaries to 64-byte lines of `C`.
    pub align_c_lines: bool,
    /// The dynamic assignment, when the policy is `DynamicTiles`. The active
    /// width reported is the job-count cap: a plan is built before an executor
    /// is chosen.
    pub dynamic: Option<DynamicReport>,
    /// The matrix shape and folded axes the index analysis reached.
    pub stats: PlanStats,
    /// Whether execution computes `D^T = B^T A^T`: the planner exchanged the
    /// row and column operands to make the micro-tile's rows a run of `D`.
    pub swapped: bool,
    /// Fraction of the row operand's `MR` blocks that are one regular run
    /// (1.0 is a plain strided panel, 0.0 is entirely the gather path), at the
    /// oriented register block. The observed regularity any awkward-stride claim
    /// must quote.
    pub regular_rows: f64,
    /// As [`regular_rows`](Self::regular_rows), for the column operand at `NR`.
    pub regular_cols: f64,
    /// Estimated serial scratch in bytes: one packed `A` block and one packed
    /// `B` panel, at the serial blocking.
    pub scratch_bytes: usize,
}

/// What a plan decided, immutable once built.
///
/// A non-packed plan reports no family.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct PlanReport {
    /// Which implementation runs.
    pub algorithm: Algorithm,
    /// Which selection rule chose it.
    pub reason: Reason,
    /// Which of A, B, C are copied into compact buffers on every execution.
    /// Always false here: no strategy of this crate copies a whole operand
    /// (bounded packing inside the packed driver is not a materialization).
    pub materialized: [bool; 3],
    /// The packed resolution, for the packed strategy.
    pub packed: Option<PackedReport>,
}
