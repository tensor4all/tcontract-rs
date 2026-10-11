//! The whole-backend trait: an extension seam for a second implementation of a
//! contraction.
//!
//! Object safe for each fixed `T`: a runtime slot holds
//! `Box<dyn ContractionBackend<T>>` and dispatches the dtype in the consumer.
//! The trait lives inside this crate by maintainer decision; it is an
//! extension seam here, not a claim of dependency-isolated implementation
//! choice.

use std::fmt::Debug;

use strided_view::{StridedView, StridedViewMut};
use tcontract_exec::Exec;

use crate::api::{Problem, Result, Scalar};

/// What a prepared plan reports about itself, independent of the backend.
///
/// # Examples
///
/// ```
/// let d = tcontract::api::Diagnostics::new("naive", "loop-nest");
/// assert_eq!(d.materialized, [false; 3]);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Diagnostics {
    /// Backend identity.
    pub backend: &'static str,
    /// Selected algorithm identifier (backend-defined, stable per backend).
    pub algorithm: &'static str,
    /// Which of A, B, C are copied into compact buffers on every execution.
    pub materialized: [bool; 3],
}

impl Diagnostics {
    /// Diagnostics of a plan that copies nothing.
    pub fn new(backend: &'static str, algorithm: &'static str) -> Self {
        Self {
            backend,
            algorithm,
            materialized: [false; 3],
        }
    }

    /// Record which operands are materialized.
    #[must_use]
    pub fn with_materialized(mut self, materialized: [bool; 3]) -> Self {
        self.materialized = materialized;
        self
    }
}

/// What the consumer demands of the plan beyond the numerical result.
///
/// # Examples
///
/// ```
/// let r = tcontract::api::Requirements::new().no_materialize(true);
/// assert!(r.no_materialize);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Requirements {
    /// Refuse a plan that copies a whole operand
    /// ([`Unsupported::WouldMaterialize`](crate::api::Unsupported)).
    pub no_materialize: bool,
}

impl Requirements {
    /// No requirements.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the copy-refusal flag.
    #[must_use]
    pub fn no_materialize(mut self, yes: bool) -> Self {
        self.no_materialize = yes;
        self
    }
}

/// The host width a plan is expected to run on; a planning hint, not a
/// binding. A plan runs on any compatible host width.
///
/// # Examples
///
/// ```
/// assert_eq!(tcontract::api::PlanningBudget::serial().threads, 1);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanningBudget {
    /// Expected most threads an execution occupies (at least 1).
    pub threads: usize,
}

impl PlanningBudget {
    /// A budget of `threads` (clamped to at least one).
    pub fn new(threads: usize) -> Self {
        Self {
            threads: threads.max(1),
        }
    }

    /// One thread.
    pub fn serial() -> Self {
        Self { threads: 1 }
    }
}

/// Where the `beta * op_C(C)` term of an accumulation is read from.
///
/// `Output` reads through the mutable output view, so a safe caller never
/// creates an aliasing immutable view just to accumulate in place. `Separate`
/// borrows a distinct C view without owning or copying its payload.
#[derive(Clone, Copy, Debug)]
pub enum AccumulationSource<'a, T> {
    /// No C term. Accepted only with zero beta, for any prepared C mode.
    Absent,
    /// The previous contents of the output: C is D.
    Output,
    /// A separately described C with the problem's C layout.
    Separate(&'a StridedView<'a, T>),
}

/// A prepared contraction for fixed layouts, reusable across executions.
///
/// The plan snapshots layout and configuration metadata; it holds no operand
/// payload or pointer, no worker thread, and nothing borrowed from the
/// backend or the executor used to prepare it. It works with different
/// buffers and different [`Exec`] budgets, and concurrent executions on
/// independent outputs are supported.
pub trait PreparedContraction<T: Scalar> {
    /// `D = op_D(alpha * dot_general(op_A(A), op_B(B)))`: the overwrite form.
    /// No previous output value is read.
    ///
    /// # Errors
    ///
    /// [`Error::Layout`](crate::api::Error::Layout) when a view differs from
    /// the plan and [`Error::Exec`](crate::api::Error::Exec) when the executor
    /// cannot serve the call: both before any write.
    /// [`Error::Backend`](crate::api::Error::Backend) for an implementation
    /// failure, after which D may be partially written.
    fn execute_into(
        &self,
        exec: &Exec<'_>,
        alpha: T,
        a: &StridedView<'_, T>,
        b: &StridedView<'_, T>,
        d: &mut StridedViewMut<'_, T>,
    ) -> Result<()>;

    /// `D = op_D(alpha * dot_general(op_A(A), op_B(B)) + beta * op_C(C))`.
    ///
    /// `beta == 0` reads no previous C value; `alpha == 0` or an empty
    /// contraction reads no A or B value. `source` must match the planned C
    /// mode: [`AccumulationSource::Output`] only when C maps to D, and
    /// [`AccumulationSource::Separate`] only for a separately described C.
    ///
    /// # Errors
    ///
    /// As [`execute_into`](Self::execute_into), and
    /// [`LayoutError::CMode`](crate::api::LayoutError) for a source that does
    /// not match the planned C mode.
    #[allow(clippy::too_many_arguments)] // INVARIANT: the contraction argument set.
    fn execute_into_accum(
        &self,
        exec: &Exec<'_>,
        alpha: T,
        a: &StridedView<'_, T>,
        b: &StridedView<'_, T>,
        beta: T,
        source: AccumulationSource<'_, T>,
        d: &mut StridedViewMut<'_, T>,
    ) -> Result<()>;

    /// Immutable plan diagnostics.
    fn diagnostics(&self) -> &Diagnostics;
}

/// A prepared plan that can be stored in a runtime slot and shared across
/// threads.
pub type BoxedPlan<T> = Box<dyn PreparedContraction<T> + Send + Sync>;

/// A complete contraction implementation for storage type `T`.
pub trait ContractionBackend<T: Scalar>: Debug + Send + Sync {
    /// Backend identity.
    fn id(&self) -> &'static str;

    /// Validate `problem`, check `requirements` and build a reusable plan.
    ///
    /// `budget` is advisory: a backend may use it to size planning choices, but
    /// the plan runs correctly on any host width and no backend may reject a
    /// host later because it differs from the budget.
    ///
    /// # Errors
    ///
    /// Validation errors, [`Unsupported::WouldMaterialize`](crate::api::Unsupported)
    /// under `no_materialize` and [`Error::Unsupported`](crate::api::Error::Unsupported)
    /// for anything the backend cannot do. No output exists at this point.
    fn prepare(
        &self,
        problem: &Problem,
        requirements: &Requirements,
        budget: &PlanningBudget,
    ) -> Result<BoxedPlan<T>>;
}
