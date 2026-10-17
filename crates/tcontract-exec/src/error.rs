/// Errors from building or using an [`Exec`](crate::Exec).
#[derive(Clone, Copy, Debug, thiserror::Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExecError {
    /// A thread budget of zero was requested.
    #[error("thread budget must be at least 1")]
    ZeroBudget,
    /// An SPMD width larger than the pool was requested.
    #[error("SPMD width {width} exceeds pool size {pool}")]
    WidthExceedsPool {
        /// Requested width.
        width: usize,
        /// Workers in the pool.
        pool: usize,
    },
    /// An SPMD width larger than the context's thread budget was requested.
    #[error("SPMD width {width} exceeds thread budget {budget}")]
    WidthExceedsBudget {
        /// Requested width.
        width: usize,
        /// The context's budget.
        budget: usize,
    },
    /// Co-scheduled execution cannot be guaranteed here (serial context, or
    /// the caller is already a worker of the pool).
    #[error("co-scheduled execution unavailable in this context")]
    Unavailable,
    /// A forced batch strategy is unavailable on this route (currently only
    /// [`BatchStrategy::WholeBatchVendor`](crate::batch::BatchStrategy::WholeBatchVendor)
    /// without a vendor batch call).
    #[error("batch strategy {strategy:?} is unavailable for this route")]
    StrategyUnavailable {
        /// The requested strategy.
        strategy: crate::batch::BatchStrategy,
    },
}
