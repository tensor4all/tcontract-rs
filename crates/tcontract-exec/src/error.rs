/// Errors from building or using an [`Exec`](crate::Exec).
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExecError {
    /// The placement layer refused a token or a pool (for example a thread
    /// budget of zero).
    #[error("placement: {0}")]
    Placement(#[from] threadteam::Error),
    /// A forced batch strategy is unavailable on this route: for example a
    /// [`BatchStrategy::Split`](crate::batch::BatchStrategy::Split) that asks for
    /// both lanes and item threads on a route that runs each item serially inside
    /// its lane.
    #[error("batch strategy {strategy:?} is unavailable for this route")]
    StrategyUnavailable {
        /// The requested strategy.
        strategy: crate::batch::BatchStrategy,
    },
}
