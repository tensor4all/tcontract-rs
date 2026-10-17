use std::num::NonZeroUsize;

use crate::{BatchDecision, BatchPolicy, BatchQuery, ExecError, Pool};

/// Parallelism granted to an operation by [`Exec::install`].
///
/// # Examples
///
/// ```
/// assert_eq!(tcontract_exec::Par::Seq.threads(), 1);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Par {
    /// Run serially on the current thread.
    Seq,
    /// Up to this many workers of the pool the closure runs in.
    Threads(NonZeroUsize),
}

impl Par {
    /// Worker count (1 for `Seq`).
    pub fn threads(self) -> usize {
        match self {
            Par::Seq => 1,
            Par::Threads(n) => n.get(),
        }
    }
}

/// Execution context passed to every operation that may run in parallel.
///
/// # Examples
///
/// ```
/// use tcontract_exec::Exec;
/// assert_eq!(Exec::serial().budget(), 1);
/// assert!(!Exec::serial().is_worker());
/// ```
#[derive(Clone, Copy)]
#[non_exhaustive]
pub enum Exec<'a> {
    /// Everything on the calling thread, with call-local scratch. Carries the
    /// batch policy so a forced strategy is still resolved at one thread.
    Serial {
        /// The batch lane policy this context applies.
        policy: BatchPolicy,
    },
    /// Everything on the calling thread, with caller-owned reusable scratch.
    SerialWithWorkspace {
        /// The caller-owned storage.
        workspace: &'a dyn crate::WorkspaceProvider,
        /// The batch lane policy this context applies.
        policy: BatchPolicy,
    },
    /// A borrowed pool with a thread budget `<= pool.size()`.
    Rayon {
        /// The borrowed pool.
        pool: &'a Pool<'a>,
        /// Most threads an operation may occupy.
        budget: NonZeroUsize,
        /// The batch lane policy this context applies.
        policy: BatchPolicy,
    },
}

// A `&dyn WorkspaceProvider` is not `Debug`, so the derive cannot cover the
// serial-with-workspace variant. The provider is an anonymous implementation
// detail here, so only the variant name is shown.
impl std::fmt::Debug for Exec<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Exec::Serial { .. } => f.write_str("Serial"),
            Exec::SerialWithWorkspace { .. } => f.write_str("SerialWithWorkspace(..)"),
            Exec::Rayon {
                pool,
                budget,
                policy,
            } => f
                .debug_struct("Rayon")
                .field("pool", pool)
                .field("budget", budget)
                .field("policy", policy)
                .finish(),
        }
    }
}

impl<'a> Exec<'a> {
    /// The serial context.
    pub const fn serial() -> Exec<'static> {
        Exec::Serial {
            policy: crate::batch::DEFAULT_BATCH_POLICY,
        }
    }

    /// Serial execution with caller-owned reusable storage.
    pub fn serial_with_workspace(workspace: &'a dyn crate::WorkspaceProvider) -> Self {
        Exec::SerialWithWorkspace {
            workspace,
            policy: crate::batch::DEFAULT_BATCH_POLICY,
        }
    }

    /// Use `pool` with its full size as budget and the default batch policy.
    pub fn rayon(pool: &'a Pool<'a>) -> Self {
        let budget = NonZeroUsize::new(pool.size()).unwrap_or(NonZeroUsize::MIN);
        Exec::Rayon {
            pool,
            budget,
            policy: BatchPolicy::default(),
        }
    }

    /// Limit the budget; values above the pool size are clamped.
    ///
    /// # Errors
    ///
    /// [`ExecError::ZeroBudget`] when `max_threads == 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tcontract_exec::{Exec, ExecError};
    /// assert_eq!(Exec::serial().with_budget(0).unwrap_err(), ExecError::ZeroBudget);
    /// ```
    pub fn with_budget(self, max_threads: usize) -> Result<Self, ExecError> {
        let max = NonZeroUsize::new(max_threads).ok_or(ExecError::ZeroBudget)?;
        Ok(match self {
            Exec::Serial { .. } | Exec::SerialWithWorkspace { .. } => self,
            Exec::Rayon { pool, .. } => {
                let cap = NonZeroUsize::new(pool.size()).unwrap_or(NonZeroUsize::MIN);
                Exec::Rayon {
                    pool,
                    budget: max.min(cap),
                    policy: *self.batch_policy(),
                }
            }
        })
    }

    /// The batch lane policy this context applies.
    ///
    /// A serial context has one thread but still carries a policy, so a forced
    /// strategy is resolved (and can be refused) at one thread as well.
    ///
    /// # Examples
    ///
    /// ```
    /// use tcontract_exec::{BatchPolicy, BatchStrategy, Exec};
    /// assert_eq!(Exec::serial().batch_policy().strategy, BatchStrategy::Auto);
    /// let tp = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
    /// let pool = tcontract_exec::Pool::borrow(&tp);
    /// let exec = Exec::rayon(&pool).with_batch_policy(BatchPolicy::new(BatchStrategy::Outer));
    /// assert_eq!(exec.batch_policy().strategy, BatchStrategy::Outer);
    /// ```
    pub fn batch_policy(&self) -> &BatchPolicy {
        match self {
            Exec::Serial { policy } | Exec::SerialWithWorkspace { policy, .. } => policy,
            Exec::Rayon { policy, .. } => policy,
        }
    }

    /// Return this context with a different batch lane policy.
    ///
    /// A serial context has one thread, so the policy cannot change its lane
    /// count, but it is preserved: a forced
    /// [`BatchStrategy::WholeBatchVendor`](crate::batch::BatchStrategy::WholeBatchVendor)
    /// is still refused at one thread instead of silently running `Auto`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tcontract_exec::{BatchPolicy, BatchStrategy, Exec};
    /// let exec = Exec::serial().with_batch_policy(BatchPolicy::new(BatchStrategy::Outer));
    /// assert_eq!(exec.batch_policy().strategy, BatchStrategy::Outer);
    /// ```
    #[must_use]
    pub fn with_batch_policy(self, policy: BatchPolicy) -> Self {
        match self {
            Exec::Serial { .. } => Exec::Serial { policy },
            Exec::SerialWithWorkspace { workspace, .. } => {
                Exec::SerialWithWorkspace { workspace, policy }
            }
            Exec::Rayon { pool, budget, .. } => Exec::Rayon {
                pool,
                budget,
                policy,
            },
        }
    }

    /// Decide the batch route for `query` on this context's budget and policy.
    ///
    /// This is the one decision function every operation that owns batch lanes
    /// calls; see [`crate::batch`]. It resolves availability before the caller
    /// writes anything.
    ///
    /// # Errors
    ///
    /// [`ExecError::StrategyUnavailable`] for a forced
    /// [`BatchStrategy::WholeBatchVendor`](crate::batch::BatchStrategy::WholeBatchVendor)
    /// on a query without a vendor batch route.
    ///
    /// # Examples
    ///
    /// ```
    /// use tcontract_exec::{BatchQuery, Exec};
    /// let d = Exec::serial().batch_decision(&BatchQuery::new(4, 1000.0)).unwrap();
    /// assert_eq!((d.lanes, d.item_threads), (1, 1));
    /// ```
    pub fn batch_decision(&self, query: &BatchQuery) -> Result<BatchDecision, ExecError> {
        // INVARIANT: `Exec` budgets are `NonZeroUsize` by construction.
        let budget = NonZeroUsize::new(self.budget()).unwrap_or(NonZeroUsize::MIN);
        crate::batch::batch_decision(self.batch_policy(), budget, query)
    }

    /// Most threads an operation may occupy (1 for a serial context).
    pub fn budget(&self) -> usize {
        match self {
            Exec::Serial { .. } | Exec::SerialWithWorkspace { .. } => 1,
            Exec::Rayon { budget, .. } => budget.get(),
        }
    }

    /// Storage the operations on this context may share, or none.
    ///
    /// A pool lends its own arena to every operation that runs on it; a serial
    /// context owns nothing, so a caller that wants reuse keeps a provider of
    /// its own and lends it through [`Exec::serial_with_workspace`].
    pub fn workspace(&self) -> Option<&'a dyn crate::WorkspaceProvider> {
        match self {
            Exec::Serial { .. } => None,
            Exec::SerialWithWorkspace { workspace, .. } => Some(*workspace),
            Exec::Rayon { pool, .. } => Some(pool.workspace()),
        }
    }

    /// Whether the calling thread is a worker of this context's pool.
    pub fn is_worker(&self) -> bool {
        match self {
            Exec::Serial { .. } | Exec::SerialWithWorkspace { .. } => false,
            Exec::Rayon { pool, .. } => pool.is_worker(),
        }
    }

    fn width(&self, k: usize) -> usize {
        k.clamp(1, self.budget())
    }

    /// Run `op` with `min(k, budget)` threads of parallelism.
    ///
    /// Width one runs `op(Par::Seq)` inline on the caller. Otherwise `op`
    /// runs inside the pool, which is entered once, or in place when the
    /// caller is already one of its workers.
    pub fn install<R: Send>(&self, k: usize, op: impl FnOnce(Par) -> R + Send) -> R {
        let k = self.width(k);
        match (self, NonZeroUsize::new(k)) {
            (Exec::Rayon { pool, .. }, Some(n)) if k > 1 => {
                if pool.is_worker() {
                    pool.count_inline();
                    op(Par::Threads(n))
                } else {
                    pool.count_entry();
                    pool.tp().install(|| op(Par::Threads(n)))
                }
            }
            _ => op(Par::Seq),
        }
    }

    /// Barrier-free partition: runs `f(i)` exactly once for each `i < k`,
    /// on at most `budget` workers in contiguous lanes.
    ///
    /// Tasks are not guaranteed to run concurrently, so `f` must never wait
    /// on another index.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::atomic::{AtomicUsize, Ordering};
    /// let n = AtomicUsize::new(0);
    /// tcontract_exec::Exec::serial().for_each_partition(5, &|_| {
    ///     n.fetch_add(1, Ordering::Relaxed);
    /// });
    /// assert_eq!(n.into_inner(), 5);
    /// ```
    pub fn for_each_partition(&self, k: usize, f: &(dyn Fn(usize) + Sync)) {
        if k == 0 {
            return;
        }
        let lanes = self.width(k);
        if lanes == 1 {
            (0..k).for_each(f);
            return;
        }
        self.install(lanes, |_| {
            rayon::scope(|s| {
                for lane in 0..lanes {
                    s.spawn(move |_| {
                        let lo = lane * k / lanes;
                        let hi = (lane + 1) * k / lanes;
                        (lo..hi).for_each(f);
                    });
                }
            })
        });
    }

    /// Co-scheduled SPMD execution: `f(t)` for every `t < width` runs
    /// concurrently on distinct workers, so `f` may use barriers counting
    /// `width` participants.
    ///
    /// Width one runs `f(0)` inline. On a pool the whole pool is dispatched
    /// (Rayon `broadcast`); workers with index `>= width` return at once.
    /// Broadcasts through one [`Pool`] wrapper are serialized (use one
    /// wrapper per `ThreadPool`). A broadcast waits until every worker reaches
    /// a scheduling point, so a worker busy in another host's long job delays
    /// it. A panic in a barrier-free `f` propagates to the caller after every
    /// participant has finished; if `f` panics before a barrier the other
    /// participants wait at it forever, so barrier-bearing `f` must not panic.
    ///
    /// # Errors
    ///
    /// [`ExecError::Unavailable`] when co-scheduling cannot be guaranteed
    /// (`Serial` with `width > 1`, or the caller is already a worker of the
    /// pool, where an outer job may hold workers at a barrier); nothing runs
    /// then, and the caller uses its barrier-free variant.
    /// [`ExecError::WidthExceedsPool`] when `width` exceeds the pool, and
    /// [`ExecError::WidthExceedsBudget`] when it exceeds the budget (the
    /// active width is capped by the budget even though the whole pool is
    /// dispatched).
    ///
    /// # Examples
    ///
    /// ```
    /// use tcontract_exec::{Exec, ExecError};
    /// assert_eq!(Exec::serial().broadcast(2, &|_| {}), Err(ExecError::Unavailable));
    /// ```
    pub fn broadcast(&self, width: usize, f: &(dyn Fn(usize) + Sync)) -> Result<(), ExecError> {
        if width == 0 {
            return Ok(());
        }
        match self {
            Exec::Serial { .. } | Exec::SerialWithWorkspace { .. } => {
                if width == 1 {
                    f(0);
                    Ok(())
                } else {
                    Err(ExecError::Unavailable)
                }
            }
            Exec::Rayon { pool, budget, .. } => {
                let size = pool.size();
                if width > size {
                    return Err(ExecError::WidthExceedsPool { width, pool: size });
                }
                if width > budget.get() {
                    return Err(ExecError::WidthExceedsBudget {
                        width,
                        budget: budget.get(),
                    });
                }
                if pool.is_worker() {
                    return Err(ExecError::Unavailable);
                }
                if width == 1 {
                    f(0);
                    return Ok(());
                }
                // INVARIANT: the SPMD mutex serializes broadcasts on this pool, so
                // no worker ever holds two barrier-bearing jobs; it guards no data,
                // so a poisoned lock (panic in an earlier `f`) is safe to reuse.
                let _guard = pool
                    .spmd
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                pool.count_broadcast();
                pool.tp().broadcast(|ctx| {
                    let t = ctx.index();
                    if t < width {
                        f(t)
                    }
                });
                Ok(())
            }
        }
    }

    /// Partition width for work estimated at `serial_ns` on one thread,
    /// minimizing the [`WidthPolicy`](crate::WidthPolicy) cost model within the budget.
    ///
    /// # Examples
    ///
    /// ```
    /// use tcontract_exec::{Exec, WidthPolicy};
    /// assert_eq!(Exec::serial().width_for(1e9, &WidthPolicy::default()), 1);
    /// ```
    pub fn width_for(&self, serial_ns: f64, policy: &crate::WidthPolicy) -> usize {
        crate::width::width_for(serial_ns, policy, self.budget())
    }
}
