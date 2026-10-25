use std::num::NonZeroUsize;

use crate::{BatchDecision, BatchPolicy, BatchQuery, ExecError, Pool, WorkspaceProvider};

/// The resources one operation runs with: a placement token, optional reusable
/// scratch, and the batch lane policy.
///
/// The [`strided_exec::Exec`] token decides where the work runs and how many
/// tasks may run at once. This type adds what is specific to contraction: the
/// workspace the call may reuse and the [`BatchPolicy`]. It owns no placement
/// code. To start parallel work take the token with [`Exec::token`] (a reborrow,
/// so one `Exec` serves a sequence of calls) or [`Exec::into_token`], and call
/// its `fork`, `for_ranges`, `broadcast` or `external`; build a child bundle
/// for each forked job with [`Exec::from_token`].
///
/// Like the token it wraps, an `Exec` is neither `Clone` nor `Copy`, so two
/// operations cannot both claim the full budget; operations take `&mut Exec`.
///
/// # Examples
///
/// ```
/// use tcontract_exec::Exec;
/// let mut exec = Exec::serial();
/// assert_eq!(exec.budget(), 1);
/// assert!(exec.token().is_serial());
/// ```
pub struct Exec<'a> {
    token: strided_exec::Exec<'a>,
    workspace: Option<&'a dyn WorkspaceProvider>,
    policy: BatchPolicy,
}

// A `&dyn WorkspaceProvider` is not `Debug`, so the derive cannot cover the
// workspace. The provider is an anonymous implementation detail here, so only
// its presence is shown.
impl std::fmt::Debug for Exec<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Exec")
            .field("token", &self.token)
            .field("workspace", &self.workspace.is_some())
            .field("policy", &self.policy)
            .finish()
    }
}

impl<'a> Exec<'a> {
    /// The serial context: the calling thread only, call-local scratch.
    pub const fn serial() -> Exec<'static> {
        Exec {
            token: strided_exec::Exec::serial(),
            workspace: None,
            policy: crate::batch::DEFAULT_BATCH_POLICY,
        }
    }

    /// Serial execution with caller-owned reusable storage.
    pub fn serial_with_workspace(workspace: &'a dyn WorkspaceProvider) -> Self {
        Exec {
            token: strided_exec::Exec::serial(),
            workspace: Some(workspace),
            policy: crate::batch::DEFAULT_BATCH_POLICY,
        }
    }

    /// `budget` slots of `pool`, with the pool's own workspace and the default
    /// batch policy. This is the only place a budget is clamped: to the pool
    /// width.
    ///
    /// # Errors
    ///
    /// [`ExecError::Placement`] when `budget == 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tcontract_exec::{Exec, Pool};
    /// let sp = strided_exec::Pool::builder().threads(2).build().unwrap();
    /// let pool = Pool::borrow(&sp);
    /// assert_eq!(Exec::new(&pool, 64).unwrap().budget(), pool.size());
    /// assert!(Exec::new(&pool, 0).is_err());
    /// ```
    pub fn new(pool: &'a Pool<'_>, budget: usize) -> Result<Self, ExecError> {
        Ok(Exec {
            token: strided_exec::Exec::new(pool.inner(), budget)?,
            workspace: Some(pool.workspace()),
            policy: BatchPolicy::default(),
        })
    }

    /// Bundle an existing token with a workspace and a policy: the way to give a
    /// forked job its own bundle, since `fork` hands out bare child tokens.
    ///
    /// # Examples
    ///
    /// ```
    /// use tcontract_exec::{BatchPolicy, Exec};
    /// let exec = Exec::from_token(strided_exec::Exec::serial(), None, BatchPolicy::default());
    /// assert_eq!(exec.budget(), 1);
    /// ```
    pub fn from_token(
        token: strided_exec::Exec<'a>,
        workspace: Option<&'a dyn WorkspaceProvider>,
        policy: BatchPolicy,
    ) -> Self {
        Exec {
            token,
            workspace,
            policy,
        }
    }

    /// The placement token, lent for one call. The bundle keeps its budget, so
    /// several calls can follow one another.
    pub fn token(&mut self) -> strided_exec::Exec<'_> {
        self.token.reborrow()
    }

    /// The placement token, consuming the bundle (for `fork`, `broadcast`, or
    /// any entry that takes the token by value and outlives this borrow).
    pub fn into_token(self) -> strided_exec::Exec<'a> {
        self.token
    }

    /// The same resources, lent for as long as `self` is exclusively borrowed.
    pub fn reborrow(&mut self) -> Exec<'_> {
        Exec {
            token: self.token.reborrow(),
            workspace: self.workspace,
            policy: self.policy,
        }
    }

    /// Narrow the budget to `min(budget, max)`; the dropped slots stay unused.
    ///
    /// # Examples
    ///
    /// ```
    /// use tcontract_exec::Exec;
    /// assert_eq!(Exec::serial().limit(4.try_into().unwrap()).budget(), 1);
    /// ```
    #[must_use]
    pub fn limit(self, max: NonZeroUsize) -> Self {
        Exec {
            token: self.token.limit(max),
            workspace: self.workspace,
            policy: self.policy,
        }
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
    /// let exec = Exec::serial().with_batch_policy(BatchPolicy::new(BatchStrategy::Outer));
    /// assert_eq!(exec.batch_policy().strategy, BatchStrategy::Outer);
    /// ```
    pub fn batch_policy(&self) -> &BatchPolicy {
        &self.policy
    }

    /// Return this context with a different batch lane policy.
    ///
    /// A serial context has one thread, so the policy cannot change its lane
    /// count, but it is preserved: a forced strategy is still honoured (or
    /// refused with a typed error) at one thread instead of silently running
    /// `Auto`.
    #[must_use]
    pub fn with_batch_policy(mut self, policy: BatchPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Decide the batch route for `query` on this context's budget and policy.
    ///
    /// This is the one decision function every operation that owns batch lanes
    /// calls; see [`crate::batch`]. Whether a *route* can honour the returned
    /// decision is the route's own preflight (see
    /// [`crate::batch::BatchDecision`]'s route contract); the decision itself
    /// is arithmetic and never fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tcontract_exec::{BatchQuery, Exec};
    /// let d = Exec::serial().batch_decision(&BatchQuery::new(4, 1000.0));
    /// assert_eq!((d.lanes, d.item_threads), (1, 1));
    /// ```
    pub fn batch_decision(&self, query: &BatchQuery) -> BatchDecision {
        crate::batch::batch_decision(&self.policy, self.token.budget(), query)
    }

    /// Most threads an operation may occupy (1 for a serial context).
    pub fn budget(&self) -> usize {
        self.token.budget().get()
    }

    /// Storage the operations on this context may share, or none.
    ///
    /// A pool lends its own arena to every operation that runs on it; a serial
    /// context owns nothing, so a caller that wants reuse keeps a provider of
    /// its own and lends it through [`Exec::serial_with_workspace`].
    pub fn workspace(&self) -> Option<&'a dyn WorkspaceProvider> {
        self.workspace
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
