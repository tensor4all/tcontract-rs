use crate::ArenaProvider;
use std::sync::Arc;

/// The CPU resources one owner lends to every operation: a
/// [`strided_exec::Pool`] (where tasks run) and an [`ArenaProvider`] (where
/// scratch lives).
///
/// The wrapper never starts threads itself; [`strided_exec::PoolBuilder`] is
/// the only place a pool is made. Retain the wrapper across calls so its arena
/// is reused; [`Exec::new`](crate::Exec::new) mints the per-call resource
/// bundle from it, and that constructor is the only place a thread budget is
/// clamped to the pool width.
///
/// # Examples
///
/// ```
/// let sp = strided_exec::Pool::builder().threads(3).build().unwrap();
/// let pool = tcontract_exec::Pool::borrow(&sp);
/// assert_eq!(pool.size(), sp.width().get());
/// ```
pub struct Pool<'p> {
    pool: PoolRef<'p>,
    /// Retained storage for operations through this wrapper. Reusing the wrapper
    /// reuses its worker buffers, B panel and barriered team sets.
    workspace: ArenaProvider,
}

enum PoolRef<'p> {
    Borrowed(&'p strided_exec::Pool),
    Shared(Arc<strided_exec::Pool>),
}

impl<'p> Pool<'p> {
    /// Borrow a pool the host owns (an engine that keeps its own
    /// [`strided_exec::Pool`]).
    pub fn borrow(pool: &'p strided_exec::Pool) -> Self {
        Self::with(PoolRef::Borrowed(pool))
    }

    fn with(pool: PoolRef<'p>) -> Self {
        Self {
            pool,
            workspace: ArenaProvider::new(),
        }
    }

    /// The storage this pool lends to every operation that runs on it.
    ///
    /// Borrowed, not copied: one pool is one owner, and a driver that is given
    /// this provider returns its team sets to it on drop.
    pub fn workspace(&self) -> &ArenaProvider {
        &self.workspace
    }

    /// Release the pool's idle workspace storage. Live leases and borrowed
    /// worker buffers survive; a worker parked on this pool keeps its slot
    /// until it is idle again.
    pub fn trim_workspace(&self) {
        use crate::WorkspaceProvider;
        self.workspace.trim();
    }

    /// Number of workers, the largest budget [`Exec::new`](crate::Exec::new)
    /// grants.
    pub fn size(&self) -> usize {
        self.inner().width().get()
    }

    pub(crate) fn inner(&self) -> &strided_exec::Pool {
        match &self.pool {
            PoolRef::Borrowed(p) => p,
            PoolRef::Shared(p) => p,
        }
    }
}

impl Pool<'static> {
    /// Take ownership of a pool, for hosts without one of their own (the C
    /// ABI's executors). Dropping the wrapper shuts the workers down.
    ///
    /// # Examples
    ///
    /// ```
    /// let sp = strided_exec::Pool::builder().threads(2).build().unwrap();
    /// let pool = tcontract_exec::Pool::owned(sp);
    /// assert!(pool.size() >= 1);
    /// ```
    pub fn owned(pool: strided_exec::Pool) -> Self {
        Self::shared(Arc::new(pool))
    }

    /// Share one [`strided_exec::Pool`] between several wrappers, each with its
    /// own workspace (the C ABI lends one pool handle to several executors). The
    /// workers stop when the last wrapper is dropped.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// let sp = Arc::new(strided_exec::Pool::builder().threads(2).build().unwrap());
    /// let a = tcontract_exec::Pool::shared(Arc::clone(&sp));
    /// let b = tcontract_exec::Pool::shared(sp);
    /// assert_eq!(a.size(), b.size());
    /// ```
    pub fn shared(pool: Arc<strided_exec::Pool>) -> Self {
        Self::with(PoolRef::Shared(pool))
    }
}

impl std::fmt::Debug for Pool<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pool").field("size", &self.size()).finish()
    }
}
