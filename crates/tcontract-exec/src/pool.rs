use crate::ArenaProvider;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// A wrapper around a host's borrowed, owned or shared Rayon pool.
///
/// tcontract never creates threads beyond this pool. Its size is fixed at
/// construction (Rayon pools cannot resize). The embedded mutex serializes
/// SPMD broadcasts through this wrapper. Hosts sharing a raw pool must retain
/// one wrapper and serialize potentially conflicting numerical work, rather
/// than constructing independent wrappers with separate SPMD mutexes.
///
/// # Examples
///
/// ```
/// let tp = rayon::ThreadPoolBuilder::new().num_threads(3).build().unwrap();
/// let pool = tcontract_exec::Pool::borrow(&tp);
/// assert_eq!(pool.size(), 3);
/// assert_eq!(pool.stats().entries, 0);
/// ```
pub struct Pool<'p> {
    pool: PoolRef<'p>,
    pub(crate) spmd: Mutex<()>,
    /// Retained storage for operations through this wrapper. Reusing the wrapper
    /// reuses its worker buffers, B panel and barriered team sets.
    workspace: ArenaProvider,
    entries: AtomicU64,
    broadcasts: AtomicU64,
    inline_runs: AtomicU64,
}

enum PoolRef<'p> {
    Borrowed(&'p rayon::ThreadPool),
    Owned(Box<rayon::ThreadPool>),
    Shared(Arc<rayon::ThreadPool>),
}

/// Counters of how often a [`Pool`] was entered, for tests and benchmarks.
///
/// # Examples
///
/// ```
/// assert_eq!(tcontract_exec::PoolStats::default().entries, 0);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// Entries from outside the pool (`install` or partitions).
    pub entries: u64,
    /// SPMD broadcasts started from outside the pool.
    pub broadcasts: u64,
    /// Parallel requests served in place because the caller was a worker.
    pub inline_runs: u64,
}

impl<'p> Pool<'p> {
    /// Borrow a host pool.
    pub fn borrow(pool: &'p rayon::ThreadPool) -> Self {
        Self::with(PoolRef::Borrowed(pool))
    }

    fn with(pool: PoolRef<'p>) -> Self {
        Self {
            pool,
            spmd: Mutex::new(()),
            workspace: ArenaProvider::new(),
            entries: AtomicU64::new(0),
            broadcasts: AtomicU64::new(0),
            inline_runs: AtomicU64::new(0),
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

    /// Number of workers.
    pub fn size(&self) -> usize {
        self.tp().current_num_threads()
    }

    pub(crate) fn tp(&self) -> &rayon::ThreadPool {
        match &self.pool {
            PoolRef::Borrowed(p) => p,
            PoolRef::Owned(p) => p,
            PoolRef::Shared(p) => p,
        }
    }

    /// Give back an owned pool (`None` for a borrowed or shared one), for example
    /// so a C host can drop it and join its workers. A shared wrapper releases
    /// only its Arc reference; other owners keep the raw pool alive.
    pub fn into_owned(self) -> Option<rayon::ThreadPool> {
        match self.pool {
            PoolRef::Owned(p) => Some(*p),
            PoolRef::Borrowed(_) | PoolRef::Shared(_) => None,
        }
    }

    /// Snapshot of the entry counters.
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            entries: self.entries.load(Ordering::Relaxed),
            broadcasts: self.broadcasts.load(Ordering::Relaxed),
            inline_runs: self.inline_runs.load(Ordering::Relaxed),
        }
    }

    /// Zero the counters.
    pub fn reset_stats(&self) {
        self.entries.store(0, Ordering::Relaxed);
        self.broadcasts.store(0, Ordering::Relaxed);
        self.inline_runs.store(0, Ordering::Relaxed);
    }

    pub(crate) fn is_worker(&self) -> bool {
        self.tp().current_thread_index().is_some()
    }

    pub(crate) fn count_entry(&self) {
        self.entries.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn count_broadcast(&self) {
        self.broadcasts.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn count_inline(&self) {
        self.inline_runs.fetch_add(1, Ordering::Relaxed);
    }
}

impl Pool<'static> {
    /// Share the host's existing Rayon pool without creating another worker team.
    ///
    /// Retain this wrapper across calls and share it (for example through an
    /// `Arc<Pool<'static>>`) across host clones to reuse workspace and SPMD
    /// serialization. Other numerical libraries may borrow the host's original
    /// raw pool, but their work must obey the host's same resource-loan protocol.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// let raw = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap());
    /// let pool = tcontract_exec::Pool::shared(Arc::clone(&raw));
    /// assert_eq!(pool.size(), raw.current_num_threads());
    /// ```
    pub fn shared(pool: Arc<rayon::ThreadPool>) -> Self {
        Self::with(PoolRef::Shared(pool))
    }

    /// Take ownership of a pool, for hosts without one of their own (the C
    /// ABI's `tcontract_tapp_executor_create_rayon`). Rust hosts normally [`Pool::borrow`].
    ///
    /// # Examples
    ///
    /// ```
    /// let tp = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
    /// let pool = tcontract_exec::Pool::owned(tp);
    /// assert_eq!(pool.size(), 2);
    /// ```
    pub fn owned(pool: rayon::ThreadPool) -> Self {
        Self::with(PoolRef::Owned(Box::new(pool)))
    }
}

impl std::fmt::Debug for Pool<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pool")
            .field("size", &self.size())
            .field("stats", &self.stats())
            .finish()
    }
}
