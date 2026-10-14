//! Executors: the `TAPP_executor` of the TAPP C API, serial or owning a Rayon
//! pool that tcontract creates and joins for C hosts without one, plus the
//! host-owned pool handle a C host shares with it.
//!
//! A `TAPP_executor` is an `intptr_t` holding a pointer to an [`Executor`];
//! zero selects the default serial executor (a tcontract policy, not in the
//! pinned TAPP headers). `TAPP_create_executor` makes a serial executor;
//! `tcontract_tapp_executor_create_rayon` makes one that owns a pool;
//! `TAPP_destroy_executor` stops and joins that pool. Operations of every
//! part of `libtcontract_capi` (TAPP products) take the same executor.
//!
//! A `tcontract_pool` is the same discipline for a pool the *host* owns:
//! `tcontract_pool_create_rayon` builds it, `tcontract_pool_destroy` stops and
//! joins it, and `tcontract_tapp_executor_borrow_pool` makes an executor that
//! runs on it instead of creating a second pool. The host keeps the pool alive
//! for every executor built from it and every call through one.
//!
//! Rust hosts do not use this module: they lend their pool to
//! [`tcontract_exec::Pool::borrow`] and never create a second one.
#![allow(non_camel_case_types)]

use std::ffi::c_int;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use ::tcontract_exec::{ArenaProvider, Exec, Pool, PoolStats};

use crate::abi::null;
use crate::status::*;

/// `TAPP_executor`: `intptr_t`, zero meaning the default serial executor.
pub type TAPP_executor = isize;

/// `tcontract_pool`: `intptr_t` holding a host-owned Rayon pool from
/// [`tcontract_pool_create_rayon`]; zero means no pool.
pub type tcontract_pool = isize;

/// Options for [`tcontract_tapp_executor_create_rayon`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct tcontract_rayon_opts {
    /// Worker stack size in bytes; 0 = 16 MiB (provider kernels recurse
    /// deeply, see tenferro-rs's CPU threading contract).
    pub stack_size: usize,
}

enum Kind {
    /// No workers, and the storage its calls reuse across a serial session.
    Serial(ArenaProvider),
    Pool {
        // The executor's `Pool` wrapper: its SPMD gate is wrapper-wide, so
        // every call through this executor goes through it. One executor per
        // pool is the intended configuration (create it once, share it across
        // plans); when a host pool is lent to several borrowed executors, each
        // one keeps its own workspace and gate, so concurrent numerical work
        // belongs on one of them.
        pool: Box<Pool<'static>>,
        joins: Mutex<Vec<JoinHandle<()>>>,
        budget: AtomicUsize,
    },
}

/// What a `TAPP_executor` points at (opaque in C).
pub struct Executor {
    inflight: AtomicUsize,
    kind: Kind,
}

impl std::fmt::Debug for Executor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (pool_size, budget) = self.threads();
        f.debug_struct("Executor")
            .field("pool_size", &pool_size)
            .field("budget", &budget)
            .finish()
    }
}

/// Holds one call in flight; released on drop, also when unwinding.
struct Flight<'a>(&'a AtomicUsize);

impl Drop for Flight<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Release);
    }
}

/// Resolve a `TAPP_executor` handle to the executor it names.
///
/// This is the Rust-level seam for the sibling C-API crates of this stack
/// (`cpueinsum-capi`, `tlinalg-capi`): they accept the same handle, so a host's
/// pool is shared with them instead of being replaced by a second one. Zero
/// names no executor, which every caller treats as a serial call.
///
/// # Safety
///
/// `handle` is zero or a live `TAPP_executor` from this library, and the
/// returned reference is not used after that executor is destroyed.
pub unsafe fn executor_from_handle<'a>(handle: TAPP_executor) -> Option<&'a Executor> {
    // SAFETY: forwarded contract.
    unsafe { resolve(handle) }
}

impl Executor {
    fn serial() -> Box<Self> {
        Box::new(Self {
            inflight: AtomicUsize::new(0),
            kind: Kind::Serial(ArenaProvider::new()),
        })
    }

    /// A pool of `nthreads >= 2` workers. `hook` runs for each worker's index
    /// before it is spawned and may refuse (a failed OS spawn); the value it
    /// returns lives as long as that worker does.
    fn pooled(
        nthreads: usize,
        stack: usize,
        hook: impl Fn(usize) -> std::io::Result<Box<dyn std::any::Any + Send>> + Send + Sync + 'static,
    ) -> Result<Box<Self>, FfiError> {
        let (tp, joins) = build_workers(nthreads, stack, hook)?;
        Ok(Box::new(Self {
            inflight: AtomicUsize::new(0),
            kind: Kind::Pool {
                pool: Box::new(Pool::owned(tp)),
                joins: Mutex::new(joins),
                budget: AtomicUsize::new(nthreads),
            },
        }))
    }

    /// An executor that borrows `pool` (a raw pool the host owns) with `budget`
    /// clamped to its width. It keeps one `Pool` wrapper of that raw pool, so
    /// the pool's workspace and SPMD gate belong to this executor; the host
    /// creates one borrowed executor and shares it across plans.
    fn borrowing(pool: Arc<rayon::ThreadPool>, budget: usize) -> Box<Self> {
        let pool = Pool::shared(pool);
        let width = pool.size().max(1);
        Box::new(Self {
            inflight: AtomicUsize::new(0),
            kind: Kind::Pool {
                pool: Box::new(pool),
                joins: Mutex::new(Vec::new()),
                budget: AtomicUsize::new(budget.min(width)),
            },
        })
    }

    /// Run `f` with an [`Exec`] for this executor, holding it in flight. The
    /// budget is read once, here: a later `set_budget` does not affect `f`.
    ///
    /// # Errors
    ///
    /// Whatever `f` returns.
    pub fn with<R>(&self, f: impl FnOnce(&Exec<'_>) -> Result<R, FfiError>) -> Result<R, FfiError> {
        self.inflight.fetch_add(1, Ordering::Acquire);
        let _flight = Flight(&self.inflight);
        match &self.kind {
            Kind::Serial(workspace) => f(&Exec::serial_with_workspace(workspace)),
            Kind::Pool { pool, budget, .. } => {
                let exec = Exec::rayon(pool)
                    .with_budget(budget.load(Ordering::Relaxed))
                    .map_err(|e| FfiError::new(TCONTRACT_ERR_INVALID_ARGUMENT, e.to_string()))?;
                f(&exec)
            }
        }
    }

    /// Run `f` with this executor's pool and budget.
    ///
    /// This is the seam for a provider that takes a `rayon::ThreadPool` directly
    /// rather than an [`Exec`] (`tlinalg-capi`): it runs on the host's pool, and
    /// the in-flight guard keeps `tcontract_pool_destroy` from releasing that
    /// pool under the call. `None` means a serial executor, which has no pool.
    pub fn with_pool<R>(&self, f: impl FnOnce(Option<(&rayon::ThreadPool, usize)>) -> R) -> R {
        self.inflight.fetch_add(1, Ordering::Acquire);
        let _flight = Flight(&self.inflight);
        match &self.kind {
            Kind::Serial(_) => f(None),
            Kind::Pool { pool, budget, .. } => {
                let width = pool.size().max(1);
                let budget = budget.load(Ordering::Relaxed).clamp(1, width);
                f(Some((pool.rayon_pool(), budget)))
            }
        }
    }

    /// `(pool_size, budget)`: `(0, 1)` for a serial executor. The budget is an
    /// upper bound on a call's width, not the active width.
    pub fn threads(&self) -> (usize, usize) {
        match &self.kind {
            Kind::Serial(_) => (0, 1),
            Kind::Pool { pool, budget, .. } => (
                pool.size(),
                budget.load(Ordering::Relaxed).clamp(1, pool.size().max(1)),
            ),
        }
    }

    /// Entry counters of the owned pool, or `None` for a serial executor
    /// (which has no pool to enter). For tests and benchmarks.
    pub fn pool_stats(&self) -> Option<PoolStats> {
        match &self.kind {
            Kind::Serial(_) => None,
            Kind::Pool { pool, .. } => Some(pool.stats()),
        }
    }

    /// Whether the calling thread is a worker of this executor's pool.
    fn on_own_worker(&self) -> bool {
        match &self.kind {
            Kind::Serial(_) => false,
            Kind::Pool { pool, .. } => Exec::rayon(pool).is_worker(),
        }
    }

    /// Stop and join an owned pool, or release a borrowed one, then free.
    /// Consumes the box only on success.
    fn destroy(this: *mut Executor) -> Result<(), FfiError> {
        // SAFETY: `this` is a live executor handle (caller contract).
        let e = unsafe { &*this };
        // A worker cannot join its own pool.
        if e.on_own_worker() {
            return Err(FfiError::new(
                TCONTRACT_ERR_WOULD_DEADLOCK,
                "executor destroyed from one of its own workers",
            ));
        }
        if e.inflight.load(Ordering::Acquire) != 0 {
            return Err(FfiError::new(
                TCONTRACT_BUSY,
                "executor has calls in flight",
            ));
        }
        // SAFETY: nothing is in flight and the caller synchronizes new calls
        // with this destruction; the handle was produced by `Box::into_raw`.
        let boxed = unsafe { Box::from_raw(this) };
        if let Kind::Pool { pool, joins, .. } = boxed.kind {
            // Take the join handles before shutdown starts, join after: when
            // this returns no tcontract worker code, TLS teardown included, runs.
            let handles = joins
                .into_inner()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            drop(pool.into_owned()); // starts worker shutdown
            for h in handles {
                let _ = h.join();
            }
        }
        Ok(())
    }
}

/// A host-owned Rayon pool behind a `tcontract_pool` (opaque in C), built by
/// [`tcontract_pool_create_rayon`].
///
/// The handle holds the only `Arc<rayon::ThreadPool>` of a dedicated pool;
/// every borrowed executor holds one clone, so `Arc::strong_count` is one plus
/// the number of live borrowers and [`PoolHandle::destroy`] can refuse while
/// one is live. The join handles are the OS threads the custom `spawn_handler`
/// created: with that handler Rayon keeps no `JoinHandle` of its own, so
/// dropping the `ThreadPool` signals shutdown but joins nothing. They are set
/// once at creation and only read at destruction, hence a plain `Vec`.
struct PoolHandle {
    pool: Arc<rayon::ThreadPool>,
    joins: Vec<JoinHandle<()>>,
}

impl std::fmt::Debug for PoolHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PoolHandle")
            .field("size", &self.pool.current_num_threads())
            .field("borrowers", &(Arc::strong_count(&self.pool) - 1))
            .finish()
    }
}

impl PoolHandle {
    /// Build the pool: workers are spawned through the same handler as
    /// [`tcontract_tapp_executor_create_rayon`], so worker names and `stack`
    /// are identical.
    fn create(nthreads: usize, stack: usize) -> Result<Box<Self>, FfiError> {
        let (tp, joins) = build_workers(nthreads, stack, |_| Ok(Box::new(())))?;
        Ok(Box::new(Self {
            pool: Arc::new(tp),
            joins,
        }))
    }

    /// Stop, join and free the pool. Consumes the box only on success.
    fn destroy(handle: tcontract_pool) -> Result<(), FfiError> {
        if handle == 0 {
            return Err(null("pool"));
        }
        // SAFETY: `handle` is a live pool handle (caller contract).
        let h = unsafe { &*(handle as *const Self) };
        // A worker cannot join its own pool.
        if h.pool.current_thread_index().is_some() {
            return Err(FfiError::new(
                TCONTRACT_ERR_WOULD_DEADLOCK,
                "pool destroyed from one of its own workers",
            ));
        }
        // INVARIANT: this handle owns one Arc of its own pool and each live
        // borrowed executor holds exactly one clone, so a count above one means
        // an executor is still live. Nothing else clones this `Arc`.
        if Arc::strong_count(&h.pool) != 1 {
            return Err(FfiError::new(
                TCONTRACT_BUSY,
                "pool is still borrowed by a live executor",
            ));
        }
        // SAFETY: no borrowed executor is live and the caller synchronizes
        // destruction with new borrows; the handle was produced by
        // `Box::into_raw`.
        let boxed = unsafe { Box::from_raw(handle as *mut Self) };
        let Self { pool, joins } = *boxed;
        drop(pool); // the last Arc: starts worker shutdown
        for h in joins {
            let _ = h.join();
        }
        Ok(())
    }
}

/// Spawn `nthreads` Rayon workers on `stack`-sized threads, `hook(i)` before
/// worker `i` starts. Returns the pool and the join handles of the OS threads,
/// which the caller needs because a custom `spawn_handler` leaves Rayon without
/// any of its own. On failure every worker already started is stopped and
/// joined before the error is returned, thread-exit work included.
fn build_workers(
    nthreads: usize,
    stack: usize,
    hook: impl Fn(usize) -> std::io::Result<Box<dyn std::any::Any + Send>> + Send + Sync + 'static,
) -> Result<(rayon::ThreadPool, Vec<JoinHandle<()>>), FfiError> {
    let joins = Arc::new(Mutex::new(Vec::new()));
    let j2 = joins.clone();
    let built = rayon::ThreadPoolBuilder::new()
        .num_threads(nthreads)
        .spawn_handler(move |t| {
            let held = hook(t.index())?;
            let h = std::thread::Builder::new()
                .name(format!("tcontract-worker-{}", t.index()))
                .stack_size(stack)
                .spawn(move || {
                    let _held = held;
                    t.run()
                })?;
            j2.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(h);
            Ok(())
        })
        .build();
    let mut handles = std::mem::take(
        &mut *joins
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    );
    match built {
        Ok(tp) => Ok((tp, handles)),
        Err(e) => {
            // Rayon terminated the workers it had started; wait for them
            // (and their TLS teardown) so a failed create leaves no thread.
            for h in handles.drain(..) {
                let _ = h.join();
            }
            Err(FfiError::new(TCONTRACT_ERR_INTERNAL, e.to_string()))
        }
    }
}

/// The executor behind a handle; `None` for the default (zero).
///
/// # Safety
///
/// `e` is zero or a live handle from this library.
unsafe fn resolve<'a>(e: TAPP_executor) -> Option<&'a Executor> {
    if e == 0 {
        None
    } else {
        // SAFETY: per the contract.
        Some(unsafe { &*(e as *const Executor) })
    }
}

/// Run `f` on the executor behind `exec` (zero: the default serial executor).
///
/// # Safety
///
/// `exec` is zero or a live handle, not destroyed during the call.
pub unsafe fn with_executor<R>(
    exec: TAPP_executor,
    f: impl FnOnce(&Exec<'_>) -> Result<R, FfiError>,
) -> Result<R, FfiError> {
    // SAFETY: forwarded contract.
    match unsafe { resolve(exec) } {
        Some(e) => e.with(f),
        None => f(&Exec::serial()),
    }
}

/// Borrow the [`Executor`] behind a nonzero handle, for diagnostics.
///
/// # Safety
///
/// `exec` is a live handle from this library, not destroyed while borrowed.
pub unsafe fn executor_ref<'a>(exec: TAPP_executor) -> Option<&'a Executor> {
    // SAFETY: forwarded contract.
    unsafe { resolve(exec) }
}

fn out_handle(out: *mut TAPP_executor, e: Box<Executor>) -> Result<(), FfiError> {
    // SAFETY: non-null, checked by the callers, writable per the contract.
    unsafe { *out = Box::into_raw(e) as TAPP_executor };
    Ok(())
}

/// Reject a null `TAPP_executor` or `tcontract_pool` out-parameter (`what`
/// names which, for the message).
fn null_out(out: *mut isize, what: &str) -> Result<(), FfiError> {
    if out.is_null() {
        Err(FfiError::new(
            TCONTRACT_ERR_INVALID_ARGUMENT,
            format!("null {what} out-parameter"),
        ))
    } else {
        Ok(())
    }
}

/// Reject a zero worker count: it is not a request for the environment's width.
fn nonzero(nthreads: usize) -> Result<(), FfiError> {
    if nthreads == 0 {
        Err(FfiError::new(
            TCONTRACT_ERR_INVALID_ARGUMENT,
            "nthreads must be at least 1",
        ))
    } else {
        Ok(())
    }
}

/// The worker stack size `opts` asks for; null `opts` or `stack_size == 0`
/// means 16 MiB (provider kernels recurse deeply, see tenferro-rs's CPU
/// threading contract).
///
/// # Safety
///
/// `opts` is null or valid.
unsafe fn stack_size(opts: *const tcontract_rayon_opts) -> usize {
    // SAFETY: null or valid per the contract.
    let o = if opts.is_null() {
        tcontract_rayon_opts::default()
    } else {
        unsafe { *opts }
    };
    if o.stack_size == 0 {
        16 << 20
    } else {
        o.stack_size
    }
}

/// `TAPP_create_executor`: a serial executor (no workers). Destroy it with
/// [`TAPP_destroy_executor`].
///
/// # Safety
///
/// `exec` is null or writable.
#[no_mangle]
pub unsafe extern "C" fn TAPP_create_executor(exec: *mut TAPP_executor) -> c_int {
    ffi(|| {
        null_out(exec, "executor")?;
        out_handle(exec, Executor::serial())
    })
}

/// `TAPP_destroy_executor`: destroy a serial executor, or stop and join the
/// pool an owned executor created with [`tcontract_tapp_executor_create_rayon`]
/// and free it. An executor built with
/// [`tcontract_tapp_executor_borrow_pool`] releases only its borrow of the
/// host's pool; the pool itself is destroyed by [`tcontract_pool_destroy`].
/// Zero is the default executor: success, nothing to do.
///
/// `TCONTRACT_BUSY` while calls are in flight and `TCONTRACT_ERR_WOULD_DEADLOCK`
/// from one of the executor's own workers; both leave the handle live, so the
/// caller may retry. A live handle is destroyed successfully once: double
/// destruction and stale handles are unsupported, and the caller must
/// synchronize destruction with the start of new calls.
///
/// # Safety
///
/// `exec` is zero or a live handle from this library.
#[no_mangle]
pub unsafe extern "C" fn TAPP_destroy_executor(exec: TAPP_executor) -> c_int {
    ffi(|| {
        if exec == 0 {
            return Ok(());
        }
        Executor::destroy(exec as *mut Executor)
    })
}

/// Create an executor that owns a private Rayon pool of `nthreads` workers,
/// reusable across plans and shared by every part of the library.
/// `nthreads == 0` is an error; `nthreads == 1` is a serial executor with no
/// workers. The width is never inferred from the environment. `opts` may be
/// null.
///
/// # Safety
///
/// `out` is null or writable; `opts` is null or valid.
#[no_mangle]
pub unsafe extern "C" fn tcontract_tapp_executor_create_rayon(
    out: *mut TAPP_executor,
    nthreads: usize,
    opts: *const tcontract_rayon_opts,
) -> c_int {
    ffi(|| {
        null_out(out, "executor")?;
        nonzero(nthreads)?;
        // SAFETY: null or valid per the contract.
        let stack = unsafe { stack_size(opts) };
        let e = if nthreads == 1 {
            Executor::serial()
        } else {
            Executor::pooled(nthreads, stack, |_| Ok(Box::new(())))?
        };
        out_handle(out, e)
    })
}

/// Create a Rayon pool the *host* owns and shares: an opaque handle that
/// [`tcontract_tapp_executor_borrow_pool`] can lend to any number of executors,
/// so a C host that manages its own threads keeps one pool instead of two.
/// `nthreads == 0` is an error. Unlike
/// [`tcontract_tapp_executor_create_rayon`], `nthreads == 1` is a real
/// one-worker pool: the handle is a pool, not an executor. `opts` may be null;
/// `stack_size` is honoured exactly as in
/// [`tcontract_tapp_executor_create_rayon`]. Free the handle with
/// [`tcontract_pool_destroy`].
///
/// Each borrowed executor keeps its own `Pool` wrapper of the shared raw pool
/// (its own workspace and SPMD gate), so one executor shared across plans is
/// what keeps `tcontract-exec`'s one-wrapper-per-raw-pool rule.
///
/// # Safety
///
/// `out` is null or writable; `opts` is null or valid.
#[no_mangle]
pub unsafe extern "C" fn tcontract_pool_create_rayon(
    out: *mut tcontract_pool,
    nthreads: usize,
    opts: *const tcontract_rayon_opts,
) -> c_int {
    ffi(|| {
        null_out(out, "pool")?;
        nonzero(nthreads)?;
        // SAFETY: null or valid per the contract.
        let stack = unsafe { stack_size(opts) };
        let handle = PoolHandle::create(nthreads, stack)?;
        // SAFETY: non-null and writable per the contract.
        unsafe { *out = Box::into_raw(handle) as tcontract_pool };
        Ok(())
    })
}

/// `tcontract_pool_destroy`: stop and join a pool from
/// [`tcontract_pool_create_rayon`], then free the handle. Zero is an error.
///
/// `TCONTRACT_BUSY` while an executor built from the pool is live, and
/// `TCONTRACT_ERR_WOULD_DEADLOCK` from one of the pool's own workers; both leave
/// the handle live, so the caller may destroy the executor and retry. A live
/// handle is destroyed successfully once: double destruction and stale handles
/// are unsupported, and the caller synchronizes destruction with the start of
/// new borrows.
///
/// # Safety
///
/// `pool` is zero or a live handle from this library.
#[no_mangle]
pub unsafe extern "C" fn tcontract_pool_destroy(pool: tcontract_pool) -> c_int {
    ffi(|| PoolHandle::destroy(pool))
}

/// Create an executor that borrows `pool` instead of creating a private one:
/// it runs on the host's pool, so the process keeps one pool and the width is
/// the pool's. `budget >= 1` is the most threads a call may use, clamped to the
/// pool width exactly as [`tcontract_tapp_executor_set_budget`] clamps it.
/// Destroy the executor with [`TAPP_destroy_executor`], which releases the
/// borrow and never stops the borrowed pool. The pool must outlive the executor
/// and every call through it. Build one executor per pool and share it across
/// plans; a second executor on the same pool gets its own workspace and SPMD
/// gate, so concurrent numerical work belongs on one of them.
///
/// # Safety
///
/// `out` is null or writable; `pool` is a live handle from
/// [`tcontract_pool_create_rayon`] that outlives the executor built here.
#[no_mangle]
pub unsafe extern "C" fn tcontract_tapp_executor_borrow_pool(
    out: *mut TAPP_executor,
    pool: tcontract_pool,
    budget: usize,
) -> c_int {
    ffi(|| {
        null_out(out, "executor")?;
        if budget == 0 {
            return Err(FfiError::new(
                TCONTRACT_ERR_INVALID_ARGUMENT,
                "budget must be at least 1",
            ));
        }
        if pool == 0 {
            return Err(null("pool"));
        }
        // SAFETY: a live pool handle per the contract.
        let h = unsafe { &*(pool as *const PoolHandle) };
        out_handle(out, Executor::borrowing(Arc::clone(&h.pool), budget))
    })
}

/// Set the thread budget of calls started afterwards (clamped to the pool
/// width; a serial executor stays at 1). The pool is not resized.
///
/// # Safety
///
/// `exec` is zero or a live handle.
#[no_mangle]
pub unsafe extern "C" fn tcontract_tapp_executor_set_budget(
    exec: TAPP_executor,
    budget: usize,
) -> c_int {
    ffi(|| {
        if budget == 0 {
            return Err(FfiError::new(
                TCONTRACT_ERR_INVALID_ARGUMENT,
                "budget must be at least 1",
            ));
        }
        // SAFETY: per the contract.
        if let Some(Executor {
            kind: Kind::Pool {
                pool, budget: b, ..
            },
            ..
        }) = unsafe { resolve(exec) }
        {
            b.store(budget.min(pool.size().max(1)), Ordering::Relaxed);
        }
        Ok(())
    })
}

/// Report the pool width and the budget; either out-parameter may be null. A
/// serial executor (including zero) reports `pool_size == 0`, `budget == 1`.
/// The budget bounds a call's width; it is not the active width or an
/// affinity promise.
///
/// # Safety
///
/// `exec` is zero or a live handle; the out-parameters are null or writable.
#[no_mangle]
pub unsafe extern "C" fn tcontract_tapp_executor_get_threads(
    exec: TAPP_executor,
    pool_size: *mut usize,
    budget: *mut usize,
) -> c_int {
    ffi(|| {
        // SAFETY: per the contract.
        let (p, b) = unsafe { resolve(exec) }.map_or((0, 1), Executor::threads);
        // SAFETY: null or writable per the contract.
        unsafe {
            if !pool_size.is_null() {
                *pool_size = p;
            }
            if !budget.is_null() {
                *budget = b;
            }
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use super::*;

    /// A worker's exit takes this long to finish; `Executor::pooled` must
    /// not return before it did, also when it fails.
    struct SlowExit(Arc<AtomicUsize>);

    impl Drop for SlowExit {
        fn drop(&mut self) {
            std::thread::sleep(std::time::Duration::from_millis(50));
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }

    /// A spawn failure at worker `at` joins the workers already started,
    /// including their thread-exit work.
    #[test]
    fn failed_creation_reclaims_started_workers() {
        for at in [1usize, 2, 3] {
            let alive = Arc::new(AtomicUsize::new(0));
            let a2 = alive.clone();
            let err = Executor::pooled(4, 1 << 20, move |i| {
                if i == at {
                    return Err(std::io::Error::other("injected spawn failure"));
                }
                a2.fetch_add(1, Ordering::SeqCst);
                Ok(Box::new(SlowExit(a2.clone())))
            })
            .unwrap_err();
            assert_eq!(err.status, TCONTRACT_ERR_INTERNAL);
            assert_eq!(
                alive.load(Ordering::SeqCst),
                0,
                "creation failed at worker {at} but returned before the started workers finished"
            );
        }
    }

    /// A successful creation does not wait for or touch the hook values.
    #[test]
    fn successful_creation_keeps_workers_until_destroy() {
        let alive = Arc::new(AtomicUsize::new(0));
        let a2 = alive.clone();
        let e = Executor::pooled(3, 1 << 20, move |_| {
            a2.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(SlowExit(a2.clone())))
        })
        .unwrap();
        assert_eq!(alive.load(Ordering::SeqCst), 3);
        assert_eq!(e.threads(), (3, 3));
        Executor::destroy(Box::into_raw(e)).unwrap();
        assert_eq!(alive.load(Ordering::SeqCst), 0, "destroy returned early");
    }
}
