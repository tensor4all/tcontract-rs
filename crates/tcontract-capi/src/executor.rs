//! Executors: the `TAPP_executor` of the TAPP C API, serial or owning a worker
//! pool that tcontract creates and stops for C hosts without one, plus the
//! host-owned pool handle a C host shares with it.
//!
//! A `TAPP_executor` is an `intptr_t` holding a pointer to an [`Executor`];
//! zero selects the default serial executor (a tcontract policy, not in the
//! pinned TAPP headers). `TAPP_create_executor` makes a serial executor;
//! `tcontract_tapp_executor_create_rayon` makes one that owns a pool;
//! `TAPP_destroy_executor` stops that pool and waits for its workers to exit. Operations of every
//! part of `libtcontract_capi` (TAPP products) take the same executor.
//!
//! A `tcontract_pool` is the same discipline for a pool the *host* owns:
//! `tcontract_pool_create_rayon` builds it, `tcontract_pool_destroy` stops it and
//! waits for its workers, and `tcontract_tapp_executor_borrow_pool` makes an executor that
//! runs on it instead of creating a second pool. The host keeps the pool alive
//! for every executor built from it and every call through one.
//!
//! The pools are [`strided_exec::Pool`]s: the C ABI never takes a pool made
//! elsewhere. Rust hosts do not use this module: they own a
//! [`strided_exec::Pool`], lend it to [`tcontract_exec::Pool::borrow`] and never
//! create a second one.
#![allow(non_camel_case_types)]

use std::cell::Cell;
use std::ffi::c_int;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use ::tcontract_exec::{ArenaProvider, Exec, Pool};

use crate::abi::null;
use crate::status::*;

/// `TAPP_executor`: `intptr_t`, zero meaning the default serial executor.
pub type TAPP_executor = isize;

/// `tcontract_pool`: `intptr_t` holding a host-owned worker pool from
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
        // The executor's `Pool` wrapper: it carries this executor's workspace,
        // so every call through the executor goes through it. One executor per
        // pool is the intended configuration (create it once, share it across
        // plans); when a host pool is lent to several borrowed executors, each
        // one keeps its own workspace, so concurrent numerical work belongs on
        // one of them.
        pool: Box<Pool<'static>>,
        /// Present for an owned pool, whose workers `destroy` refuses to be
        /// called from.
        worker_id: Option<usize>,
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

    /// A pool of `nthreads >= 2` workers that this executor owns.
    fn pooled(nthreads: usize, stack: usize) -> Result<Box<Self>, FfiError> {
        let (pool, worker_id) = Workers::start(nthreads, stack)?;
        Ok(Box::new(Self {
            inflight: AtomicUsize::new(0),
            kind: Kind::Pool {
                pool: Box::new(Pool::shared(pool)),
                worker_id: Some(worker_id),
                budget: AtomicUsize::new(nthreads),
            },
        }))
    }

    /// An executor that borrows `pool` (a pool the host owns) with `budget`
    /// clamped to its width. It keeps one `Pool` wrapper of that pool, so the
    /// workspace belongs to this executor; the host creates one borrowed
    /// executor and shares it across plans.
    fn borrowing(pool: Arc<strided_exec::Pool>, budget: usize) -> Box<Self> {
        let pool = Pool::shared(pool);
        let width = pool.size().max(1);
        Box::new(Self {
            inflight: AtomicUsize::new(0),
            kind: Kind::Pool {
                pool: Box::new(pool),
                worker_id: None,
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
    pub fn with<R>(
        &self,
        f: impl FnOnce(&mut Exec<'_>) -> Result<R, FfiError>,
    ) -> Result<R, FfiError> {
        self.inflight.fetch_add(1, Ordering::Acquire);
        let _flight = Flight(&self.inflight);
        match &self.kind {
            Kind::Serial(workspace) => f(&mut Exec::serial_with_workspace(workspace)),
            Kind::Pool { pool, budget, .. } => {
                let mut exec = Exec::new(pool, budget.load(Ordering::Relaxed))
                    .map_err(|e| FfiError::new(TCONTRACT_ERR_INVALID_ARGUMENT, e.to_string()))?;
                f(&mut exec)
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

    /// Whether the calling thread is a worker of this executor's pool.
    fn on_own_worker(&self) -> bool {
        match &self.kind {
            Kind::Serial(_) => false,
            Kind::Pool { worker_id, .. } => worker_id.is_some_and(on_worker_of),
        }
    }

    /// Stop an owned pool and wait for its workers, or release a borrowed one,
    /// then free. Consumes the box only on success.
    fn destroy(this: *mut Executor) -> Result<(), FfiError> {
        // SAFETY: `this` is a live executor handle (caller contract).
        let e = unsafe { &*this };
        // Dropping the pool joins its workers, which a worker cannot do to itself.
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
        // Dropping an owned pool joins its workers, thread-local destructors
        // included, so none of this library's code runs after `destroy` returns.
        drop(boxed);
        Ok(())
    }
}

/// A host-owned worker pool behind a `tcontract_pool` (opaque in C), built by
/// [`tcontract_pool_create_rayon`].
///
/// The handle holds the only `Arc<strided_exec::Pool>` of a dedicated pool;
/// every borrowed executor holds one clone, so `Arc::strong_count` is one plus
/// the number of live borrowers and [`PoolHandle::destroy`] can refuse while
/// one is live. `worker_id` names the pool's workers (see [`on_worker_of`]).
struct PoolHandle {
    pool: Arc<strided_exec::Pool>,
    worker_id: usize,
}

impl std::fmt::Debug for PoolHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PoolHandle")
            .field("size", &self.pool.width())
            .field("borrowers", &(Arc::strong_count(&self.pool) - 1))
            .finish()
    }
}

impl PoolHandle {
    /// Build the pool: workers are started exactly as for
    /// [`tcontract_tapp_executor_create_rayon`], so worker names and `stack`
    /// are identical.
    fn create(nthreads: usize, stack: usize) -> Result<Box<Self>, FfiError> {
        let (pool, worker_id) = Workers::start(nthreads, stack)?;
        Ok(Box::new(Self { pool, worker_id }))
    }

    /// Stop the pool, wait for its workers and free the handle. Consumes the
    /// box only on success.
    fn destroy(handle: tcontract_pool) -> Result<(), FfiError> {
        if handle == 0 {
            return Err(null("pool"));
        }
        // SAFETY: `handle` is a live pool handle (caller contract).
        let h = unsafe { &*(handle as *const Self) };
        // Dropping the pool joins its workers, which a worker cannot do to itself.
        if on_worker_of(h.worker_id) {
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
        // The last `Arc`: dropping the pool joins its workers, thread-local
        // destructors included.
        drop(boxed);
        Ok(())
    }
}

/// Identity of the pool whose worker the current thread is (zero: none),
/// installed by each worker when it starts. It exists only so that `destroy`
/// can refuse to run on one of the pool's own workers.
static NEXT_POOL: AtomicUsize = AtomicUsize::new(1);

thread_local! {
    static WORKER_OF: Cell<usize> = const { Cell::new(0) };
}

/// Whether the calling thread is a worker of the pool named `id`.
fn on_worker_of(id: usize) -> bool {
    WORKER_OF.try_with(Cell::get).is_ok_and(|w| w == id)
}

/// Starts a pool of workers named `tcontract-worker-<i>` on `stack`-sized threads.
struct Workers;

impl Workers {
    /// Build the pool; the second value names its workers for [`on_worker_of`].
    fn start(nthreads: usize, stack: usize) -> Result<(Arc<strided_exec::Pool>, usize), FfiError> {
        let id = NEXT_POOL.fetch_add(1, Ordering::Relaxed);
        let pool = strided_exec::Pool::builder()
            .threads(nthreads)
            .stack_size(stack)
            .thread_name(|i| format!("tcontract-worker-{i}"))
            .on_start(move |_| {
                // INVARIANT: the hook runs once per worker, on its own thread.
                let _ = WORKER_OF.try_with(|w| w.set(id));
            })
            .build()
            .map_err(|e| FfiError::new(TCONTRACT_ERR_INTERNAL, e.to_string()))?;
        Ok((Arc::new(pool), id))
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
    f: impl FnOnce(&mut Exec<'_>) -> Result<R, FfiError>,
) -> Result<R, FfiError> {
    // SAFETY: forwarded contract.
    match unsafe { resolve(exec) } {
        Some(e) => e.with(f),
        None => f(&mut Exec::serial()),
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

/// Create an executor that owns a private worker pool of `nthreads` workers,
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
            Executor::pooled(nthreads, stack)?
        };
        out_handle(out, e)
    })
}

/// Create a worker pool the *host* owns and shares: an opaque handle that
/// [`tcontract_tapp_executor_borrow_pool`] can lend to any number of executors,
/// so a C host that manages its own threads keeps one pool instead of two.
/// `nthreads == 0` is an error. Unlike
/// [`tcontract_tapp_executor_create_rayon`], `nthreads == 1` is a real
/// one-worker pool: the handle is a pool, not an executor. `opts` may be null;
/// `stack_size` is honoured exactly as in
/// [`tcontract_tapp_executor_create_rayon`]. Free the handle with
/// [`tcontract_pool_destroy`].
///
/// Each borrowed executor keeps its own `Pool` wrapper of the shared pool
/// (its own workspace), so one executor shared across plans is
/// what keeps `tcontract-exec`'s one-wrapper-per-pool rule.
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
/// plans; a second executor on the same pool gets its own workspace, so concurrent
/// numerical work belongs on one of them.
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
    use super::*;

    /// `destroy` returns only after every worker of an owned pool has exited,
    /// thread-exit work included: a thread-local destructor registered at
    /// worker start has run by then.
    #[test]
    fn destroy_waits_for_every_worker_to_exit() {
        use std::sync::atomic::AtomicUsize as Count;
        thread_local! { static GUARD: std::cell::RefCell<Option<Guard>> = const { std::cell::RefCell::new(None) }; }
        static FINISHED: Count = Count::new(0);
        struct Guard;
        impl Drop for Guard {
            fn drop(&mut self) {
                // Slow enough that an early return would be seen.
                std::thread::sleep(std::time::Duration::from_millis(50));
                FINISHED.fetch_add(1, Ordering::SeqCst);
            }
        }
        let pool = strided_exec::Pool::builder()
            .threads(3)
            .on_start(|_| {
                let _ = GUARD.try_with(|g| *g.borrow_mut() = Some(Guard));
            })
            .build()
            .unwrap();
        let e = Box::new(Executor {
            inflight: AtomicUsize::new(0),
            kind: Kind::Pool {
                pool: Box::new(Pool::shared(Arc::new(pool))),
                worker_id: None,
                budget: AtomicUsize::new(3),
            },
        });
        Executor::destroy(Box::into_raw(e)).unwrap();
        assert_eq!(FINISHED.load(Ordering::SeqCst), 3, "destroy returned early");
    }

    /// A worker is recognised as a member of its own pool and of no other.
    #[test]
    fn a_worker_is_a_member_of_its_own_pool_only() {
        let a = Executor::pooled(2, 1 << 20).unwrap();
        let b = Executor::pooled(2, 1 << 20).unwrap();
        assert!(!a.on_own_worker() && !b.on_own_worker());
        let pool_a = match &a.kind {
            Kind::Pool { pool, .. } => pool,
            Kind::Serial(_) => unreachable!("pooled executor"),
        };
        let seen = std::sync::Mutex::new(Vec::new());
        let mut exec = Exec::new(pool_a, 2).unwrap();
        exec.token().fork(2, |_, _| {
            seen.lock()
                .unwrap()
                .push((a.on_own_worker(), b.on_own_worker()));
        });
        // Never a member of the other pool; at least one job ran on a worker of
        // this one (the caller may take the other).
        let seen = seen.into_inner().unwrap();
        assert!(seen.iter().all(|&(_, in_b)| !in_b));
        assert!(seen.iter().any(|&(in_a, _)| in_a));
        Executor::destroy(Box::into_raw(a)).unwrap();
        Executor::destroy(Box::into_raw(b)).unwrap();
    }
}
