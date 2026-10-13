//! A Rayon pool the *host* owns, lent to a TAPP executor through
//! `tcontract_pool_create_rayon` / `tcontract_tapp_executor_borrow_pool`: work
//! runs on that pool (not on the global one, and not on a second pool), the
//! budget is clamped to its width, and destroying it while a borrowed executor
//! is live is refused.
mod raw;

use std::collections::HashSet;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Mutex;

use raw::*;
use tcontract_capi::executor::{with_executor, TAPP_executor};
use tcontract_capi::*;

unsafe fn pool(nthreads: usize) -> tcontract_pool {
    let mut p = 0;
    assert_eq!(
        tcontract_pool_create_rayon(&mut p, nthreads, std::ptr::null()),
        TAPP_SUCCESS
    );
    p
}

unsafe fn borrow(pool: tcontract_pool, budget: usize) -> TAPP_executor {
    let mut e = 0;
    assert_eq!(
        tcontract_tapp_executor_borrow_pool(&mut e, pool, budget),
        TAPP_SUCCESS
    );
    e
}

/// The workers of the executor's pool, as `(index, thread id)`.
unsafe fn workers(exec: TAPP_executor, width: usize) -> HashSet<(usize, std::thread::ThreadId)> {
    let seen = Mutex::new(HashSet::new());
    with_executor(exec, |x| {
        x.broadcast(width, &|t| {
            seen.lock()
                .unwrap()
                .insert((t, std::thread::current().id()));
        })
        .unwrap();
        Ok(())
    })
    .unwrap();
    seen.into_inner().unwrap()
}

#[test]
fn a_borrowed_executor_runs_on_the_hosts_pool() {
    const WIDTH: usize = 3;
    unsafe {
        let p = pool(WIDTH);
        let e = borrow(p, WIDTH);
        let on_pool = workers(e, WIDTH);

        // The effective width is the pool's, on its own workers.
        assert_eq!(on_pool.len(), WIDTH, "not the pool's width: {on_pool:?}");
        assert_eq!(
            on_pool.iter().map(|(i, _)| *i).collect::<HashSet<_>>(),
            (0..WIDTH).collect::<HashSet<_>>()
        );
        let ids: HashSet<_> = on_pool.iter().map(|(_, id)| *id).collect();

        // Those are not the global pool's threads: the call neither fell back to
        // the global Rayon pool nor created a pool of its own.
        let global = Mutex::new(HashSet::new());
        rayon::broadcast(|_| {
            global.lock().unwrap().insert(std::thread::current().id());
        });
        assert!(
            ids.is_disjoint(&global.into_inner().unwrap()),
            "the borrowed executor ran on the global Rayon pool"
        );

        // A second executor on the same handle is the same pool, not a new one.
        let e2 = borrow(p, WIDTH);
        assert_eq!(
            workers(e2, WIDTH)
                .iter()
                .map(|(_, id)| *id)
                .collect::<HashSet<_>>(),
            ids,
            "two borrowed executors must share one pool"
        );

        assert_eq!(TAPP_destroy_executor(e), TAPP_SUCCESS);
        assert_eq!(TAPP_destroy_executor(e2), TAPP_SUCCESS);
        assert_eq!(tcontract_pool_destroy(p), TAPP_SUCCESS);
    }
}

#[test]
fn a_product_runs_on_the_borrowed_pool() {
    let (m, n, k) = (200usize, 190usize, 180usize);
    let (ea, eb, ed) = (
        [m as i64, k as i64],
        [k as i64, n as i64],
        [m as i64, n as i64],
    );
    let (sa, sb, sd) = (cm(&ea), cm(&eb), cm(&ed));
    let (la, lb, ld) = ([I, K], [K, J], [I, J]);
    unsafe {
        let (rc, plan) = plan_f64(
            (&ea, &sa, &la),
            (&eb, &sb, &lb),
            (&ed, &sd, &ld),
            (&ed, &sd, &ld),
        );
        assert_eq!(rc, TAPP_SUCCESS);
        let a = seq(m * k, 1);
        let b = seq(k * n, 2);
        let want = naive(m, n, k, &a, &b);

        let p = pool(4);
        let e = borrow(p, 4);
        let mut d = vec![f64::NAN; m * n];
        assert_eq!(
            exec_f64(
                plan,
                e,
                1.0,
                a.as_ptr(),
                b.as_ptr(),
                0.0,
                std::ptr::null(),
                d.as_mut_ptr()
            ),
            TAPP_SUCCESS
        );
        assert_close(&d, &want, 1e-12);
        let s = pool_stats(e).unwrap();
        assert_eq!(
            s.entries + s.broadcasts,
            1,
            "the product entered the borrowed pool exactly once"
        );

        assert_eq!(TAPP_destroy_executor(e), TAPP_SUCCESS);
        assert_eq!(tcontract_pool_destroy(p), TAPP_SUCCESS);
        TAPP_destroy_tensor_product(plan);
    }
}

#[test]
fn the_budget_is_clamped_to_the_pool_width() {
    unsafe {
        let p = pool(4);
        let (mut width, mut budget) = (0usize, 0usize);

        // Above the width: clamped, as set_budget clamps it.
        let e = borrow(p, 99);
        assert_eq!(
            tcontract_tapp_executor_get_threads(e, &mut width, &mut budget),
            TAPP_SUCCESS
        );
        assert_eq!((width, budget), (4, 4), "borrow_pool did not clamp");

        // Below it: kept, and a wider broadcast is refused, as set_budget
        // refuses it.
        assert_eq!(tcontract_tapp_executor_set_budget(e, 2), TAPP_SUCCESS);
        assert_eq!(
            tcontract_tapp_executor_get_threads(e, &mut width, &mut budget),
            TAPP_SUCCESS
        );
        assert_eq!((width, budget), (4, 2));
        assert!(
            with_executor(e, |x| Ok(x.broadcast(4, &|_| {}).is_err())).unwrap(),
            "a width above the budget is refused"
        );
        assert!(
            with_executor(e, |x| Ok(x.broadcast(2, &|_| {}).is_ok())).unwrap(),
            "a width within the budget runs"
        );

        // A second borrow keeps its own (clamped) budget.
        let e2 = borrow(p, 3);
        assert_eq!(
            tcontract_tapp_executor_get_threads(e2, &mut width, &mut budget),
            TAPP_SUCCESS
        );
        assert_eq!((width, budget), (4, 3));

        assert_eq!(TAPP_destroy_executor(e), TAPP_SUCCESS);
        assert_eq!(TAPP_destroy_executor(e2), TAPP_SUCCESS);
        assert_eq!(tcontract_pool_destroy(p), TAPP_SUCCESS);
    }
}

#[test]
fn a_one_worker_pool_is_a_real_pool() {
    // Unlike tcontract_tapp_executor_create_rayon(1), which is a serial
    // executor, the handle is the host's pool object: one worker, still a pool.
    unsafe {
        let p = pool(1);
        let e = borrow(p, 1);
        let (mut width, mut budget) = (0usize, 0usize);
        assert_eq!(
            tcontract_tapp_executor_get_threads(e, &mut width, &mut budget),
            TAPP_SUCCESS
        );
        assert_eq!((width, budget), (1, 1));
        assert!(pool_stats(e).is_some(), "a one-worker pool is not serial");
        assert_eq!(TAPP_destroy_executor(e), TAPP_SUCCESS);
        assert_eq!(tcontract_pool_destroy(p), TAPP_SUCCESS);
    }
}

#[test]
fn invalid_arguments_are_rejected() {
    unsafe {
        let mut e: TAPP_executor = 0;
        let mut p: tcontract_pool = 0;

        // Null out-parameter and zero workers.
        assert_eq!(
            tcontract_pool_create_rayon(std::ptr::null_mut(), 2, std::ptr::null()),
            TAPP_ERROR_NULL
        );
        assert_eq!(
            tcontract_pool_create_rayon(&mut p, 0, std::ptr::null()),
            TAPP_ERROR_NULL
        );
        assert_eq!(p, 0, "a rejected create wrote the out-parameter");

        // Zero (absent) pool handle, and null out-parameters.
        assert_eq!(
            tcontract_tapp_executor_borrow_pool(&mut e, 0, 1),
            TAPP_ERROR_NULL
        );
        assert_eq!(
            tcontract_tapp_executor_borrow_pool(std::ptr::null_mut(), 0, 1),
            TAPP_ERROR_NULL
        );
        assert_eq!(tcontract_pool_destroy(0), TAPP_ERROR_NULL);

        // A live pool, but a null out-parameter or a zero budget.
        let p = pool(2);
        assert_eq!(
            tcontract_tapp_executor_borrow_pool(std::ptr::null_mut(), p, 1),
            TAPP_ERROR_NULL
        );
        assert_eq!(
            tcontract_tapp_executor_borrow_pool(&mut e, p, 0),
            TAPP_ERROR_NULL
        );
        assert_eq!(e, 0, "a rejected borrow wrote the out-parameter");
        assert_eq!(tcontract_pool_destroy(p), TAPP_SUCCESS);
    }
}

#[test]
fn destroying_a_pool_with_a_live_executor_is_refused() {
    unsafe {
        let p = pool(2);
        let e = borrow(p, 2);
        // Reject, not accept-with-unsupported-semantics: the pool knows it has
        // a live borrower, so the handle stays live and the caller retries.
        assert_eq!(tcontract_pool_destroy(p), TAPP_ERROR_BUSY);
        with_executor(e, |x| {
            x.broadcast(2, &|_| {}).unwrap();
            Ok(())
        })
        .unwrap();
        assert_eq!(TAPP_destroy_executor(e), TAPP_SUCCESS);
        assert_eq!(tcontract_pool_destroy(p), TAPP_SUCCESS);

        // From one of the pool's own workers: a destroy would join the calling
        // thread, so it is refused rather than deadlocking.
        let p = pool(2);
        let e = borrow(p, 2);
        let code = AtomicI32::new(i32::MIN);
        with_executor(e, |x| {
            x.broadcast(2, &|_| {
                code.store(tcontract_pool_destroy(p), Ordering::Relaxed);
            })
            .unwrap();
            Ok(())
        })
        .unwrap();
        assert_eq!(code.load(Ordering::Relaxed), TAPP_ERROR_WOULD_DEADLOCK);
        assert_eq!(TAPP_destroy_executor(e), TAPP_SUCCESS);
        assert_eq!(tcontract_pool_destroy(p), TAPP_SUCCESS);
    }
}
