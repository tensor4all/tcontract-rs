//! The executor lifecycle through the TAPP/tcontract C entry points.
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tcontract_capi::executor::*;
use tcontract_capi::status::*;

fn create_rayon(n: usize) -> TAPP_executor {
    let mut e = 0;
    let st = unsafe { tcontract_tapp_executor_create_rayon(&mut e, n, std::ptr::null()) };
    assert_eq!(st, TCONTRACT_OK);
    assert_ne!(e, 0);
    e
}

fn threads(e: TAPP_executor) -> (usize, usize) {
    let (mut p, mut b) = (99, 99);
    assert_eq!(
        unsafe { tcontract_tapp_executor_get_threads(e, &mut p, &mut b) },
        TCONTRACT_OK
    );
    (p, b)
}

#[test]
fn serial_executor_has_no_pool() {
    let mut e = 0;
    assert_eq!(unsafe { TAPP_create_executor(&mut e) }, TCONTRACT_OK);
    assert_ne!(e, 0);
    assert_eq!(threads(e), (0, 1));
    assert!(unsafe { executor_ref(e) }.unwrap().pool_stats().is_none());
    // A serial executor accepts a budget and stays at one.
    assert_eq!(
        unsafe { tcontract_tapp_executor_set_budget(e, 8) },
        TCONTRACT_OK
    );
    assert_eq!(threads(e), (0, 1));
    assert_eq!(unsafe { TAPP_destroy_executor(e) }, TCONTRACT_OK);
}

#[test]
fn zero_is_the_default_serial_executor() {
    assert_eq!(threads(0), (0, 1));
    assert_eq!(unsafe { TAPP_destroy_executor(0) }, TCONTRACT_OK);
    assert_eq!(
        unsafe { TAPP_destroy_executor(0) },
        TCONTRACT_OK,
        "a no-op, repeatedly"
    );
    let width = unsafe { with_executor(0, |x| Ok(x.install(4, |p| p.threads()))) }.unwrap();
    assert_eq!(width, 1);
}

#[test]
fn one_thread_is_serial_and_zero_threads_is_an_error() {
    let e = create_rayon(1);
    assert_eq!(threads(e), (0, 1), "nthreads == 1 creates no workers");
    assert!(unsafe { executor_ref(e) }.unwrap().pool_stats().is_none());
    assert_eq!(unsafe { TAPP_destroy_executor(e) }, TCONTRACT_OK);

    let mut out = 7;
    assert_eq!(
        unsafe { tcontract_tapp_executor_create_rayon(&mut out, 0, std::ptr::null()) },
        TCONTRACT_ERR_INVALID_ARGUMENT
    );
    assert_eq!(out, 7, "failed creation leaves the out-parameter alone");
    assert_eq!(
        unsafe { tcontract_tapp_executor_create_rayon(std::ptr::null_mut(), 2, std::ptr::null()) },
        TCONTRACT_ERR_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe { TAPP_create_executor(std::ptr::null_mut()) },
        TCONTRACT_ERR_INVALID_ARGUMENT
    );
}

#[test]
fn width_ignores_the_environment() {
    std::env::set_var("RAYON_NUM_THREADS", "7");
    std::env::set_var("TENSORCONTRACT_THREADS", "5");
    let e = create_rayon(3);
    assert_eq!(threads(e), (3, 3));
    assert_eq!(unsafe { TAPP_destroy_executor(e) }, TCONTRACT_OK);
    std::env::remove_var("RAYON_NUM_THREADS");
    std::env::remove_var("TENSORCONTRACT_THREADS");
}

#[test]
fn budget_is_positive_clamped_and_snapshotted_per_call() {
    let e = create_rayon(3);
    assert_eq!(
        unsafe { tcontract_tapp_executor_set_budget(e, 2) },
        TCONTRACT_OK
    );
    assert_eq!(threads(e), (3, 2));
    assert_eq!(
        unsafe { tcontract_tapp_executor_set_budget(e, 99) },
        TCONTRACT_OK
    );
    assert_eq!(threads(e), (3, 3), "clamped to the pool width");
    assert_eq!(
        unsafe { tcontract_tapp_executor_set_budget(e, 0) },
        TCONTRACT_ERR_INVALID_ARGUMENT
    );
    assert_eq!(threads(e), (3, 3));
    // Null out-parameters are allowed.
    assert_eq!(
        unsafe {
            tcontract_tapp_executor_get_threads(e, std::ptr::null_mut(), std::ptr::null_mut())
        },
        TCONTRACT_OK
    );

    // A budget change during a call does not affect that call.
    unsafe { tcontract_tapp_executor_set_budget(e, 2) };
    let seen = unsafe {
        with_executor(e, |x| {
            let before = x.budget();
            tcontract_tapp_executor_set_budget(e, 3);
            Ok((before, x.budget()))
        })
    }
    .unwrap();
    assert_eq!(seen, (2, 2));
    assert_eq!(threads(e), (3, 3));
    assert_eq!(unsafe { TAPP_destroy_executor(e) }, TCONTRACT_OK);
}

#[test]
fn destroy_joins_workers_and_waits_for_tls_teardown() {
    let e = create_rayon(3);
    static DROPPED: AtomicUsize = AtomicUsize::new(0);
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            std::thread::sleep(Duration::from_millis(20));
            DROPPED.fetch_add(1, Ordering::SeqCst);
        }
    }
    thread_local! { static G: std::cell::RefCell<Option<Guard>> = const { std::cell::RefCell::new(None) }; }
    unsafe {
        with_executor(e, |x| {
            x.broadcast(2, &|_| G.with(|g| *g.borrow_mut() = Some(Guard)))
                .unwrap();
            Ok(())
        })
    }
    .unwrap();
    assert_eq!(unsafe { TAPP_destroy_executor(e) }, TCONTRACT_OK);
    assert_eq!(
        DROPPED.load(Ordering::SeqCst),
        2,
        "destroy returned before worker TLS teardown"
    );
}

#[test]
fn busy_and_self_worker_destroy_leave_the_handle_live() {
    let e = create_rayon(2);
    let inside = Arc::new(AtomicBool::new(false));
    let release = Arc::new(AtomicBool::new(false));
    let (i2, r2) = (inside.clone(), release.clone());
    let ptr = e as usize;
    let t = std::thread::spawn(move || {
        unsafe {
            with_executor(ptr as TAPP_executor, |_| {
                i2.store(true, Ordering::SeqCst);
                while !r2.load(Ordering::SeqCst) {
                    std::thread::yield_now();
                }
                Ok(())
            })
        }
        .unwrap();
    });
    while !inside.load(Ordering::SeqCst) {
        std::thread::yield_now();
    }
    assert_eq!(unsafe { TAPP_destroy_executor(e) }, TCONTRACT_BUSY);
    assert!(!unsafe { std::ffi::CStr::from_ptr(tcontract_last_error()) }
        .to_bytes()
        .is_empty());
    release.store(true, Ordering::SeqCst);
    t.join().unwrap();

    // From a worker of the pool itself (the call is also in flight).
    let st = unsafe {
        with_executor(e, |x| {
            Ok(x.install(2, move |_| TAPP_destroy_executor(ptr as TAPP_executor)))
        })
    }
    .unwrap();
    assert_eq!(st, TCONTRACT_ERR_WOULD_DEADLOCK);
    // Both refusals left the handle usable, and it can now be destroyed once.
    assert_eq!(threads(e), (2, 2));
    assert_eq!(unsafe { TAPP_destroy_executor(e) }, TCONTRACT_OK);
}

#[test]
fn last_error_and_panics() {
    let st = ffi(|| Err(FfiError::new(TCONTRACT_ERR_SHAPE, "bad shape")));
    assert_eq!(st, TCONTRACT_ERR_SHAPE);
    let msg = unsafe { std::ffi::CStr::from_ptr(tcontract_last_error()) }
        .to_str()
        .unwrap()
        .to_string();
    assert_eq!(msg, "bad shape");
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let st = ffi(|| panic!("boom"));
    std::panic::set_hook(prev);
    assert_eq!(st, TCONTRACT_ERR_PANIC);
    let msg = unsafe { std::ffi::CStr::from_ptr(tcontract_last_error()) }
        .to_str()
        .unwrap()
        .to_string();
    assert!(msg.contains("boom"));
    assert_eq!(ffi(|| Ok(())), TCONTRACT_OK);
}

#[test]
fn tapp_error_helpers() {
    assert!(TAPP_check_success(0));
    assert!(!TAPP_check_success(TCONTRACT_BUSY));
    let n = unsafe { TAPP_explain_error(TCONTRACT_BUSY, 0, std::ptr::null_mut()) };
    assert!(n > 0);
    let mut buf = vec![0 as std::ffi::c_char; n + 1];
    assert_eq!(
        unsafe { TAPP_explain_error(TCONTRACT_BUSY, buf.len(), buf.as_mut_ptr()) },
        n
    );
    assert_eq!(
        unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
            .to_bytes()
            .len(),
        n
    );
    // Truncation keeps the NUL.
    let mut small = [0x7f as std::ffi::c_char; 4];
    assert_eq!(
        unsafe { TAPP_explain_error(TCONTRACT_BUSY, 4, small.as_mut_ptr()) },
        n
    );
    assert_eq!(small[3], 0);
}
