//! The Rust seam a sibling C-API crate uses to run on the host's pool: it
//! receives an `Exec` for the executor's pool and budget, never the pool.

use std::ffi::c_void;

use tcontract_capi::{
    executor_from_handle, tcontract_pool, tcontract_pool_create_rayon, tcontract_pool_destroy,
    tcontract_tapp_executor_borrow_pool, TAPP_check_success, TAPP_create_executor,
    TAPP_destroy_executor, TAPP_ERROR_BUSY,
};

#[test]
fn a_borrowed_executor_lends_its_pool_and_budget() {
    let mut pool: tcontract_pool = 0;
    // SAFETY: a writable out-parameter and a null options pointer.
    let status = unsafe { tcontract_pool_create_rayon(&mut pool, 2, std::ptr::null()) };
    assert!(TAPP_check_success(status), "create the pool");

    let mut exec: isize = 0;
    // SAFETY: a live pool handle and a writable out-parameter.
    let status = unsafe { tcontract_tapp_executor_borrow_pool(&mut exec, pool, 2) };
    assert!(TAPP_check_success(status), "borrow the pool");

    // SAFETY: a live handle.
    let executor = unsafe { executor_from_handle(exec) }.expect("a live executor");
    let observed = executor
        .with(|exec| Ok((executor.threads().0, exec.budget())))
        .unwrap();
    assert_eq!(observed, (2, 2));

    // While the call is in flight the executor cannot be destroyed: the seam
    // holds the same guard the C entries hold.
    let busy = executor
        .with(|_| {
            // SAFETY: a live handle, destroyed by nobody else.
            Ok(unsafe { TAPP_destroy_executor(exec) })
        })
        .unwrap();
    assert_eq!(busy, TAPP_ERROR_BUSY);
    // And the pool cannot be released from under the borrowing executor.
    // SAFETY: a live pool handle.
    assert_eq!(unsafe { tcontract_pool_destroy(pool) }, TAPP_ERROR_BUSY);

    // SAFETY: a live handle no call uses afterwards.
    assert!(TAPP_check_success(unsafe { TAPP_destroy_executor(exec) }));
    // SAFETY: the pool is no longer borrowed.
    assert!(TAPP_check_success(unsafe { tcontract_pool_destroy(pool) }));
}

#[test]
fn a_serial_executor_has_no_pool_to_lend() {
    let mut serial: isize = 0;
    // SAFETY: a writable out-parameter.
    assert!(TAPP_check_success(unsafe {
        TAPP_create_executor(&mut serial)
    }));

    // SAFETY: a live handle.
    let executor = unsafe { executor_from_handle(serial) }.expect("a live executor");
    assert_eq!(executor.threads(), (0, 1));
    assert!(executor.with(|exec| Ok(exec.token().is_serial())).unwrap());

    // SAFETY: a live handle no call uses afterwards.
    assert!(TAPP_check_success(unsafe { TAPP_destroy_executor(serial) }));

    // A zero handle names no executor at all, which the ABI treats as serial.
    // SAFETY: zero is explicitly allowed.
    assert!(unsafe { executor_from_handle(0) }.is_none());
    let _ = std::marker::PhantomData::<*const c_void>;
}
