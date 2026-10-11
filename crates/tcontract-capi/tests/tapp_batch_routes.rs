//! Batch resource admission precedes every output write.
mod raw;

use raw::*;
use tcontract_capi::*;

#[test]
fn mixed_storage_batch_checks_nested_fresh_route_before_first_write() {
    // SAFETY: every descriptor, scalar and pointer covers its declared reach;
    // outputs are disjoint and remain owned until synchronous execution ends.
    unsafe {
        let (m, n, k) = (200usize, 190usize, 180usize);
        let (ea, eb, ed) = (
            [m as i64, k as i64],
            [k as i64, n as i64],
            [m as i64, n as i64],
        );
        let (sa, sb, sd) = (cm(&ea), cm(&eb), cm(&ed));
        let (rc, plan) = plan_f64(
            (&ea, &sa, &[I, K]),
            (&eb, &sb, &[K, J]),
            (&ed, &sd, &[I, J]),
            (&ed, &sd, &[I, J]),
        );
        assert_eq!(rc, TAPP_SUCCESS);
        let exec = rayon_exec(4);
        let a = vec![1.0_f64; m * k];
        let b = vec![1.0_f64; k * n];
        let c = vec![23.0_f64; m * n];
        let mut d0 = vec![17.0_f64; m * n];
        let mut d1 = vec![19.0_f64; m * n];
        let (ap, bp, cp, p0, p1) = (
            a.as_ptr() as usize,
            b.as_ptr() as usize,
            c.as_ptr() as usize,
            d0.as_mut_ptr() as usize,
            d1.as_mut_ptr() as usize,
        );
        let rc = tcontract_capi::executor::with_executor(exec, |x| {
            Ok(x.install(2, move |_| {
                let (alpha, beta) = (1.0_f64, 1.0_f64);
                let aa = [ap as *const std::ffi::c_void; 2];
                let bb = [bp as *const std::ffi::c_void; 2];
                let cc = [p0 as *const std::ffi::c_void, cp as *const std::ffi::c_void];
                let mut dd = [p0 as *mut std::ffi::c_void, p1 as *mut std::ffi::c_void];
                TAPP_execute_batched_product(
                    plan,
                    exec,
                    std::ptr::null_mut(),
                    2,
                    (&alpha as *const f64).cast(),
                    aa.as_ptr(),
                    bb.as_ptr(),
                    (&beta as *const f64).cast(),
                    cc.as_ptr(),
                    dd.as_mut_ptr(),
                )
            }))
        })
        .unwrap();
        eprintln!("nested mixed-storage batch status={rc}");
        if rc == TAPP_ERROR_UNSUPPORTED {
            assert!(
                d0.iter().all(|&v| v == 17.0),
                "initialized item wrote before fresh admission"
            );
            assert!(d1.iter().all(|&v| v == 19.0));
        } else {
            // A backend's barrier-free route is legal on its own pool worker.
            assert_eq!(rc, TAPP_SUCCESS);
            assert!(d0.iter().all(|&v| v == k as f64 + 17.0));
            assert!(d1.iter().all(|&v| v == k as f64 + 23.0));
        }
        assert_eq!(TAPP_destroy_executor(exec), TAPP_SUCCESS);
        assert_eq!(TAPP_destroy_tensor_product(plan), TAPP_SUCCESS);
    }
}
