//! Concurrent and nested products on one executor finish in bounded time and
//! add no threads. Alone in its test binary: it counts the process's threads.
mod raw;

use std::sync::mpsc;
use std::time::Duration;

use raw::*;
use tcontract_capi::*;

fn threads() -> usize {
    std::fs::read_dir("/proc/self/task").map_or(0, Iterator::count)
}

#[test]
fn concurrent_nested_and_over_budget_products_neither_deadlock_nor_add_threads() {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || unsafe {
        let (m, n, k) = (200usize, 190usize, 180usize);
        let (ea, eb, ed) = (
            [m as i64, k as i64],
            [k as i64, n as i64],
            [m as i64, n as i64],
        );
        let (sa, sb, sd) = (cm(&ea), cm(&eb), cm(&ed));
        let (la, lb, ld) = ([I, K], [K, J], [I, J]);
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

        let exec = rayon_exec(4);
        // Warm: the pool's workers exist now.
        let baseline = threads();

        // Concurrent callers (more than the pool has workers) on one executor.
        let ok = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..6 {
                s.spawn(|| {
                    for _ in 0..4 {
                        let mut d = vec![f64::NAN; m * n];
                        let rc = exec_f64(
                            plan,
                            exec,
                            1.0,
                            a.as_ptr(),
                            b.as_ptr(),
                            0.0,
                            std::ptr::null(),
                            d.as_mut_ptr(),
                        );
                        assert_eq!(rc, TAPP_SUCCESS);
                        assert_close(&d, &want, 1e-12);
                        ok.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                });
            }
        });
        assert_eq!(ok.load(std::sync::atomic::Ordering::Relaxed), 24);

        // From a worker of the executor's own pool a packed team cannot be
        // co-scheduled; the call takes the barrier-free route and completes.
        let mut d = vec![17.0; m * n];
        let dp = d.as_mut_ptr() as usize;
        let (ap, bp) = (a.as_ptr() as usize, b.as_ptr() as usize);
        let rc = tcontract_capi::executor::with_executor(exec, |x| {
            Ok(x.token().external(move |_| {
                exec_f64(
                    plan,
                    exec,
                    1.0,
                    ap as *const f64,
                    bp as *const f64,
                    0.0,
                    std::ptr::null(),
                    dp as *mut f64,
                )
            }))
        })
        .unwrap();
        assert_eq!(rc, TAPP_SUCCESS);
        assert_close(&d, &want, 1e-12);

        // Budget below the pool width: the product still completes.
        assert_eq!(tcontract_tapp_executor_set_budget(exec, 2), TAPP_SUCCESS);
        let mut d = vec![f64::NAN; m * n];
        assert_eq!(
            exec_f64(
                plan,
                exec,
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

        let after = threads();
        assert_eq!(TAPP_destroy_executor(exec), TAPP_SUCCESS);
        TAPP_destroy_tensor_product(plan);
        tx.send((baseline, after)).unwrap();
    });
    let (baseline, after) = rx
        .recv_timeout(Duration::from_secs(120))
        .expect("concurrent/nested products did not finish: deadlock");
    assert_eq!(baseline, after, "products added threads");
}
