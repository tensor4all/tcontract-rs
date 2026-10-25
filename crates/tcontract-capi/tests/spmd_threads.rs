//! Nested and concurrent co-scheduled teams on one executor: no deadlock, no
//! extra threads.
//! Alone in its test binary so the process-wide thread count is meaningful.
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tcontract_capi::executor::*;
use tcontract_capi::status::*;

fn create_rayon(n: usize) -> TAPP_executor {
    let mut e = 0;
    let st = unsafe { tcontract_tapp_executor_create_rayon(&mut e, n, std::ptr::null()) };
    assert_eq!(st, TCONTRACT_OK);
    e
}

#[test]
fn nested_and_concurrent_spmd_neither_deadlock_nor_add_threads() {
    let e = create_rayon(4);
    let (tx, rx) = std::sync::mpsc::channel();
    let ptr = e as usize;
    std::thread::spawn(move || {
        let live = || std::fs::read_dir("/proc/self/task").map_or(0, Iterator::count);
        let ex: TAPP_executor = ptr as TAPP_executor;
        // A broadcast of the full width returns only once all four workers are
        // running, so the baseline below cannot miss a worker that is still
        // starting up.
        unsafe {
            with_executor(ex, |x| {
                x.token().broadcast(|_| {}).unwrap();
                Ok(())
            })
        }
        .unwrap();
        let before = live();
        // Concurrent broadcasts from several callers do not interleave.
        let hits = Arc::new(AtomicUsize::new(0));
        let callers: Vec<_> = (0..3)
            .map(|_| {
                let hits = hits.clone();
                std::thread::spawn(move || {
                    for _ in 0..20 {
                        unsafe {
                            with_executor(ex, |x| {
                                x.token()
                                    .broadcast(|_| {
                                        hits.fetch_add(1, Ordering::Relaxed);
                                    })
                                    .unwrap();
                                Ok(())
                            })
                        }
                        .unwrap();
                    }
                })
            })
            .collect();
        for c in callers {
            c.join().unwrap();
        }
        // Nested: a broadcast from a worker is refused, never queued.
        let nested = unsafe {
            with_executor(ex, |x| {
                Ok(x.token().external(|_| {
                    with_executor(ex, |y| Ok(y.token().broadcast(|_| {}).is_err())).unwrap()
                }))
            })
        }
        .unwrap();
        // A broadcast runs exactly the budget: a smaller budget is a smaller team.
        unsafe { tcontract_tapp_executor_set_budget(ex, 2) };
        let team = Arc::new(AtomicUsize::new(0));
        let t2 = team.clone();
        unsafe {
            with_executor(ex, |x| {
                x.token()
                    .broadcast(|_| _ = t2.fetch_add(1, Ordering::Relaxed))
                    .unwrap();
                Ok(())
            })
        }
        .unwrap();
        let over = team.load(Ordering::Relaxed) == 2;
        // The three callers are joined, so only the pool's workers remain. A
        // joined thread can stay in /proc/self/task for a moment after `join`
        // returns, so wait (bounded) for the count to settle; a thread that was
        // really added never goes away and still fails the assertion.
        let mut after = live();
        for _ in 0..500 {
            if after <= before {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
            after = live();
        }
        tx.send((hits.load(Ordering::Relaxed), nested, over, before, after))
            .unwrap();
    });
    let (hits, nested, over, before, after) = rx
        .recv_timeout(Duration::from_secs(60))
        .expect("nested/concurrent SPMD did not finish: deadlock");
    assert_eq!(hits, 3 * 20 * 4);
    assert!(nested, "a nested broadcast must be refused");
    assert!(over, "a smaller budget must be a smaller team");
    assert_eq!(before, after, "no thread was added");
    assert_eq!(unsafe { TAPP_destroy_executor(e) }, TCONTRACT_OK);
}
