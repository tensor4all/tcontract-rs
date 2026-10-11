use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Barrier;
use std::time::Duration;
use tcontract_exec::{Exec, ExecError, Pool, WidthPolicy};

// The OS-thread-count test needs no other test spawning threads meanwhile.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn pool(n: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .unwrap()
}

fn with_timeout<F: FnOnce() + Send + 'static>(secs: u64, f: F) {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        f();
        let _ = tx.send(());
    });
    rx.recv_timeout(Duration::from_secs(secs))
        .expect("deadlock: timed out");
}

#[test]
fn broadcast_runs_width_participants_concurrently() {
    let _serial = lock();
    let tp = pool(4);
    let p = Pool::borrow(&tp);
    let exec = Exec::rayon(&p);
    let bar = Barrier::new(3);
    let n = AtomicUsize::new(0);
    exec.broadcast(3, &|t| {
        assert!(t < 3);
        bar.wait();
        n.fetch_add(1, Ordering::Relaxed);
    })
    .unwrap();
    assert_eq!(n.load(Ordering::Relaxed), 3);
    assert_eq!(p.stats().broadcasts, 1);
}

#[test]
fn broadcast_rejects_width_above_pool_and_serial_width_two() {
    let _serial = lock();
    let tp = pool(2);
    let p = Pool::borrow(&tp);
    assert_eq!(
        Exec::rayon(&p).broadcast(3, &|_| {}),
        Err(ExecError::WidthExceedsPool { width: 3, pool: 2 })
    );
    assert_eq!(
        Exec::serial().broadcast(2, &|_| {}),
        Err(ExecError::Unavailable)
    );
    let ran = AtomicUsize::new(0);
    Exec::serial()
        .broadcast(1, &|_| {
            ran.fetch_add(1, Ordering::Relaxed);
        })
        .unwrap();
    assert_eq!(ran.load(Ordering::Relaxed), 1);
}

#[test]
fn nested_broadcast_from_a_worker_is_unavailable_and_runs_nothing() {
    let _serial = lock();
    with_timeout(20, || {
        let tp = pool(4);
        let p = Pool::borrow(&tp);
        let exec = Exec::rayon(&p);
        let inner = AtomicUsize::new(0);
        exec.broadcast(4, &|_| {
            let r = exec.broadcast(2, &|_| {
                inner.fetch_add(1, Ordering::Relaxed);
            });
            assert_eq!(r, Err(ExecError::Unavailable));
        })
        .unwrap();
        assert_eq!(inner.load(Ordering::Relaxed), 0);
    });
}

#[test]
fn concurrent_broadcasts_from_two_host_threads_do_not_deadlock() {
    let _serial = lock();
    with_timeout(60, || {
        let tp = pool(4);
        let p = Pool::borrow(&tp);
        std::thread::scope(|s| {
            for _ in 0..2 {
                s.spawn(|| {
                    let exec = Exec::rayon(&p);
                    for _ in 0..50 {
                        let bar = Barrier::new(4);
                        exec.broadcast(4, &|_| {
                            bar.wait();
                        })
                        .unwrap();
                    }
                });
            }
        });
        assert_eq!(p.stats().broadcasts, 100);
    });
}

#[test]
fn panic_in_broadcast_propagates_and_pool_stays_usable() {
    let _serial = lock();
    let tp = pool(2);
    let p = Pool::borrow(&tp);
    let exec = Exec::rayon(&p);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = exec.broadcast(2, &|t| {
            if t == 1 {
                panic!("boom")
            }
        });
    }));
    assert!(r.is_err());
    exec.broadcast(2, &|_| {}).unwrap();
}

#[test]
fn width_policy_is_serial_for_small_work_and_capped_by_budget() {
    let _serial = lock();
    let tp = pool(4);
    let p = Pool::borrow(&tp);
    let exec = Exec::rayon(&p);
    let pol = WidthPolicy::default();
    assert_eq!(exec.width_for(1_000.0, &pol), 1);
    assert_eq!(exec.width_for(1e9, &pol), 4);
    assert_eq!(exec.with_budget(2).unwrap().width_for(1e9, &pol), 2);
    assert_eq!(Exec::serial().width_for(1e9, &pol), 1);
    assert_eq!(exec.width_for(f64::NAN, &pol), 1);
}

#[test]
fn os_thread_count_does_not_grow_after_partitions() {
    let _serial = lock();
    fn os_threads() -> usize {
        std::fs::read_dir("/proc/self/task")
            .map(|d| d.count())
            .unwrap_or(0)
    }
    let tp = pool(4);
    let p = Pool::borrow(&tp);
    let exec = Exec::rayon(&p).with_budget(2).unwrap();
    exec.for_each_partition(64, &|_| {});
    exec.broadcast(2, &|_| {}).unwrap();
    let before = os_threads();
    for _ in 0..20 {
        exec.for_each_partition(64, &|_| {});
        exec.broadcast(2, &|_| {}).unwrap();
    }
    // Threads of a pool dropped by an earlier test may still be exiting, so
    // the count can shrink meanwhile; it must never grow.
    if before > 0 {
        let after = os_threads();
        assert!(after <= before, "OS threads grew from {before} to {after}");
    }
}

#[test]
fn broadcast_width_is_capped_by_the_budget() {
    let _serial = lock();
    let tp = pool(4);
    let p = Pool::borrow(&tp);
    let exec = Exec::rayon(&p).with_budget(2).unwrap();
    assert_eq!(
        exec.broadcast(3, &|_| {}),
        Err(ExecError::WidthExceedsBudget {
            width: 3,
            budget: 2
        })
    );
    assert_eq!(p.stats().broadcasts, 0);
    exec.broadcast(2, &|_| {}).unwrap();
}
