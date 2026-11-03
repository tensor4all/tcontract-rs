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
    let pool = threadteam::Pool::builder()
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
    // The token borrows `a`'s pool; end the borrow before `a` is destroyed.
    drop(exec);
    Executor::destroy(Box::into_raw(a)).unwrap();
    Executor::destroy(Box::into_raw(b)).unwrap();
}
