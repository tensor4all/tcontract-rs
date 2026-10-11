use std::sync::atomic::{AtomicUsize, Ordering};
use tcontract_exec::{Exec, Par, Pool};

fn pool(n: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .unwrap()
}

#[test]
fn shared_pool_preserves_selected_workers_workspace_and_arc_ownership() {
    use std::collections::HashSet;
    use std::sync::{Arc, Mutex};
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Pool<'static>>();
    let raw = Arc::new(pool(2));
    let weak = Arc::downgrade(&raw);
    let expected: HashSet<_> = raw
        .broadcast(|_| std::thread::current().id())
        .into_iter()
        .collect();
    let wrapper = Arc::new(Pool::shared(Arc::clone(&raw)));
    let alias = Arc::clone(&wrapper);
    assert!(std::ptr::eq(wrapper.workspace(), alias.workspace()));
    let exec = Exec::rayon(&wrapper);
    let caller = std::thread::current().id();
    assert_eq!(
        exec.install(1, |par| (par, std::thread::current().id())),
        (Par::Seq, caller)
    );
    assert_eq!(wrapper.stats().entries, 0);
    for _ in 0..3 {
        exec.install(2, |par| {
            assert_eq!(par.threads(), 2);
            assert!(raw.current_thread_index().is_some());
        });
    }
    let observed = Mutex::new(HashSet::new());
    exec.broadcast(2, &|_| {
        observed.lock().unwrap().insert(std::thread::current().id());
    })
    .unwrap();
    assert_eq!(observed.into_inner().unwrap(), expected);
    // Keep the pre-existing worker-context refusal, including width one.
    assert_eq!(
        raw.install(|| exec.broadcast(1, &|_| ())),
        Err(tcontract_exec::ExecError::Unavailable)
    );
    drop(raw);
    assert!(weak.upgrade().is_some());
    drop(alias);
    let wrapper = Arc::try_unwrap(wrapper).unwrap();
    assert!(wrapper.into_owned().is_none());
    // Ownership retention, not a claim about worker shutdown completion.
    assert!(weak.upgrade().is_none());
}

#[test]
fn serial_install_runs_inline_with_seq() {
    let caller = std::thread::current().id();
    let got = Exec::serial().install(4, |par| (par, std::thread::current().id()));
    assert_eq!(got, (Par::Seq, caller));
}

#[test]
fn width_one_never_enters_the_pool() {
    let tp = pool(4);
    let p = Pool::borrow(&tp);
    let exec = Exec::rayon(&p);
    let caller = std::thread::current().id();
    let tid = exec.install(1, |par| {
        assert_eq!(par, Par::Seq);
        std::thread::current().id()
    });
    assert_eq!(tid, caller);
    exec.for_each_partition(1, &|i| assert_eq!(i, 0));
    assert_eq!(p.stats().entries, 0);
}

#[test]
fn parallel_install_enters_once_and_runs_on_a_worker() {
    let tp = pool(4);
    let p = Pool::borrow(&tp);
    let exec = Exec::rayon(&p);
    let (par, idx) = exec.install(3, |par| (par, tp.current_thread_index()));
    assert_eq!(par.threads(), 3);
    assert!(idx.is_some());
    assert_eq!(p.stats().entries, 1);
}

#[test]
fn nested_install_on_a_worker_does_not_reenter() {
    let tp = pool(4);
    let p = Pool::borrow(&tp);
    let exec = Exec::rayon(&p);
    exec.install(2, |_| {
        assert!(exec.is_worker());
        exec.install(2, |par| assert_eq!(par.threads(), 2));
    });
    let s = p.stats();
    assert_eq!((s.entries, s.inline_runs), (1, 1));
}

#[test]
fn budget_caps_width_and_zero_budget_is_an_error() {
    let tp = pool(4);
    let p = Pool::borrow(&tp);
    let exec = Exec::rayon(&p).with_budget(2).unwrap();
    assert_eq!(exec.budget(), 2);
    assert_eq!(exec.install(8, |par| par.threads()), 2);
    assert!(Exec::rayon(&p).with_budget(0).is_err());
    assert_eq!(Exec::rayon(&p).with_budget(99).unwrap().budget(), 4);
}

#[test]
fn partition_runs_every_index_once_and_repartitions_above_budget() {
    let tp = pool(4);
    let p = Pool::borrow(&tp);
    let exec = Exec::rayon(&p).with_budget(2).unwrap();
    let hits: Vec<AtomicUsize> = (0..7).map(|_| AtomicUsize::new(0)).collect();
    exec.for_each_partition(7, &|i| {
        hits[i].fetch_add(1, Ordering::Relaxed);
    });
    assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
    assert_eq!(p.stats().entries, 1);
}

#[test]
fn faer_matmul_runs_on_the_borrowed_pool() {
    use faer::{linalg::matmul::matmul, Accum, Mat};
    let tp = pool(4);
    let p = Pool::borrow(&tp);
    let exec = Exec::rayon(&p);
    let a = Mat::<f64>::from_fn(256, 256, |i, j| (i + 2 * j) as f64 * 1e-3);
    let b = Mat::<f64>::from_fn(256, 256, |i, j| (i * j % 7) as f64);
    let mut c = Mat::<f64>::zeros(256, 256);
    let mut r = Mat::<f64>::zeros(256, 256);
    matmul(
        r.as_mut(),
        Accum::Replace,
        a.as_ref(),
        b.as_ref(),
        1.0,
        faer::Par::Seq,
    );
    exec.install(4, |par| {
        let fp = match par {
            Par::Seq => faer::Par::Seq,
            Par::Threads(n) => faer::Par::rayon(n.get()),
        };
        matmul(c.as_mut(), Accum::Replace, a.as_ref(), b.as_ref(), 1.0, fp);
    });
    assert!((&c - &r).norm_max() <= 1e-9 * r.norm_max());
    assert_eq!(p.stats().entries, 1);
}
