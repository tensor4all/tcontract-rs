use tcontract_exec::{Exec, Pool};

#[test]
fn owned_pool_runs_and_hands_its_pool_back() {
    let tp = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    let pool: Pool<'static> = Pool::owned(tp);
    assert_eq!(pool.size(), 3);
    let exec = Exec::rayon(&pool);
    assert_eq!(exec.install(3, |p| p.threads()), 3);
    exec.broadcast(3, &|_| {}).unwrap();
    assert_eq!(pool.stats().broadcasts, 1);
    let back = pool.into_owned().expect("owned pool is returned");
    assert_eq!(back.current_num_threads(), 3);
    let tp2 = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    assert!(Pool::borrow(&tp2).into_owned().is_none());
}
