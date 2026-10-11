//! strided-rs kernels driven by a `tcontract_exec::Exec` (feature `strided`).
#![cfg(feature = "strided")]

use strided_basic::{map_into, StridedArray};
use tcontract_exec::strided::run_with_exec;
use tcontract_exec::{Exec, Pool};

fn pool(n: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .unwrap()
}

#[test]
fn small_map_stays_on_caller_and_never_enters() {
    let tp = pool(4);
    let p = Pool::borrow(&tp);
    let exec = Exec::rayon(&p);
    let src = StridedArray::<f64>::from_fn_col_major(&[64, 8], |i| (i[0] + i[1]) as f64);
    let mut dst = StridedArray::<f64>::col_major(&[64, 8]);
    let caller = std::thread::current().id();
    run_with_exec(&exec, 64 * 8, |ctx| {
        assert!(ctx.is_serial());
        assert_eq!(std::thread::current().id(), caller);
        map_into(&mut dst.view_mut(), &src.view(), |x| 2.0 * x).unwrap()
    });
    assert_eq!(p.stats().entries, 0);
    assert_eq!(dst.view().get(&[3, 2]), 10.0);
}

#[test]
fn large_map_enters_once_and_matches_serial() {
    let tp = pool(4);
    let p = Pool::borrow(&tp);
    let exec = Exec::rayon(&p);
    let dims = [512, 256];
    let src = StridedArray::<f64>::from_fn_col_major(&dims, |i| (i[0] * 3 + i[1]) as f64);
    let src_view = src.view();
    let src_t = src_view.permute(&[1, 0]).unwrap();
    let mut par = StridedArray::<f64>::col_major(&[256, 512]);
    let mut ser = StridedArray::<f64>::col_major(&[256, 512]);
    run_with_exec(&exec, 512 * 256, |ctx| {
        assert_eq!(ctx.max_threads_limit().map(|n| n.get()), Some(4));
        map_into(&mut par.view_mut(), &src_t, |x| x + 1.0).unwrap()
    });
    run_with_exec(&Exec::serial(), 512 * 256, |ctx| {
        assert!(ctx.is_serial());
        map_into(&mut ser.view_mut(), &src_t, |x| x + 1.0).unwrap()
    });
    assert_eq!(par.into_data(), ser.into_data());
    assert_eq!(p.stats().entries, 1);
}

#[test]
fn budget_bounds_the_strided_context() {
    let tp = pool(4);
    let p = Pool::borrow(&tp);
    let exec = Exec::rayon(&p).with_budget(2).unwrap();
    let limit = run_with_exec(&exec, 1 << 20, |ctx| {
        ctx.max_threads_limit().map(|n| n.get())
    });
    assert_eq!(limit, Some(2));
}
