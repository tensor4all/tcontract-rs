//! One pool is one workspace owner: every operation on it shares the storage,
//! and no other pool can see it.
use strided_view::{StridedView, StridedViewMut};
use tcontract::{Plan, PlanConfig};
use tcontract_exec::{Exec, Pool};
use tcontract_kernel::KernelChoice;

mod common;
use common::plans::matmul_problem;

const M: usize = 192;
const N: usize = 160;
const K: usize = 128;

/// One `ij,jk->ik` contraction on `width` workers of the pool.
fn run(exec: &mut Exec<'_>, width: usize) {
    let mut exec = exec.reborrow().limit(width.try_into().unwrap());
    let av: Vec<f64> = (0..M * K).map(|x| (x % 11) as f64 - 5.0).collect();
    let bv: Vec<f64> = (0..K * N).map(|x| (x % 7) as f64 * 0.5).collect();
    let config = PlanConfig {
        kernel: KernelChoice::Id("ref.f64.real-scalar.4x4".into()),
        ..PlanConfig::default()
    };
    let plan = Plan::<f64>::new(&matmul_problem(M, N, K), &config).unwrap();
    let mut out = vec![0.0; M * N];
    plan.execute_into(
        &mut exec,
        1.0,
        &StridedView::new(&av, &[M, K], &[1, M as isize], 0).unwrap(),
        &StridedView::new(&bv, &[K, N], &[1, K as isize], 0).unwrap(),
        &mut StridedViewMut::new(&mut out, &[M, N], &[1, M as isize], 0).unwrap(),
    )
    .unwrap();
}

#[test]
fn one_pool_lends_one_workspace_and_two_pools_never_share() {
    let tp = threadteam::Pool::builder().threads(4).build().unwrap();
    let pool_a = Pool::borrow(&tp);
    let pool_b = Pool::borrow(&tp);
    let mut exec = Exec::new(&pool_a, pool_a.size()).unwrap();

    // The context answers with a workspace, and the only one it can answer with
    // is this pool's: the run below is what proves it did.
    assert!(exec.workspace().is_some(), "a pool lends a workspace");
    // Both pools start empty and stay empty until something runs on them, which
    // is what makes the run below a statement about *this* pool's arena.
    assert_eq!(pool_a.workspace().retained_bytes(), 0);
    assert_eq!(pool_b.workspace().retained_bytes(), 0);
    // A serial context owns nothing; its caller lends a provider explicitly.
    assert!(Exec::serial().workspace().is_none());
    assert!(Exec::serial_with_workspace(pool_a.workspace())
        .workspace()
        .is_some());

    // Two operations on one pool reuse its storage: it grows on the first and
    // stops growing after that.
    run(&mut exec, 3);
    let panel = pool_a.workspace().retained_panel_bytes();
    assert!(panel > 0, "the pool's workspace was never used");
    assert_eq!(
        pool_b.workspace().retained_bytes(),
        0,
        "an operation on one pool touched another pool's workspace"
    );
    // The panel a shape needs does not depend on which workers took part, so a
    // second run of the same shape must reuse the very same allocation.
    run(&mut exec, 3);
    assert_eq!(
        pool_a.workspace().retained_panel_bytes(),
        panel,
        "the second operation grew the pool's panel again"
    );
    // Trimming releases idle storage and leaves the pool usable.
    pool_a.trim_workspace();
    assert!(pool_a.workspace().retained_panel_bytes() <= panel);
    run(&mut exec, 3);
    assert!(pool_a.workspace().retained_panel_bytes() > 0);
}
