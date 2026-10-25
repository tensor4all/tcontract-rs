//! The packed SPMD driver on a pool through the token's `broadcast`, and the
//! barrier-free route it falls back on when the token cannot co-schedule.

use std::time::Duration;

use strided_view::{StridedView, StridedViewMut};
use tcontract::api::Error;
use tcontract::{ExecutionRoute, OutputContract, PackedRoute, Plan, PlanConfig};
use tcontract_exec::{BatchPolicy, Exec, Pool};
use tcontract_kernel::KernelChoice;

mod common;
use common::placement::{installs, max_live, on_worker, pool, root};
use common::plans::{matmul_problem, packed_plan};

const M: usize = 256;
const N: usize = 240;
const K: usize = 200;

/// One contraction of `plan` into `out`.
fn fill(plan: &Plan<f64>, exec: &mut Exec<'_>, out: &mut [f64]) -> Result<(), Error> {
    let av: Vec<f64> = (0..M * K).map(|x| (x % 13) as f64 - 6.0).collect();
    let bv: Vec<f64> = (0..K * N).map(|x| (x % 7) as f64 * 0.5).collect();
    plan.execute_into(
        exec,
        1.0,
        &StridedView::new(&av, &[M, K], &[1, M as isize], 0).unwrap(),
        &StridedView::new(&bv, &[K, N], &[1, K as isize], 0).unwrap(),
        &mut StridedViewMut::new(out, &[M, N], &[1, M as isize], 0).unwrap(),
    )
}

fn gemm(plan: &Plan<f64>, exec: &mut Exec<'_>) -> Vec<f64> {
    let mut out = vec![0.0; M * N];
    fill(plan, exec, &mut out).unwrap();
    out
}

/// A deadlock in a fallback must fail the test, not hang it.
fn within_a_minute(what: &'static str, f: impl FnOnce() + Send + 'static) {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        f();
        let _ = tx.send(());
    });
    rx.recv_timeout(Duration::from_secs(60))
        .unwrap_or_else(|_| panic!("{what} deadlocked"));
}

#[test]
fn contraction_runs_spmd_on_the_pool() {
    let plan = packed_plan(M, N, K);
    let sp = pool(4);
    let pool = Pool::borrow(&sp);
    let mut exec = Exec::new(&pool, 4).unwrap();
    let serial = gemm(&plan, &mut Exec::serial());
    let route = plan
        .execution_route(&exec, 1.0, OutputContract::Initialized)
        .unwrap();
    let ExecutionRoute::Packed { kind, width } = route else {
        panic!("a packed plan takes the packed route: {route:?}")
    };
    assert_eq!(kind, PackedRoute::Spmd);
    let root = root(&mut exec);
    let par = gemm(&plan, &mut exec);
    assert_eq!(par, serial);
    // The whole team ran at once (its barriers need it) and entered the pool once.
    let live = max_live(root);
    assert!(
        (2..=width).contains(&live),
        "the team did not run at once: {live} live, width {width}"
    );
    assert_eq!(installs(root), 1);
}

/// A caller that is a worker of the pool cannot be co-scheduled; the call
/// takes the barrier-free cells route in place and the numbers are the team's.
#[test]
fn a_worker_caller_falls_back_to_barrier_free_cells() {
    within_a_minute("nested SPMD", || {
        let plan = packed_plan(M, N, K);
        let sp = pool(4);
        let pool = Pool::borrow(&sp);
        let serial = gemm(&plan, &mut Exec::serial());
        let (out, root) = on_worker(&pool, || {
            let mut inner = Exec::new(&pool, 4).unwrap();
            let root = root(&mut inner);
            let mut out = vec![-7.0; M * N];
            fill(&plan, &mut inner, &mut out).expect("the refused team falls back");
            (out, root)
        });
        assert_eq!(out, serial);
        assert_eq!(installs(root), 0, "the cells ran in place on the worker");
        assert!(max_live(root) <= 4);
    });
}

/// A token produced by `fork` is refused too, whatever its share: two graph
/// nodes running at once each fall back and stay within the parent's budget.
#[test]
fn a_forked_child_falls_back_and_stays_within_the_budget() {
    let plan = packed_plan(M, N, K);
    let sp = pool(4);
    let pool = Pool::borrow(&sp);
    let serial = gemm(&plan, &mut Exec::serial());
    let exec = Exec::new(&pool, 4).unwrap();
    let workspace = exec.workspace();
    let mut token = exec.into_token();
    let root = token.ledger_root().unwrap();
    let outs = std::sync::Mutex::new(Vec::new());
    token.reborrow().fork(2, |_, child| {
        assert_eq!(child.budget().get(), 2);
        let mut e = Exec::from_token(child, workspace, BatchPolicy::default());
        let mut out = vec![0.0; M * N];
        fill(&plan, &mut e, &mut out).expect("a forked child falls back");
        outs.lock().unwrap().push(out);
    });
    let outs = outs.into_inner().unwrap();
    assert_eq!(outs.len(), 2);
    assert!(outs.iter().all(|o| *o == serial));
    assert!(max_live(root) <= 4, "{}", max_live(root));
}

/// `pm == 1` takes no barrier -- every column group has one thread and reads
/// only its own slice of the panel -- so a direct-B plan runs the same cells
/// from a worker without any fallback.
#[test]
fn a_direct_b_plan_runs_barrier_free_cells_from_a_worker() {
    within_a_minute("barrier-free cells", || {
        let config = PlanConfig {
            kernel: KernelChoice::Id("ref.f64.direct-b.4x4".into()),
            ..PlanConfig::default()
        };
        let plan = Plan::<f64>::new(&matmul_problem(M, N, K), &config).unwrap();
        let sp = pool(4);
        let pool = Pool::borrow(&sp);
        let serial = gemm(&plan, &mut Exec::serial());
        let (out, root) = on_worker(&pool, || {
            let mut inner = Exec::new(&pool, 4).unwrap();
            let root = root(&mut inner);
            let mut out = vec![0.0; M * N];
            fill(&plan, &mut inner, &mut out).expect("a direct-B plan is barrier-free");
            (out, root)
        });
        assert_eq!(out, serial);
        assert_eq!(installs(root), 0, "the cells ran in place on the worker");
    });
}
