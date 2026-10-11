//! The packed SPMD driver on a borrowed pool through `Exec::broadcast`.

use std::time::Duration;

use strided_view::{StridedView, StridedViewMut};
use tcontract::api::Error;
use tcontract::{Plan, PlanConfig};
use tcontract_exec::{Exec, Pool};
use tcontract_kernel::KernelChoice;

mod common;
use common::plans::{matmul_problem, packed_plan};

const M: usize = 256;
const N: usize = 240;
const K: usize = 200;

/// One contraction of `plan` into `out`.
fn fill(plan: &Plan<f64>, exec: &Exec<'_>, out: &mut [f64]) -> Result<(), Error> {
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

fn gemm(plan: &Plan<f64>, exec: &Exec<'_>) -> Vec<f64> {
    let mut out = vec![0.0; M * N];
    fill(plan, exec, &mut out).unwrap();
    out
}

fn pool4() -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
}

#[test]
fn contraction_runs_spmd_on_the_borrowed_pool() {
    let plan = packed_plan(M, N, K);
    let tp = pool4();
    let pool = Pool::borrow(&tp);
    let exec = Exec::rayon(&pool);
    let serial = gemm(&plan, &Exec::serial());
    let par = gemm(&plan, &exec);
    assert_eq!(par, serial);
    assert_eq!(pool.stats().broadcasts, 1);
}

#[test]
fn nested_contraction_on_a_worker_is_refused_without_a_write() {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let plan = packed_plan(M, N, K);
        let tp = pool4();
        let pool = Pool::borrow(&tp);
        let exec = Exec::rayon(&pool);
        // A barrier-bearing team cannot be co-scheduled from a worker of its
        // own pool; the call says so and leaves the output alone.
        let mut out = vec![-7.0; M * N];
        let err = exec
            .install(2, |_| fill(&plan, &exec, &mut out))
            .unwrap_err();
        assert!(matches!(err, Error::Exec(_)), "{err}");
        assert!(out.iter().all(|&v| v == -7.0), "a refused route wrote");
        assert_eq!(pool.stats().broadcasts, 0);
        let _ = tx.send(());
    });
    rx.recv_timeout(Duration::from_secs(60))
        .expect("nested SPMD deadlocked");
}

/// `pm == 1` takes no barrier -- every column group has one thread and reads
/// only its own slice of the panel -- so a worker caller runs those cells in
/// place instead of being refused.
#[test]
fn a_direct_b_plan_runs_barrier_free_cells_from_a_worker() {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let config = PlanConfig {
            kernel: KernelChoice::Id("ref.f64.direct-b.4x4".into()),
            ..PlanConfig::default()
        };
        let plan = Plan::<f64>::new(&matmul_problem(M, N, K), &config).unwrap();
        let tp = pool4();
        let pool = Pool::borrow(&tp);
        let exec = Exec::rayon(&pool);
        let serial = gemm(&plan, &Exec::serial());
        pool.reset_stats();
        let mut out = vec![0.0; M * N];
        tp.install(|| fill(&plan, &exec, &mut out))
            .expect("a direct-B plan is barrier-free");
        assert_eq!(out, serial);
        let stats = pool.stats();
        assert_eq!(stats.broadcasts, 0, "a barrier-free plan never broadcasts");
        assert_eq!(stats.inline_runs, 1, "the cells ran in place on the worker");
        let _ = tx.send(());
    });
    rx.recv_timeout(Duration::from_secs(60))
        .expect("barrier-free cells deadlocked");
}
