//! A pinned static grid (`Partition::StaticGrid { pin: Some(..) }`) must not
//! widen the SPMD team beyond the `Exec`'s budget, nor make a refused team
//! spawn threads. Own test binary: it counts the process's OS threads.

use std::time::Duration;

use strided_view::{StridedView, StridedViewMut};
use tcontract::api::Error;
use tcontract::{Partition, Plan, PlanConfig};
use tcontract_exec::{Exec, Pool};

mod common;
use common::plans::matmul_problem;

fn os_threads() -> usize {
    std::fs::read_dir("/proc/self/task")
        .map(|d| d.count())
        .unwrap_or(0)
}

fn plan(config: PlanConfig) -> Plan<f64> {
    let (m, n, k) = (256usize, 240usize, 200usize);
    Plan::<f64>::new(&matmul_problem(m, n, k), &config).unwrap()
}

/// One contraction into `out`.
fn fill(plan: &Plan<f64>, exec: &Exec<'_>, out: &mut [f64]) -> Result<(), Error> {
    let (m, n, k) = (256usize, 240usize, 200usize);
    let av: Vec<f64> = (0..m * k).map(|x| (x % 13) as f64).collect();
    let bv: Vec<f64> = (0..k * n).map(|x| (x % 7) as f64).collect();
    plan.execute_into(
        exec,
        1.0,
        &StridedView::new(&av, &[m, k], &[1, m as isize], 0).unwrap(),
        &StridedView::new(&bv, &[k, n], &[1, k as isize], 0).unwrap(),
        &mut StridedViewMut::new(out, &[m, n], &[1, m as isize], 0).unwrap(),
    )
}

fn gemm(plan: &Plan<f64>, exec: &Exec<'_>) -> Vec<f64> {
    let (m, n) = (256usize, 240usize);
    let mut out = vec![0.0; m * n];
    fill(plan, exec, &mut out).unwrap();
    out
}

#[test]
fn pinned_partition_respects_host_width_and_never_spawns() {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let tp = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let pool = Pool::borrow(&tp);
        let plan = plan(PlanConfig {
            partition: Some(Partition::StaticGrid {
                pin: Some((4, 2)),
                align_c_lines: false,
            }),
            ..PlanConfig::default()
        });
        // Budget below the pool size: the pinned 4x2 grid must shrink to it.
        let exec = Exec::rayon(&pool).with_budget(3).unwrap();
        let wide = gemm(&plan, &exec);
        assert_eq!(pool.stats().broadcasts, 1);
        // A worker caller cannot be co-scheduled: the pinned team is refused
        // without spawning and without a write.
        let before = os_threads();
        let mut out = vec![-7.0; 256 * 240];
        let err = exec
            .install(2, |_| fill(&plan, &exec, &mut out))
            .unwrap_err();
        assert!(matches!(err, Error::Exec(_)), "{err}");
        assert!(out.iter().all(|&v| v == -7.0), "a refused route wrote");
        assert!(!wide.is_empty(), "the wide run produced no output");
        if before > 0 {
            assert_eq!(os_threads(), before);
        }
        let _ = tx.send(());
    });
    rx.recv_timeout(Duration::from_secs(60))
        .expect("pinned partition test failed or deadlocked");
}
