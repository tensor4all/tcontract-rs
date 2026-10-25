//! A pinned static grid (`Partition::StaticGrid { pin: Some(..) }`) must not
//! widen the SPMD team beyond the `Exec`'s budget, nor make a refused team
//! (which falls back to barrier-free cells) spawn threads. Own test binary: it
//! counts the process's OS threads.

use std::time::Duration;

use strided_view::{StridedView, StridedViewMut};
use tcontract::api::Error;
use tcontract::{Partition, Plan, PlanConfig};
use tcontract_exec::{Exec, Pool};

mod common;
use common::placement::{installs, max_live, on_worker, pool, root};
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
fn fill(plan: &Plan<f64>, exec: &mut Exec<'_>, out: &mut [f64]) -> Result<(), Error> {
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

fn gemm(plan: &Plan<f64>, exec: &mut Exec<'_>) -> Vec<f64> {
    let (m, n) = (256usize, 240usize);
    let mut out = vec![0.0; m * n];
    fill(plan, exec, &mut out).unwrap();
    out
}

#[test]
fn pinned_partition_respects_host_width_and_never_spawns() {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let sp = pool(4);
        let pool = Pool::borrow(&sp);
        let plan = plan(PlanConfig {
            partition: Some(Partition::StaticGrid {
                pin: Some((4, 2)),
                align_c_lines: false,
            }),
            ..PlanConfig::default()
        });
        let serial = gemm(&plan, &mut Exec::serial());
        // Budget below the pool size: the pinned 4x2 grid must shrink to it.
        let mut exec = Exec::new(&pool, 3).unwrap();
        let team = root(&mut exec);
        let wide = gemm(&plan, &mut exec);
        assert_eq!(wide, serial);
        assert_eq!(
            max_live(team),
            3,
            "the team is the budget, not the pinned grid"
        );
        // A worker caller cannot be co-scheduled: the pinned team is refused
        // without spawning, and the call runs barrier-free cells in place.
        let before = os_threads();
        let (out, root) = on_worker(&pool, || {
            let mut inner = Exec::new(&pool, 3).unwrap();
            let root = root(&mut inner);
            let mut out = vec![-7.0; 256 * 240];
            fill(&plan, &mut inner, &mut out).expect("the refused team falls back");
            (out, root)
        });
        assert_eq!(out, serial);
        assert_eq!(installs(root), 0);
        assert!(max_live(root) <= 3);
        if before > 0 {
            assert_eq!(os_threads(), before);
        }
        let _ = tx.send(());
    });
    rx.recv_timeout(Duration::from_secs(60))
        .expect("pinned partition test failed or deadlocked");
}
