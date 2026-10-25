//! No `Exec` route of this crate starts Rayon's process-global pool: serial
//! and pooled execution of the packed driver (team, lanes and the barrier-free
//! fallback), the faer strategy and the elementwise strategy all run on the
//! pool the host built. Alone in its test binary, and a single test, because the
//! check starts the global pool itself (`ledger::global_pool_unstarted`) and
//! passes only when nothing earlier did.
use strided_view::{StridedView, StridedViewMut};
use tcontract::api::{DType, DotGeneral, LayoutSpec, OperandSpec, Problem};
use tcontract::{Plan, PlanConfig};
use tcontract_exec::{Exec, Pool};

mod common;
use common::placement::{on_worker, pool};
use common::plans::{matmul_problem, packed_plan};

fn gemm(plan: &Plan<f64>, exec: &mut Exec<'_>, m: usize, n: usize, k: usize) -> Vec<f64> {
    let a: Vec<f64> = (0..m * k).map(|x| (x % 13) as f64 - 6.0).collect();
    let b: Vec<f64> = (0..k * n).map(|x| (x % 7) as f64 * 0.5).collect();
    let mut d = vec![0.0; m * n];
    plan.execute_into(
        exec,
        1.0,
        &StridedView::new(&a, &[m, k], &[1, m as isize], 0).unwrap(),
        &StridedView::new(&b, &[k, n], &[1, k as isize], 0).unwrap(),
        &mut StridedViewMut::new(&mut d, &[m, n], &[1, m as isize], 0).unwrap(),
    )
    .unwrap();
    d
}

/// `D[i,j,h] = sum_k A[i,k,h] B[k,j,h]`: a batch the faer strategy serves.
fn batched(m: usize, n: usize, k: usize, items: usize) -> (Plan<f64>, Vec<f64>, Vec<f64>) {
    let spec = |d: &[usize]| {
        let mut s = vec![1isize];
        for &x in &d[..d.len() - 1] {
            s.push(s.last().unwrap() * x as isize);
        }
        OperandSpec::new(LayoutSpec::new(d, &s, 0).unwrap())
    };
    let problem = Problem::from_dot_general(
        DType::F64,
        spec(&[m, k, items]),
        spec(&[k, n, items]),
        spec(&[m, n, items]),
        &DotGeneral::new(&[1], &[0], &[2], &[2]),
    )
    .unwrap();
    let plan = Plan::<f64>::new(&problem, &PlanConfig::default()).unwrap();
    let a = (0..m * k * items).map(|x| (x % 13) as f64 - 6.0).collect();
    let b = (0..k * n * items).map(|x| (x % 7) as f64 * 0.5).collect();
    (plan, a, b)
}

fn run_batched(plan: &Plan<f64>, a: &[f64], b: &[f64], exec: &mut Exec<'_>, d: [usize; 4]) {
    let [m, n, k, h] = d;
    let mut out = vec![0.0; m * n * h];
    plan.execute_into(
        exec,
        1.0,
        &StridedView::new(a, &[m, k, h], &[1, m as isize, (m * k) as isize], 0).unwrap(),
        &StridedView::new(b, &[k, n, h], &[1, k as isize, (k * n) as isize], 0).unwrap(),
        &mut StridedViewMut::new(&mut out, &[m, n, h], &[1, m as isize, (m * n) as isize], 0)
            .unwrap(),
    )
    .unwrap();
}

#[test]
fn no_route_starts_the_global_pool() {
    let (m, n, k) = (192usize, 160, 128);
    let plan = packed_plan(m, n, k);
    let (fplan, fa, fb) = batched(8, 8, 8, 512);
    let (bplan, ba, bb) = batched(128, 128, 128, 2);
    let elementwise =
        Plan::<f64>::new(&matmul_problem(300, 300, 1), &PlanConfig::default()).unwrap();

    // Serial: every strategy on the calling thread.
    let mut serial = Exec::serial();
    let want = gemm(&plan, &mut serial, m, n, k);
    run_batched(&fplan, &fa, &fb, &mut serial, [8, 8, 8, 512]);
    run_batched(&bplan, &ba, &bb, &mut serial, [128, 128, 128, 2]);
    gemm(&elementwise, &mut serial, 300, 300, 1);

    // Pooled: the packed team, the batch lanes, faer through `external`, the
    // elementwise pass, and the barrier-free fallback from a worker.
    let sp = pool(4);
    let pool = Pool::borrow(&sp);
    let mut exec = Exec::new(&pool, 4).unwrap();
    assert_eq!(gemm(&plan, &mut exec, m, n, k), want);
    run_batched(&fplan, &fa, &fb, &mut exec, [8, 8, 8, 512]);
    run_batched(&bplan, &ba, &bb, &mut exec, [128, 128, 128, 2]);
    gemm(&elementwise, &mut exec, 300, 300, 1);
    let nested = on_worker(&pool, || {
        let mut inner = Exec::new(&pool, 4).unwrap();
        gemm(&plan, &mut inner, m, n, k)
    });
    assert_eq!(nested, want);

    assert!(
        threadteam::ledger::global_pool_unstarted(),
        "something started the process-global Rayon pool"
    );
}
