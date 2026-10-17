//! #123: the batch policy carried on `Exec` drives the packed route.
//!
//! Forcing `Outer`, `Inner` or `Split` must give the bitwise-serial result for
//! an item-serial kernel (the packed driver guarantees that at any grid), and a
//! forced `WholeBatchVendor` on a route without a vendor batched call must be
//! refused before any output is written.
use std::num::NonZeroUsize;

use strided_view::{StridedView, StridedViewMut};
use tcontract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
use tcontract::{Plan, PlanConfig};
use tcontract_exec::{BatchPolicy, BatchStrategy, Exec, ExecError, Pool};

/// `D[i,j] = sum_k A[i,k] B[k,j]`, column-major.
fn matmul(m: usize, n: usize, k: usize) -> Problem {
    let spec = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
    Problem::from_labels(
        DType::F64,
        spec(&[m, k], &[1, m as isize]),
        spec(&[k, n], &[1, k as isize]),
        CSpec::Absent,
        spec(&[m, n], &[1, m as isize]),
        &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    )
    .unwrap()
}

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
}

fn run(plan: &Plan<f64>, exec: &Exec<'_>, a: &[f64], b: &[f64], d: &mut [f64], m: usize, n: usize) {
    let k = a.len() / m;
    plan.execute_into(
        exec,
        1.0,
        &StridedView::new(a, &[m, k], &[1, m as isize], 0).unwrap(),
        &StridedView::new(b, &[k, n], &[1, k as isize], 0).unwrap(),
        &mut StridedViewMut::new(d, &[m, n], &[1, m as isize], 0).unwrap(),
    )
    .unwrap();
}

#[test]
fn forced_lanes_are_bitwise_serial_for_the_packed_route() {
    let (m, n, k) = (12usize, 9usize, 7usize);
    let a: Vec<f64> = (0..m * k)
        .map(|i| (i * 31 + 7) as f64 * 0.13 - 2.0)
        .collect();
    let b: Vec<f64> = (0..k * n)
        .map(|i| (i * 17 + 3) as f64 * 0.07 - 1.0)
        .collect();
    // `PlanConfig::packed` forces the packed driver, which is the item-serial
    // route the identity is stated for.
    let plan = Plan::<f64>::new(&matmul(m, n, k), &PlanConfig::packed()).unwrap();
    let tp = pool(8);
    let p = Pool::borrow(&tp);

    let mut want = vec![0.0f64; m * n];
    run(&plan, &Exec::serial(), &a, &b, &mut want, m, n);

    for (strategy, budget) in [
        (BatchStrategy::Sequential, 4),
        (BatchStrategy::Outer, 4),
        (BatchStrategy::Inner, 4),
        (
            BatchStrategy::Split {
                lanes: NonZeroUsize::new(3).unwrap(),
            },
            6,
        ),
    ] {
        let exec = Exec::rayon(&p)
            .with_budget(budget)
            .unwrap()
            .with_batch_policy(BatchPolicy::new(strategy));
        let mut got = vec![0.0f64; m * n];
        run(&plan, &exec, &a, &b, &mut got, m, n);
        assert_eq!(got, want, "strategy {strategy:?} at budget {budget}");
    }
}

#[test]
fn forced_vendor_is_refused_before_any_write() {
    let (m, n, k) = (4usize, 3usize, 5usize);
    let a: Vec<f64> = (0..m * k).map(|i| i as f64 + 1.0).collect();
    let b: Vec<f64> = (0..k * n).map(|i| i as f64 * 0.5 + 1.0).collect();
    let plan = Plan::<f64>::new(&matmul(m, n, k), &PlanConfig::packed()).unwrap();
    let tp = pool(4);
    let p = Pool::borrow(&tp);
    let exec = Exec::rayon(&p)
        .with_budget(4)
        .unwrap()
        .with_batch_policy(BatchPolicy::new(BatchStrategy::WholeBatchVendor));

    let mut d = vec![17.5f64; m * n];
    let err = plan
        .execute_into(
            &exec,
            1.0,
            &StridedView::new(&a, &[m, k], &[1, m as isize], 0).unwrap(),
            &StridedView::new(&b, &[k, n], &[1, k as isize], 0).unwrap(),
            &mut StridedViewMut::new(&mut d, &[m, n], &[1, m as isize], 0).unwrap(),
        )
        .unwrap_err();
    assert!(
        matches!(
            err,
            tcontract::Error::Exec(ExecError::StrategyUnavailable { .. })
        ),
        "{err}"
    );
    assert!(d.iter().all(|&x| x == 17.5), "a refused route wrote to D");

    // `execution_route` reports the same refusal, still without a write.
    let err = plan
        .execution_route(&exec, 1.0, tcontract::OutputContract::Initialized)
        .unwrap_err();
    assert!(
        matches!(
            err,
            tcontract::Error::Exec(ExecError::StrategyUnavailable { .. })
        ),
        "{err}"
    );
}
