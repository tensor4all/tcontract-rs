//! #123: the batch policy carried on `Exec` drives the packed route.
//!
//! Forcing `Outer`, `Inner` or `Split` must give the bitwise-serial result for
//! an item-serial kernel (the packed driver guarantees that at any grid), and a
//! forced strategy the route cannot serve must be refused before any output is
//! written.
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

mod common;
use common::placement::{installs, pool};

fn run(
    plan: &Plan<f64>,
    exec: &mut Exec<'_>,
    a: &[f64],
    b: &[f64],
    d: &mut [f64],
    m: usize,
    n: usize,
) {
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
    let sp = pool(8);
    let p = Pool::borrow(&sp);

    let mut want = vec![0.0f64; m * n];
    run(&plan, &mut Exec::serial(), &a, &b, &mut want, m, n);

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
        let mut exec = Exec::new(&p, budget)
            .unwrap()
            .with_batch_policy(BatchPolicy::new(strategy));
        let mut got = vec![0.0f64; m * n];
        run(&plan, &mut exec, &a, &b, &mut got, m, n);
        assert_eq!(got, want, "strategy {strategy:?} at budget {budget}");
    }
}

/// `D[i,j,h] = sum_k A[i,k,h] B[k,j,h]`, column-major, `h` outermost.
fn batched(m: usize, n: usize, k: usize, h: usize) -> Problem {
    let spec = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
    let (mi, ni, ki) = (m as isize, n as isize, k as isize);
    Problem::from_labels(
        DType::F64,
        spec(&[m, k, h], &[1, mi, mi * ki]),
        spec(&[k, n, h], &[1, ki, ki * ni]),
        CSpec::Absent,
        spec(&[m, n, h], &[1, mi, mi * ni]),
        &Labels::new(&[0, 2, 3], &[2, 1, 3], &[0, 1, 3]),
    )
    .unwrap()
}

/// A genuine batch axis: the forced lane route must report and run its lane
/// width, and an unsupported split must be refused before any write.
#[test]
fn forced_lanes_on_a_batch_report_and_run_their_width() {
    use tcontract::{Error, ExecutionRoute, OutputContract, PackedRoute};

    let (m, n, k, h) = (8usize, 8, 8, 8);
    let plan = Plan::<f64>::new(&batched(m, n, k, h), &PlanConfig::packed()).unwrap();
    let sp = pool(8);
    let p = Pool::borrow(&sp);
    let exec = |strategy| {
        Exec::new(&p, 8)
            .unwrap()
            .with_batch_policy(BatchPolicy::new(strategy))
    };
    let (mi, ni, ki) = (m as isize, n as isize, k as isize);
    let a: Vec<f64> = (0..m * k * h)
        .map(|i| (i * 7 + 3) as f64 * 0.11 - 1.0)
        .collect();
    let b: Vec<f64> = (0..k * n * h)
        .map(|i| (i * 5 + 1) as f64 * 0.07 - 0.5)
        .collect();
    let mut want = vec![0.0f64; m * n * h];
    plan.execute_into(
        &mut Exec::serial(),
        1.0,
        &StridedView::new(&a, &[m, k, h], &[1, mi, mi * ki], 0).unwrap(),
        &StridedView::new(&b, &[k, n, h], &[1, ki, ki * ni], 0).unwrap(),
        &mut StridedViewMut::new(&mut want, &[m, n, h], &[1, mi, mi * ni], 0).unwrap(),
    )
    .unwrap();
    let run = |exec: &mut Exec<'_>, d: &mut [f64]| {
        plan.execute_into(
            exec,
            1.0,
            &StridedView::new(&a, &[m, k, h], &[1, mi, mi * ki], 0).unwrap(),
            &StridedView::new(&b, &[k, n, h], &[1, ki, ki * ni], 0).unwrap(),
            &mut StridedViewMut::new(d, &[m, n, h], &[1, mi, mi * ni], 0).unwrap(),
        )
    };

    // Outer at budget eight over eight items: eight lane workers.
    let mut outer = exec(BatchStrategy::Outer);
    assert_eq!(
        plan.execution_route(&outer, 1.0, OutputContract::Initialized)
            .unwrap(),
        ExecutionRoute::Packed {
            kind: PackedRoute::BatchLanes,
            width: 8
        }
    );
    let mut got = vec![0.0f64; m * n * h];
    run(&mut outer, &mut got).unwrap();
    assert_eq!(got, want, "Outer lanes changed the result");

    // Sequential forces the serial route even at budget eight.
    let mut sequential = exec(BatchStrategy::Sequential);
    assert_eq!(
        plan.execution_route(&sequential, 1.0, OutputContract::Initialized)
            .unwrap(),
        ExecutionRoute::Packed {
            kind: PackedRoute::Serial,
            width: 1
        }
    );
    let mut got = vec![0.0f64; m * n * h];
    run(&mut sequential, &mut got).unwrap();
    assert_eq!(got, want, "Sequential changed the result");

    // Split { lanes: 2 } at budget eight asks for two lanes and four item
    // threads; the packed route runs items serially per lane, so it is refused
    // before any write rather than silently discarding the item width.
    let mut split = exec(BatchStrategy::Split {
        lanes: NonZeroUsize::new(2).unwrap(),
    });
    let err = plan
        .execution_route(&split, 1.0, OutputContract::Initialized)
        .unwrap_err();
    assert!(
        matches!(err, Error::Exec(ExecError::StrategyUnavailable { .. })),
        "{err}"
    );
    let mut sentinel = vec![7.0f64; m * n * h];
    let err = run(&mut split, &mut sentinel).unwrap_err();
    assert!(
        matches!(err, Error::Exec(ExecError::StrategyUnavailable { .. })),
        "{err}"
    );
    assert!(
        sentinel.iter().all(|&x| x == 7.0),
        "a refused split wrote to D"
    );
}

/// A forced strategy is resolved before a pinned grid: `Sequential` still
/// serializes it, and an outer route that cannot honor the pin is refused.
#[test]
fn forced_policy_overrides_a_pinned_grid() {
    use tcontract::{Error, ExecutionRoute, OutputContract, PackedRoute, Partition};

    let (m, n, k, h) = (8usize, 8, 8, 8);
    let config = PlanConfig {
        partition: Some(Partition::StaticGrid {
            pin: Some((4, 1)),
            align_c_lines: false,
        }),
        ..PlanConfig::default()
    };
    let plan = Plan::<f64>::new(&batched(m, n, k, h), &config).unwrap();
    let sp = pool(8);
    let p = Pool::borrow(&sp);
    let with = |strategy| {
        Exec::new(&p, 8)
            .unwrap()
            .with_batch_policy(BatchPolicy::new(strategy))
    };

    // Sequential wins over the pinned grid.
    let sequential = with(BatchStrategy::Sequential);
    assert_eq!(
        plan.execution_route(&sequential, 1.0, OutputContract::Initialized)
            .unwrap(),
        ExecutionRoute::Packed {
            kind: PackedRoute::Serial,
            width: 1
        }
    );

    // Outer would use batch lanes and silently ignore the pin: reject.
    let outer = with(BatchStrategy::Outer);
    let err = plan
        .execution_route(&outer, 1.0, OutputContract::Initialized)
        .unwrap_err();
    assert!(
        matches!(err, Error::Exec(ExecError::StrategyUnavailable { .. })),
        "{err}"
    );
}

/// `D[i,j] = sum_k A[i,k] B[k,j] + beta * D[i,j]`, in-place C.
fn matmul_output_c(m: usize, n: usize, k: usize) -> Problem {
    let spec = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
    Problem::from_labels(
        DType::F64,
        spec(&[m, k], &[1, m as isize]),
        spec(&[k, n], &[1, k as isize]),
        CSpec::Output(tcontract::api::Op::Identity),
        spec(&[m, n], &[1, m as isize]),
        &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    )
    .unwrap()
}

/// A forced `Sequential` limits the faer `beta * D` output pass too: an output
/// above the parallel threshold must not enter the pool.
#[test]
fn forced_sequential_serializes_the_faer_output_pass() {
    use tcontract::api::AccumulationSource;
    use tcontract::Algorithm;

    // A 256x256 output (65536 elements) is above the strided parallel threshold;
    // the contraction is shallow so faer still serves it.
    let (m, n, k) = (256usize, 256usize, 8usize);
    let plan = Plan::<f64>::new(&matmul_output_c(m, n, k), &PlanConfig::default()).unwrap();
    assert_eq!(plan.report().algorithm, Algorithm::Faer);
    let sp = pool(4);
    let p = Pool::borrow(&sp);
    let mut exec = Exec::new(&p, 4)
        .unwrap()
        .with_batch_policy(BatchPolicy::new(BatchStrategy::Sequential));

    let a = vec![1.0f64; m * k];
    let b = vec![1.0f64; k * n];
    let mut d = vec![1.0f64; m * n];
    let av = StridedView::new(&a, &[m, k], &[1, m as isize], 0).unwrap();
    let bv = StridedView::new(&b, &[k, n], &[1, k as isize], 0).unwrap();
    let mut dv = StridedViewMut::new(&mut d, &[m, n], &[1, m as isize], 0).unwrap();
    let root = common::placement::root(&mut exec);
    plan.execute_into_accum(
        &mut exec,
        1.0,
        &av,
        &bv,
        0.5,
        AccumulationSource::Output,
        &mut dv,
    )
    .unwrap();
    assert_eq!(
        installs(root),
        0,
        "a forced sequential route entered the pool for the output pass"
    );
}
