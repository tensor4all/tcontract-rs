//! tensor4all/tcontract-rs#163: a plan's own time estimate, and the work count behind it.
//!
//! The work is a fact about the problem (`Problem::macs`, weighted for the scalar); the time is
//! that over a caller-supplied rate, divided by the width and added to the token's dispatch cost —
//! the expression the shared width rule minimises. Nothing here asserts an absolute wall-clock
//! value: the rate is supplied by the test, so every expectation is a ratio or a model identity.

use std::time::Duration;

use num_complex::Complex64;
use tcontract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
use tcontract::{Plan, PlanConfig};
use tcontract_exec::host::muladds_per_ns;
use tcontract_exec::{Exec, Pool};

/// `D[i,j] = sum_k A[i,k] B[k,j]`, column-major.
fn matmul(m: usize, n: usize, k: usize, dtype: DType) -> Problem {
    let spec = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
    Problem::from_labels(
        dtype,
        spec(&[m, k], &[1, m as isize]),
        spec(&[k, n], &[1, k as isize]),
        CSpec::Absent,
        spec(&[m, n], &[1, m as isize]),
        &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    )
    .unwrap()
}

/// `D = A * B` elementwise: every axis is a batch axis, so `macs` is the element count.
fn hadamard(dims: &[usize]) -> Problem {
    let strides: Vec<isize> = {
        let mut acc = 1isize;
        dims.iter()
            .map(|d| {
                let s = acc;
                acc *= *d as isize;
                s
            })
            .collect()
    };
    let spec = |d: &[usize]| OperandSpec::new(LayoutSpec::new(d, &strides, 0).unwrap());
    let labels: Vec<i64> = (0..dims.len() as i64).collect();
    Problem::from_labels(
        DType::F64,
        spec(dims),
        spec(dims),
        CSpec::Absent,
        spec(dims),
        &Labels::new(&labels, &labels, &labels),
    )
    .unwrap()
}

fn nanoseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e9
}

/// A four-slot pool whose dispatch costs are synthetic and unmistakable, so an estimate that
/// reconstructed a host default instead of reading the token would be caught.
fn four_slots() -> Pool<'static> {
    let flat = |micros| threadteam::WidthCosts::flat(Duration::from_micros(micros));
    let costs = threadteam::DispatchCosts::new(flat(11), flat(7), flat(5));
    let pool = threadteam::Pool::builder()
        .threads(4)
        .dispatch_costs(costs)
        .build()
        .unwrap();
    Pool::owned(pool)
}

#[test]
fn the_work_is_the_problem_count_weighted_for_the_scalar() {
    let plan = Plan::<f64>::new(&matmul(2, 4, 3, DType::F64), &PlanConfig::default()).unwrap();
    // 2 * 4 * 3 multiply-adds, whatever the width, the budget or the rate.
    assert_eq!(plan.muladds(), 24.0);
    assert_eq!(plan.muladds(), plan.problem().macs() as f64);

    // A complex multiply-add is four real ones, so a complex plan counts four times as much.
    let complex = Plan::<Complex64>::new(&matmul(2, 4, 3, DType::C64), &PlanConfig::default())
        .expect("a complex plan");
    assert_eq!(complex.muladds(), 4.0 * 24.0);

    // An all-batch (Hadamard) problem has no contraction: its work is the element count.
    let elementwise = Plan::<f64>::new(&hadamard(&[2, 3]), &PlanConfig::default()).unwrap();
    assert_eq!(elementwise.muladds(), 6.0);
}

#[test]
fn the_work_does_not_depend_on_the_strategy() {
    // The same problem planned on the faer route (the default for a small GEMM) and on the packed
    // driver: one work count, because the model is the problem's, not the route's.
    let problem = matmul(64, 64, 64, DType::F64);
    let fusable = Plan::<f64>::new(&problem, &PlanConfig::default()).unwrap();
    let packed = Plan::<f64>::new(&problem, &PlanConfig::packed()).unwrap();
    assert_ne!(fusable.report().algorithm, packed.report().algorithm);
    assert_eq!(fusable.muladds(), packed.muladds());
}

#[test]
fn the_estimate_is_ideal_division_plus_the_token_dispatch_cost() {
    let pool = four_slots();
    let mut exec = Exec::new(&pool, 4).unwrap();
    let token = exec.reborrow();
    // 64 multiply-adds at 8 per nanosecond: 8 ns of compute, then divided by the width.
    let plan = Plan::<f64>::new(&matmul(4, 4, 4, DType::F64), &PlanConfig::default()).unwrap();
    assert_eq!(plan.muladds(), 64.0);
    let rate = 8.0;
    let dispatch = Duration::from_micros(5);
    assert_eq!(plan.estimate(1, rate, &token), Duration::from_nanos(8));
    assert_eq!(
        plan.estimate(2, rate, &token),
        Duration::from_nanos(4) + dispatch
    );
    assert_eq!(
        plan.estimate(4, rate, &token),
        Duration::from_nanos(2) + dispatch
    );
    // Width is clamped to the token's budget: asking for more slots than there are is not an
    // estimate of a width this token cannot run.
    assert_eq!(
        plan.estimate(9, rate, &token),
        plan.estimate(4, rate, &token)
    );
    // Zero and one mean the serial estimate, and neither dispatches.
    assert_eq!(plan.estimate(0, rate, &token), Duration::from_nanos(8));
}

#[test]
fn a_boundary_the_model_cannot_express_is_saturated_not_panicked() {
    let exec = Exec::serial();
    let plan = Plan::<f64>::new(&matmul(8, 8, 8, DType::F64), &PlanConfig::default()).unwrap();
    // A positive rate small enough that the quotient cannot be a `Duration`.
    assert_eq!(plan.estimate(1, 1e-300, &exec), Duration::MAX);
    // No output at all: no work, and a serial context dispatches nothing, so no time.
    let empty = Plan::<f64>::new(&matmul(0, 8, 8, DType::F64), &PlanConfig::default()).unwrap();
    assert_eq!(empty.muladds(), 0.0);
    assert_eq!(empty.estimate(1, muladds_per_ns(), &exec), Duration::ZERO);
    // An empty contraction has no multiply-adds: the model counts the contraction, exactly as the
    // width rule does, and the output pass it still runs is not part of that count.
    let no_k = Plan::<f64>::new(&matmul(8, 8, 0, DType::F64), &PlanConfig::default()).unwrap();
    assert_eq!(no_k.muladds(), 0.0);
    assert_eq!(no_k.estimate(1, muladds_per_ns(), &exec), Duration::ZERO);
}

#[test]
fn the_estimate_scales_with_the_rate_not_with_the_network() {
    let exec = Exec::serial();
    let plan = Plan::<f64>::new(&matmul(64, 64, 64, DType::F64), &PlanConfig::default()).unwrap();
    let one = plan.estimate(1, muladds_per_ns(), &exec);
    let double = plan.estimate(1, 2.0 * muladds_per_ns(), &exec);
    assert!(
        (nanoseconds(one) / 2.0 - nanoseconds(double)).abs() <= 1.0,
        "{one:?} vs {double:?}"
    );
    // A bigger contraction is more work and more time, at the same rate.
    let bigger =
        Plan::<f64>::new(&matmul(128, 128, 128, DType::F64), &PlanConfig::default()).unwrap();
    assert!(bigger.estimate(1, muladds_per_ns(), &exec) > one);
}

#[test]
fn a_rate_that_cannot_produce_a_time_is_refused() {
    let exec = Exec::serial();
    let plan = Plan::<f64>::new(&matmul(8, 8, 8, DType::F64), &PlanConfig::default()).unwrap();
    for rate in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(plan.estimate(1, rate, &exec), Duration::MAX, "rate {rate}");
    }
}

#[test]
fn the_estimate_is_in_the_unit_the_width_rule_prices() {
    // The shared rule is asked for a width in nanoseconds. The published estimate must be in that
    // unit, or a caller comparing the two is comparing seconds with microseconds. The dispatch row
    // is synthetic, so the boundary is exact: 1 us costs, threshold `PAYS = 3` dispatches.
    let pool = four_slots();
    let mut exec = Exec::new(&pool, 4).unwrap();
    let token = exec.reborrow();
    let small = Plan::<f64>::new(&matmul(16, 16, 16, DType::F64), &PlanConfig::default()).unwrap();
    let big = Plan::<f64>::new(&matmul(512, 512, 512, DType::F64), &PlanConfig::default()).unwrap();
    let nanos = |plan: &Plan<f64>| plan.estimate(1, muladds_per_ns(), &token).as_secs_f64() * 1e9;
    // 16^3 = 4096 multiply-adds is about 260 ns of serial work: far below one 5 us dispatch, so
    // the rule keeps it on the caller. 512^3 is about 8.6 ms: far above, so it grants a team.
    assert!(
        nanos(&small) < nanoseconds(Duration::from_micros(5)),
        "{}",
        nanos(&small)
    );
    assert!(nanos(&big) > nanoseconds(Duration::from_micros(5)));
    assert_eq!(token.width_for(nanos(&small), 1), 1);
    assert!(token.width_for(nanos(&big), 1) > 1);
}
