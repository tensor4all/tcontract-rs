//! Kernel selection at plan creation: the claim of this API is that a plan
//! resolves its kernel when it is built, reports it, and refuses an unusable
//! one there rather than inside a contraction.
use tcontract::api::{DType, DotGeneral, Error, LayoutSpec, Op, OperandSpec, Problem};
use tcontract::{Algorithm, Plan, PlanConfig};
use tcontract_kernel::{KernelChoice, SelectError};

mod common;
use common::{out_dims, T};

fn spec<S>(t: &T<S>) -> OperandSpec {
    OperandSpec::new(LayoutSpec::new(&t.dims, &t.strides, 0).unwrap()).with_op(Op::Identity)
}

/// The corpus case whose operands cannot both fuse copy-free, so that the
/// planner chooses the packed driver and there is a resolution to report.
fn copying_problem() -> Problem {
    let cfg = DotGeneral::new(&[1, 2], &[2, 0], &[], &[]);
    let a = T::<f64>::new(&[3, 4, 5], 1);
    let b = T::<f64>::new(&[5, 6, 4], 2);
    let c = T::<f64>::new(&out_dims(&cfg, &a.dims, &b.dims), 3);
    Problem::from_dot_general(DType::F64, spec(&a), spec(&b), spec(&c), &cfg).unwrap()
}

fn plan_with(config: &PlanConfig) -> tcontract::Result<Plan<f64>> {
    Plan::<f64>::new(&copying_problem(), config)
}

#[test]
fn a_plan_resolves_and_reports_its_kernel() {
    let plan = plan_with(&PlanConfig::default()).unwrap();
    assert_eq!(plan.report().algorithm, Algorithm::Packed);
    let report = plan
        .report()
        .packed
        .as_ref()
        .expect("the planner chose the packed driver");
    assert!(!report.family_id.is_empty());
    assert!(report.mr > 0 && report.nr > 0 && report.kc > 0);
    // The family id names the ISA; the tuning class names the datapath width
    // the default menu was ordered by, which the id cannot carry (#156).
    assert_eq!(
        report.tuning_class,
        tcontract_kernel::TuningClass::detect(),
        "the report names the process's tuning class"
    );

    // A forced family is the one reported.
    let forced = PlanConfig {
        kernel: KernelChoice::Id("ref.f64.real.4x4".into()),
        ..PlanConfig::default()
    };
    assert_eq!(
        plan_with(&forced)
            .unwrap()
            .report()
            .packed
            .as_ref()
            .unwrap()
            .family_id,
        "ref.f64.real.4x4"
    );

    // An id that does not exist fails at creation, not at execution, with the
    // typed selection error.
    let bad = PlanConfig {
        kernel: KernelChoice::Id("not.a.kernel".into()),
        ..PlanConfig::default()
    };
    let err = plan_with(&bad).unwrap_err();
    assert!(
        matches!(&err, Error::Select(SelectError::UnknownId { id }) if id == "not.a.kernel"),
        "unexpected error: {err}"
    );
}

#[test]
fn a_plan_that_does_not_use_the_packed_driver_reports_no_family() {
    // Column-major matmul fuses copy-free: faer, no family.
    let cfg = DotGeneral::new(&[1], &[0], &[], &[]);
    let a = T::<f64>::new(&[5, 4], 1);
    let b = T::<f64>::new(&[4, 3], 2);
    let c = T::<f64>::new(&out_dims(&cfg, &a.dims, &b.dims), 3);
    let problem =
        Problem::from_dot_general(DType::F64, spec(&a), spec(&b), spec(&c), &cfg).unwrap();
    let plan = Plan::<f64>::new(&problem, &PlanConfig::default()).unwrap();
    assert_eq!(plan.report().algorithm, Algorithm::Faer);
    assert!(plan.report().packed.is_none());
}
