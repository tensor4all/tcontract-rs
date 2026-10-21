use core::mem::MaybeUninit;
use tcontract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
use tcontract::{
    Algorithm, ExecutionRoute, OutputContract, Plan, PlanConfig, SliceAccumulationSource,
};
use tcontract_exec::Exec;

fn operand(dims: &[usize], strides: &[isize]) -> OperandSpec {
    OperandSpec::new(LayoutSpec::new(dims, strides, 0).unwrap())
}

#[test]
fn faer_matvec_serves_a_fresh_output_on_the_faer_route() {
    let problem = Problem::from_labels(
        DType::F64,
        operand(&[2, 2], &[2, 1]),
        operand(&[2, 1], &[1, 2]),
        CSpec::Absent,
        operand(&[2, 1], &[1, 2]),
        &Labels::new(&[0, 1], &[1, 2], &[0, 2]),
    )
    .unwrap();
    // A plan needs no preparation for a fresh output: `fresh_output` is
    // accepted but changes nothing.
    let unprepared = Plan::<f64>::new(&problem, &PlanConfig::default()).unwrap();
    assert_eq!(
        unprepared
            .execution_route(&Exec::serial(), 1.0, OutputContract::Fresh)
            .unwrap(),
        ExecutionRoute::Faer
    );
    assert_eq!(
        unprepared
            .execution_route(&Exec::serial(), 0.0, OutputContract::Fresh)
            .unwrap(),
        ExecutionRoute::OutputOnly
    );
    let config = PlanConfig {
        fresh_output: true,
        ..PlanConfig::default()
    };
    let plan = Plan::<f64>::new(&problem, &config).unwrap();
    assert_eq!(plan.report().algorithm, Algorithm::Faer);
    assert_eq!(
        plan.execution_route(&Exec::serial(), 1.0, OutputContract::Initialized)
            .unwrap(),
        ExecutionRoute::Faer
    );
    assert_eq!(
        plan.execution_route(&Exec::serial(), 1.0, OutputContract::Fresh)
            .unwrap(),
        ExecutionRoute::Faer
    );
    let mut output = [MaybeUninit::uninit(); 2];
    let d = plan
        .execute_uninit_slices(
            &Exec::serial(),
            1.0,
            (&[1., 2., 3., 4.], 0),
            (&[5., 6.], 0),
            0.0,
            SliceAccumulationSource::Absent,
            (&mut output, 0),
        )
        .unwrap();
    assert_eq!(d, &[17., 39.]);
}

#[test]
fn alpha_zero_does_not_read_nan_inputs_and_initializes_the_entire_output() {
    let problem = Problem::from_labels(
        DType::F64,
        operand(&[3], &[1]),
        operand(&[3], &[1]),
        CSpec::Absent,
        operand(&[3], &[-1]),
        &Labels::new(&[0], &[0], &[0]),
    )
    .unwrap();
    let plan = Plan::<f64>::new(&problem, &PlanConfig::default()).unwrap();
    let mut output = [MaybeUninit::uninit(); 3];
    let d = plan
        .execute_uninit_slices(
            &Exec::serial(),
            0.0,
            (&[f64::NAN; 3], 0),
            (&[f64::NAN; 3], 0),
            0.0,
            SliceAccumulationSource::Absent,
            (&mut output, 2),
        )
        .unwrap();
    assert_eq!(d, &[0.; 3]);
}

#[test]
fn reduced_diagonal_output_cannot_claim_initialization_of_holes() {
    let problem = Problem::from_labels(
        DType::F64,
        operand(&[2], &[1]),
        operand(&[2], &[1]),
        CSpec::Absent,
        operand(&[2, 2], &[1, 2]),
        &Labels::new(&[0], &[0], &[0, 0]),
    )
    .unwrap();
    let plan = Plan::<f64>::new(&problem, &PlanConfig::default()).unwrap();
    let mut output = [MaybeUninit::new(123.); 4];
    assert!(plan
        .execute_uninit_slices(
            &Exec::serial(),
            1.0,
            (&[2., 3.], 0),
            (&[4., 5.], 0),
            0.0,
            SliceAccumulationSource::Absent,
            (&mut output, 0)
        )
        .is_err());
    // SAFETY: all slots started initialized, and before-write rejection leaves them so.
    assert_eq!(output.map(|x| unsafe { x.assume_init() }), [123.; 4]);
}

#[test]
fn absent_nonzero_or_nan_beta_and_fresh_output_accumulation_are_rejected() {
    let l = operand(&[1], &[1]);
    let absent = Problem::from_labels(
        DType::F64,
        l.clone(),
        l.clone(),
        CSpec::Absent,
        l.clone(),
        &Labels::new(&[0], &[0], &[0]),
    )
    .unwrap();
    let plan = Plan::<f64>::new(&absent, &PlanConfig::default()).unwrap();
    for beta in [1.0, f64::NAN] {
        let mut d = [777.];
        assert!(plan
            .execute_slices_accum(
                &Exec::serial(),
                1.0,
                (&[2.], 0),
                (&[3.], 0),
                beta,
                SliceAccumulationSource::Absent,
                (&mut d, 0)
            )
            .is_err());
        assert_eq!(d, [777.]);
    }
    let output_problem = Problem::from_labels(
        DType::F64,
        l.clone(),
        l.clone(),
        CSpec::Output(Default::default()),
        l,
        &Labels::new(&[0], &[0], &[0]),
    )
    .unwrap();
    let output_plan = Plan::<f64>::new(&output_problem, &PlanConfig::default()).unwrap();
    let mut d = [MaybeUninit::new(777.)];
    assert!(output_plan
        .execute_uninit_slices(
            &Exec::serial(),
            1.0,
            (&[2.], 0),
            (&[3.], 0),
            1.0,
            SliceAccumulationSource::Output,
            (&mut d, 0)
        )
        .is_err());
    // SAFETY: initialized sentinel remains after rejection.
    assert_eq!(unsafe { d[0].assume_init() }, 777.);
}

#[test]
fn rank_zero_and_empty_output_have_exact_completion_contracts() {
    let l = operand(&[], &[]);
    let p = Problem::from_labels(
        DType::F64,
        l.clone(),
        l.clone(),
        CSpec::Absent,
        l,
        &Labels::new(&[], &[], &[]),
    )
    .unwrap();
    let plan = Plan::<f64>::new(&p, &PlanConfig::default()).unwrap();
    let mut d = [MaybeUninit::uninit()];
    assert_eq!(
        plan.execute_uninit_slices(
            &Exec::serial(),
            1.0,
            (&[2.], 0),
            (&[3.], 0),
            0.0,
            SliceAccumulationSource::Absent,
            (&mut d, 0)
        )
        .unwrap(),
        &[6.]
    );
    let l = operand(&[0], &[1]);
    let p = Problem::from_labels(
        DType::F64,
        l.clone(),
        l.clone(),
        CSpec::Absent,
        l,
        &Labels::new(&[0], &[0], &[0]),
    )
    .unwrap();
    let plan = Plan::<f64>::new(&p, &PlanConfig::default()).unwrap();
    let mut empty = [];
    assert!(plan
        .execute_uninit_slices(
            &Exec::serial(),
            1.0,
            (&[], 0),
            (&[], 0),
            0.0,
            SliceAccumulationSource::Absent,
            (&mut empty, 0)
        )
        .unwrap()
        .is_empty());
    let mut extra = [MaybeUninit::uninit()];
    assert!(plan
        .execute_uninit_slices(
            &Exec::serial(),
            1.0,
            (&[], 0),
            (&[], 0),
            0.0,
            SliceAccumulationSource::Absent,
            (&mut extra, 0)
        )
        .is_err());
}

/// Every write-only C mode serves a fresh output on the Faer route: no C, the
/// output itself at `beta == 0`, and a separate C at zero and nonzero `beta`.
/// The 2x2 problem contracts `A` (row-major) with `B`; `D = alpha * A * B`
/// plus `beta * C` where C is read.
#[test]
fn fresh_faer_outputs_cover_every_write_only_c_mode() {
    use tcontract::api::Op;
    let a = [1.0f64, 2.0, 3.0, 4.0]; // row-major 2x2: [[1, 2], [3, 4]]
    let b = [5.0f64, 6.0, 7.0, 8.0]; // column-major 2x2: [[5, 7], [6, 8]]
    let c = [10.0f64, 20.0, 30.0, 40.0]; // column-major
                                         // A * B, column-major: [[17, 23], [39, 53]] -> [17, 39, 23, 53]
    let product = [17.0f64, 39.0, 23.0, 53.0];
    let build = |c_spec: CSpec| {
        let labels = Labels::new(&[0, 1], &[1, 2], &[0, 2]);
        let labels = if matches!(c_spec, CSpec::Separate(_)) {
            labels.with_c(&[0, 2])
        } else {
            labels
        };
        Problem::from_labels(
            DType::F64,
            operand(&[2, 2], &[2, 1]),
            operand(&[2, 2], &[1, 2]),
            c_spec,
            operand(&[2, 2], &[1, 2]),
            &labels,
        )
        .unwrap()
    };
    let cases: [(CSpec, f64, SliceAccumulationSource<'_, f64>, [f64; 4]); 4] = [
        (CSpec::Absent, 0.0, SliceAccumulationSource::Absent, product),
        (
            CSpec::Output(Op::Identity),
            0.0,
            SliceAccumulationSource::Output,
            product,
        ),
        (
            CSpec::Separate(operand(&[2, 2], &[1, 2])),
            0.0,
            SliceAccumulationSource::Separate((&c, 0)),
            product,
        ),
        (
            CSpec::Separate(operand(&[2, 2], &[1, 2])),
            2.0,
            SliceAccumulationSource::Separate((&c, 0)),
            [37.0, 79.0, 83.0, 133.0],
        ),
    ];
    for (c_spec, beta, source, want) in cases {
        let plan = Plan::<f64>::new(&build(c_spec), &PlanConfig::default()).unwrap();
        assert_eq!(plan.report().algorithm, Algorithm::Faer);
        assert_eq!(
            plan.execution_route(&Exec::serial(), 1.0, OutputContract::Fresh)
                .unwrap(),
            ExecutionRoute::Faer
        );
        let mut output = [MaybeUninit::uninit(); 4];
        let d = plan
            .execute_uninit_slices(
                &Exec::serial(),
                1.0,
                (&a, 0),
                (&b, 0),
                beta,
                source,
                (&mut output, 0),
            )
            .unwrap();
        assert_eq!(d, &want, "beta {beta}");
    }
}
