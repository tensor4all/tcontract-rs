//! Regressions from the Phase 1c review.
use tcontract::api::{DType, DotGeneral, Error, LayoutSpec, OperandSpec, Problem, ShapeError};
use tcontract::{add, Algorithm, Plan, PlanConfig};
use tcontract_exec::Exec;

mod common;
use common::*;

fn spec(dims: &[usize], strides: &[isize]) -> OperandSpec {
    OperandSpec::new(LayoutSpec::new(dims, strides, 0).unwrap())
}

fn forced_packed() -> PlanConfig {
    PlanConfig::packed()
}

#[test]
fn extent_one_axes_with_huge_strides_do_not_overflow() {
    // add: C [2, 1, 3] with an isize::MAX stride on the unit axis.
    let a = T::<f64>::new(&[2, 1, 3], 1);
    let mut c = T::<f64> {
        data: vec![1.0; 6],
        dims: vec![2, 1, 3],
        strides: vec![1, isize::MAX, 2],
        offset: 0,
    };
    add(&mut Exec::serial(), 2.0, &a.view(), 3.0, &mut c.view_mut()).unwrap();
    for_each_index(&[2, 1, 3], |i| assert_eq!(c.get(i), 2.0 * a.get(i) + 3.0));
    // alpha == 0 through a plan scales C, same layout, under every strategy.
    let cfg = DotGeneral::new(&[], &[], &[], &[]);
    let x = T::<f64>::new(&[2], 1);
    let y = T::<f64>::new(&[1, 3], 2);
    for config in [PlanConfig::default(), forced_packed()] {
        let mut c = T::<f64> {
            data: vec![1.0; 6],
            dims: vec![2, 1, 3],
            strides: vec![1, isize::MAX, 2],
            offset: 0,
        };
        let problem = Problem::from_dot_general(
            DType::F64,
            spec(&x.dims, &x.strides),
            spec(&y.dims, &y.strides),
            spec(&c.dims, &c.strides),
            &cfg,
        )
        .unwrap();
        let plan = Plan::<f64>::new(&problem, &config).unwrap();
        plan.execute_into_accum(
            &mut Exec::serial(),
            0.0,
            &x.view(),
            &y.view(),
            4.0,
            tcontract::api::AccumulationSource::Output,
            &mut c.view_mut(),
        )
        .unwrap();
        assert!(c.data.iter().all(|&v| v == 4.0), "{:?}", plan.report());
    }
}

#[test]
fn overflowing_element_counts_are_shape_errors() {
    let d = [1usize << 33, 1 << 33];
    let s = [1isize, 1 << 33];
    let cfg = DotGeneral::new(&[0, 1], &[0, 1], &[], &[]);
    let e = Problem::from_dot_general(DType::F64, spec(&d, &s), spec(&d, &s), spec(&[], &[]), &cfg);
    assert!(
        matches!(e, Err(Error::Shape(ShapeError::Overflow { .. }))),
        "{e:?}"
    );
}

#[test]
fn all_batch_contractions_run_elementwise() {
    let cfg = DotGeneral::new(&[], &[], &[0, 1], &[1, 0]);
    let a = T::<f64>::new(&[3, 4], 1);
    let b = T::<f64>::new(&[4, 3], 2);
    let c0 = T::<f64>::new(&[3, 4], 3);
    let want = reference(&cfg, 1.5, &a, false, &b, false, 0.5, &c0);
    let mut c = c0.clone();
    let problem = Problem::from_dot_general(
        DType::F64,
        spec(&a.dims, &a.strides),
        spec(&b.dims, &b.strides),
        spec(&c.dims, &c.strides),
        &cfg,
    )
    .unwrap();
    let plan = Plan::<f64>::new(&problem, &PlanConfig::default()).unwrap();
    assert_eq!(plan.report().algorithm, Algorithm::Elementwise);
    plan.execute_into_accum(
        &mut Exec::serial(),
        1.5,
        &a.view(),
        &b.view(),
        0.5,
        tcontract::api::AccumulationSource::Output,
        &mut c.view_mut(),
    )
    .unwrap();
    assert!(rel_err(&c, &want) < 1e-14);
}

#[test]
fn empty_problems_are_copy_free() {
    let cfg = DotGeneral::new(&[], &[], &[], &[]);
    let a = T::<f64>::new(&[2, 0, 3], 1);
    let a = T {
        strides: vec![1, 2, 2],
        ..a
    };
    let b = T::<f64>::new(&[4], 2);
    let c = T::<f64>::new(&[2, 0, 3, 4], 3);
    let problem = Problem::from_dot_general(
        DType::F64,
        spec(&a.dims, &a.strides),
        spec(&b.dims, &b.strides),
        spec(&c.dims, &c.strides),
        &cfg,
    )
    .unwrap();
    let config = PlanConfig {
        no_materialize: true,
        ..PlanConfig::default()
    };
    let plan = Plan::<f64>::new(&problem, &config).unwrap();
    assert_eq!(plan.report().materialized, [false; 3]);
}
