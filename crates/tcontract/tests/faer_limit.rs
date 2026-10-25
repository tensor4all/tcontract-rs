//! Rule 3's per-dtype GEMM volume bound (#63): large fused problems take the
//! packed driver, small ones and unbounded dtypes keep faer.

use num_complex::Complex64;
use strided_view::{StridedView, StridedViewMut};
use tcontract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
use tcontract::{Algorithm, FaerLimit, Plan, PlanConfig, Reason};
use tcontract_exec::Exec;

fn col_major(dims: &[usize]) -> Vec<isize> {
    let mut s = Vec::new();
    let mut acc = 1isize;
    for &d in dims {
        s.push(acc);
        acc *= d as isize;
    }
    s
}

/// The MPS step `ab,asc->bsc` at bond `chi` and physical dimension 2.
fn mps_step(dtype: DType, chi: usize) -> Problem {
    let l = |d: &[usize]| OperandSpec::new(LayoutSpec::new(d, &col_major(d), 0).unwrap());
    Problem::from_labels(
        dtype,
        l(&[chi, chi]),
        l(&[chi, 2, chi]),
        CSpec::Absent,
        l(&[chi, 2, chi]),
        &Labels::new(&[0, 1], &[0, 2, 3], &[1, 2, 3]),
    )
    .unwrap()
}

#[test]
fn c64_above_the_bound_runs_packed() {
    let plan = Plan::<Complex64>::new(&mps_step(DType::C64, 64), &PlanConfig::default()).unwrap();
    let r = plan.report();
    assert_eq!(r.algorithm, Algorithm::Packed);
    assert_eq!(
        r.reason,
        Reason::AboveFaerLimit {
            volume: 2 * 64 * 64 * 64,
            limit: 1 << 17
        }
    );
}

#[test]
fn c64_below_the_bound_keeps_faer() {
    for chi in [4, 16, 32] {
        let plan =
            Plan::<Complex64>::new(&mps_step(DType::C64, chi), &PlanConfig::default()).unwrap();
        assert_eq!(plan.report().algorithm, Algorithm::Faer, "chi = {chi}");
        assert_eq!(plan.report().reason, Reason::Fused);
    }
}

#[test]
fn unbounded_dtypes_and_no_limit_keep_faer() {
    let plan = Plan::<f64>::new(&mps_step(DType::F64, 64), &PlanConfig::default()).unwrap();
    assert_eq!(plan.report().algorithm, Algorithm::Faer);
    let cfg = PlanConfig {
        faer_limit: FaerLimit::NONE,
        ..PlanConfig::default()
    };
    assert!(!cfg.requires_packed());
    let plan = Plan::<Complex64>::new(&mps_step(DType::C64, 64), &cfg).unwrap();
    assert_eq!(plan.report().algorithm, Algorithm::Faer);
}

#[test]
fn a_custom_bound_applies_to_its_dtype() {
    let cfg = PlanConfig {
        faer_limit: FaerLimit {
            f64: Some(1000),
            ..FaerLimit::NONE
        },
        ..PlanConfig::default()
    };
    let plan = Plan::<f64>::new(&mps_step(DType::F64, 8), &cfg).unwrap();
    assert_eq!(plan.report().algorithm, Algorithm::Packed);
    assert_eq!(
        plan.report().reason,
        Reason::AboveFaerLimit {
            volume: 1024,
            limit: 1000
        }
    );
}

#[test]
fn reasons_of_the_other_rules() {
    assert_eq!(
        Plan::<Complex64>::new(&mps_step(DType::C64, 4), &PlanConfig::packed())
            .unwrap()
            .report()
            .reason,
        Reason::Forced
    );
    let l = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
    let hadamard = Problem::from_labels(
        DType::F64,
        l(&[3, 3], &[1, 3]),
        l(&[3, 3], &[1, 3]),
        CSpec::Absent,
        l(&[3, 3], &[1, 3]),
        &Labels::new(&[0, 1], &[0, 1], &[0, 1]),
    )
    .unwrap();
    let r = Plan::<f64>::new(&hadamard, &PlanConfig::default()).unwrap();
    assert_eq!(r.report().reason, Reason::AllBatch);
    // A K axis split across non-chaining strides in A cannot fuse.
    let split = Problem::from_labels(
        DType::F64,
        l(&[3, 2, 2], &[1, 6, 3]),
        l(&[2, 2, 3], &[1, 2, 4]),
        CSpec::Absent,
        l(&[3, 3], &[1, 3]),
        &Labels::new(&[0, 1, 2], &[1, 2, 3], &[0, 3]),
    )
    .unwrap();
    let r = Plan::<f64>::new(&split, &PlanConfig::default()).unwrap();
    assert_eq!(r.report().reason, Reason::NotFusable);
}

#[test]
fn the_packed_route_above_the_bound_is_correct() {
    let chi = 64;
    let p = mps_step(DType::C64, chi);
    let packed = Plan::<Complex64>::new(&p, &PlanConfig::default()).unwrap();
    let faer = Plan::<Complex64>::new(
        &p,
        &PlanConfig {
            faer_limit: FaerLimit::NONE,
            ..PlanConfig::default()
        },
    )
    .unwrap();
    let a: Vec<Complex64> = (0..chi * chi)
        .map(|i| Complex64::new((i % 7) as f64 - 3.0, (i % 5) as f64))
        .collect();
    let b: Vec<Complex64> = (0..chi * 2 * chi)
        .map(|i| Complex64::new((i % 3) as f64, 1.0 - (i % 4) as f64))
        .collect();
    let ad = [chi, chi];
    let a_strides = col_major(&ad);
    let av = StridedView::new(&a, &ad, &a_strides, 0).unwrap();
    let bd = [chi, 2, chi];
    let b_strides = col_major(&bd);
    let bv = StridedView::new(&b, &bd, &b_strides, 0).unwrap();
    let mut d1 = vec![Complex64::new(0.0, 0.0); chi * 2 * chi];
    let mut d2 = d1.clone();
    let mut v1 = StridedViewMut::new(&mut d1, &bd, &b_strides, 0).unwrap();
    packed
        .execute_into(
            &mut Exec::serial(),
            Complex64::new(1.0, 0.0),
            &av,
            &bv,
            &mut v1,
        )
        .unwrap();
    let mut v2 = StridedViewMut::new(&mut d2, &bd, &b_strides, 0).unwrap();
    faer.execute_into(
        &mut Exec::serial(),
        Complex64::new(1.0, 0.0),
        &av,
        &bv,
        &mut v2,
    )
    .unwrap();
    // Integer-valued inputs: both routes are exact.
    assert_eq!(d1, d2);
}
