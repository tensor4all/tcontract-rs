use super::super::{c_for_call, PackedPlan};
use crate::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
use crate::PlanConfig;

#[test]
fn absent_c_at_zero_beta_uses_only_destination_batch_offsets() {
    let s = |dims: &[usize], strides: &[isize]| {
        OperandSpec::new(LayoutSpec::new(dims, strides, 0).unwrap())
    };
    let p = Problem::from_labels(
        DType::F64,
        s(&[2, 2, 1], &[2, 1, 2]),
        s(&[2, 1, 2], &[2, 1, 1]),
        CSpec::Separate(s(&[2, 2, 2], &[10_000, 1, 2])),
        s(&[2, 2, 2], &[4, 1, 2]),
        &Labels::new(&[0, 1, 2], &[0, 2, 3], &[0, 1, 3]).with_c(&[0, 1, 3]),
    )
    .unwrap();
    let plan = PackedPlan::from_problem(&p, &PlanConfig::packed()).unwrap();
    assert_eq!(plan.h_c, [0, 10_000]);
    let mut d = [0_f64; 8];
    let dp = d.as_mut_ptr();
    for beta in [0.0, -0.0] {
        let (cp, hc) = c_for_call(&plan, beta, core::ptr::null(), dp);
        assert_eq!(cp, dp.cast_const());
        assert_eq!(hc, &[0, 4]); // All resulting offsets fit the actual D.
    }
    let c = [0_f64; 10_004];
    for beta in [1.0, f64::NAN] {
        let (cp, hc) = c_for_call(&plan, beta, c.as_ptr(), dp);
        assert_eq!(cp, c.as_ptr());
        assert_eq!(hc, &[0, 10_000]);
    }
    // Exercise the same normalization through the public, safe fresh boundary.
    let public = crate::Plan::<f64>::new(&p, &PlanConfig::packed()).unwrap();
    let mut fresh = [core::mem::MaybeUninit::uninit(); 8];
    let initialized = public
        .execute_uninit_slices(
            &tcontract_exec::Exec::serial(),
            1.0,
            (&[1., 2., 3., 4.], 0),
            (&[1.; 4], 0),
            0.0,
            crate::SliceAccumulationSource::Absent,
            (&mut fresh, 0),
        )
        .unwrap();
    assert_eq!(initialized, &[1., 2., 1., 2., 3., 4., 3., 4.]);
}
