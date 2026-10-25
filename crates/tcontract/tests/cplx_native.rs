//! The native interleaved complex families through the packed contraction
//! path: a contraction whose operands cannot fuse, forced onto
//! `avx2.c64.native.4x4`, reported, and equal to the reference.
use num_complex::Complex64;
use tcontract::api::{AccumulationSource, DType, DotGeneral, LayoutSpec, Op, OperandSpec, Problem};
use tcontract::{Plan, PlanConfig};
use tcontract_exec::Exec;
use tcontract_kernel::KernelChoice;

mod common;
use common::{out_dims, reference, T};

fn have_isa() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[test]
fn a_packed_contraction_runs_the_native_family_and_matches_the_reference() {
    // Contract A[., 1, 2] with B[2, ., 0] (the copying case of kernel_select);
    // B stored transposed and A reversed so nothing fuses.
    let cfg = DotGeneral::new(&[1, 2], &[2, 0], &[], &[]);
    let a0 = T::<Complex64>::new(&[5, 4, 6], 1);
    let b0 = T::<Complex64>::new(&[6, 7, 4], 2);
    let a = a0.reversed();
    let b = b0.restride(&[2, 0, 1]);
    let od = out_dims(&cfg, &a.dims, &b.dims);
    let c0 = T::<Complex64>::new(&od, 3);
    let spec = |t: &T<Complex64>, op| {
        OperandSpec::new(LayoutSpec::new(&t.dims, &t.strides, t.offset).unwrap()).with_op(op)
    };
    let problem = Problem::from_dot_general(
        DType::C64,
        spec(&a, Op::Conjugate),
        spec(&b, Op::Identity),
        spec(&c0, Op::Identity),
        &cfg,
    )
    .unwrap();
    let config = PlanConfig {
        kernel: KernelChoice::Id("avx2.c64.native.4x4".into()),
        ..PlanConfig::default()
    };
    let plan = Plan::<Complex64>::new(&problem, &config);
    if !have_isa() {
        // Refused at plan creation, never run.
        assert!(plan.is_err(), "an unavailable family must be refused");
        return;
    }
    let plan = plan.unwrap();
    let report = plan.report().packed.as_ref().expect("packed plan");
    assert_eq!(report.family_id, "avx2.c64.native.4x4");
    let (alpha, beta) = (Complex64::new(0.8, -0.3), Complex64::new(0.4, 0.2));
    let mut c = c0.clone();
    plan.execute_into_accum(
        &mut Exec::serial(),
        alpha,
        &a.view(),
        &b.view(),
        beta,
        AccumulationSource::Output,
        &mut c.view_mut(),
    )
    .unwrap();
    let want = reference(&cfg, alpha, &a, true, &b, false, beta, &c0);
    let (mut worst, mut scale) = (0.0f64, 0.0f64);
    common::for_each_index(&c.dims, |i| {
        worst = worst.max((c.get(i) - want.get(i)).norm());
        scale = scale.max(want.get(i).norm());
    });
    // K = 4 * 6 = 24 products of O(1) values: a few K eps.
    assert!(
        worst <= 24.0 * 8.0 * f64::EPSILON * scale.max(1.0),
        "{worst:e}"
    );
}
