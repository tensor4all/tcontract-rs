//! Regression: a forced family id resolves in a contraction plan with no
//! earlier call having registered the providers (single test, own binary).
// The cplx families exist only on x86_64 (`tcontract-kernel` is empty elsewhere), so
// off x86_64 there is no provider whose registration order could be tested.
#![cfg(target_arch = "x86_64")]
use num_complex::Complex64;
use tcontract::api::{DType, DotGeneral, LayoutSpec, OperandSpec, Problem};
use tcontract::{Plan, PlanConfig};
use tcontract_kernel::KernelChoice;

#[test]
fn a_forced_id_resolves_on_the_packed_path_first_thing() {
    let cfg = DotGeneral::new(&[1], &[0], &[], &[]);
    let spec = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
    let problem = Problem::from_dot_general(
        DType::C64,
        spec(&[3, 4], &[1, 3]),
        spec(&[4, 5], &[1, 4]),
        spec(&[3, 5], &[1, 3]),
        &cfg,
    )
    .unwrap();
    let config = PlanConfig {
        kernel: KernelChoice::Id("avx2.c64.native.4x4".into()),
        ..PlanConfig::default()
    };
    let plan = Plan::<Complex64>::new(&problem, &config);
    // UnknownId would mean the plan resolved the id before registering; a host
    // without AVX2+FMA refuses it as unsupported, which also proves it is known.
    match plan {
        Ok(p) => assert_eq!(
            p.report().packed.as_ref().unwrap().family_id,
            "avx2.c64.native.4x4"
        ),
        Err(e) => assert!(!format!("{e:?}").contains("UnknownId"), "{e:?}"),
    }
}
