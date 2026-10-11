//! The knobs that used to be `TENSORCONTRACT_*` variables are explicit inputs:
//! each one changes the resolution it is passed to, and the process
//! environment changes nothing.
use tcontract_kernel::blocking::BlockModel;
use tcontract_kernel::{
    resolve_legacy_auto, BlockingOverride, ComplexMethod, Isa, KernelChoice, KernelForce,
    ResolvedGemm, Tuning, C64,
};

fn tuned(t: Tuning) -> ResolvedGemm<f64> {
    ResolvedGemm::<f64>::resolve::<f64>(&KernelChoice::Id("ref.f64.real.4x4".into()), 1)
        .unwrap()
        .with_tuning(&t)
        .unwrap()
}

#[test]
fn blocking_overrides_apply_to_the_resolution_they_are_given() {
    let base = tuned(Tuning::default());
    let kc = tuned(Tuning {
        blocking: BlockingOverride {
            kc: Some(8),
            ..Default::default()
        },
        ..Tuning::default()
    });
    assert_eq!(kc.kc, 8);
    assert_eq!((kc.mc, kc.nc), (base.mc, base.nc));
    let half = tuned(Tuning {
        blocking: BlockingOverride {
            mc_pct: Some(50),
            ..Default::default()
        },
        ..Tuning::default()
    });
    assert_eq!(
        half.mc,
        (base.mc / 2).next_multiple_of(half.mr).max(half.mr)
    );
    // The model derives its own numbers; the default tuning is the legacy one.
    let model = tuned(Tuning {
        block_model: BlockModel::Analytical,
        ..Tuning::default()
    });
    assert_eq!(model.family().id, base.family().id);
}

#[test]
fn legacy_auto_honours_the_isa_preference_and_the_method() {
    let scalar = Tuning {
        kernel_force: KernelForce::Scalar,
        ..Tuning::default()
    };
    let rg = resolve_legacy_auto::<f64>(ComplexMethod::Planar, &scalar, 1).unwrap();
    assert_eq!(rg.family().isa, Isa::Portable);
    for method in ComplexMethod::ALL {
        let rg = resolve_legacy_auto::<C64>(method, &scalar, 1).unwrap();
        assert_eq!(rg.family().isa, Isa::Portable, "{method}");
        let m = rg.family().complex.unwrap().method;
        assert_eq!(
            m,
            match method {
                ComplexMethod::Planar => tcontract_kernel::Method::Native,
                ComplexMethod::OneM => tcontract_kernel::Method::OneM,
                ComplexMethod::ThreeM => tcontract_kernel::Method::ThreeM,
            }
        );
    }
}

#[test]
fn the_process_environment_is_not_read() {
    let before = tuned(Tuning::default());
    // SAFETY: the only test in this binary that touches the environment.
    unsafe {
        std::env::set_var("TENSORCONTRACT_KC", "999");
        std::env::set_var("TENSORCONTRACT_BLOCKMODEL", "model");
        std::env::set_var("TENSORCONTRACT_KERNEL", "scalar");
        std::env::set_var("TPRIMS_GEMM_KERNEL", "does.not.exist");
    }
    let after = tuned(Tuning::default());
    assert_eq!(
        (before.mc, before.kc, before.nc),
        (after.mc, after.kc, after.nc)
    );
    let auto = ResolvedGemm::<f64>::resolve::<f64>(&KernelChoice::Auto, 1).unwrap();
    assert!(auto.family().allow_auto);
}

#[test]
fn a_coupled_kc_rederives_mc_and_nc_at_that_depth() {
    let base = tuned(Tuning::default());
    let coupled = tuned(Tuning {
        kc_couple: Some(64),
        ..Tuning::default()
    });
    assert_eq!(coupled.kc, 64);
    // A shallower depth leaves room for more rows and columns in the same budget.
    assert!(coupled.mc > base.mc && coupled.nc > base.nc);
    // The plain kc override changes kc alone.
    let plain = tuned(Tuning {
        blocking: BlockingOverride {
            kc: Some(64),
            ..Default::default()
        },
        ..Tuning::default()
    });
    assert_eq!((plain.mc, plain.nc), (base.mc, base.nc));
}
