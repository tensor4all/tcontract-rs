use super::*;

#[test]
fn synthetic_shared_l3_retargets_nc_to_effective_width() {
    let mut rg = ResolvedGemm::<f32>::resolve::<f32>(&KernelChoice::Auto, 1).unwrap();
    rg.policy = BlockingPolicy {
        model: BlockModel::Analytical,
        hierarchy: cache::BUILTIN,
        overrides: None,
        explicit: None,
        kc_couple: None,
    };
    let budget = rg.with_threads(8).unwrap();
    let serial = budget.with_threads(1).unwrap();
    assert!(
        serial.nc > budget.nc,
        "serial NC={} budget NC={}",
        serial.nc,
        budget.nc
    );
    assert_eq!((serial.mc, serial.kc), (budget.mc, budget.kc));
    assert_eq!(serial.effective_threads, 1);
}

#[test]
fn percentage_overrides_apply_once_and_overflow_is_typed() {
    // Forced: the built-in menus are always present, so Auto is the widest
    // ISA's family and its blocking is machine-dependent.
    let mut rg =
        ResolvedGemm::<f64>::resolve::<f64>(&KernelChoice::Id("ref.f64.real.4x4".into()), 1)
            .unwrap();
    rg.policy = BlockingPolicy {
        model: BlockModel::Legacy,
        hierarchy: cache::BUILTIN,
        explicit: None,
        kc_couple: None,
        overrides: Some(BlockingOverride {
            mc: None,
            kc: None,
            nc: None,
            mc_pct: Some(50),
            nc_pct: Some(50),
        }),
    };
    let scaled = rg.with_threads(8).unwrap();
    assert_eq!((scaled.mc, scaled.kc, scaled.nc), (32, 256, 512));
    assert_eq!(scaled.with_threads(1).unwrap().mc, 32);
    assert_eq!(scaled.with_threads(1).unwrap().nc, 512);
    rg.policy.overrides = Some(BlockingOverride {
        mc: None,
        kc: None,
        nc: None,
        mc_pct: Some(usize::MAX),
        nc_pct: Some(100),
    });
    assert!(matches!(
        rg.with_threads(1),
        Err(SelectError::Incompatible {
            reason: "blocking arithmetic overflow",
            ..
        })
    ));
}
