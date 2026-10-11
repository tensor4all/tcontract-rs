//! Plan configuration is explicit: every knob that used to be a
//! `TENSORCONTRACT_*` variable is a field of [`PlanConfig`], the environment
//! changes nothing, and a bad request is refused rather than weakened.
use tcontract::api::{ConfigError, Error};
use tcontract::{CacheModel, Orient, Partition, Plan, PlanConfig, RowBlock, Writeback};
use tcontract_kernel::blocking::BlockModel;
use tcontract_kernel::{BlockingOverride, KernelChoice, KernelForce, Method, SelectError};

mod common;
use common::plans::matmul_problem;

fn plan(config: &PlanConfig) -> Result<Plan<f64>, Error> {
    Plan::<f64>::new(&matmul_problem(96, 80, 64), config)
}

#[test]
fn a_pinned_scalar_kernel_and_a_blocking_override_reach_the_resolution() {
    let config = PlanConfig {
        isa: KernelForce::Scalar,
        blocking: BlockingOverride {
            kc: Some(8),
            ..Default::default()
        },
        ..PlanConfig::default()
    };
    let p = plan(&config).unwrap();
    let report = p
        .report()
        .packed
        .as_ref()
        .expect("a tuning request forces packed");
    assert!(report.family_id.starts_with("ref."), "{}", report.family_id);
    assert_eq!(report.kc, 8);
}

#[test]
fn every_tuning_request_forces_the_packed_driver() {
    use tcontract::Algorithm;
    // The default configuration fuses this matmul copy-free: faer.
    assert_eq!(
        plan(&PlanConfig::default()).unwrap().report().algorithm,
        Algorithm::Faer
    );
    let requests = [
        PlanConfig {
            kernel: KernelChoice::Id("ref.f64.real.4x4".into()),
            ..PlanConfig::default()
        },
        PlanConfig {
            partition: Some(Partition::StaticGrid {
                pin: None,
                align_c_lines: false,
            }),
            ..PlanConfig::default()
        },
        PlanConfig {
            blocking: BlockingOverride {
                mc: Some(16),
                ..Default::default()
            },
            ..PlanConfig::default()
        },
        PlanConfig {
            cache_model: CacheModel {
                block_model: BlockModel::Analytical,
                ..Default::default()
            },
            ..PlanConfig::default()
        },
        PlanConfig {
            writeback: Writeback::Gather,
            ..PlanConfig::default()
        },
    ];
    for r in requests {
        assert_eq!(
            plan(&r).unwrap().report().algorithm,
            Algorithm::Packed,
            "{r:?}"
        );
    }
    // Orientation and row block shape the packed plan but do not force it.
    let shaped = PlanConfig {
        orientation: Orient::Force(true),
        row_block: RowBlock::Base,
        ..PlanConfig::default()
    };
    assert_eq!(plan(&shaped).unwrap().report().algorithm, Algorithm::Faer);
}

#[test]
fn a_pinned_grid_is_reported_and_honoured() {
    let config = PlanConfig {
        partition: Some(Partition::StaticGrid {
            pin: Some((3, 2)),
            align_c_lines: true,
        }),
        ..PlanConfig::default()
    };
    let p = plan(&config).unwrap();
    let report = p.report().packed.as_ref().unwrap();
    assert_eq!(
        report.partition,
        tcontract_kernel::PartitionPolicy::StaticGrid { pm: 3, pn: 2 }
    );
    assert!(report.align_c_lines);
}

/// A pinned grid whose product overflows must be rejected when the plan is
/// built, not when the driver multiplies `pm * pn` unchecked.
#[test]
fn an_overflowing_pinned_grid_is_rejected() {
    let err = plan(&PlanConfig {
        partition: Some(Partition::StaticGrid {
            pin: Some((usize::MAX / 2 + 1, 2)),
            align_c_lines: false,
        }),
        ..PlanConfig::default()
    })
    .expect_err("an overflowing grid built a plan");
    assert!(matches!(err, Error::Select(_)), "{err}");
}

#[test]
fn blocking_overrides_are_validated_when_the_config_is_used() {
    let bad = |blocking: BlockingOverride| {
        plan(&PlanConfig {
            blocking,
            ..PlanConfig::default()
        })
        .err()
        .unwrap()
    };
    // Positivity.
    for blocking in [
        BlockingOverride {
            mc: Some(0),
            ..Default::default()
        },
        BlockingOverride {
            kc: Some(0),
            ..Default::default()
        },
        BlockingOverride {
            nc_pct: Some(0),
            ..Default::default()
        },
    ] {
        assert!(
            matches!(
                bad(blocking),
                Error::Config(ConfigError::NotPositive { .. })
            ),
            "{blocking:?}"
        );
    }
    // An absolute size and a percentage for one dimension are exclusive.
    assert!(matches!(
        bad(BlockingOverride {
            mc: Some(32),
            mc_pct: Some(50),
            ..Default::default()
        }),
        Error::Config(ConfigError::BlockingExclusive { dim: "mc" })
    ));
    assert!(matches!(
        bad(BlockingOverride {
            nc: Some(32),
            nc_pct: Some(50),
            ..Default::default()
        }),
        Error::Config(ConfigError::BlockingExclusive { dim: "nc" })
    ));
    // A zero coupling or L3 domain count.
    for cache_model in [
        CacheModel {
            kc_couple: Some(0),
            ..Default::default()
        },
        CacheModel {
            l3_domains: Some(0),
            ..Default::default()
        },
    ] {
        let e = plan(&PlanConfig {
            cache_model,
            ..PlanConfig::default()
        })
        .err()
        .unwrap();
        assert!(
            matches!(e, Error::Config(ConfigError::NotPositive { .. })),
            "{e:?}"
        );
    }
}

/// An explicit complex scheme must agree with a forced family, and a family the
/// registry cannot supply is a typed selection error at planning.
#[test]
fn a_complex_method_must_agree_with_the_forced_family() {
    use tcontract::api::{DType, DotGeneral, LayoutSpec, OperandSpec, Problem};
    let spec = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
    let problem = Problem::from_dot_general(
        DType::C64,
        spec(&[8, 8], &[1, 8]),
        spec(&[8, 8], &[1, 8]),
        spec(&[8, 8], &[1, 8]),
        &DotGeneral::new(&[1], &[0], &[], &[]),
    )
    .unwrap();
    let config = PlanConfig {
        method: Some(Method::ThreeM),
        kernel: KernelChoice::Id("ref.c64.native.4x4".into()),
        ..PlanConfig::default()
    };
    let e = Plan::<num_complex::Complex64>::new(&problem, &config)
        .err()
        .unwrap();
    assert!(
        matches!(e, Error::Select(SelectError::Incompatible { .. })),
        "{e:?}"
    );
}

#[test]
fn spellings_parse_and_unknown_ones_are_rejected() {
    assert_eq!(Orient::parse("swap"), Some(Orient::Force(true)));
    assert_eq!(Orient::parse("bogus"), None);
    assert_eq!(RowBlock::parse("idx=2"), Some(RowBlock::Index(2)));
    assert_eq!(RowBlock::parse("mr=16"), Some(RowBlock::Pin(16)));
    assert_eq!(RowBlock::parse("idx=x"), None);
    assert_eq!(KernelForce::parse("AVX2"), Some(KernelForce::Avx2));
    assert_eq!(KernelForce::parse("avx3"), None);
}

#[test]
fn the_process_environment_is_not_read() {
    let packed_cfg = PlanConfig {
        partition: Some(Partition::StaticGrid {
            pin: None,
            align_c_lines: false,
        }),
        ..PlanConfig::default()
    };
    let before = (
        plan(&PlanConfig::default()).unwrap().report().clone(),
        plan(&packed_cfg).unwrap().report().clone(),
    );
    // SAFETY: the only test in this binary that touches the environment.
    unsafe {
        std::env::set_var("TENSORCONTRACT_KERNEL", "scalar");
        std::env::set_var("TENSORCONTRACT_KC", "999");
        std::env::set_var("TENSORCONTRACT_THREADS", "7");
        std::env::set_var("TENSORCONTRACT_ORIENT", "swap");
        std::env::set_var("TENSORCONTRACT_PARTITION", "9x9");
        std::env::set_var("TPRIMS_GEMM_KERNEL", "does.not.exist");
    }
    let after = (
        plan(&PlanConfig::default()).unwrap().report().clone(),
        plan(&packed_cfg).unwrap().report().clone(),
    );
    assert_eq!(before, after);
}
