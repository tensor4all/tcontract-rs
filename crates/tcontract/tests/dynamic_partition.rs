//! `Partition::DynamicTiles` through `Plan`: a plan-level option of the packed
//! driver, validated when the plan is built, honoured on a non-fusable
//! contraction, and -- since an explicit request forces the packed driver --
//! honoured on an all-batch problem too.
use tcontract::api::{
    AccumulationSource, ConfigError, DType, DotGeneral, Error, LayoutSpec, OperandSpec, Problem,
};
use tcontract::{Algorithm, Partition, Plan, PlanConfig};
use tcontract_exec::{Exec, Pool};
use tcontract_kernel::{KernelChoice, PartitionPolicy, SelectError};

mod common;
use common::{out_dims, reference, rel_err, T};

fn spec<S>(t: &T<S>) -> OperandSpec {
    OperandSpec::new(LayoutSpec::new(&t.dims, &t.strides, 0).unwrap())
}

/// Two contracted axes in permuted order: faer would have to copy.
fn copying_case() -> (DotGeneral, Vec<usize>, Vec<usize>) {
    (
        DotGeneral::new(&[1, 2], &[2, 0], &[], &[]),
        vec![24, 20, 9],
        vec![9, 30, 20],
    )
}

fn dyn_cfg(job_m: usize, job_n: usize) -> PlanConfig {
    PlanConfig {
        kernel: KernelChoice::Id("ref.f64.real.4x4".into()),
        partition: Some(Partition::DynamicTiles { job_m, job_n }),
        ..PlanConfig::default()
    }
}

fn plan(cfg: &PlanConfig) -> Result<Plan<f64>, Error> {
    let (dg, ad, bd) = copying_case();
    let (a, b) = (T::<f64>::new(&ad, 1), T::<f64>::new(&bd, 2));
    let c = T::<f64>::new(&out_dims(&dg, &a.dims, &b.dims), 3);
    Plan::<f64>::new(
        &Problem::from_dot_general(DType::F64, spec(&a), spec(&b), spec(&c), &dg).unwrap(),
        cfg,
    )
}

fn run(plan: &Plan<f64>, exec: &mut Exec<'_>) -> Vec<f64> {
    let (dg, ad, bd) = copying_case();
    let (a, b) = (T::<f64>::new(&ad, 1), T::<f64>::new(&bd, 2));
    let mut c = T::<f64>::new(&out_dims(&dg, &a.dims, &b.dims), 3);
    plan.execute_into_accum(
        exec,
        1.0,
        &a.view(),
        &b.view(),
        0.5,
        AccumulationSource::Output,
        &mut c.view_mut(),
    )
    .unwrap();
    c.data
}

#[test]
fn a_non_fusable_contraction_runs_dynamically_and_reports_the_policy() {
    let tp = strided_exec::Pool::builder().threads(4).build().unwrap();
    let pool = Pool::borrow(&tp);
    let exec = Exec::new(&pool, pool.size()).unwrap();
    let stat = plan(&PlanConfig {
        kernel: KernelChoice::Id("ref.f64.real.4x4".into()),
        ..PlanConfig::default()
    })
    .unwrap();
    let dynp = plan(&dyn_cfg(8, 8)).unwrap();
    assert_eq!(dynp.report().algorithm, Algorithm::Packed);
    let report = dynp.report().packed.as_ref().unwrap();
    assert_eq!(
        report.partition,
        PartitionPolicy::DynamicTiles { job_m: 8, job_n: 8 }
    );
    let d = report.dynamic.expect("assignment is reported");
    assert_eq!((d.job_m, d.job_n, d.row_bands), (8, 8, 3));
    assert!(stat.report().packed.as_ref().unwrap().dynamic.is_none());
    for mut e in [Exec::serial(), exec] {
        assert_eq!(run(&dynp, &mut e), run(&stat, &mut Exec::serial()));
    }
}

#[test]
fn a_partition_request_alone_selects_the_packed_driver() {
    // Without an explicit kernel, a partition policy alone is a requirement,
    // so the planner chooses the packed driver. 48 x 48 is a whole number of
    // register blocks for every built-in family (4x4, 8x6, 8x4, NEON 16x3,
    // ...), since job extents must be multiples of the selected family's
    // logical MR and NR.
    let cfg = PlanConfig {
        partition: Some(Partition::DynamicTiles {
            job_m: 48,
            job_n: 48,
        }),
        ..PlanConfig::default()
    };
    assert_eq!(plan(&cfg).unwrap().report().algorithm, Algorithm::Packed);
}

#[test]
fn invalid_policies_are_typed_errors_at_planning() {
    // A zero extent is a configuration error, caught before any problem is seen.
    for (jm, jn) in [(0, 8), (8, 0)] {
        let e = plan(&dyn_cfg(jm, jn)).expect_err("rejected");
        assert!(
            matches!(e, Error::Config(ConfigError::NotPositive { .. })),
            "{jm}x{jn}: {e:?}"
        );
    }
    // A job that is not a multiple of the family's register block is refused
    // when the family is resolved.
    for (jm, jn) in [(6, 8), (8, 6)] {
        let e = plan(&dyn_cfg(jm, jn)).expect_err("rejected");
        assert!(
            matches!(e, Error::Select(SelectError::Incompatible { .. })),
            "{jm}x{jn}: {e:?}"
        );
    }
    // A pinned grid with a zero side.
    let pinned = PlanConfig {
        partition: Some(Partition::StaticGrid {
            pin: Some((0, 2)),
            align_c_lines: false,
        }),
        ..PlanConfig::default()
    };
    assert!(matches!(
        plan(&pinned),
        Err(Error::Config(ConfigError::Option(_)))
    ));
}

/// The behaviour change of the consolidation: an explicit grid request on an
/// all-batch (Hadamard) problem used to be refused because the elementwise pass
/// has no grid; it now forces the packed driver, which honours it.
#[test]
fn an_explicit_partition_on_an_all_batch_problem_runs_packed() {
    let dg = DotGeneral::new(&[], &[], &[0, 1], &[1, 0]);
    let a = T::<f64>::new(&[48, 8], 1);
    let b = T::<f64>::new(&[8, 48], 2);
    let c0 = T::<f64>::new(&[48, 8], 3);
    let want = reference(&dg, 1.5, &a, false, &b, false, 0.5, &c0);
    for partition in [
        Partition::StaticGrid {
            pin: None,
            align_c_lines: false,
        },
        Partition::StaticGrid {
            pin: Some((2, 2)),
            align_c_lines: false,
        },
        Partition::DynamicTiles {
            job_m: 48,
            job_n: 48,
        },
    ] {
        let config = PlanConfig {
            partition: Some(partition),
            ..PlanConfig::default()
        };
        let problem =
            Problem::from_dot_general(DType::F64, spec(&a), spec(&b), spec(&c0), &dg).unwrap();
        let plan = Plan::<f64>::new(&problem, &config).unwrap();
        assert_eq!(plan.report().algorithm, Algorithm::Packed, "{partition:?}");
        let tp = strided_exec::Pool::builder().threads(4).build().unwrap();
        let pool = Pool::borrow(&tp);
        for mut exec in [Exec::serial(), Exec::new(&pool, pool.size()).unwrap()] {
            let mut c = c0.clone();
            plan.execute_into_accum(
                &mut exec,
                1.5,
                &a.view(),
                &b.view(),
                0.5,
                AccumulationSource::Output,
                &mut c.view_mut(),
            )
            .unwrap();
            assert!(rel_err(&c, &want) < 1e-13, "{partition:?}");
        }
    }
    // The planner's own choice for the same problem stays elementwise.
    let problem =
        Problem::from_dot_general(DType::F64, spec(&a), spec(&b), spec(&c0), &dg).unwrap();
    assert_eq!(
        Plan::<f64>::new(&problem, &PlanConfig::default())
            .unwrap()
            .report()
            .algorithm,
        Algorithm::Elementwise
    );
}
