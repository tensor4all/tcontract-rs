use num_complex::Complex64;
use tcontract::api::{
    AccumulationSource, AliasError, ConfigError, DType, DotGeneral, Error, LayoutError, LayoutSpec,
    Op, OperandSpec, Problem, Scalar, ShapeError,
};
use tcontract::{Algorithm, Plan, PlanConfig};
use tcontract_exec::{Exec, Pool};
use tcontract_kernel::Element;

mod common;
use common::*;

/// The configurations every case runs under: the planner's own choice, and
/// the packed driver forced by an explicit grid request.
fn configs() -> [PlanConfig; 2] {
    let packed = PlanConfig::packed();
    [PlanConfig::default(), packed]
}

fn spec<S>(t: &T<S>, op: Op) -> OperandSpec {
    OperandSpec::new(LayoutSpec::new(&t.dims, &t.strides, 0).unwrap()).with_op(op)
}

fn problem<S: Scalar>(
    cfg: &DotGeneral,
    a: &T<S>,
    ca: bool,
    b: &T<S>,
    cb: bool,
    c: &T<S>,
) -> Problem {
    let op = |x: bool| if x { Op::Conjugate } else { Op::Identity };
    Problem::from_dot_general(
        S::STORAGE,
        spec(a, op(ca)),
        spec(b, op(cb)),
        spec(c, Op::Identity),
        cfg,
    )
    .unwrap()
}

struct Case {
    name: &'static str,
    a: Vec<usize>,
    b: Vec<usize>,
    cfg: DotGeneral,
}

fn corpus() -> Vec<Case> {
    let dg =
        |lc: &[usize], rc: &[usize], lb: &[usize], rb: &[usize]| DotGeneral::new(lc, rc, lb, rb);
    vec![
        Case {
            name: "matmul ij,jk",
            a: vec![5, 4],
            b: vec![4, 3],
            cfg: dg(&[1], &[0], &[], &[]),
        },
        Case {
            name: "batched bij,bjk",
            a: vec![3, 5, 4],
            b: vec![3, 4, 2],
            cfg: dg(&[2], &[1], &[0], &[0]),
        },
        Case {
            name: "two contracted axes in different orders",
            a: vec![3, 4, 5],
            b: vec![5, 6, 4],
            cfg: dg(&[1, 2], &[2, 0], &[], &[]),
        },
        Case {
            name: "outer product",
            a: vec![3, 2],
            b: vec![4],
            cfg: dg(&[], &[], &[], &[]),
        },
        Case {
            name: "full contraction to rank 0",
            a: vec![3, 4],
            b: vec![4, 3],
            cfg: dg(&[0, 1], &[1, 0], &[], &[]),
        },
        Case {
            name: "hadamard batch only",
            a: vec![3, 4],
            b: vec![4, 3],
            cfg: dg(&[], &[], &[0, 1], &[1, 0]),
        },
        Case {
            name: "4-index network ijkl,klmn",
            a: vec![2, 3, 4, 2],
            b: vec![4, 2, 3, 2],
            cfg: dg(&[2, 3], &[0, 1], &[], &[]),
        },
        Case {
            name: "batch in the middle",
            a: vec![4, 3, 5],
            b: vec![5, 3, 2],
            cfg: dg(&[2], &[0], &[1], &[1]),
        },
        Case {
            name: "empty free extent",
            a: vec![0, 3],
            b: vec![3, 2],
            cfg: dg(&[1], &[0], &[], &[]),
        },
        Case {
            name: "empty contraction",
            a: vec![3, 0],
            b: vec![0, 2],
            cfg: dg(&[1], &[0], &[], &[]),
        },
    ]
}

fn out_dims(cfg: &DotGeneral, a: &[usize], b: &[usize]) -> Vec<usize> {
    cfg.validate(a, b).unwrap().out_dims.to_vec()
}

#[allow(clippy::too_many_arguments)]
fn run<S: Scalar>(
    exec: &Exec<'_>,
    config: &PlanConfig,
    case: &Case,
    layout: u8,
    ca: bool,
    cb: bool,
    beta: S,
) -> Algorithm {
    let base_a = T::<S>::new(&case.a, 1);
    let base_b = T::<S>::new(&case.b, 2);
    // layout 0: column-major, 1: axes reversed in storage order, 2: negative strides
    let rev = |d: usize| (0..d).rev().collect::<Vec<_>>();
    let (a, b) = match layout {
        0 => (base_a, base_b),
        1 => (
            base_a.restride(&rev(case.a.len())),
            base_b.restride(&rev(case.b.len())),
        ),
        _ => (base_a.reversed(), base_b.reversed()),
    };
    let od = out_dims(&case.cfg, &case.a, &case.b);
    let c0 = if layout == 2 {
        T::<S>::new(&od, 3).reversed()
    } else {
        T::<S>::new(&od, 3)
    };
    let alpha = <S as Element>::from_parts(
        tcontract_kernel::Real::from_f64(0.75),
        tcontract_kernel::Real::from_f64(0.25),
    );
    let want = reference(&case.cfg, alpha, &a, ca, &b, cb, beta, &c0);
    let mut c = c0.clone();
    let plan = Plan::<S>::new(&problem(&case.cfg, &a, ca, &b, cb, &c), config).unwrap();
    plan.execute_into_accum(
        exec,
        alpha,
        &a.view(),
        &b.view(),
        beta,
        AccumulationSource::Output,
        &mut c.view_mut(),
    )
    .unwrap();
    let tol = if std::mem::size_of::<<S as Scalar>::Re>() == 4 {
        1e-4
    } else {
        1e-12
    };
    let err = rel_err(&c, &want);
    assert!(
        err < tol,
        "{:?} {} layout{layout} {ca:?}/{cb:?}: {err}",
        plan.report().algorithm,
        case.name
    );
    plan.report().algorithm
}

#[test]
fn corpus_matches_the_reference() {
    for config in configs() {
        for case in corpus() {
            for layout in 0..3u8 {
                run::<f64>(&Exec::serial(), &config, &case, layout, false, false, 1.5);
                run::<Complex64>(
                    &Exec::serial(),
                    &config,
                    &case,
                    layout,
                    true,
                    false,
                    Complex64::new(0.0, 0.0),
                );
                run::<Complex64>(
                    &Exec::serial(),
                    &config,
                    &case,
                    layout,
                    false,
                    true,
                    Complex64::new(1.0, -1.0),
                );
            }
        }
    }
}

#[test]
fn larger_contractions_on_a_pool_match() {
    let tp = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let pool = Pool::borrow(&tp);
    let exec = Exec::rayon(&pool);
    let big = Case {
        name: "big batched",
        a: vec![8, 40, 30],
        b: vec![8, 30, 50],
        cfg: DotGeneral::new(&[2], &[1], &[0], &[0]),
    };
    let net = Case {
        name: "big network",
        a: vec![6, 7, 8, 9],
        b: vec![8, 9, 5, 4],
        cfg: DotGeneral::new(&[2, 3], &[0, 1], &[], &[]),
    };
    for config in configs() {
        for layout in 0..3u8 {
            run::<f64>(&exec, &config, &big, layout, false, false, 0.5);
            run::<f64>(&exec, &config, &net, layout, false, false, 0.0);
        }
    }
}

/// The strategy rules, in order: an explicit request forces the packed driver
/// (even for an all-batch problem); otherwise all-batch is elementwise; a
/// problem that fuses copy-free runs on faer; the rest runs packed. No
/// strategy copies a whole operand, so `no_materialize` is always met.
#[test]
fn strategy_selection_follows_the_rules() {
    let algo = |case: &Case, config: &PlanConfig| {
        let a = T::<f64>::new(&case.a, 1);
        let b = T::<f64>::new(&case.b, 2);
        let c = T::<f64>::new(&out_dims(&case.cfg, &case.a, &case.b), 3);
        Plan::<f64>::new(&problem(&case.cfg, &a, false, &b, false, &c), config)
            .unwrap()
            .report()
            .algorithm
    };
    let default = PlanConfig::default();
    let [_, packed] = configs();
    let corpus = corpus();
    // 3. column-major matmul fuses without a copy.
    assert_eq!(algo(&corpus[0], &default), Algorithm::Faer);
    // 4. contracted axes in different orders on A and B cannot both fuse.
    assert_eq!(algo(&corpus[2], &default), Algorithm::Packed);
    // 2. a Hadamard product is elementwise.
    assert_eq!(algo(&corpus[5], &default), Algorithm::Elementwise);
    // 1. an explicit request wins, all-batch included.
    assert_eq!(algo(&corpus[0], &packed), Algorithm::Packed);
    assert_eq!(algo(&corpus[5], &packed), Algorithm::Packed);
    // A kernel id or a complex method is such a request too.
    let id = PlanConfig {
        kernel: tcontract_kernel::KernelChoice::Id("ref.f64.real.4x4".into()),
        ..PlanConfig::default()
    };
    assert_eq!(algo(&corpus[0], &id), Algorithm::Packed);
    assert_eq!(algo(&corpus[5], &id), Algorithm::Packed);
    // Nothing is copied, so a strict requirement is met everywhere.
    let strict = PlanConfig {
        no_materialize: true,
        ..PlanConfig::default()
    };
    for case in &corpus {
        let a = T::<f64>::new(&case.a, 1);
        let b = T::<f64>::new(&case.b, 2);
        let c = T::<f64>::new(&out_dims(&case.cfg, &case.a, &case.b), 3);
        let plan =
            Plan::<f64>::new(&problem(&case.cfg, &a, false, &b, false, &c), &strict).unwrap();
        assert_eq!(plan.report().materialized, [false; 3], "{}", case.name);
    }
}

#[test]
fn validation_errors_are_typed_and_nothing_is_written() {
    let a = T::<f64>::new(&[3, 4], 1);
    let b = T::<f64>::new(&[4, 2], 2);
    let cfg = DotGeneral::new(&[1], &[0], &[], &[]);
    assert!(matches!(
        DotGeneral::new(&[1, 1], &[0, 0], &[], &[]).validate(&[3, 4], &[4, 2]),
        Err(Error::Config(ConfigError::AxisRepeated { .. }))
    ));
    assert!(matches!(
        DotGeneral::new(&[2], &[0], &[], &[]).validate(&[3, 4], &[4, 2]),
        Err(Error::Config(ConfigError::AxisOutOfRange { .. }))
    ));
    assert!(matches!(
        DotGeneral::new(&[0], &[0], &[], &[]).validate(&[3, 4], &[4, 2]),
        Err(Error::Shape(ShapeError::PairedExtent { .. }))
    ));
    for config in configs() {
        // A broadcast (stride-zero) output addresses an element more than once.
        let c = T::<f64> {
            data: vec![7.0; 3],
            dims: vec![3, 2],
            strides: vec![1, 0],
            offset: 0,
        };
        let e = Problem::from_dot_general(
            DType::F64,
            spec(&a, Op::Identity),
            spec(&b, Op::Identity),
            spec(&c, Op::Identity),
            &cfg,
        );
        assert!(matches!(
            e,
            Err(Error::Alias(AliasError::OutputNotInjective))
        ));
        // A view that differs from the planned layout is refused before any write.
        let planned = T::<f64>::new(&[3, 2], 3);
        let plan =
            Plan::<f64>::new(&problem(&cfg, &a, false, &b, false, &planned), &config).unwrap();
        let mut c = c;
        let e = plan.execute_into(
            &Exec::serial(),
            1.0,
            &a.view(),
            &b.view(),
            &mut c.view_mut(),
        );
        assert!(
            matches!(e, Err(Error::Layout(LayoutError::Mismatch { .. }))),
            "{e:?}"
        );
        assert!(c.data.iter().all(|&x| x == 7.0));
    }
}

/// `execute_into` is the overwrite form: no previous output value is read, so
/// NaN garbage in the output is replaced, not propagated.
#[test]
fn overwrite_reads_no_previous_output() {
    for config in configs() {
        for case in corpus() {
            let a = T::<f64>::new(&case.a, 1);
            let b = T::<f64>::new(&case.b, 2);
            let od = out_dims(&case.cfg, &case.a, &case.b);
            let mut c = T::<f64>::new(&od, 3);
            c.data.iter_mut().for_each(|x| *x = f64::NAN);
            let want = reference(&case.cfg, 2.0, &a, false, &b, false, 0.0, &c);
            let plan =
                Plan::<f64>::new(&problem(&case.cfg, &a, false, &b, false, &c), &config).unwrap();
            plan.execute_into(
                &Exec::serial(),
                2.0,
                &a.view(),
                &b.view(),
                &mut c.view_mut(),
            )
            .unwrap();
            assert!(rel_err(&c, &want) < 1e-12, "{}", case.name);
        }
    }
}
