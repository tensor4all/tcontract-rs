//! The neutral contraction interface, exercised through trait objects with two
//! independent implementations: the tcontract backend and the test-only naive one.

use num_complex::{Complex32, Complex64};
use tcontract::api::{
    AccumulationSource, AliasError, BoxedPlan, ContractionBackend, DotGeneral, Error, LayoutError,
    LayoutSpec, Op, OperandSpec, PlanningBudget, Problem, Requirements, Scalar, ShapeError,
};
use tcontract::{PlanConfig, TcontractBackend};
use tcontract_exec::{Exec, Pool};
use tcontract_kernel::{Element, KernelChoice, Real, SelectError};
use tcontract_testkit::NaiveBackend;

mod common;
use common::*;

fn op(c: bool) -> Op {
    if c {
        Op::Conjugate
    } else {
        Op::Identity
    }
}

fn spec<S>(t: &T<S>, o: Op) -> OperandSpec {
    OperandSpec::new(LayoutSpec::new(&t.dims, &t.strides, 0).unwrap()).with_op(o)
}

fn problem<S: Scalar>(
    cfg: &DotGeneral,
    a: &T<S>,
    b: &T<S>,
    c: &T<S>,
    conj: (bool, bool),
) -> Problem {
    Problem::from_dot_general(
        S::STORAGE,
        spec(a, op(conj.0)),
        spec(b, op(conj.1)),
        spec(c, Op::Identity),
        cfg,
    )
    .unwrap()
}

/// The planner's own choice, the packed driver forced, and the naive loop nest.
fn backends<S: Scalar>() -> Vec<Box<dyn ContractionBackend<S>>> {
    let packed = PlanConfig::packed();
    vec![
        Box::new(TcontractBackend::default()),
        Box::new(TcontractBackend { config: packed }),
        Box::new(NaiveBackend),
    ]
}

fn scalar<S: Scalar>(re: f64, im: f64) -> S {
    <S as Element>::from_parts(Real::from_f64(re), Real::from_f64(im))
}

struct Case {
    name: &'static str,
    a: Vec<usize>,
    b: Vec<usize>,
    cfg: DotGeneral,
}

fn corpus() -> Vec<Case> {
    let dg = DotGeneral::new;
    vec![
        Case {
            name: "matmul",
            a: vec![5, 4],
            b: vec![4, 3],
            cfg: dg(&[1], &[0], &[], &[]),
        },
        Case {
            name: "batched",
            a: vec![3, 5, 4],
            b: vec![3, 4, 2],
            cfg: dg(&[2], &[1], &[0], &[0]),
        },
        Case {
            name: "two contracted",
            a: vec![3, 4, 5],
            b: vec![5, 6, 4],
            cfg: dg(&[1, 2], &[2, 0], &[], &[]),
        },
        Case {
            name: "outer",
            a: vec![3, 2],
            b: vec![4],
            cfg: dg(&[], &[], &[], &[]),
        },
        Case {
            name: "scalar result",
            a: vec![3, 4],
            b: vec![4, 3],
            cfg: dg(&[0, 1], &[1, 0], &[], &[]),
        },
        Case {
            name: "hadamard",
            a: vec![3, 4],
            b: vec![4, 3],
            cfg: dg(&[], &[], &[0, 1], &[1, 0]),
        },
        Case {
            name: "singleton dims",
            a: vec![1, 4, 1],
            b: vec![4, 1, 3],
            cfg: dg(&[1], &[0], &[2], &[1]),
        },
        Case {
            name: "batch middle",
            a: vec![4, 3, 5],
            b: vec![5, 3, 2],
            cfg: dg(&[2], &[0], &[1], &[1]),
        },
        Case {
            name: "empty free",
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

/// Column-major, permuted-storage and negative-stride variants of one tensor.
fn layouts<S: Scalar>(t: &T<S>) -> Vec<T<S>> {
    let rev: Vec<usize> = (0..t.dims.len()).rev().collect();
    vec![t.clone(), t.restride(&rev), t.reversed()]
}

fn check_all<S: Scalar>(tol: f64) {
    let scalars = [
        (scalar::<S>(1.0, 0.0), scalar::<S>(0.0, 0.0)),
        (scalar::<S>(0.5, 0.25), scalar::<S>(-0.75, 0.5)),
    ];
    let conjs = [(false, false), (true, false), (true, true)];
    for case in corpus() {
        let out = out_dims(&case.cfg, &case.a, &case.b);
        let a0 = T::<S>::new(&case.a, 1);
        let b0 = T::<S>::new(&case.b, 2);
        let c0 = T::<S>::new(&out, 3);
        for (li, (la, lb)) in layouts(&a0).into_iter().zip(layouts(&b0)).enumerate() {
            let lc = layouts(&c0).swap_remove(li);
            for &(alpha, beta) in &scalars {
                for &(ca, cb) in &conjs {
                    let want = reference(&case.cfg, alpha, &la, ca, &lb, cb, beta, &lc);
                    let p = problem(&case.cfg, &la, &lb, &lc, (ca, cb));
                    for be in backends::<S>() {
                        let plan = be
                            .prepare(&p, &Requirements::new(), &PlanningBudget::serial())
                            .unwrap_or_else(|e| panic!("{} {}: {e}", case.name, be.id()));
                        let mut got = lc.clone();
                        plan.execute_into_accum(
                            &mut Exec::serial(),
                            alpha,
                            &la.view(),
                            &lb.view(),
                            beta,
                            AccumulationSource::Output,
                            &mut got.view_mut(),
                        )
                        .unwrap();
                        let err = rel_err(&got, &want);
                        assert!(err < tol, "{} {} layout {li}: {err}", case.name, be.id());
                    }
                }
            }
        }
    }
}

#[test]
fn all_backends_match_the_reference_f64() {
    check_all::<f64>(1e-12);
}
#[test]
fn all_backends_match_the_reference_f32() {
    check_all::<f32>(2e-5);
}
#[test]
fn all_backends_match_the_reference_c64() {
    check_all::<Complex64>(1e-12);
}
#[test]
fn all_backends_match_the_reference_c32() {
    check_all::<Complex32>(2e-5);
}

fn matmul_problem(m: usize, k: usize, n: usize) -> (DotGeneral, T<f64>, T<f64>, T<f64>) {
    (
        DotGeneral::new(&[1], &[0], &[], &[]),
        T::new(&[m, k], 11),
        T::new(&[k, n], 12),
        T::new(&[m, n], 13),
    )
}

fn accum(
    plan: &BoxedPlan<f64>,
    exec: &mut Exec<'_>,
    alpha: f64,
    a: &T<f64>,
    b: &T<f64>,
    beta: f64,
    out: &mut T<f64>,
) -> Result<(), Error> {
    plan.execute_into_accum(
        exec,
        alpha,
        &a.view(),
        &b.view(),
        beta,
        AccumulationSource::Output,
        &mut out.view_mut(),
    )
}

#[test]
fn beta_zero_reads_no_output_and_zero_alpha_or_empty_k_reads_no_inputs() {
    let (cfg, a, b, c) = matmul_problem(4, 3, 5);
    let p = problem(&cfg, &a, &b, &c, (false, false));
    let nan = |t: &T<f64>| T {
        data: vec![f64::NAN; t.data.len()],
        ..t.clone()
    };
    for be in backends::<f64>() {
        let plan = be
            .prepare(&p, &Requirements::new(), &PlanningBudget::serial())
            .unwrap();
        let mut exec = Exec::serial();
        // beta == 0: previous C values (NaN) must not leak.
        let mut out = nan(&c);
        accum(&plan, &mut exec, 1.0, &a, &b, 0.0, &mut out).unwrap();
        let want = reference(&cfg, 1.0, &a, false, &b, false, 0.0, &c);
        assert!(rel_err(&out, &want) < 1e-12, "{}", be.id());
        // The overwrite form reads no previous output either.
        let mut out = nan(&c);
        plan.execute_into(&mut exec, 1.0, &a.view(), &b.view(), &mut out.view_mut())
            .unwrap();
        assert!(rel_err(&out, &want) < 1e-12, "{} overwrite", be.id());
        // alpha == 0: A and B (NaN) must not be read.
        let mut out = c.clone();
        accum(&plan, &mut exec, 0.0, &nan(&a), &nan(&b), 2.0, &mut out).unwrap();
        for_each_index(&c.dims, |i| {
            assert_eq!(out.get(i), 2.0 * c.get(i), "{}", be.id())
        });
    }
    // Empty contraction: C = beta * C, inputs untouched.
    let (cfg, a, b, c) = matmul_problem(3, 0, 2);
    let p = problem(&cfg, &a, &b, &c, (false, false));
    for be in backends::<f64>() {
        let plan = be
            .prepare(&p, &Requirements::new(), &PlanningBudget::serial())
            .unwrap();
        let mut out = c.clone();
        accum(&plan, &mut Exec::serial(), 1.0, &a, &b, 3.0, &mut out).unwrap();
        for_each_index(&c.dims, |i| {
            assert_eq!(out.get(i), 3.0 * c.get(i), "{}", be.id())
        });
    }
}

/// Validation happens when the problem is described, once, before any backend
/// or any no-op shortcut sees it.
#[test]
fn invalid_metadata_is_rejected_before_the_no_op_shortcuts() {
    // Empty contraction plus an aliased output: refused, not shortcut.
    let cfg = DotGeneral::new(&[1], &[0], &[], &[]);
    let s = |d: &[usize], st: &[isize]| OperandSpec::new(LayoutSpec::new(d, st, 0).unwrap());
    let e = Problem::from_dot_general(
        tcontract::api::DType::F64,
        s(&[3, 0], &[1, 3]),
        s(&[0, 2], &[1, 1]),
        s(&[3, 2], &[1, 1]),
        &cfg,
    )
    .unwrap_err();
    assert!(
        matches!(e, Error::Alias(AliasError::OutputNotInjective)),
        "{e:?}"
    );
    // Empty output with mismatching extents of B.
    let e = Problem::from_dot_general(
        tcontract::api::DType::F64,
        s(&[0, 3], &[1, 1]),
        s(&[4, 2], &[1, 4]),
        s(&[0, 2], &[1, 1]),
        &cfg,
    )
    .unwrap_err();
    assert!(
        matches!(e, Error::Shape(ShapeError::PairedExtent { .. })),
        "{e:?}"
    );
}

#[test]
fn plans_outlive_problem_and_backend_and_reject_other_layouts_before_writing() {
    let (cfg, a, b, c) = matmul_problem(4, 3, 5);
    let plans: Vec<_> = {
        let p = problem(&cfg, &a, &b, &c, (false, false));
        let bes = backends::<f64>();
        bes.iter()
            .map(|be| {
                be.prepare(&p, &Requirements::new(), &PlanningBudget::serial())
                    .unwrap()
            })
            .collect()
        // problem and backends dropped here
    };
    for plan in &plans {
        // Different buffers with the planned layouts, run repeatedly.
        for seed in [1u64, 2, 3] {
            let a2 = T::<f64>::new(&a.dims, seed);
            let b2 = T::<f64>::new(&b.dims, seed + 10);
            let mut out = c.clone();
            accum(plan, &mut Exec::serial(), 1.0, &a2, &b2, 0.5, &mut out).unwrap();
            let want = reference(&cfg, 1.0, &a2, false, &b2, false, 0.5, &c);
            assert!(
                rel_err(&out, &want) < 1e-12,
                "{}",
                plan.diagnostics().backend
            );
        }
        // A transposed C layout is another problem: rejected, C untouched.
        let wrong = c.restride(&[1, 0]);
        let mut out = wrong.clone();
        let e = accum(plan, &mut Exec::serial(), 1.0, &a, &b, 0.0, &mut out).unwrap_err();
        assert!(
            matches!(e, Error::Layout(LayoutError::Mismatch { .. })),
            "{e}"
        );
        assert_eq!(out.data, wrong.data);
        // An accumulation source that does not match the planned C mode.
        let mut out = c.clone();
        let e = plan
            .execute_into_accum(
                &mut Exec::serial(),
                1.0,
                &a.view(),
                &b.view(),
                0.5,
                AccumulationSource::Separate(&c.view()),
                &mut out.view_mut(),
            )
            .unwrap_err();
        assert!(matches!(e, Error::Layout(LayoutError::CMode)), "{e}");
    }
}

/// No strategy copies a whole operand, so the shared flag is always met, and
/// every backend reports that it materializes nothing.
#[test]
fn materialization_is_reported_and_the_shared_flag_is_met() {
    // Contracted axes in different orders on A and B cannot both fuse.
    let cfg = DotGeneral::new(&[1, 2], &[2, 0], &[], &[]);
    let a = T::<f64>::new(&[3, 4, 5], 1);
    let b = T::<f64>::new(&[5, 6, 4], 2);
    let c = T::<f64>::new(&[3, 6], 3);
    let p = problem(&cfg, &a, &b, &c, (false, false));
    for be in backends::<f64>() {
        let plan = be
            .prepare(
                &p,
                &Requirements::new().no_materialize(true),
                &PlanningBudget::serial(),
            )
            .unwrap();
        assert_eq!(plan.diagnostics().materialized, [false; 3], "{}", be.id());
    }
    let plan = ContractionBackend::<f64>::prepare(
        &TcontractBackend::default(),
        &p,
        &Requirements::new(),
        &PlanningBudget::serial(),
    )
    .unwrap();
    assert_eq!(plan.diagnostics().algorithm, "packed");
    assert_eq!(plan.diagnostics().backend, "tcontract");
}

#[test]
fn a_forced_unusable_kernel_fails_at_preparation_without_output() {
    let (cfg, a, b, c) = matmul_problem(4, 3, 5);
    let p = problem(&cfg, &a, &b, &c, (false, false));
    let be = TcontractBackend {
        config: PlanConfig {
            kernel: KernelChoice::Id("not.a.kernel".into()),
            ..PlanConfig::default()
        },
    };
    let e = ContractionBackend::<f64>::prepare(
        &be,
        &p,
        &Requirements::new(),
        &PlanningBudget::serial(),
    )
    .err()
    .unwrap();
    assert!(
        matches!(e, Error::Select(SelectError::UnknownId { .. })),
        "{e}"
    );
    // The factory is not mutated by a requirement: no_materialize ORs in per call.
    assert!(!be.config.no_materialize);
    let _ = ContractionBackend::<f64>::prepare(
        &TcontractBackend::default(),
        &p,
        &Requirements::new().no_materialize(true),
        &PlanningBudget::serial(),
    )
    .unwrap();
}

#[test]
fn a_plan_for_another_dtype_is_a_configuration_error() {
    let (cfg, a, b, c) = matmul_problem(4, 3, 5);
    let p = problem(&cfg, &a, &b, &c, (false, false));
    for e in [
        ContractionBackend::<f32>::prepare(
            &TcontractBackend::default(),
            &p,
            &Requirements::new(),
            &PlanningBudget::serial(),
        )
        .err()
        .unwrap(),
        ContractionBackend::<f32>::prepare(
            &NaiveBackend,
            &p,
            &Requirements::new(),
            &PlanningBudget::serial(),
        )
        .err()
        .unwrap(),
    ] {
        assert!(
            matches!(
                e,
                Error::Config(tcontract::api::ConfigError::DtypeMismatch { .. })
            ),
            "{e}"
        );
    }
}

#[test]
fn one_thread_exec_enters_no_pool_and_a_four_thread_exec_stays_in_budget() {
    let (cfg, a, b, c) = matmul_problem(96, 80, 72);
    let p = problem(&cfg, &a, &b, &c, (false, false));
    let want = reference(&cfg, 1.0, &a, false, &b, false, 0.5, &c);
    let tp = threadteam::Pool::builder().threads(4).build().unwrap();
    let pool = Pool::borrow(&tp);
    for be in backends::<f64>() {
        let plan = be
            .prepare(&p, &Requirements::new(), &PlanningBudget::new(4))
            .unwrap();
        // 1T on a pool-backed context: no pool entry.
        let mut exec = Exec::new(&pool, 1).unwrap();
        let one_t = exec.token().ledger_root().unwrap();
        let mut out = c.clone();
        accum(&plan, &mut exec, 1.0, &a, &b, 0.5, &mut out).unwrap();
        assert!(rel_err(&out, &want) < 1e-12, "{}", be.id());
        assert_eq!(threadteam::ledger::installs(one_t), 0, "{}", be.id());
        // The same plan at 4T.
        let mut exec = Exec::new(&pool, pool.size()).unwrap();
        assert_eq!(exec.budget(), 4);
        let four_t = exec.token().ledger_root().unwrap();
        let mut out = c.clone();
        accum(&plan, &mut exec, 1.0, &a, &b, 0.5, &mut out).unwrap();
        assert!(rel_err(&out, &want) < 1e-12, "{}", be.id());
        assert!(threadteam::ledger::max_live(four_t) <= 4, "{}", be.id());
        // Nested entry from inside the pool: a plan that needs a barrier-bearing
        // team takes its barrier-free route, so every backend completes and
        // agrees instead of deadlocking or running serially.
        let mut out = c.clone();
        exec.reborrow()
            .into_token()
            .external(|_| {
                let mut inner = Exec::new(&pool, 4).unwrap();
                accum(&plan, &mut inner, 1.0, &a, &b, 0.5, &mut out)
            })
            .unwrap();
        assert!(rel_err(&out, &want) < 1e-12, "{} nested", be.id());
    }
}

#[test]
fn a_shared_plan_runs_concurrently_on_independent_outputs() {
    let (cfg, a, b, c) = matmul_problem(64, 48, 40);
    let p = problem(&cfg, &a, &b, &c, (false, false));
    let want = reference(&cfg, 1.0, &a, false, &b, false, 0.0, &c);
    let tp = threadteam::Pool::builder().threads(4).build().unwrap();
    let pool = Pool::borrow(&tp);
    for be in backends::<f64>() {
        let plan = be
            .prepare(&p, &Requirements::new(), &PlanningBudget::new(4))
            .unwrap();
        std::thread::scope(|s| {
            for t in 0..4 {
                let (plan, a, b, c, want, pool) = (&plan, &a, &b, &c, &want, &pool);
                s.spawn(move || {
                    // Half the callers use the shared pool, half run serially.
                    let mut exec = if t % 2 == 0 {
                        Exec::new(pool, pool.size()).unwrap()
                    } else {
                        Exec::serial()
                    };
                    for _ in 0..3 {
                        let mut out = c.clone();
                        accum(plan, &mut exec, 1.0, a, b, 0.0, &mut out).unwrap();
                        assert!(rel_err(&out, want) < 1e-12);
                    }
                });
            }
        });
    }
}
