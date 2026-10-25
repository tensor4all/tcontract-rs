//! The output update `D = op_D(alpha * op_A(A) * op_B(B) + beta * op_C(C))` in
//! every C mode, under every strategy, against the independent label oracle.
//!
//! The modes: overwrite (`beta` fixed to zero, no previous value read),
//! accumulation in D (`C` is `D`, with `op_C` applied to the old `D`), and a
//! separately described C with its own layout. `op_D` applies to the whole
//! result. `alpha == 0` and an empty contraction compute `op_D(beta * op_C(C))`
//! and read no input, for every strategy.

use num_complex::Complex64 as C64;
use strided_view::{StridedView, StridedViewMut};
use tcontract::api::{
    AccumulationSource, CSpec, DType, Labels, LayoutSpec, Op, OperandSpec, Problem,
};
use tcontract::{Algorithm, Plan, PlanConfig};
use tcontract_exec::Exec;
use tcontract_testkit::fixtures::{col_major, row_major, seeded, storage_len};
use tcontract_testkit::oracle::{contract_reference, RefOperand, RefOutput};

/// A contraction shape: extents and label lists of A, B and D.
struct Shape {
    name: &'static str,
    a: Vec<usize>,
    b: Vec<usize>,
    d: Vec<usize>,
    la: Vec<i64>,
    lb: Vec<i64>,
    ld: Vec<i64>,
    /// The algorithm the default configuration is expected to choose.
    default: Algorithm,
}

fn shapes() -> Vec<Shape> {
    vec![
        Shape {
            name: "matmul",
            a: vec![7, 5],
            b: vec![5, 6],
            d: vec![7, 6],
            la: vec![0, 2],
            lb: vec![2, 1],
            ld: vec![0, 1],
            default: Algorithm::Faer,
        },
        Shape {
            name: "hadamard",
            a: vec![4, 3],
            b: vec![3, 4],
            d: vec![4, 3],
            la: vec![0, 1],
            lb: vec![1, 0],
            ld: vec![0, 1],
            default: Algorithm::Elementwise,
        },
        Shape {
            name: "two contracted, permuted",
            a: vec![3, 4, 5],
            b: vec![5, 6, 4],
            d: vec![3, 6],
            la: vec![0, 1, 2],
            lb: vec![2, 3, 1],
            ld: vec![0, 3],
            default: Algorithm::Packed,
        },
        Shape {
            name: "batched with a reduction",
            a: vec![2, 4, 5, 3],
            b: vec![2, 5, 6],
            d: vec![2, 4, 6],
            la: vec![9, 0, 2, 7],
            lb: vec![9, 2, 1],
            ld: vec![9, 0, 1],
            default: Algorithm::Packed,
        },
    ]
}

fn configs() -> [PlanConfig; 2] {
    let packed = PlanConfig::packed();
    [PlanConfig::default(), packed]
}

fn op(c: bool) -> Op {
    if c {
        Op::Conjugate
    } else {
        Op::Identity
    }
}

fn spec(dims: &[usize], strides: &[isize], conj: bool) -> OperandSpec {
    OperandSpec::new(LayoutSpec::new(dims, strides, 0).unwrap()).with_op(op(conj))
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    Overwrite,
    Output,
    Separate,
}

#[derive(Clone, Copy, Debug)]
struct Flags {
    conj_a: bool,
    conj_b: bool,
    conj_c: bool,
    conj_d: bool,
}

const NONE: Flags = Flags {
    conj_a: false,
    conj_b: false,
    conj_c: false,
    conj_d: false,
};
const ALL: Flags = Flags {
    conj_a: true,
    conj_b: false,
    conj_c: true,
    conj_d: true,
};

/// Run one case and compare with the oracle. Returns the plan's algorithm.
fn check(
    shape: &Shape,
    config: &PlanConfig,
    mode: Mode,
    flags: Flags,
    alpha: C64,
    beta: C64,
    nan_inputs: bool,
) -> Algorithm {
    // A and B are column-major; D is row-major; a separate C is column-major,
    // so it differs from D in layout.
    let (sa, sb) = (col_major(&shape.a), col_major(&shape.b));
    let sd = row_major(&shape.d);
    let sc = col_major(&shape.d);
    let a: Vec<C64> = if nan_inputs {
        vec![C64::new(f64::NAN, f64::NAN); storage_len(&shape.a, &sa)]
    } else {
        seeded(1, storage_len(&shape.a, &sa))
    };
    let b: Vec<C64> = if nan_inputs {
        vec![C64::new(f64::NAN, f64::NAN); storage_len(&shape.b, &sb)]
    } else {
        seeded(2, storage_len(&shape.b, &sb))
    };
    let start: Vec<C64> = seeded(3, storage_len(&shape.d, &sd));
    let c_sep: Vec<C64> = seeded(4, storage_len(&shape.d, &sc));

    let c_spec = match mode {
        Mode::Overwrite => CSpec::Absent,
        Mode::Output => CSpec::Output(op(flags.conj_c)),
        Mode::Separate => CSpec::Separate(spec(&shape.d, &sc, flags.conj_c)),
    };
    let mut labels = Labels::new(&shape.la, &shape.lb, &shape.ld);
    if matches!(mode, Mode::Separate) {
        labels = labels.with_c(&shape.ld);
    }
    let problem = Problem::from_labels(
        DType::C64,
        spec(&shape.a, &sa, flags.conj_a),
        spec(&shape.b, &sb, flags.conj_b),
        c_spec,
        spec(&shape.d, &sd, flags.conj_d),
        &labels,
    )
    .unwrap();
    let plan = Plan::<C64>::new(&problem, config).unwrap();

    // The oracle, from the original description.
    let mut want = start.clone();
    let c_ref_data: &[C64] = match mode {
        Mode::Separate => &c_sep,
        _ => &start,
    };
    let c_ref_strides = match mode {
        Mode::Separate => &sc,
        _ => &sd,
    };
    let beta_eff = if matches!(mode, Mode::Overwrite) {
        C64::new(0.0, 0.0)
    } else {
        beta
    };
    let c_ref = RefOperand {
        data: c_ref_data,
        dims: &shape.d,
        strides: c_ref_strides,
        offset: 0,
        labels: &shape.ld,
        conj: flags.conj_c,
    };
    contract_reference(
        alpha,
        &RefOperand {
            data: &a,
            dims: &shape.a,
            strides: &sa,
            offset: 0,
            labels: &shape.la,
            conj: flags.conj_a,
        },
        &RefOperand {
            data: &b,
            dims: &shape.b,
            strides: &sb,
            offset: 0,
            labels: &shape.lb,
            conj: flags.conj_b,
        },
        beta_eff,
        (!matches!(mode, Mode::Overwrite)).then_some(&c_ref),
        &mut RefOutput {
            data: &mut want,
            dims: &shape.d,
            strides: &sd,
            offset: 0,
            labels: &shape.ld,
            conj: flags.conj_d,
        },
    )
    .unwrap();

    let mut got = start.clone();
    let av = StridedView::new(&a, &shape.a, &sa, 0).unwrap();
    let bv = StridedView::new(&b, &shape.b, &sb, 0).unwrap();
    let cv = StridedView::new(&c_sep, &shape.d, &sc, 0).unwrap();
    {
        let mut dv = StridedViewMut::new(&mut got, &shape.d, &sd, 0).unwrap();
        match mode {
            Mode::Overwrite => plan
                .execute_into(&mut Exec::serial(), alpha, &av, &bv, &mut dv)
                .unwrap(),
            Mode::Output => plan
                .execute_into_accum(
                    &mut Exec::serial(),
                    alpha,
                    &av,
                    &bv,
                    beta,
                    AccumulationSource::Output,
                    &mut dv,
                )
                .unwrap(),
            Mode::Separate => plan
                .execute_into_accum(
                    &mut Exec::serial(),
                    alpha,
                    &av,
                    &bv,
                    beta,
                    AccumulationSource::Separate(&cv),
                    &mut dv,
                )
                .unwrap(),
        }
    }
    let err = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (g - w).norm())
        .fold(0.0, f64::max);
    assert!(
        err < 1e-12 * 50.0 && got.iter().all(|z| z.is_finite()),
        "{} {mode:?} {flags:?} alpha {alpha} beta {beta}: err {err} ({:?})",
        shape.name,
        plan.report().algorithm
    );
    plan.report().algorithm
}

#[test]
fn every_c_mode_and_conjugation_matches_the_oracle_under_every_strategy() {
    let (alpha, beta) = (C64::new(0.7, -0.3), C64::new(-0.4, 0.6));
    for shape in shapes() {
        for config in configs() {
            for mode in [Mode::Overwrite, Mode::Output, Mode::Separate] {
                for flags in [
                    NONE,
                    ALL,
                    Flags {
                        conj_a: false,
                        conj_b: true,
                        conj_c: false,
                        conj_d: true,
                    },
                ] {
                    check(&shape, &config, mode, flags, alpha, beta, false);
                }
            }
        }
    }
}

/// The default configuration chooses the documented strategy for each shape in
/// every C mode (a separately described C included: faer writes it into D in
/// one output-sized pass and accumulates the product), and every shape gets the
/// same answer.
#[test]
fn the_planner_chooses_the_same_strategy_in_every_c_mode() {
    let (alpha, beta) = (C64::new(0.7, -0.3), C64::new(-0.4, 0.6));
    for shape in shapes() {
        let overwrite = check(
            &shape,
            &PlanConfig::default(),
            Mode::Overwrite,
            ALL,
            alpha,
            beta,
            false,
        );
        let output = check(
            &shape,
            &PlanConfig::default(),
            Mode::Output,
            ALL,
            alpha,
            beta,
            false,
        );
        let separate = check(
            &shape,
            &PlanConfig::default(),
            Mode::Separate,
            ALL,
            alpha,
            beta,
            false,
        );
        assert_eq!(overwrite, shape.default, "{} overwrite", shape.name);
        assert_eq!(output, shape.default, "{} output", shape.name);
        assert_eq!(separate, shape.default, "{} separate", shape.name);
    }
}

/// `alpha == 0` computes `op_D(beta * op_C(C))` and reads neither input (NaN
/// inputs must not leak), in every mode and strategy.
#[test]
fn zero_alpha_updates_the_output_without_reading_the_inputs() {
    let beta = C64::new(-0.4, 0.6);
    for shape in shapes() {
        for config in configs() {
            for mode in [Mode::Overwrite, Mode::Output, Mode::Separate] {
                for flags in [NONE, ALL] {
                    check(&shape, &config, mode, flags, C64::new(0.0, 0.0), beta, true);
                }
            }
        }
    }
}

/// An empty contraction has no terms: the product is zero and the update is the
/// same output-only pass, again reading no input.
#[test]
fn an_empty_contraction_updates_the_output_only() {
    let beta = C64::new(0.5, 0.25);
    let shape = Shape {
        name: "empty K",
        a: vec![4, 0],
        b: vec![0, 3],
        d: vec![4, 3],
        la: vec![0, 2],
        lb: vec![2, 1],
        ld: vec![0, 1],
        default: Algorithm::Packed,
    };
    for config in configs() {
        for mode in [Mode::Overwrite, Mode::Output, Mode::Separate] {
            for flags in [NONE, ALL] {
                check(
                    &shape,
                    &config,
                    mode,
                    flags,
                    C64::new(1.0, 0.5),
                    beta,
                    false,
                );
            }
        }
    }
}
