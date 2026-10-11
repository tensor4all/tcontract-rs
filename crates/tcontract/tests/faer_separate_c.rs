//! A separately described C on a GEMM-fusable problem runs on faer when its
//! output pass pays (Phase 2 W2/W2b) and on the packed driver otherwise:
//! `D = op_D(alpha * op_A(A) * op_B(B) + beta * op_C(C))` against the
//! independent label oracle, for a C that is D itself (in place, same mapping
//! and origin), a C of D's layout in its own buffer, and a C of another
//! layout, over every conjugation combination and the scalar corner cases
//! (`beta == 0` reads no C, `alpha == 0` reads no A or B), in f64 and c64.
use num_complex::Complex64 as C64;
use tcontract::api::{CSpec, DType, Labels, LayoutSpec, Op, OperandSpec, Problem, Scalar};
use tcontract::{Algorithm, FaerLimit, Plan, PlanConfig};
use tcontract_exec::Exec;
use tcontract_kernel::Element;
use tcontract_testkit::fixtures::{col_major, row_major, seeded, storage_len};
use tcontract_testkit::oracle::{contract_reference, RefOperand, RefOutput};

struct Shape {
    name: &'static str,
    a: Vec<usize>,
    b: Vec<usize>,
    d: Vec<usize>,
    la: Vec<i64>,
    lb: Vec<i64>,
    ld: Vec<i64>,
}

/// GEMM-fusable shapes: a plain matmul, a batched one, and two fused K axes.
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
        },
        Shape {
            name: "batched",
            a: vec![4, 3, 5],
            b: vec![3, 6, 5],
            d: vec![4, 6, 5],
            la: vec![0, 2, 9],
            lb: vec![2, 1, 9],
            ld: vec![0, 1, 9],
        },
        Shape {
            name: "two fused K",
            a: vec![3, 4, 5],
            b: vec![4, 5, 6],
            d: vec![3, 6],
            la: vec![0, 1, 2],
            lb: vec![1, 2, 3],
            ld: vec![0, 3],
        },
    ]
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
enum Source {
    /// C is D itself: same mapping, same origin.
    InPlace,
    /// A separate buffer with D's layout.
    SameLayout,
    /// A separate buffer, row-major while D is column-major.
    OtherLayout,
}

fn dtype<T: Scalar>() -> DType {
    T::STORAGE
}

fn check<T: Scalar + Element>(
    shape: &Shape,
    source: Source,
    flags: [bool; 4],
    alpha: T,
    beta: T,
    nan_c: bool,
) -> Algorithm {
    let [ca, cb, cc, cd] = flags;
    let (sa, sb, sd) = (
        col_major(&shape.a),
        col_major(&shape.b),
        col_major(&shape.d),
    );
    let sc = match source {
        Source::InPlace | Source::SameLayout => sd.clone(),
        Source::OtherLayout => row_major(&shape.d),
    };
    let a: Vec<T> = seeded(1, storage_len(&shape.a, &sa));
    let b: Vec<T> = seeded(2, storage_len(&shape.b, &sb));
    let start: Vec<T> = seeded(3, storage_len(&shape.d, &sd));
    let c_sep: Vec<T> = if nan_c {
        let nan = <T as Element>::from_parts(
            <<T as Element>::Real as tcontract_kernel::Real>::from_f64(f64::NAN),
            <<T as Element>::Real as tcontract_kernel::Real>::from_f64(f64::NAN),
        );
        vec![nan; storage_len(&shape.d, &sc)]
    } else {
        seeded(4, storage_len(&shape.d, &sc))
    };

    let problem = Problem::from_labels(
        dtype::<T>(),
        spec(&shape.a, &sa, ca),
        spec(&shape.b, &sb, cb),
        CSpec::Separate(spec(&shape.d, &sc, cc)),
        spec(&shape.d, &sd, cd),
        &Labels::new(&shape.la, &shape.lb, &shape.ld).with_c(&shape.ld),
    )
    .unwrap();
    // The separate-C routing under test, without rule 3's volume bound (#63),
    // which would send the large c64 shapes to packed at every beta.
    let config = PlanConfig {
        faer_limit: FaerLimit::NONE,
        ..PlanConfig::default()
    };
    let plan = Plan::<T>::new(&problem, &config).unwrap();
    let routes = plan.report().algorithm;

    let c_data: &[T] = match source {
        Source::InPlace => &start,
        _ => &c_sep,
    };
    let mut want = start.clone();
    contract_reference(
        alpha,
        &RefOperand {
            data: &a,
            dims: &shape.a,
            strides: &sa,
            offset: 0,
            labels: &shape.la,
            conj: ca,
        },
        &RefOperand {
            data: &b,
            dims: &shape.b,
            strides: &sb,
            offset: 0,
            labels: &shape.lb,
            conj: cb,
        },
        beta,
        Some(&RefOperand {
            data: c_data,
            dims: &shape.d,
            strides: &sc,
            offset: 0,
            labels: &shape.ld,
            conj: cc,
        }),
        &mut RefOutput {
            data: &mut want,
            dims: &shape.d,
            strides: &sd,
            offset: 0,
            labels: &shape.ld,
            conj: cd,
        },
    )
    .unwrap();

    let mut got = start.clone();
    let dp = got.as_mut_ptr();
    let cp = match source {
        Source::InPlace => dp as *const T,
        _ => c_sep.as_ptr(),
    };
    // SAFETY: every pointer addresses its operand's full layout; D is
    // exclusive, and C is D itself only for the in-place mapping.
    unsafe {
        plan.execute_raw(&Exec::serial(), alpha, a.as_ptr(), b.as_ptr(), beta, cp, dp)
            .unwrap();
    }
    let err = got
        .iter()
        .zip(&want)
        .map(|(g, w)| g.sub(*w).norm())
        .fold(0.0, f64::max);
    assert!(
        err < 1e-12 * 50.0 && got.iter().all(|z| z.norm().is_finite()),
        "{} {source:?} flags {flags:?} alpha {alpha:?} beta {beta:?}: err {err}",
        shape.name
    );
    routes
}

fn scalars<T: Element>() -> Vec<(T, T)> {
    let s = |re: f64, im: f64| {
        T::from_parts(
            <<T as Element>::Real as tcontract_kernel::Real>::from_f64(re),
            <<T as Element>::Real as tcontract_kernel::Real>::from_f64(im),
        )
    };
    vec![
        (s(1.0, 0.0), s(0.0, 0.0)),
        (s(0.7, -0.3), s(0.0, 0.0)),
        (s(1.0, 0.0), s(1.0, 0.0)),
        (s(0.7, -0.3), s(1.0, 0.0)),
        (s(0.7, -0.3), s(-0.4, 0.6)),
        (s(0.0, 0.0), s(-0.4, 0.6)),
    ]
}

fn sweep<T: Scalar + Element>() {
    for shape in shapes() {
        for source in [Source::InPlace, Source::SameLayout, Source::OtherLayout] {
            for bits in 0..16u32 {
                let flags = [bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0];
                for (alpha, beta) in scalars::<T>() {
                    let route = check(&shape, source, flags, alpha, beta, false);
                    assert_eq!(
                        route,
                        Algorithm::Faer,
                        "{} {source:?}: GEMM-fusable, so faer",
                        shape.name
                    );
                }
            }
        }
    }
}

#[test]
fn a_separate_c_matches_the_oracle_in_f64() {
    sweep::<f64>();
}

#[test]
fn a_separate_c_matches_the_oracle_in_c64() {
    sweep::<C64>();
}

/// With `beta == 0` the separate C is never read: NaN in it must not leak.
#[test]
fn beta_zero_does_not_read_a_separate_c() {
    let zero = <C64 as Element>::zero();
    for shape in shapes() {
        for source in [Source::SameLayout, Source::OtherLayout] {
            for bits in [0u32, 15] {
                let flags = [bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0];
                let route = check::<C64>(&shape, source, flags, C64::new(0.7, -0.3), zero, true);
                assert_eq!(route, Algorithm::Faer, "{}", shape.name);
            }
        }
    }
}

/// A TAPP product describes C separately (here with D's layout): the plan
/// still reports faer, not materializing any operand.
#[test]
fn a_tapp_style_gemm_product_reports_faer() {
    let (sa, sb, sd) = (col_major(&[8, 9]), col_major(&[9, 7]), col_major(&[8, 7]));
    let problem = Problem::from_labels(
        DType::F64,
        spec(&[8, 9], &sa, false),
        spec(&[9, 7], &sb, false),
        CSpec::Separate(spec(&[8, 7], &sd, false)),
        spec(&[8, 7], &sd, false),
        &Labels::new(&[0, 2], &[2, 1], &[0, 1]).with_c(&[0, 1]),
    )
    .unwrap();
    let plan = Plan::<f64>::new(&problem, &PlanConfig::default()).unwrap();
    assert_eq!(plan.report().algorithm, Algorithm::Faer);
    assert_eq!(plan.report().materialized, [false; 3]);
}

fn matmul_problem(m: usize, n: usize, k: usize, separate: bool) -> Problem {
    let (sa, sb, sd) = (col_major(&[m, k]), col_major(&[k, n]), col_major(&[m, n]));
    let (c, labels) = if separate {
        (
            CSpec::Separate(spec(&[m, n], &sd, false)),
            Labels::new(&[0, 2], &[2, 1], &[0, 1]).with_c(&[0, 1]),
        )
    } else {
        (CSpec::Absent, Labels::new(&[0, 2], &[2, 1], &[0, 1]))
    };
    Problem::from_labels(
        DType::F64,
        spec(&[m, k], &sa, false),
        spec(&[k, n], &sb, false),
        c,
        spec(&[m, n], &sd, false),
        &labels,
    )
    .unwrap()
}

/// A separate C costs an output pass: a large output with a small K, or a
/// transposed A, plans on the packed driver for every beta (measured, Phase 2
/// W2/W2b and the clean W2b confirmation); a cache-resident output, a large K
/// with a column-major A and a matrix-vector shape go to faer. The same GEMM
/// without a separate C is faer.
#[test]
fn a_separate_c_on_a_large_output_with_small_k_is_packed() {
    let route = |m, n, k, sep| {
        let plan = Plan::<f64>::new(&matmul_problem(m, n, k, sep), &PlanConfig::default()).unwrap();
        plan.report().algorithm
    };
    assert_eq!(route(2048, 1024, 8, true), Algorithm::Packed);
    assert_eq!(route(512, 512, 8, true), Algorithm::Packed);
    assert_eq!(route(2048, 1024, 8, false), Algorithm::Faer);
    assert_eq!(route(2048, 1024, 512, true), Algorithm::Faer);
    assert_eq!(route(128, 128, 8, true), Algorithm::Faer);
    assert_eq!(route(1, 4_000_000, 8, true), Algorithm::Faer);
}

/// A plan that declines faer for a separate C matches the oracle at every
/// beta, and `beta == 0` never reads C.
#[test]
fn a_declined_separate_c_matches_the_oracle_and_beta_zero_never_reads_c() {
    // out = 300 * 300 = 90000 > 2^16 and K = 4: the packed driver, every beta.
    let shape = Shape {
        name: "large output, small K",
        a: vec![300, 4],
        b: vec![4, 300],
        d: vec![300, 300],
        la: vec![0, 2],
        lb: vec![2, 1],
        ld: vec![0, 1],
    };
    let one = <C64 as Element>::one();
    let zero = <C64 as Element>::zero();
    for source in [Source::SameLayout, Source::OtherLayout] {
        for bits in [0u32, 15, 6, 9] {
            let flags = [bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0];
            // beta == 0, C full of NaN: packed, C unread.
            let r = check::<C64>(&shape, source, flags, C64::new(0.7, -0.3), zero, true);
            assert_eq!(r, Algorithm::Packed);
            // beta != 0: the packed route.
            for beta in [one, C64::new(-0.4, 0.6)] {
                check::<C64>(&shape, source, flags, C64::new(0.7, -0.3), beta, false);
            }
        }
    }
    // The in-place C (C is D) is packed too.
    check::<C64>(&shape, Source::InPlace, [false; 4], one, zero, false);
    check::<C64>(
        &shape,
        Source::InPlace,
        [true; 4],
        one,
        C64::new(0.5, 0.5),
        false,
    );
}
