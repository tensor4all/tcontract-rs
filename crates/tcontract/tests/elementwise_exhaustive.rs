//! The elementwise (all-batch) strategy against the label oracle, for every
//! combination of the four conjugations, the three C modes, zero and nonzero
//! `alpha` and `beta`, in `f64` and `c64`.
//!
//! The layouts are awkward on purpose: A has negative strides (a reversed
//! walk), B broadcasts along an axis (stride 0), D is row-major while A is
//! column-major (a permuted layout), and a separate C reverses one axis of D's
//! shape. The folding of `op_D` into the other terms and the dispatch on the
//! remaining conjugations are exercised in full.

use num_complex::Complex64 as C64;
use strided_view::{StridedView, StridedViewMut};
use tcontract::api::{
    AccumulationSource, CSpec, Labels, LayoutSpec, Op, OperandSpec, Problem, Scalar,
};
use tcontract::{Algorithm, Plan, PlanConfig};
use tcontract_exec::Exec;
use tcontract_kernel::Element;
use tcontract_testkit::fixtures::{row_major, seeded};
use tcontract_testkit::oracle::{contract_reference, RefOperand, RefOutput};

const DIMS: [usize; 2] = [3, 4];
const LABELS: [i64; 2] = [0, 1];

fn op(c: bool) -> Op {
    if c {
        Op::Conjugate
    } else {
        Op::Identity
    }
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    Overwrite,
    Output,
    Separate,
}

/// A layout: strides, logical offset and the storage it needs.
struct Layout {
    strides: [isize; 2],
    offset: isize,
    len: usize,
}

fn operand<'a, T>(data: &'a [T], l: &'a Layout, conj: bool) -> RefOperand<'a, T> {
    RefOperand {
        data,
        dims: &DIMS,
        strides: &l.strides,
        offset: l.offset,
        labels: &LABELS,
        conj,
    }
}

fn run<T: Scalar + 'static>(mode: Mode, conj: [bool; 4], alpha: T, beta: T, tol: f64) {
    // A: both axes reversed, offset at the far corner.
    let la = Layout {
        strides: [-1, -3],
        offset: 11,
        len: 12,
    };
    // B: broadcast along axis 0.
    let lb = Layout {
        strides: [0, 1],
        offset: 0,
        len: 4,
    };
    // D: row-major, a permutation of A's column-major order.
    let ld = Layout {
        strides: row_major(&DIMS).try_into().unwrap(),
        offset: 0,
        len: 12,
    };
    // A separate C: column-major, axis 1 reversed.
    let lc = Layout {
        strides: [1, -3],
        offset: 9,
        len: 12,
    };
    let a: Vec<T> = seeded(1, la.len);
    let b: Vec<T> = seeded(2, lb.len);
    let start: Vec<T> = seeded(3, ld.len);
    let c_sep: Vec<T> = seeded(4, lc.len);

    let spec = |l: &Layout, conj: bool| {
        OperandSpec::new(LayoutSpec::new(&DIMS, &l.strides, l.offset).unwrap()).with_op(op(conj))
    };
    let c_spec = match mode {
        Mode::Overwrite => CSpec::Absent,
        Mode::Output => CSpec::Output(op(conj[2])),
        Mode::Separate => CSpec::Separate(spec(&lc, conj[2])),
    };
    let mut labels = Labels::new(&LABELS, &LABELS, &LABELS);
    if matches!(mode, Mode::Separate) {
        labels = labels.with_c(&LABELS);
    }
    let problem = Problem::from_labels(
        T::STORAGE,
        spec(&la, conj[0]),
        spec(&lb, conj[1]),
        c_spec,
        spec(&ld, conj[3]),
        &labels,
    )
    .unwrap();
    let plan = Plan::<T>::new(&problem, &PlanConfig::default()).unwrap();
    assert_eq!(plan.report().algorithm, Algorithm::Elementwise);

    // The oracle, from the original description.
    let mut want = start.clone();
    let beta_eff = if matches!(mode, Mode::Overwrite) {
        <T as Element>::zero()
    } else {
        beta
    };
    let c_ref = match mode {
        Mode::Overwrite => None,
        Mode::Output => Some(operand(&start[..], &ld, conj[2])),
        Mode::Separate => Some(operand(&c_sep[..], &lc, conj[2])),
    };
    contract_reference(
        alpha,
        &operand(&a, &la, conj[0]),
        &operand(&b, &lb, conj[1]),
        beta_eff,
        c_ref.as_ref(),
        &mut RefOutput {
            data: &mut want,
            dims: &DIMS,
            strides: &ld.strides,
            offset: ld.offset,
            labels: &LABELS,
            conj: conj[3],
        },
    )
    .unwrap();

    let mut got = start.clone();
    let av = StridedView::new(&a, &DIMS, &la.strides, la.offset).unwrap();
    let bv = StridedView::new(&b, &DIMS, &lb.strides, lb.offset).unwrap();
    let cv = StridedView::new(&c_sep, &DIMS, &lc.strides, lc.offset).unwrap();
    {
        let mut dv = StridedViewMut::new(&mut got, &DIMS, &ld.strides, ld.offset).unwrap();
        let source = match mode {
            Mode::Overwrite => None,
            Mode::Output => Some(AccumulationSource::Output),
            Mode::Separate => Some(AccumulationSource::Separate(&cv)),
        };
        match source {
            None => plan
                .execute_into(&mut Exec::serial(), alpha, &av, &bv, &mut dv)
                .unwrap(),
            Some(s) => plan
                .execute_into_accum(&mut Exec::serial(), alpha, &av, &bv, beta, s, &mut dv)
                .unwrap(),
        }
    }
    let err = got
        .iter()
        .zip(&want)
        .map(|(g, w)| Element::sub(*g, *w).norm())
        .fold(0.0, f64::max);
    assert!(
        err < tol,
        "{mode:?} conj [a b c d] {conj:?} alpha {alpha:?} beta {beta:?}: err {err}"
    );
}

fn sweep<T: Scalar + 'static>(scalars: [T; 2], tol: f64) {
    let zero = <T as Element>::zero();
    for mode in [Mode::Overwrite, Mode::Output, Mode::Separate] {
        for bits in 0..16u8 {
            let conj = [bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0];
            for alpha in [zero, scalars[0]] {
                for beta in [zero, scalars[1]] {
                    run(mode, conj, alpha, beta, tol);
                }
            }
        }
    }
}

#[test]
fn every_conjugation_c_mode_and_scalar_matches_the_oracle_in_c64() {
    sweep([C64::new(0.7, -0.3), C64::new(-0.4, 0.6)], 1e-12);
}

#[test]
fn every_conjugation_c_mode_and_scalar_matches_the_oracle_in_f64() {
    sweep([0.7f64, -0.4], 1e-12);
}

/// The special-cased scalars (`alpha == 1`, `beta == 1`) take dedicated
/// strided-rs kernels; they must agree with the oracle too.
#[test]
fn unit_scalars_match_the_oracle() {
    let one = C64::new(1.0, 0.0);
    for mode in [Mode::Overwrite, Mode::Output, Mode::Separate] {
        for bits in 0..16u8 {
            let conj = [bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0];
            run(mode, conj, one, one, 1e-12);
            run(mode, conj, one, C64::new(-0.4, 0.6), 1e-12);
            run(mode, conj, C64::new(0.7, -0.3), one, 1e-12);
        }
    }
}

/// Above strided-rs's parallel threshold the pass fans out on the borrowed
/// pool; the in-place and separate-C forms (raw-pointer in-place body and the
/// checked three-operand kernel) must give the serial, formula result.
#[test]
fn a_threaded_pass_matches_the_formula_and_the_serial_pass() {
    use tcontract_exec::Pool;
    const N: usize = 300; // 90 000 elements, over 2^15
    let dims = [N, N];
    let labels = [0i64, 1];
    let (alpha, beta) = (C64::new(0.7, -0.3), C64::new(-0.4, 0.6));
    let a: Vec<C64> = seeded(1, N * N); // row-major: a permuted layout
    let b: Vec<C64> = seeded(2, N * N);
    let start: Vec<C64> = seeded(3, N * N);
    let sa = [N as isize, 1];
    let sd = [1isize, N as isize];
    let spec = |s: &[isize; 2], conj| {
        OperandSpec::new(LayoutSpec::new(&dims, s, 0).unwrap()).with_op(op(conj))
    };
    let problem = Problem::from_labels(
        C64::STORAGE,
        spec(&sa, true),
        spec(&sd, false),
        CSpec::Output(op(true)),
        spec(&sd, true),
        &Labels::new(&labels, &labels, &labels),
    )
    .unwrap();
    let plan = Plan::<C64>::new(&problem, &PlanConfig::default()).unwrap();
    assert_eq!(plan.report().algorithm, Algorithm::Elementwise);
    let av = StridedView::new(&a, &dims, &sa, 0).unwrap();
    let bv = StridedView::new(&b, &dims, &sd, 0).unwrap();
    let go = |exec: &mut Exec<'_>| {
        let mut out = start.clone();
        let mut dv = StridedViewMut::new(&mut out, &dims, &sd, 0).unwrap();
        plan.execute_into_accum(
            exec,
            alpha,
            &av,
            &bv,
            beta,
            AccumulationSource::Output,
            &mut dv,
        )
        .unwrap();
        out
    };
    let serial = go(&mut Exec::serial());
    let tp = threadteam::Pool::builder().threads(4).build().unwrap();
    let pool = Pool::borrow(&tp);
    let par = go(&mut Exec::new(&pool, pool.size()).unwrap());
    assert_eq!(par, serial);
    // D = conj(alpha * conj(A) * B + beta * conj(D)), D and B column-major.
    for j in 0..N {
        for i in 0..N {
            let x = a[i * N + j].conj();
            let want = (alpha * x * b[j * N + i] + beta * start[j * N + i].conj()).conj();
            assert!((serial[j * N + i] - want).norm() < 1e-12, "({i}, {j})");
        }
    }
}
