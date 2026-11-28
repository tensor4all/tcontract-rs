//! Fixed-size `2x2` / `3x3` batched products (tensor4all/tenferro-rs#2745).
//!
//! The faer strategy hands a batch whose `m = n = k` is 2 or 3 to a fixed-size
//! kernel. Every case here runs the default plan (which must be that kernel's
//! route: `Algorithm::Faer`) and the same problem forced through the packed
//! driver, an independent general path, and requires the two to agree within a
//! rounding tolerance. Layouts cover a batch-contiguous and a batch-strided
//! operand, a padded batch stride, a transposed operand folded into strides,
//! and both conjugations; update modes cover overwrite, `alpha != 1`, and
//! accumulation in D.

use num_complex::{Complex32, Complex64};
use strided_view::{StridedView, StridedViewMut};
use tcontract::api::{
    AccumulationSource, CSpec, Labels, LayoutSpec, Op, OperandSpec, Problem, Scalar,
};
use tcontract::{Algorithm, Plan, PlanConfig};
use tcontract_exec::Exec;
use tcontract_kernel::Real;

mod common;
use common::fill;

/// Where the batch axis sits in memory and how the matrix is stored.
#[derive(Clone, Copy, Debug)]
enum Layout {
    /// `[N, N, h]` column-major: matrices back to back.
    ColMajor,
    /// `[N, N, h]` with the matrix transposed in memory (`A^T` is stored).
    Transposed,
    /// Batch axis fastest: element `(i, j, b)` at `i * h + j * N * h + b`.
    BatchFirst,
    /// Matrices separated by one padding element.
    Padded,
}

fn strides(l: Layout, n: usize, h: usize) -> Vec<isize> {
    let (n, h) = (n as isize, h as isize);
    match l {
        Layout::ColMajor => vec![1, n, n * n],
        Layout::Transposed => vec![n, 1, n * n],
        Layout::BatchFirst => vec![h, n * h, 1],
        Layout::Padded => vec![1, n, n * n + 1],
    }
}

fn storage_len(dims: &[usize], s: &[isize]) -> usize {
    1 + dims
        .iter()
        .zip(s)
        .map(|(&d, &s)| (d - 1) * s as usize)
        .sum::<usize>()
}

#[derive(Clone, Copy, Debug)]
struct Case {
    n: usize,
    h: usize,
    la: Layout,
    lb: Layout,
    ld: Layout,
    conj_a: bool,
    conj_b: bool,
    alpha: f64,
    /// `Some(beta)`: accumulate in D (`D = alpha*A*B + beta*D`).
    beta: Option<f64>,
}

fn run<T: Scalar>(c: Case, config: &PlanConfig) -> (Vec<T>, Algorithm) {
    let dims = [c.n, c.n, c.h];
    let (sa, sb, sd) = (
        strides(c.la, c.n, c.h),
        strides(c.lb, c.n, c.h),
        strides(c.ld, c.n, c.h),
    );
    let spec = |s: &[isize], conj: bool| {
        let o = OperandSpec::new(LayoutSpec::new(&dims, s, 0).unwrap());
        if conj {
            o.conj()
        } else {
            o
        }
    };
    let problem = Problem::from_labels(
        T::STORAGE,
        spec(&sa, c.conj_a),
        spec(&sb, c.conj_b),
        match c.beta {
            Some(_) => CSpec::Output(Op::Identity),
            None => CSpec::Absent,
        },
        spec(&sd, false),
        // D[i, j, h] = sum_k A[i, k, h] B[k, j, h]
        &Labels::new(&[0, 2, 3], &[2, 1, 3], &[0, 1, 3]),
    )
    .unwrap();
    let plan = Plan::<T>::new(&problem, config).unwrap();
    let algorithm = plan.report().algorithm;
    let a: Vec<T> = fill(storage_len(&dims, &sa), 11);
    let b: Vec<T> = fill(storage_len(&dims, &sb), 29);
    let mut d: Vec<T> = fill(storage_len(&dims, &sd), 47);
    let (va, vb) = (
        StridedView::new(&a, &dims, &sa, 0).unwrap(),
        StridedView::new(&b, &dims, &sb, 0).unwrap(),
    );
    let alpha = T::from_parts(T::Real::from_f64(c.alpha), T::Real::from_f64(0.0));
    let mut exec = Exec::serial();
    {
        let mut vd = StridedViewMut::new(&mut d, &dims, &sd, 0).unwrap();
        match c.beta {
            None => plan.execute_into(&mut exec, alpha, &va, &vb, &mut vd),
            Some(beta) => {
                let beta = T::from_parts(T::Real::from_f64(beta), T::Real::from_f64(0.0));
                plan.execute_into_accum(
                    &mut exec,
                    alpha,
                    &va,
                    &vb,
                    beta,
                    AccumulationSource::Output,
                    &mut vd,
                )
            }
        }
        .unwrap();
    }
    (d, algorithm)
}

fn check<T: Scalar>(tol: f64) {
    let layouts = [
        Layout::ColMajor,
        Layout::Transposed,
        Layout::BatchFirst,
        Layout::Padded,
    ];
    let mut cases = 0;
    for n in [2usize, 3] {
        // 1 (no tail), 5 and 33 (odd tails), 64.
        for h in [1usize, 5, 33, 64] {
            for (ia, &la) in layouts.iter().enumerate() {
                for (ib, &lb) in layouts.iter().enumerate() {
                    // Pair the output layout with the operands so every layout
                    // is an output once without the full cube.
                    let ld = layouts[(ia + ib) % 2];
                    for (conj_a, conj_b) in [(false, false), (false, true), (true, false)] {
                        for (alpha, beta) in [(1.0, None), (0.75, None), (1.0, Some(0.5))] {
                            let c = Case {
                                n,
                                h,
                                la,
                                lb,
                                ld,
                                conj_a,
                                conj_b,
                                alpha,
                                beta,
                            };
                            let (got, algo) = run::<T>(c, &PlanConfig::default());
                            let (want, general) = run::<T>(c, &PlanConfig::packed());
                            assert_eq!(algo, Algorithm::Faer, "{c:?} must reach the faer route");
                            assert_eq!(general, Algorithm::Packed, "{c:?} reference is packed");
                            assert_eq!(got.len(), want.len());
                            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                                let err = ((g.re().to_f64() - w.re().to_f64()).powi(2)
                                    + (g.im().to_f64() - w.im().to_f64()).powi(2))
                                .sqrt();
                                assert!(
                                    err <= tol,
                                    "{c:?} element {i}: {g:?} vs {w:?} (err {err:e})"
                                );
                            }
                            cases += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(cases > 1000, "ran {cases} cases");
}

#[test]
fn c64_matches_the_general_path() {
    check::<Complex64>(1e-13);
}
#[test]
fn c32_matches_the_general_path() {
    check::<Complex32>(1e-4);
}
#[test]
fn f64_matches_the_general_path() {
    check::<f64>(1e-13);
}
#[test]
fn f32_matches_the_general_path() {
    check::<f32>(1e-4);
}

/// Sizes without a fixed-size kernel keep the general route and its result.
#[test]
fn other_sizes_are_untouched() {
    for n in [1usize, 4] {
        let c = Case {
            n,
            h: 7,
            la: Layout::ColMajor,
            lb: Layout::ColMajor,
            ld: Layout::ColMajor,
            conj_a: false,
            conj_b: true,
            alpha: 1.0,
            beta: None,
        };
        let (got, _) = run::<Complex64>(c, &PlanConfig::default());
        let (want, _) = run::<Complex64>(c, &PlanConfig::packed());
        assert!(
            got.iter().zip(&want).all(|(g, w)| (g - w).norm() < 1e-13),
            "n = {n}"
        );
    }
}
