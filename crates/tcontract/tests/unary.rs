//! The labels-based unary update, `D = alpha * op_A(A[labels_a]) + beta * D_old`:
//! a permutation, a diagonal, a reduction, conjugation and in-place
//! accumulation, for every dtype, against an independent label oracle.

use num_complex::Complex32 as C32;
use strided_view::StridedView;
use tcontract::api::{DType, Error, Scalar};
use tcontract::unary::{add, Unary};
use tcontract_exec::Exec;
use tcontract_kernel::Element;

mod common;
use common::*;

/// `D = alpha * op_A(A[labels_a]) + beta * D_old`, computed from the labels
/// alone: `A` indices contribute to the output index their labels name, axes
/// sharing a label must agree with each other, and `beta` applies to the
/// original output once, not once per contributing term.
fn oracle<S: Scalar>(
    alpha: S,
    a: &T<S>,
    la: &[i64],
    conj_a: bool,
    beta: S,
    ld: &[i64],
    d0: &T<S>,
) -> T<S> {
    let mut sum = T::<S> {
        data: vec![<S as Element>::zero(); d0.data.len()],
        dims: d0.dims.clone(),
        strides: d0.strides.clone(),
        offset: d0.offset,
    };
    for_each_index(&a.dims, |ia| {
        // Axes sharing a label read one value: the diagonal constraint.
        for (k, (&lk, &ik)) in la.iter().zip(ia).enumerate() {
            for (&lj, &ij) in la.iter().zip(ia).skip(k + 1) {
                if lk == lj && ik != ij {
                    return;
                }
            }
        }
        let mut io = vec![0usize; ld.len()];
        for (j, &l) in ld.iter().enumerate() {
            let Some(k) = la.iter().position(|&al| al == l) else {
                return;
            };
            io[j] = ia[k];
        }
        let x = if conj_a {
            Element::conj(a.get(ia))
        } else {
            a.get(ia)
        };
        sum.set(&io, Element::add(sum.get(&io), x));
    });
    let mut out = d0.clone();
    for_each_index(&d0.dims, |io| {
        let base = if beta == <S as Element>::zero() {
            <S as Element>::zero()
        } else {
            Element::mul(beta, d0.get(io))
        };
        out.set(io, Element::add(Element::mul(alpha, sum.get(io)), base));
    });
    out
}

fn run<S: Scalar>(
    dtype: DType,
    alpha: S,
    beta: S,
    a: &T<S>,
    la: &[i64],
    conj_a: bool,
    ld: &[i64],
    d: &mut T<S>,
) -> Result<(), Error> {
    let spec = if conj_a {
        Unary::new(la, ld).with_conj_a()
    } else {
        Unary::new(la, ld)
    };
    assert_eq!(<S as Scalar>::STORAGE, dtype);
    add(
        &Exec::serial(),
        &spec,
        alpha,
        &a.view(),
        beta,
        &mut d.view_mut(),
    )
}

/// One case per dtype: a permuted output with a conjugated input and a
/// non-zero beta, checked against the oracle.
fn case<S: Scalar>(dtype: DType, alpha: S, beta: S, seed: u64) {
    let a = T::<S>::new(&[3, 4], seed);
    let d0 = T::<S>::new(&[4, 3], seed + 1);
    let mut d = d0.clone();
    run(dtype, alpha, beta, &a, &[0, 1], true, &[1, 0], &mut d).unwrap();
    let want = oracle(alpha, &a, &[0, 1], true, beta, &[1, 0], &d0);
    assert!(rel_err(&d, &want) < 1e-6, "{dtype:?}");
}

#[test]
fn every_dtype_updates_with_conjugation_and_beta() {
    case(DType::F32, 2.0f32, 0.5f32, 1);
    case(DType::F64, -1.5f64, 0.25f64, 2);
    case(DType::C32, C32::new(0.0, 2.0), C32::new(-1.0, 0.5), 3);
    case(DType::F64, 1.0f64, 0.0f64, 4);
}

#[test]
fn a_diagonal_with_a_reduced_label() {
    // d[i] = sum_j a[i, j, j], with A's axes 1 and 2 sharing a label.
    let a = T::<f64>::new(&[2, 3, 3], 5);
    let d0 = T::<f64>::new(&[2], 6);
    let mut d = d0.clone();
    run(DType::F64, 2.0, 0.0, &a, &[0, 1, 1], false, &[0], &mut d).unwrap();
    let want = oracle(2.0, &a, &[0, 1, 1], false, 0.0, &[0], &d0);
    assert!(rel_err(&d, &want) < 1e-12);
}

#[test]
fn a_permutation_with_a_free_output_order() {
    let a = T::<f64>::new(&[3, 4, 2], 7);
    let d0 = T::<f64>::new(&[2, 3, 4], 8);
    let mut d = d0.clone();
    // D[2, 0, 1] = A[0, 1, 2]
    run(
        DType::F64,
        1.0,
        0.0,
        &a,
        &[0, 1, 2],
        false,
        &[2, 0, 1],
        &mut d,
    )
    .unwrap();
    let want = oracle(1.0, &a, &[0, 1, 2], false, 0.0, &[2, 0, 1], &d0);
    assert!(rel_err(&d, &want) < 1e-12);
    for_each_index(&[3, 4, 2], |i| {
        assert_eq!(d.get(&[i[2], i[0], i[1]]), a.get(i))
    });
}

#[test]
fn an_isolated_reduction_over_a_rank_two_input() {
    // d[j] = sum_{i,k} a[i, j, k]
    let a = T::<f64>::new(&[3, 2, 4], 9);
    let d0 = T::<f64>::new(&[2], 10);
    let mut d = d0.clone();
    run(DType::F64, 1.0, 3.0, &a, &[0, 1, 2], false, &[1], &mut d).unwrap();
    let want = oracle(1.0, &a, &[0, 1, 2], false, 3.0, &[1], &d0);
    assert!(rel_err(&d, &want) < 1e-12);
}

#[test]
fn beta_zero_never_reads_the_output() {
    let a = T::<f64>::new(&[2, 2], 11);
    let mut d = T::<f64> {
        data: vec![f64::NAN; 4],
        dims: vec![2, 2],
        strides: vec![1, 2],
        offset: 0,
    };
    run(DType::F64, 1.0, 0.0, &a, &[0, 1], false, &[1, 0], &mut d).unwrap();
    for_each_index(&[2, 2], |i| assert_eq!(d.get(&[i[1], i[0]]), a.get(i)));
}

#[test]
fn an_output_label_that_a_does_not_carry_is_refused() {
    let a = T::<f64>::new(&[2, 2], 12);
    let mut d = T::<f64>::new(&[2], 13);
    assert!(run(DType::F64, 1.0, 0.0, &a, &[0, 1], false, &[2], &mut d).is_err());
    // A label list of the wrong length is a configuration error, not a panic.
    assert!(run(DType::F64, 1.0, 0.0, &a, &[0], false, &[0], &mut d).is_err());
}

#[test]
fn a_non_injective_output_is_refused_before_any_write() {
    let a = T::<f64>::new(&[2, 2], 14);
    let mut d = T::<f64> {
        data: vec![5.0; 4],
        dims: vec![2, 2],
        strides: vec![0, 1],
        offset: 0,
    };
    assert!(matches!(
        run(DType::F64, 1.0, 0.0, &a, &[0, 1], false, &[0, 1], &mut d),
        Err(Error::Alias(_)) | Err(Error::Layout(_))
    ));
    assert!(d.data.iter().all(|&x| x == 5.0));
}

#[test]
fn the_same_plan_runs_on_different_buffers() {
    // The public `add` builds a plan per call; a caller that keeps one is the
    // `Plan::execute_into_accum` path. Pin that the two agree.
    use tcontract::api::{CSpec, Labels, LayoutSpec, OperandSpec, Problem};
    use tcontract::{Plan, PlanConfig};
    let a = T::<f64>::new(&[3, 4], 15);
    let d0 = T::<f64>::new(&[4, 3], 16);
    let mut via_api = d0.clone();
    run(
        DType::F64,
        1.0,
        0.0,
        &a,
        &[0, 1],
        false,
        &[1, 0],
        &mut via_api,
    )
    .unwrap();

    let l = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
    let p = Problem::from_labels(
        DType::F64,
        l(&[3, 4], &[1, 3]),
        l(&[], &[]),
        CSpec::Absent,
        l(&[4, 3], &[1, 4]),
        &Labels::new(&[0, 1], &[], &[1, 0]),
    )
    .unwrap();
    let plan = Plan::<f64>::new(&p, &PlanConfig::default()).unwrap();
    let b = [1.0f64];
    let bv = StridedView::new(&b[..], &[], &[], 0).unwrap();
    let exec = Exec::serial();
    for seed in 0..3 {
        let a = T::<f64>::new(&[3, 4], 17 + seed);
        let mut d = T::<f64>::new(&[4, 3], 20 + seed);
        plan.execute_into(&exec, 1.0, &a.view(), &bv, &mut d.view_mut())
            .unwrap();
        for_each_index(&[3, 4], |i| assert_eq!(d.get(&[i[1], i[0]]), a.get(i)));
    }
}
