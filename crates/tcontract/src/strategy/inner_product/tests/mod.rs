//! Route selection, numerical agreement and the output contract of the
//! batched-inner-product strategy.
//!
//! The fixture in [`FAER_BASELINE_F64`] / [`FAER_BASELINE_C64`] is the
//! *previous* route's output: bit patterns captured from `Strategy::Faer` — one
//! degenerate `1 x 1 x k` GEMM per batch item — on `origin/main` before this
//! strategy existed, for f64 and c64 in both layouts. It is the behavioural
//! baseline the change has to stay inside.

use super::{plan as inner_product_plan, Write};
use crate::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
use crate::{Algorithm, ExecutionRoute, Plan, PlanConfig, Reason};
use num_complex::Complex64;
use strided_basic::Identity;
use strided_view::{StridedView, StridedViewMut};
use tcontract_exec::Exec;

/// Deterministic operand values, distinct per operand and per memory slot.
fn val(i: usize, salt: usize) -> f64 {
    (((i * 37 + salt * 101 + 13) % 251) as f64 / 251.0) - 0.5
}

/// `(dims, strides, a_labels, b_labels)` of one case, column-major throughout.
/// `hc` keeps the batch axis contiguous (`pi,pi->p`), `kc` the contracted one
/// (`ip,ip->p`).
fn layout(h: usize, k: usize, hc: bool) -> (Vec<usize>, Vec<isize>, Vec<i64>, Vec<i64>) {
    if hc {
        (vec![h, k], vec![1, h as isize], vec![0, 1], vec![0, 1])
    } else {
        (vec![k, h], vec![1, k as isize], vec![1, 0], vec![1, 0])
    }
}

/// The problem of `sum_k A[h, k] B[h, k]` over `h` batch items.
fn problem(
    dtype: DType,
    dims: &[usize],
    strides: &[isize],
    h: usize,
    la: &[i64],
    lb: &[i64],
) -> Problem {
    let spec = || OperandSpec::new(LayoutSpec::new(dims, strides, 0).unwrap());
    Problem::from_labels(
        dtype,
        spec(),
        spec(),
        CSpec::Absent,
        OperandSpec::new(LayoutSpec::new(&[h], &[1], 0).unwrap()),
        &Labels::new(la, lb, &[0]),
    )
    .unwrap()
}

/// A real GEMM must keep its existing route.
#[test]
fn a_real_gemm_keeps_its_own_route() {
    let l = || LayoutSpec::new(&[2, 2], &[1, 2], 0).unwrap();
    let p = Problem::from_labels(
        DType::F64,
        OperandSpec::new(l()),
        OperandSpec::new(l()),
        CSpec::Absent,
        OperandSpec::new(l()),
        &Labels::new(&[0, 1], &[1, 2], &[0, 2]),
    )
    .unwrap();
    assert!(inner_product_plan(&p).is_none());
    let plan = Plan::<f64>::from_problem(p, &PlanConfig::default()).unwrap();
    assert_ne!(plan.report().algorithm, Algorithm::InnerProduct);
}

/// A batch of dots selects the new rule in both layouts and in both dtypes, and
/// the route it reports is the new one.
#[test]
fn a_batch_of_dots_selects_the_inner_product_route() {
    for (h, k) in [(64usize, 32usize), (3, 5), (1, 7), (7, 1)] {
        for hc in [true, false] {
            let (dims, strides, la, lb) = layout(h, k, hc);
            for dtype in [DType::F64, DType::C64] {
                let p = problem(dtype, &dims, &strides, h, &la, &lb);
                let planned = inner_product_plan(&p);
                assert!(planned.is_some(), "h={h} k={k} hc={hc} {dtype:?}");
                // The traversal follows the unit-stride axis, never a size.
                // One pass per contracted element needs the batch axis at unit
                // stride in A, B and D, and enough items to pay a call; with
                // k == 1 both axes have unit stride.
                let per_k = h >= 16 && (hc || k == 1);
                assert_eq!(planned.unwrap().per_k_for_test(), per_k);
            }
            let p = problem(DType::F64, &dims, &strides, h, &la, &lb);
            let plan = Plan::<f64>::from_problem(p, &PlanConfig::default()).unwrap();
            assert_eq!(plan.report().algorithm, Algorithm::InnerProduct);
            assert_eq!(plan.report().reason, Reason::InnerProduct);
            assert_eq!(
                plan.execution_route(&mut Exec::serial(), 1.0, crate::OutputContract::Fresh)
                    .unwrap(),
                ExecutionRoute::InnerProduct
            );
        }
    }
}

/// The captured baseline of the previous faer route, reproduced inside the
/// stated reordering bound.
///
/// Both routes sum the same `k` products of the same values, so the only
/// difference a correct route can show is a different summation order. For a
/// sum of `k` terms the reordering bound is
/// `(k-1) * eps / (1 - (k-1) * eps) * sum|a_j b_j|`, which for `k <= 64` is
/// below `64 * eps * sum|a_j b_j|`; the assertion uses that (larger, still
/// tiny) bound so it states one number for the whole fixture.
#[test]
fn matches_the_captured_faer_baseline_within_the_reordering_bound() {
    const EPS: f64 = f64::EPSILON;
    for &(tag, h, k, bits) in FAER_BASELINE_F64 {
        let hc = tag == "hc";
        let (dims, strides, la, lb) = layout(h, k, hc);
        let n: usize = dims.iter().product();
        let a: Vec<f64> = (0..n).map(|i| val(i, 1)).collect();
        let b: Vec<f64> = (0..n).map(|i| val(i, 2)).collect();
        let got = Plan::<f64>::from_problem(
            problem(DType::F64, &dims, &strides, h, &la, &lb),
            &PlanConfig::default(),
        )
        .unwrap();
        let mut d = vec![0.0f64; h];
        let av = StridedView::<f64, Identity>::new(&a, &dims, &strides, 0).unwrap();
        let bv = StridedView::<f64, Identity>::new(&b, &dims, &strides, 0).unwrap();
        let mut dv = StridedViewMut::new(&mut d, &[h], &[1], 0).unwrap();
        got.execute_into(&mut Exec::serial(), 1.0, &av, &bv, &mut dv)
            .unwrap();
        // sum|a_j b_j| over the contracted axis, from the same values.
        for (i, &want) in bits.iter().enumerate() {
            let want = f64::from_bits(want);
            let mut magnitude = 0.0f64;
            for kk in 0..k {
                let idx = if hc { i + h * kk } else { i * k + kk };
                magnitude += (a[idx] * b[idx]).abs();
            }
            let bound = 64.0 * EPS * magnitude.max(f64::MIN_POSITIVE);
            assert!(
                (d[i] - want).abs() <= bound,
                "{tag} h={h} k={k} item {i}: {} against {want}, bound {bound:e}",
                d[i]
            );
        }
    }
    for &(tag, h, k, re, im) in FAER_BASELINE_C64 {
        let hc = tag == "hc";
        let (dims, strides, la, lb) = layout(h, k, hc);
        let n: usize = dims.iter().product();
        let a: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(val(i, 1), val(i, 3)))
            .collect();
        let b: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(val(i, 2), val(i, 4)))
            .collect();
        let plan = Plan::<Complex64>::from_problem(
            problem(DType::C64, &dims, &strides, h, &la, &lb),
            &PlanConfig::default(),
        )
        .unwrap();
        assert_eq!(plan.report().algorithm, Algorithm::InnerProduct);
        let mut d = vec![Complex64::new(0.0, 0.0); h];
        let av = StridedView::<Complex64, Identity>::new(&a, &dims, &strides, 0).unwrap();
        let bv = StridedView::<Complex64, Identity>::new(&b, &dims, &strides, 0).unwrap();
        let mut dv = StridedViewMut::new(&mut d, &[h], &[1], 0).unwrap();
        plan.execute_into(
            &mut Exec::serial(),
            Complex64::new(1.0, 0.0),
            &av,
            &bv,
            &mut dv,
        )
        .unwrap();
        for (i, (&wr, &wi)) in re.iter().zip(im).enumerate() {
            let want = Complex64::new(f64::from_bits(wr), f64::from_bits(wi));
            let mut magnitude = 0.0f64;
            for kk in 0..k {
                let idx = if hc { i + h * kk } else { i * k + kk };
                let term = a[idx] * b[idx];
                magnitude += term.re.abs() + term.im.abs();
            }
            let bound = 64.0 * EPS * magnitude.max(f64::MIN_POSITIVE);
            assert!(
                (d[i] - want).norm() <= bound,
                "{tag} h={h} k={k} item {i}: {} against {want}, bound {bound:e}",
                d[i]
            );
        }
    }
}

/// The existing packed route is an independent implementation of the same
/// semantics: the two must agree on the shapes this issue measured.
#[test]
fn matches_the_packed_route_on_measured_shapes() {
    for (h, k) in [(64usize, 1024usize), (1024, 64), (256, 256), (33, 129)] {
        for hc in [true, false] {
            let (dims, strides, la, lb) = layout(h, k, hc);
            let n: usize = dims.iter().product();
            let a: Vec<f64> = (0..n).map(|i| val(i, 1)).collect();
            let b: Vec<f64> = (0..n).map(|i| val(i, 2)).collect();
            let new = Plan::<f64>::from_problem(
                problem(DType::F64, &dims, &strides, h, &la, &lb),
                &PlanConfig::default(),
            )
            .unwrap();
            let packed = Plan::<f64>::from_problem(
                problem(DType::F64, &dims, &strides, h, &la, &lb),
                &PlanConfig {
                    // Any forced partition selects the packed driver.
                    partition: Some(crate::Partition::StaticGrid {
                        pin: None,
                        align_c_lines: false,
                    }),
                    ..PlanConfig::default()
                },
            )
            .unwrap();
            assert_eq!(new.report().algorithm, Algorithm::InnerProduct);
            assert_eq!(packed.report().algorithm, Algorithm::Packed);
            let eval = |plan: &Plan<f64>| {
                let mut d = vec![0.0f64; h];
                let av = StridedView::<f64, Identity>::new(&a, &dims, &strides, 0).unwrap();
                let bv = StridedView::<f64, Identity>::new(&b, &dims, &strides, 0).unwrap();
                let mut dv = StridedViewMut::new(&mut d, &[h], &[1], 0).unwrap();
                plan.execute_into(&mut Exec::serial(), 1.0, &av, &bv, &mut dv)
                    .unwrap();
                d
            };
            let (want, got) = (eval(&packed), eval(&new));
            let mut magnitude = 0.0f64;
            for i in 0..h {
                for kk in 0..k {
                    let idx = if hc { i + h * kk } else { i * k + kk };
                    magnitude += (a[idx] * b[idx]).abs();
                }
            }
            let bound = 64.0 * f64::EPSILON * magnitude;
            for i in 0..h {
                assert!(
                    (got[i] - want[i]).abs() <= bound,
                    "h={h} k={k} hc={hc} item {i}: {} against {}",
                    got[i],
                    want[i]
                );
            }
        }
    }
}

/// Nothing outside the output is written, including under the accumulation
/// contract.
#[test]
fn writes_only_the_output() {
    let (h, k) = (5usize, 9usize);
    let (dims, strides, la, lb) = layout(h, k, true);
    let n: usize = dims.iter().product();
    let a: Vec<f64> = (0..n).map(|i| val(i, 1)).collect();
    let b: Vec<f64> = (0..n).map(|i| val(i, 2)).collect();
    let plan = Plan::<f64>::from_problem(
        problem(DType::F64, &dims, &strides, h, &la, &lb),
        &PlanConfig::default(),
    )
    .unwrap();
    // A strided destination with one guard element on each side.
    const GUARD: f64 = -12345.0;
    let mut storage = vec![GUARD; h + 2];
    let av = StridedView::<f64, Identity>::new(&a, &dims, &strides, 0).unwrap();
    let bv = StridedView::<f64, Identity>::new(&b, &dims, &strides, 0).unwrap();
    {
        let mut dv = StridedViewMut::new(&mut storage[1..1 + h], &[h], &[1], 0).unwrap();
        plan.execute_into(&mut Exec::serial(), 1.0, &av, &bv, &mut dv)
            .unwrap();
    }
    assert_eq!(storage[0], GUARD);
    assert_eq!(storage[h + 1], GUARD);
}

/// The accumulation contract adds to the existing output.
#[test]
fn accumulate_adds_to_the_existing_output() {
    let (h, k) = (4usize, 6usize);
    let (dims, strides, la, lb) = layout(h, k, false);
    let n: usize = dims.iter().product();
    let a: Vec<f64> = (0..n).map(|i| val(i, 1)).collect();
    let b: Vec<f64> = (0..n).map(|i| val(i, 2)).collect();
    let spec = || OperandSpec::new(LayoutSpec::new(&dims, &strides, 0).unwrap());
    let p = Problem::from_labels(
        DType::F64,
        spec(),
        spec(),
        CSpec::Output(crate::api::Op::Identity),
        OperandSpec::new(LayoutSpec::new(&[h], &[1], 0).unwrap()),
        &Labels::new(&la, &lb, &[0]),
    )
    .unwrap();
    let plan = Plan::<f64>::from_problem(p, &PlanConfig::default()).unwrap();
    assert_eq!(plan.report().algorithm, Algorithm::InnerProduct);
    let mut fresh = vec![0.0f64; h];
    let mut accumulated = vec![7.5f64; h];
    let av = StridedView::<f64, Identity>::new(&a, &dims, &strides, 0).unwrap();
    let bv = StridedView::<f64, Identity>::new(&b, &dims, &strides, 0).unwrap();
    {
        let mut dv = StridedViewMut::new(&mut fresh, &[h], &[1], 0).unwrap();
        plan.execute_into(&mut Exec::serial(), 1.0, &av, &bv, &mut dv)
            .unwrap();
    }
    {
        let mut dv = StridedViewMut::new(&mut accumulated, &[h], &[1], 0).unwrap();
        plan.execute_into_accum(
            &mut Exec::serial(),
            1.0,
            &av,
            &bv,
            1.0,
            crate::api::AccumulationSource::Output,
            &mut dv,
        )
        .unwrap();
    }
    for i in 0..h {
        assert!((accumulated[i] - (fresh[i] + 7.5)).abs() <= 1e-12);
    }
    let _ = Write::Overwrite;
}

/// `alpha` is a runtime argument, not a planning one: one plan serves any
/// scalar, and the product is scaled by it.
#[test]
fn scales_by_a_runtime_alpha() {
    for hc in [true, false] {
        let (h, k) = (20usize, 7usize);
        let (dims, strides, la, lb) = layout(h, k, hc);
        let n: usize = dims.iter().product();
        let a: Vec<f64> = (0..n).map(|i| val(i, 1)).collect();
        let b: Vec<f64> = (0..n).map(|i| val(i, 2)).collect();
        let plan = Plan::<f64>::from_problem(
            problem(DType::F64, &dims, &strides, h, &la, &lb),
            &PlanConfig::default(),
        )
        .unwrap();
        let eval = |alpha: f64| {
            let mut d = vec![0.0f64; h];
            let av = StridedView::<f64, Identity>::new(&a, &dims, &strides, 0).unwrap();
            let bv = StridedView::<f64, Identity>::new(&b, &dims, &strides, 0).unwrap();
            let mut dv = StridedViewMut::new(&mut d, &[h], &[1], 0).unwrap();
            plan.execute_into(&mut Exec::serial(), alpha, &av, &bv, &mut dv)
                .unwrap();
            d
        };
        let (one, three) = (eval(1.0), eval(3.0));
        for i in 0..h {
            assert!((three[i] - 3.0 * one[i]).abs() <= 1e-12, "hc={hc} item {i}");
        }
    }
}

/// The overwrite contract never reads the destination: a poisoned one comes out
/// clean, and a genuinely uninitialized one is written before it is read.
#[test]
fn overwrite_never_reads_the_destination() {
    for hc in [true, false] {
        let (h, k) = (18usize, 5usize);
        let (dims, strides, la, lb) = layout(h, k, hc);
        let n: usize = dims.iter().product();
        let a: Vec<f64> = vec![1.0; n];
        let b: Vec<f64> = vec![1.0; n];
        let plan = Plan::<f64>::from_problem(
            problem(DType::F64, &dims, &strides, h, &la, &lb),
            &PlanConfig::default(),
        )
        .unwrap();
        let expected = k as f64;
        let mut poisoned = vec![f64::NAN; h];
        {
            let av = StridedView::<f64, Identity>::new(&a, &dims, &strides, 0).unwrap();
            let bv = StridedView::<f64, Identity>::new(&b, &dims, &strides, 0).unwrap();
            let mut dv = StridedViewMut::new(&mut poisoned, &[h], &[1], 0).unwrap();
            plan.execute_into(&mut Exec::serial(), 1.0, &av, &bv, &mut dv)
                .unwrap();
        }
        for (i, value) in poisoned.iter().enumerate() {
            assert_eq!(*value, expected, "hc={hc} item {i} kept the old value");
        }
    }
}

/// `<layout, h, k, output bits>` from the faer route on `origin/main`.
const FAER_BASELINE_F64: &[(&str, usize, usize, &[u64])] = &[
    ("hc", 1, 1, &[0xbf90baa85909aee5]),
    ("kc", 1, 1, &[0xbf90baa85909aee5]),
    ("hc", 2, 3, &[0xbfb41f8414eb2477, 0xbfc45bfad0a83182]),
    ("kc", 2, 3, &[0xbfc3a6fa7a513a5a, 0xbfb58984c19912c9]),
    (
        "hc",
        3,
        2,
        &[0xbfb89998f9d1cc80, 0xbf9a7a2ff0b8b50b, 0xbfbd9f54c03b8db9],
    ),
    (
        "kc",
        3,
        2,
        &[0xbfb1152884352a79, 0xbfc551dda9fe5580, 0xbf71e95de09b2024],
    ),
    (
        "hc",
        7,
        5,
        &[
            0x3fa87e0801448dd7,
            0xbfd63879385d223e,
            0xbfdbdbcd90929bc5,
            0xbfcdbeab0fea9c31,
            0xbf833852bb3c0fc6,
            0xbfc7609db9db6bc8,
            0xbfc1bd4961a5f243,
        ],
    ),
    (
        "kc",
        7,
        5,
        &[
            0xbfcab878b336a1ff,
            0xbfccb034db68a965,
            0xbfc0194a701c26c4,
            0xbfd10ab0dd6022fa,
            0xbfccd779b63b877d,
            0xbfbf5821b5cf5831,
            0xbfbffc7d3c1e502f,
        ],
    ),
    (
        "hc",
        9,
        17,
        &[
            0xbfe341752f68734e,
            0xbfe37180f3ef88f5,
            0xbfe51ac40b4bdc32,
            0xbfe4392a0f7f7736,
            0xbfe4f16a9677fe36,
            0xbfe2fe2ada581e97,
            0xbfe78d505688e599,
            0xbfe498bc720d628b,
            0xbfe4536e27cc15ee,
        ],
    ),
    (
        "kc",
        9,
        17,
        &[
            0xbfe703ceb3d6d450,
            0xbfe21c3da67b919f,
            0xbfe72cc44bca8252,
            0xbfdda0d38eec45a7,
            0xbfe72defe26b1241,
            0xbfe21e94d3bcb183,
            0xbfe7075177b88424,
            0xbfe245faf82f9f95,
            0xbfe6b8e90bb2d7f9,
        ],
    ),
    (
        "hc",
        64,
        33,
        &[
            0xbff3bc6ac547874d,
            0xbff245ff2163a196,
            0xbff469f96492ead3,
            0xbff4ee36e172745e,
            0xbff41c26c3dd86a2,
            0xbff321ae873d2edb,
            0xbff439f5f273d92b,
            0xbff369116b7f7b5e,
            0xbff29890c16b4d8c,
            0xbff64f1519f3ff26,
            0xbff4c3b1d5fe0672,
            0xbff28530e741fe7e,
            0xbff37f2c4fca1a4a,
            0xbff438453553093e,
            0xbff4775e03f95a2d,
            0xbff29c56188d1d5f,
            0xbff4e6699feeb4bf,
            0xbff44a6081a3bc5e,
            0xbff0afc059b8716d,
            0xbff53e500a98f075,
            0xbff3e7c82f4c5d2f,
            0xbff0d90929bc4766,
            0xbff2514f26c91109,
            0xbff33f4374fb656c,
            0xbff3f47e0c2a788f,
            0xbff22dae5977f2c6,
            0xbff250d252b0d50d,
            0xbff3baba0826b763,
            0xbff09e6474c01a44,
            0xbff42bc146e505dd,
            0xbff65aa7b2998e93,
            0xbff29424fa292dc4,
            0xbff2b17754a74454,
            0xbff4158d6b625af2,
            0xbff3b11abfe217d8,
            0xbff3e81314f4812b,
            0xbff4aa24d2a1bfb2,
            0xbff4d9bc15789d5d,
            0xbff29571da69cdb4,
            0xbff358af2eb79c2b,
            0xbff5bc7103158a4d,
            0xbff358af2eb79c2c,
            0xbff29571da69cdb4,
            0xbff4d9bc15789d5e,
            0xbff4aa24d2a1bfb1,
            0xbff3e81314f4812d,
            0xbff3b11abfe217d9,
            0xbff4158d6b625af4,
            0xbff2b17754a74456,
            0xbff29424fa292dc5,
            0xbff65aa7b2998e95,
            0xbff29bf23baced62,
            0xbff09e6474c01a44,
            0xbff3baba0826b764,
            0xbff250d252b0d510,
            0xbff22dae5977f2c7,
            0xbff264af00f26016,
            0xbff33f4374fb6568,
            0xbff3e11e32012984,
            0xbff0d90929bc4767,
            0xbff3e7c82f4c5d2f,
            0xbff3ae80ff60d7fc,
            0xbff0afc059b8716a,
            0xbff5da2f8cdbd4db,
        ],
    ),
    (
        "kc",
        64,
        33,
        &[
            0xbff3ebd86c1650fe,
            0xbff291ffbb5825e1,
            0xbff4fb7a3738d3b8,
            0xbff35113dba3f48a,
            0xbff326b019cf969d,
            0xbff3a0016e29e0af,
            0xbff3046cd18f2049,
            0xbff419cf969c66bd,
            0xbff4227536b08e50,
            0xbff30e0c19d3bfd0,
            0xbff39a48c6a720f9,
            0xbff33242b275260a,
            0xbff34d4e848224b9,
            0xbff4f8ae88477bdd,
            0xbff2902db49745f6,
            0xbff3eb000d85e907,
            0xbff2b997ce3b2bf1,
            0xbff39b10806780f0,
            0xbff3160b49c7976c,
            0xbff31417f966a785,
            0xbff3b5d9bf345fa0,
            0xbff2c028d44e539e,
            0xbff3fdfc0acf0818,
            0xbff42f121c569db3,
            0xbff3431970ed3d39,
            0xbff37539cf7552c8,
            0xbff335a42cb6c5e0,
            0xbff3792070373297,
            0xbff4ca6425b13e20,
            0xbff28a53c3747640,
            0xbff40d968dd68757,
            0xbff4841c333f798c,
            0xbff20c7db0a00064,
            0xbff32f665eb3c631,
            0xbff335404fd695e6,
            0xbff4fc633a9943ab,
            0xbff29f53b5ee8d3a,
            0xbff26576bab2c009,
            0xbff42e2918f62dc1,
            0xbff36aa0df003b4d,
            0xbff342a4ef3d053e,
            0xbff32b7fbdf1e65f,
            0xbff3976c72e5c11c,
            0xbff48e93da148109,
            0xbff276f3e94b2732,
            0xbff422a72520a64e,
            0xbff4c19d3bfd068b,
            0xbff1efe26b1241c9,
            0xbff33b3b8a99759e,
            0xbff36985ed2fb357,
            0xbff4d68c89a71586,
            0xbff431bc81a7e593,
            0xbff25f8c24bfe855,
            0xbff450aef476c410,
            0xbff3b5972bf43fa3,
            0xbff1830cb2bdf71b,
            0xbff313d566268788,
            0xbff3a8328c8dd04a,
            0xbff4453da57144a0,
            0xbff2560e261b58cd,
            0xbff42a31d36445ef,
            0xbff4f1985bb41435,
            0xbff1c5c13c7e03d8,
            0xbff3398acd78a5b1,
        ],
    ),
];

/// `<layout, h, k, real bits, imaginary bits>` from the faer route on
/// `origin/main`.
const FAER_BASELINE_C64: &[(&str, usize, usize, &[u64], &[u64])] = &[
    ("hc", 1, 1, &[0x3f9718b5af18df47], &[0xbfb7e51dc9fb1551]),
    ("kc", 1, 1, &[0x3f9718b5af18df47], &[0xbfb7e51dc9fb1551]),
    (
        "hc",
        2,
        3,
        &[0x3f9718b5af18df4d, 0x3c50dcca57aa1538],
        &[0x3fb8a9b8a3598bb7, 0xbfa581fa13557325],
    ),
    (
        "kc",
        2,
        3,
        &[0xbfbc3a8794901f26, 0x3fc1005a802b2b7d],
        &[0x3fb44d8be4414237, 0xbf9993412a49c049],
    ),
    (
        "hc",
        3,
        2,
        &[0x3f9718b5af18df4d, 0x3fb700c8c40d6075, 0xbfb700c8c40d6075],
        &[0xbfd29f25f0b2773b, 0x3fc61954d71e2bc2, 0x3fc61954d71e2bc5],
    ),
    (
        "kc",
        3,
        2,
        &[0x3f518dd3686f23da, 0xbfbc80bee231dbb4, 0x3fc1005a802b2b7c],
        &[0xbf8fcefa93487274, 0xbfba500ec2247715, 0x3fc61954d71e2bc4],
    ),
    (
        "hc",
        7,
        5,
        &[
            0x3fce5d2ca516c472,
            0xbfd34f140fa6fea4,
            0xbfd360a1e30f6dc9,
            0x3fc95ce6f2efa2f8,
            0x3fd731ce30c4ee11,
            0xbfc6567202fb88c8,
            0x3f87e498a37ad545,
        ],
        &[
            0xbfd5411790564652,
            0x3fd92478219485b7,
            0xbfd78c0bc8b049aa,
            0xbfe51054b6ded8b4,
            0x3fd9e103262f1c83,
            0xbfc7d1df3971d640,
            0xbfdeae7164ddd07b,
        ],
    ),
    (
        "kc",
        7,
        5,
        &[
            0x3f518dd3686f24d7,
            0xbfc1a1d9b2b8c399,
            0x3fb2f325ba9ad321,
            0xbfaa09d77482ba84,
            0xbf3c9b458dbea034,
            0xbf5da5928e3e8d03,
            0x3fc2b7556efe1608,
        ],
        &[
            0xbf980a0b778cd383,
            0x3fb5a8773467f147,
            0xbfd4899ec31e1f48,
            0xbfd2093956ca6e8d,
            0xbfdc83ddc9335b8f,
            0xbfa61bee979f6ba9,
            0xbfc91df7c0516608,
        ],
    ),
    (
        "hc",
        9,
        17,
        &[
            0xbfb996e537cb8026,
            0xbf9055c12bd933e8,
            0xbfa540712035f65f,
            0x3fc1ef16881ddfd2,
            0xbf74598259c701be,
            0x3fc20573ffa89eb8,
            0xbfc5f786108ded6a,
            0xbf986ffcf1bdce81,
            0xbfb13dba3f48a87c,
        ],
        &[
            0x3fd95838986d6331,
            0xbff74ea3b72ac8a7,
            0x3f9b6d9a532da934,
            0xbff4564a7b8d75c9,
            0xbfd8f759559ed7ed,
            0xbfda7e7a6e5ac4d7,
            0xbff2ad9d2f816a88,
            0xbfc3c11d48fdc90d,
            0xbff75e0c4bc22fe8,
        ],
    ),
    (
        "kc",
        9,
        17,
        &[
            0xbfc3eccdeb12c6f0,
            0x3fc456e89945c1be,
            0xbfc4fd58b995b9a1,
            0x3fd0a63e31dfdfe0,
            0xbfc403b0891dc5d0,
            0x3faaf1d087f22f2e,
            0xbfc30a0858a5d203,
            0x3fa8b9d84ce14ae1,
            0xbfa43d6c3ab982fb,
        ],
        &[
            0xbfd79be8def7e8e5,
            0xbfe88e384f9c550e,
            0xbfe85a3545835799,
            0xbfdde972ae9f2a1a,
            0xbfe89af67ee27470,
            0xbfddda1abed7cad9,
            0xbfe88c23b59b5527,
            0xbfde30b4494166a0,
            0xbff0d69d2e771d85,
        ],
    ),
    (
        "hc",
        64,
        33,
        &[
            0x3fb84d23de2d1040,
            0x3f62025518a71df3,
            0xbfcd2e771d85533a,
            0xbfb0d39f9115ad9d,
            0xbfab131a28022d9c,
            0x3fa966863b344279,
            0xbfcc77a4c06d7c27,
            0xbfb1702dd5e0e605,
            0x3fc53a75e57316ab,
            0xbfd69a736cfc3575,
            0xbfa12e1fbc412936,
            0x3fc2420c0505bbca,
            0xbfac8039bbb19bbe,
            0x3f74727991d3017a,
            0xbf947ce093d80005,
            0x3fb086a54ef0b169,
            0xbfc42e144af223c2,
            0x3f9b24c9450aac9d,
            0x3fd6a988c98374ba,
            0xbfc51c4b2c649821,
            0xbfa0ab0dd6022fad,
            0x3fca8b9c7a811432,
            0x3faf2a9f0cf97a5d,
            0x3fab8bc50c3c27c7,
            0x3fae0b84086f886f,
            0x3fc35ae93b8cae0d,
            0xbf54ef4daa0efa8d,
            0xbf38f7380bfecf53,
            0x3fd3cb41b7c2a896,
            0xbfb8a05bee550c26,
            0xbfce8e95eeae8210,
            0x3fc17e7b78a7c555,
            0xbf77fd8fdb86d44c,
            0xbfbf5b833010f804,
            0x3f6ffcbfcf5e717d,
            0xbfb9cc77b5653d8f,
            0xbfd076c838a91232,
            0xbfc30b5538e671f0,
            0x3fc1885d542c84d9,
            0x3fa292ece7ec97e8,
            0xbfd34b4eb8852ed3,
            0x3fa278eb62e0192c,
            0x3fb222946bb69d52,
            0xbfc4e7c2fbcb5aaf,
            0x3fb41fc6a82b4476,
            0x3fb04626a8d1b489,
            0xbfb1a9e9877ca331,
            0xbfb0ec96c921ac73,
            0x3fb89cb7e0d34c54,
            0x3fc38f716c25eb80,
            0xbfce4d0cfb8f0545,
            0x3fc13d77ac088880,
            0x3fd094f2f1b790ba,
            0xbfb29b3f4ff09772,
            0x3fc5aa06a7e8b137,
            0x3fc06ca3c9e432ab,
            0x3fa4429fbbbc02b2,
            0xbfa518e9b222f846,
            0x3fb0b574d8072f19,
            0x3fd44c604da0b248,
            0xbfc9d69c242a1d0d,
            0x3f62344388bf1b9e,
            0x3fc9e5b180b15c4c,
            0xbfd3c248df9e5904,
        ],
        &[
            0xbff481f2cb3a6fa8,
            0xbff0194a701c26c5,
            0xbfdd4e5293b4a1ad,
            0xbff7731ce30c4ee4,
            0xbff2e3ffb9438bde,
            0xbfe8a7402c785bd7,
            0xbff6a87e7c82f4c7,
            0xbff13f6afc697868,
            0xbff35cbb424d8dfb,
            0xbff27bd85b7180f3,
            0xbff7a1dbc752c499,
            0xbff9162a7ecda66d,
            0xbfecfe077c1e0d97,
            0xbffa488413e0d776,
            0xbffa604fb54c464d,
            0xbfed4b86e4c349d0,
            0xbff90a97e6281701,
            0xbff7a53d4194646e,
            0xbff28e2de89a500d,
            0xbff35d61b26dddf0,
            0xbff5ee21092169de,
            0xbff6c70d1271a34f,
            0xbfec014064a499f2,
            0xbff2df3015213c18,
            0xbff77d4151d12e60,
            0xbfe81791609352dd,
            0xbff020b3d4bfb667,
            0xbff617cdb6056fd7,
            0xbfed288c87927b82,
            0xbfe825174999d230,
            0xbff1482993b5abfa,
            0xbff2dc64662fe43e,
            0xbfeb5f3c0b96c1dc,
            0xbfed341f20380af1,
            0xbff6161cf8e49fe9,
            0xbfed301735d61b26,
            0xbfeb368906e333d2,
            0xbff91594b37d5e71,
            0xbff2e03a6221bc0b,
            0xbff2ca40c7772d1d,
            0xbff549f7714289e5,
            0xbff5e90ed1befa1d,
            0xbff4f0f614c7c63b,
            0xbff29acace405f6f,
            0xbffc59cd3f6f259a,
            0xbff77871adaede9e,
            0xbfed5887a749892e,
            0xbffbff76b04bbe08,
            0xbff590a340947e6f,
            0xbff31e83249fa904,
            0xbff78f96def9fd7c,
            0xbff2e3de6fa37bdf,
            0xbff405eebf26d9ba,
            0xbfed60da0f4d88c7,
            0xbff5face93978143,
            0xbff50cda45652ce0,
            0xbfde438efcea75b0,
            0xbff479a8b59e7411,
            0xbff132409ddb250c,
            0xbfe814727991d301,
            0xbff01907dcdc06c8,
            0xbfef959219bcdd32,
            0xbff02081e64f9e6c,
            0xbfe811b76f708325,
        ],
    ),
    (
        "kc",
        64,
        33,
        &[
            0x3fa1303456422913,
            0x3fb0ae2cbd03af7d,
            0xbfbcc77b5653d841,
            0xbfbbd51b40df6422,
            0x3fbe3ad8b8064629,
            0xbfae759eb6a2833b,
            0xbfa468128fce00ea,
            0x3f6e2ab90e7e87ff,
            0x3faa379cb098b839,
            0x3f8ed97b96d27e6a,
            0xbfc03649928a155c,
            0x3f9b14247502ad74,
            0x3fb47727ec554033,
            0xbfc3e438efcea75a,
            0x3fb96b3495b68249,
            0x3fa93178e41ac514,
            0x3fa90c061008c6db,
            0xbf97a205635ad884,
            0xbfb4c52c7b7abc65,
            0x3fbd21fb817f53d6,
            0x3f9e90aa88af81ff,
            0xbf98cd9c03eac9f3,
            0x3f93c7e03d8108e8,
            0xbf9f82005123764f,
            0xbf8ae23604eab019,
            0xbfbf7b3d5ca0367c,
            0x3f9ed97b96d27e40,
            0x3faa7f6371bb34d6,
            0xbfc36b8e0b94ad3e,
            0x3fba5c8a5e2a767f,
            0x3fab14247502ad8f,
            0xbfb2a94a5f7756c3,
            0x3fb300aba3a1527a,
            0xbfb3d3d6b306c82a,
            0x3fb7f4b84d02948a,
            0xbfaab04794d2b256,
            0xbf819625d0732435,
            0x3fc17a5244a5c58d,
            0xbfa1f2ba959f9fb1,
            0xbf91d48fdc912129,
            0x3f62344388bf1c14,
            0x3f8c71a985aa9c47,
            0x3fa22f1007bc9cbc,
            0xbfc16b3ce81e8643,
            0x3f8ab04794d2b278,
            0x3facf6d005ea9604,
            0xbfb9e0c092eefc8b,
            0x3fb3f2016c1546b4,
            0xbfb2e280ea92d3f5,
            0x3fb0bd42198aeeba,
            0xbfa8cd9c03eac9e7,
            0xbfba3e5fa51bf7f3,
            0x3fc37aa3681bec85,
            0xbfaa430dff9e37bd,
            0xbfa13a9b584728b3,
            0x3fbf996815aeb50d,
            0x3f83aabfd17309f2,
            0x3f9ffaab355d7037,
            0xbf8e759eb6a28339,
            0x3f9d5aace61a9119,
            0xbf9e17ffa47587a3,
            0xbfbd03d0c870d55d,
            0x3fb4e35734893ae5,
            0x3f94064a499f05b5,
        ],
        &[
            0xbff0c5df67371259,
            0xbff4d2cb5bb947b6,
            0xbff3aec7bbd4f9f8,
            0xbff06921060282de,
            0xbff3b558c1e821a5,
            0xbff4f03effd76e42,
            0xbfeb948bf5f05f3e,
            0xbff0a76176186bd3,
            0xbff7ba906c1ea365,
            0xbff4fdb4440de59e,
            0xbff0b2135dc58f4f,
            0xbff574bf0ff717ce,
            0xbff3665a8a922d82,
            0xbff2f920f55db2d7,
            0xbff23b59b552841b,
            0xbff10438ce85074d,
            0xbff4df78e62f5f18,
            0xbff57b82047a5778,
            0xbff0640ecea0131c,
            0xbff37e9aadadd451,
            0xbff36a412b540d4d,
            0xbfeb039049cab652,
            0xbff0afc482ec736c,
            0xbff364673a313d9e,
            0xbff4f4578f096611,
            0xbff0770acbe93232,
            0xbff58a97610196b9,
            0xbff34a86fec4cede,
            0xbff2aba18cb8769b,
            0xbff23ebb2f9423ef,
            0xbff1587b2bad832d,
            0xbff9a12af0306fa6,
            0xbff0cd511e42a5fa,
            0xbff0a5da54ffafe9,
            0xbff9ad535426470b,
            0xbff187f124e450db,
            0xbfeb00501929267b,
            0xbff27e82c0c2c8d3,
            0xbff351076007ee8e,
            0xbff531d897c6f312,
            0xbfede621d92d9249,
            0xbff533cbe827e2f7,
            0xbff3449c68d1f727,
            0xbff2740b19edc154,
            0xbfeb0fea9c30a5b9,
            0xbff191b1b6c90065,
            0xbff9a8b59e740f45,
            0xbff0a32fefae6809,
            0xbff0cc9a09524e05,
            0xbff9a2672ba10793,
            0xbff14b591f8733d4,
            0xbff243de0bc69bb1,
            0xbff2b2b7b94bde46,
            0xbff353907bb9266d,
            0xbff58542965f06ff,
            0xbff083fae99f698f,
            0xbff4f2e96528b621,
            0xbff3549ac8b9a65f,
            0xbff0b23cf9cda34a,
            0xbfeb0c67d84ef5e2,
            0xbff3604eaaff45cd,
            0xbff3973eada9ab21,
            0xbff05e02ef0d2b69,
            0xbff56717dd508876,
        ],
    ),
];
