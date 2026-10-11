//! The native interleaved AVX2+FMA complex families (`tcontract-kernel`,
//! issue #30) through the real planner and driver: arithmetic, all
//! conjugations, orientation, alpha/beta, `C == D`, strides (including negative
//! ones and offsets), genuine block-scatter contraction with batch labels,
//! untouched padding, residuals scaled to `K` and `|A||B|`, and bitwise
//! equality across worker counts for one family.
use super::common::{all_families, check_family_vs_oracle, run_with_width, Opts, Shape};
use super::compat::{contract_reference, RefOperand};
use super::compat::{ElementOp, Operand};
use super::compat::{Layout, Plan};
use crate::api::Scalar;
use tcontract_kernel::{Blocking, KernelChoice, C32, C64};
use tcontract_kernel::{Element, Real};

fn cplx_ids<T: Scalar>() -> Vec<&'static str> {
    let ids: Vec<_> = all_families::<T>()
        .into_iter()
        .filter(|id| id.starts_with("avx2.") && id.contains(".native."))
        .collect();
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        assert_eq!(
            ids.len(),
            1,
            "the AVX2+FMA family must be available here: {ids:?}"
        );
    }
    ids
}

fn to64<T: Element>(x: T) -> C64 {
    C64::new(x.re().to_f64(), x.im().to_f64())
}
fn eps<T: Element>() -> f64 {
    if core::mem::size_of::<T::Real>() == 4 {
        f32::EPSILON as f64
    } else {
        f64::EPSILON
    }
}

// ------------------------------------------------------------------ oracle

#[test]
fn every_conjugation_orientation_and_scalar_combination_matches_the_oracle() {
    fn run<T>(tol: f64)
    where
        T: Scalar,
    {
        let ids = cplx_ids::<T>();
        if ids.is_empty() {
            eprintln!("no cplx family available on this host (no AVX2+FMA)");
            return;
        }
        for id in ids {
            for mask in 0u8..16 {
                for row_major_d in [false, true] {
                    for (alpha, beta, alias) in [
                        ((1.3, 0.2), (-0.4, 0.1), false),
                        ((1.3, 0.2), (-0.4, 0.1), true),
                        ((1.0, 0.0), (0.0, 0.0), false),
                        ((1.0, 0.0), (1.0, 0.0), true),
                        ((0.0, 1.0), (0.0, -1.0), false),
                        ((0.0, 0.0), (2.0, 0.5), false),
                    ] {
                        check_family_vs_oracle::<T>(
                            id,
                            Opts {
                                alpha_re: alpha.0,
                                alpha_im: alpha.1,
                                beta_re: beta.0,
                                beta_im: beta.1,
                                alias,
                                conj_a: mask & 1 != 0,
                                conj_b: mask & 2 != 0,
                                conj_c: mask & 4 != 0,
                                conj_d: mask & 8 != 0,
                                row_major_d,
                                width: 1,
                            },
                            tol,
                        );
                    }
                }
            }
        }
    }
    run::<C64>(1e-12);
    run::<C32>(3e-5);
}

// ------------------------------------------------------- strided residuals

/// A tensor with arbitrary signed strides inside a guarded allocation.
struct Tensor<T> {
    extents: Vec<i64>,
    strides: Vec<i64>,
    labels: Vec<i64>,
    buf: Vec<T>,
    base: isize,
    sentinel: T,
}

const GUARD: isize = 3;

impl<T: Element> Tensor<T> {
    fn new(extents: &[i64], strides: &[i64], labels: &[i64], seed: usize) -> Self {
        let (mut lo, mut hi) = (0i64, 0i64);
        for (&e, &s) in extents.iter().zip(strides) {
            let reach = (e - 1).max(0) * s;
            if reach < 0 {
                lo += reach;
            } else {
                hi += reach;
            }
        }
        let len = (hi - lo) as usize + 1 + 2 * GUARD as usize;
        let sentinel = T::from_parts(T::Real::from_f64(1234.5), T::Real::from_f64(-678.25));
        let mut t = Self {
            extents: extents.to_vec(),
            strides: strides.to_vec(),
            labels: labels.to_vec(),
            buf: vec![sentinel; len],
            base: GUARD - lo as isize,
            sentinel,
        };
        let mut n = 0usize;
        t.walk(|t, off, _| {
            t.buf[off] = super::common::value::<T>(seed + n);
            n += 1;
        });
        t
    }

    fn offset(&self, idx: &[usize]) -> usize {
        let o: i64 = idx
            .iter()
            .zip(&self.strides)
            .map(|(&i, &s)| i as i64 * s)
            .sum();
        (self.base + o as isize) as usize
    }

    /// Visit every logical element in column-major logical order.
    fn walk(&mut self, mut f: impl FnMut(&mut Self, usize, &[usize])) {
        let mut idx = vec![0usize; self.extents.len()];
        if self.extents.contains(&0) {
            return;
        }
        loop {
            let off = self.offset(&idx);
            f(self, off, &idx);
            let mut d = 0;
            loop {
                if d == idx.len() {
                    return;
                }
                idx[d] += 1;
                if (idx[d] as i64) < self.extents[d] {
                    break;
                }
                idx[d] = 0;
                d += 1;
            }
        }
    }

    fn dense(&self) -> (Vec<C64>, Layout) {
        let layout = Layout::col_major(&self.extents);
        let mut out = vec![C64::new(0.0, 0.0); layout.storage_len().max(1) as usize];
        let mut me = Self {
            extents: self.extents.clone(),
            strides: self.strides.clone(),
            labels: self.labels.clone(),
            buf: self.buf.clone(),
            base: self.base,
            sentinel: self.sentinel,
        };
        let mut lin = 0usize;
        me.walk(|t, off, _| {
            out[lin] = to64(t.buf[off]);
            lin += 1;
        });
        (out, layout)
    }

    fn layout(&self) -> Layout {
        Layout::new(self.extents.clone(), self.strides.clone()).unwrap()
    }

    fn ptr(&self) -> *const T {
        // SAFETY: `base` keeps every addressed element inside `buf`.
        unsafe { self.buf.as_ptr().offset(self.base) }
    }
    fn ptr_mut(&mut self) -> *mut T {
        // SAFETY: as above.
        unsafe { self.buf.as_mut_ptr().offset(self.base) }
    }
}

struct Case<'a> {
    id: &'static str,
    a: (&'a [i64], &'a [i64], &'a [i64]),
    b: (&'a [i64], &'a [i64], &'a [i64]),
    d: (&'a [i64], &'a [i64], &'a [i64]),
    /// Separate `C` strides (same extents and labels as `D`), else in place.
    c_strides: Option<&'a [i64]>,
    blocking: Option<Blocking>,
    mask: u8,
    alpha: (f64, f64),
    beta: (f64, f64),
}

fn refop<'a>(
    data: &'a [C64],
    layout: &'a Layout,
    idx: &'a [i64],
    op: ElementOp,
) -> RefOperand<'a, C64> {
    RefOperand {
        data,
        layout,
        idx,
        op,
    }
}

fn op(c: bool) -> ElementOp {
    if c {
        ElementOp::Conjugate
    } else {
        ElementOp::Identity
    }
}

fn run_case<T>(c: &Case<'_>)
where
    T: Scalar,
{
    let a = Tensor::<T>::new(c.a.0, c.a.1, c.a.2, 1);
    let b = Tensor::<T>::new(c.b.0, c.b.1, c.b.2, 101);
    let mut d = Tensor::<T>::new(c.d.0, c.d.1, c.d.2, 211);
    let cc = c.c_strides.map(|s| Tensor::<T>::new(c.d.0, s, c.d.2, 307));
    let d0 = d.dense();
    let (conj_a, conj_b, conj_c, conj_d) = (
        c.mask & 1 != 0,
        c.mask & 2 != 0,
        c.mask & 4 != 0,
        c.mask & 8 != 0,
    );
    let alpha = T::from_parts(T::Real::from_f64(c.alpha.0), T::Real::from_f64(c.alpha.1));
    let beta = T::from_parts(T::Real::from_f64(c.beta.0), T::Real::from_f64(c.beta.1));
    let (la, lb, ld) = (a.layout(), b.layout(), d.layout());
    let lc = cc.as_ref().map_or_else(|| ld.clone(), |t| t.layout());
    let mut plan = Plan::new(
        Operand {
            layout: &la,
            idx: &a.labels,
            op: op(conj_a),
        },
        Operand {
            layout: &lb,
            idx: &b.labels,
            op: op(conj_b),
        },
        Some(Operand {
            layout: &lc,
            idx: &d.labels,
            op: op(conj_c),
        }),
        Operand {
            layout: &ld,
            idx: &d.labels,
            op: op(conj_d),
        },
    )
    .unwrap()
    .with_kernel(KernelChoice::Id(c.id.into()))
    .unwrap()
    .with_threads(1);
    if let Some(blk) = c.blocking {
        plan = plan.with_blocking(blk);
    }

    // Oracle on dense copies in C64, and |A||B| for the residual scale.
    let (ad, ald) = a.dense();
    let (bd, bld) = b.dense();
    let (cd, cld) = cc.as_ref().map_or_else(|| d0.clone(), |t| t.dense());
    let dlay = Layout::col_major(&d.extents);
    let (a64, b64, c64) = (ad.clone(), bd.clone(), cd.clone());
    let mut want = d0.0.clone();
    contract_reference::<C64>(
        to64(alpha),
        &refop(&a64, &ald, &a.labels, op(conj_a)),
        &refop(&b64, &bld, &b.labels, op(conj_b)),
        to64(beta),
        Some(&refop(&c64, &cld, &d.labels, op(conj_c))),
        &mut want,
        &dlay,
        &d.labels,
        op(conj_d),
    )
    .unwrap();
    let absv = |v: &[C64]| -> Vec<C64> { v.iter().map(|z| C64::new(z.norm(), 0.0)).collect() };
    let (aa, ba) = (absv(&ad), absv(&bd));
    let mut mag = vec![C64::new(0.0, 0.0); want.len()];
    contract_reference::<C64>(
        C64::new(1.0, 0.0),
        &refop(&aa, &ald, &a.labels, ElementOp::Identity),
        &refop(&ba, &bld, &b.labels, ElementOp::Identity),
        C64::new(0.0, 0.0),
        None,
        &mut mag,
        &dlay,
        &d.labels,
        ElementOp::Identity,
    )
    .unwrap();
    let k_total: i64 = {
        let mut k = 1i64;
        for (l, e) in a.labels.iter().zip(&a.extents) {
            if b.labels.contains(l) && !d.labels.contains(l) {
                k *= e;
            }
        }
        k
    };

    // The call under test.
    let c_ptr = match &cc {
        Some(t) => t.ptr(),
        None => d.ptr(),
    };
    // SAFETY: all buffers are guarded allocations covering their layouts; the
    // plan was built from these layouts; D is exclusive (C is either a distinct
    // tensor or D itself, the supported in-place form).
    unsafe { plan.run_raw::<T>(alpha, a.ptr(), b.ptr(), beta, c_ptr, d.ptr_mut()) };

    // Residuals, element by element, scaled to K and |A||B| (+ |beta||C|).
    let tol = 8.0 * (k_total as f64 + 3.0) * eps::<T>();
    let mut got = vec![C64::new(0.0, 0.0); want.len()];
    let mut lin = 0usize;
    let mut touched = vec![false; d.buf.len()];
    let mut dd = Tensor::<T> {
        extents: d.extents.clone(),
        strides: d.strides.clone(),
        labels: d.labels.clone(),
        buf: d.buf.clone(),
        base: d.base,
        sentinel: d.sentinel,
    };
    dd.walk(|t, off, _| {
        got[lin] = to64(t.buf[off]);
        touched[off] = true;
        lin += 1;
    });
    let abs_alpha = to64(alpha).norm();
    let abs_beta = to64(beta).norm();
    for (i, ((g, w), m)) in got.iter().zip(&want).zip(&mag).enumerate() {
        let scale = abs_alpha * m.re + abs_beta * cd[i].norm();
        let err = (*g - *w).norm();
        assert!(
            err <= tol * scale + 1e-30,
            "{} case mask={:#06b}: element {i} err {err:e} > {:e} (got {g}, want {w})",
            c.id,
            c.mask,
            tol * scale
        );
    }
    // Padding and guards are untouched.
    for (off, was) in touched.iter().enumerate() {
        if !was {
            assert_eq!(d.buf[off], d.sentinel, "{}: wrote outside D at {off}", c.id);
        }
    }
    // Inputs are unchanged (compare against a fresh copy).
    let a2 = Tensor::<T>::new(c.a.0, c.a.1, c.a.2, 1);
    let b2 = Tensor::<T>::new(c.b.0, c.b.1, c.b.2, 101);
    assert_eq!(a.buf, a2.buf, "{}: A was modified", c.id);
    assert_eq!(b.buf, b2.buf, "{}: B was modified", c.id);
}

fn cases<'a>(id: &'static str, mr: usize, nr: usize) -> Vec<(String, Case<'a>)> {
    // Leak the shapes: they must outlive the borrowed slices; small, test-only.
    fn leak(v: Vec<i64>) -> &'static [i64] {
        Box::leak(v.into_boxed_slice())
    }
    let (i, j, k) = (0i64, 1, 2);
    let mut out = Vec::new();
    let blk = |mr: usize, nr: usize| Blocking {
        mc: 2 * mr,
        kc: 5,
        nc: 2 * nr,
    };
    let (m, n) = ((2 * mr + 3) as i64, (2 * nr + 1) as i64);
    for kk in [0i64, 1, 2, 17] {
        out.push((
            format!("gemm m={m} n={n} k={kk}"),
            Case {
                id,
                a: (leak(vec![m, kk]), leak(vec![1, m]), leak(vec![i, k])),
                b: (
                    leak(vec![kk, n]),
                    leak(vec![1, kk.max(1)]),
                    leak(vec![k, j]),
                ),
                d: (leak(vec![m, n]), leak(vec![1, m]), leak(vec![i, j])),
                c_strides: None,
                blocking: Some(blk(mr, nr)),
                mask: 0,
                alpha: (1.3, 0.2),
                beta: (-0.4, 0.1),
            },
        ));
    }
    // Row-major inputs with a different-layout separate C.
    out.push((
        "row-major A,B; transposed separate C".into(),
        Case {
            id,
            a: (leak(vec![m, 11]), leak(vec![11, 1]), leak(vec![i, k])),
            b: (leak(vec![11, n]), leak(vec![n, 1]), leak(vec![k, j])),
            d: (leak(vec![m, n]), leak(vec![1, m]), leak(vec![i, j])),
            c_strides: Some(leak(vec![n, 1])),
            blocking: Some(blk(mr, nr)),
            mask: 0b0101,
            alpha: (0.7, -0.3),
            beta: (0.5, 0.25),
        },
    ));
    // Signed strides: A reversed in both modes, B in one, D in one.
    out.push((
        "negative strides".into(),
        Case {
            id,
            a: (leak(vec![m, 9]), leak(vec![-1, -m]), leak(vec![i, k])),
            b: (leak(vec![9, n]), leak(vec![1, -9]), leak(vec![k, j])),
            d: (leak(vec![m, n]), leak(vec![-1, m]), leak(vec![i, j])),
            c_strides: None,
            blocking: Some(blk(mr, nr)),
            mask: 0b1010,
            alpha: (1.1, 0.4),
            beta: (-0.6, 0.2),
        },
    ));
    // Genuine block-scatter contraction with a batch label and gapped output:
    // A[i1,i2,k1,k2,h] B[k1,k2,j1,j2,h] -> D[i1,i2,j1,j2,h]; every mode has a
    // non-nested stride, so row/column offsets are true scatter vectors.
    let (i1, i2, j1, j2, k1, k2, h) = (10i64, 11, 12, 13, 14, 15, 16);
    out.push((
        "block-scatter contraction with batch".into(),
        Case {
            id,
            a: (
                leak(vec![3, 4, 2, 3, 2]),
                leak(vec![5, 17, 150, 300, 1000]),
                leak(vec![i1, i2, k1, k2, h]),
            ),
            b: (
                leak(vec![2, 3, 5, 2, 2]),
                leak(vec![7, 40, 130, 900, 2000]),
                leak(vec![k1, k2, j1, j2, h]),
            ),
            d: (
                leak(vec![3, 4, 5, 2, 2]),
                leak(vec![2, 9, 80, 500, 4000]),
                leak(vec![i1, i2, j1, j2, h]),
            ),
            c_strides: None,
            blocking: Some(blk(mr, nr)),
            mask: 0b0110,
            alpha: (0.9, -0.8),
            beta: (0.3, 0.6),
        },
    ));
    out
}

#[test]
fn strided_scatter_and_batched_cases_meet_the_k_scaled_residual_bound() {
    fn run<T>(mr_nr: impl Fn(&str) -> (usize, usize))
    where
        T: Scalar,
    {
        for id in cplx_ids::<T>() {
            let (mr, nr) = mr_nr(id);
            for (name, case) in cases(id, mr, nr) {
                for mask in [0u8, 0b0001, 0b0010, 0b0100, 0b1000, 0b1111, case.mask] {
                    let c = Case {
                        mask,
                        ..case_clone(&case)
                    };
                    let _ = &name;
                    run_case::<T>(&c);
                }
            }
        }
    }
    run::<C64>(|_| (4, 4));
    run::<C32>(|_| (8, 4));
}

fn case_clone<'a>(c: &Case<'a>) -> Case<'a> {
    Case {
        id: c.id,
        a: c.a,
        b: c.b,
        d: c.d,
        c_strides: c.c_strides,
        blocking: c.blocking,
        mask: c.mask,
        alpha: c.alpha,
        beta: c.beta,
    }
}

// ------------------------------------------------------ workers, selection

#[test]
fn one_family_is_bitwise_identical_across_worker_counts() {
    fn run<T>()
    where
        T: Scalar,
    {
        for id in cplx_ids::<T>() {
            let shape = Shape {
                m: 70,
                n: 61,
                k: 90,
            };
            let one = run_with_width::<T>(id, 1, shape, false);
            for w in [2usize, 3, 4] {
                let many = run_with_width::<T>(id, w, shape, false);
                assert!(
                    one.iter().zip(&many).all(|(x, y)| x == y),
                    "{id}: width {w} differs bitwise from width 1"
                );
            }
        }
    }
    run::<C64>();
    run::<C32>();
}

#[test]
fn the_default_selection_does_not_change() {
    let l = Layout::col_major(&[9, 9]);
    let plan = |t: usize| {
        Plan::new(
            Operand::new(&l, &[0, 2]),
            Operand::new(&l, &[2, 1]),
            None,
            Operand::new(&l, &[0, 1]),
        )
        .unwrap()
        .with_threads(t)
    };
    for t in [1, 4] {
        for id in [
            plan(t).resolved::<C64>().unwrap().family().id,
            plan(t).resolved::<C32>().unwrap().family().id,
        ] {
            assert!(
                !(id.starts_with("avx2.") && id.contains(".native.")),
                "{id}"
            );
        }
    }
    // An explicit id is honoured, or refused at plan creation on a host
    // without the instructions: never silently replaced.
    let r = plan(1)
        .with_kernel(KernelChoice::Id("avx2.c64.native.4x4".into()))
        .unwrap()
        .resolved::<C64>();
    match r {
        Ok(rg) => assert_eq!(rg.family().id, "avx2.c64.native.4x4"),
        Err(_) => assert!(all_families::<C64>()
            .iter()
            .all(|id| !(id.starts_with("avx2.") && id.contains(".native.")))),
    }
}
