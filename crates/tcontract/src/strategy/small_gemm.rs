//! Fixed-size kernels for the tiny square products of a batched contraction.
//!
//! A batch of `2x2 * 2x2` or `3x3 * 3x3` products (tenferro-rs#2745) spends
//! its time in the general GEMM's per-item setup, not in arithmetic: 8 or 27
//! multiply-adds per product against a plan, feature dispatch and a wrapper
//! stack. The faer strategy therefore routes an item whose `m = n = k` is 2 or
//! 3 here. The kernel reads both operands through the strides the strategy
//! already fused, so conjugation flags, transposed operands (a stride
//! permutation) and any batch stride need no copy and no layout condition.
//! It allocates nothing and packs nothing.

use crate::api::Scalar;
use tcontract_kernel::Element;

/// Whether `(m, n, k)` has a fixed-size kernel.
#[inline]
pub(crate) fn supports(m: usize, n: usize, k: usize) -> bool {
    m == n && n == k && (m == 2 || m == 3)
}

/// `D (+)= alpha * op(A) * op(B)` for one `N x N` product.
///
/// Both operands are loaded whole before `D` is written, so the update is
/// correct even if the arithmetic order differs from the general path.
///
/// # Safety
///
/// The pointers with these strides address valid elements of an `N x N`
/// matrix each; `d` is exclusive and does not alias `a` or `b`. With
/// `accumulate` false, `d` is only written.
#[allow(clippy::too_many_arguments)] // INVARIANT: the GEMM argument set.
#[inline(always)]
pub(crate) unsafe fn matmul<T: Scalar, const N: usize>(
    accumulate: bool,
    d: *mut T,
    (drs, dcs): (isize, isize),
    a: *const T,
    (ars, acs): (isize, isize),
    conj_a: bool,
    b: *const T,
    (brs, bcs): (isize, isize),
    conj_b: bool,
    alpha: T,
) {
    // SAFETY: every offset below stays inside the N x N extents the caller
    // vouches for.
    unsafe {
        let load = |p: *const T, rs: isize, cs: isize, conj: bool| {
            let mut m = [[*p; N]; N]; // [column][row]
            for (j, col) in m.iter_mut().enumerate() {
                for (i, v) in col.iter_mut().enumerate() {
                    let x = *p.offset(i as isize * rs + j as isize * cs);
                    *v = if conj { Element::conj(x) } else { x };
                }
            }
            m
        };
        let am = load(a, ars, acs, conj_a);
        let bm = load(b, brs, bcs, conj_b);
        let scale = alpha != <T as Element>::one();
        for j in 0..N {
            for i in 0..N {
                let mut s = am[0][i] * bm[j][0];
                for k in 1..N {
                    s = s + am[k][i] * bm[j][k];
                }
                if scale {
                    s = alpha * s;
                }
                let dst = d.offset(i as isize * drs + j as isize * dcs);
                *dst = if accumulate { *dst + s } else { s };
            }
        }
    }
}

/// `matmul` over the items `items` of one batch: item `i` starts `i * h`
/// elements after the batch origin of each operand.
///
/// # Safety
///
/// As [`matmul`] for every item's origins; output items are disjoint.
#[inline(always)]
pub(crate) unsafe fn batch<T: Scalar, const N: usize>(
    items: core::ops::Range<usize>,
    accumulate: bool,
    (d, hd, ds): (*mut T, isize, (isize, isize)),
    (a, ha, as_, ca): (*const T, isize, (isize, isize), bool),
    (b, hb, bs, cb): (*const T, isize, (isize, isize), bool),
    alpha: T,
) {
    for i in items {
        let i = i as isize;
        // SAFETY: the caller's contract.
        unsafe {
            matmul::<T, N>(
                accumulate,
                d.offset(i * hd),
                ds,
                a.offset(i * ha),
                as_,
                ca,
                b.offset(i * hb),
                bs,
                cb,
                alpha,
            );
        }
    }
}
