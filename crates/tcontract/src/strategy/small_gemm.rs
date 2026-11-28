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
//!
//! # Vectorising across the batch
//!
//! The products of one batch are independent, so for `f64` and `Complex<f64>`
//! the kernel runs several of them side by side, one per SIMD lane: for a fixed
//! `(i, j, k)` the update `c[i][j] += a[i][k] * b[k][j]` is the same
//! instruction for every item. Operand elements are gathered lane by lane into
//! registers (the batch stride is arbitrary; a unit stride is one vector
//! load), the arithmetic is lane-wise, and results are scattered back. The
//! code runs through `pulp::Arch::dispatch`, so the vector width is chosen at
//! run time from the CPU, never from the build target.
//!
//! **Bit-identical to the scalar kernel.** Every lane performs exactly the
//! scalar kernel's IEEE operations in its order (separate multiply and add,
//! never a fused multiply-add; the complex product is
//! `(ar*br - ai*bi, ar*bi + ai*br)` as in `num_complex`), so the result does
//! not depend on the lane count, the dispatched ISA or the item's position in
//! the batch. Items past the last full group take the scalar kernel.

use core::any::TypeId;

use num_complex::Complex64;
use pulp::{Arch, Simd, WithSimd};

use crate::api::Scalar;
use tcontract_kernel::Element;

/// Whether `T` has batch-vectorised kernels (`f64`, `Complex<f64>`).
#[inline]
fn vectorised<T: Scalar>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f64>() || TypeId::of::<T>() == TypeId::of::<Complex64>()
}

/// Whether `(m, n, k)` has a fixed-size kernel for `T`: `2x2` and `3x3` for
/// every dtype (scalar, plus the batch-vectorised path for `f64` / `c64`), and
/// `4x4` / `5x5` for the dtypes with a vectorised kernel.
#[inline]
pub(crate) fn supports<T: Scalar>(m: usize, n: usize, k: usize) -> bool {
    m == n && n == k && (m == 2 || m == 3 || ((m == 4 || m == 5) && vectorised::<T>()))
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
    d: (*mut T, isize, (isize, isize)),
    a: (*const T, isize, (isize, isize), bool),
    b: (*const T, isize, (isize, isize), bool),
    alpha: T,
) {
    // SAFETY: the caller's contract; `T` is checked against the pointee type
    // of each cast, so the casts are identities.
    unsafe {
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            let (d, a, b) = (
                (d.0 as *mut f64, d.1, d.2),
                (a.0 as *const f64, a.1, a.2, a.3),
                (b.0 as *const f64, b.1, b.2, b.3),
            );
            let alpha = *(&alpha as *const T as *const f64);
            Arch::new().dispatch(RealBatch::<N> {
                items,
                accumulate,
                d,
                a,
                b,
                alpha,
            });
        } else if TypeId::of::<T>() == TypeId::of::<Complex64>() {
            let (d, a, b) = (
                (d.0 as *mut Complex64, d.1, d.2),
                (a.0 as *const Complex64, a.1, a.2, a.3),
                (b.0 as *const Complex64, b.1, b.2, b.3),
            );
            let alpha = *(&alpha as *const T as *const Complex64);
            Arch::new().dispatch(ComplexBatch::<N> {
                items,
                accumulate,
                d,
                a,
                b,
                alpha,
            });
        } else {
            batch_scalar::<T, N>(items, accumulate, d, a, b, alpha);
        }
    }
}

/// The scalar item loop: the reference the vector kernels must reproduce bit
/// for bit, and the route for dtypes without a vector kernel.
///
/// # Safety
///
/// As [`batch`].
#[inline(always)]
pub(crate) unsafe fn batch_scalar<T: Scalar, const N: usize>(
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

/// Most lanes of any `pulp` f64 vector (AVX-512).
const MAX_LANES: usize = 8;
/// Registers staging one operand's `N x N` block, `N <= 5`.
const MAX_BLOCK: usize = 25;

/// `S::F64_LANES` consecutive batch items of a real element at `p`, item `l`
/// at `p + l * stride`, as one vector.
///
/// # Safety
///
/// Each of the addressed elements is valid for reads.
#[inline(always)]
unsafe fn gather<S: Simd>(simd: S, p: *const f64, stride: isize) -> S::f64s {
    // SAFETY: the caller's contract; a unit stride addresses `F64_LANES`
    // contiguous elements, and `tmp` holds `MAX_LANES >= F64_LANES`.
    unsafe {
        if stride == 1 {
            simd.partial_load_f64s(core::slice::from_raw_parts(p, S::F64_LANES))
        } else {
            let mut tmp = [0.0f64; MAX_LANES];
            for (l, t) in tmp.iter_mut().enumerate().take(S::F64_LANES) {
                *t = *p.offset(l as isize * stride);
            }
            simd.partial_load_f64s(&tmp[..S::F64_LANES])
        }
    }
}

/// `gather` for complex elements: the real and imaginary parts as two vectors.
///
/// # Safety
///
/// As [`gather`].
#[inline(always)]
unsafe fn gather_c<S: Simd>(simd: S, p: *const Complex64, stride: isize) -> (S::f64s, S::f64s) {
    // SAFETY: the caller's contract; `tmp` holds `MAX_LANES >= F64_LANES`.
    unsafe {
        let mut re = [0.0f64; MAX_LANES];
        let mut im = [0.0f64; MAX_LANES];
        for l in 0..S::F64_LANES {
            let z = *p.offset(l as isize * stride);
            re[l] = z.re;
            im[l] = z.im;
        }
        (
            simd.partial_load_f64s(&re[..S::F64_LANES]),
            simd.partial_load_f64s(&im[..S::F64_LANES]),
        )
    }
}

/// Write `v` to `S::F64_LANES` items of a real destination at `p`, item `l` at
/// `p + l * stride`.
///
/// # Safety
///
/// Each of the addressed elements is valid for writes.
#[inline(always)]
unsafe fn scatter<S: Simd>(simd: S, p: *mut f64, stride: isize, v: S::f64s) {
    // SAFETY: the caller's contract, as `gather`.
    unsafe {
        if stride == 1 {
            simd.partial_store_f64s(core::slice::from_raw_parts_mut(p, S::F64_LANES), v);
        } else {
            let mut tmp = [0.0f64; MAX_LANES];
            simd.partial_store_f64s(&mut tmp[..S::F64_LANES], v);
            for (l, t) in tmp.iter().enumerate().take(S::F64_LANES) {
                *p.offset(l as isize * stride) = *t;
            }
        }
    }
}

type Strides = (isize, isize);

/// `f64` items `items`, one item per lane.
struct RealBatch<const N: usize> {
    items: core::ops::Range<usize>,
    accumulate: bool,
    d: (*mut f64, isize, Strides),
    a: (*const f64, isize, Strides, bool),
    b: (*const f64, isize, Strides, bool),
    alpha: f64,
}

impl<const N: usize> WithSimd for RealBatch<N> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let Self {
            items,
            accumulate,
            d: (d, hd, (drs, dcs)),
            a: (a, ha, (ars, acs), _),
            b: (b, hb, (brs, bcs), _),
            alpha,
        } = self;
        let lanes = S::F64_LANES;
        let scale = alpha != 1.0;
        let valpha = simd.splat_f64s(alpha);
        let mut i = items.start;
        // SAFETY (whole body): the caller of `batch` vouches for every
        // addressed element of every item in `items`; each group below is
        // `lanes` such items and the groups are disjoint.
        unsafe {
            while i + lanes <= items.end {
                let (pa, pb, pd) = (
                    a.offset(i as isize * ha),
                    b.offset(i as isize * hb),
                    d.offset(i as isize * hd),
                );
                let mut av = [simd.splat_f64s(0.0); MAX_BLOCK];
                let mut bv = [simd.splat_f64s(0.0); MAX_BLOCK];
                for c in 0..N {
                    for r in 0..N {
                        av[r + c * N] =
                            gather(simd, pa.offset(r as isize * ars + c as isize * acs), ha);
                        bv[r + c * N] =
                            gather(simd, pb.offset(r as isize * brs + c as isize * bcs), hb);
                    }
                }
                for j in 0..N {
                    for r in 0..N {
                        let mut s = simd.mul_f64s(av[r], bv[j * N]);
                        for k in 1..N {
                            s = simd.add_f64s(s, simd.mul_f64s(av[r + k * N], bv[k + j * N]));
                        }
                        if scale {
                            s = simd.mul_f64s(valpha, s);
                        }
                        let dst = pd.offset(r as isize * drs + j as isize * dcs);
                        if accumulate {
                            s = simd.add_f64s(gather(simd, dst, hd), s);
                        }
                        scatter(simd, dst, hd, s);
                    }
                }
                i += lanes;
            }
            batch_scalar::<f64, N>(
                i..items.end,
                accumulate,
                (d, hd, (drs, dcs)),
                (a, ha, (ars, acs), false),
                (b, hb, (brs, bcs), false),
                alpha,
            );
        }
    }
}

/// `Complex<f64>` items `items`, one item per lane, real and imaginary parts
/// in separate vectors.
struct ComplexBatch<const N: usize> {
    items: core::ops::Range<usize>,
    accumulate: bool,
    d: (*mut Complex64, isize, Strides),
    a: (*const Complex64, isize, Strides, bool),
    b: (*const Complex64, isize, Strides, bool),
    alpha: Complex64,
}

impl<const N: usize> WithSimd for ComplexBatch<N> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let Self {
            items,
            accumulate,
            d: (d, hd, (drs, dcs)),
            a: (a, ha, (ars, acs), conj_a),
            b: (b, hb, (brs, bcs), conj_b),
            alpha,
        } = self;
        let lanes = S::F64_LANES;
        let scale = alpha != Complex64::new(1.0, 0.0);
        let (alr, ali) = (simd.splat_f64s(alpha.re), simd.splat_f64s(alpha.im));
        let zero = simd.splat_f64s(0.0);
        let mut i = items.start;
        // SAFETY (whole body): as `RealBatch`.
        unsafe {
            while i + lanes <= items.end {
                let (pa, pb, pd) = (
                    a.offset(i as isize * ha),
                    b.offset(i as isize * hb),
                    d.offset(i as isize * hd),
                );
                let mut ar = [zero; MAX_BLOCK];
                let mut ai = [zero; MAX_BLOCK];
                let mut br = [zero; MAX_BLOCK];
                let mut bi = [zero; MAX_BLOCK];
                for c in 0..N {
                    for r in 0..N {
                        let x = gather_c(simd, pa.offset(r as isize * ars + c as isize * acs), ha);
                        let y = gather_c(simd, pb.offset(r as isize * brs + c as isize * bcs), hb);
                        // Conjugation negates the imaginary part, as `Element::conj`.
                        (ar[r + c * N], ai[r + c * N]) =
                            (x.0, if conj_a { simd.neg_f64s(x.1) } else { x.1 });
                        (br[r + c * N], bi[r + c * N]) =
                            (y.0, if conj_b { simd.neg_f64s(y.1) } else { y.1 });
                    }
                }
                for j in 0..N {
                    for r in 0..N {
                        // num_complex: (a*c - b*d, a*d + b*c), then `s + p`.
                        let (mut sr, mut si) = (
                            simd.sub_f64s(
                                simd.mul_f64s(ar[r], br[j * N]),
                                simd.mul_f64s(ai[r], bi[j * N]),
                            ),
                            simd.add_f64s(
                                simd.mul_f64s(ar[r], bi[j * N]),
                                simd.mul_f64s(ai[r], br[j * N]),
                            ),
                        );
                        for k in 1..N {
                            let (xr, xi) = (ar[r + k * N], ai[r + k * N]);
                            let (yr, yi) = (br[k + j * N], bi[k + j * N]);
                            let pr = simd.sub_f64s(simd.mul_f64s(xr, yr), simd.mul_f64s(xi, yi));
                            let pi = simd.add_f64s(simd.mul_f64s(xr, yi), simd.mul_f64s(xi, yr));
                            sr = simd.add_f64s(sr, pr);
                            si = simd.add_f64s(si, pi);
                        }
                        if scale {
                            (sr, si) = (
                                simd.sub_f64s(simd.mul_f64s(alr, sr), simd.mul_f64s(ali, si)),
                                simd.add_f64s(simd.mul_f64s(alr, si), simd.mul_f64s(ali, sr)),
                            );
                        }
                        let dst = pd.offset(r as isize * drs + j as isize * dcs);
                        if accumulate {
                            let (dr, di) = gather_c(simd, dst, hd);
                            sr = simd.add_f64s(dr, sr);
                            si = simd.add_f64s(di, si);
                        }
                        let (mut re, mut im) = ([0.0f64; MAX_LANES], [0.0f64; MAX_LANES]);
                        simd.partial_store_f64s(&mut re[..lanes], sr);
                        simd.partial_store_f64s(&mut im[..lanes], si);
                        for l in 0..lanes {
                            *dst.offset(l as isize * hd) = Complex64::new(re[l], im[l]);
                        }
                    }
                }
                i += lanes;
            }
            batch_scalar::<Complex64, N>(
                i..items.end,
                accumulate,
                (d, hd, (drs, dcs)),
                (a, ha, (ars, acs), conj_a),
                (b, hb, (brs, bcs), conj_b),
                alpha,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic values in roughly `[-1, 1)` with varied exponents, so a
    /// reordered sum or a fused multiply-add changes low bits.
    fn lcg(n: usize, mut x: u64) -> Vec<f64> {
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = (x >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
                u * f64::from_bits(((x >> 3) & 0x3 | 0x3fc) << 52 | (x & 0xfffff) << 20)
                // 2^-3..2^0 scale
            })
            .collect()
    }

    trait Case: Scalar + core::fmt::Debug {
        fn make(n: usize, seed: u64) -> Vec<Self>;
        fn alpha(i: usize) -> Self;
        fn bits(&self) -> [u64; 2];
    }
    impl Case for f64 {
        fn make(n: usize, seed: u64) -> Vec<Self> {
            lcg(n, seed)
        }
        fn alpha(i: usize) -> Self {
            [1.0, 0.75, -1.3][i]
        }
        fn bits(&self) -> [u64; 2] {
            [self.to_bits(), 0]
        }
    }
    impl Case for Complex64 {
        fn make(n: usize, seed: u64) -> Vec<Self> {
            let v = lcg(2 * n, seed);
            v.chunks(2).map(|c| Complex64::new(c[0], c[1])).collect()
        }
        fn alpha(i: usize) -> Self {
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.75, 0.0),
                Complex64::new(-1.3, 0.4),
            ][i]
        }
        fn bits(&self) -> [u64; 2] {
            [self.re.to_bits(), self.im.to_bits()]
        }
    }

    /// The vector kernel and the scalar kernel agree to the last bit for every
    /// batch size (full groups and tails), layout (unit and non-unit batch
    /// stride per operand), conjugation, `alpha` and update mode.
    fn bit_identical<T: Case, const N: usize>() {
        let nn = N * N;
        for count in [1usize, 2, 3, 4, 5, 7, 8, 9, 16, 17, 33] {
            // (row stride, column stride, batch stride, storage length)
            let aos = (1isize, N as isize, nn as isize + 1);
            let soa = (count as isize, (N * count) as isize, 1isize);
            for (la, lb, ld) in [
                (aos, aos, aos),
                (soa, soa, soa),
                (aos, soa, aos),
                (soa, aos, soa),
            ] {
                for (conj_a, conj_b) in [(false, false), (true, false), (false, true)] {
                    for ialpha in 0..3 {
                        for accumulate in [false, true] {
                            let len = nn * count + count + 1;
                            let a = T::make(len, 1);
                            let b = T::make(len, 2);
                            let d0 = T::make(len, 3);
                            let (mut dv, mut ds) = (d0.clone(), d0.clone());
                            let alpha = T::alpha(ialpha);
                            let args = |d: *mut T| {
                                (
                                    (d, ld.2, (ld.0, ld.1)),
                                    (a.as_ptr(), la.2, (la.0, la.1), conj_a),
                                    (b.as_ptr(), lb.2, (lb.0, lb.1), conj_b),
                                )
                            };
                            let (dvp, dsp) = (dv.as_mut_ptr(), ds.as_mut_ptr());
                            // SAFETY: every layout addresses `< len` elements.
                            unsafe {
                                let (d, a, b) = args(dvp);
                                batch::<T, N>(0..count, accumulate, d, a, b, alpha);
                                let (d, a, b) = args(dsp);
                                batch_scalar::<T, N>(0..count, accumulate, d, a, b, alpha);
                            }
                            for (i, (x, y)) in dv.iter().zip(&ds).enumerate() {
                                assert_eq!(
                                    x.bits(),
                                    y.bits(),
                                    "N={N} count={count} la={la:?} lb={lb:?} ld={ld:?} \
                                     conj=({conj_a},{conj_b}) alpha={ialpha} acc={accumulate} \
                                     element {i}: {x:?} vs {y:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn f64_vector_kernel_is_bit_identical_to_scalar() {
        bit_identical::<f64, 2>();
        bit_identical::<f64, 3>();
        bit_identical::<f64, 4>();
        bit_identical::<f64, 5>();
    }

    #[test]
    fn c64_vector_kernel_is_bit_identical_to_scalar() {
        bit_identical::<Complex64, 2>();
        bit_identical::<Complex64, 3>();
        bit_identical::<Complex64, 4>();
        bit_identical::<Complex64, 5>();
    }

    #[test]
    fn supports_follows_the_dtype() {
        assert!(supports::<f64>(5, 5, 5) && supports::<Complex64>(4, 4, 4));
        assert!(!supports::<f32>(4, 4, 4) && supports::<f32>(3, 3, 3));
        assert!(!supports::<f64>(6, 6, 6) && !supports::<f64>(2, 3, 2));
    }
}
