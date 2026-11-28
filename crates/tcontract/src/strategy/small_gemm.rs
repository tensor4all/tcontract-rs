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

/// Whether `T` is `f64`.
#[inline]
fn real_f64<T: Scalar>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f64>()
}

/// Whether `(m, n, k)` has a fixed-size kernel for `T`: `2x2` and `3x3` for
/// every dtype (scalar, plus the batch-vectorised path for `f64` / `c64`), and
/// `4x4` / `5x5` for `f64` only. A `c64` `4x4` / `5x5` vector kernel measured
/// no faster than the scalar one and slower than the general path on batches
/// that miss the cache, so those sizes keep the general route.
#[inline]
pub(crate) fn supports<T: Scalar>(m: usize, n: usize, k: usize) -> bool {
    m == n && n == k && (m == 2 || m == 3 || ((m == 4 || m == 5) && real_f64::<T>()))
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
            run(RealBatch::<N> {
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
            run(ComplexBatch::<N> {
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

/// Run `op` at the widest level this kernel is tuned for, chosen at run time.
///
/// `pulp::Arch::new().dispatch` picks the widest level the CPU has. This
/// kernel's register transposes are written for two and four lanes, and a
/// 256-bit level (`V3`) measured at least as fast as 512-bit on the
/// `2x2` .. `5x5` products (their data is 32 to 400 bytes per matrix, too
/// little to fill wider registers), so on x86-64 a CPU with `V3` runs it and
/// every other CPU takes `Arch`'s choice (NEON: two lanes; none: scalar). The
/// result is bit-identical at every level.
#[inline(always)]
fn run<Op: WithSimd<Output = ()>>(op: Op) {
    #[cfg(target_arch = "x86_64")]
    if let Some(v3) = pulp::x86::V3::try_new() {
        // `Simd::vectorize` (not a closure around `with_simd`) enters a
        // `#[target_feature]` function, so the body inlines with the level's
        // instructions; a closure leaves every vector op an out-of-line call.
        Simd::vectorize(v3, op);
        return;
    }
    Arch::new().dispatch(op);
}

/// Most lanes of any `pulp` f64 vector (AVX-512).
const MAX_LANES: usize = 8;
/// Registers staging one operand's `N x N` block, `N <= 5`.
const MAX_BLOCK: usize = 25;

/// Lane `l` of `S::f64s` from `lanes[l]` (`lanes` holds at least `F64_LANES`
/// values). A transmute of a register-sized prefix, which the optimiser keeps
/// in registers; a store followed by a wider load would stall on store
/// forwarding for every gathered operand.
#[inline(always)]
fn pack<S: Simd>(lanes: &[f64; MAX_LANES]) -> S::f64s {
    debug_assert!(core::mem::size_of::<S::f64s>() <= core::mem::size_of::<[f64; MAX_LANES]>());
    // SAFETY: `f64s` is plain old data of at most `MAX_LANES` f64s.
    unsafe { core::ptr::read_unaligned(lanes.as_ptr() as *const S::f64s) }
}

/// The lanes of `v`, in the first `F64_LANES` entries.
#[inline(always)]
fn unpack<S: Simd>(v: S::f64s) -> [f64; MAX_LANES] {
    let mut out = [0.0f64; MAX_LANES];
    // SAFETY: as `pack`.
    unsafe { core::ptr::write_unaligned(out.as_mut_ptr() as *mut S::f64s, v) };
    out
}

/// `S::F64_LANES` consecutive batch items of a real element at `p`, item `l`
/// at `p + l * stride`, as one vector.
///
/// # Safety
///
/// Each of the addressed elements is valid for reads.
#[inline(always)]
unsafe fn gather<S: Simd>(simd: S, p: *const f64, stride: isize) -> S::f64s {
    // SAFETY: the caller's contract; a unit stride addresses `F64_LANES`
    // contiguous elements.
    unsafe {
        if stride == 1 {
            simd.partial_load_f64s(core::slice::from_raw_parts(p, S::F64_LANES))
        } else {
            let mut tmp = [0.0f64; MAX_LANES];
            for (l, t) in tmp.iter_mut().enumerate().take(S::F64_LANES) {
                *t = *p.offset(l as isize * stride);
            }
            pack::<S>(&tmp)
        }
    }
}

/// `gather` for complex elements: the real and imaginary parts as two vectors.
///
/// # Safety
///
/// As [`gather`].
#[inline(always)]
unsafe fn gather_c<S: Simd>(_simd: S, p: *const Complex64, stride: isize) -> (S::f64s, S::f64s) {
    // SAFETY: the caller's contract.
    unsafe {
        let mut re = [0.0f64; MAX_LANES];
        let mut im = [0.0f64; MAX_LANES];
        for l in 0..S::F64_LANES {
            let z = *p.offset(l as isize * stride);
            re[l] = z.re;
            im[l] = z.im;
        }
        (pack::<S>(&re), pack::<S>(&im))
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
            let t = unpack::<S>(v);
            for l in 0..S::F64_LANES {
                *p.offset(l as isize * stride) = t[l];
            }
        }
    }
}

/// Whether items sit back to back as column-major `N x N` matrices.
#[inline(always)]
fn aos(h: isize, (rs, cs): (isize, isize), n: isize) -> bool {
    rs == 1 && cs == n && h == n * n
}

/// Most real fields of one item (`Complex64`, `N = 5`).
const MAX_FIELDS: usize = 2 * MAX_BLOCK;

/// `K` registers, for `pulp`'s lane transposes (which take a plain-old-data
/// value; bytemuck implements `Pod` only for arrays of fixed sizes).
#[derive(Clone, Copy)]
#[repr(transparent)]
struct Rows<R, const K: usize>([R; K]);
// SAFETY: a transparent array of plain-old-data registers has no padding and
// every bit pattern valid.
unsafe impl<R: pulp::bytemuck::Zeroable, const K: usize> pulp::bytemuck::Zeroable for Rows<R, K> {}
// SAFETY: as above.
unsafe impl<R: pulp::bytemuck::Pod, const K: usize> pulp::bytemuck::Pod for Rows<R, K> {}

/// Whether back-to-back items can use [`load_fields`] / [`store_fields`] at this
/// lane count.
#[inline(always)]
fn transposable<S: Simd>() -> bool {
    matches!(S::F64_LANES, 2 | 4)
}

/// The `nf` contiguous reals of each of `S::F64_LANES` back-to-back items at
/// `p`, as `nf` vectors (field `f` of item `l` in lane `l`): `K x K` register
/// transposes (`K` the lane count) of each block of `K` fields, then a lane
/// gather for the `nf % K` left over.
///
/// # Safety
///
/// `K * nf` reals at `p` are valid for reads, `S::F64_LANES == K`.
#[inline(always)]
unsafe fn load_fields_k<S: Simd, const K: usize>(
    simd: S,
    p: *const f64,
    nf: usize,
    out: &mut [S::f64s; MAX_FIELDS],
) {
    // SAFETY: the caller's contract; each unaligned read is one register of
    // `K` reals inside item `l`'s `nf` fields.
    unsafe {
        for q in 0..nf / K {
            let mut rows = [simd.splat_f64s(0.0); K];
            for (l, row) in rows.iter_mut().enumerate() {
                *row = core::ptr::read_unaligned(p.add(l * nf + K * q) as *const S::f64s);
            }
            let t = simd.deinterleave_shfl_f64s(Rows(rows)).0;
            out[K * q..K * q + K].copy_from_slice(&t);
        }
        for f in nf / K * K..nf {
            out[f] = gather(simd, p.add(f), nf as isize);
        }
    }
}

/// [`load_fields_k`] at the lane count of `S`.
///
/// # Safety
///
/// As [`load_fields_k`], with `transposable::<S>()`.
#[inline(always)]
unsafe fn load_fields<S: Simd>(simd: S, p: *const f64, nf: usize, out: &mut [S::f64s; MAX_FIELDS]) {
    // SAFETY: the caller's contract.
    unsafe {
        match S::F64_LANES {
            2 => load_fields_k::<S, 2>(simd, p, nf, out),
            4 => load_fields_k::<S, 4>(simd, p, nf, out),
            _ => unreachable!("guarded by `transposable`"),
        }
    }
}

/// The inverse of [`load_fields_k`].
///
/// # Safety
///
/// `K * nf` reals at `p` are valid for writes, `S::F64_LANES == K`.
#[inline(always)]
unsafe fn store_fields_k<S: Simd, const K: usize>(
    simd: S,
    p: *mut f64,
    nf: usize,
    v: &[S::f64s; MAX_FIELDS],
) {
    // SAFETY: as `load_fields_k`.
    unsafe {
        for q in 0..nf / K {
            let mut rows = [simd.splat_f64s(0.0); K];
            rows.copy_from_slice(&v[K * q..K * q + K]);
            let t = simd.interleave_shfl_f64s(Rows(rows)).0;
            for (l, row) in t.iter().enumerate() {
                core::ptr::write_unaligned(p.add(l * nf + K * q) as *mut S::f64s, *row);
            }
        }
        for f in nf / K * K..nf {
            scatter(simd, p.add(f), nf as isize, v[f]);
        }
    }
}

/// [`store_fields_k`] at the lane count of `S`.
///
/// # Safety
///
/// As [`store_fields_k`], with `transposable::<S>()`.
#[inline(always)]
unsafe fn store_fields<S: Simd>(simd: S, p: *mut f64, nf: usize, v: &[S::f64s; MAX_FIELDS]) {
    // SAFETY: the caller's contract.
    unsafe {
        match S::F64_LANES {
            2 => store_fields_k::<S, 2>(simd, p, nf, v),
            4 => store_fields_k::<S, 4>(simd, p, nf, v),
            _ => unreachable!("guarded by `transposable`"),
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
        let nn = N * N;
        let scale = alpha != 1.0;
        let valpha = simd.splat_f64s(alpha);
        let zero = simd.splat_f64s(0.0);
        // Back-to-back matrices are read and written as register transposes.
        let fast = |h, s: Strides| transposable::<S>() && aos(h, s, N as isize);
        let (fa, fb, fd) = (
            fast(ha, (ars, acs)),
            fast(hb, (brs, bcs)),
            fast(hd, (drs, dcs)),
        );
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
                let mut av = [zero; MAX_FIELDS];
                let mut bv = [zero; MAX_FIELDS];
                // Element `(r, c)` is `av[r + c * N]` either way.
                if fa {
                    load_fields(simd, pa, nn, &mut av);
                } else {
                    for c in 0..N {
                        for r in 0..N {
                            av[r + c * N] =
                                gather(simd, pa.offset(r as isize * ars + c as isize * acs), ha);
                        }
                    }
                }
                if fb {
                    load_fields(simd, pb, nn, &mut bv);
                } else {
                    for c in 0..N {
                        for r in 0..N {
                            bv[r + c * N] =
                                gather(simd, pb.offset(r as isize * brs + c as isize * bcs), hb);
                        }
                    }
                }
                let mut ov = [zero; MAX_FIELDS];
                for j in 0..N {
                    for r in 0..N {
                        let mut s = simd.mul_f64s(av[r], bv[j * N]);
                        for k in 1..N {
                            s = simd.add_f64s(s, simd.mul_f64s(av[r + k * N], bv[k + j * N]));
                        }
                        if scale {
                            s = simd.mul_f64s(valpha, s);
                        }
                        ov[r + j * N] = s;
                    }
                }
                if fd {
                    if accumulate {
                        let mut cur = [zero; MAX_FIELDS];
                        load_fields(simd, pd, nn, &mut cur);
                        for e in 0..nn {
                            ov[e] = simd.add_f64s(cur[e], ov[e]);
                        }
                    }
                    store_fields(simd, pd, nn, &ov);
                } else {
                    for j in 0..N {
                        for r in 0..N {
                            let dst = pd.offset(r as isize * drs + j as isize * dcs);
                            let mut s = ov[r + j * N];
                            if accumulate {
                                s = simd.add_f64s(gather(simd, dst, hd), s);
                            }
                            scatter(simd, dst, hd, s);
                        }
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

/// The `N x N` complex block of `S::F64_LANES` items at `p` as real and
/// imaginary vectors, entry `(r, c)` at index `r + c * N`. With back-to-back
/// items (`back_to_back`) the real part of entry `e` is field
/// `2 * e` and its imaginary part field `2 * e + 1`. Conjugation negates the
/// imaginary part, as `Element::conj`.
///
/// # Safety
///
/// The block's elements are valid for reads; `back_to_back` only if the items
/// are contiguous `N x N` matrices and `transposable::<S>()`.
#[inline(always)]
unsafe fn load_c<S: Simd, const N: usize>(
    simd: S,
    p: *const Complex64,
    h: isize,
    (rs, cs): Strides,
    back_to_back: bool,
    conj: bool,
) -> ([S::f64s; MAX_BLOCK], [S::f64s; MAX_BLOCK]) {
    let zero = simd.splat_f64s(0.0);
    let (mut re, mut im) = ([zero; MAX_BLOCK], [zero; MAX_BLOCK]);
    // SAFETY: the caller's contract.
    unsafe {
        if back_to_back {
            let mut f = [zero; MAX_FIELDS];
            load_fields(simd, p as *const f64, 2 * N * N, &mut f);
            for e in 0..N * N {
                re[e] = f[2 * e];
                im[e] = if conj {
                    simd.neg_f64s(f[2 * e + 1])
                } else {
                    f[2 * e + 1]
                };
            }
        } else {
            for c in 0..N {
                for r in 0..N {
                    let x = gather_c(simd, p.offset(r as isize * rs + c as isize * cs), h);
                    re[r + c * N] = x.0;
                    im[r + c * N] = if conj { simd.neg_f64s(x.1) } else { x.1 };
                }
            }
        }
    }
    (re, im)
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
        let nn = N * N;
        let scale = alpha != Complex64::new(1.0, 0.0);
        let (alr, ali) = (simd.splat_f64s(alpha.re), simd.splat_f64s(alpha.im));
        let zero = simd.splat_f64s(0.0);
        let fast = |h, s: Strides| transposable::<S>() && aos(h, s, N as isize);
        let (fa, fb, fd) = (
            fast(ha, (ars, acs)),
            fast(hb, (brs, bcs)),
            fast(hd, (drs, dcs)),
        );
        let mut i = items.start;
        // SAFETY (whole body): as `RealBatch`.
        unsafe {
            while i + lanes <= items.end {
                let (pa, pb, pd) = (
                    a.offset(i as isize * ha),
                    b.offset(i as isize * hb),
                    d.offset(i as isize * hd),
                );
                let (ar, ai) = load_c::<S, N>(simd, pa, ha, (ars, acs), fa, conj_a);
                let (br, bi) = load_c::<S, N>(simd, pb, hb, (brs, bcs), fb, conj_b);
                let mut or = [zero; MAX_BLOCK];
                let mut oi = [zero; MAX_BLOCK];
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
                        (or[r + j * N], oi[r + j * N]) = (sr, si);
                    }
                }
                if fd {
                    let mut f = [zero; MAX_FIELDS];
                    if accumulate {
                        load_fields(simd, pd as *const f64, 2 * nn, &mut f);
                    }
                    for e in 0..nn {
                        (f[2 * e], f[2 * e + 1]) = if accumulate {
                            (
                                simd.add_f64s(f[2 * e], or[e]),
                                simd.add_f64s(f[2 * e + 1], oi[e]),
                            )
                        } else {
                            (or[e], oi[e])
                        };
                    }
                    store_fields(simd, pd as *mut f64, 2 * nn, &f);
                } else {
                    for j in 0..N {
                        for r in 0..N {
                            let dst = pd.offset(r as isize * drs + j as isize * dcs);
                            let (mut sr, mut si) = (or[r + j * N], oi[r + j * N]);
                            if accumulate {
                                let (dr, di) = gather_c(simd, dst, hd);
                                sr = simd.add_f64s(dr, sr);
                                si = simd.add_f64s(di, si);
                            }
                            let (re, im) = (unpack::<S>(sr), unpack::<S>(si));
                            for l in 0..lanes {
                                *dst.offset(l as isize * hd) = Complex64::new(re[l], im[l]);
                            }
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
            let exact = (1isize, N as isize, nn as isize);
            let soa = (count as isize, (N * count) as isize, 1isize);
            for (la, lb, ld) in [
                (aos, aos, aos),
                (soa, soa, soa),
                (aos, soa, aos),
                (soa, aos, soa),
                (exact, exact, exact),
                (exact, exact, aos),
                (aos, exact, exact),
                (exact, soa, exact),
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

    /// The result does not depend on the dispatched ISA: the portable scalar
    /// `Simd` (one lane), SSE2-class (`V2`, two lanes), AVX2 (`V3`, four lanes),
    /// AVX-512 (`V4`, eight lanes) where the CPU has them, and the
    /// runtime-chosen level all equal the scalar kernel bit for bit.
    #[test]
    fn every_dispatch_level_is_bit_identical() {
        const N: usize = 3;
        let nn = N * N;
        let count = 21usize;
        let len = nn * count;
        let (a, b, d0) = (
            <Complex64 as Case>::make(len, 5),
            <Complex64 as Case>::make(len, 6),
            <Complex64 as Case>::make(len, 7),
        );
        let alpha = Complex64::new(0.75, -0.25);
        let l = (1isize, N as isize, nn as isize);
        let mut want = d0.clone();
        // SAFETY: back-to-back `N x N` items inside `len` elements.
        unsafe {
            batch_scalar::<Complex64, N>(
                0..count,
                true,
                (want.as_mut_ptr(), l.2, (l.0, l.1)),
                (a.as_ptr(), l.2, (l.0, l.1), true),
                (b.as_ptr(), l.2, (l.0, l.1), false),
                alpha,
            );
        }
        let op = |d: &mut Vec<Complex64>| ComplexBatch::<N> {
            items: 0..count,
            accumulate: true,
            d: (d.as_mut_ptr(), l.2, (l.0, l.1)),
            a: (a.as_ptr(), l.2, (l.0, l.1), true),
            b: (b.as_ptr(), l.2, (l.0, l.1), false),
            alpha,
        };
        // One result per level; a level the CPU lacks reproduces `want`.
        let mut got = vec![want.clone(); 5];
        let fresh = || d0.clone();
        got[0] = fresh();
        op(&mut got[0]).with_simd(pulp::Scalar::new());
        #[cfg(target_arch = "x86_64")]
        {
            if let Some(v) = pulp::x86::V2::try_new() {
                got[1] = fresh();
                op(&mut got[1]).with_simd(v);
            }
            if let Some(v) = pulp::x86::V3::try_new() {
                got[2] = fresh();
                op(&mut got[2]).with_simd(v);
            }
            // Eight lanes (the `x86-v4` dev-dependency feature compiles it in).
            if let Some(v) = pulp::x86::V4::try_new() {
                got[3] = fresh();
                op(&mut got[3]).with_simd(v);
            }
        }
        got[4] = fresh();
        Arch::new().dispatch(op(&mut got[4]));
        for g in &got {
            assert!(g.iter().zip(&want).all(|(x, y)| x.bits() == y.bits()));
        }
    }

    #[test]
    fn supports_follows_the_dtype() {
        assert!(supports::<f64>(5, 5, 5) && supports::<f64>(4, 4, 4));
        assert!(!supports::<Complex64>(4, 4, 4) && supports::<Complex64>(3, 3, 3));
        assert!(!supports::<f32>(4, 4, 4) && supports::<f32>(3, 3, 3));
        assert!(!supports::<f64>(6, 6, 6) && !supports::<f64>(2, 3, 2));
    }
}
