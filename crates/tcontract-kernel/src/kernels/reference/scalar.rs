//! Portable scalar micro-kernels.
//!
//! These are the reference semantics for every vectorised kernel, and the path
//! by which an arbitrary [`Real`] type (extended precision, dual numbers for
//! forward-mode AD, ...) gets a working contraction engine.
//!
//! They are written so that LLVM can auto-vectorise the `f32`/`f64`
//! instantiations reasonably well, but they are not the fast path.
//!
//! # Element types
//!
//! The generic builders [`config_real`] and [`config_cplx`] work for any
//! [`Real`] type, which is how the reference families are built for the four
//! storage types. The set of storage types the registry and the contraction
//! planner accept is sealed to `f32`, `f64`, `Complex<f32>` and `Complex<f64>`;
//! an arbitrary scalar (extended precision, dual numbers) no longer gets a
//! contraction engine by implementing a kernel-set trait.
//!
//! The register block is `MR x NR` in *reals*, and 4x4 is a reasonable default:
//! the accumulator is a stack array of `MR * NR` values.

use crate::{Blocking, ComplexMethod, KernelConfig, PackFormat, Real, TileFormat, Ukr};

/// `ab[j * MR + i] = sum_p a[p * MR + i] * b[p * NR + j]`
///
/// Also the engine of the 1m path: see [`onem_ukr`].
///
/// # Safety
/// `a` must be valid for `MR * kc` reads, `b` for `NR * kc`, `ab` for
/// `MR * NR` writes.
pub unsafe fn real_ukr<T: Real, const MR: usize, const NR: usize>(
    kc: usize,
    a: *const T,
    b: *const T,
    ab: *mut T,
) {
    let mut acc = [[T::ZERO; MR]; NR];
    for p in 0..kc {
        let ap = a.add(p * MR);
        let bp = b.add(p * NR);
        for (j, accj) in acc.iter_mut().enumerate() {
            let bv = *bp.add(j);
            for (i, s) in accj.iter_mut().enumerate() {
                *s += *ap.add(i) * bv;
            }
        }
    }
    for (j, accj) in acc.iter().enumerate() {
        for (i, &v) in accj.iter().enumerate() {
            *ab.add(j * MR + i) = v;
        }
    }
}

/// Planar (split-complex) kernel.
///
/// `a` holds `[re_0..re_{MR-1}, im_0..im_{MR-1}]` per k-step, `b` likewise
/// with `NR`. Writes the real plane to `ab[0 .. MR*NR]` and the imaginary
/// plane to `ab[MR*NR .. 2*MR*NR]`, both column-major within the tile.
///
/// # Safety
/// `a` valid for `2 * MR * kc` reads, `b` for `2 * NR * kc`, `ab` for
/// `2 * MR * NR` writes.
pub unsafe fn planar_ukr<T: Real, const MR: usize, const NR: usize>(
    kc: usize,
    a: *const T,
    b: *const T,
    ab: *mut T,
) {
    let mut acc_re = [[T::ZERO; MR]; NR];
    let mut acc_im = [[T::ZERO; MR]; NR];
    for p in 0..kc {
        let are = a.add(p * 2 * MR);
        let aim = are.add(MR);
        let bre = b.add(p * 2 * NR);
        let bim = bre.add(NR);
        for j in 0..NR {
            let br = *bre.add(j);
            let bi = *bim.add(j);
            for i in 0..MR {
                let ar = *are.add(i);
                let ai = *aim.add(i);
                acc_re[j][i] += ar * br;
                acc_re[j][i] += -(ai * bi);
                acc_im[j][i] += ar * bi;
                acc_im[j][i] += ai * br;
            }
        }
    }
    for j in 0..NR {
        for i in 0..MR {
            *ab.add(j * MR + i) = acc_re[j][i];
            *ab.add(MR * NR + j * MR + i) = acc_im[j][i];
        }
    }
}

/// Van Zee's 1m method: a *real* micro-kernel of shape `2*MR x NR` run over
/// `2*kc` real steps.
///
/// `MR2` is the real row count, so the complex micro-tile is `MR2/2 x NR`.
/// Operand A arrives in "1e" format (four reals per complex element) and B in
/// "1r" (two), which is exactly what makes one real product yield the complex
/// one. The tile comes out as a real `MR2 x NR` matrix whose row `2i` is the
/// real part and row `2i+1` the imaginary part of complex row `i`.
///
/// # Safety
/// `a` valid for `2 * MR2 * kc` reads, `b` for `2 * NR * kc`, `ab` for
/// `MR2 * NR` writes.
pub unsafe fn onem_ukr<T: Real, const MR2: usize, const NR: usize>(
    kc: usize,
    a: *const T,
    b: *const T,
    ab: *mut T,
) {
    // One logical (complex) k-step is two real k-steps in both panels.
    real_ukr::<T, MR2, NR>(2 * kc, a, b, ab)
}

/// Karatsuba 3m kernel.
///
/// `a` holds `[re, im, re+im]` planes of `MR` per k-step, `b` likewise with
/// `NR`. Writes three `MR x NR` planes:
///
/// ```text
/// M1 = Ar*Br     M2 = Ai*Bi     M3 = (Ar+Ai)*(Br+Bi)
/// ```
///
/// from which write-back forms `Cr = M1 - M2` and `Ci = M3 - M1 - M2`. Three
/// products instead of four: 25% fewer flops, at the cost of a third
/// accumulator plane and a weaker error bound.
///
/// # Safety
/// `a` valid for `3 * MR * kc` reads, `b` for `3 * NR * kc`, `ab` for
/// `3 * MR * NR` writes.
pub unsafe fn threem_ukr<T: Real, const MR: usize, const NR: usize>(
    kc: usize,
    a: *const T,
    b: *const T,
    ab: *mut T,
) {
    let mut m1 = [[T::ZERO; MR]; NR];
    let mut m2 = [[T::ZERO; MR]; NR];
    let mut m3 = [[T::ZERO; MR]; NR];
    for p in 0..kc {
        let are = a.add(p * 3 * MR);
        let aim = are.add(MR);
        let asum = are.add(2 * MR);
        let bre = b.add(p * 3 * NR);
        let bim = bre.add(NR);
        let bsum = bre.add(2 * NR);
        for j in 0..NR {
            let br = *bre.add(j);
            let bi = *bim.add(j);
            let bs = *bsum.add(j);
            for i in 0..MR {
                m1[j][i] += *are.add(i) * br;
                m2[j][i] += *aim.add(i) * bi;
                m3[j][i] += *asum.add(i) * bs;
            }
        }
    }
    let plane = MR * NR;
    for j in 0..NR {
        for i in 0..MR {
            *ab.add(j * MR + i) = m1[j][i];
            *ab.add(plane + j * MR + i) = m2[j][i];
            *ab.add(2 * plane + j * MR + i) = m3[j][i];
        }
    }
}

/// Real configuration at register block `MR x NR`.
///
/// `MR`/`NR` are used exactly as given — unlike [`config_cplx`], which halves
/// the row block to make room for the extra accumulator planes.
pub fn config_real<T: Real, const MR: usize, const NR: usize>() -> KernelConfig<T> {
    KernelConfig {
        ukr: Ukr {
            mr: MR,
            nr: NR,
            a_per_k: MR,
            b_per_k: NR,
            tile: MR * NR,
            a_pack: PackFormat::Real,
            b_pack: PackFormat::Real,
            tile_fmt: TileFormat::Real,
            func: real_ukr::<T, MR, NR>,
            name: "scalar-real",
        },
        blk: Blocking::derive(core::mem::size_of::<T>(), 1, 1),
    }
}

/// Complex configuration for one method.
///
/// `MR`/`NR` are the *real* register block. The complex micro-tile is smaller
/// because complex needs more accumulator state: half the rows for planar and
/// 1m (two planes, or a doubled real row count), and here also half for 3m so
/// that three planes fit. That shrinkage is inherent to complex arithmetic,
/// not to any one method.
pub fn config_cplx<T: Real, const MR: usize, const NR: usize>(
    method: ComplexMethod,
) -> KernelConfig<T> {
    let sz = core::mem::size_of::<T>();
    // Halve the row block for complex; keep at least one row.
    const fn half(x: usize) -> usize {
        if x >= 2 {
            x / 2
        } else {
            1
        }
    }
    match method {
        ComplexMethod::Planar => {
            const fn mr_of(mr: usize) -> usize {
                half(mr)
            }
            let _ = mr_of;
            KernelConfig {
                ukr: Ukr {
                    mr: half(MR),
                    nr: NR,
                    a_per_k: 2 * half(MR),
                    b_per_k: 2 * NR,
                    tile: 2 * half(MR) * NR,
                    a_pack: PackFormat::Planar,
                    b_pack: PackFormat::Planar,
                    tile_fmt: TileFormat::Planar,
                    func: planar_dispatch::<T, MR, NR>(),
                    name: "scalar-planar",
                },
                blk: Blocking::derive(sz, 2, 2),
            }
        }
        ComplexMethod::OneM => KernelConfig {
            ukr: Ukr {
                // The real kernel is MR x NR, so the complex tile is MR/2 x NR.
                mr: half(MR),
                nr: NR,
                a_per_k: 4 * half(MR),
                b_per_k: 2 * NR,
                tile: MR * NR,
                a_pack: PackFormat::OneE,
                b_pack: PackFormat::Planar,
                tile_fmt: TileFormat::OneM,
                func: onem_ukr::<T, MR, NR>,
                name: "scalar-1m",
            },
            // Packed A carries four reals per complex element, so the same L2
            // budget buys half the rows planar gets.
            blk: Blocking::derive(sz, 4, 2),
        },
        ComplexMethod::ThreeM => KernelConfig {
            ukr: Ukr {
                mr: half(MR),
                nr: NR,
                a_per_k: 3 * half(MR),
                b_per_k: 3 * NR,
                tile: 3 * half(MR) * NR,
                a_pack: PackFormat::ThreeM,
                b_pack: PackFormat::ThreeM,
                tile_fmt: TileFormat::ThreeM,
                func: threem_dispatch::<T, MR, NR>(),
                name: "scalar-3m",
            },
            blk: Blocking::derive(sz, 3, 3),
        },
    }
}

/// `planar_ukr` instantiated at half the real row block.
///
/// Const generics cannot do arithmetic in a type position on stable, so the
/// halving is spelled out for the block sizes actually used.
const fn planar_dispatch<T: Real, const MR: usize, const NR: usize>(
) -> unsafe fn(usize, *const T, *const T, *mut T) {
    match MR {
        2 => planar_ukr::<T, 1, NR>,
        4 => planar_ukr::<T, 2, NR>,
        8 => planar_ukr::<T, 4, NR>,
        16 => planar_ukr::<T, 8, NR>,
        32 => planar_ukr::<T, 16, NR>,
        _ => planar_ukr::<T, 1, NR>,
    }
}

const fn threem_dispatch<T: Real, const MR: usize, const NR: usize>(
) -> unsafe fn(usize, *const T, *const T, *mut T) {
    match MR {
        2 => threem_ukr::<T, 1, NR>,
        4 => threem_ukr::<T, 2, NR>,
        8 => threem_ukr::<T, 4, NR>,
        16 => threem_ukr::<T, 8, NR>,
        32 => threem_ukr::<T, 16, NR>,
        _ => threem_ukr::<T, 1, NR>,
    }
}
