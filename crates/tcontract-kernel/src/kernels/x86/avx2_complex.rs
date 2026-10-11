//! AVX2+FMA interleaved complex tile kernels.
//!
//! One vector holds adjacent complex pairs `[re, im, re, im, ...]`. For each k
//! step the signed swap `[-ai, ar, ...]` of each A vector is formed once and
//! reused for every B column; every accumulator then takes two FMAs:
//! `acc += a * br + swap(a) * bi`.

use crate::*;
use core::arch::x86_64::*;

/// Columns per tile for both dtypes (two vectors of A per column).
const NR: usize = 4;

macro_rules! native_tile {
    ($tile:ident, $inner:ident, $r:ty, $lanes:literal,
     $zero:ident, $load:ident, $store:ident, $fma:ident, $set1:ident, $xor:ident,
     $swap:expr, $sign:expr) => {
        #[target_feature(enable = "avx2,fma")]
        unsafe fn $inner(kc: usize, a: *const $r, b: *const $r, tile: *mut $r) {
            // SAFETY: the caller's panel/tile footprint contract (see `$tile`).
            unsafe {
                let sign = $sign;
                let mut acc = [[$zero(); 2]; NR];
                for p in 0..kc {
                    let ap = a.add(p * 2 * $lanes);
                    let bp = b.add(p * 2 * NR);
                    let a0 = $load(ap);
                    let a1 = $load(ap.add($lanes));
                    let s0 = $xor($swap(a0), sign);
                    let s1 = $xor($swap(a1), sign);
                    for (j, col) in acc.iter_mut().enumerate() {
                        let br = $set1(*bp.add(2 * j));
                        let bi = $set1(*bp.add(2 * j + 1));
                        col[0] = $fma(a0, br, col[0]);
                        col[1] = $fma(a1, br, col[1]);
                        col[0] = $fma(s0, bi, col[0]);
                        col[1] = $fma(s1, bi, col[1]);
                    }
                }
                for (j, col) in acc.iter().enumerate() {
                    let t = tile.add(j * 2 * $lanes);
                    $store(t, col[0]);
                    $store(t.add($lanes), col[1]);
                }
            }
        }

        /// Overwrite an interleaved complex tile with the packed product.
        ///
        /// # Safety
        /// AVX2 and FMA are available; A and B cover `2*MR*kc` and
        /// `2*NR*kc` readable reals (MR = 2 vectors of complex pairs), the
        /// tile covers `2*MR*NR` writable reals, and the tile does not alias
        /// either panel.
        unsafe fn $tile(kc: usize, a: *const $r, b: *const $r, tile: *mut $r) {
            // SAFETY: forwarded; the family is only usable when `required`
            // (AVX2+FMA) is present, which selection checks before any call.
            unsafe { $inner(kc, a, b, tile) }
        }
    };
}

native_tile!(
    tile_c64,
    tile_c64_avx2,
    f64,
    4,
    _mm256_setzero_pd,
    _mm256_loadu_pd,
    _mm256_storeu_pd,
    _mm256_fmadd_pd,
    _mm256_set1_pd,
    _mm256_xor_pd,
    |v| _mm256_permute_pd::<0b0101>(v),
    _mm256_set_pd(0.0, -0.0, 0.0, -0.0)
);
native_tile!(
    tile_c32,
    tile_c32_avx2,
    f32,
    8,
    _mm256_setzero_ps,
    _mm256_loadu_ps,
    _mm256_storeu_ps,
    _mm256_fmadd_ps,
    _mm256_set1_ps,
    _mm256_xor_ps,
    |v| _mm256_permute_ps::<0b10_11_00_01>(v),
    _mm256_set_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0)
);

const ORIGIN: Origin = Origin::Cplx;

macro_rules! family {
    ($r:ty, $id:literal, $mr:literal, $tile:ident, $mc:expr, $kc:expr, $nc:expr) => {
        KernelFamily::<$r> {
            id: $id,
            origin: ORIGIN,
            isa: Isa::Avx2,
            required: CpuFeatures {
                avx2: true,
                fma: true,
                ..CpuFeatures::NONE
            },
            imp: KernelImpl::Optimized,
            // Below the tensorcontract AVX2 families; irrelevant to Auto,
            // which skips every family here.
            priority: 100,
            complex: Some(ComplexScheme {
                method: Method::Native,
                a: Layout::Interleaved,
                b: Layout::Interleaved,
                tile: TileFormat::Interleaved,
            }),
            mr: $mr,
            nr: NR,
            a_per_k: 2 * $mr,
            b_per_k: 2 * NR,
            tile_bound: 2 * $mr * NR,
            c_pref: CPref::Col,
            ukr: UkrFn::Tile($tile),
            b_access: BAccess::Packed,
            c_update: CUpdate::ScratchTile,
            // Conservative seeds from the packed footprints (c64 4x4: a
            // 64x128 A block is 128 KiB), not tuned; the resolver derives the
            // working blocking from the cache model.
            blocks: Blocksizes {
                mc: $mc,
                kc: $kc,
                nc: $nc,
            },
            caps: Caps {
                scatter_pack: true,
                conj_a: true,
                conj_b: true,
            },
            opaque: core::ptr::null(),
            inner: None,
            allow_auto: false,
        }
    };
}

static C64: KernelFamily<f64> = family!(
    f64,
    "avx2.c64.native.4x4",
    4,
    tile_c64,
    (64, 256),
    (128, 512),
    (512, 4096)
);
static C32: KernelFamily<f32> = family!(
    f32,
    "avx2.c32.native.8x4",
    8,
    tile_c32,
    (64, 256),
    (256, 1024),
    (512, 4096)
);

pub(crate) fn families_f64() -> &'static [&'static KernelFamily<f64>] {
    static LIST: [&KernelFamily<f64>; 1] = [&C64];
    &LIST
}

pub(crate) fn families_f32() -> &'static [&'static KernelFamily<f32>] {
    static LIST: [&KernelFamily<f32>; 1] = [&C32];
    &LIST
}
