//! Isolated f64 OpenBLAS kernel-import experiment.
//!
//! Vendors OpenBLAS's `dgemm_kernel_16x2_skylakex` (commit
//! `31e82fa8c509e6f0d96288de3c20d8916d894e72`) as an external
//! [`KernelFamily`](tcontract_kernel::KernelFamily) for the tcontract packed driver.
//!
//! The kernel is a `MR=16 x NR=2` panel-panel product over column-major packed
//! panels, which matches tcontract's `PackFormat::Real` / `TileFormat::Real`
//! contract exactly: A is packed `A[i + MR*p]`, B is packed `B[j + NR*p]`, and
//! the tile is `C[j*MR + i]`. The only adapter needed is the scratch-tile
//! wrapper: zero the `MR x NR` tile, then call the OpenBLAS kernel with
//! `alpha = 1.0` and `ldc = MR` (the kernel always accumulates, so this yields
//! `A*B`). There is no beta parameter at this level; no packing or
//! contraction logic is reimplemented.
//!
//! On Zen 5 OpenBLAS selects this SkylakeX/Cooperlake f64 kernel (there is no
//! Zen-5-exclusive f64 kernel), which is why the label stays honest: it is the
//! kernel OpenBLAS actually maps to Zen 5, not a Zen 5 kernel.
//!
//! # Safety
//!
//! [`OPENBLAS_FAMILY`] is only admitted through
//! [`KernelCatalog::from_static_families`](tcontract_kernel::KernelCatalog::from_static_families),
//! which is the single `unsafe` trust boundary for external kernels. See
//! `PROVENANCE.md` for the source, license and the callable symbol.

use tcontract_kernel::{
    BAccess, Blocksizes, CPref, CUpdate, Caps, CpuFeatures, Isa, KernelCatalog, KernelFamily,
    KernelImpl, Origin, UkrFn,
};

/// Register rows of the micro-tile.
pub const MR: usize = 16;
/// Register columns of the micro-tile.
pub const NR: usize = 2;

extern "C" {
    /// OpenBLAS `dgemm_kernel_16x2_skylakex` (see `c/dgemm_kernel_16x2_skylakex.c`).
    ///
    /// `C = alpha * A * B + C` over an `m x n` column-major `C` (`ldc = m`).
    /// `A` is an `MR x k` column-major packed panel, `B` an `NR x k`
    /// column-major packed panel. `BLASLONG` is `int64_t` (see `c/common.h`).
    fn dgemm_kernel_16x2_skylakex(
        m: i64,
        n: i64,
        k: i64,
        alpha: f64,
        a: *const f64,
        b: *const f64,
        c: *mut f64,
        ldc: i64,
    ) -> i32;
}

/// The scratch-tile arm: overwrite `tile` (`MR x NR`, column-major) with
/// `A(16 x kc) * B(2 x kc)`.
///
/// # Safety
/// `a` addresses `MR * kc`, `b` addresses `NR * kc`, `tile` addresses `MR * NR`
/// reals; none may alias.
unsafe fn tile_ukr(kc: usize, a: *const f64, b: *const f64, tile: *mut f64) {
    // Zero the tile, then let the kernel accumulate `C += 1.0 * A*B` into it.
    // SAFETY: caller guarantees `tile` covers MR*NR reals and does not alias.
    unsafe { core::slice::from_raw_parts_mut(tile, MR * NR) }.fill(0.0);
    // SAFETY: caller satisfied the panel/tile extents above.
    unsafe {
        dgemm_kernel_16x2_skylakex(MR as i64, NR as i64, kc as i64, 1.0, a, b, tile, MR as i64);
    }
}

/// The OpenBLAS family descriptor, admitted through
/// [`catalog`]. `required` is AVX-512F: the kernel is inline assembly over
/// zmm registers (`vfmadd231pd`, `vbroadcastsd`, `vpermpd`, ...).
pub static OPENBLAS_FAMILY: KernelFamily<f64> = KernelFamily {
    id: "openblas.dgemm_kernel_16x2_skylakex.f64",
    origin: Origin::External {
        crate_name: "openblas-kernel",
        license: "BSD-3-Clause",
    },
    isa: Isa::Avx512,
    required: CpuFeatures {
        avx2: true,
        fma: true,
        avx512f: true,
        ..CpuFeatures::NONE
    },
    imp: KernelImpl::Optimized,
    priority: 2000,
    complex: None,
    mr: MR,
    nr: NR,
    a_per_k: MR,
    b_per_k: NR,
    tile_bound: MR * NR,
    c_pref: CPref::Col,
    ukr: UkrFn::Tile(tile_ukr),
    edge: None,
    b_access: BAccess::Packed,
    c_update: CUpdate::ScratchTile,
    blocks: Blocksizes {
        mc: (256, 256),
        kc: (256, 256),
        nc: (1536, 1536),
    },
    caps: Caps {
        scatter_pack: true,
        conj_a: false,
        conj_b: false,
    },
    opaque: core::ptr::null(),
    inner: None,
    allow_auto: false,
};

/// The wider OpenBLAS path, with explicit B-format adaptation.
pub static OPENBLAS_WIDE_FAMILY: KernelFamily<f64> = KernelFamily {
    id: "openblas.dgemm_kernel_16x12_skylakex.f64",
    nr: 12,
    b_per_k: 12,
    tile_bound: MR * 12,
    ukr: UkrFn::Tile(tile_ukr_wide),
    blocks: Blocksizes {
        mc: (256, 256),
        kc: (128, 128),
        nc: (1536, 1536),
    },
    ..OPENBLAS_FAMILY
};

unsafe fn tile_ukr_wide(kc: usize, a: *const f64, b: *const f64, tile: *mut f64) {
    // ponytail: bounded stack adapter for kc<=256; a native pair-panel pack
    // format would remove this copy if whole-call measurements justify it.
    const MAX_K: usize = 256;
    if kc > MAX_K {
        // SAFETY: the same validated 16 x kc, 12 x kc and 16 x 12 buffers.
        unsafe {
            tcontract_kernel::kernels::reference::portable::real_tile::<f64, MR, 12>(kc, a, b, tile)
        };
        return;
    }
    let mut pairs = [core::mem::MaybeUninit::<f64>::uninit(); MAX_K * 12];
    for group in 0..6 {
        for p in 0..kc {
            for j in 0..2 {
                // SAFETY: 12-wide input and exactly 12*kc initialized output
                // elements; OpenBLAS reads only this initialized prefix.
                pairs[group * 2 * kc + p * 2 + j].write(unsafe { *b.add(p * 12 + group * 2 + j) });
            }
        }
    }
    // SAFETY: exclusive scratch tile spans 16*12 elements.
    unsafe { core::slice::from_raw_parts_mut(tile, MR * 12) }.fill(0.0);
    // SAFETY: A is unchanged; B has six contiguous kc x 2 pair panels,
    // matching OpenBLAS's n12 path; C is column-major with ldc=16.
    unsafe {
        dgemm_kernel_16x2_skylakex(
            MR as i64,
            12,
            kc as i64,
            1.0,
            a,
            pairs.as_ptr().cast(),
            tile,
            MR as i64,
        )
    };
}

/// Wider tile product; includes B repacking and falls back above kc=256.
///
/// # Safety
/// `a` spans 16*kc, `b` spans 12*kc, and exclusive `tile` spans 16*12 f64s.
pub unsafe fn tile_product_wide(kc: usize, a: *const f64, b: *const f64, tile: *mut f64) {
    // SAFETY: caller supplies the documented panel/tile extents.
    unsafe { tile_ukr_wide(kc, a, b, tile) }
}

static FAMILIES: [&KernelFamily<f64>; 2] = [&OPENBLAS_FAMILY, &OPENBLAS_WIDE_FAMILY];

/// A single-family [`KernelCatalog`] holding [`OPENBLAS_FAMILY`].
///
/// # Panics
/// If the descriptor fails validation; it is checked at admission time so a
/// mistake is caught before any kernel runs.
pub fn catalog() -> KernelCatalog<f64> {
    // SAFETY: OPENBLAS_FAMILY is immutable and process-constant, its kernel
    // implements the declared `MR x NR` column-major panel product, requires
    // no more than AVX-512F, and never unwinds. See PROVENANCE.md.
    unsafe { KernelCatalog::from_static_families(&FAMILIES) }.expect("OpenBLAS family is valid")
}

/// The raw tile product, for the self-check and any direct caller: identical
/// to what the driver invokes, without packing or write-back.
///
/// # Safety
/// As [`tile_ukr`].
pub unsafe fn tile_product(kc: usize, a: *const f64, b: *const f64, tile: *mut f64) {
    // SAFETY: caller's obligation, forwarded.
    unsafe { tile_ukr(kc, a, b, tile) }
}
