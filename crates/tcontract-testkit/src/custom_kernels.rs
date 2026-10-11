//! A downstream crate's own packed microkernels.
//!
//! This module plays the part of a user of tcontract: it depends on the public
//! kernel contract only, implements its kernels itself, and never calls the
//! process-wide `register`. Its kernels are deliberately tiny scalar loops;
//! they exist to prove the selection, ownership and diagnostics contracts,
//! not to be fast.
//!
//! The ABI each kernel implements is the documented one of
//! [`tcontract_kernel::TileUkrFn`]: A packed `MR` reals per k-step, B packed
//! `NR` reals per k-step, and the complete column-major `MR x NR` tile
//! overwritten.
//!
//! What safe downstream code cannot do with a handle is pinned by the
//! `compile_fail` doctests of `tcontract_kernel::KernelCatalog`.
use tcontract_kernel::{
    portable, Axis, BAccess, Blocksizes, CPref, CUpdate, Caps, ComplexScheme, CpuFeatures, Isa,
    KernelFamily, KernelImpl, Layout, Method, Origin, TileFormat, UkrFn,
};

/// How this crate reports itself in diagnostics.
pub const ORIGIN: Origin = Origin::External {
    crate_name: "tcontract-testkit",
    license: "MIT OR Apache-2.0",
};

/// `tile[j * MR + i] = sum_p a[p * MR + i] * b[p * NR + j]`, overwriting the
/// whole tile.
///
/// # Safety
/// `a` covers `kc * MR` reals, `b` covers `kc * NR` reals and `tile` covers
/// `MR * NR` writable reals not aliasing either.
unsafe fn tile_kernel<R, const MR: usize, const NR: usize>(
    kc: usize,
    a: *const R,
    b: *const R,
    tile: *mut R,
) where
    R: tcontract_kernel::Real,
{
    // SAFETY: the function's extent contract.
    unsafe {
        for j in 0..NR {
            for i in 0..MR {
                let mut sum = R::ZERO;
                for p in 0..kc {
                    sum += *a.add(p * MR + i) * *b.add(p * NR + j);
                }
                *tile.add(j * MR + i) = sum;
            }
        }
    }
}

macro_rules! real_family {
    ($name:ident, $r:ty, $id:literal, $mr:literal, $nr:literal, $mc:literal) => {
        static $name: KernelFamily<$r> = KernelFamily {
            id: $id,
            origin: ORIGIN,
            isa: Isa::Portable,
            required: CpuFeatures::NONE,
            imp: KernelImpl::Reference,
            priority: 0,
            complex: None,
            mr: $mr,
            nr: $nr,
            a_per_k: $mr,
            b_per_k: $nr,
            tile_bound: $mr * $nr,
            c_pref: CPref::Any,
            ukr: UkrFn::Tile(tile_kernel::<$r, $mr, $nr>),
            b_access: BAccess::Packed,
            c_update: CUpdate::ScratchTile,
            blocks: Blocksizes {
                mc: ($mc, $mc),
                kc: (256, 256),
                nc: (1024, 1024),
            },
            caps: Caps {
                scatter_pack: true,
                conj_a: true,
                conj_b: true,
            },
            opaque: core::ptr::null(),
            inner: None,
            allow_auto: false,
        };
    };
}

macro_rules! complex_family {
    ($name:ident, $real:ident, $r:ty, $id:literal, $mr:literal, $nr:literal) => {
        static $name: KernelFamily<$r> = KernelFamily {
            id: $id,
            complex: Some(ComplexScheme {
                method: Method::Native,
                a: Layout::Interleaved,
                b: Layout::Interleaved,
                tile: TileFormat::Interleaved,
            }),
            a_per_k: 2 * $mr,
            b_per_k: 2 * $nr,
            tile_bound: 2 * $mr * $nr,
            // The project's reference loop for the interleaved layout; the
            // descriptor, id and provenance are this crate's own.
            ukr: UkrFn::Tile(portable::cplx_tile::<$r, $mr, $nr>),
            ..$real
        };
    };
}

real_family!(F64_2X2, f64, "custom.f64.2x2", 2, 2, 64);
real_family!(F64_3X4, f64, "custom.f64.3x4", 3, 4, 63);
real_family!(F32_2X2, f32, "custom.f32.2x2", 2, 2, 64);
complex_family!(C64_2X2, F64_2X2, f64, "custom.c64.native.2x2", 2, 2);
complex_family!(C32_2X2, F32_2X2, f32, "custom.c32.native.2x2", 2, 2);

/// Direct-update kernel reading B in place, built from the contract's public
/// reference loop, so the direct-B guards run under a downstream descriptor.
static F64_DIRECT_B: KernelFamily<f64> = KernelFamily {
    id: "custom.f64.4x4.direct-b",
    mr: 4,
    nr: 4,
    a_per_k: 4,
    b_per_k: 4,
    tile_bound: 16,
    ukr: UkrFn::Direct(portable::real_direct::<f64, 4, 4>),
    b_access: BAccess::Direct {
        unit_stride: Axis::Col,
    },
    c_update: CUpdate::Direct,
    blocks: Blocksizes {
        mc: (64, 64),
        kc: (256, 256),
        nc: (1024, 1024),
    },
    ..F64_2X2
};

/// Two real f64 kernels with different tile shapes.
pub fn f64_families() -> &'static [&'static KernelFamily<f64>] {
    static LIST: [&KernelFamily<f64>; 2] = [&F64_2X2, &F64_3X4];
    &LIST
}
/// A direct-update f64 kernel that reads B in place when it can.
pub fn f64_direct_families() -> &'static [&'static KernelFamily<f64>] {
    static LIST: [&KernelFamily<f64>; 1] = [&F64_DIRECT_B];
    &LIST
}
/// One real f32 kernel.
pub fn f32_families() -> &'static [&'static KernelFamily<f32>] {
    static LIST: [&KernelFamily<f32>; 1] = [&F32_2X2];
    &LIST
}
/// One complex (`Complex<f64>`) kernel in the interleaved layout.
pub fn c64_families() -> &'static [&'static KernelFamily<f64>] {
    static LIST: [&KernelFamily<f64>; 1] = [&C64_2X2];
    &LIST
}
/// One complex (`Complex<f32>`) kernel in the interleaved layout.
pub fn c32_families() -> &'static [&'static KernelFamily<f32>] {
    static LIST: [&KernelFamily<f32>; 1] = [&C32_2X2];
    &LIST
}

/// A kernel whose truthful ISA requirement no CPU meets (AVX2 and NEON), for
/// testing that a masked-out choice is rejected before any compute.
static F64_IMPOSSIBLE: KernelFamily<f64> = KernelFamily {
    id: "custom.f64.impossible",
    required: CpuFeatures {
        avx2: true,
        neon: true,
        ..CpuFeatures::NONE
    },
    isa: Isa::Avx2,
    ..F64_2X2
};
/// The never-runnable kernel, alone.
pub fn f64_impossible_families() -> &'static [&'static KernelFamily<f64>] {
    static LIST: [&KernelFamily<f64>; 1] = [&F64_IMPOSSIBLE];
    &LIST
}
