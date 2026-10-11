//! Built-in kernel families and their registration order.
//!
//! The built-in set is present without any caller registration step: the
//! portable reference families first, then Lukas Devos's tensorcontract menus
//! (descriptor adapters that call the existing configs and bodies from
//! lkdvos/tensorprimitives-rs, MIT OR Apache-2.0; no arithmetic is duplicated
//! here), then the native complex families. The order is the tie-break of
//! equal priorities, so it is part of the contract and pinned by tests.

use crate::*;
use std::sync::OnceLock;

// The shared micro-kernel bodies. **Must be declared before the ISA modules
// that invoke them**: `#[macro_use]` makes `macro_rules!` visible to items that
// follow it textually, not to the module graph, so moving this line below
// `mod x86` breaks the build with a bare "cannot find macro".
//
// Gated on the union of its two consumers below. The module defines macros and
// nothing else, so on a target that is neither x86 nor aarch64 nothing invokes
// them and `unused_macros` fires, which under CI's `-D warnings` is a hard
// error. Keep this cfg equal to the disjunction of the `x86` and `aarch64`
// cfgs; a new ISA module must be added here as well as below.
#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
#[macro_use]
mod macros;

#[doc(hidden)]
pub mod cplx;
#[cfg(test)]
mod menu_tests;
pub mod reference;

// The x86-64 SIMD kernels: public for `examples/kernel_shapes`, `doc(hidden)`
// and outside the semver guarantee. See the module's own docs.
#[doc(hidden)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86;

// The AArch64/NEON kernels, on the same terms.
#[doc(hidden)]
#[cfg(target_arch = "aarch64")]
pub mod aarch64;

#[cfg(target_arch = "aarch64")]
use aarch64 as simd_isa;
/// The vectorised kernel module for this target, under one name.
///
/// Exists so the legacy menu and the dead-code guards name *a* SIMD module
/// rather than enumerating architectures at every site. Both modules expose the
/// same entry points by construction, so the alias is total.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use x86 as simd_isa;

/// The instruction set whose built-in families form the default menu under an
/// ISA preference: the widest one the CPU supports (or the pinned one), the
/// portable kernels when the preference selects no vectorised module.
///
/// The contraction planner builds its default family menu from this (the
/// families of this ISA, in registry order, default first).
pub fn menu_isa(force: KernelForce) -> Isa {
    if force == KernelForce::Scalar {
        return Isa::Portable;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    return match x86::selected_isa(force) {
        Some(x86::Isa::Avx2) => Isa::Avx2,
        Some(x86::Isa::Avx512) => Isa::Avx512,
        None => Isa::Portable,
    };
    #[cfg(target_arch = "aarch64")]
    return Isa::Neon;
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    Isa::Portable
}

type List<R> = Vec<&'static KernelFamily<R>>;

/// The id of a built-in tensorcontract-menu family:
/// `{isa}.{dtype}.{scheme}.{MR}x{NR}` from the menu label (`scalar`, `avx2`,
/// `avx512`, `neon`), the storage dtype, the tile format and the logical tile.
///
/// The scalar menu is the `ref` ISA, and its real family is `real-scalar`, so
/// it stays distinct from the portable `ref.*.real` family of the same
/// geometry; the SIMD menus' real families are `real`.
fn builtin_id(label: &str, dtype: &str, tile: TileFormat, mr: usize, nr: usize) -> String {
    let isa = if label == "scalar" { "ref" } else { label };
    let scheme = match (tile, label) {
        (TileFormat::Real, "scalar") => "real-scalar",
        (TileFormat::Real, _) => "real",
        (TileFormat::Planar, _) => "planar",
        (TileFormat::OneM, _) => "1m",
        (TileFormat::ThreeM, _) => "3m",
        (TileFormat::Interleaved, _) => "native",
        (TileFormat::FourM, _) => "4m",
    };
    format!("{isa}.{dtype}.{scheme}.{mr}x{nr}")
}

#[cfg(test)]
mod id_tests {
    use super::*;

    #[test]
    fn built_in_ids_follow_isa_dtype_scheme_geometry() {
        let id = builtin_id;
        assert_eq!(
            id("avx2", "f64", TileFormat::Real, 8, 6),
            "avx2.f64.real.8x6"
        );
        assert_eq!(
            id("avx512", "c32", TileFormat::OneM, 8, 12),
            "avx512.c32.1m.8x12"
        );
        assert_eq!(
            id("scalar", "f64", TileFormat::Real, 4, 4),
            "ref.f64.real-scalar.4x4"
        );
        assert_eq!(
            id("scalar", "c64", TileFormat::Planar, 2, 4),
            "ref.c64.planar.2x4"
        );
        // The NEON menus cannot be listed on an x86 host; their names come from
        // the same function.
        assert_eq!(
            id("neon", "f64", TileFormat::Real, 16, 3),
            "neon.f64.real.16x3"
        );
        assert_eq!(
            id("neon", "c64", TileFormat::ThreeM, 2, 8),
            "neon.c64.3m.2x8"
        );
        assert_eq!(
            id("neon", "c32", TileFormat::Planar, 8, 6),
            "neon.c32.planar.8x6"
        );
    }
}

fn append_config<R: RealSlot>(
    out: &mut List<R>,
    label: &str,
    isa: Isa,
    required: CpuFeatures,
    priority: u16,
    cfg: KernelConfig<R>,
) {
    // Configs are process-constant compiled menus; effective-width blocking
    // is resolved separately when a plan selects a descriptor.
    // Descriptor defaults are unscaled. Resolution applies environment/model
    // overrides once; aligning the finite compiled menu needs no env lookup.
    let cfg = cfg.with_blocking(cfg.blk);
    let u = cfg.ukr;
    let complex = match u.tile_fmt {
        TileFormat::Real => None,
        TileFormat::Planar => Some(ComplexScheme {
            method: Method::Native,
            a: Layout::Planar,
            b: Layout::Planar,
            tile: u.tile_fmt,
        }),
        TileFormat::OneM => Some(ComplexScheme {
            method: Method::OneM,
            a: Layout::OneE,
            b: Layout::OneR,
            tile: u.tile_fmt,
        }),
        TileFormat::ThreeM => Some(ComplexScheme {
            method: Method::ThreeM,
            a: Layout::ThreeM,
            b: Layout::ThreeM,
            tile: u.tile_fmt,
        }),
        // These formats are not offered by the existing tc menus.
        TileFormat::Interleaved => Some(ComplexScheme {
            method: Method::Native,
            a: Layout::Interleaved,
            b: Layout::Interleaved,
            tile: u.tile_fmt,
        }),
        TileFormat::FourM => Some(ComplexScheme {
            method: Method::FourM,
            a: Layout::Planar,
            b: Layout::Planar,
            tile: u.tile_fmt,
        }),
    };
    // INVARIANT: RealSlot is sealed to f32/f64; ids name the storage dtype.
    let dtype = match (core::mem::size_of::<R>(), complex.is_some()) {
        (4, false) => "f32",
        (4, true) => "c32",
        (_, false) => "f64",
        (_, true) => "c64",
    };
    let id = builtin_id(label, dtype, u.tile_fmt, u.mr, u.nr);
    let f = KernelFamily {
        id: Box::leak(id.into_boxed_str()),
        origin: Origin::Tensorcontract,
        isa,
        required,
        imp: if isa == Isa::Portable {
            KernelImpl::Reference
        } else {
            KernelImpl::Optimized
        },
        priority: if u.tile_fmt == TileFormat::OneM {
            priority.saturating_sub(50)
        } else {
            priority
        },
        complex,
        mr: u.mr,
        nr: u.nr,
        a_per_k: u.a_per_k,
        b_per_k: u.b_per_k,
        tile_bound: u.tile,
        c_pref: CPref::Col,
        ukr: UkrFn::Tile(u.func),
        b_access: BAccess::Packed,
        c_update: CUpdate::ScratchTile,
        blocks: Blocksizes {
            mc: (cfg.blk.mc, cfg.blk.mc),
            kc: (cfg.blk.kc, cfg.blk.kc),
            nc: (cfg.blk.nc, cfg.blk.nc),
        },
        caps: Caps {
            scatter_pack: true,
            conj_a: true,
            conj_b: true,
        },
        opaque: core::ptr::null(),
        inner: None,
        allow_auto: u.tile_fmt != TileFormat::ThreeM,
    };
    // INVARIANT: this immutable descriptor set is finite (compiled menu
    // entries only), initialized once and retained for the process lifetime.
    out.push(Box::leak(Box::new(f)));
}

fn scalar_families<R: RealSlot>() -> List<R> {
    let mut out = Vec::new();
    append_config(
        &mut out,
        "scalar",
        Isa::Portable,
        CpuFeatures::NONE,
        10,
        reference::scalar::config_real::<R, 4, 4>(),
    );
    for method in ComplexMethod::ALL {
        append_config(
            &mut out,
            "scalar",
            Isa::Portable,
            CpuFeatures::NONE,
            10,
            reference::scalar::config_cplx::<R, 4, 4>(method),
        );
    }
    out
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
fn append_menu<R: RealSlot>(
    out: &mut List<R>,
    label: &str,
    isa: Isa,
    required: CpuFeatures,
    priority: u16,
    menu: simd_isa::IsaConfigs<R>,
) {
    for (complex, method) in [
        (false, ComplexMethod::Planar),
        (true, ComplexMethod::Planar),
        (true, ComplexMethod::OneM),
        (true, ComplexMethod::ThreeM),
    ] {
        for i in 0..(menu.row_blocks)(complex, method).len() {
            // INVARIANT: indices come from this compiled menu; existing
            // row_block_menus_are_well_formed tests cover config_at coverage.
            let cfg = (menu.config_at)(complex, method, i).expect("compiled menu has a config");
            append_config(
                out,
                label,
                isa,
                required,
                priority.saturating_sub(u16::try_from(i).unwrap_or(u16::MAX)),
                cfg,
            );
        }
    }
}

macro_rules! family_list {
    ($r:ty, $name:ident, $sets:ident) => {
        /// All compiled real/complex tensorcontract families, including
        /// CPU-unavailable ISAs. Descriptors are a finite process-constant
        /// manifest, not a data cache.
        fn $name() -> &'static [&'static KernelFamily<$r>] {
            static LIST: OnceLock<List<$r>> = OnceLock::new();
            LIST.get_or_init(|| {
                #[allow(unused_mut)] // INVARIANT: SIMD targets append to the scalar list below.
                let mut out = scalar_families::<$r>();
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    append_menu(
                        &mut out,
                        "avx512",
                        Isa::Avx512,
                        CpuFeatures {
                            avx512f: true,
                            ..CpuFeatures::NONE
                        },
                        300,
                        simd_isa::$sets(simd_isa::Isa::Avx512),
                    );
                    append_menu(
                        &mut out,
                        "avx2",
                        Isa::Avx2,
                        CpuFeatures {
                            avx2: true,
                            fma: true,
                            ..CpuFeatures::NONE
                        },
                        200,
                        simd_isa::$sets(simd_isa::Isa::Avx2),
                    );
                }
                #[cfg(target_arch = "aarch64")]
                append_menu(
                    &mut out,
                    "neon",
                    Isa::Neon,
                    CpuFeatures {
                        neon: true,
                        ..CpuFeatures::NONE
                    },
                    200,
                    simd_isa::$sets(simd_isa::Isa::Neon),
                );
                out
            })
        }
    };
}
family_list!(f32, families_f32, isa_configs_f32);
family_list!(f64, families_f64, isa_configs_f64);

macro_rules! builtin_list {
    ($r:ty, $name:ident, $reference:path, $tc:ident, $cplx:path) => {
        /// The built-in families for one real type, in registration order:
        /// portable, tensorcontract menus, native complex.
        pub(crate) fn $name() -> &'static [&'static KernelFamily<$r>] {
            static LIST: OnceLock<Vec<&'static KernelFamily<$r>>> = OnceLock::new();
            LIST.get_or_init(|| {
                $reference()
                    .iter()
                    .chain($tc().iter())
                    .chain($cplx().iter())
                    .copied()
                    .collect()
            })
        }
    };
}
builtin_list!(
    f32,
    builtin_f32,
    reference::portable::families_f32,
    families_f32,
    cplx::families_f32
);
builtin_list!(
    f64,
    builtin_f64,
    reference::portable::families_f64,
    families_f64,
    cplx::families_f64
);
