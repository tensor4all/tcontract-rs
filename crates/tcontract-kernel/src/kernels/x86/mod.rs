//! x86-64 vectorised micro-kernels.
//!
//! **Not part of the public API.** This module is `#[doc(hidden)]` and outside
//! the crate's semver guarantee; it is public only so that
//! `examples/kernel_shapes` can measure one kernel family at a time. The
//! register-block menus below are re-measured whenever the reference machine
//! changes, so nothing here is stable. Reach the kernels through
//! registered families.
//!
//! These honour exactly the same [`PackFormat`](crate::PackFormat) / [`TileFormat`](crate::TileFormat) contract as
//! [`scalar`](super::reference::scalar), which is what keeps the three complex methods
//! interchangeable: the driver, the packing traversal and the write-back do not
//! know which kernel they are running.
//!
//! # Register blocking
//!
//! Write `MV` for the number of vector registers an `A` sliver occupies per
//! plane per k-step, `L` for the lane count (8 for AVX-512 `f64`, 16 for
//! AVX-512 `f32`; 4 and 8 for AVX2) and `NR` for the column block. Then per
//! logical k-step a kernel issues
//!
//! | | accumulator registers | load-port uops | FMA uops |
//! |---|---|---|---|
//! | real | `MV*NR` | `MV + NR` | `MV*NR` |
//! | planar | `2*MV*NR` | `2*MV + 2*NR` | `4*MV*NR` |
//! | 1m | `MV*NR` (real `2*MR x NR`) | `2*(MV + NR)` | `2*MV*NR` |
//! | 3m | `3*MV*NR` | `3*MV + 3*NR` | `3*MV*NR` |
//!
//! Skylake-SP retires two 512-bit FMAs and two loads per cycle, and LLVM emits
//! `vbroadcastsd` as its own uop rather than folding it into an FMA memory
//! operand (checked in the disassembly), so a kernel is issue-bound on FMAs
//! exactly when `FMA > loads`. Two things then decide the shape, and they pull
//! in opposite directions:
//!
//! 1. **The register file.** Accumulators plus one A plane plus the live
//!    broadcasts must fit in the architectural register file — 32 zmm under
//!    AVX-512, **16 ymm under AVX2**. Every candidate that does not spills and
//!    loses 30–50% — the sweep is full of these cliffs, and they are the reason
//!    3m cannot simply be given a big block.
//! 2. **Bytes per useful flop.** Once a shape is issue-bound the A sliver is a
//!    stream out of L2, not an L1 resident, and the packed footprint per flop
//!    becomes the binding constraint.
//!
//! The second point is the measured story, and it is not the one the flop
//! counts predict:
//!
//! * **planar** has the lowest bytes-per-flop of the three (two reals per
//!   complex element in *both* panels, and four FMAs from every (A-vector,
//!   B-scalar) pair with no in-register shuffling). It wins the corpus on both
//!   AVX-512 machines measured.
//! * **1m** issues the same FMAs but spread over two real k-steps and reads a
//!   packed `A` panel twice the size — 1.5x planar's bytes per flop.
//! * **3m** does 25% *fewer* FMAs and still loses, because it loads three
//!   planes of both operands to do it.
//!
//! The accounting above — bytes moved per useful flop, not flop count — is
//! arithmetic and holds everywhere. **Whether 3m's flop saving ever *pays* is a
//! per-microarchitecture question with no settled answer.** This module used to
//! claim that it pays with both panels L1-resident; that is a Cascade Lake
//! result and it does not transfer. At `kc = 16`, which is that regime, 3m leads
//! planar by 1.105 (`c64`) and 1.155 (`c32`) on Cascade Lake and trails at 0.567
//! and 0.662 on Ice Lake, where it wins at no depth, no shape and neither
//! precision while already running its own Ice Lake-optimal block. Ranking
//! anything below planar without naming the machine is a mistake this project
//! has made twice.
//!
//! The **AVX-512** shapes were chosen by measuring, not by the reasoning above:
//! the retired `kernel_shapes` example (see `docs/provenance.md`), raw output
//! in `tensorprimitives/bench-results/phase3-kernel-shapes.txt`.
//!
//! The **AVX2** shapes are *provisional and unmeasured* — see
//! [`cfg_avx2_f64`] — because the reference machine has no AVX2-only CPU to
//! measure them on and the same `examples/kernel_shapes` run is the calibration
//! path once one is available. Halving the register file is not a small change:
//! with 16 ymm, planar's two accumulator planes and 3m's three leave almost no
//! choice of aspect ratio, so AVX2 register blocks are 2–6x smaller than their
//! AVX-512 counterparts and are much closer to the register bound.

use crate::{ComplexMethod, KernelConfig};

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2_complex;

mod avx2;
mod avx512;

pub use avx2::{avx2_f32, avx2_f64, cfg_avx2_f32, cfg_avx2_f64};
pub use avx512::{avx512_f32, avx512_f64, cfg_avx512_f32, cfg_avx512_f64};

// ---------------------------------------------------------------------------
// Runtime dispatch
// ---------------------------------------------------------------------------

/// An instruction set this file has kernels for, widest first.
///
/// Not a CPU-feature bitset: it names a *kernel family*, i.e. one instantiation
/// of `simd_kernels!` plus the register-block menus that go with it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isa {
    /// AVX-512F. Measured; the reference machine.
    Avx512,
    /// AVX2 + FMA3. Provisional register blocks — see [`cfg_avx2_f64`].
    Avx2,
}

impl Isa {
    /// The name used in test failure messages and in `tcbench info` output.
    pub fn name(self) -> &'static str {
        match self {
            Isa::Avx512 => "avx512",
            Isa::Avx2 => "avx2",
        }
    }
}

/// Whether AVX-512F is usable. Checked once per process.
fn have_avx512() -> bool {
    #[cfg(feature = "std")]
    {
        use std::sync::OnceLock;
        static YES: OnceLock<bool> = OnceLock::new();
        *YES.get_or_init(|| is_x86_feature_detected!("avx512f"))
    }
    #[cfg(not(feature = "std"))]
    {
        // No runtime detection without `std`; take it only if the whole crate
        // was compiled for it.
        cfg!(target_feature = "avx512f")
    }
}

/// Whether AVX2 *and* FMA3 are usable. Checked once per process.
///
/// Both bits are required and neither implies the other in CPUID, even though
/// no shipping CPU has one without the other: the kernels' loads and broadcasts
/// are AVX/AVX2 and their `vfmadd`/`vfnmadd` are FMA3.
fn have_avx2() -> bool {
    #[cfg(feature = "std")]
    {
        use std::sync::OnceLock;
        static YES: OnceLock<bool> = OnceLock::new();
        *YES.get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
    }
    #[cfg(not(feature = "std"))]
    {
        cfg!(all(target_feature = "avx2", target_feature = "fma"))
    }
}

/// Every ISA whose kernels this CPU can execute, widest first, **ignoring
/// any pinned [`crate::KernelForce`]**.
///
/// Dispatch uses [`selected_isa`]; this exists for the kernel-contract tests,
/// which must check every kernel the machine can run rather than only the one
/// it would choose. That is the only way the AVX2 kernels get exercised at all
/// on an AVX-512 reference machine under a plain `cargo test`.
pub fn available_isas() -> &'static [Isa] {
    match (have_avx512(), have_avx2()) {
        (true, true) => &[Isa::Avx512, Isa::Avx2],
        (true, false) => &[Isa::Avx512],
        (false, true) => &[Isa::Avx2],
        (false, false) => &[],
    }
}

/// The ISA the engine will dispatch to under `force`, or `None` for the portable
/// scalar path. The feature detection it wraps is cached for the process.
///
/// `force` overrides the choice: `scalar` takes the portable
/// path, `avx2` and `avx512` pin a family. A pinned family the CPU cannot run
/// falls through to scalar rather than faulting — pinning is a testing and A/B
/// facility, not a promise that the hardware exists.
pub fn selected_isa(force: crate::KernelForce) -> Option<Isa> {
    use crate::KernelForce;
    match force {
        KernelForce::Scalar => None,
        KernelForce::Avx2 => have_avx2().then_some(Isa::Avx2),
        KernelForce::Avx512 => have_avx512().then_some(Isa::Avx512),
        // Pinning NEON on x86 is a request this target cannot honour, and
        // falling through to scalar is what pinning an absent family already
        // does here. `kernel::aarch64` declines `avx2`/`avx512` the same way, so
        // one sweep script can pass the same arm list to every machine.
        KernelForce::Neon => None,
        // Widest first. AVX-512 is measured and AVX2 is not, so there is no
        // shape-dependent choice between them to make here.
        KernelForce::Auto => available_isas().first().copied(),
    }
}

/// One ISA's five kernel-set entry points for one element type.
///
/// A table of function pointers rather than five `match isa` statements per
/// type, because everything that wants this wants *all* of it at once: dispatch
/// wants the selected ISA's five, and the contract tests want each available
/// ISA's five in turn.
#[derive(Clone, Copy)]
pub struct IsaConfigs<T> {
    pub isa: Isa,
    pub real: fn() -> KernelConfig<T>,
    pub cplx: fn(ComplexMethod) -> KernelConfig<T>,
    pub row_blocks: fn(bool, ComplexMethod) -> &'static [(usize, usize)],
    pub config_at: fn(bool, ComplexMethod, usize) -> Option<KernelConfig<T>>,
}

/// Written by hand for two reasons: the derive would demand `T: Debug` for a
/// struct that never stores a `T`, and the other five fields are function
/// pointers whose addresses tell a reader nothing. The `isa` *is* the identity of
/// the table.
impl<T> core::fmt::Debug for IsaConfigs<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("IsaConfigs")
            .field("isa", &self.isa)
            .finish()
    }
}

/// The five entry points the default menu is built from, per type: the
/// default shape, the menu of row blocks, and the config at a chosen one, each
/// resolved through [`selected_isa`]. All yield `None`/`&[]` when no vectorised
/// ISA is available or `force` is `Scalar`, which sends the
/// caller to the portable scalar path.
macro_rules! dispatch {
    ($t:ty, $cfg512:ident, $cfg2:ident, $sets:ident,
     $real:ident, $cplx:ident, $rows:ident, $real_at:ident, $cplx_at:ident) => {
        /// This type's kernel set for a named ISA, whether or not the CPU has it.
        pub fn $sets(isa: Isa) -> IsaConfigs<$t> {
            match isa {
                Isa::Avx512 => IsaConfigs {
                    isa,
                    real: $cfg512::real_config,
                    cplx: $cfg512::cplx_config,
                    row_blocks: $cfg512::row_blocks,
                    config_at: $cfg512::config_at,
                },
                Isa::Avx2 => IsaConfigs {
                    isa,
                    real: $cfg2::real_config,
                    cplx: $cfg2::cplx_config,
                    row_blocks: $cfg2::row_blocks,
                    config_at: $cfg2::config_at,
                },
            }
        }

        pub fn $real(force: crate::KernelForce) -> Option<KernelConfig<$t>> {
            let s = $sets(selected_isa(force)?);
            Some((s.real)())
        }

        pub fn $cplx(force: crate::KernelForce, method: ComplexMethod) -> Option<KernelConfig<$t>> {
            let s = $sets(selected_isa(force)?);
            Some((s.cplx)(method))
        }

        /// Row blocks with a kernel, default first; empty when unavailable.
        pub fn $rows(
            force: crate::KernelForce,
            complex: bool,
            method: ComplexMethod,
        ) -> &'static [(usize, usize)] {
            match selected_isa(force) {
                Some(isa) => ($sets(isa).row_blocks)(complex, method),
                None => &[],
            }
        }

        pub fn $real_at(force: crate::KernelForce, mr: usize) -> Option<KernelConfig<$t>> {
            let s = $sets(selected_isa(force)?);
            (s.config_at)(false, ComplexMethod::Planar, mr)
        }

        pub fn $cplx_at(
            force: crate::KernelForce,
            method: ComplexMethod,
            mr: usize,
        ) -> Option<KernelConfig<$t>> {
            let s = $sets(selected_isa(force)?);
            (s.config_at)(true, method, mr)
        }
    };
}

dispatch!(
    f64,
    cfg_avx512_f64,
    cfg_avx2_f64,
    isa_configs_f64,
    config_real_f64,
    config_cplx_f64,
    row_blocks_f64,
    config_real_f64_at,
    config_cplx_f64_at
);
dispatch!(
    f32,
    cfg_avx512_f32,
    cfg_avx2_f32,
    isa_configs_f32,
    config_real_f32,
    config_cplx_f32,
    row_blocks_f32,
    config_real_f32_at,
    config_cplx_f32_at
);
