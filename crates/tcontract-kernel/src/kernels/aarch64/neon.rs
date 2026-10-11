//! AArch64 NEON vectorised micro-kernels.
//!
//! **Not part of the public API.** `#[doc(hidden)]` and outside the semver
//! guarantee, on the same terms as `kernel::x86`: public only so that
//! `examples/kernel_shapes` can measure one kernel family at a time, and the
//! register-block menus are re-measured whenever the machine changes. Reach the
//! kernels through the registered families.
//!
//! The bodies are not written here. They come from `simd_kernels!` in
//! `kernel::simd`, the same macro that generates the AVX-512 and AVX2 kernels,
//! so a NEON-vs-AVX comparison is not also a comparison between two people's
//! hand-written kernels (D17).
//!
//! # Why this exists, and what it is worth
//!
//! Not because the portable path was unvectorised — **it was already
//! vectorised.** LLVM compiles `scalar::real_ukr`'s 4x4 tile into eight
//! `float64x2_t` accumulators with `ld1r.2d` broadcasts and lane-indexed
//! `fmul.2d`, unasked. What it will not emit is `fmla`, because `acc += a * b`
//! is two roundings and IEEE forbids contracting them. So the portable path
//! spends **two instructions per multiply-accumulate where the machine offers
//! one**, which halves the ceiling available to it, and that single instruction
//! form is most of the gap to a tuned baseline (A58).
//!
//! The consequence for this module: its headline win is **FMA contraction, worth
//! about 2x, not 10x**, plus whatever better register blocking buys — the scalar
//! tile uses 8 of 32 vector registers. Do not expect more, and do not justify
//! work here with the older "no vectorised kernel off x86" claim, which was true
//! of the source and false of the binary.
//!
//! # Register blocking
//!
//! NEON is **32 x 128-bit** registers: `L = 2` lanes for `f64`, `4` for `f32`.
//! Narrower than AVX2's 4 and AVX-512's 8, so **no shape transfers from either**
//! — the menus in `x86.rs` are not a starting point. The budget is the same
//! accounting `x86`'s module docs set out, `live = accumulators + A planes +
//! broadcasts <= 32`:
//!
//! | method | `live` | candidates (`MV x NR`) → logical `MR x NR`, `f64` |
//! |---|---|---|
//! | real | `MV*NR + MV + 1` | `(3,8)` → `6x8`, `(4,6)` → `8x6`, `(2,10)` → `4x10` |
//! | planar | `2*MV*NR + 2*MV + 2` | `(2,6)` → `4x6`, `(3,4)` → `6x4`, `(1,12)` → `2x12` |
//! | 1m | `MV*NR + MV + 1` | `(3,8)` → `6x8` complex `3x8`… (real block, halved) |
//! | 3m | `3*MV*NR + MV + 1` | `(1,8)` → `2x8`, `(2,4)` → `4x4`, `(3,3)` → `6x3` |
//!
//! **The two leading `real` candidates are exactly the shapes BLIS's own AArch64
//! `dgemm` kernels use** — `bli_dgemm_armv8a_asm_6x8` and `8x6r` — and the `f32`
//! ones land on its `12x8r` and `8x12`. That is independent corroboration that
//! the budget arithmetic is right, from a library that measured it; it is *not* a
//! measurement of this engine's kernels.
//!
//! # The shipped shapes are MEASURED, and the budget did not pick them
//!
//! Three arms of `examples/kernel_shapes` on an M3 Max
//! (`tensorprimitives/bench-results/CKF6QCDVPD-m3max/kernel-shapes*.txt`, `docs/archive/tensorprimitives/notebook/apple-silicon.md` part
//! 21). The menus in [`cfg_neon_f64`] / [`cfg_neon_f32`] are that measurement,
//! ordered by it, and each carries its own margin against the session floor.
//!
//! **The budget filtered half the columns well and chose badly in three**,
//! which is A34 restated rather than a new finding. The `!` over-budget flag is
//! a clean separator for `planar` and `3m` — its shapes are the bottom of those
//! columns in both dtypes — and unreliable for `real` and `1m`, where
//! over-budget shapes appear at the *top*. `f64` real is the sharp case: the
//! budget proposed `6x8`, BLIS's own shape, and the machine prefers `16x3` by
//! 4.9% — a shape the budget calls *over* its limit.
//!
//! Register blocks are per-microarchitecture, so this is an M3 Max result and
//! not a NEON one. Re-run the sweep on any other Apple part before quoting it.

use crate::{Blocking, ComplexMethod, KernelConfig, PackFormat, TileFormat, Ukr};

use core::arch::aarch64::*;

// ---------------------------------------------------------------------------
// Shims: NEON spelled in the macro's x86-shaped vocabulary
// ---------------------------------------------------------------------------
//
// `simd_kernels!` takes six operations with x86 argument order. Three of NEON's
// six match directly (`vld1q_*`, `vdupq_n_*`, `vst1q_*` — note `vst1q` takes
// `(ptr, value)`, which is what the macro passes). Three do not, and each shim
// says why. All are `#[inline(always)]`, so none survives into the kernel body.

/// `$zero` must take no arguments; `vdupq_n_f64` takes the value to splat.
#[inline(always)]
fn zero_f64() -> float64x2_t {
    unsafe { vdupq_n_f64(0.0) }
}

/// `$fmadd(a, b, c) = a * b + c`, the x86 order. **NEON reverses it**:
/// `vfmaq_f64(a, b, c) = a + b * c`, accumulator first.
///
/// # Safety
/// None required beyond the caller's — this is arithmetic on values, and the
/// `unsafe` is only the intrinsic's own historical requirement.
#[inline(always)]
unsafe fn fmadd_f64(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
    vfmaq_f64(c, a, b)
}

/// `$fnmadd(a, b, c) = -(a * b) + c`. `vfmsq_f64(a, b, c) = a - b * c`, so the
/// same reordering plus the sign the planar kernel needs for `re*re - im*im`.
///
/// # Safety
/// As [`fmadd_f64`].
#[inline(always)]
unsafe fn fnmadd_f64(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
    vfmsq_f64(c, a, b)
}

/// `f32` counterpart of [`zero_f64`].
#[inline(always)]
fn zero_f32() -> float32x4_t {
    unsafe { vdupq_n_f32(0.0) }
}

/// `f32` counterpart of [`fmadd_f64`].
///
/// # Safety
/// As [`fmadd_f64`].
#[inline(always)]
unsafe fn fmadd_f32(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
    vfmaq_f32(c, a, b)
}

/// `f32` counterpart of [`fnmadd_f64`].
///
/// # Safety
/// As [`fmadd_f64`].
#[inline(always)]
unsafe fn fnmadd_f32(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
    vfmsq_f32(c, a, b)
}

// `neon` is architecturally guaranteed on aarch64 and is in every supported
// target's default feature set, so the `#[target_feature]` the macro applies is
// a no-op here rather than an unlock. It is passed anyway: the macro requires at
// least one feature, and naming the requirement keeps the two ISA modules
// readable side by side. The trampolines it also generates are redundant for the
// same reason — nothing prevents these kernels being coerced to a fn pointer
// directly — but they cost one `call` per micro-tile against `kc * MR * NR`
// FMAs, and diverging from `x86.rs`'s structure to save it is not worth the
// asymmetry until a measurement asks for it.
simd_kernels!(
    neon_f64,
    f64,
    float64x2_t,
    2,
    ["neon"],
    zero_f64,
    vld1q_f64,
    vdupq_n_f64,
    fmadd_f64,
    fnmadd_f64,
    vst1q_f64
);

simd_kernels!(
    neon_f32,
    f32,
    float32x4_t,
    4,
    ["neon"],
    zero_f32,
    vld1q_f32,
    vdupq_n_f32,
    fmadd_f32,
    fnmadd_f32,
    vst1q_f32
);

// ---------------------------------------------------------------------------
// Configuration builders
// ---------------------------------------------------------------------------

/// NEON shapes for `f64` / `c64` (`L = 2`). **Measured on an M3 Max**, three
/// arms, `kc = 256` — the depth [`Blocking::derive`] actually uses for 8-byte
/// reals, so these are the shapes the driver runs and not a nearby regime.
///
/// | kernel | `MV x NR` | `MR x NR` (logical) | GF/s | over the runner-up |
/// |---|---|---|---|---|
/// | real | `8 x 3` | `16 x 3` | 58.3 | +4.9%, 2.6x the floor |
/// | planar | `2 x 6` | `4 x 6` | 56.2 | +0.2%, a tie with `4x5` and `2x12` |
/// | 1m | `2 x 8` | `2 x 8` | 55.9 | +1.1%, a tie with `4x6` and `3x8` |
/// | 3m | `1 x 8` | `2 x 8` | 79.3 | +7.6%, 4.0x the floor |
///
/// Two of the four are **ties inside the session floor** (p90 1.89%), so their
/// order below the first entry is not a measured ranking and the incumbent
/// budget-derived shape was kept where it tied. The two that are not ties are
/// `real` and `3m`.
///
/// `real`'s winner is the surprise and it is worth stating plainly: `16 x 3` is
/// **over the register budget** (`live = 33`), which every other ISA here
/// punishes by 30–50%, and it wins anyway at the shipping depth. It is also the
/// one shape whose two regimes disagree — perfectly reproducible at `kc = 64`
/// and `256` (0.0% across three arms) and bimodal at `kc = 16` (51.5 / 51.5 /
/// 40.9). The engine never runs it at 16. See `docs/archive/tensorprimitives/notebook/apple-silicon.md` part 21.
pub mod cfg_neon_f64 {
    use super::*;
    configs!(
        f64,
        neon_f64,
        "neon",
        real = [(8, 3), (2, 8), (4, 6), (3, 8)],
        planar = [(2, 6), (2, 5), (1, 12)],
        onem = [(2, 8), (4, 6), (3, 8)],
        threem = [(1, 8), (2, 3), (2, 4)],
    );
}

/// NEON shapes for `f32` / `c32` (`L = 4`). **Measured on an M3 Max**, three
/// arms, `kc = 384`.
///
/// | kernel | `MV x NR` | `MR x NR` (logical) | GF/s | over the runner-up |
/// |---|---|---|---|---|
/// | real | `2 x 8` | `8 x 8` | 111.5 | +0.5%, a tie six ways |
/// | planar | `2 x 6` | `8 x 6` | 112.1 | +0.1%, a tie with `8x5` and `4x12` |
/// | 1m | `4 x 6` | `8 x 6` | 111.2 | +1.4%, a tie with `6x8` and `12x4` |
/// | 3m | `1 x 8` | `4 x 8` | 159.1 | +7.1%, 2.0x the floor |
///
/// The `MV x NR` grid is the same one `f64` is swept over — the budget counts
/// *registers*, not lanes — but **the winners are not the same shapes**, which
/// is the point of measuring each. `real` moves from BLIS's `12x8r` to `8x8`,
/// and `1m` prefers `NR = 6` where `f64` prefers `NR = 8`.
///
/// Three of the four are ties inside a wider floor than `f64`'s (p90 3.64%),
/// and the wider floor is itself a finding: at `L = 4` the same shapes run at
/// twice the throughput and are correspondingly more exposed to this laptop's
/// clock. Only `3m` clears it, and only just.
pub mod cfg_neon_f32 {
    use super::*;
    configs!(
        f32,
        neon_f32,
        "neon",
        real = [(2, 8), (4, 6), (6, 4), (3, 8)],
        planar = [(2, 6), (2, 5), (1, 12)],
        onem = [(4, 6), (3, 8), (6, 4)],
        threem = [(1, 8), (2, 3), (2, 4)],
    );
}

// ---------------------------------------------------------------------------
// Runtime dispatch
// ---------------------------------------------------------------------------

/// An instruction set this file has kernels for.
///
/// One variant, unlike x86's two, and that is the whole difference in this
/// section: there is no widest-first choice to make on aarch64 because NEON is
/// the only vector ISA the architecture guarantees. SVE would add a variant here
/// and is **absent on Apple Silicon** (`hw.optional.arm.FEAT_SVE` does not
/// exist), so it is not speculated about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isa {
    /// NEON / AdvSIMD. Register blocks measured on an M3 Max — see
    /// [`cfg_neon_f64`].
    Neon,
}

impl Isa {
    pub fn name(self) -> &'static str {
        match self {
            Isa::Neon => "neon",
        }
    }
}

/// Every ISA whose kernels this CPU can execute, **ignoring
/// any pinned [`crate::KernelForce`]**, for the kernel-contract tests.
///
/// Always exactly one: NEON is mandatory in the AArch64 base architecture, so
/// there is nothing to detect. `is_aarch64_feature_detected!("neon")` would
/// answer `true` unconditionally on every target this compiles for, and calling
/// it would imply a runtime question that does not exist.
pub fn available_isas() -> &'static [Isa] {
    &[Isa::Neon]
}

/// The ISA the engine will dispatch to, or `None` for the portable scalar path.
///
/// `KernelForce::Scalar` takes the portable path; `Neon` and `Auto`
/// take the kernels. **`avx2` and `avx512` fall through to scalar rather than
/// faulting** — the same answer the x86 dispatch gives for a family the CPU
/// lacks, which is what lets one sweep script pass the same arm list to every
/// machine.
pub fn selected_isa(force: crate::KernelForce) -> Option<Isa> {
    use crate::KernelForce;
    match force {
        KernelForce::Scalar => None,
        KernelForce::Neon | KernelForce::Auto => Some(Isa::Neon),
        // Pinning an x86 family on aarch64 is a request this target cannot
        // honour, and scalar is the documented answer to that everywhere else.
        KernelForce::Avx2 | KernelForce::Avx512 => None,
    }
}

/// One ISA's kernel-set entry points for one element type.
///
/// Deliberately the same shape as x86's `IsaConfigs` rather than shared
/// with it: it carries an `isa: Isa`, and the two `Isa` enums are different
/// types. Eight lines of duplication buys `examples/kernel_shapes` and the
/// contract tests one interface across both architectures.
#[derive(Clone, Copy)]
pub struct IsaConfigs<T> {
    pub isa: Isa,
    pub real: fn() -> KernelConfig<T>,
    pub cplx: fn(ComplexMethod) -> KernelConfig<T>,
    pub row_blocks: fn(bool, ComplexMethod) -> &'static [(usize, usize)],
    pub config_at: fn(bool, ComplexMethod, usize) -> Option<KernelConfig<T>>,
}

/// As x86's, written by hand: the derive would demand `T: Debug` for a struct
/// that never stores a `T`, and function-pointer addresses are not information.
impl<T> core::fmt::Debug for IsaConfigs<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("IsaConfigs")
            .field("isa", &self.isa)
            .finish()
    }
}

/// The five entry points the default menu is built from, per type. Mirrors
/// `x86`'s `dispatch!` and yields `None`/`&[]` under
/// `KernelForce::Scalar`, which sends the caller to the portable path.
macro_rules! dispatch {
    ($t:ty, $cfg:ident, $sets:ident,
     $real:ident, $cplx:ident, $rows:ident, $real_at:ident, $cplx_at:ident) => {
        /// This type's kernel set for a named ISA.
        pub fn $sets(isa: Isa) -> IsaConfigs<$t> {
            match isa {
                Isa::Neon => IsaConfigs {
                    isa,
                    real: $cfg::real_config,
                    cplx: $cfg::cplx_config,
                    row_blocks: $cfg::row_blocks,
                    config_at: $cfg::config_at,
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
    cfg_neon_f64,
    isa_configs_f64,
    config_real_f64,
    config_cplx_f64,
    row_blocks_f64,
    config_real_f64_at,
    config_cplx_f64_at
);
dispatch!(
    f32,
    cfg_neon_f32,
    isa_configs_f32,
    config_real_f32,
    config_cplx_f32,
    row_blocks_f32,
    config_real_f32_at,
    config_cplx_f32_at
);
