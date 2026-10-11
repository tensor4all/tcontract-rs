//! The AVX-512F kernels and their register-block menus, from Lukas Devos's
//! tensorcontract (see the module documentation of [`super`]).

use crate::{Blocking, ComplexMethod, KernelConfig, PackFormat, TileFormat, Ukr};
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

simd_kernels!(
    avx512_f64,
    f64,
    __m512d,
    8,
    ["avx512f"],
    _mm512_setzero_pd,
    _mm512_loadu_pd,
    _mm512_set1_pd,
    _mm512_fmadd_pd,
    _mm512_fnmadd_pd,
    _mm512_storeu_pd
);

simd_kernels!(
    avx512_f32,
    f32,
    __m512,
    16,
    ["avx512f"],
    _mm512_setzero_ps,
    _mm512_loadu_ps,
    _mm512_set1_ps,
    _mm512_fmadd_ps,
    _mm512_fnmadd_ps,
    _mm512_storeu_ps
);

// ---------------------------------------------------------------------------
// Configuration builders
// ---------------------------------------------------------------------------

/// AVX-512 shapes for `f64` / `c64` (`L = 8`), measured at the `kc = 256` that
/// [`Blocking::derive`] picks for 8-byte reals.
///
/// | kernel | `MV x NR` | `MR x NR` (logical) | acc | load | FMA | bytes/flop | GF/s |
/// |---|---|---|---|---|---|---|---|
/// | real | `3 x 8` | `24 x 8` | 24 | 11 | 24 | 0.67 | 88.4 |
/// | planar | `2 x 6` | `16 x 6` | 24 | 16 | 48 | **0.46** | **102.8** |
/// | 1m | `3 x 8` | `12 x 8` | 24 | 22 | 48 | 0.67 | 91.1 |
/// | 3m | `1 x 10` | `8 x 10` | 30 | 33 | 30 | 0.68 | 87.8 |
///
/// Note the shapes are *not* interchangeable between methods. Every candidate
/// needing more than 32 live vector registers — accumulators plus one A plane
/// plus the broadcasts — falls off a cliff of 30–50%, which is what rules out
/// e.g. `3m 24 x 4` (36 accumulators). Within the survivors the ranking follows
/// bytes-per-useful-flop, not FMA count: see the module docs.
///
/// The alternates on each menu, and what they cost at the operating `kc`
/// (fraction of the default's throughput, from the Phase 4.1c re-run of
/// `examples/kernel_shapes`):
///
/// | method | menu, `MR` (`NR`) | cost of each alternate |
/// |---|---|---|
/// | real | 24 (8), 16 (8), 8 (8) | 0.82, 0.79 |
/// | planar | 16 (6), 24 (3), 8 (8) | 1.02, 0.93 |
/// | 1m | 12 (8), 16 (6), 8 (8) | 1.02, 0.86 |
/// | 3m | 8 (10), 16 (4), 24 (3) | 0.89, 0.87 |
///
/// Those within a couple of percent of 1.00 are ties at the sweep's own
/// repeatability, not free lunches; the menu is ordered by the Phase 3 choice,
/// which is not re-litigated here.
pub mod cfg_avx512_f64 {
    use super::*;
    configs!(
        f64,
        avx512_f64,
        "avx512",
        real = [(3, 8), (2, 8), (1, 8)],
        planar = [(2, 6), (3, 3), (1, 8)],
        onem = [(3, 8), (4, 6), (2, 8)],
        threem = [(1, 10), (2, 4), (3, 3)],
    );
}

/// AVX-512 shapes for `f32` / `c32` (`L = 16`), measured at `kc = 384`.
///
/// | kernel | `MV x NR` | `MR x NR` (logical) | bytes/flop | GF/s |
/// |---|---|---|---|---|
/// | real | `3 x 8` | `48 x 8` | 0.29 | 179.9 |
/// | planar | `2 x 6` | `32 x 6` | **0.20** | 195.7 |
/// | 1m | `4 x 6` | `32 x 6` | 0.37 | 175.4 |
/// | 3m | `1 x 10` | `16 x 10` | 0.24 | 194.9 |
///
/// 1m prefers a wider real row block here than it does in `f64` because the
/// doubled lane count already makes its "1e" panel large; the other three
/// methods land on the same `MV x NR` in both precisions.
///
/// The alternates and their cost at the operating `kc`, as for `f64`:
///
/// | method | menu, `MR` (`NR`) | cost of each alternate |
/// |---|---|---|
/// | real | 48 (8), 32 (8), 16 (10) | 0.87, 1.02 |
/// | planar | 32 (6), 48 (4), 16 (12), **32 (5)** | 0.95, 0.90, **1.078** |
/// | 1m | 32 (6), 24 (8), 16 (8), 8 (12) | 1.03, 0.86, 0.86 |
/// | 3m | 16 (10), 32 (4), 48 (3) | 0.91, 0.84 |
///
/// **`planar 32 (5)` is faster than the default it sits behind**, which no other
/// entry on any menu is: 210.9 GF/s against 195.7 in the same Phase 3 sweep, and
/// ahead at `kc = 64` too. The default appears to have been chosen on the
/// bolded bytes-per-flop above rather than on the throughput in the same output
/// file (A35) — the failure mode this project has now recorded four times. It is
/// **last on the menu, not first**, because a kernel margin is not a corpus
/// margin: `NR` also sets the `jr` loop count and the packed-`B` sliver
/// geometry. `RowBlock::Index(3)` is the arm that settles it.
///
/// The 32-bit menus are the ones that matter for the write-back: the corpus
/// rounds every stride-1 index up to a multiple of **24**, and at `L = 16` no
/// full-width `MR` divides 24 except 1m's, so the others can only reduce the
/// straddling fraction rather than eliminate it. `real 16 (10)` costing nothing
/// measurable is the important entry — it is the shape the `f32` cases still on
/// the gather path would need.
pub mod cfg_avx512_f32 {
    use super::*;
    configs!(
        f32,
        avx512_f32,
        "avx512",
        real = [(3, 8), (2, 8), (1, 10)],
        // `(2, 5)` is the shape A35 found and could not reach: Phase 3's own
        // sweep names `32x5` at **210.9 GF/s** against the shipped `32x6`'s
        // **195.7**, 7.8% faster at the operating `kc` and also ahead at
        // `kc = 64`. It is **appended, not inserted**, on purpose — the entries
        // before it keep the positions `tensorprimitives/bench-results/phase4c` swept, so that
        // grid's `idx=` numbering still means what it meant.
        //
        // It cannot be *chosen by the rule*, which reads `MR` and sees a tie
        // with the default. That is deliberate: a 7.8% kernel margin is not a
        // 7.8% corpus margin, since `NR` also moves the `jr` loop count and the
        // packed-`B` sliver geometry. Making it reachable at `idx=3` is what
        // turns A35 from untestable into an A/B, which is the whole change.
        planar = [(2, 6), (3, 4), (1, 12), (2, 5)],
        onem = [(4, 6), (3, 8), (2, 8), (1, 12)],
        threem = [(1, 10), (2, 4), (3, 3)],
    );
}
