//! The AVX2+FMA kernels and their register-block menus, from Lukas Devos's
//! tensorcontract (see the module documentation of [`super`]).

use crate::{Blocking, ComplexMethod, KernelConfig, PackFormat, TileFormat, Ukr};
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// AVX2 needs `fma` as well as `avx2`: the 256-bit loads, broadcasts and stores
// come from AVX/AVX2 but `_mm256_fmadd_*` / `_mm256_fnmadd_*` are FMA3. Every
// CPU that has AVX2 has FMA3 in practice, but they are separate CPUID bits, so
// both are detected and both are enabled here.
simd_kernels!(
    avx2_f64,
    f64,
    __m256d,
    4,
    ["avx2", "fma"],
    _mm256_setzero_pd,
    _mm256_loadu_pd,
    _mm256_set1_pd,
    _mm256_fmadd_pd,
    _mm256_fnmadd_pd,
    _mm256_storeu_pd
);

simd_kernels!(
    avx2_f32,
    f32,
    __m256,
    8,
    ["avx2", "fma"],
    _mm256_setzero_ps,
    _mm256_loadu_ps,
    _mm256_set1_ps,
    _mm256_fmadd_ps,
    _mm256_fnmadd_ps,
    _mm256_storeu_ps
);

// ---------------------------------------------------------------------------
// Configuration builders
// ---------------------------------------------------------------------------

/// AVX2 shapes for `f64` / `c64` (`L = 4`). **Provisional and unmeasured** — see
/// the note below on which half of the choice is modelled.
///
/// 16 ymm instead of 32 zmm, and the same accumulator-plane counts, so the
/// register bound bites much harder than it does on AVX-512. With `live =
/// accumulators + one A plane + the live broadcasts`:
///
/// | kernel | `MV x NR` | `MR x NR` (logical) | acc | live | load | FMA | f/l | bytes/flop |
/// |---|---|---|---|---|---|---|---|---|
/// | real | `2 x 6` | `8 x 6` | 12 | 15 | 8 | 12 | 1.50 | **1.17** |
/// | planar | `1 x 5` | `4 x 5` | 10 | 14 | 12 | 20 | 1.67 | **0.90** |
/// | 1m | `2 x 6` | `4 x 6` | 12 | 15 | 16 | 24 | 1.50 | 1.17 |
/// | 3m | `1 x 4` | `4 x 4` | 12 | 14 | 15 | 12 | 0.80 | 1.50 |
///
/// # What is modelled and what is guessed
///
/// **Modelled, and trustworthy** (D19: the uop model gets the *cliffs* right):
///
/// * `live <= 16` for every shape on every menu. That is the only hard
///   statement here, and it is the one that matters most, because a spill costs
///   30–50% — far more than any ranking error among shapes that fit. It was
///   **checked in the disassembly**, which needs no benchmark: every shipped
///   default in both precisions (six distinct kernels — 1m reuses `real`)
///   allocates 15–16 distinct ymm with *zero* stack traffic in the loop, while
///   `planar 1 x 6` (`live = 16`) spills one register and `planar 2 x 3`
///   (`live = 18`) spills seven. The model's cliff is where the model says.
/// * `acc >= 10`, i.e. enough independent accumulator chains to cover an FMA
///   latency of ~5 cycles against two FMA ports. Below that the kernel is
///   latency-bound however good its byte traffic is. Both survive only just:
///   planar's default has exactly 10 and 3m's exactly 12.
/// * Which shape has the lowest bytes-per-useful-flop within each method, which
///   is the quantity Phase 3 measured to be decisive at the operating `kc`.
///
/// **Guessed** (D19: the same model gets the *ranking* wrong):
///
/// * That the byte-traffic argument still picks the winner when the register
///   file is half as large. It may not: with `MR` down to 4 complex rows the A
///   sliver is small enough that it could stay L1-resident, which is precisely
///   the regime where Phase 3 measured 3m to be *fastest*. So the ranking
///   between the three complex methods is genuinely open on AVX2, more open
///   than it was on AVX-512.
/// * `NR` for planar — though less than it was. `1 x 6` reaches `live = 16`
///   exactly and would give 12 accumulators and 0.83 bytes/flop, strictly
///   better on both counts *if* it allocated cleanly; the disassembly says it
///   does not quite (one spill/reload pair per k-step, against `1 x 5`'s none),
///   which settles the choice without a measurement. What stays guessed is the
///   *price*: one spill of a broadcast is nothing like the 30–50% a spilled
///   accumulator costs, so `1 x 6` might still win. **It could now go on the menu
///   as an alternate** — that was impossible when this was written, because the
///   menu was keyed by `MR` and both shapes have `MR = 4`, but D43 re-keyed it by
///   *position* precisely so an `NR`-only alternate is expressible (that is what
///   made A35's `32x5` reachable in `f32` planar). Appending `(1, 6)` here would
///   make it an `idx=` arm; it has not been done because nothing has asked to
///   measure it and an unmeasured menu entry is a liability.
/// * 3m's default. `1 x 4` is load-port-bound (`f/l = 0.80`) and `2 x 2`
///   (`MR = 8`, on the menu) is exactly balanced at `f/l = 1.00` but moves 25%
///   more bytes per flop. `1 x 4` follows the AVX-512 precedent, where the
///   measured 3m winner `1 x 10` was *also* load-bound and *also* the lowest
///   bytes/flop of its candidates — but that is an analogy, not a measurement.
/// * That one shape per method can be right for all AVX2 hardware at all. The
///   AVX2 microarchitectures spread wider than the AVX-512 ones: FMA latency is
///   5 on Haswell/Broadwell and 4 on later Intel and Zen 3, which moves the
///   `acc >= 10` floor, and Zen 1 splits every 256-bit op into two 128-bit
///   halves, which moves everything. These shapes are a starting point on
///   whatever machine gets measured first, not a tuning.
///
/// Calibrate with the retired `kernel_shapes` example (deleted in the source
/// integration, see `docs/provenance.md`) **on an AVX2 machine** — it swept the AVX2 grid too
/// and prints `acc`/`load`/`FMA`/`bytes-per-flop` next to measured GF/s, so the
/// menus below can be replaced from its output the way the AVX-512 ones were.
///
/// # The menus
///
/// | method | menu, `MR` (`NR`) | why the alternates exist |
/// |---|---|---|
/// | real | 8 (6), 12 (4), 4 (8) | `12 (4)` is `live = 16` with a better `f/l`; `4 (8)` is load-bound and only there for the row-block rule |
/// | planar | 4 (5), 8 (2) | `8 (2)` is the only other aspect ratio that fits at all |
/// | 1m | 4 (6), 6 (4), 2 (8) | as real, halved: 1m's complex tile is half its real row block |
/// | 3m | 4 (4), 8 (2) | the load-bound / balanced pair described above |
///
/// Note what the small blocks do to the write-back: **every** `MR` on every
/// menu here (2, 4, 6, 8, 12) divides 24, and the TCCG corpus rounds every
/// stride-1 extent up to a multiple of 24, so the gather-path problem that
/// motivated the row-block menu on AVX-512 does not arise at all in `f64` on
/// AVX2, which is one reason these menus can stay short.
///
/// `tcbench shapes` says so quantitatively, and it costs no CPU to re-check
/// (a pinned-AVX2 run of the retired `tcbench shapes`). Over the 392 corpus
/// case-dtype-methods, `Plan::row_block` would change shape on **26 under
/// AVX-512 and 12 under AVX2**, and the 12 are all `f32` — it is inert in `f64`
/// and in all three complex methods in both precisions. The sharper number is
/// the other one: **81** case-dtype-methods have no shape on their AVX-512 menu
/// that clears the gather path at all, and **0** on their AVX2 menu. Smaller
/// register blocks are worse for the kernel and better for the write-back, and
/// on AVX2 the write-back side of that trade is simply won.
pub mod cfg_avx2_f64 {
    use super::*;
    configs!(
        f64,
        avx2_f64,
        "avx2",
        real = [(2, 6), (3, 4), (1, 8)],
        planar = [(1, 5), (2, 2)],
        onem = [(2, 6), (3, 4), (1, 8)],
        threem = [(1, 4), (2, 2)],
    );
}

/// AVX2 shapes for `f32` / `c32` (`L = 8`). **Provisional and unmeasured**; the
/// reasoning, and the modelled/guessed split, is in [`cfg_avx2_f64`].
///
/// The `MV x NR` grid is deliberately *identical* to the `f64` one, so that a
/// single-vs-double comparison on AVX2 is a comparison of the hardware and not
/// of two shape choices — the same reason `examples/kernel_shapes` sweeps one
/// candidate list for both types. (AVX-512 1m is the one place the project
/// departed from that, and it departed from it on measured evidence, which
/// there is none of here.) So the logical row blocks are twice the `f64` ones:
///
/// | kernel | `MV x NR` | `MR x NR` (logical) | acc | live | bytes/flop |
/// |---|---|---|---|---|---|
/// | real | `2 x 6` | `16 x 6` | 12 | 15 | 0.46 |
/// | planar | `1 x 5` | `8 x 5` | 10 | 14 | **0.33** |
/// | 1m | `2 x 6` | `8 x 6` | 12 | 15 | 0.46 |
/// | 3m | `1 x 4` | `8 x 4` | 12 | 14 | 0.56 |
///
/// Menus: real 16 (6), 24 (4), 8 (8); planar 8 (5), 16 (2); 1m 8 (6), 12 (4),
/// 4 (8); 3m 8 (4), 16 (2). Here `real 24 (4)`, `1m 12 (4)` and the `8`s divide
/// the corpus's multiple-of-24 extents while the `16`s do not, so the row-block
/// rule has a real choice to make in `f32` on AVX2 as it does on AVX-512 —
/// unlike `f64`, where every shape on every menu divides 24 already.
pub mod cfg_avx2_f32 {
    use super::*;
    configs!(
        f32,
        avx2_f32,
        "avx2",
        real = [(2, 6), (3, 4), (1, 8)],
        planar = [(1, 5), (2, 2)],
        onem = [(2, 6), (3, 4), (1, 8)],
        threem = [(1, 4), (2, 2)],
    );
}
