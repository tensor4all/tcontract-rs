# Native three-provider contraction comparison

## Protocol (recorded before timing)

User requested tcontract-rs, original tensorprimitives-rs and TBLIS; final thread
counts are 1,4,8,12 (not 16), native CPU builds, and the measurement procedure
fixed in the tcontract-benchmark skill. Existing TCCG `tcbench` has 49 cases and
TBLIS FFI, but was serial-only and no longer had the upstream baseline.
Reuse it; add an optional pinned upstream dependency and the existing
BenchThreads host/pool helper instead of a new suite.

Baseline/library commit: `d726474701dce6ac571758755a8d60d6600df546` plus
recorded local harness changes (no numerical-library changes).
Upstream: `8cda75e11ed26f46c0c22f9629004c84dbabc8e5`.
TBLIS: 2.0, `20cc0bcdb13ddf9fbf619138b6081a16e9e48b7f`.
Bundled static BLIS: 3.0, `358e689cadd6757f564a2992cf46a2f7d6fa6bb0`;
system BLIS 0.9.0 is installed but is not this TBLIS baseline's provider.
TBLIS CMake Release flags: C/C++ `-O3 -march=native`, BLIS family `zen3`,
pthread BLIS / OpenMP TCI. Auto-detection initially chose generic on the
unrecognized Zen5 CPU; that configuration was rejected before timing.
BLIS's latest available AMD family in the pinned tree is Zen3 (AVX2), not
Zen5-native AVX512. This baseline limitation must accompany comparisons.

Hardware: AMD Ryzen AI 9 HX 470, 12 physical cores / 24 logical CPUs,
heterogeneous L3 domains: cores 0-3 (L3 #0), 4-11 (L3 #1).
Core sets: 1T=4, 4T=4-7, 8T=4-11, 12T=0-11; no SMT.
The 12T arm is explicitly user-requested full-physical-core scaling across
both L3s, not the single-L3 protocol. Record observed clocks/topology/load.
Library cache/domain estimates in `info` are not actual affinity/topology.

Complete case list: `corpus::corpus()`'s 49 cases, no filter, contiguous
TCCG layouts, dtype f64,c64, nominal tensor sizes 1 and 16 MiB. Seed 0x5EED
for timing and 0xA11CE for verification; alpha=1,beta=0. TCCG sizing includes
extent rounding/minima; record equivalent M/N/K per row, never imply exact
1/16 MiB allocations. Engines: plan,packed,upstream,tblis. Primary comparison:
plan vs external upstream/TBLIS. Packed is diagnostic only.

Rust: rustc 1.99.0, release opt-level=3, thin LTO, codegen-units=1,
`RUSTFLAGS=-C target-cpu=native` for all engines. Execution-only boundary:
exclude input/descriptor/plan/pool construction; include packing, call/FFI,
executor entry and provider internal scratch/allocation/scoped threads.
Keep provider default algorithms/pool policies; document rather than silently
normalize their ownership differences.

Correctness gate before timing: known-value ones GEMM (D=K) in f64/c64,
then full corpus comparison against plan at both sizes and every thread
count, finite Frobenius residual <=1e-10. All engines receive identical data.
One warm-up, five timed reps, best wall time; whole suite A then A2, all
thread counts in order, never parallel. Existing pinned.sh validity gate:
3-second idle windows, <=5% selected-CPU busy, maximum three attempts.
No compilation or other benchmark while timing. Invalid suite means the
paired experiment is inconclusive, not a selectively repaired result.

Report every case, speedup, 1T scaling, and per-row A/A relative spread
abs(A2/A-1); geometric means summarize the complete list. No optimization
promotion threshold: this is a descriptive baseline comparison, not an
optimization candidate. Differences within observed noise are inconclusive.
Retain unfavorable scaling and results. Raw results and reproduction metadata:
`benchmarks/benchmarks/tcbench/results/2026-10-07-native/`.

## Verification before timing

- Native release integration smoke test: known values, all 49 cases in f64/c64
  at 1T/4T/8T/12T, invalid zero-thread rejection: passed with upstream,TBLIS.
- `cargo clippy -p tcontract-bench --all-targets --features upstream,tblis -- -D warnings`: passed.
- Skill mirror check: passed.

## Result

The first invocation exceeded its 600-second controller deadline while
measuring 16 MiB at 1T. Retained as `2026-10-07-native-interrupted`, excluded
in full. Reran the complete procedure (not just favorable cases); both full
A/A suites finished. All eight correctness processes and sixteen timing
processes passed idle guards on attempt 1, all 6,272 measurement rows were
present and mismatch-free. The stdlib summarizer validates completeness and
produces 784 per-case comparisons.

At 16 MiB, 8T geometric-mean tcontract speedup against its own 1T is 5.06x f64
and 5.84x c64; upstream is very close (time ratios 1.04x/1.02x), while this
Zen3/AVX2 TBLIS build has time ratios 1.31x/1.14x. Kernel/ISA confounding
prevents a driver-only claim. All cases and A/A spreads remain in the linked
result; small differences below individual noise spreads are inconclusive.

12T has severe runtime variance despite idle windows: TBLIS's 1 MiB c64
`abjc-cbka-kj` changes from 0.277 ms to 29.991 ms; tcontract small cases also
vary substantially. Cause unestablished, **12T ranking inconclusive**.
No rows were dropped and no performance optimization was promoted.

Final checks include native/default-feature and upstream/TBLIS integration
checks, feature-enabled clippy, rustfmt, script syntax, skill mirror equality,
and the complete-result summarizer. No PR/push or library changes were made.
