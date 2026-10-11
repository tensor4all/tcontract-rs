# Phase 2 S0: attribution (tensor4all/tprims-rs #50)

Baseline numbers: `decision.txt`, `targets.txt`. Raw profile: `perf/` (`attribution-raw.txt`, `*.cycles.gz`,
`*.pf.gz`). Host: EPYC 7713P, idle pinned cores of one CCD, `perf_event_paranoid=1`. Method: `perf record -F 4999`
cycles over a process that runs the plan row (faer in every profiled case) and then the packed row; the packed
window is the span of `tprims_kernel` kernel/pack samples. Kernel-mode samples are unresolved (kptr_restrict), so
kernel-mode time is split into page faults vs. futex/scheduler by the page-fault event count and `perf stat`
(`cycles:k`, context switches). "Off-CPU" = 1 - samples/(T x window x F): time threads were parked (no cycles).

## Baseline (packed/plan workload ratio; >1 = packed slower; noise 0.3-2.8%)

| corpus | 1T faer-subset / whole | 4T | 8T |
|---|---|---|---|
| tenferro-p1 (19 faer cases) | 1.034 / 1.002 | 1.832 / 1.053 | 2.314 / 1.084 |
| tenferro-p1-gemm (53, all faer) | 1.078 | 1.173 | 1.231 |
| large-batched-gemm (16, all faer) | 1.064 | 1.140 | 1.153 |
| phase2-extra (11, all faer) | 2.399 | 11.94 | 16.68 |

The gate (faer-subset <= max(1.05, 1+noise), cap 1.5x above 50 us) fails in all 12 cells today. The 1T ratios are
modest except for small/rank-1 shapes; 4T and 8T are dominated by a few pathological cases (below).

## Profile: cycles by category inside the packed row

| case | T | packed/plan | kernel | pack | write-back | driver | kernel-mode | off-CPU of T x window |
|---|---|---|---|---|---|---|---|---|
| 1024^3 f64 | 1 | 1.13 | 85.2% | 4.6% | 6.5% | 2.4% | 1.3% | 0.3% |
| 1024^3 f64 | 8 | 1.13 | 77.1% | 7.5% | 7.2% | 2.1% | 5.9% | 1.1% |
| 1024^3 c64 | 1 | 1.10 | 89.6% | 3.0% | 5.4% | 1.2% | 0.7% | 0.2% |
| 1024^3 c64 | 8 | 1.21 | 86.4% | 5.7% | 5.1% | 1.4% | 1.3% | 3.3% |
| gemm_batched_008 (m9 n19683 k27 h243) | 1 | 2.35 | 34.3% | 41.2% | 11.0% | 7.5% | 5.9% | 0.8% |
| gemm_batched_008 | 8 | 1.87 | 24.5% | 30.4% | 7.0% | 7.1% | 30.3% | 6.4% |
| gemm_batched_012 (m27 n729 k3 h2187) | 8 | 3.33 | 6.6% | 3.6% | 25.4% | 12.5% | 48.7% | 6.2% |
| gemm_batched_047 (m1 n144 k11 h1100) | 8 | 77 (18.9 ms vs 0.25 ms) | 2.1% | 3.9% | 0.5% | 2.8% | 83.3% | 34.1% |
| small_n4_k32_h8192 | 1 | 1.72 | 16.3% | 70.5% | 3.4% | 4.6% | 5.2% | 0.4% |
| small_n4_k32_h8192 | 8 | 70.9 | 2.8% | 8.8% | 0.5% | 3.2% | 77.8% | 55.8% |

(Ratios here are from the profile runs, 10-60 samples, on a possibly different core set than `decision.txt`.)

Supporting counts:
- gemm_batched_047, 8T, `perf stat`: 3.0e9 `cycles:k` against 4.4e8 `cycles:u`, 248,571 context switches over 20 runs
  (about 12k per call), only 4,460 page faults. The kernel-mode time is therefore futex wait/wake in
  `std::sync::Barrier` and the scheduler, not page faults. `Barrier::wait`/`futex` are the top user symbols.
- Page faults by faulting code (whole process, includes the warm-up where the freshly allocated C is first touched):
  pack 0.6k-1.8k events for 1024^3 and 35-1300 for the batched cases; write-back 2k-283k, which is first-touch of the
  harness output `vec![0; n]` (not a per-call cost; it only inflates kernel-mode share of 008/012 8T, so those rows
  overstate kernel-mode time). No allocator symbol is in any top list.

## Reading against the spec's causes

1. **Cause 2 (barriers, serial batch loop, width not matched to the work). Dominant, by far.**
   For batches of small items (047, small n<=4 / large-H, 012) the packed route at 4T/8T is 10-77x slower than at 1T:
   87% of cycles in kernel mode, 248k context switches, 34-56% of thread time parked. The in-plan batch loop runs the
   per-item driver with two futex barriers per (jc, pc) for every tiny item, so barrier cost scales with H, not with
   the work. Weighted loss: gemm_batched_047 / dot_general_049 is 157-181 ms at 4T/8T, 3x the next entry; the
   small n<=4 entries of phase2-extra are 75 ms (n4_k32), 43-56 ms (m1_n64_k64) at 4T/8T. Fixing only the barrier
   primitive makes each wait cheaper, but 1100-8192 items x 2 barriers still run; batch-axis job claiming (S3) removes
   them for this class, and a plan-time width choice is needed so tiny per-item work never pays a team barrier.
   For large GEMMs at 8T the sync loss is small here (off-CPU 1-3%, barrier 0.4-0.8% user), so the barrier primitive
   (S2) matters for the 1.13-1.21x 8T large-GEMM gap only a little, as far as this profile shows.
2. **Cause 6 and the small-k / small-m class (S3). Second.**
   Where packing cannot be amortized, pack takes 41% (008, 1T) and 70% (n4_k32, 1T) of packed cycles and the kernel
   only 16-34%; faer reads B in place. 1T ratios: 008 2.35, 022 2.45, 023 2.07, n2/n3/n4 small-H 2.4-3.3,
   rank-1 K=1 2.2-4.6, m1_n64_k64 5.3. This accounts for most 1T per-case-cap failures (11 in tenferro-p1-gemm).
   B packed and used once is exactly cause 6; the S3 small-problem family (B in place, A packed in leased scratch)
   and Direct-B target it.
3. **Cause 1 (pack/write-back/driver overhead) explains the whole large-GEMM 1T gap; cause 5 (tiles) does not show up.**
   1024^3 f64 1T: packed 52.9 ms, plan 47.0 ms; kernel samples are 85.2% of 52.9 = 45.1 ms, already below faer's total.
   c64: kernel 89.6% of 193.4 = 173.3 ms vs faer 176.2 ms. So the micro-kernel is on par with faer's whole routine
   and the gap is pack + write-back + driver = 10-15% of packed cycles. (Approximate: the faer figure includes its own
   packing; this says the AVX2 tile is not the bottleneck at this size, not that it cannot improve.)
   Write-back alone is 5-7% at 1024^3 and 25% in 012 (m27 n729 k3, k too small to amortize the scratch tile): the
   scratch tile + scalar write-back with no `target_feature` listed in the spec is confirmed as a measurable cost.
   large-batched-gemm weighted ratio 1.06-1.15 and tenferro-p1's top 1T losses (dot_general_016/031 c64, 1.11x) are this class.
4. **Cause 3 (allocation): verified fixed.** No allocator symbols in any profile; pack faults are a few thousand
   in total.
5. **Cause 4 (A_c = 512 KiB blocking): not isolated.** Needs the one-arm blocking experiment of S1; the profile
   cannot separate it. Evidence of impact is weak: pack/write-back shares at m>=256 are 10-15%, in line with item 3.
6. **8T on large GEMMs.** Ratio 1.13 (f64) and 1.21 (c64) vs 1.13/1.10 at 1T: c64 loses 11 more points at 8T with
   3.3% off-CPU and 5.7% pack (up from 3.0%), so a modest parallel-scaling cost (B pack not overlapped with compute,
   1-D static split) is present, but the spec's "0.40 at 8T" gap from the old figures is not reproduced here
   for large GEMMs; the large 8T gap comes from the batch/small-item class above.

## Recommended order for S1-S3

1. **S3 batch-axis claiming plus a plan-time width rule for tiny items** (with the small-problem family for the
   same shapes). Largest by weighted loss (047: 157-181 ms; small n<=4 / large-H entries) and the only fix for the
   per-case-cap failures at 4T/8T (ratios up to 51x). Without it the gate cannot pass at any width above 1.
2. **S3 small-problem family (B in place, A packed in scratch) and rank-1 / matvec shapes.** Accounts for the 1T
   cap failures (cases at 1.5-5x) and, after step 1, the remaining 4T/8T small-item losses.
3. **S1 pack/write-back codegen (const-generic flags, `target_feature` variants, SIMD pack) and then the
   one-arm blocking experiment.** This is the 1T large-GEMM gap (10-15% of cycles outside the kernel) and the bulk
   of the large-batched-gemm and tenferro-p1 weighted ratios (1.06-1.15, 1.03 faer subset at 1T). Needed to reach
   <= 1.05. Direct f64 kernels follow only if pack/write-back gains leave a gap; this profile does not show the
   tile as the bottleneck.
4. **S2 spin-then-park barrier, medium-width rule, Direct-B, DynamicTiles.** The barrier primitive matters mainly
   for the batched class (covered by step 1) and, for large GEMMs at 8T, by only about 1-3% off-CPU in this profile;
   measure after steps 1-3 and keep only what the 8T check shows.

Caveats: single-run profiles on a shared host (cores idle-checked); kernel symbols unresolved; the harness runs the
plan row first in the same process, separated by symbol, so cache/page state carries over; the n<=4 / rank-1
entries are synthetic.
