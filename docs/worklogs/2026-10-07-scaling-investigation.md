# Scaling investigation (current TBLIS unchanged)

> **Two standing contracts for this work.** (1) Every optimisation claimed in
> a PR must be backed by evidence recorded here: the measurement, the exact
> conditions, and a link to the raw data under
> `benchmarks/benchmarks/tcbench/results/` or `experiments/`. (2) Optimisations
> that were tried and did **not** work are recorded here too, with the numbers
> that rejected them, and are never quietly dropped. The ledger for (2) is
> "Ledger of attempts that were rejected" below.
>
> **Status: experimental, not promoted.** The partition change's primary gate
> passes, but its predeclared non-regression gate does **not**: unresolved
> individual regressions remain (see "Promotion status" at the end). The
> axis-derived scatter wiring was reverted. Nothing here is a shipped default.


## Need gate and diagnostic protocol, before new measurements

User requests root-cause analysis and optimization of tcontract scaling, retaining
the existing Zen3 TBLIS baseline. Existing complete 49-case measurements
show repeatable 4T->8T c64 regressions in `ajbc-ckba-jk` and
`ajbdc-ckbad-jk` at nominal16MiB, affecting whole-call times (15->40 ms,
15->28 ms), not merely an isolated helper. These are the predeclared
representative cases; both input preparation and planning remain outside
execution timing. The affected execution path is the entire measured call,
so its end-to-end share exceeds a 20% need threshold.

Initial source evidence: `blocking/probe.rs::probe_sysfs` probes CPU0,
whose L3 is 16MiB shared by four physical cores on this heterogeneous host.
The benchmarks are pinned to CPUs4-11, whose L3 is 8MiB shared by eight
physical cores. Thus the model reports two domains at 8T where actual
placement has one. `plan/orientation.rs::columns_beat_rows` then switches
shallow contractions from 8 row strips to 8 column strips. Both cases have
n/nr=8 exactly. Column partitioning also changes packing/output locality.
These are hypotheses, not yet a measured cause.

Diagnostic arms (same existing native binary): default; forced
`TCBENCH_L3_DOMAINS=1` (same physical affinity, changes only the domain
estimate); pinned row grid `TCBENCH_PARTITION=8x1` at8T. Measure both cases,
c64 nominal16MiB, requested4T/8T (grid arm8T only), plan/packed, one warm-up
and five reps best wall time, complete diagnostic set repeated twice.
Known values/full selected-case verification before timing; <=1e-10 finite
residual. Pin4T=4-7,8T=4-11 with existing <=5%/3-second idle guards.
Sequential, no simultaneous builds/tests/benchmarks. Preserve every arm.
A large repeatable whole-call improvement (>20%, above A/A spread) justifies
an affinity-aware cache probe experiment; no library edits before this gate.

Raw diagnostics: `benchmarks/benchmarks/tcbench/results/scaling-diagnostic/`.

## Diagnostic result / implementation gate

Both full diagnostic repeats passed guards and output comparisons. Plan times
in ms: `ajbc` default8T 47.036/46.972 vs domain1 10.171/10.504;
`ajbdc` default8T 32.658/35.088 vs domain1 9.849/9.899. Explicit8x1 rows
match domain1, while 4T is essentially unchanged. The measured need and >20%
gate pass. The mechanism is the domain-dependent grid change; the specific
relative contributions of duplicated packing and output cache-line traffic
have not been separately counted, so do not claim either as the sole cause.

## Baseline/candidate protocol (before candidate implementation)

Capture the complete baseline anew with existing `tcbench/run.sh` and current
Zen3 TBLIS, then the candidate with the same runner. Each full49-case,
f64/c64,1/16MiB,1/4/8/12T,all4-engine set has two full A/A runs and five
best-of repetitions. This is descriptive optimization evidence, not comparing
against older-session clocks. Baseline binary SHA256:
`a75da0382c8696d246743a2183f5d7b9af807339ccefaf9703c61b6a0e3035df`.
Candidate changes only the sysfs cache-probe CPU from unconditional CPU0 to
the first CPU in the calling thread's initial allowed affinity mask, with
legacy CPU0 fallback if procfs information is absent/unusable. Hardware
cache still probed once; no filesystem/syscall in execution hot paths and no
implicit global knob. Mixed/scattered cpusets and later affinity changes
remain explicit-model limitations rather than a new scheduler.

Primary gate: each of the two representative c64 nominal16MiB 8T cases
improves >=2x on geometric means over A/A, with improvement exceeding both
arms' per-case A/A spread. Numerical checks: all known-value/full-corpus
comparisons finite <=1e-10, kernel unit tests and release contraction tests.
Nonregression: no >10% geometric-mean tcontract slowdown in any size/dtype/thread
group. Retain and inspect every individual >10% regression; do not promote
if an unexplained regression exceeds both arms' A/A spread. If a validity
or nonregression gate fails, classify the pair inconclusive/failed; do not
selectively replace cases. Existing <=5%/3-second idle guards, no parallel
runs/builds, per-process observations retained. Add Hadamard regression
measurement at1T/4T since all-batch behavior must not regress.

Raw complete sets: `results/scaling-baseline/` and `results/scaling-candidate/`.

The first full baseline attempt exited75 on a12T idle gate (CPUs0-11 remained
busy over all three pre-run observations). It is **inconclusive**, not a
baseline for promotion; keep it as `scaling-baseline-invalid`. A complete
rerun uses the saved baseline binary (`TCBENCH_BIN`, exact hash above), not
rebuilding from candidate sources. Its source identity is the original
native-run harness snapshot plus the unchanged library at the base commit.
The runner's captured current source.patch may contain candidate changes;
binary SHA/source identity, not current checkout contents, identify the
baseline. No idle threshold is relaxed and no12T case is dropped.

## Candidate implementation checks

`probe_sysfs` now selects the first allowed CPU from initial
`/proc/thread-self/status` with CPU0 fallback. Parser tests cover range,
comma-separated, full-affinity and malformed/missing cases. Native release
kernel tests161 and contraction tests225 passed. Pinned `info` onCPUs4-11
now reports8MiB L3 shared by16 logical CPUs and1 domain at8T (before:16MiB,
8 logical CPUs,2 domains). This is the expected mechanism correction, not a
hardcoded per-host optimization.

## Explicit single-L3 follow-up protocol (before candidate timings)

The second full1/4/8/12T baseline attempt again exited75 on a12T pre-run
idle gate. No complete baseline/candidate12T pair is available and neither
failed full experiment passes the original promotion gate. All failed files
remain; this is not repaired by selectively inserting12T observations.

Proceed with a **separate, explicitly single-L31/4/8T experiment** for the
actual affinity bug, not a claim about12T. Complete49 cases, both sizes and
dtypes, all4 engines, five reps and fullA/A repetitions as above; primary
cases,>=2x gate,10% group/nonregression and individual-noise conditions are
unchanged within this scope. Pair baseline/candidate at each size/thread
group withAB thenBA on the second complete repeat. Existing idle thresholds
and guards are unchanged. No per-case exclusions. Results cannot establish
12T nonregression;12T performance remains blocked by host load. Numerically,
the12T mask0-11 selectsCPU0 in both old and new probes, so that path retains
the same descriptors/partition rule, but this is not a new12T timing claim.

Runner:`tcbench/pair-single-l3.sh`; results:`results/scaling-single-l3/`.
This restricted scope is fixed before any candidate performance observation.

## Single-L3 result: primary passes, nonregression does not

Complete paired AB/BA measurements and correctness/idle guards passed.
Representative16MiB/c64/8T plan improvements are4.544x and3.020x.
Whole49-case plan ratios at8T:1MiB f641.055/c641.151;
16MiB f640.985/c641.095. All group geometric-mean10% nonregression gates
pass, but11 individual plan regressions exceed10% and both A/A spreads.
Therefore this candidate **does not pass promotion**, notwithstanding the
primary gains. Full output and all individual regressions are retained in
`results/scaling-single-l3/comparison.txt` and candidate/vs-baseline.csv.

The six16MiB/f64/8T `abcijk-*` regressions also occur in packed execution;
unchanged upstream/TBLIS controls for the representative `abcijk-eiab-jkec`
are stable. This is a candidate grid effect requiring isolation. In contrast,
1MiB `aqrs-pa-pqrs` uses faer in the plan arm, and the1T `abjc-cbka-kj`
regression cannot come from the8T domain-dependent grid switch. The
1MiB `ijk-il-jlk` packed control is unchanged while one plan sample is slow.
Do not explain all11 as a single partition bug or discard these observations.

## Grid isolation protocol, fixed before further measurements

Use the current saved native candidate binary, no library edits yet.
Filters `abcijk` (all mirror cases, not selected winners) and `ajb` (both
primary cases), f64/c64,1/16MiB, budgets1/4/8 on CPUs4/4-7/4-11.
Force packed execution with existing TCBENCH_PARTITION, using all meaningful
row/column/balanced grids:1T1x1;4T4x1,1x4,2x2;
8T8x1,1x8,4x2,2x4. All cases verified before timings, finite<=1e-10;
five reps and one warm-up, full repeats A/A2 with reverse grid order on A2.
Same <=5%/3-second guard, all results preserved; whole diagnostic inconclusive
if a guard fails. Native binary/layout/kernel/block sizes remain constant.
This separates grid choice from code generation and unrelated faer effects.

Need gate for a rule change: recover>=10% of an affected whole-call cost,
above row/column A/A spreads, while retaining>=2x primary improvement versus
original baseline. Choose conditions based on geometry/packing/output layout,
not CPU names or case IDs; no production rule before these measurements.
Any eventual rule must then pass the complete49-case paired validation and
Hadamard check; this diagnostic cannot substitute for those gates.

## Grid result and second candidate, before implementation/timing

The complete diagnostic passed all numerical/idle checks. Both filters include
21 distinct cases (18 mirror cases and3 `ajb` cases, including both primary
cases); all of them remain in the raw result, not only the originally named2.
At16MiB/f64/8T, the six regressed mirror cases run4.128-4.221ms in8x1 and
3.614-3.677ms in1x8, recovering11-14% of the whole call with unchanged
kernel/block sizes. Other mirror layouts do not show the same gain; several
c64 layouts regress15-24% under column splitting. Thus removing the domain
condition globally or simply selecting columns for all square cases is wrong.
The primary c64 cases remain much faster in rows:9.968vs46.802ms and
9.271vs33.910ms.4T column choices are less beneficial and occasionally lose
>10% in a1MiB c64 case. No sole packing/false-sharing mechanism is asserted:
packing/barrier contributions have not been counted separately.

Proposed conservative local-column predicate: only a fully occupied, detected
single L3 domain; cols>=rows; unit-stride output row run covering more than
one MR panel per worker; enough column blocks; K<=64. On this host that makes
4T keep its existing rule and allows8T to recover the six long-run layouts,
without changing narrow-output primary cases. Derive saturation from cached
physical cores per L3, not a Zen5 name or hardcoded8-thread threshold. Keep
unknown-cache fallback on the old rule. The predicate is a measured heuristic,
not a universal cost model; other CPU validation remains required.

Save the current affinity-only native binary as `/tmp/tcbench-affinity-candidate`.
Second candidate changes the shared partition predicate only, with pure boundary
regression tests. Before any second-candidate timing: repeat the same complete
single-L3 paired49-case protocol against the original binary, retaining all
engines/cases and original primary/nonregression gates. Supplement with the
Hadamard corpus at1T/4T, median20 runs after3 warm-ups, complete A/A repeats
with AB/BA order; same pinning/idle gate. No candidate is promoted unless
all required gates pass; full12T remains unvalidated.

### How TBLIS and BLIS represent the same thing (source evidence)

TBLIS, whose block-scatter-matrix design `pack/scatter.rs` cites
(arXiv:1607.00291), is the reference for this code. Its
`frame/base/block_scatter.hpp` carries

```c
struct bsmtc_params {
    std::array<len_type,2> nblock;                  // block counts per dim
    std::array<const stride_type*,2> block_off;     // per-block offsets
    std::array<int,2> ndim;
    std::array<const len_type*,2> len;              // rank-sized extents
    std::array<const stride_type*,2> stride;        // rank-sized strides
    std::array<bool,2> pack_3d;
};
void fill_block_scatter(len_type type_size, len_type nblock,
                        const stride_type* block_off, int ndim,
                        const len_type* len, const stride_type* stride,
                        len_type BS, len_type off, len_type size,
                        stride_type* scat, stride_type* bs, bool pack_3d);
```

and `block_scatter.cxx` computes the block scatter analytically in
`fill_scatter_3d`, which walks the mixed-radix index space with a
`viterator` and emits work for full and partial blocks. So TBLIS's gather
metadata is **O(blocks + rank)**: it takes the `(len, stride)` descriptors it
already has and derives per-block offsets. It does not build a per-element
offset vector, and its block granularity `BS` is the panel width, not the
register tile.

tcontract instead builds `scat` of length **elements** per role and then
`bs` of length `elements / mr`. The per-element array is the part TBLIS does
not have; that, not the existence of block scatter, is the actionable
difference. BLIS's pack kernels, examined in the same investigation, address
source elements with base+index*scale addressing and pass strides into the
kernel (`inca`/`lda`, with an explicit general-stride branch), never through
a materialised index vector.

The first survey attempt reported "no network", but that was a missing tool in
that read-only agent, not a host limitation; the survey was redone from the
main session with live sources.

* **cuTENSOR** describes each operand as a list of mode/*extent*/*stride*
triplets and has no index array; its user guide's performance guidelines ask
the caller to keep batched modes the slowest-varying and "the extent of the
fastest-varying mode (a.k.a. stride-one mode) as large as possible" -- i.e. to
maximise exactly the long contiguous run that makes a per-element scatter
pure overhead.
* **cuTENSOR plan cache**: a bounded, LRU, per-handle lookup from
`cutensorOperationDescriptor_t` to `cutensorPlan_t`, resizable
(`cutensorHandleResizePlanCachelines`), serialisable to disk, and disableable
per contraction (`CUTENSOR_CACHE_MODE_NONE`). The documented purpose is
"minimize launch-related overhead", and it is called out as "particularly
helpful if the same contraction is planned multiple times in the same
application". That is the industry-standard answer to the dynamic-shape
problem, and its shape (owner, bound, clear/disable, stats) matches this
repository's `Cache Ownership` rules.
* **CUTLASS**: the batched path passes one batch stride for the strided form,
or an array of pointers; no per-element index structure.
* **opt_einsum** / numpy-style einsum: reduce to reshape + transpose +
`tensordot` (BLAS). **ITensor/NDTensors** likewise permutes and calls GEMM,
handling memory permutations internally. **CTF** uses cyclic/blocked layouts
and local BLAS kernels, with a performance model choosing the layout.
* The block-scatter-matrix format itself is from the TBLIS paper
(Matthews, arXiv:1607.00291), confirmed by its own text; the
Strassen-for-TC follow-up (arXiv:1704.03092 / SISC18) also describes a
Block-Scatter-Matrix format.

So a per-element gather index array is **not** the norm across libraries; the
norm is `(extent, stride)` descriptors plus, where a gather is genuinely
needed, a scatter at block/panel granularity. tcontract is unusual only in
materialising the per-element offsets at plan time and in having no plan
cache.

## (#16) What the45% actually is: not the driver loop

Two probes, both in `experiments/avx512-ukr-asm/`:

| measurement | per core | aggregate |
|---|---:|---:|
| kernel, L1-resident panels (`multicore`) | 52.6 | 421.1 (8T) |
| kernel, streaming (`stream`, 1T, kc 64 / 256) | 49.6 / 48.3 | - |
| same, 8T, workers sharing the two read-only buffers | 22.8 | 182.1 |

Retained as `experiments/avx512-ukr-asm/results/stream-{1t,8t}.txt`; the
footprints are **256 MiB of A and 32 MiB of B**, chosen so every call reads a
fresh panel, and the 8T run has all workers reading the *same* two buffers --
not eight independent per-thread panel streams. Earlier session readings of
47.7-49.9 (1T) and 19.4-24.7 per core / 155-198 aggregate (8T) were taken at a
32 MiB + 4 MiB footprint and are quoted as **historical, not retained**.

Library ablations on the one corpus case that is a plain GEMM
(`ij-ik-kj`, m1464 x n1448 x k1464, f64,16 MiB,8T CPU4-11, packed,
warm-up plus5 reps, pinned):

| configuration | ms | GFLOP/s |
|---|---:|---:|
| default (`nc` 1536) | 24.482 | 253.5 |
| `TCBENCH_WRITEBACK=gather` | 25.145 | 246.9 |
| `TCBENCH_NC=512` | 25.381 | 244.6 |
| `TCBENCH_NC=256` | 26.708 | 232.4 |
| `TCBENCH_KC=64` | 26.956 | 230.3 |
| `TCBENCH_KC=128` | 25.188 | 246.4 |
| `TCBENCH_KC=192` | 24.631 | 252.0 |

Reading: forcing the slow output path costs **2.7%**, so the write-back is not
the gap. Shrinking `nc` from1536 to256 multiplies A repacking by six and costs
**9%**; the extra ~86 MB of panel traffic then implies ~39 GB/s aggregate for
the pack loop, which bounds the default packing at a few percent. `kc` is
**better** large (256 beats64 by10%), so the micro-panel exceeding L1 is not a
problem here.

Therefore the earlier framing -- "45% of the whole call is driver, packing and
write-back" -- was **wrong**, and the measurements are the reason it is
retracted. The controllable driver stages account for roughly a tenth. The
rest is the kernel's own per-core rate falling from ~50 to ~30 when eight
cores run it against real panel traffic: the single-core kernel streams a47-50
GiB/s class of panels without difficulty, but eight cores doing the same thing
reach only155-198 GFLOP/s in the probe, and the library lands in the same
band at253 GFLOP/s. That is a memory-system/locality effect, not a loop in
`driver/`, and `driver_sim.rs` (a model of the nest with the driver stages
switched off) reproduced the library's aggregate while being *more* pessimistic
than the library in absolute terms, so it is retained only as an ablation, not
as a model.

Caveat on precision: repeated runs of the identical `stream 256` configuration
read 35.7 and 48.0 GFLOP/s per core **(historical session readings, no retained
output; the retained warmed run reads 48.3)**. The multi-core numbers move by up to30%
between runs, so the apportionment above is qualitative -- the ablations bound
the driver stages, but no precise percentage should be quoted, and no absolute
rate without its warm-up and run order.

Note also that the ablations required rebuilding `tcbench`, which a previous
`cargo clean -p tcontract-bench` had removed; the first attempts silently produced
no data because the binary was missing.

## Evidence index (contract 1)

A PR for this work must link these rather than restate them. Most numbers in
this file are traceable to one of them; the exceptions are labelled
"historical, not retained" in place, and a claim without that label and
without an artifact should be treated as unsupported:

| claim | evidence |
|---|---|
| primary scaling fix, all49 cases | `benchmarks/benchmarks/tcbench/results/scaling-single-l3/` (paired A/A, `comparison.txt`, `vs-baseline.csv`), `results/scaling-layout/` |
| partition predicate isolated from the driver | `results/grid-diagnostic/` (same binary, forced grids) |
| plan-time scatter cost, the reverted primitive, and the fast path | `experiments/scatter-cost/results/` (probe + `manifest.txt`; self-contained because the primitive was reverted) and `benchmarks/benchmarks/tcbench/results/scatter-fastpath/` (the `packed_plan` rows) |
| kernel ceiling, clock ramp, driver share | `experiments/avx512-ukr-asm/results/` -- `bench-1t.csv`, `peak.txt`, `multicore-{1t,8t}.txt`, `stream-{1t,8t}.txt`, `driver-sim-8t.txt`, plus guard logs and a manifest |
| library ablations (write-back, `nc`, `kc`) | `benchmarks/benchmarks/tcbench/results/ablations/` (nine CSVs + `manifest.txt`); reproduce with `TCBENCH_WRITEBACK` / `TCBENCH_NC` / `TCBENCH_KC` on `ij-ik-kj` |
| rejected OpenBLAS import | `experiments/openblas-kernel/results/2026-10-07{,-wide,-invalid}/`, `docs/worklogs/2026-10-07-openblas-kernel-import.md` |
| rejected scatter wiring | `benchmarks/benchmarks/tcbench/results/scatter-wired/comparison.txt` (the paired run that rejected it). The reverted sources were an out-of-tree archive of the attempt and are **not** retained here; the rejection rests on the retained comparison, not on the archive |
| host, build flags, guard logs | each result directory's `manifest.txt`, `*.guard` |

## Ledger of attempts that were rejected

Every entry was implemented (or measured) and then dropped on evidence. None
of them is shipped, and none of the numbers below was re-selected.

| attempt | what was measured | outcome |
|---|---|---|
| Import OpenBLAS's Zen5-selected f64 kernel (16x2 and 16x12 with B repacking) | full 3-case f64 A/A at 1/4/8/12T | **slower everywhere**: ratio 0.61-0.81. `experiments/openblas-kernel/`, worklog `2026-10-07-openblas-kernel-import.md` |
| Freeze the compiler's AVX-512 ukr assembly and ship it | pinned 1T, kc 16/64/256 | **identical to intrinsics** (52.49/52.56, 52.69/52.70, 52.71/52.70 GFLOP/s). Nothing to gain; `experiments/avx512-ukr-asm/` |
| Hand-written assembly as the next lever | kernel ceiling probe | **deferred, not attempted**: the kernel keeps 47.7-52.7 GFLOP/s even streaming from 32 MiB, so there is no demonstrated headroom to hand-tune against |
| Attribute the end-to-end gap to the driver loop (my own earlier claim) | library ablations on `ij-ik-kj` 8T | **refuted and retracted**: write-back 2.7%, A repacking 9%, and `kc` is *better* large |
| `SegmentedScan`-style coarser write-back blocks | -- | not attempted; superseded by the ablation above |
| Replace the plan's per-element `scat` with axis-derived blocks (the wiring) | full 49-case paired run | **execution regression, reverted** -- see the next section |
| Keep the wiring and instead fix only the write-back | -- | not attempted: the write-back panel copy was estimated at ~1.3% and the measured regression is 19%, so the estimation was wrong and the mechanism is elsewhere in the wiring |
| Use a smaller `kc` so the micro-panel fits L1 | `TCBENCH_KC` sweep 64-256 | **refuted**: `kc` 256 is 10% *faster* than 64 |
| `BlockModel::Analytical` blocking for this host | - | not attempted; the family constants beat `kc` 64 by hand already |
| Attribute the residual 1T/4T regressions to the partition predicate | source trace of `local_columns` | **refuted**: the predicate cannot fire at 1T or 4T on this host |
| `Cargo.lock`/dependency difference as the cause of the plan-time delta | lock-file diff | **refuted**: only the optional `tensorcontract` git dependency differs |
| Let TBLIS/BLIS auto-detect the host configuration | configure output | **rejected before any timing**: auto detection selected a generic configuration, so the shipped baseline is an explicit `zen3` build instead |
| Integrate AMD AOCL's Zen5 BLIS as the TBLIS backend | configure attempt | **blocked, no timing numbers**: AOCL supports Zen5 but does not provide the TBLIS 2.x plugin API. `docs/worklogs/2026-10-07-zen5-tblis-blocker.md` |
| LTO as the cause of the plan-time delta | thin-LTO vs `LTO=false` | **confirmed as code layout**, not an algorithm change, and not stable across build configuration. The raw run list and the symbol-shift counts were session notes, not retained artifacts: treat those particular figures as unsupported |

Discarded measurements, kept because they document the protocol failing rather
than a result: `results/scaling-baseline-invalid/`,
`results/scaling-layout-build-invalid/`, `results/2026-10-07-wide-invalid/`
(12T idle gates, and one binary that did not contain the change under test),
`results/scaling-layout/hadamard-timeout/` (a shell timeout after the useful
repeat had completed), and three `bench`/`tcbench` ablations that silently
produced no data because `tcbench` had been removed by a `cargo clean`.

## Rejected: the axis-derived scatter wiring (reverted)

What it was: `Scatter { Run, Full }` plus `ScatSlice`, replacing the plan's
twelve `Vec<i64>` scatters and the `&[i64]` borrowed by `Ctx`, `pack_panel`,
`pack_b_needed`, `scale_only` and `packed_report`. The full attempt is saved
outside the tree (`/tmp/scatter-wiring-attempt/`, 17 files, 2296 diff lines)
and was reverted with `git checkout HEAD -- <files>`.

What it achieved: **plan construction collapsed from `O(elements)` to
`O(rank)`**, which was the point.

| case, `packed_plan` 1T CPU4 | before | fast path only | wired |
|---|---:|---:|---:|
| `hadamard_vec_2e24_f64` | 190.2 ms | 142.5 ms | **2.7 us** |
| `hadamard_mat_2048_f64` | 47.1 ms | 41.7 ms | **2.8 us** |
| `hadamard_mat_256_f64` | 0.211 ms | 0.465 ms | **3.0 us** |

The plan-time columns are historical observations (see the note under the
fast-path table); the wired column is the session reading, and the wired
binary is not retained because the wiring was reverted. What is retained and
reproducible is the *reverted* run's plan-time rows above, which is the state
this branch actually ships.

Why it was rejected: the complete paired49-case run (baseline = the saved
pre-change binary, `1/4/8T`, `1/16 MiB`, both dtypes, AB/BA) reported **139
individual regressions above both A/A spreads**, 0.75-0.91x, concentrated in
the `abcijk` family and in1T/4T, against group geometric means of 0.889-1.089.
The primary cases still passed (4.533x, 2.867x), so the primary gate did not
catch it; the non-regression gate did.

The mechanism was then isolated by rebuilding each arm and re-measuring one
representative case (`abcijk-eiab-jkec`, 16 MiB, f64, 1T, packed, reps5,
pinned CPU4):

| binary | time | GFLOP/s |
|---|---:|---:|
| pre-change baseline | 19.439 ms | 29.5 |
| scatter-wired | 23.937 ms | 24.0 |
| reverted (fast path kept) | 19.356 ms | 29.6 |

So the wiring cost 19% on the packed execution path, and reverting restores
the baseline exactly. I had estimated the write-back's per-tile panel copy at
~1.3% and therefore expected the packer's `ScatSlice` change to be harmless;
that estimate was wrong, and the honest reading is that introducing an
indirect accessor into the two hottest loops (pack and write-back) cost more
than the plan-time work it removed. The plan-time win is real but it is paid
for in execution, which is the opposite of the required trade.

What is kept: the additive, execution-neutral part -- the single-axis fast
path in `build_scatter` (`crates/tcontract-kernel/src/pack/scatter.rs`), with
its own regression test, worth 1.4-2.7x on plan-time scatter construction and
**no** change to any executed instruction path. Verified after the revert:
2e24 = 142.5 ms, mat_2048 = 41.7 ms, and the representative case back at
19.356 ms.

## Kernel ceiling and the driver share (#16)

Probe: `experiments/avx512-ukr-asm/src/bin/multicore.rs` -- the *same* intrinsic
micro-kernel, data and loop on N threads, reporting per-thread and aggregate
GFLOP/s. Pinned by the usual protocol.

| threads | per-core | aggregate |
|---|---:|---:|
| 1 (CPU4) | 52.7 | 52.7 |
| 8 (CPU4-11) | 52.6 | **421.1 GFLOP/s** |

Per-core throughput is unchanged from1T to8T, so there is **no** clock or power
throttling with all eight cores running AVX-512 FMA, and the kernel is not
limited by memory at this size. End-to-end, a1024^3 f64 GEMM runs in ~9.32 ms
at8T = 230.5 GFLOP/s, i.e. 54.8% of the kernel-only ceiling. The first reading
of that gap -- "about45% of the whole call is driver, packing and write-back"
-- is **retracted** by the ablations in the next section; only the measurement
above survives. What is supported: the remaining efficiency is *not* in
hand-written assembly, and it is not in the driver's identifiable stages.

### Measurement caveat found while doing this (affects the whole protocol)

The first version of the probe (no warm-up loop; that build is not retained,
so the reading is quoted as a historical session observation) reported 39.8
GFLOP/s per core at every thread count -- a 25% under-read. The cause is **clock ramp, not thread migration**:
the affinity mask at1T is CPU4 alone, and `ps -L -o psr` showed every thread
parked on its assigned CPU across repeated samples (4 for1T; 4-11, one each,
for8T). What differed was warm-up: after the three-second idle gate, a ~1.2 s
FMA burst measures the ramp rather than the settled clock, and does so
reproducibly (session notes: 39.8 four times under `pinned.sh`, 48.4-49.6 for
immediate back-to-back direct runs; neither set is retained). With a two-second warm-up loop the same binary reads
52.7/52.6 GFLOP/s. `bench.rs` reads52.7 because its first kc case warms the
core before the second is timed.

Consequences: (a) a microbenchmark whose whole measurement is shorter than a
couple of seconds after an idle gate under-reports, and the effect is large
enough to be mistaken for a kernel defect; (b) arm-versus-arm ratios remain
valid only when both arms get the same warm-up, so an A/B whose two arms have
different durations can be skewed; (c) no absolute rate from this session
should be quoted without its warm-up. The tcbench protocol runs a warm-up
before its five repetitions, so its comparisons stand, but its first cases in
each process are the most exposed.

## Axis-driven block scatter, measured (first step of the plan-cost fix)

**Reverted.** The whole of this subsection describes the axis-driven primitive
that was removed from the tree; it is kept as the record of what was measured,
not as a description of this checkout. `experiments/scatter-cost/src/main.rs`
carries its own copy so the numbers below stay reproducible.

The primitive was, in `crates/tcontract-kernel/src/pack/scatter.rs`:
* `role_len(extents)` -- the product `build_scatter` would allocate, without
allocating it.
* `block_strides_from_axes(extents, strides, blk, out)` -- the block-scatter
entries, walked straight off the mixed-radix axes. It steps one position at a
time and keeps only the running block, so the per-element offset vector is
never materialised and never written. Single-axis roles skip the odometer
entirely.

Equality with the old path is the contract, so the test asserts it over
shapes that carry, tie, repeat, straddle blocks and overflow
(`block_strides_from_axes_matches_the_materialised_path`, plus a partial-tail
case); it caught two real bugs during development (`blk = 1` blocks have no
difference to report; and the walker has to reposition onto each block start
without treating that step as one of the block's differences).

Isolated measurement, `experiments/scatter-cost`, pinned1T on CPU4, best of7,
equality asserted in the same run (`old` = `build_scatter` + `append_block_scatter`):

The retained run is `experiments/scatter-cost/results/reverted-primitive.csv`
(per-arm time-based warm-up; the earlier no-warm-up run is kept beside it as
`reverted-primitive-no-warmup.csv` and is protocol-deficient):

| role | elements | old ms | new ms | speedup |
|---|---:|---:|---:|---:|
| one axis, 2^24 | 16 777 216 | 39.93 | 0.637 | 62.7x |
| one axis, 2^20 | 1 048 576 | 0.522 | 0.0398 | 13.1x |
| one axis, 230400 | 230 400 | 0.104 | 0.0088 | 11.7x |
| two axes, 256x256, blk24 | 65 536 | 0.0879 | 0.0991 | **0.89x** |
| three axes, 16x8x4 | 512 | 0.0010 | 0.0009 | 1.16x |

**What survived is narrower than the original plan.** Everything below that
is marked *reverted* was measured and then removed from the tree; only the
`build_scatter` fast path is in this PR:

1. ~~`block_strides_from_axes` + `role_len`~~ -- **reverted.** These were the
   axis-driven primitive; they are no longer in the tree, and the probe that
   measured them (`experiments/scatter-cost`) now carries its own copy so its
   historical numbers stay reproducible. The primitive's measured effect is
   recorded below as *reverted*, not as a property of this checkout.
2. A single-axis fast path **inside `build_scatter`** -- **this is the part in
   this PR**, and it needs no API change: `role_axes` already folds compatible axes, so a role with a large
   element count arrives as one axis, and one axis is a ramp rather than an
   odometer. Measured effect on the materialise-then-compress path, re-run in
   the same binary and order as before:

| role | elements | before the fast path | after |
|---|---:|---:|---:|
| one axis 2^24 | 16 777 216 | 55.02 ms | 39.89 ms |
| one axis 2^20 | 1 048 576 | 1.309 ms | 0.550 ms |
| one axis 230400 | 230 400 | 0.287 ms | 0.107 ms |

Those three rows are the *original* session observations; their raw output was
not retained, so they are quoted as historical, not as reproducible. The
retained, reproducible measurement of the fast path is the library-level table
below, and the retained equivalent of this micro-table is the "old ms" column
of `reverted-primitive.csv`, which is the same `build_scatter` plus
`append_block_scatter` path measured before and after the fast path.

The remaining cost there is `append_block_scatter` re-deriving block strides
from the materialised vector (an `O(elements)` scan of 24-entry windows), not
`build_scatter`. Removing that is what the reverted primitive would have done,
and it is **not** in this PR. The numbers in the table are the *reverted*
measurement, retained as the record of why the fast path was worth keeping on
its own.

**Why the wiring was first deferred and then rejected.** Recorded in two
stages, because the reason changed: see "Rejected: the axis-derived scatter
wiring (reverted)" above for the execution regression that finally rejected
it, and this paragraph for the need gate that had deferred it. Deferral
reasoning: The consumers hold `&[i64]` borrowed from the
plan: `driver/mod.rs` (`Ctx.am`/`ak`/`bn`/`dm`/`dn`), `tile.rs` (panel slices
`&am[ic..ic+ic_len]`), the batch loop (`h_d[h]`), `packed_report`, and
`select.rs` — 34 sites across 7 files outside tests. Replacing the stored
vector with an axis description therefore changes the driver's borrow
structure and the packer signatures, and `PERFORMANCE_TIPS.md` requires a
complete paired49-case benchmark for a change to the packed driver, not a
spot check. Against that: the affected population is a contraction whose
*folded batch role* is large under the forced-packed route, which no corpus
case is (all49 have batch elements of1), and the plan-time cost for the
largest corpus roles is now ~0.1-0.3 ms against millisecond-scale execution.
The need gate was not met, so the refactor is recorded rather than landed.
Effect at library level, `packed_plan` (plan construction, forced-packed route,
1T CPU4, `BENCH_RUNS=5`/`BENCH_WARMUP=1`, `benchmarks/benchmarks/tcontract/corpus/hadamard.json`),
against the recorded pre-change values from the same protocol:

| case | before | after the fast path |
|---|---:|---:|
| `hadamard_vec_2e24_f64` | 190.2 ms | **142.5 ms** |
| `hadamard_mat_2048_f64` | 47.1 ms | 41.7 ms |
| `hadamard_mat_256_f64` | 0.211 ms | 0.465 ms |

The "after" column is the retained run in
`benchmarks/benchmarks/tcbench/results/scatter-fastpath/plan-packed_plan.txt`.
Repeat runs of the same configuration read 131-142 ms for the 2^24 case, so
quote the range, not a single figure. The "before" column is the pre-change
value recorded during the investigation; its raw output was not retained, so
it is quoted as historical -- the *direction and order of magnitude* are
reproduced by the retained probe's `old ms` column above, which measures the
same `build_scatter` + `append_block_scatter` path.

The2^24 case is the informative one: a1.45x whole-plan improvement is far
above any layout or run-order effect this session has seen, and its direction
matches the microbenchmark. The two smaller cases sit inside the cross-build
noise band already documented (an isolated `mat_256` reading was730 us against
an in-context211 us before this change), so they are reported, not summarised.

The residual is now understood: for2^24 the plan still allocates and fills
four batch-role scatter vectors of16 777 216 entries each -- 536 MB of writes
-- which at131 ms is about4 GB/s and therefore page-fault/memory bound rather
than CPU bound. Speeding up the fill cannot remove traffic that the
representation demands; only the wiring does. That is the strongest argument
for the deferral being a *measurement* decision rather than a choice to stop:
the case that would justify the refactor is exactly the case the corpus does
not contain, and it is now quantified at536 MB per plan.

The primitive was reverted with the wiring, so nothing of it is in the tree;
the fast path above is the whole of the production change.

A single-axis role was 12-63x cheaper with the (now reverted) axis-driven
path. The second row is the honest exception:
when the role genuinely has several incompatible axes the new walk is ~12%
slower than materialising, because it pays per-block bookkeeping that the
bulk `Vec` fill does not. That shape is what `fold_axes` removes whenever the
strides are compatible, and the two-axis case above is written without
folding to expose the worst case; the wiring step must therefore keep the
materialised path for a role that survives folding with more than one axis
rather than switch unconditionally.

Scope of the reverted attempt: it would have removed one `O(elements)` pass
and the `O(elements/blk)` block buffer per role, but **not** the per-element
`scat` itself, which the packers, the write-back and the batch loop still index.
Finishing the job means the plan stops storing `scat` for a role it can
describe by `(len, stride)` -- the same information `stats.*_axes` already
holds -- and the consumers generate the panel slice they need. That is a
wiring change across `plan/analysis.rs` and `driver/`, and it needs its own
correctness and non-regression evidence before it is promoted.

## AVX-512 micro-kernel assembly experiment (predeclared)

User requests freezing this host's compiled kernel assembly as an optimized,
distributable kernel, plus further assembly-level tuning. Source facts: the
f64 AVX-512 micro-kernel is generated by `simd_kernels!` from
`core::arch::x86_64` intrinsics under `#[target_feature(enable="avx512f")]`,
so it does **not** depend on `-C target-cpu=native`; the `native` flag affects
the rest of the binary, and a distributable build needs a baseline target for
that part regardless. Freezing assembly is therefore an optimization question,
not a portability fix, unless the frozen instructions are measured to differ.

Protocol, fixed before the experiment's timings: isolated crate
`experiments/avx512-ukr-asm/**` (own workspace, no production edits); emit the
`real::<3,8>`/`tramp_real::<3,8>` bodies with `--emit=asm` for both native and
a baseline target and diff them; assemble the frozen body with the system
assembler; correctness first (K=0,1,2,3,7,16,33 and exact small integers, plus
a256x256 case, relative Frobenius residual against a naive reference); then
microbenchmark intrinsic versus frozen assembly at1T CPU4 and4T CPU4-7,
pinned, warm-up plus at least five repetitions, best wall time, all results
retained including losses. No kernel-family adoption or production default
changes without a separate end-to-end measurement; a micro-kernel result alone
is effect, not adoption evidence.

## Asm experiment result: freezing the compiler's assembly buys nothing

Isolated experiment `experiments/avx512-ukr-asm/` (own workspace, no
production edits). Correctness passed first: `intrinsic` and `frozen` both
match a naive reference for kc in {0,1,2,3,7,16,33} with exact small integers,
and the two were cross-checked at the measured kc.

Pinned1T on CPU4, 7 reps, warm-up present, guard logs kept
(`results/bench-1t.csv`):

| kc | intrinsic best/mean ns | frozen best/mean ns | GFLOP/s |
|---|---:|---:|---:|
| 16 | 117.04 / 133.00 | 116.89 / 116.93 | 52.49 / 52.56 |
| 64 | 466.41 / 467.12 | 466.32 / 466.65 | 52.69 / 52.70 |
| 256 | 1865.18 / 1865.85 | 1865.34 / 1867.96 | 52.71 / 52.70 |

Both arms are equal to within noise at every kc. The reason is visible in the
retained `asm/native-vs-baseline.diff`: the two target configurations do not
produce byte-identical text (stack setup order and `vxorpd xmm` vs `vpxord
zmm` zeroing differ), but the arithmetic loop is the same instruction
sequence, so pinning one variant cannot be faster. **Conclusion: embedding
this host's compiled kernel as "the optimized kernel" has no measured benefit;
it buys build reproducibility only.** A hand-written assembly kernel would have
to beat the intrinsic body on its own merits.

Machine ceiling, for context: a separate probe (`src/bin/peak.rs`) saturates at
**37.6 GFLOP/s** in pure 512-bit FMA chains (4 chains suffice; more do not
help), with a dependent single chain at 9.4 GFLOP/s. The micro-kernel reaches
52.7 GFLOP/s, i.e. **above** the pure-FMA probe. The two numbers are not
reconcilable as a simple issue-width limit, and the likely explanation is
clock behaviour under a pure-FMA power load on this mobile APU; the probe is
retained as evidence but is **not** used as a ceiling claim. What can be said
without overclaiming: at 24x8 and kc=64 a k-step costs ~7.3 ns, no freeze
variant changes that, and the end-to-end8T figure (~231 GFLOP/s for a1024^3
GEMM) is well below8x the kernel's single-core rate. An earlier reading put
that gap in the driver/packing/planning path; the ablations above **retract**
it, and what the probes support is the kernel's own rate falling under real
multi-core panel traffic.

Direction: do not invest in hand assembly on this evidence. The measured
inefficiencies are (a) plan-time O(elements) scatter materialisation (above),
and (b) the gap between kernel rate and whole-call rate.

## Build-identity failure, not a usable second-candidate comparison

The first `scaling-layout` complete numeric/idle-guarded pair did not contain
the intended kernel library: the executable's pinned `info` still reports
CPU0's16MiB/8-logical L3, and its text lacks `/proc/thread-self/status`,
whereas the saved affinity-only executable reports the corrected cache and
contains that string. Local package artifacts had been built using a shared
target across the baseline worktree and candidate. Exact cache-invalidation
cause is not established; do not treat the resulting metrics as this candidate.
Retain the whole result as `scaling-layout-build-invalid/` and the saved
invalid binary separately. Rebuild the three local packages and check the
actual8T cache report **before** rerunning the entire unchanged protocol.

The subsequent Hadamard launch used an incorrect repository-relative path
and failed before cases ran. The actual repository path is
`benchmarks/benchmarks/tcontract/corpus/hadamard.json`; use its absolute path
and record its hash. This is a harness setup error, not a numerical result.

Rebuilt native candidate now reports8MiB L3/16 logical sharers and1 domain
at8T before measurements. Kernel/contract release tests388 passed;
clippy/fmt/skill-mirror checks passed. The runner now saves each binary's
pinned1/4/8T info along with hashes to expose cache/build mismatches.
Hadamard's existing harness caps requested20 repetitions at10 when work is
>=2^24; thus the2^24-vector cases use10, all others20, with3 warm-ups.
Both arms use the same cap and all24 entries, both plan/packed variants.

## Correctly rebuilt second candidate: partial success, not promoted

Complete fresh49-case correctness and idle guards passed. The primary cases
improve4.589x and3.091x.8T whole-case ratios:1MiB f641.111/c641.157;
16MiB f641.002/c641.089. The six16MiB/f64 grid regressions are recovered;
all size/dtype/thread group gates pass. Three individual directional-A/A
flags remain, all at1MiB/f64 and1T or4T, where the domain/grid change does
not apply. Two are repeatable ~10% differences (`abjc-cbka-kj`1T and
`ajbc-ckba-jk`4T); the third `abjc-kbac-jk`4T has one plan outlier
0.383ms versus0.102ms, while its packed control stays0.097-0.100ms.
These are unresolved; do not call the original promotion gate passed.

The combined long command timed out only after the full contraction pair was
completed and during the second Hadamard repeat. Keep the incomplete
Hadamard attempt under `hadamard-timeout/`; it supplies no complete A/A gate.
A complete Hadamard-only paired rerun passed all192 whole-output checks and
all idle guards. Default execution geometric means are0.993 at1T and0.989
at4T; packed execution1.039 and1.007. One4T/1024-element c64 default case
exceeds10% above noise; forced-packed planning is slower in22/24 cases at
each width (geometric ratios0.791/0.799). All planning rows remain recorded
although the contraction primary metric excludes planning.

Hadamard candidate was built with optional upstream/tblis features whereas
its original saved baseline used default features. Dynamic library lists
are equal, so this alone is not evidence of the slowdown's cause; nevertheless
match the baseline's feature configuration before drawing a shipping-config
conclusion. Before a further complete Hadamard A/A rerun, build a default-
feature native candidate and preserve the previous full observations under
`hadamard-provider-features/`. Keep repetition/idle/case gates unchanged.

The complete default-feature Hadamard rerun also passed all192 numerical
checks and all guards. Execution ratios are1T plan0.994/packed0.995 and
4T plan1.002/packed1.013. Most execution is unchanged, but the4T
1024-element c64 default case still exceeds the individual noise gate.
Forced-packed planning remains slower in22/24 cases at each width, with
geometric ratios0.807/0.783. Thus feature matching alone does **not** resolve
the observed planning differences. This is retained as an open finding,
not attributed to the new partition predicate, which is not called by
plan construction. Default elementwise planning is approximately unchanged.

## Plan-construction slowdown: cause is thin-LTO code generation

All measurements here use native release, `hadamard_mat_256_f64` at1T on CPU4,
`BENCH_CASE`/`BENCH_RUNS=20`/`BENCH_WARMUP=3`, each binary measured under the
normal pinned/idle guard. Median `packed_plan` (plan construction) in ns:

| binary | build setting | runs |
|---|---|---|
| baseline (saved) | thin-LTO | 723250, 731126 |
| baseline (fresh rebuild) | thin-LTO | 732758, 723030, 729943, 720765 |
| candidate | thin-LTO | 861809, 862720, 860576, 873831, 863431 |
| baseline | `CARGO_PROFILE_RELEASE_LTO=false` | 775127, 774095, 774086 |
| candidate | `CARGO_PROFILE_RELEASE_LTO=false` | 745211, 726306, 727900 |

With thin-LTO the candidate is ~+19%; without LTO the candidate is **faster**
than the baseline and the baseline itself moves from ~728us to ~774us. The
difference is therefore a code-generation/layout effect of thin-LTO, not an
algorithmic regression, and it is not stable against a build-configuration
change.

Supporting isolation (all native release, same case):

* `partition_with`, the only function the change touches, is executed **only
during execution**: an `eprintln!` trace printed once between the
`packed_exec` header and no occurrence before `packed_plan`.
* Phase instrumentation inside `plan_packed` attributes ~850us of an ~880us
`Plan::new` to `PackedPlan::from_problem` and only ~47us to `resolve`.
* Candidate with only the affinity probe change, and candidate with only the
partition change, each reproduce the baseline ~728us; only both together
show ~870us. A semantics-preserving edit to the baseline
(`k <= BANDWIDTH_BOUND_K` written as `k < BANDWIDTH_BOUND_K + 1`) does not
reproduce it.
* `Cargo.lock` differs between the two trees only by the optional
`tensorcontract` git dependency, which the default-feature contract binary
does not build.

Consequence for the promotion gate: `packed_plan`/`plan_plan` rows are
plan-construction cost, not the contraction primary metric, and they are not
reproducible across build configuration. They must not be used to reject the
candidate, and equally the candidate must not claim the no-LTO win. The
methodological cost is that plan construction cannot be compared across
separately built binaries at this resolution; compare it only within one
build, or fix the build configuration. The shipped configuration is thin-LTO,
so a shipping conclusion needs one of: a deliberate deterministic alignment
flag, or removal of the code that perturbs `tcontract` codegen.

## Why plan construction costs microseconds to hundreds of milliseconds

`PackedPlan::from_problem` is not complex; it is **linear in the number of
logical elements**, because `build_scatter_for`/`build_scatter` materialise one
`i64` scatter entry per element of each role, for twelve (role, operand)
combinations (`analysis.rs:252-263`). The four batch-role scatters alone are
`4 * elements * 8` bytes of writes plus an index odometer pass.

Measured `packed_plan` (plan construction, packed route) on the baseline,
1T/CPU4, over the Hadamard corpus -- linear in element count at roughly
11 ns/element, or ~2.7 GB/s of scatter writes. **These are historical session
observations whose raw output was not retained**, the two largest rows
included; the retained, reproducible equivalent is the "old ms" column of
`experiments/scatter-cost/results/reverted-primitive.csv` and the plan-time
table in the reverted-attempt section, which reproduce the same scaling from a
retained artifact:

| case | elements | packed_plan |
|---|---:|---:|
| `hadamard_vec_2e10` | 1 024 | 0.008 ms |
| `hadamard_vec_2e14` | 16 384 | 0.06-0.16 ms |
| `hadamard_mat_256` | 65 536 | 0.211 ms |
| `hadamard_rank4_32` | 1 048 576 | 11.0 ms |
| `hadamard_mat_2048` | 4 194 304 | 47.1 ms |
| `hadamard_vec_2e24` | 16 777 216 | 190.2 ms |

The same case measured in isolation (only `hadamard_mat_256`) reads ~0.73 ms
(another historical reading, not retained), about3.5x the in-context0.211 ms. The isolated run allocates four512KiB
scatter vectors per plan, above glibc's mmap threshold, so it pays
mmap/munmap and page faults every iteration; in the full corpus the allocator
reuses freed large blocks. Plan-construction numbers are therefore
allocation-context dependent, in addition to the layout sensitivity above.

This does **not** mean every contraction pays it. Role-scatter length is the
role's element count, so a plain `M x N x K` GEMM builds only `O(M+N+K)`
entries (about `8*(M+N+K)` words), i.e. roughly0.1 ms at1024 and0.37 ms at
4096 per dimension -- not ~1 ms.

Population check: **all49 TCCG corpus cases have batch (H-role) elements of
1**, i.e. `h_ax` is empty after extent-1 filtering, so their plan cost is
`~8*(m+n+k)` entries -- at most ~28k entries, a few tens of microseconds
against millisecond-scale execution. The512-element `hadamard_vec_2e10` case
(0.008 ms) is that shape. The genuinely expensive population is the
**forced-packed all-batch route** (the Hadamard matrix/vector cases, up to
190 ms at2^24) and, in general, any contraction whose folded **batch role has
a large element count**. For the unforced all-batch route `from_problem` is
not called at all, because `all_batch()` selects `Elementwise`.

Consequence for callers: a plan is normally built once per shape, and the
system enforces no plan cache, so a workload whose shapes change every call
pays this whenever the packed route is chosen with a large batch role. The
TCCG corpus cannot exhibit that, so no existing repository measurement is
affected. Measuring a batch-heavy packed case is a prerequisite for any
optimization here.

### The materialisation re-derives information the plan already had

`build_scatter` runs a mixed-radix odometer over `(extents, strides)` to
produce the offset vector. The module documentation states the blocks are "the
block-scatter-matrix layout of Matthews (arXiv:1607.00291)", and
`run_structure` then *re-derives* a compact description: `Some((len, stride))`
when the vector is a concatenation of equal-length arithmetic runs. Its own
doc says that shape is "the shape every output scatter in the corpus has".
`unbroken_fraction` already computes regularity from `(total, len, blk)` in
`O(len)` without the vector.

Measured regularity of the corpus in the stored16MiB/8T run, for the packed
engine (`regular_a`, `regular_b` columns): 76 of98 rows have both`= 1.0`
(every aligned block an arithmetic progression), only five distinct pairs
occur, and the minimum is0.667. The largest roles are `m = 230400`, giving
`3m + 3n + 2k + 4 = 691316` scatter entries for `ajbdc-ckbad-jk`,
`abjcd-dkbac-jk` and `adbjc-cbdka-kj` at16MiB -- roughly0.55-1.9 ms per plan
depending on how warm the allocator is, against ~9 ms of8T execution.

How the value is consumed: `driver/tile.rs::pack_a_rows` slices the
**block**-scatter by block index (`a_m_bs[ic / mr ..]`) and the per-element
scatter by row (`am[ic .. ic + ic_len]`) for one `mc` panel, and
`static_grid.rs`/`dynamics.rs` index the batch scatter per batch item
(`plan.h_d[h]`). Both are affine in the index for a periodic run, so a compact
`(len, stride)` descriptor can answer the same queries without the array.
The public surface would change: `oriented_scatters` and the `scatter` module
are documented as the harness-visible description of the traversal, so any
compact form must either stay behind that API or come with an accessor that
returns the same values.

Avoidable cost, ranked (no change made yet, and no promotion without a gated
measurement): (1) do not build packed scatters when the route is not packed --
already true for the default all-batch route; (2) represent a foldable,
affine batch/contraction role by `(stride, extent)` runs instead of one entry
per element; (3) pool/reuse the large scatter buffers across plans to avoid
repeated mmap. Plans are constructed once and reused, so only workloads that
build many plans, or whose roles are tensor-sized, are affected.

Static symbol-table comparison of the two thin-LTO binaries (8259 common
text symbols) shows the change shifted the whole layout: 7846 symbols by
exactly +128 bytes,377 by +528, and `PackedPlan::from_problem` itself by +528
(entry offset within its64-byte line moving from16 to32 bytes). `.text` grew
2368 bytes. A uniform layout shift that preserves alignment for most symbols
is the signature of a front-end/uop-cache or branch-predictor aliasing effect,
not of a change in executed work -- which the trace already excluded. The
exact micro-architectural mechanism is not proven, so this is reported as
"layout-sensitive code generation", not as a named CPU effect.

## Residual1T/4T regressions run an unchanged partition path

The new predicate cannot change any decision at1T or4T on this host:
`local_columns` requires `cores == p` and `p > 1`, and the corrected probe
reports8 physical cores per L3, so at4T (`p=4`) and1T (`p=1`) it is false and
the grid is the baseline row split. Domain counts are also equal (`1` at both
widths for both probes). Therefore the three residual `scaling-layout`
individual flags -- `abjc-cbka-kj` at1T, `ajbc-ckba-jk` and `abjc-kbac-jk` at
4T, all f64/1MiB -- run the **same partition as the baseline**, so the
predicate cannot be their cause. That is a statement about the code path, not
a proof that the difference is noise: the measured cross-binary layout swing
is the only mechanism this work has identified, and it is not ruled out. The predeclared
gate's "individual regression above both A/A spreads" cannot separate that,
because each A/A spread is measured inside one binary.

A follow-up that does not relax the gate: rebuild each arm and re-measure the
flagged cases; a layout artifact is expected to move or vanish, whereas a real
regression would persist across rebuilds. This is a reproducibility check, not
a re-selection of favorable cases.

Final current disposition: keep the patch as an **uncommitted experimental
candidate**, not a promoted optimization. The8T primary/layout results are
repeatable, but individual1T/4T execution findings and forced-packed planning
still need diagnosis;12T remains unvalidated. In particular, `abjc-kbac-jk`
has a large directional A/A outlier: the original A2/A-based spread is
order-dependent, while its max/min range is275%. Do not interpret that
single flag as a reliable2x algorithmic slowdown or silently change the
predeclared gate to promote the candidate. Raw observations and original
classification remain intact. Further diagnostics should capture dispatch
and allocation/alignment observables on the unchanged paths before adding
more partition conditions.

## Promotion status (the banner's reference)

What passes:

* the predeclared **primary** gate: both representative c64 16 MiB 8T cases
  improve >= 2x above both A/A spreads -- 4.589x and 3.091x in the retained
  rebuilt candidate's paired run (`results/scaling-layout/comparison.txt`),
  with the mechanism isolated in a single binary by the forced-grid
  diagnostic. (The 4.53x/2.87x pair belongs to the *rejected* scatter-wired
  run and is not the partition candidate's result.)
* **numerical correctness**: `tcbench verify` at 1/4/8T and both sizes, all 49
  cases, zero mismatches, residuals ~1e-16;
* the workspace gate: 592 tests, clippy `-D warnings`, fmt, the skill-mirror
  check, the aarch64 check, and the MSRV build.

What does **not** pass, and therefore blocks promotion:

* the predeclared **non-regression** gate. The retained paired run records
  individual regressions above both A/A spreads, and after a rebuild of both
  arms three of them remain at 1 MiB f64 (0.905x, 0.905x and one 4T case whose
  two candidate repeats differ by 275%). The work log's own reading is that
  those cases follow an unchanged partition path, which makes the predicate
  not their cause, but it is **not** proof that the difference is build noise
  -- the cross-binary layout swing is the only mechanism identified so far and
  is not ruled out.
* 12T was never validated: three complete baseline attempts were refused by
  the host idle gate, so no 12T result exists in either arm.

Consequently this branch is an **experimental** result. Promoting it requires
either resolving the retained individual regressions under the unchanged
acceptance criteria, or a maintainer decision to accept them with the mechanism
documented. The scatter wiring is **not** part of it: it was rejected on its
own execution regression and reverted.
