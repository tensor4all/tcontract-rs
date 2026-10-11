---
name: tcontract-benchmark
description: Use when running, adding or reporting a tcontract-rs benchmark — the tcontract-bench binaries (exec_entry, blas, contract, capi_rust), the C benchmark, or tenferro-benchmark runs that compare tcontract against tenferro's default backend. Covers building, choosing idle cores in one L3 domain, pinning, paired thread counts, the A/A noise floor and recording results; the rules themselves live in PERFORMANCE_TIPS.md.
---

# tcontract benchmark

The procedure for measuring. The rules it applies are in `PERFORMANCE_TIPS.md`
(`Performance-Sensitive Tests And Benchmarks`, `Performance-Gated Experiment
Protocol`, `CPU Threading Contract`); read those sections first.

1. **Quiet host.** Never run two benchmarks at once, and do not build or test
   while measuring. Look for other users' load:
   `ps -eo pid,user,psr,%cpu,comm --sort=-%cpu | head`.
2. **Build once, native release profile.** Use the identical
   `RUSTFLAGS="-C target-cpu=native"` for every Rust provider and harness.
   (`--target-cpu=native` is not a Cargo flag.) Record the complete effective
   flags; do not reuse a baseline built without them. Native binaries are
   host-specific and must not be moved to a different CPU for measurement.
   Build C/C++ baselines in Release with `-O3 -march=native`, recording their
   actual configuration and ISA selection. The build job count depends on the host:
   set `CARGO_BUILD_JOBS` for it (for example 16 on the shared 64-core EPYC)
   instead of hardcoding `-j`; cargo and the runners read it.
   `RUSTFLAGS="-C target-cpu=native" cargo build --release -p tcontract-bench --bins`
   (the C benchmark builds itself in `benchmarks/c/run.sh`). Build
   parallelism does not change the measured thread count.
3. **Choose idle cores in one L3 domain:**
   `python3 benchmarks/scripts/idle_cpus.py pick N` prints N idle CPUs of one
   L3 domain (one hardware thread per core), for example `8,9,10,11`. Pick for
   the largest thread count of the run and use a prefix for the smaller ones
   (1T = the first CPU). A full-domain count (8 on the EPYC 7713P) needs a
   whole idle L3 domain; if `pick` exits 2, wait, do not widen across domains.
4. **Run each measurement through `benchmarks/scripts/pinned.sh CPUS -- CMD`.**
   It pins with `taskset`, checks the cores are idle before and after, keeps
   only valid runs and retries spoiled ones. The per-benchmark runners already
   use it and pair thread counts per case, one process per case:
   `benchmarks/benchmarks/tcontract/contract/run.sh OUT CPUS 1 4 8` and
   `CORPUS=file benchmarks/benchmarks/tcontract/blas/run.sh OUT CPUS 1 4 8`
   (both `benchmarks/scripts/paired.sh`, which also writes `manifest.txt`),
   `benchmarks/c/run.sh CPUS1 CPUS4 OUT`. A recorded workload is replayed
   with `CORPUS=file` (`contract --corpus`, `blas --corpus`; format in
   `benchmarks/src/corpus.rs`). For a single binary:
   `BENCH_CASE=<case> benchmarks/scripts/pinned.sh 8 -- target/release/blas --threads 1`.
5. **Thread counts:** the default comparison is **1T, 4T, 8T, 12T**, in
   that order; no 16T arm. Use one logical CPU per physical core, not SMT.
   Inspect `lscpu -e=CPU,CORE,SOCKET,CACHE` first: requested threads are not
   proof of the physical-core count. Keep 1T/4T/8T within one L3 domain,
   with prefix core sets; 12T may span L3 domains only as an explicitly
   requested full-physical-core scaling arm. This is an exception to the
   single-L3 selection in step 3: choose and check the 12 distinct physical
   cores explicitly, do not use `idle_cpus.py pick 12` when no domain holds 12.
   Label its domain count, and do not interpret 8T-to-12T as pure thread
   scaling. On the Ryzen AI 9 HX 470 host, use 1T=`4`, 4T=`4-7`, 8T=`4-11`,
   12T=`0-11`, **only if idle**; the last arm spans both L3 domains. The binary
   asserts its effective width at startup and rejects conflicting
   `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` and any
   variable of the removed library knobs (the `tcbench` harness knobs are `TCBENCH_*`); do not set them.
6. **Correctness before timing:** run exact known-value examples and the
   corpus's `verify` command at every requested thread count before timed
   runs. Compare full outputs and record relative Frobenius residuals; reject
   non-finite residuals and out-of-tolerance results. Use the same case list,
   shapes, strides, dtype, input seed, alpha/beta and output semantics across
   providers. In particular a tcontract `packed` row is not an upstream
   tensorprimitives-rs measurement: pin and run upstream separately.
7. **Noise floor:** run the same binary twice on the same cores minutes apart
   (A/A) and report the spread; differences below it are not findings.
8. **Record** beside every published table: tcontract-rs commit (and tenferro-rs /
   tenferro-benchmark commits for tenferro runs), CPU model, core set, profile,
   thread counts, `pinned.sh` attempts, and the A/A spread. Raw CSVs go under
   the benchmark's `results/`, summaries in its `README.md`. Report negative
   and inconclusive results as such.
9. **tenferro-benchmark runs** (tcontract provider vs default backend) use the
   same core choice and `pinned.sh`, with tenferro's paired ABBA runner
   (`scripts/run_paired_timing.sh`) as the command and its idle-host guard
   left enabled. tenferro-benchmark's devcontainer suites do not pin by
   default; these runs do, and say so in the result.
10. **CPU affinity exists only on Linux.** On other hosts (macOS, Windows)
   `idle_cpus.py` exits 3 and `pinned.sh` runs the command unpinned with a
   note; state in the result that pinning was unavailable.

## Contraction comparison across providers

Reuse `tcbench`'s full TCCG corpus (49 rows including GEMM), not a new
hand-picked suite. Pin upstream tensorprimitives-rs to a full git SHA and
build it into the same harness behind an optional feature, so all engines
receive identical buffers. Compare tcontract `plan` (default API strategy)
with upstream tensorprimitives; retain `packed` as a separate diagnostic,
never call it upstream tensorprimitives.

Before the first timed run, record this protocol in a work log:
- all 49 cases; `f64,c64`; nominal tensor sizes **1 and 16 MiB**;
- all four thread counts **1,4,8,12** and exact core sets;
- contiguous TCCG layouts first; `--stress padded` as a separately labelled
  follow-up, not silently mixed into the primary result;
- **1500 ms of untimed priming by wall clock** per arm (`--prime-ms`, default
  1500, echoed by `run`) then **5 repetitions**, best wall time per engine/case,
  with the `spread` across those repetitions recorded (the existing `tcbench`
  statistic); complete suite repeated twice (A/A);
- timing includes execution, packing, call/FFI and provider-owned allocation;
  excludes input generation, plan/descriptor construction and tcontract pool
  construction; document scratch/pool reuse differences between providers;
- correctness: known values first, `verify` at each size/thread count;
  relative Frobenius error <= `1e-10` for `f64,c64`, finite outputs;
- idle checks: the existing 3-second window and <=5% busy per selected CPU
  **and each of its SMT siblings** (a busy sibling shares the measured core),
  3 attempts maximum; no compilation or other benchmarks during timing;
- **clock ramp**: this host drops about a quarter of its throughput for the
  first ~1-2 s of sustained AVX-512 work after an idle gate. A microbenchmark
  whose whole timed window is that short measures the ramp, not the hardware,
  and the error is large enough to look like a kernel defect (measured:
  39.8 GFLOP/s for a kernel that reads 52.7 once warm). `tcbench` primes every
  arm for `--prime-ms` of wall time before its repetitions, but that only fixes
  the arm's *own* start: the arms still run one after another in a fixed order, so
  the first one absorbs the ramp the later ones inherit, which cost 40% of latency
  at 500 ms and disappeared at 1500 ms. Any new micro-probe or ablation must warm
  up for **1.5-2 s of wall time**, not a fixed call count, must use the *same*
  warm-up on every arm, and must keep the scatter across repetitions — a slow first
  repetition is the signature of an unsettled arm.
  Never quote an absolute rate without its warm-up and its run order.
- report per-case latency, GFLOP/s, speedup vs each provider and 1T scaling;
  geometric-mean ratios are summaries, not substitutes for all rows;
- report A/A spread for every case, and treat changes below that spread as
  inconclusive. Do not choose new cases, repetitions or exclusions after
  inspecting the results. A failed validity gate makes the paired comparison
  inconclusive; retain failed observations and rerun the entire pair.

## Where a campaign result is published

This skill owns the **library-level** protocol and the host preparation that
external projects also rely on: `benchmarks/scripts/idle_cpus.py` and
`benchmarks/scripts/pinned.sh` (tenferro-benchmark's `docs/tcontract-provider.md`
names both, so keep their paths and behaviour stable), the 1T/4T rows of this
package, and the harness in `benchmarks/benchmarks/tcbench`.

The **campaign** — which commits were measured on which machines, the run
manifests that carry each measured revision, the published per-cell reports and
their staleness — lives in
[tensor4all/tcontract-benchmark](https://github.com/tensor4all/tcontract-benchmark).
Record campaign runs there with `scripts/record_run.py`, which builds `tcbench`
from a pinned checkout of this repository, verifies before timing, validates the
manifest and regenerates `result/INDEX.md`.

Two consequences for work here:

* `benchmarks/benchmarks/tcbench/results/` is historical evidence for the PRs
  that produced it and is frozen. Adding a new result directory here does not
  publish anything: no index, no staleness, no revision binding.
* A campaign number is quoted only with the commit it was measured at, the
  hardware profile and its coverage. `result/INDEX.md` in that repository is the
  place to see what exists and how current it is; a report there is a claim about
  one cell, not about a population the run did not contain.
