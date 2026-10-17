# TCCG contraction benchmark

`tcbench` reuses the full 49-case TCCG corpus (CCSD, AO2MO, InTensLi,
CCSD(T), transposed output). Shapes, signed element strides, seeded random
inputs, alpha=1, beta=0 and output buffers are identical across engines.
The algorithms originate in Matthews, *High-Performance Tensor Contraction
without Transposition*, [arXiv:1607.00291](https://arxiv.org/abs/1607.00291).

| Engine | What is measured |
| --- | --- |
| `plan` | tcontract's default planner: packed or copy-free faer |
| `packed` | tcontract packed driver forced (diagnostic) |
| `ttgt` | CBLAS TTGT baseline (`--features blas`, needs a BLAS) |
| `tblis` | Actual C++ TBLIS through the direct FFI adapter, the independent reference |

The first arm measured for a case is the reference every later arm is compared
against, and `MISMATCH` in its notes fails the run; a run that selects a single arm
therefore validates nothing, and says so.

Do not label `packed` as external TBLIS.

## Fixed procedure

The source of truth is
[the tcontract-benchmark skill](../../../.agents/skills/tcontract-benchmark/SKILL.md).
Use `RUSTFLAGS="-C target-cpu=native"` and release for all Rust code. TBLIS must be
a Release build of a tagged release with an explicitly chosen BLIS configuration
family - `benchmarks/scripts/build_tblis.sh` picks one for the host (it does not use
`-march=native`) and records the tag, the commits and the artifact hash.

```sh
export TBLIS_ROOT=/path/to/native-release-tblis
export CARGO_BUILD_JOBS=8  # choose for the host
# Ryzen AI 9 HX 470: 1/4/8 physical cores in L3 #1; 12 physical cores in two L3s.
bash benchmarks/benchmarks/tcbench/run.sh \
  benchmarks/benchmarks/tcbench/results/native-12t 4 4-7 4-11 0-11
```

The runner builds once, checks known-value GEMM and all corpus outputs at
1/16 MiB and 1T/4T/8T/12T, then measures the complete suite twice sequentially.
The baselines are cargo features, not always-on arms: `--features tblis` adds
`tblis` and `--features blas` adds `ttgt`, and `--engines` selects among the arms a
binary was built with. Naming an arm this binary does not have is an error, not an
empty table. The runner above measures the two in-repo arms; a ratio against the
independent baseline is published by whichever campaign cell enables the feature.

Every process goes through `pinned.sh`; do not run another benchmark or build
alongside it. A failed idle/correctness gate stops the suite; keep failed
observations and classify the pair inconclusive. Core lists are explicit
arguments: validate them against `lscpu`, not a guessed core count. 12T across
L3s is a user-requested full-physical-core exception to single-L3 measurement.
No SMT or 16T arm.

1500 ms of untimed, wall-clock priming per arm by default (`--prime-ms`, echoed by
`run`, `0` skips it; the published procedure requires at least the default), then
five repetitions, best wall time, including packing, executor/FFI entry and
provider-internal allocation. Priming is time-based, not a call count: a fixed
number of warm-up calls removes a different share of the clock ramp in every arm,
most in the fastest one. Input generation, planning/descriptor construction and
tcontract pool construction are excluded. tcontract retains a caller-owned scratch
arena/pool. These are library execution comparisons, not isolated microkernel
timings. The CSV carries, beside the best time, the `spread` across the
repetitions, `(max - min) / best`.

The default is 1500 ms rather than 500 because 500 was measured to be too short
on this host. For a case whose call is about 2 ms, the *first* arm measured read
2.89 ms against 2.06 ms once settled — 40% higher latency, a 29% lower rate — and
the arm measured after it read 2.20 ms, so it looked 29% faster than the same work
measured on its own. That is a position bias, not a kernel difference; at 1500 ms
(and at 3000 ms) the case read 2.06 ms whether measured alone or after another arm.
It is not a guarantee for every case: the arms are still measured one after another
in a fixed order, the harness compares outputs and never compares two arms'
timings, and an arm that settles late shows up only in the `spread` column. The
diagnostic to apply by hand is that two arms whose rows carry the same family,
blocking and partition policy must agree.


CSV rows include the thread budget, dtype, the *declared* shape spec (a non-`gemm` case carries
`m = n = --n`, `k` empty, so the census matches the declaration; the case's resolved extents stay
in `macs` and the human table), GFLOP/s, latency
and tcontract strategy/kernel notes. `manifest.txt` records the base commit,
dirty status, binary SHA256, compiler, flags and actual topology;
`source.patch` records tracked local harness edits. Archive new source files
next to results too. The legacy `info` cache
and domain estimates describe the library's model, **not** actual affinity;
use the captured `lscpu` topology to interpret this heterogeneous host.

[2026-10-07 native run](results/2026-10-07-native/README.md).

## Timing-v2 evidence

The toolkit-driven runs pass a few flags this harness did not always have: `--regime` (the run label),
`--n` (the declared size spec, also the MiB target; `--size` stays as the alias), `--batch` (must be
`1`), `--family` (exact case ids), `--rows` (select and order engine rows), `--rotate-rows` (rotate the
execution order by *k*, which is how whole-process repeats see each engine first), `--statistic
best|median`, and `--census-csv` (one row per declared cell, including declared skips). The evidence CSV
gains `regime,family,order,batch,row,status,note,total_ms,per_item_us,samples_ms,rayon_threads,vendor_threads`;
`samples_ms` is the chronological per-call list as JSON and `per_item_us` is `total_ms * 1000 / batch`.
None of it changes the harness's behaviour when the flags are absent. The samples are serial observations
within one process, not replicates.
