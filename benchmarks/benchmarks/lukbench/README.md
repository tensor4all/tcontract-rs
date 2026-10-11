# Lukas Devos's per-shape corpus

`lukbench` is the second suite the
[tcontract-benchmark campaign](https://github.com/tensor4all/tcontract-benchmark)
drives out of this repository, beside `tcbench`. It runs the per-shape corpus
that [tcontract-rs#61](https://github.com/tensor4all/tcontract-rs/issues/61) rebuilt —
Lukas Devos's tensorcontract/tenferro figure shapes — with a TBLIS arm.

The case definitions and step programs are copied from
`experiments/three-engine-contract/src/main.rs` in this repository; they are
that experiment's, not new shapes. The corpus is our reconstruction of the
figure, so the shapes carry the experiment's working assumptions.

## Corpus

Column-major throughout, `alpha = 1`, `beta = 0`, fixed extents, one dtype per
case. Each case is a *program* of pairwise steps; one timed call is the whole
program.

| Family (`group`) | dtype | cases |
| --- | --- | --- |
| `ikb_knb_inb` | f64 | `i=k=n` in {2,4,8,16} × batch in {16,64,256} |
| `ijk_jkl_il` | f64 | 8x16x8x8 |
| `ij_jk_ik` | c64 / f64 | n=32 / n=64 |
| `ij_jk_kl_il` | f64 | n=64, fixed pairwise order `((ij,jk),kl)` |
| `mps_chain` | c64 | L=32, physical dim 2, uniform chi in {4,8,16,32,64} |

The MPS chain is the bilinear `<phi|psi>` (no conjugation) of two random
open-chain MPS from a random `chi x chi` left environment, two steps per site
(`ab,asc->bsc`, `bsc,bsd->cd`), 64 pairwise contractions; its inputs are scaled
by `1/sqrt(2*chi)` so the intermediates stay in range.

`lukbench info` prints the corpus as this binary sees it — case, group, dtype,
steps, MACs and the `m`/`n`/`k` the rows will carry — so a recorded run can be
checked against the corpus it claims to have measured.

## Engines

| Engine | What is measured | Feature |
| --- | --- | --- |
| `plan` | `PlanConfig::default()`: the planner's own choice | — |
| `packed` | `PlanConfig::packed()`: the packed driver forced | — |
| `tblis` | the C++ TBLIS baseline, one `tblis_tensor_mult` per step | `tblis` / `tblis13` |

The `tblis` arm binds the published `libtblis` directly
(`benchmarks/benchmarks/tcbench/tblis.rs`, shared with `tcbench`'s retired arm)
because the benchmark has to control which TBLIS and which BLIS configuration it
measures; `scripts/build_tblis.sh` pins and installs one into a prefix, and
`TBLIS_ROOT` names it. `tblis13` selects the 1.3 ABI; 1.3 and 2.x swap the
`TYPE_DOUBLE`/`TYPE_SCOMPLEX` enumerators, so the harness self-checks the ABI
at startup and refuses to run on a mismatch.

## Procedure

The source of truth is
[the tcontract-benchmark skill](../../../.agents/skills/tcontract-benchmark/SKILL.md).
Use `RUSTFLAGS="-C target-cpu=native"` and release for all Rust code.

```sh
export CARGO_BUILD_JOBS=16
RUSTFLAGS="-C target-cpu=native" cargo build --release -p tcontract-bench --bin lukbench
cpus=$(python3 benchmarks/scripts/idle_cpus.py pick 4)
benchmarks/scripts/pinned.sh "${cpus%%,*}" -- target/release/lukbench verify --threads 1
benchmarks/scripts/pinned.sh "$cpus"       -- target/release/lukbench run --threads 4 --csv /tmp/lukbench-4t.csv
```

- **A prebuilt plan and a preallocated output.** All steps are planned before
  the timed region; each timed call runs the whole program into slots allocated
  once. Planning, input generation and (`N > 1`) tcontract's pool construction are
  excluded.
- **Time-based priming, then best of `--reps`.** 1500 ms of untimed, wall-clock
  priming per arm by default (`--prime-ms`, echoed by `run` in its
  `... N ms priming ...` banner; `0` skips it), then five repetitions, best wall
  time, and the `spread` across those repetitions `(max - min) / best` in the CSV.
  The default is 1500 ms rather than 500 because 500 was measured to be too short
  on this host: for a case whose call is about 2 ms, the *first* arm measured read
  2.89 ms against 2.06 ms once settled — 40% higher latency, a 29% lower rate — and
  the arm measured after it read 2.20 ms, so it looked 29% faster than the same work
  measured on its own. At 1500 ms (and at 3000 ms) the case read 2.06 ms whether
  measured alone or after another arm; that is those cases, not a guarantee. The
  arms are still measured in a fixed order and the harness compares outputs, never
  two arms' timings, so the diagnostic to apply by hand is that two arms whose rows
  carry the same family, blocking and partition policy must agree.

  Priming is time-based, not a call count: a fixed number of warm-up calls
  removes a different share of the clock ramp in every arm, most in the fastest
  one.
- **One thread or a borrowed host pool.** `--threads 1` is
  `BenchThreads`'s serial `Exec` with no pool entry; `--threads N` borrows the
  host's pool through `Exec::rayon`. Every process goes through `pinned.sh`;
  do not run another benchmark or build alongside it.
- **`--size` is accepted and ignored.** This corpus has no size knob — the
  extents are the issue's, not a function of a nominal tensor size. The flag
  exists so the campaign's runner can pass the same command line to `lukbench`
  as to `tcbench`. It is echoed as ignored in `run`'s banner and in `--help`.
- **`verify` gates `run`.** Every arm is compared with the naive label-loop
  reference (the experiment's) before any timing, and `verify` prints
  `all comparisons within tolerance` only when every comparison is inside
  tolerance; it exits non-zero otherwise. `verify` runs one untimed pass per
  arm, so `--reps` and `--prime-ms` do not apply to it.

## CSV

`run --csv <path>` writes exactly `tcbench`'s columns:

```
case,group,dtype,engine,threads,m,n,k,macs,seconds,gflops,regular_a,regular_b,notes
```

- `macs` is the whole program's multiply-accumulate count and `seconds` the
  whole program's time, so `gflops` is the chain's throughput, not one step's.
- `m`, `n`, `k` are the equivalent GEMM shape by the same split `tcbench` uses.
  A program with more than one step has no single equivalent shape: those rows
  carry the **final step's** `m`/`n`/`k`, and `notes` says
  `steps=<n> m/n/k=final-step per-step MxNxK=<shape>[x<count>];...`.
- `regular_a` and `regular_b` are 0 on every row. `tcbench` reports the packed
  driver's block regularity, which is one number for one contraction; a
  program's steps have different values and this harness does not invent an
  aggregate. The per-step strategy is in `notes` instead.
- `notes` carries the max relative error against the naive reference
  (`rel_err=...`), the per-step strategy for `plan`/`packed`, and the chain
  suffix above. `MISMATCH` in a note means that arm disagreed with the
  reference: `run` then exits non-zero and the row is not publishable.
