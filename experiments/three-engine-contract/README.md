# three-engine-contract

tcontract vs tensorprimitives-rs (`tensorcontract`) vs tenferro on the per-shape
corpus of tensor4all/tcontract-rs#61, at one thread. The corpus is our own
reconstruction of the cases in Lukas Devos's tensorcontract-rs vs tenferro-rs
figure (that data is unpublished); definitions are in `corpus()` in
`src/main.rs`.

A standalone crate: tenferro-rs pins strided-rs v0.4.4 and tcontract v0.4.5, so
the two must not be unified with the root workspace.

## Cases

Column-major throughout. Each case is a program of pairwise steps; all engines
run the same steps on the same data.

- `ikb,knb->inb` f64, i=k=n in {2,4,8,16}, batch in {16,64,256}.
- `ijk,jkl->il` f64, 8x16x8x8.
- `ij,jk->ik` c64 n=32 and f64 n=64.
- `ij,jk,kl->il` f64 n=64, order `((ij,jk),kl)`.
- MPS chain, c64, L=32, physical dim 2, uniform chi in {4,...,64}: bilinear
  `<phi|psi>` (no conjugation) from a random chi x chi left environment, two
  steps per site (`ab,asc->bsc`, `bsc,bsd->cd`), 64 pairwise contractions.

## Arms

| arm | timed boundary |
| --- | --- |
| `tc_exec` | prebuilt `tensorcontract::Plan::run`, preallocated outputs |
| `tc_call` | `tensorcontract::contract` (plans) + fresh outputs |
| `tp_exec` / `tp_packed_exec` | prebuilt `tcontract::Plan::execute_into`, `Exec::serial()`; planner default / packed forced |
| `tp_call` | `Problem` + `Plan::new` + fresh outputs per step |
| `ce_exec` | cpueinsum: prebuilt `EinsumPlan::execute_into`, kept `Scratch`, views built once per program call |
| `ce_call` | cpueinsum: `EinsumSpec::new` + `einsum_into` + fresh output per program call |
| `tf_exec` | prepared `ConcreteEinsumPlan::execute_into`, one backend session around the loop |
| `tf_call` | ordinary `einsum` per step, one backend session around the loop |
| `tf_call_spc` | ordinary `einsum`, a new backend session per step |
| `tf_call_subs` | ordinary `einsum_subscripts` with integer labels built outside timing (no string parsing) |
| `tf_prep_exec` | `ConcreteEinsumPlan::prepare_subscripts` + allocating `execute` per step (planning + allocation) |
| `tf_plan_exec` | prebuilt plan, allocating `execute` per step (allocation only) |
| `tf_eager` | `EagerRuntime` session einsum on constants, one eager session per program |
| `tf_eager_scoped` | `tf_eager` inside one `CpuBackend::with_execution_scope` |
| `tf_traced` | one traced einsum per step; trace, compile and `prepare_compiled` untimed; `run_prepared` timed inside one execution scope |
| `tf_traced_nary` | the whole program as one N-ary traced einsum with `EinsumOptimize::Path` equal to the step order (integer labels) |

tenferro uses `CpuBackend::with_threads(1)` built once per arm, never in the
timed region. Every arm is checked against a naive label-loop reference
(relative max error < 1e-10) before timing. Timing: a calibrated repeat count
of about 20 ms per sample, one warm-up sample, 11 samples, per-call median.

## Run

```bash
cargo build --release
for s in 1 2 3; do ./target/release/three-engine-contract > results/session$s.csv; done
python3 summarize.py
```

Options: `--case <substring>`, `--arms a,b`, `--sample-ms`, `--samples`,
`--seed`, `--list`, `--report` (the tcontract planner strategy per step). macOS has no `taskset`; threads are pinned to one by API
(`Plan::threads()==1` asserted for tensorcontract, `Exec::serial()` for tcontract,
`num_threads()==1` asserted for tenferro), not by CPU affinity.

## Recorded run

`results/` holds three sessions on Apple M5 Max, macOS 26.5.1,
rustc 1.96.0, release profile opt 3 + thin LTO + codegen-units 1, default
target CPU. tenferro rev 8402ad24, tensorprimitives-rs rev 8cda75e, tcontract at
`4a44a7f0` (the cpueinsum arms pin cpueinsum-rs `6364da4` and that same tcontract
rev), strided-rs c12d96fa (tcontract) and v0.4.4 (tenferro). Summary:
`results/summary.md`.

The cpueinsum arms need a checkout of tensor4all/cpueinsum-rs next to this
repository (`../../../cpueinsum-rs`); their results are in `results/cpueinsum/`.

## What this record does not cover

Unresolved in #61, recorded here rather than implied to be done:

- **One thread only, and the 4T arm is not run.** The issue asked for 1T and
  4T. This harness is 1T *by construction*, not by a flag: every tcontract arm is
  `Exec::serial()`, tenferro is `CpuBackend::with_threads(1)` with
  `assert_eq!(num_threads(), 1)`, tensorcontract asserts `pl.threads() == 1`, and
  there is no `--threads` option. A 4T arm is therefore new code in all three
  engines, and the recorded session is on an Apple M5 Max, which is not available
  for a follow-up — a run on another machine would be a different host's
  campaign, and this record's own rule is that ratios are only comparable within
  one host. What the arm would add that is nowhere else is the *tenferro*
  comparison at 4T on the recorded host; that needs that host. The width half of
  the question is measured on the zen5 host instead, at these same corpus shapes,
  by [`../faer-limit-sweep/`](../faer-limit-sweep/README.md), which compares
  tcontract' own two routes at 1T and 4T. That sweep finds the 4T ranking
  **reproducibly different** from 1T's for 18 of 79 groups (packed ahead at 4T in
  both of its recordings, faer ahead at 1T in every one of them), so the width
  matters — but a rule that used it would need the executor width at plan time,
  which a `Plan` does not have. See the 2026-10-09 row in `docs/decision-log.md`.
- **The corpus is our reconstruction.** `corpus()` in `src/main.rs` carries
  working assumptions where the figure is ambiguous, not definitions confirmed
  against the unpublished script.
- **One host, one rustc, one revision each.** The three sessions are A/A
  repeats of one host state, not a cross-host comparison.

The per-case ratios in the original figure's form, the per-engine per-call
overhead, and what reproduces from that figure and what does not are in
[`results/findings.md`](results/findings.md), generated by
[`results/findings.py`](results/findings.py) from the recorded sessions.
