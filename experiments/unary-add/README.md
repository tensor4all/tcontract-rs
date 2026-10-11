# unary-add

Per-call cost of `tcontract::unary::add` on small blocks, at one thread:

```text
D = alpha * op_A(A[labels_a]) + beta * D_old
```

One entry point for a permutation (`tensoradd!`), a diagonal and a reduction
(`tensortrace!`). TensorKit.jl calls it once per subblock, so what matters is the
per-call cost, not the bandwidth of a large block. This probe answers "what does
one such call cost, and how much of it is planning?" — the question
[tcontract-rs#82](https://github.com/tensor4all/tcontract-rs/issues/82) asks before the
one-shot C ABI (its P2) is designed.

## Two timed boundaries

| arm | what one call pays |
| --- | --- |
| `api` | `unary::add`: build the `Problem`, plan it, execute |
| `plan` | a prebuilt `Plan` + `AccumulationSource::Output`, executed |

Views are descriptors over fixed buffers and are built outside the clock. Every
arm is checked against a label oracle before any timing; a mismatch aborts the
run rather than publishing a number.

## Shapes

A small permutation with an accumulated output (`tensoradd!`'s form), a bigger
one, a site tensor's permutation, and a diagonal that is then reduced:

| shape | A | labels | D |
| --- | --- | --- | --- |
| `perm_8x8` | 8x8 | `[0,1]` | 8x8, `[1,0]` |
| `perm_32x32` | 32x32 | `[0,1]` | 32x32, `[1,0]` |
| `site_chi32` | 32x2x32 | `[0,1,2]` | 2x32x32, `[1,0,2]` |
| `trace_32` | 32x32x32 | `[0,1,1]` | 32, `[0]` |

## Run

```bash
bash experiments/unary-add/run.sh 3   # three sessions, each idle-gated and pinned
```

`run.sh` picks one idle CPU through the measured checkout's `idle_cpus.py` and
runs each session through that checkout's `pinned.sh`, so the guard log records
the window, the threshold and the SMT siblings it checked. One thread
(`Exec::serial()`), 500 ms of untimed priming per arm, best of 5 calls.

## Recorded run

`results/manifest.txt` carries the revision, host, compiler, CPU set, guard and
the method. `results/session{1,2,3}.{csv,txt,guard}` are the raw rows, the CHECK
lines and the guard logs.

The finding, on this host: **planning is three to five times the execution**.
`api` pays 4.1 µs for an 8x8 permutation and 24.8 µs for a `c64` site block,
against 0.9 µs and 9.9 µs on a prebuilt plan. Even the prebuilt plan is
microseconds for an 8x8 permutation, because the update runs as a general
contraction with a rank-zero `B` rather than as a dedicated strided pass. So the
C ABI either caches plans keyed by labels and layouts, or the small-block path
needs a lowering that skips the binary `Problem` analysis.

## What this does not cover

- One host, one rustc, one session order; the spread across the three sessions is
  1-5%, which is the band this probe can see.
- `f64` throughout plus one `c64` site case; `f32`/`c32` are not measured here
  (the API and its tests cover all four dtypes).
- Nothing about multi-threaded calls, and nothing about a plan cache: the `plan`
  arm reuses one plan for one shape, built outside the clock.
