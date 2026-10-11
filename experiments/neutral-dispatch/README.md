# neutral-dispatch

What the neutral contraction interface costs per call — the item
[tcontract-rs#31](https://github.com/tensor4all/tcontract-rs/issues/31) deferred:
"the paired 1T/4T timing comparison of the concrete and trait paths. It was
deferred; only allocation parity was checked."

The question matters because the interface is the runtime backend slot: a
consumer selects an implementation as `Box<dyn ContractionBackend<T>>`, prepares
a plan through it and then calls that plan on every operation. If the seam cost
anything per call, that would be a reason not to put it on a hot path.

## Arms

| arm | what one call pays |
| --- | --- |
| `prepare_concrete` | `Plan::<T>::new(&problem, &PlanConfig::default())` |
| `prepare_trait` | `Box<dyn ContractionBackend<T>>::prepare` with `Requirements::new()` and `PlanningBudget::serial()` |
| `exec_concrete` | `Plan::execute_into` on a prebuilt plan |
| `exec_trait` | `BoxedPlan<T>::execute_into` on the plan the trait prepared |

Both arms run the same kernels by construction: `TcontractBackend::prepare` is
`Plan::new` with the trait's `no_materialize` OR-ed in, and the trait's
`execute_into` forwards to the same execution. Any difference is dispatch and
boxing, not arithmetic. Views are descriptors over fixed buffers, built outside
the clock; the output is preallocated.

## Shapes

| shape | |
| --- | --- |
| `gemm_n64` | `D[i,k] = sum_j A[i,j] B[j,k]`, 64 × 64 × 64 |
| `mps_env_chi32` | the #61 corpus's MPS environment step at chi = 32, `ab,asc->bsc`: `D[b,s,c] = sum_a A[a,b] B[a,s,c]` |

## Run

```bash
bash experiments/neutral-dispatch/run.sh 3
```

Three sessions, each pinned to four CPUs of one idle L3 domain chosen by the
measured checkout's own `idle_cpus.py`, run through its `pinned.sh`; 1T is
`Exec::serial()` (no pool entered), 4T is a host-owned 4-worker Rayon pool
borrowed once outside the timed region. 500 ms of untimed priming per arm, best
of 5 calls, median of 3 sessions.

## Recorded run

`results/manifest.txt` carries the revision, host, compiler, guard, method and
the full table; `results/session{1,2,3}.{csv,txt,guard}` are the raw rows, the
8 CHECK lines per session and the guard logs.

**The finding: the interface is free per call, and costs one `Box` — about
150 ns — per prepared plan.**

| case | dtype | T | prepare trait/concrete | exec trait/concrete | prepare delta |
|---|---|---|---:|---:|---:|
| `gemm_n64` | f64 | 1 | 1.46 | 1.00 | +151 ns |
| `gemm_n64` | f64 | 4 | 1.48 | 1.00 | +160 ns |
| `gemm_n64` | c64 | 1 | 1.05 | 1.00 | +270 ns |
| `gemm_n64` | c64 | 4 | 1.03 | 0.99 | +191 ns |
| `mps_env_chi32` | f64 | 1 | 1.47 | 0.99 | +161 ns |
| `mps_env_chi32` | f64 | 4 | 1.41 | 1.00 | +139 ns |
| `mps_env_chi32` | c64 | 1 | 1.48 | 1.00 | +160 ns |
| `mps_env_chi32` | c64 | 4 | 1.41 | 1.00 | +141 ns |

`exec_trait / exec_concrete` is 0.99–1.00 in every row at both widths. The
preparation ratio looks worse on the cheap f64 rows (1.46) than on the c64 GEMM
(1.03) only because that concrete preparation is 5.5 µs, so the same absolute
overhead is a smaller share of it: the delta is the same +140 to +270 ns
everywhere.

So a consumer that keeps a prepared plan pays nothing measurable for runtime
backend selection, and pays it once at preparation. Allocation counts are not
re-measured here: `crates/tcontract/tests/plan_alloc.rs` pins the
*concrete* planning counts and `tests/neutral_alloc.rs` pins that the
**execution** boundary adds no allocation and no copy. The preparation-time
`Box` this probe measures in time is not allocation-counted by either, so that
attribution is an inference from the delta.

## What this does not cover

- One host, one rustc, one revision, three sessions. The `gemm_n64` c64 4T
  `exec` row is the noisiest at 17% spread, so its 3% difference is not a
  finding.
- Two shapes and one dtype pair (f64, c64); the point is the seam, not the
  kernel.
- No complex reference: c64 is timed, f64 is checked against the probe's naive
  reference. The two arms run the same plan code, and the f64 rows agree at
  machine precision.
