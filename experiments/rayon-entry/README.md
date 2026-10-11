# Rayon pool entry latency

**Question:** Where does the cost of entering a Rayon pool from a non-worker thread come from: waking sleeping workers, waking the blocked caller, or fanning out to many workers? How does it compare with an OpenMP parallel region?

This is a preliminary measurement for [Prototype 1](../../docs/experiments.md#prototype-1-execution-and-ffi-boundary). It times empty closures only, so there is no numerical correctness gate. Each configuration was run one to three times in one session; no paired statistics or noise estimate yet.

## Setup

| Item | Value |
| --- | --- |
| Date | 2026-09-29 |
| CPU | Apple M5 Max, 6 performance and 12 efficiency cores (`hw.ncpu` = 18), no affinity control |
| OS | macOS 26.5.1 (Darwin 25.5.0) |
| Rust | rustc 1.96.0, release profile, `opt-level = 3` |
| Rayon | rayon 1.11.0, rayon-core 1.13.0 (`Cargo.lock`) |
| OpenMP | Homebrew libomp 22.1.5, Apple clang 21.0.0, `-O2` |
| Timed boundary | From just before the entry call to its return on the calling thread. Pool construction and 1000 warm-up entries are outside. |
| Idle gap | The caller busy-waits for the gap (it never sleeps), so only the pool's workers can go to sleep. |
| Timer | `Instant` (Rust); `CLOCK_MONOTONIC_RAW` (C). The first OpenMP run used `CLOCK_MONOTONIC`, which rounds to 1 µs on macOS. |

Run from this directory:

```sh
cargo run --release --locked --bin rayon-entry   # src/main.rs: install, par_iter, broadcast vs gap (default-run)
cargo run --release --locked --bin split         # src/bin/split.rs: caller wake vs worker wake
cargo run --release --locked --bin width         # src/bin/width.rs: active width on a fixed 18-worker pool
clang -O2 -Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include \
  -L$(brew --prefix libomp)/lib -lomp omp.c -o omp && ./omp 4
```

Raw outputs of the reruns are in [`results/`](results/).

## Results (median µs; p10 to p90 in the raw output)

| Operation | Threads | Gap 0 | Gap 100 µs or more |
| --- | --- | --- | --- |
| `install(empty)` | 1 to 4 | 3.6 to 6 | 3.5 to 12.7 |
| `spawn`, caller spins on an atomic | 1 to 4 | 0.4 to 0.7 | 6 to 8 |
| `install` then `par_iter` over 4n items | 4 | 6.5 | 9 to 16 |
| `broadcast(empty)` | 4 | 8 | 12 to 17 |
| `install(empty)` | 18 | 23 | 8 to 12.5 |
| `install` then `par_iter` | 18 | 70 | 140 to 220 |
| `broadcast(empty)` | 18 | 55 | 120 to 200 |
| nested `install` inside the same pool | 1 to 18 | 0.008 to 0.033 | same |
| raw two-thread round trip, both spin | 2 | 0.12 | |
| raw two-thread round trip, `Mutex` + `Condvar` | 2 | 3.0 | |
| OpenMP `parallel` region, default settings | 4 | 12 | 10 to 11.5 |
| OpenMP, `OMP_WAIT_POLICY=active KMP_BLOCKTIME=infinite` | 4 | 0.42 | 0.46 to 1.1 (p90 4.3 at 10 ms) |
| OpenMP, default settings | 18 | 53 | 57 to 94 |
| OpenMP, active wait | 18 | 1.8 | 2.6 to 59 (p90 166 µs to 1.1 ms) |

OpenMP rows are from the rerun of 2026-09-29 12:16 UTC ([raw](results/omp-2026-09-29.txt)) after removing a data race ([#4](https://github.com/tensor4all/tcontract-rs/issues/4)): the first version wrote every worker's thread number to one shared `volatile int`, which is a race and adds cache-line contention. Each worker now passes its own thread number through an empty `asm volatile` barrier. ThreadSanitizer (`-O1 -fsanitize=thread`, 4 and 18 threads, warm-up and timed regions) reports no race; the `-O2` disassembly of `main.omp_outlined` still calls `omp_get_thread_num`, so the region is not optimized away. The default-policy numbers did not change materially; the active-wait numbers at 18 threads became worse than first reported.

## Active width on a fixed pool

`width.rs` keeps one 18-worker pool and varies the active width `k` ([raw](results/width-2026-09-29.txt)). `broadcast` with workers at index `k` or above returning immediately, as in the adapter proposed in [tensorprimitives-rs #1](https://github.com/lkdvos/tensorprimitives-rs/issues/1), against `install` plus `scope` spawning `k` empty tasks. Median µs:

| k | broadcast, gap 0 | broadcast, gap 1 ms | install + scope, gap 0 | install + scope, gap 1 ms |
| --- | --- | --- | --- | --- |
| 1 | 72 | 165 | 21 | 17 |
| 2 | 65 | 164 | 23 | 19 |
| 4 | 61 | 164 | 22 | 50 |
| 8 | 58 | 167 | 23 | 141 |
| 18 | 49 | 166 | 24 | 144 |

`broadcast` returns 18 results for every `k`: all workers are dispatched and awaited, so its cost is set by the pool size, not by the active width. `install` plus `k` tasks scales with `k` from idle, but its tasks are not guaranteed to run concurrently and cannot synchronize with barriers. This is the evidence for the width rules in the [architecture](../../docs/architecture.md#execution-context).

## Reading

- **Worker wake-up is real.** Idle Rayon workers go to sleep within 100 µs (32 `yield_now` rounds, `rayon-core/src/sleep/mod.rs`). Handing a job to a sleeping worker and seeing it run costs 6 to 8 µs.
- **Caller wake-up is the other half.** Even with awake workers, `install` from outside costs 3 to 6 µs because the caller blocks on a lock latch and must be woken; this matches the raw `Condvar` round trip. A spinning caller with awake workers sees 0.4 to 0.7 µs.
- **Full-width fan-out dominates.** Waking all 18 workers costs 55 to 200 µs. The heterogeneous efficiency cores are a likely contributor; a homogeneous CPU should be measured before generalizing.
- **Already inside the pool, entry is nearly free** (8 to 33 ns).
- **OpenMP is not intrinsically cheaper.** With default libomp settings it is in the same range as Rayon. Its sub-microsecond figures come from an active wait policy, which Rayon does not offer. Active waiting on all 18 cores degraded to a 59 µs median and 1.1 ms p90 after a 10 ms gap.

## Decision taken from this result

Recorded in the [decision log](../../docs/decision-log.md): an entry cost of about 10 µs per parallel kernel is acceptable, because serial operations never pay it and parallel width is chosen from the amount of work. Spin-wait policies are deferred until a measurement shows a need.

## Not yet measured

Break-even of real small GEMM and contraction kernels against thread count; a homogeneous x86 CPU (the [tenferro #1945](https://github.com/tensor4all/tenferro-rs/issues/1945) numbers are from AMD EPYC); barrier cost inside one SPMD call on a borrowed pool; a subset-broadcast primitive.
