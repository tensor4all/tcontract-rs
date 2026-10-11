# exec_entry

Entry cost of the `tcontract-exec` primitives, and two kernels driven through
`Exec`: a transposing strided map (`tcontract_exec::strided::run_with_exec`) and a
512³ f64 GEMM on `Exec` (`Plan::run_with`, which broadcasts through
`Exec::broadcast` and uses the pool's own workspace). `# stats` lines give pool entry and broadcast counts per
case, so "serial work never enters the pool" is counted, not inferred.

```bash
cargo build --release -p tcontract-bench --bin exec_entry
taskset -c 19    target/release/exec_entry --threads 1
taskset -c 19-22 target/release/exec_entry --threads 4
```

Environment: `BENCH_RUNS` (default 200; large cases cap at 30 / 20),
`BENCH_WARMUP` (default 20), `BENCH_FILTER`. Timed boundary: one call of the
primitive or kernel; the GEMM plan is built outside the timed region, and the
GEMM result is checked bitwise against the serial run (`CHECK ... ok`).

## Observation, 2026-09-29 (single run, not a claim)

- tcontract-rs `2fee942` (clean tree), release profile (thin LTO,
  codegen-units 1), rustc 1.97.1.
- AMD EPYC 7713P (64 cores, 1 thread/core, 8 L3 domains of 8 cores), shared
  host with load average about 15; cores 19-22 (one L3 domain) measured idle
  (busy fraction ≤ 0.01) over 3 s before and after the runs.
- Raw output: [results/2026-09-29-1t.csv](results/2026-09-29-1t.csv),
  [results/2026-09-29-4t.csv](results/2026-09-29-4t.csv).

| Case | Variant | 1T median | 4T median | 4T entries / broadcasts |
| --- | --- | ---: | ---: | --- |
| `install_empty` | k1 | 40 ns | 30 ns | 0 / 0 |
| `install_empty` | k2 | — | 6.89 µs | 1 per call |
| `install_empty` | k4 | — | 6.75 µs | 1 per call |
| `partition_empty` | k2 | — | 6.85 µs | 1 per call |
| `partition_empty` | k4 | — | 6.82 µs | 1 per call |
| `broadcast_empty` | w4 | — | 7.62 µs | 1 broadcast per call |
| `strided_map_f64` | transpose 2^12 (below threshold) | 3.18 µs | 3.15 µs | 0 / 0 |
| `strided_map_f64` | transpose 2^22 | 14.22 ms | 4.07 ms (3.5x) | 1 per call |
| `packed_gemm_f64` (recorded under an earlier name in the CSVs) | 512³ | 5.61 ms | 1.98 ms (2.8x) | 1 broadcast per call |

Width-one requests cost tens of nanoseconds and never touch the pool; a small
map below strided's `MINTHREADLENGTH` stays on the caller at 4T. Entering the
pool costs about 7 µs at width 2 to 4 on this machine, in line with the
`experiments/rayon-entry` measurement on the M5 Max.
