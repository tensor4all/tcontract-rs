# fused_gemm

The fused grouped-GEMM route (`tcontract::FusedGemmPlan`, tcontract-rs#208)
against the per-job route: `contract_batched` with one `Plan` per distinct
geometry, once with the packed driver forced (`per_job_packed`, the reference
the fused result must equal bitwise, checked per case as `CHECK ... ok`) and
once with the planner's own strategy (`per_job_default`). All jobs are `(N, N)`
f64, compact and back to back in shared buffers.

Cases (the issue's list): `16384x32^3`, `4096x8^3`,
`issue46_1x1024x256x1024+8192x32^3` (one 1024x1024x256 job plus 8192 of
32^3, cpueinsum issue46), and `uniform46_d{2,4,8,16}` (46 blocks of
degeneracy d).

Run it through the paired runner, which runs every thread count of a case back
to back, each pinned to idle cores of one L3 domain by `scripts/pinned.sh`
(Linux only; elsewhere record that the run was unpinned):

```bash
cargo build --release -p tcontract-bench --bin fused_gemm
benchmarks/scripts/paired.sh target/release/fused_gemm out/fused 8-15 1 4
```

Before a case is timed, every variant's result is checked on the measured
context (bitwise for `fused` and `per_job_packed` against the per-job packed
result, exactly by value for `per_job_default`), and a mismatch exits with
status 1, so a wrong route produces no rows. Environment: `BENCH_RUNS`
(default 50), `BENCH_WARMUP_MS` (default 1500 ms of untimed calls per arm,
the top of the PERFORMANCE_TIPS range for short calls), `BENCH_WARMUP`
(default 5 more calls), `BENCH_CASE` (exact name, as `paired.sh` passes it),
`BENCH_FILTER` (substring). Timed boundary: one execute of every job; inputs
and output pass through `black_box` on every call. Plans are built outside
the timed region; the per-job route builds its `BatchItem` views inside it,
as its callers do.

No result is recorded here yet: campaign runs belong to
[tcontract-benchmark](https://github.com/tensor4all/tcontract-benchmark).
