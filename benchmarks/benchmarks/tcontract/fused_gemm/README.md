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

```bash
cargo build --release -p tcontract-bench --bin fused_gemm
taskset -c 19    target/release/fused_gemm --threads 1
taskset -c 19-22 target/release/fused_gemm --threads 4
```

Environment: `BENCH_RUNS` (default 50), `BENCH_WARMUP` (default 5),
`BENCH_FILTER` (substring of the case name). Timed boundary: one execute of
every job. Plans are built outside the timed region; the per-job route builds
its `BatchItem` views inside it, as its callers do.

No result is recorded here yet: campaign runs belong to
[tcontract-benchmark](https://github.com/tensor4all/tcontract-benchmark).
