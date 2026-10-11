# faer-limit-sweep

Per-call cost of the two engines the `tcontract` planner can choose for a
copy-free fusable contraction, on the same `Problem`, at one and four threads:

```text
D = alpha * A * B          alpha = 1
```

Two C modes are recorded for every case:

| `c_mode` | problem | timed call |
| --- | --- | --- |
| `absent` | `CSpec::Absent`, a zero `D` | `Plan::execute_into` — `D = alpha * A * B`, overwriting |
| `output` | `CSpec::Output(Op::Identity)`, a non-zero `D` | `Plan::execute_into_accum` with `beta = 1` and `AccumulationSource::Output` — `D += alpha * A * B` in place |

`output` is the form an MPS step (`D += A*B`) pays; `absent` is the overwrite
form the earlier recorded run used.

| arm | what one call pays |
| --- | --- |
| `default` | `PlanConfig::default()` — the planner's own route: faer below the dtype's `FaerLimit`, the packed driver above it, elementwise for all-batch |
| `packed` | `PlanConfig::packed()` — the packed TBLIS-style driver, forced |
| `faer_forced` | `PlanConfig { faer_limit: FaerLimit::NONE, ..PlanConfig::default() }` — faer for every fusable problem, the rule before #63 |

The question is [tcontract-rs#63](https://github.com/tensor4all/tcontract-rs/issues/63)
(the size/dtype threshold on the copy-free fusion rule) and its follow-up
[#69](https://github.com/tensor4all/tcontract-rs/issues/69): where the crossover
between the two engines lies, per dtype and per thread count, on the classes
that fuse to one strided batched GEMM. This probe only records the arms;
the `FaerLimit` values and the decision-log row are decided elsewhere.

## Timed boundary

Only the plan execution is timed, on a prebuilt `Plan` and a preallocated
output: `Plan::execute_into` in the `absent` mode and
`Plan::execute_into_accum(.., AccumulationSource::Output)` in the `output` mode.
Building the `Problem`, planning it, building the views and allocating
`A`/`B`/`D` all happen outside the clock, and one arm is measured at a time.

| threads | executor |
| --- | --- |
| 1 | `Exec::serial()`, process pinned to one CPU |
| 4 | a host-owned `rayon::ThreadPool` of 4 workers built once, borrowed through `tcontract_exec::Pool::borrow`, handed to `Exec::rayon` |

The binary asserts the pool width, the `Exec` budget and that
`Cpus_allowed_list` holds exactly the requested CPUs, and refuses
`RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` /
`MKL_NUM_THREADS` / `VECLIB_MAXIMUM_THREADS` that contradict `--threads`.

Before any timing, every case is checked against a naive label oracle and the
arms against each other (`rel < 1e-10` for `f64`/`c64`, `< 1e-5` for
`f32`/`c32`). A mismatch aborts the run instead of publishing a number. The
`CHECK` lines are in `results/triarm-session*.txt` (three arms, both C modes),
in `results/cmode-session*.txt` (two arms, both C modes) and in the earlier
`results/session*.txt` (`absent` only).

## Shapes

Column-major contiguous; the labels are the ones
`experiments/three-engine-contract/src/main.rs` already uses. All four dtypes
`DType::{F32, F64, C32, C64}` are measured for every shape.

| class | equation | extents |
| --- | --- | --- |
| `gemm` | `D[i,k] = sum_j A[i,j] B[j,k]` | `n` in {32, 64, 128, 256} |
| `gemm_batched` | `D[i,k,b] = sum_j A[i,j,b] B[j,k,b]` | `(n, batch)` in {(16,16), (32,16), (64,16), (32,64), (64,64)} |
| `mps_env` | `ab,asc->bsc` | `A = chi x chi`, `B = chi x 2 x chi`, `D = chi x 2 x chi` |
| `mps_site` | `bsc,bsd->cd` | `A = B = chi x 2 x chi`, `D = chi x chi` |

`chi` is {16, 24, 32, 48, 64, 96, 128} for the two MPS classes. `mnk` in the
table below is the fused `m * n * k` per batch item — the quantity the planner's
`FaerLimit` compares — read from the forced-packed plan's `PlanStats` for both
rows. The MPS steps both have `m * n * k = 2 chi^3`.

`mps_env`'s output extents follow the label order of the equation (`bsc`), so
`D` is `chi x 2 x chi`; the "2 x chi x chi" written in the task text is not the
order the labels give. `results/manifest.txt` records this.

## Run

```bash
bash experiments/faer-limit-sweep/run.sh 3            # the three-arm set
bash experiments/faer-limit-sweep/run.sh 3 cmode      # the earlier two-arm set
```

`run.sh [sessions] [set]` is the one entry point; `set` is `triarm`
(`default`, `packed`, `faer_forced`; the default) or `cmode` (`default`,
`packed` only, the set the earlier `cmode-session*` files were recorded with).
It builds once, then runs `sessions` sessions; each session is two guarded,
pinned processes — a 1T run on `idle_cpus.py pick 1` and a 4T run on
`idle_cpus.py pick 4` — through `benchmarks/scripts/pinned.sh` of this
checkout. A sub-run whose guard log does not say "idle before and after" is
discarded and retried. Per arm: 500 ms of untimed wall-clock priming, then the
best of 5 calls. The one binary records both C modes for every case and the
arms of the requested set; each sub-run writes
`results/<set>-session{s}.{csv,txt,guard}`, and `run.sh` prints the summary
table below.

## Recorded runs

`results/manifest.txt` carries the revision, host, CPU sets, guard and method.

There are three recorded runs. The first measured only the `absent` mode:
`results/session{1,2,3}.{csv,txt,guard}` are its raw per-session rows (368
each), its `CHECK` lines and its guard logs. The second measures both C modes
with two arms: `results/cmode-session{1,2,3}.{csv,txt,guard}` are its raw
per-session rows (736 each: 23 shapes x 4 dtypes x 2 arms x 2 thread counts x 2
C modes), its `CHECK` lines (368 per session) and its guard logs. The third
measures both C modes with three arms:
`results/triarm-session{1,2,3}.{csv,txt,guard}` are its raw per-session rows
(1104 each: 23 shapes x 4 dtypes x 3 arms x 2 thread counts x 2 C modes), its
`CHECK` lines (368 per session) and its guard logs. All six guarded sub-runs of
each run passed at attempt 1.

The `absent` rows of the second run re-measure the same configuration as the
first run (at the newer library revision named in `results/manifest.txt`), so
the first run's `absent` table and the second run's `absent` table are
comparable; the `output` rows are the dimension the second run added, and the
`faer_forced` arm is the one the third run added.

### Earlier run (overwrite `absent` only)

`default/packed < 1` means the planner's own route was faster; `> 1` means the
forced-packed arm was faster. `spread` is the larger of the two arms'
`(max-min)/median` over the three sessions.

| threads | class | dtype | params | mnk | default | packed | ns default | ns packed | default/packed | spread |
|---|---|---|---|---:|---|---|---:|---:|---:|---:|
| 1 | `gemm` | f32 | n=32 | 32768 | faer | packed | 742 | 2554 | 0.291 | 8% |
| 1 | `gemm` | f64 | n=32 | 32768 | faer | packed | 1402 | 6412 | 0.219 | 2% |
| 1 | `gemm` | c32 | n=32 | 32768 | faer | packed | 2705 | 7454 | 0.363 | 2% |
| 1 | `gemm` | c64 | n=32 | 32768 | faer | packed | 5299 | 7794 | 0.680 | 1% |
| 1 | `gemm` | f32 | n=64 | 262144 | faer | packed | 5179 | 13475 | 0.384 | 1% |
| 1 | `gemm` | f64 | n=64 | 262144 | faer | packed | 10309 | 22772 | 0.453 | 0% |
| 1 | `gemm` | c32 | n=64 | 262144 | faer | packed | 20508 | 33783 | 0.607 | 1% |
| 1 | `gemm` | c64 | n=64 | 262144 | packed | packed | 61345 | 61225 | 1.002 | 1% |
| 1 | `gemm` | f32 | n=128 | 2097152 | faer | packed | 40375 | 69389 | 0.582 | 0% |
| 1 | `gemm` | f64 | n=128 | 2097152 | faer | packed | 80460 | 128469 | 0.626 | 1% |
| 1 | `gemm` | c32 | n=128 | 2097152 | faer | packed | 160920 | 205905 | 0.782 | 0% |
| 1 | `gemm` | c64 | n=128 | 2097152 | packed | packed | 427688 | 426246 | 1.003 | 1% |
| 1 | `gemm` | f32 | n=256 | 16777216 | faer | packed | 320969 | 397441 | 0.808 | 1% |
| 1 | `gemm` | f64 | n=256 | 16777216 | faer | packed | 644252 | 863060 | 0.746 | 1% |
| 1 | `gemm` | c32 | n=256 | 16777216 | faer | packed | 1284768 | 1353035 | 0.950 | 0% |
| 1 | `gemm` | c64 | n=256 | 16777216 | packed | packed | 2694577 | 2696541 | 0.999 | 1% |
| 1 | `gemm_batched` | f32 | n=16 batch=16 | 4096 | faer | packed | 2044 | 11732 | 0.174 | 9% |
| 1 | `gemm_batched` | f64 | n=16 batch=16 | 4096 | faer | packed | 3616 | 9969 | 0.363 | 5% |
| 1 | `gemm_batched` | c32 | n=16 batch=16 | 4096 | faer | packed | 9658 | 19787 | 0.488 | 1% |
| 1 | `gemm_batched` | c64 | n=16 batch=16 | 4096 | faer | packed | 19316 | 19457 | 0.993 | 1% |
| 1 | `gemm_batched` | f32 | n=32 batch=16 | 32768 | faer | packed | 11401 | 34965 | 0.326 | 12% |
| 1 | `gemm_batched` | f64 | n=32 batch=16 | 32768 | faer | packed | 21901 | 46276 | 0.473 | 1% |
| 1 | `gemm_batched` | c32 | n=32 batch=16 | 32768 | faer | packed | 42639 | 62156 | 0.686 | 1% |
| 1 | `gemm_batched` | c64 | n=32 batch=16 | 32768 | faer | packed | 84618 | 119092 | 0.711 | 1% |
| 1 | `gemm_batched` | f32 | n=64 batch=16 | 262144 | faer | packed | 82453 | 172602 | 0.478 | 2% |
| 1 | `gemm_batched` | f64 | n=64 batch=16 | 262144 | faer | packed | 166661 | 236832 | 0.704 | 0% |
| 1 | `gemm_batched` | c32 | n=64 batch=16 | 262144 | faer | packed | 329054 | 398994 | 0.825 | 0% |
| 1 | `gemm_batched` | c64 | n=64 batch=16 | 262144 | packed | packed | 775457 | 776239 | 0.999 | 1% |
| 1 | `gemm_batched` | f32 | n=32 batch=64 | 32768 | faer | packed | 45665 | 121576 | 0.376 | 3% |
| 1 | `gemm_batched` | f64 | n=32 batch=64 | 32768 | faer | packed | 88986 | 184013 | 0.484 | 1% |
| 1 | `gemm_batched` | c32 | n=32 batch=64 | 32768 | faer | packed | 171691 | 251990 | 0.681 | 1% |
| 1 | `gemm_batched` | c64 | n=32 batch=64 | 32768 | faer | packed | 343260 | 484864 | 0.708 | 1% |
| 1 | `gemm_batched` | f32 | n=64 batch=64 | 262144 | faer | packed | 332180 | 693653 | 0.479 | 0% |
| 1 | `gemm_batched` | f64 | n=64 batch=64 | 262144 | faer | packed | 669970 | 985188 | 0.680 | 1% |
| 1 | `gemm_batched` | c32 | n=64 batch=64 | 262144 | faer | packed | 1328108 | 1622797 | 0.818 | 1% |
| 1 | `gemm_batched` | c64 | n=64 batch=64 | 262144 | packed | packed | 3719421 | 3704311 | 1.004 | 1% |
| 1 | `mps_env` | f32 | chi=16 | 8192 | faer | packed | 320 | 1452 | 0.220 | 3% |
| 1 | `mps_env` | f64 | chi=16 | 8192 | faer | packed | 461 | 1503 | 0.307 | 3% |
| 1 | `mps_env` | c32 | chi=16 | 8192 | faer | packed | 801 | 1884 | 0.425 | 2% |
| 1 | `mps_env` | c64 | chi=16 | 8192 | faer | packed | 1492 | 2735 | 0.546 | 1% |
| 1 | `mps_env` | f32 | chi=24 | 27648 | faer | packed | 881 | 1773 | 0.497 | 1% |
| 1 | `mps_env` | f64 | chi=24 | 27648 | faer | packed | 1242 | 2324 | 0.534 | 1% |
| 1 | `mps_env` | c32 | chi=24 | 27648 | faer | packed | 2355 | 5110 | 0.461 | 1% |
| 1 | `mps_env` | c64 | chi=24 | 27648 | faer | packed | 4648 | 8085 | 0.575 | 1% |
| 1 | `mps_env` | f32 | chi=32 | 65536 | faer | packed | 1432 | 4378 | 0.327 | 2% |
| 1 | `mps_env` | f64 | chi=32 | 65536 | faer | packed | 2795 | 5901 | 0.474 | 2% |
| 1 | `mps_env` | c32 | chi=32 | 65536 | faer | packed | 5390 | 7855 | 0.686 | 1% |
| 1 | `mps_env` | c64 | chi=32 | 65536 | faer | packed | 10630 | 14117 | 0.753 | 1% |
| 1 | `mps_env` | f32 | chi=48 | 221184 | faer | packed | 4528 | 7153 | 0.633 | 1% |
| 1 | `mps_env` | f64 | chi=48 | 221184 | faer | packed | 8867 | 11923 | 0.744 | 0% |
| 1 | `mps_env` | c32 | chi=48 | 221184 | faer | packed | 17512 | 28223 | 0.620 | 0% |
| 1 | `mps_env` | c64 | chi=48 | 221184 | packed | packed | 41607 | 41587 | 1.000 | 0% |
| 1 | `mps_env` | f32 | chi=64 | 524288 | faer | packed | 10360 | 20949 | 0.495 | 1% |
| 1 | `mps_env` | f64 | chi=64 | 524288 | faer | packed | 20869 | 28463 | 0.733 | 1% |
| 1 | `mps_env` | c32 | chi=64 | 524288 | faer | packed | 41237 | 50384 | 0.818 | 1% |
| 1 | `mps_env` | c64 | chi=64 | 524288 | packed | packed | 96239 | 96209 | 1.000 | 0% |
| 1 | `mps_env` | f32 | chi=96 | 1769472 | faer | packed | 34594 | 43852 | 0.789 | 0% |
| 1 | `mps_env` | f64 | chi=96 | 1769472 | faer | packed | 68899 | 79428 | 0.867 | 0% |
| 1 | `mps_env` | c32 | chi=96 | 1769472 | faer | packed | 137106 | 153957 | 0.891 | 0% |
| 1 | `mps_env` | c64 | chi=96 | 1769472 | packed | packed | 301462 | 300501 | 1.003 | 0% |
| 1 | `mps_env` | f32 | chi=128 | 4194304 | faer | packed | 81512 | 108282 | 0.753 | 1% |
| 1 | `mps_env` | f64 | chi=128 | 4194304 | faer | packed | 162704 | 203240 | 0.801 | 0% |
| 1 | `mps_env` | c32 | chi=128 | 4194304 | faer | packed | 323924 | 357547 | 0.906 | 0% |
| 1 | `mps_env` | c64 | chi=128 | 4194304 | packed | packed | 704134 | 702912 | 1.002 | 2% |
| 1 | `mps_site` | f32 | chi=16 | 8192 | faer | packed | 311 | 1723 | 0.180 | 6% |
| 1 | `mps_site` | f64 | chi=16 | 8192 | faer | packed | 471 | 1443 | 0.326 | 2% |
| 1 | `mps_site` | c32 | chi=16 | 8192 | faer | packed | 801 | 2675 | 0.299 | 1% |
| 1 | `mps_site` | c64 | chi=16 | 8192 | faer | packed | 1473 | 2695 | 0.547 | 1% |
| 1 | `mps_site` | f32 | chi=24 | 27648 | faer | packed | 882 | 2976 | 0.296 | 6% |
| 1 | `mps_site` | f64 | chi=24 | 27648 | faer | packed | 1292 | 2374 | 0.544 | 2% |
| 1 | `mps_site` | c32 | chi=24 | 27648 | faer | packed | 2364 | 4879 | 0.485 | 0% |
| 1 | `mps_site` | c64 | chi=24 | 27648 | faer | packed | 4659 | 8085 | 0.576 | 0% |
| 1 | `mps_site` | f32 | chi=32 | 65536 | faer | packed | 1453 | 4628 | 0.314 | 9% |
| 1 | `mps_site` | f64 | chi=32 | 65536 | faer | packed | 2845 | 6222 | 0.457 | 1% |
| 1 | `mps_site` | c32 | chi=32 | 65536 | faer | packed | 5380 | 8676 | 0.620 | 0% |
| 1 | `mps_site` | c64 | chi=32 | 65536 | faer | packed | 10790 | 15078 | 0.716 | 0% |
| 1 | `mps_site` | f32 | chi=48 | 221184 | faer | packed | 4608 | 7404 | 0.622 | 1% |
| 1 | `mps_site` | f64 | chi=48 | 221184 | faer | packed | 9217 | 12203 | 0.755 | 0% |
| 1 | `mps_site` | c32 | chi=48 | 221184 | faer | packed | 17683 | 29024 | 0.609 | 0% |
| 1 | `mps_site` | c64 | chi=48 | 221184 | packed | packed | 41257 | 41207 | 1.001 | 0% |
| 1 | `mps_site` | f32 | chi=64 | 524288 | faer | packed | 10669 | 22362 | 0.477 | 0% |
| 1 | `mps_site` | f64 | chi=64 | 524288 | faer | packed | 21430 | 29575 | 0.725 | 2% |
| 1 | `mps_site` | c32 | chi=64 | 524288 | faer | packed | 41657 | 51556 | 0.808 | 0% |
| 1 | `mps_site` | c64 | chi=64 | 524288 | packed | packed | 95558 | 95508 | 1.001 | 0% |
| 1 | `mps_site` | f32 | chi=96 | 1769472 | faer | packed | 35065 | 44844 | 0.782 | 0% |
| 1 | `mps_site` | f64 | chi=96 | 1769472 | faer | packed | 69700 | 80520 | 0.866 | 0% |
| 1 | `mps_site` | c32 | chi=96 | 1769472 | faer | packed | 137316 | 156752 | 0.876 | 0% |
| 1 | `mps_site` | c64 | chi=96 | 1769472 | packed | packed | 302925 | 302194 | 1.002 | 1% |
| 1 | `mps_site` | f32 | chi=128 | 4194304 | faer | packed | 82805 | 111989 | 0.739 | 1% |
| 1 | `mps_site` | f64 | chi=128 | 4194304 | faer | packed | 167392 | 207057 | 0.808 | 2% |
| 1 | `mps_site` | c32 | chi=128 | 4194304 | faer | packed | 326739 | 379267 | 0.862 | 1% |
| 1 | `mps_site` | c64 | chi=128 | 4194304 | packed | packed | 720635 | 718541 | 1.003 | 2% |
| 4 | `gemm` | f32 | n=32 | 32768 | faer | packed | 751 | 2244 | 0.335 | 14% |
| 4 | `gemm` | f64 | n=32 | 32768 | faer | packed | 1412 | 3266 | 0.432 | 1% |
| 4 | `gemm` | c32 | n=32 | 32768 | faer | packed | 2705 | 4278 | 0.632 | 1% |
| 4 | `gemm` | c64 | n=32 | 32768 | faer | packed | 5300 | 7794 | 0.680 | 1% |
| 4 | `gemm` | f32 | n=64 | 262144 | faer | packed | 5180 | 11161 | 0.464 | 0% |
| 4 | `gemm` | f64 | n=64 | 262144 | faer | packed | 10289 | 14888 | 0.691 | 1% |
| 4 | `gemm` | c32 | n=64 | 262144 | faer | packed | 14838 | 12453 | 1.192 | 13% |
| 4 | `gemm` | c64 | n=64 | 262144 | packed | packed | 21149 | 20809 | 1.016 | 12% |
| 4 | `gemm` | f32 | n=128 | 2097152 | faer | packed | 38672 | 20288 | 1.906 | 59% |
| 4 | `gemm` | f64 | n=128 | 2097152 | faer | packed | 32100 | 42459 | 0.756 | 8% |
| 4 | `gemm` | c32 | n=128 | 2097152 | faer | packed | 57256 | 54432 | 1.052 | 3% |
| 4 | `gemm` | c64 | n=128 | 2097152 | packed | packed | 98614 | 98935 | 0.997 | 2% |
| 4 | `gemm` | f32 | n=256 | 16777216 | faer | packed | 98434 | 142456 | 0.691 | 56% |
| 4 | `gemm` | f64 | n=256 | 16777216 | faer | packed | 185015 | 204431 | 0.905 | 4% |
| 4 | `gemm` | c32 | n=256 | 16777216 | faer | packed | 353960 | 348560 | 1.015 | 0% |
| 4 | `gemm` | c64 | n=256 | 16777216 | packed | packed | 689095 | 688806 | 1.000 | 1% |
| 4 | `gemm_batched` | f32 | n=16 batch=16 | 4096 | faer | packed | 2024 | 11722 | 0.173 | 8% |
| 4 | `gemm_batched` | f64 | n=16 batch=16 | 4096 | faer | packed | 3486 | 9968 | 0.350 | 8% |
| 4 | `gemm_batched` | c32 | n=16 batch=16 | 4096 | faer | packed | 9568 | 19697 | 0.486 | 0% |
| 4 | `gemm_batched` | c64 | n=16 batch=16 | 4096 | faer | packed | 19326 | 19065 | 1.014 | 1% |
| 4 | `gemm_batched` | f32 | n=32 batch=16 | 32768 | faer | packed | 6883 | 15318 | 0.449 | 11% |
| 4 | `gemm_batched` | f64 | n=32 batch=16 | 32768 | faer | packed | 11462 | 20688 | 0.554 | 1% |
| 4 | `gemm_batched` | c32 | n=32 batch=16 | 32768 | faer | packed | 14236 | 19657 | 0.724 | 6% |
| 4 | `gemm_batched` | c64 | n=32 batch=16 | 32768 | faer | packed | 24606 | 33472 | 0.735 | 1% |
| 4 | `gemm_batched` | f32 | n=64 batch=16 | 262144 | faer | packed | 24135 | 47228 | 0.511 | 2% |
| 4 | `gemm_batched` | f64 | n=64 batch=16 | 262144 | faer | packed | 44823 | 62657 | 0.715 | 2% |
| 4 | `gemm_batched` | c32 | n=64 batch=16 | 262144 | faer | packed | 240599 | 102190 | 2.354 | 7% |
| 4 | `gemm_batched` | c64 | n=64 batch=16 | 262144 | packed | packed | 196927 | 196496 | 1.002 | 1% |
| 4 | `gemm_batched` | f32 | n=32 batch=64 | 32768 | faer | packed | 14948 | 39303 | 0.380 | 1% |
| 4 | `gemm_batched` | f64 | n=32 batch=64 | 32768 | faer | packed | 24997 | 49853 | 0.501 | 4% |
| 4 | `gemm_batched` | c32 | n=32 batch=64 | 32768 | faer | packed | 46396 | 65853 | 0.705 | 1% |
| 4 | `gemm_batched` | c64 | n=32 batch=64 | 32768 | faer | packed | 88195 | 122498 | 0.720 | 1% |
| 4 | `gemm_batched` | f32 | n=64 batch=64 | 262144 | faer | packed | 86341 | 177131 | 0.487 | 1% |
| 4 | `gemm_batched` | f64 | n=64 batch=64 | 262144 | faer | packed | 171199 | 243034 | 0.704 | 1% |
| 4 | `gemm_batched` | c32 | n=64 batch=64 | 262144 | faer | packed | 1018981 | 401940 | 2.535 | 1% |
| 4 | `gemm_batched` | c64 | n=64 batch=64 | 262144 | packed | packed | 936949 | 932430 | 1.005 | 1% |
| 4 | `mps_env` | f32 | chi=16 | 8192 | faer | packed | 320 | 1433 | 0.223 | 3% |
| 4 | `mps_env` | f64 | chi=16 | 8192 | faer | packed | 471 | 1432 | 0.329 | 2% |
| 4 | `mps_env` | c32 | chi=16 | 8192 | faer | packed | 791 | 1854 | 0.427 | 2% |
| 4 | `mps_env` | c64 | chi=16 | 8192 | faer | packed | 1483 | 2695 | 0.550 | 1% |
| 4 | `mps_env` | f32 | chi=24 | 27648 | faer | packed | 871 | 1743 | 0.500 | 1% |
| 4 | `mps_env` | f64 | chi=24 | 27648 | faer | packed | 1242 | 2234 | 0.556 | 2% |
| 4 | `mps_env` | c32 | chi=24 | 27648 | faer | packed | 2364 | 5050 | 0.468 | 1% |
| 4 | `mps_env` | c64 | chi=24 | 27648 | faer | packed | 4628 | 8075 | 0.573 | 1% |
| 4 | `mps_env` | f32 | chi=32 | 65536 | faer | packed | 1433 | 4308 | 0.333 | 1% |
| 4 | `mps_env` | f64 | chi=32 | 65536 | faer | packed | 2775 | 5830 | 0.476 | 1% |
| 4 | `mps_env` | c32 | chi=32 | 65536 | faer | packed | 5370 | 7795 | 0.689 | 1% |
| 4 | `mps_env` | c64 | chi=32 | 65536 | faer | packed | 10630 | 14056 | 0.756 | 0% |
| 4 | `mps_env` | f32 | chi=48 | 221184 | faer | packed | 4499 | 7094 | 0.634 | 1% |
| 4 | `mps_env` | f64 | chi=48 | 221184 | faer | packed | 8886 | 11872 | 0.748 | 0% |
| 4 | `mps_env` | c32 | chi=48 | 221184 | faer | packed | 13997 | 15338 | 0.913 | 25% |
| 4 | `mps_env` | c64 | chi=48 | 221184 | packed | packed | 15569 | 15549 | 1.001 | 5% |
| 4 | `mps_env` | f32 | chi=64 | 524288 | faer | packed | 10590 | 13195 | 0.803 | 41% |
| 4 | `mps_env` | f64 | chi=64 | 524288 | faer | packed | 15950 | 17883 | 0.892 | 26% |
| 4 | `mps_env` | c32 | chi=64 | 524288 | faer | packed | 18103 | 19847 | 0.912 | 29% |
| 4 | `mps_env` | c64 | chi=64 | 524288 | packed | packed | 33252 | 33302 | 0.998 | 10% |
| 4 | `mps_env` | f32 | chi=96 | 1769472 | faer | packed | 27621 | 18966 | 1.456 | 14% |
| 4 | `mps_env` | f64 | chi=96 | 1769472 | faer | packed | 38020 | 30497 | 1.247 | 15% |
| 4 | `mps_env` | c32 | chi=96 | 1769472 | faer | packed | 63097 | 46426 | 1.359 | 47% |
| 4 | `mps_env` | c64 | chi=96 | 1769472 | packed | packed | 110596 | 110416 | 1.002 | 0% |
| 4 | `mps_env` | f32 | chi=128 | 4194304 | faer | packed | 33372 | 35386 | 0.943 | 36% |
| 4 | `mps_env` | f64 | chi=128 | 4194304 | faer | packed | 54983 | 75881 | 0.725 | 14% |
| 4 | `mps_env` | c32 | chi=128 | 4194304 | faer | packed | 99025 | 97542 | 1.015 | 17% |
| 4 | `mps_env` | c64 | chi=128 | 4194304 | packed | packed | 185827 | 186268 | 0.998 | 1% |
| 4 | `mps_site` | f32 | chi=16 | 8192 | faer | packed | 311 | 1644 | 0.189 | 4% |
| 4 | `mps_site` | f64 | chi=16 | 8192 | faer | packed | 481 | 1343 | 0.358 | 6% |
| 4 | `mps_site` | c32 | chi=16 | 8192 | faer | packed | 801 | 2645 | 0.303 | 2% |
| 4 | `mps_site` | c64 | chi=16 | 8192 | faer | packed | 1472 | 2615 | 0.563 | 2% |
| 4 | `mps_site` | f32 | chi=24 | 27648 | faer | packed | 891 | 2925 | 0.305 | 10% |
| 4 | `mps_site` | f64 | chi=24 | 27648 | faer | packed | 1262 | 2284 | 0.553 | 1% |
| 4 | `mps_site` | c32 | chi=24 | 27648 | faer | packed | 2344 | 4859 | 0.482 | 1% |
| 4 | `mps_site` | c64 | chi=24 | 27648 | faer | packed | 4608 | 8015 | 0.575 | 1% |
| 4 | `mps_site` | f32 | chi=32 | 65536 | faer | packed | 1452 | 4548 | 0.319 | 1% |
| 4 | `mps_site` | f64 | chi=32 | 65536 | faer | packed | 2805 | 6152 | 0.456 | 1% |
| 4 | `mps_site` | c32 | chi=32 | 65536 | faer | packed | 5350 | 8596 | 0.622 | 0% |
| 4 | `mps_site` | c64 | chi=32 | 65536 | faer | packed | 10700 | 14978 | 0.714 | 1% |
| 4 | `mps_site` | f32 | chi=48 | 221184 | faer | packed | 4548 | 7303 | 0.623 | 1% |
| 4 | `mps_site` | f64 | chi=48 | 221184 | faer | packed | 9217 | 12102 | 0.762 | 1% |
| 4 | `mps_site` | c32 | chi=48 | 221184 | faer | packed | 21671 | 14036 | 1.544 | 10% |
| 4 | `mps_site` | c64 | chi=48 | 221184 | packed | packed | 21780 | 21420 | 1.017 | 7% |
| 4 | `mps_site` | f32 | chi=64 | 524288 | faer | packed | 12403 | 15659 | 0.792 | 9% |
| 4 | `mps_site` | f64 | chi=64 | 524288 | faer | packed | 17803 | 17273 | 1.031 | 51% |
| 4 | `mps_site` | c32 | chi=64 | 524288 | faer | packed | 27110 | 21911 | 1.237 | 20% |
| 4 | `mps_site` | c64 | chi=64 | 524288 | packed | packed | 33572 | 33433 | 1.004 | 5% |
| 4 | `mps_site` | f32 | chi=96 | 1769472 | faer | packed | 21671 | 19156 | 1.131 | 89% |
| 4 | `mps_site` | f64 | chi=96 | 1769472 | faer | packed | 75721 | 28824 | 2.627 | 49% |
| 4 | `mps_site` | c32 | chi=96 | 1769472 | faer | packed | 97472 | 50294 | 1.938 | 37% |
| 4 | `mps_site` | c64 | chi=96 | 1769472 | packed | packed | 109975 | 109425 | 1.005 | 2% |
| 4 | `mps_site` | f32 | chi=128 | 4194304 | faer | packed | 61505 | 40936 | 1.502 | 45% |
| 4 | `mps_site` | f64 | chi=128 | 4194304 | faer | packed | 59801 | 77695 | 0.770 | 32% |
| 4 | `mps_site` | c32 | chi=128 | 4194304 | faer | packed | 112179 | 103192 | 1.087 | 2% |
| 4 | `mps_site` | c64 | chi=128 | 4194304 | packed | packed | 192278 | 192239 | 1.000 | 1% |

Across the three sessions the 1T rows repeat within a median 0.4% / p90 1.9% /
max 12.3% spread; the 4T rows repeat within a median 1.2% / p90 13.8% /
max 89.1%, so 4T differences below that band are not findings.



### Second run (two arms, both C modes)

The second run re-measured the `absent` rows and added the `output` mode, on the
library revision named in `results/manifest.txt`. Both modes' rows are below;
`default/packed < 1` means the planner's own route was faster, `> 1` the
forced-packed arm.

#### `absent` (overwrite)

| c_mode | threads | class | dtype | params | mnk | default | packed | ns default | ns packed | default/packed | spread |
|---|---|---|---|---|---:|---|---|---:|---:|---:|---:|
| absent | 1 | `gemm` | f32 | n=32 | 32768 | faer | packed | 741 | 2565 | 0.289 | 0% |
| absent | 1 | `gemm` | f64 | n=32 | 32768 | faer | packed | 1403 | 6532 | 0.215 | 4% |
| absent | 1 | `gemm` | c32 | n=32 | 32768 | faer | packed | 2705 | 7494 | 0.361 | 3% |
| absent | 1 | `gemm` | c64 | n=32 | 32768 | faer | packed | 5300 | 7815 | 0.678 | 1% |
| absent | 1 | `gemm` | f32 | n=64 | 262144 | faer | packed | 5190 | 13425 | 0.387 | 9% |
| absent | 1 | `gemm` | f64 | n=64 | 262144 | faer | packed | 10339 | 22713 | 0.455 | 0% |
| absent | 1 | `gemm` | c32 | n=64 | 262144 | faer | packed | 20529 | 33764 | 0.608 | 1% |
| absent | 1 | `gemm` | c64 | n=64 | 262144 | packed | packed | 60654 | 60544 | 1.002 | 2% |
| absent | 1 | `gemm` | f32 | n=128 | 2097152 | faer | packed | 40445 | 69349 | 0.583 | 2% |
| absent | 1 | `gemm` | f64 | n=128 | 2097152 | faer | packed | 80500 | 129040 | 0.624 | 1% |
| absent | 1 | `gemm` | c32 | n=128 | 2097152 | faer | packed | 161093 | 205903 | 0.782 | 0% |
| absent | 1 | `gemm` | c64 | n=128 | 2097152 | packed | packed | 426856 | 426475 | 1.001 | 0% |
| absent | 1 | `gemm` | f32 | n=256 | 16777216 | faer | packed | 321239 | 396930 | 0.809 | 1% |
| absent | 1 | `gemm` | f64 | n=256 | 16777216 | faer | packed | 646715 | 861857 | 0.750 | 1% |
| absent | 1 | `gemm` | c32 | n=256 | 16777216 | faer | packed | 1286609 | 1353624 | 0.950 | 0% |
| absent | 1 | `gemm` | c64 | n=256 | 16777216 | packed | packed | 2703971 | 2692579 | 1.004 | 1% |
| absent | 1 | `gemm_batched` | f32 | n=16 batch=16 | 4096 | faer | packed | 2084 | 11161 | 0.187 | 64% |
| absent | 1 | `gemm_batched` | f64 | n=16 batch=16 | 4096 | faer | packed | 3496 | 9889 | 0.354 | 28% |
| absent | 1 | `gemm_batched` | c32 | n=16 batch=16 | 4096 | faer | packed | 9558 | 19997 | 0.478 | 0% |
| absent | 1 | `gemm_batched` | c64 | n=16 batch=16 | 4096 | faer | packed | 19336 | 19155 | 1.009 | 1% |
| absent | 1 | `gemm_batched` | f32 | n=32 batch=16 | 32768 | faer | packed | 11361 | 34214 | 0.332 | 1% |
| absent | 1 | `gemm_batched` | f64 | n=32 batch=16 | 32768 | faer | packed | 21781 | 45715 | 0.476 | 1% |
| absent | 1 | `gemm_batched` | c32 | n=32 batch=16 | 32768 | faer | packed | 42719 | 61895 | 0.690 | 0% |
| absent | 1 | `gemm_batched` | c64 | n=32 batch=16 | 32768 | faer | packed | 85129 | 119074 | 0.715 | 1% |
| absent | 1 | `gemm_batched` | f32 | n=64 batch=16 | 262144 | faer | packed | 82604 | 170800 | 0.484 | 0% |
| absent | 1 | `gemm_batched` | f64 | n=64 batch=16 | 262144 | faer | packed | 166270 | 236715 | 0.702 | 0% |
| absent | 1 | `gemm_batched` | c32 | n=64 batch=16 | 262144 | faer | packed | 328346 | 395197 | 0.831 | 1% |
| absent | 1 | `gemm_batched` | c64 | n=64 batch=16 | 262144 | packed | packed | 774023 | 773873 | 1.000 | 0% |
| absent | 1 | `gemm_batched` | f32 | n=32 batch=64 | 32768 | faer | packed | 45806 | 135765 | 0.337 | 11% |
| absent | 1 | `gemm_batched` | f64 | n=32 batch=64 | 32768 | faer | packed | 88406 | 183612 | 0.481 | 2% |
| absent | 1 | `gemm_batched` | c32 | n=32 batch=64 | 32768 | faer | packed | 172171 | 247442 | 0.696 | 0% |
| absent | 1 | `gemm_batched` | c64 | n=32 batch=64 | 32768 | faer | packed | 343175 | 482540 | 0.711 | 1% |
| absent | 1 | `gemm_batched` | f32 | n=64 batch=64 | 262144 | faer | packed | 333011 | 686599 | 0.485 | 1% |
| absent | 1 | `gemm_batched` | f64 | n=64 batch=64 | 262144 | faer | packed | 669979 | 991923 | 0.675 | 2% |
| absent | 1 | `gemm_batched` | c32 | n=64 batch=64 | 262144 | faer | packed | 1331532 | 1626492 | 0.819 | 2% |
| absent | 1 | `gemm_batched` | c64 | n=64 batch=64 | 262144 | packed | packed | 3751546 | 3770130 | 0.995 | 2% |
| absent | 1 | `mps_env` | f32 | chi=16 | 8192 | faer | packed | 320 | 1432 | 0.223 | 6% |
| absent | 1 | `mps_env` | f64 | chi=16 | 8192 | faer | packed | 461 | 1683 | 0.274 | 5% |
| absent | 1 | `mps_env` | c32 | chi=16 | 8192 | faer | packed | 792 | 1983 | 0.399 | 1% |
| absent | 1 | `mps_env` | c64 | chi=16 | 8192 | faer | packed | 1492 | 2765 | 0.540 | 1% |
| absent | 1 | `mps_env` | f32 | chi=24 | 27648 | faer | packed | 871 | 1883 | 0.463 | 1% |
| absent | 1 | `mps_env` | f64 | chi=24 | 27648 | faer | packed | 1252 | 2414 | 0.519 | 4% |
| absent | 1 | `mps_env` | c32 | chi=24 | 27648 | faer | packed | 2364 | 5290 | 0.447 | 1% |
| absent | 1 | `mps_env` | c64 | chi=24 | 27648 | faer | packed | 4629 | 8146 | 0.568 | 1% |
| absent | 1 | `mps_env` | f32 | chi=32 | 65536 | faer | packed | 1443 | 4448 | 0.324 | 2% |
| absent | 1 | `mps_env` | f64 | chi=32 | 65536 | faer | packed | 2785 | 5851 | 0.476 | 1% |
| absent | 1 | `mps_env` | c32 | chi=32 | 65536 | faer | packed | 5370 | 7985 | 0.673 | 0% |
| absent | 1 | `mps_env` | c64 | chi=32 | 65536 | faer | packed | 10740 | 14206 | 0.756 | 1% |
| absent | 1 | `mps_env` | f32 | chi=48 | 221184 | faer | packed | 4528 | 7303 | 0.620 | 0% |
| absent | 1 | `mps_env` | f64 | chi=48 | 221184 | faer | packed | 8937 | 12043 | 0.742 | 0% |
| absent | 1 | `mps_env` | c32 | chi=48 | 221184 | faer | packed | 17583 | 28433 | 0.618 | 0% |
| absent | 1 | `mps_env` | c64 | chi=48 | 221184 | packed | packed | 41558 | 41499 | 1.001 | 0% |
| absent | 1 | `mps_env` | f32 | chi=64 | 524288 | faer | packed | 10410 | 21020 | 0.495 | 0% |
| absent | 1 | `mps_env` | f64 | chi=64 | 524288 | faer | packed | 20899 | 28754 | 0.727 | 1% |
| absent | 1 | `mps_env` | c32 | chi=64 | 524288 | faer | packed | 41298 | 50515 | 0.818 | 1% |
| absent | 1 | `mps_env` | c64 | chi=64 | 524288 | packed | packed | 95940 | 95960 | 1.000 | 1% |
| absent | 1 | `mps_env` | f32 | chi=96 | 1769472 | faer | packed | 34595 | 43833 | 0.789 | 0% |
| absent | 1 | `mps_env` | f64 | chi=96 | 1769472 | faer | packed | 68939 | 79579 | 0.866 | 1% |
| absent | 1 | `mps_env` | c32 | chi=96 | 1769472 | faer | packed | 136989 | 154238 | 0.888 | 0% |
| absent | 1 | `mps_env` | c64 | chi=96 | 1769472 | packed | packed | 300097 | 300230 | 1.000 | 1% |
| absent | 1 | `mps_env` | f32 | chi=128 | 4194304 | faer | packed | 81293 | 108784 | 0.747 | 1% |
| absent | 1 | `mps_env` | f64 | chi=128 | 4194304 | faer | packed | 162806 | 204050 | 0.798 | 1% |
| absent | 1 | `mps_env` | c32 | chi=128 | 4194304 | faer | packed | 324540 | 358328 | 0.906 | 1% |
| absent | 1 | `mps_env` | c64 | chi=128 | 4194304 | packed | packed | 702018 | 701166 | 1.001 | 1% |
| absent | 1 | `mps_site` | f32 | chi=16 | 8192 | faer | packed | 300 | 1774 | 0.169 | 1% |
| absent | 1 | `mps_site` | f64 | chi=16 | 8192 | faer | packed | 471 | 1563 | 0.301 | 5% |
| absent | 1 | `mps_site` | c32 | chi=16 | 8192 | faer | packed | 801 | 2885 | 0.278 | 2% |
| absent | 1 | `mps_site` | c64 | chi=16 | 8192 | faer | packed | 1473 | 2846 | 0.518 | 2% |
| absent | 1 | `mps_site` | f32 | chi=24 | 27648 | faer | packed | 892 | 3086 | 0.289 | 1% |
| absent | 1 | `mps_site` | f64 | chi=24 | 27648 | faer | packed | 1262 | 2515 | 0.502 | 0% |
| absent | 1 | `mps_site` | c32 | chi=24 | 27648 | faer | packed | 2344 | 5129 | 0.457 | 0% |
| absent | 1 | `mps_site` | c64 | chi=24 | 27648 | faer | packed | 4608 | 8286 | 0.556 | 0% |
| absent | 1 | `mps_site` | f32 | chi=32 | 65536 | faer | packed | 1453 | 4668 | 0.311 | 1% |
| absent | 1 | `mps_site` | f64 | chi=32 | 65536 | faer | packed | 2835 | 6291 | 0.451 | 0% |
| absent | 1 | `mps_site` | c32 | chi=32 | 65536 | faer | packed | 5340 | 8866 | 0.602 | 1% |
| absent | 1 | `mps_site` | c64 | chi=32 | 65536 | faer | packed | 10701 | 15249 | 0.702 | 0% |
| absent | 1 | `mps_site` | f32 | chi=48 | 221184 | faer | packed | 4558 | 7494 | 0.608 | 1% |
| absent | 1 | `mps_site` | f64 | chi=48 | 221184 | faer | packed | 9197 | 12323 | 0.746 | 1% |
| absent | 1 | `mps_site` | c32 | chi=48 | 221184 | faer | packed | 17734 | 29195 | 0.607 | 1% |
| absent | 1 | `mps_site` | c64 | chi=48 | 221184 | packed | packed | 41278 | 41417 | 0.997 | 0% |
| absent | 1 | `mps_site` | f32 | chi=64 | 524288 | faer | packed | 10610 | 22091 | 0.480 | 0% |
| absent | 1 | `mps_site` | f64 | chi=64 | 524288 | faer | packed | 21470 | 29585 | 0.726 | 0% |
| absent | 1 | `mps_site` | c32 | chi=64 | 524288 | faer | packed | 41689 | 51667 | 0.807 | 0% |
| absent | 1 | `mps_site` | c64 | chi=64 | 524288 | packed | packed | 95631 | 95750 | 0.999 | 1% |
| absent | 1 | `mps_site` | f32 | chi=96 | 1769472 | faer | packed | 35126 | 45005 | 0.780 | 0% |
| absent | 1 | `mps_site` | f64 | chi=96 | 1769472 | faer | packed | 69731 | 80812 | 0.863 | 1% |
| absent | 1 | `mps_site` | c32 | chi=96 | 1769472 | faer | packed | 137437 | 157156 | 0.875 | 0% |
| absent | 1 | `mps_site` | c64 | chi=96 | 1769472 | packed | packed | 299395 | 299374 | 1.000 | 0% |
| absent | 1 | `mps_site` | f32 | chi=128 | 4194304 | faer | packed | 82444 | 113654 | 0.725 | 1% |
| absent | 1 | `mps_site` | f64 | chi=128 | 4194304 | faer | packed | 167355 | 206818 | 0.809 | 3% |
| absent | 1 | `mps_site` | c32 | chi=128 | 4194304 | faer | packed | 328442 | 376556 | 0.872 | 1% |
| absent | 1 | `mps_site` | c64 | chi=128 | 4194304 | packed | packed | 719596 | 718405 | 1.002 | 0% |
| absent | 4 | `gemm` | f32 | n=32 | 32768 | faer | packed | 761 | 2484 | 0.306 | 4% |
| absent | 4 | `gemm` | f64 | n=32 | 32768 | faer | packed | 1413 | 3206 | 0.441 | 2% |
| absent | 4 | `gemm` | c32 | n=32 | 32768 | faer | packed | 2725 | 4228 | 0.645 | 2% |
| absent | 4 | `gemm` | c64 | n=32 | 32768 | faer | packed | 5330 | 7725 | 0.690 | 1% |
| absent | 4 | `gemm` | f32 | n=64 | 262144 | faer | packed | 5219 | 11081 | 0.471 | 2% |
| absent | 4 | `gemm` | f64 | n=64 | 262144 | faer | packed | 10259 | 14898 | 0.689 | 1% |
| absent | 4 | `gemm` | c32 | n=64 | 262144 | faer | packed | 15439 | 13676 | 1.129 | 28% |
| absent | 4 | `gemm` | c64 | n=64 | 262144 | packed | packed | 20008 | 19967 | 1.002 | 16% |
| absent | 4 | `gemm` | f32 | n=128 | 2097152 | faer | packed | 21210 | 21019 | 1.009 | 88% |
| absent | 4 | `gemm` | f64 | n=128 | 2097152 | faer | packed | 39614 | 42459 | 0.933 | 35% |
| absent | 4 | `gemm` | c32 | n=128 | 2097152 | faer | packed | 59992 | 53129 | 1.129 | 7% |
| absent | 4 | `gemm` | c64 | n=128 | 2097152 | packed | packed | 99798 | 98696 | 1.011 | 2% |
| absent | 4 | `gemm` | f32 | n=256 | 16777216 | faer | packed | 113654 | 142267 | 0.799 | 24% |
| absent | 4 | `gemm` | f64 | n=256 | 16777216 | faer | packed | 183606 | 203193 | 0.904 | 5% |
| absent | 4 | `gemm` | c32 | n=256 | 16777216 | faer | packed | 347615 | 348216 | 0.998 | 0% |
| absent | 4 | `gemm` | c64 | n=256 | 16777216 | packed | packed | 696076 | 690485 | 1.008 | 3% |
| absent | 4 | `gemm_batched` | f32 | n=16 batch=16 | 4096 | faer | packed | 2084 | 11120 | 0.187 | 0% |
| absent | 4 | `gemm_batched` | f64 | n=16 batch=16 | 4096 | faer | packed | 3486 | 9628 | 0.362 | 2% |
| absent | 4 | `gemm_batched` | c32 | n=16 batch=16 | 4096 | faer | packed | 9568 | 20017 | 0.478 | 0% |
| absent | 4 | `gemm_batched` | c64 | n=16 batch=16 | 4096 | faer | packed | 19306 | 19066 | 1.013 | 1% |
| absent | 4 | `gemm_batched` | f32 | n=32 batch=16 | 32768 | faer | packed | 6623 | 15118 | 0.438 | 7% |
| absent | 4 | `gemm_batched` | f64 | n=32 batch=16 | 32768 | faer | packed | 11281 | 20147 | 0.560 | 3% |
| absent | 4 | `gemm_batched` | c32 | n=32 batch=16 | 32768 | faer | packed | 14197 | 19587 | 0.725 | 0% |
| absent | 4 | `gemm_batched` | c64 | n=32 batch=16 | 32768 | faer | packed | 24716 | 33593 | 0.736 | 3% |
| absent | 4 | `gemm_batched` | f32 | n=64 batch=16 | 262144 | faer | packed | 24125 | 46847 | 0.515 | 2% |
| absent | 4 | `gemm_batched` | f64 | n=64 batch=16 | 262144 | faer | packed | 44584 | 61985 | 0.719 | 2% |
| absent | 4 | `gemm_batched` | c32 | n=64 batch=16 | 262144 | faer | packed | 253774 | 102122 | 2.485 | 6% |
| absent | 4 | `gemm_batched` | c64 | n=64 batch=16 | 262144 | packed | packed | 196840 | 197969 | 0.994 | 1% |
| absent | 4 | `gemm_batched` | f32 | n=32 batch=64 | 32768 | faer | packed | 14848 | 38061 | 0.390 | 1% |
| absent | 4 | `gemm_batched` | f64 | n=32 batch=64 | 32768 | faer | packed | 24896 | 49383 | 0.504 | 4% |
| absent | 4 | `gemm_batched` | c32 | n=32 batch=64 | 32768 | faer | packed | 46297 | 65814 | 0.703 | 1% |
| absent | 4 | `gemm_batched` | c64 | n=32 batch=64 | 32768 | faer | packed | 88224 | 123412 | 0.715 | 1% |
| absent | 4 | `gemm_batched` | f32 | n=64 batch=64 | 262144 | faer | packed | 86391 | 175209 | 0.493 | 1% |
| absent | 4 | `gemm_batched` | f64 | n=64 batch=64 | 262144 | faer | packed | 170758 | 243494 | 0.701 | 3% |
| absent | 4 | `gemm_batched` | c32 | n=64 batch=64 | 262144 | faer | packed | 1015983 | 401273 | 2.532 | 1% |
| absent | 4 | `gemm_batched` | c64 | n=64 batch=64 | 262144 | packed | packed | 943176 | 945794 | 0.997 | 1% |
| absent | 4 | `mps_env` | f32 | chi=16 | 8192 | faer | packed | 310 | 1413 | 0.219 | 3% |
| absent | 4 | `mps_env` | f64 | chi=16 | 8192 | faer | packed | 481 | 1412 | 0.341 | 2% |
| absent | 4 | `mps_env` | c32 | chi=16 | 8192 | faer | packed | 811 | 1803 | 0.450 | 2% |
| absent | 4 | `mps_env` | c64 | chi=16 | 8192 | faer | packed | 1512 | 2655 | 0.569 | 3% |
| absent | 4 | `mps_env` | f32 | chi=24 | 27648 | faer | packed | 871 | 1744 | 0.499 | 2% |
| absent | 4 | `mps_env` | f64 | chi=24 | 27648 | faer | packed | 1272 | 2235 | 0.569 | 5% |
| absent | 4 | `mps_env` | c32 | chi=24 | 27648 | faer | packed | 2384 | 5039 | 0.473 | 3% |
| absent | 4 | `mps_env` | c64 | chi=24 | 27648 | faer | packed | 4608 | 8005 | 0.576 | 1% |
| absent | 4 | `mps_env` | f32 | chi=32 | 65536 | faer | packed | 1462 | 4258 | 0.343 | 4% |
| absent | 4 | `mps_env` | f64 | chi=32 | 65536 | faer | packed | 2755 | 5741 | 0.480 | 2% |
| absent | 4 | `mps_env` | c32 | chi=32 | 65536 | faer | packed | 5350 | 7744 | 0.691 | 1% |
| absent | 4 | `mps_env` | c64 | chi=32 | 65536 | faer | packed | 10580 | 13986 | 0.756 | 1% |
| absent | 4 | `mps_env` | f32 | chi=48 | 221184 | faer | packed | 4488 | 7133 | 0.629 | 1% |
| absent | 4 | `mps_env` | f64 | chi=48 | 221184 | faer | packed | 8856 | 11862 | 0.747 | 0% |
| absent | 4 | `mps_env` | c32 | chi=48 | 221184 | faer | packed | 11913 | 13896 | 0.857 | 25% |
| absent | 4 | `mps_env` | c64 | chi=48 | 221184 | packed | packed | 15739 | 15600 | 1.009 | 5% |
| absent | 4 | `mps_env` | f32 | chi=64 | 524288 | faer | packed | 10028 | 12163 | 0.824 | 40% |
| absent | 4 | `mps_env` | f64 | chi=64 | 524288 | faer | packed | 18244 | 17703 | 1.031 | 33% |
| absent | 4 | `mps_env` | c32 | chi=64 | 524288 | faer | packed | 23714 | 19507 | 1.216 | 51% |
| absent | 4 | `mps_env` | c64 | chi=64 | 524288 | packed | packed | 32542 | 32270 | 1.008 | 10% |
| absent | 4 | `mps_env` | f32 | chi=96 | 1769472 | faer | packed | 32671 | 18194 | 1.796 | 93% |
| absent | 4 | `mps_env` | f64 | chi=96 | 1769472 | faer | packed | 58720 | 27602 | 2.127 | 36% |
| absent | 4 | `mps_env` | c32 | chi=96 | 1769472 | faer | packed | 63610 | 46216 | 1.376 | 13% |
| absent | 4 | `mps_env` | c64 | chi=96 | 1769472 | packed | packed | 110917 | 110347 | 1.005 | 2% |
| absent | 4 | `mps_env` | f32 | chi=128 | 4194304 | faer | packed | 44223 | 35376 | 1.250 | 29% |
| absent | 4 | `mps_env` | f64 | chi=128 | 4194304 | faer | packed | 65543 | 76052 | 0.862 | 20% |
| absent | 4 | `mps_env` | c32 | chi=128 | 4194304 | faer | packed | 98954 | 98423 | 1.005 | 6% |
| absent | 4 | `mps_env` | c64 | chi=128 | 4194304 | packed | packed | 186560 | 185788 | 1.004 | 1% |
| absent | 4 | `mps_site` | f32 | chi=16 | 8192 | faer | packed | 321 | 1613 | 0.199 | 7% |
| absent | 4 | `mps_site` | f64 | chi=16 | 8192 | faer | packed | 481 | 1402 | 0.343 | 4% |
| absent | 4 | `mps_site` | c32 | chi=16 | 8192 | faer | packed | 811 | 2665 | 0.304 | 1% |
| absent | 4 | `mps_site` | c64 | chi=16 | 8192 | faer | packed | 1483 | 2605 | 0.569 | 3% |
| absent | 4 | `mps_site` | f32 | chi=24 | 27648 | faer | packed | 892 | 2895 | 0.308 | 2% |
| absent | 4 | `mps_site` | f64 | chi=24 | 27648 | faer | packed | 1262 | 2275 | 0.555 | 2% |
| absent | 4 | `mps_site` | c32 | chi=24 | 27648 | faer | packed | 2344 | 4839 | 0.484 | 2% |
| absent | 4 | `mps_site` | c64 | chi=24 | 27648 | faer | packed | 4608 | 7965 | 0.579 | 2% |
| absent | 4 | `mps_site` | f32 | chi=32 | 65536 | faer | packed | 1472 | 4489 | 0.328 | 1% |
| absent | 4 | `mps_site` | f64 | chi=32 | 65536 | faer | packed | 2835 | 6051 | 0.469 | 2% |
| absent | 4 | `mps_site` | c32 | chi=32 | 65536 | faer | packed | 5380 | 8536 | 0.630 | 1% |
| absent | 4 | `mps_site` | c64 | chi=32 | 65536 | faer | packed | 10730 | 14998 | 0.715 | 1% |
| absent | 4 | `mps_site` | f32 | chi=48 | 221184 | faer | packed | 4589 | 7324 | 0.627 | 1% |
| absent | 4 | `mps_site` | f64 | chi=48 | 221184 | faer | packed | 9217 | 12072 | 0.764 | 1% |
| absent | 4 | `mps_site` | c32 | chi=48 | 221184 | faer | packed | 18385 | 14107 | 1.303 | 23% |
| absent | 4 | `mps_site` | c64 | chi=48 | 221184 | packed | packed | 21059 | 21481 | 0.980 | 5% |
| absent | 4 | `mps_site` | f32 | chi=64 | 524288 | faer | packed | 11752 | 15338 | 0.766 | 13% |
| absent | 4 | `mps_site` | f64 | chi=64 | 524288 | faer | packed | 17783 | 16762 | 1.061 | 13% |
| absent | 4 | `mps_site` | c32 | chi=64 | 524288 | faer | packed | 29946 | 21260 | 1.409 | 27% |
| absent | 4 | `mps_site` | c64 | chi=64 | 524288 | packed | packed | 32661 | 33613 | 0.972 | 9% |
| absent | 4 | `mps_site` | f32 | chi=96 | 1769472 | faer | packed | 23033 | 19106 | 1.206 | 94% |
| absent | 4 | `mps_site` | f64 | chi=96 | 1769472 | faer | packed | 76935 | 28273 | 2.721 | 47% |
| absent | 4 | `mps_site` | c32 | chi=96 | 1769472 | faer | packed | 88815 | 50184 | 1.770 | 40% |
| absent | 4 | `mps_site` | c64 | chi=96 | 1769472 | packed | packed | 109907 | 110096 | 0.998 | 1% |
| absent | 4 | `mps_site` | f32 | chi=128 | 4194304 | faer | packed | 49533 | 41869 | 1.183 | 52% |
| absent | 4 | `mps_site` | f64 | chi=128 | 4194304 | faer | packed | 60443 | 76703 | 0.788 | 24% |
| absent | 4 | `mps_site` | c32 | chi=128 | 4194304 | faer | packed | 109805 | 102332 | 1.073 | 11% |
| absent | 4 | `mps_site` | c64 | chi=128 | 4194304 | packed | packed | 194825 | 192061 | 1.014 | 1% |

#### `output` (in-place accumulation)

| c_mode | threads | class | dtype | params | mnk | default | packed | ns default | ns packed | default/packed | spread |
|---|---|---|---|---|---:|---|---|---:|---:|---:|---:|
| output | 1 | `gemm` | f32 | n=32 | 32768 | faer | packed | 751 | 2624 | 0.286 | 1% |
| output | 1 | `gemm` | f64 | n=32 | 32768 | faer | packed | 1413 | 3477 | 0.406 | 1% |
| output | 1 | `gemm` | c32 | n=32 | 32768 | faer | packed | 2715 | 7654 | 0.355 | 3% |
| output | 1 | `gemm` | c64 | n=32 | 32768 | faer | packed | 5330 | 8546 | 0.624 | 2% |
| output | 1 | `gemm` | f32 | n=64 | 262144 | faer | packed | 5180 | 11271 | 0.460 | 1% |
| output | 1 | `gemm` | f64 | n=64 | 262144 | faer | packed | 10359 | 15359 | 0.674 | 0% |
| output | 1 | `gemm` | c32 | n=64 | 262144 | faer | packed | 20659 | 29635 | 0.697 | 0% |
| output | 1 | `gemm` | c64 | n=64 | 262144 | packed | packed | 64220 | 63679 | 1.008 | 1% |
| output | 1 | `gemm` | f32 | n=128 | 2097152 | faer | packed | 40606 | 56466 | 0.719 | 1% |
| output | 1 | `gemm` | f64 | n=128 | 2097152 | faer | packed | 80511 | 102132 | 0.788 | 1% |
| output | 1 | `gemm` | c32 | n=128 | 2097152 | faer | packed | 161882 | 198430 | 0.816 | 0% |
| output | 1 | `gemm` | c64 | n=128 | 2097152 | packed | packed | 436895 | 436734 | 1.000 | 1% |
| output | 1 | `gemm` | f32 | n=256 | 16777216 | faer | packed | 322361 | 399375 | 0.807 | 1% |
| output | 1 | `gemm` | f64 | n=256 | 16777216 | faer | packed | 649119 | 710013 | 0.914 | 1% |
| output | 1 | `gemm` | c32 | n=256 | 16777216 | faer | packed | 1290636 | 1424827 | 0.906 | 0% |
| output | 1 | `gemm` | c64 | n=256 | 16777216 | packed | packed | 2764845 | 2739847 | 1.009 | 2% |
| output | 1 | `gemm_batched` | f32 | n=16 batch=16 | 4096 | faer | packed | 2175 | 11442 | 0.190 | 60% |
| output | 1 | `gemm_batched` | f64 | n=16 batch=16 | 4096 | faer | packed | 3556 | 10109 | 0.352 | 28% |
| output | 1 | `gemm_batched` | c32 | n=16 batch=16 | 4096 | faer | packed | 9117 | 24246 | 0.376 | 1% |
| output | 1 | `gemm_batched` | c64 | n=16 batch=16 | 4096 | faer | packed | 18114 | 21691 | 0.835 | 1% |
| output | 1 | `gemm_batched` | f32 | n=32 batch=16 | 32768 | faer | packed | 11452 | 34955 | 0.328 | 1% |
| output | 1 | `gemm_batched` | f64 | n=32 batch=16 | 32768 | faer | packed | 21781 | 46757 | 0.466 | 1% |
| output | 1 | `gemm_batched` | c32 | n=32 batch=16 | 32768 | faer | packed | 42930 | 80590 | 0.533 | 0% |
| output | 1 | `gemm_batched` | c64 | n=32 batch=16 | 32768 | faer | packed | 86632 | 130365 | 0.665 | 1% |
| output | 1 | `gemm_batched` | f32 | n=64 batch=16 | 262144 | faer | packed | 82774 | 173066 | 0.478 | 0% |
| output | 1 | `gemm_batched` | f64 | n=64 batch=16 | 262144 | faer | packed | 167552 | 240992 | 0.695 | 1% |
| output | 1 | `gemm_batched` | c32 | n=64 batch=16 | 262144 | faer | packed | 330837 | 469723 | 0.704 | 0% |
| output | 1 | `gemm_batched` | c64 | n=64 batch=16 | 262144 | packed | packed | 817444 | 817674 | 1.000 | 0% |
| output | 1 | `gemm_batched` | f32 | n=32 batch=64 | 32768 | faer | packed | 46637 | 138400 | 0.337 | 12% |
| output | 1 | `gemm_batched` | f64 | n=32 batch=64 | 32768 | faer | packed | 89759 | 189233 | 0.474 | 2% |
| output | 1 | `gemm_batched` | c32 | n=32 batch=64 | 32768 | faer | packed | 173354 | 322130 | 0.538 | 1% |
| output | 1 | `gemm_batched` | c64 | n=32 batch=64 | 32768 | faer | packed | 351801 | 528005 | 0.666 | 1% |
| output | 1 | `gemm_batched` | f32 | n=64 batch=64 | 262144 | faer | packed | 333667 | 694105 | 0.481 | 0% |
| output | 1 | `gemm_batched` | f64 | n=64 batch=64 | 262144 | faer | packed | 684606 | 1028538 | 0.666 | 5% |
| output | 1 | `gemm_batched` | c32 | n=64 batch=64 | 262144 | faer | packed | 1346330 | 1926598 | 0.699 | 6% |
| output | 1 | `gemm_batched` | c64 | n=64 batch=64 | 262144 | packed | packed | 4038360 | 4042171 | 0.999 | 2% |
| output | 1 | `mps_env` | f32 | chi=16 | 8192 | faer | packed | 330 | 1613 | 0.205 | 6% |
| output | 1 | `mps_env` | f64 | chi=16 | 8192 | faer | packed | 461 | 1734 | 0.266 | 4% |
| output | 1 | `mps_env` | c32 | chi=16 | 8192 | faer | packed | 801 | 2454 | 0.326 | 1% |
| output | 1 | `mps_env` | c64 | chi=16 | 8192 | faer | packed | 1492 | 3216 | 0.464 | 4% |
| output | 1 | `mps_env` | f32 | chi=24 | 27648 | faer | packed | 1042 | 2104 | 0.495 | 1% |
| output | 1 | `mps_env` | f64 | chi=24 | 27648 | faer | packed | 1252 | 2474 | 0.506 | 3% |
| output | 1 | `mps_env` | c32 | chi=24 | 27648 | faer | packed | 2375 | 6112 | 0.389 | 1% |
| output | 1 | `mps_env` | c64 | chi=24 | 27648 | faer | packed | 4739 | 8957 | 0.529 | 2% |
| output | 1 | `mps_env` | f32 | chi=32 | 65536 | faer | packed | 1452 | 4809 | 0.302 | 1% |
| output | 1 | `mps_env` | f64 | chi=32 | 65536 | faer | packed | 2785 | 6172 | 0.451 | 1% |
| output | 1 | `mps_env` | c32 | chi=32 | 65536 | faer | packed | 5420 | 10400 | 0.521 | 1% |
| output | 1 | `mps_env` | c64 | chi=32 | 65536 | faer | packed | 10841 | 15680 | 0.691 | 0% |
| output | 1 | `mps_env` | f32 | chi=48 | 221184 | faer | packed | 4518 | 7474 | 0.604 | 0% |
| output | 1 | `mps_env` | f64 | chi=48 | 221184 | faer | packed | 8916 | 12303 | 0.725 | 0% |
| output | 1 | `mps_env` | c32 | chi=48 | 221184 | faer | packed | 17643 | 33532 | 0.526 | 1% |
| output | 1 | `mps_env` | c64 | chi=48 | 221184 | packed | packed | 44494 | 44504 | 1.000 | 0% |
| output | 1 | `mps_env` | f32 | chi=64 | 524288 | faer | packed | 10409 | 21380 | 0.487 | 1% |
| output | 1 | `mps_env` | f64 | chi=64 | 524288 | faer | packed | 20779 | 29425 | 0.706 | 1% |
| output | 1 | `mps_env` | c32 | chi=64 | 524288 | faer | packed | 41398 | 59942 | 0.691 | 1% |
| output | 1 | `mps_env` | c64 | chi=64 | 524288 | packed | packed | 101240 | 101330 | 0.999 | 1% |
| output | 1 | `mps_env` | f32 | chi=96 | 1769472 | faer | packed | 34625 | 44414 | 0.780 | 0% |
| output | 1 | `mps_env` | f64 | chi=96 | 1769472 | faer | packed | 68990 | 80431 | 0.858 | 1% |
| output | 1 | `mps_env` | c32 | chi=96 | 1769472 | faer | packed | 138041 | 175029 | 0.789 | 0% |
| output | 1 | `mps_env` | c64 | chi=96 | 1769472 | packed | packed | 312332 | 312021 | 1.001 | 1% |
| output | 1 | `mps_env` | f32 | chi=128 | 4194304 | faer | packed | 81672 | 110547 | 0.739 | 1% |
| output | 1 | `mps_env` | f64 | chi=128 | 4194304 | faer | packed | 162816 | 204344 | 0.797 | 1% |
| output | 1 | `mps_env` | c32 | chi=128 | 4194304 | faer | packed | 325862 | 394845 | 0.825 | 1% |
| output | 1 | `mps_env` | c64 | chi=128 | 4194304 | packed | packed | 725342 | 724690 | 1.001 | 1% |
| output | 1 | `mps_site` | f32 | chi=16 | 8192 | faer | packed | 300 | 1813 | 0.165 | 2% |
| output | 1 | `mps_site` | f64 | chi=16 | 8192 | faer | packed | 471 | 1613 | 0.292 | 2% |
| output | 1 | `mps_site` | c32 | chi=16 | 8192 | faer | packed | 791 | 3196 | 0.247 | 3% |
| output | 1 | `mps_site` | c64 | chi=16 | 8192 | faer | packed | 1483 | 3035 | 0.489 | 2% |
| output | 1 | `mps_site` | f32 | chi=24 | 27648 | faer | packed | 951 | 3156 | 0.301 | 1% |
| output | 1 | `mps_site` | f64 | chi=24 | 27648 | faer | packed | 1262 | 2585 | 0.488 | 2% |
| output | 1 | `mps_site` | c32 | chi=24 | 27648 | faer | packed | 2344 | 5761 | 0.407 | 1% |
| output | 1 | `mps_site` | c64 | chi=24 | 27648 | faer | packed | 4678 | 8666 | 0.540 | 1% |
| output | 1 | `mps_site` | f32 | chi=32 | 65536 | faer | packed | 1452 | 4749 | 0.306 | 1% |
| output | 1 | `mps_site` | f64 | chi=32 | 65536 | faer | packed | 2795 | 6382 | 0.438 | 1% |
| output | 1 | `mps_site` | c32 | chi=32 | 65536 | faer | packed | 5340 | 10039 | 0.532 | 0% |
| output | 1 | `mps_site` | c64 | chi=32 | 65536 | faer | packed | 10781 | 15920 | 0.677 | 1% |
| output | 1 | `mps_site` | f32 | chi=48 | 221184 | faer | packed | 4558 | 7514 | 0.607 | 1% |
| output | 1 | `mps_site` | f64 | chi=48 | 221184 | faer | packed | 9227 | 12384 | 0.745 | 0% |
| output | 1 | `mps_site` | c32 | chi=48 | 221184 | faer | packed | 17893 | 31880 | 0.561 | 0% |
| output | 1 | `mps_site` | c64 | chi=48 | 221184 | packed | packed | 42941 | 42911 | 1.001 | 0% |
| output | 1 | `mps_site` | f32 | chi=64 | 524288 | faer | packed | 10640 | 22502 | 0.473 | 0% |
| output | 1 | `mps_site` | f64 | chi=64 | 524288 | faer | packed | 21431 | 29847 | 0.718 | 0% |
| output | 1 | `mps_site` | c32 | chi=64 | 524288 | faer | packed | 41618 | 56385 | 0.738 | 0% |
| output | 1 | `mps_site` | c64 | chi=64 | 524288 | packed | packed | 98704 | 98475 | 1.002 | 1% |
| output | 1 | `mps_site` | f32 | chi=96 | 1769472 | faer | packed | 35096 | 45165 | 0.777 | 0% |
| output | 1 | `mps_site` | f64 | chi=96 | 1769472 | faer | packed | 69862 | 81232 | 0.860 | 0% |
| output | 1 | `mps_site` | c32 | chi=96 | 1769472 | faer | packed | 137960 | 167332 | 0.824 | 0% |
| output | 1 | `mps_site` | c64 | chi=96 | 1769472 | packed | packed | 305169 | 305106 | 1.000 | 0% |
| output | 1 | `mps_site` | f32 | chi=128 | 4194304 | faer | packed | 82705 | 114204 | 0.724 | 0% |
| output | 1 | `mps_site` | f64 | chi=128 | 4194304 | faer | packed | 167886 | 212328 | 0.791 | 2% |
| output | 1 | `mps_site` | c32 | chi=128 | 4194304 | faer | packed | 328844 | 394696 | 0.833 | 0% |
| output | 1 | `mps_site` | c64 | chi=128 | 4194304 | packed | packed | 730102 | 730327 | 1.000 | 0% |
| output | 4 | `gemm` | f32 | n=32 | 32768 | faer | packed | 751 | 2575 | 0.292 | 1% |
| output | 4 | `gemm` | f64 | n=32 | 32768 | faer | packed | 1413 | 3437 | 0.411 | 2% |
| output | 4 | `gemm` | c32 | n=32 | 32768 | faer | packed | 2725 | 5420 | 0.503 | 2% |
| output | 4 | `gemm` | c64 | n=32 | 32768 | faer | packed | 5390 | 8496 | 0.634 | 1% |
| output | 4 | `gemm` | f32 | n=64 | 262144 | faer | packed | 5210 | 11291 | 0.461 | 2% |
| output | 4 | `gemm` | f64 | n=64 | 262144 | faer | packed | 10309 | 15279 | 0.675 | 1% |
| output | 4 | `gemm` | c32 | n=64 | 262144 | faer | packed | 15288 | 15379 | 0.994 | 17% |
| output | 4 | `gemm` | c64 | n=64 | 262144 | packed | packed | 22552 | 21771 | 1.036 | 8% |
| output | 4 | `gemm` | f32 | n=128 | 2097152 | faer | packed | 26940 | 20488 | 1.315 | 49% |
| output | 4 | `gemm` | f64 | n=128 | 2097152 | faer | packed | 33563 | 43441 | 0.773 | 25% |
| output | 4 | `gemm` | c32 | n=128 | 2097152 | faer | packed | 57979 | 61015 | 0.950 | 5% |
| output | 4 | `gemm` | c64 | n=128 | 2097152 | packed | packed | 104356 | 104216 | 1.001 | 3% |
| output | 4 | `gemm` | f32 | n=256 | 16777216 | faer | packed | 120807 | 143860 | 0.840 | 27% |
| output | 4 | `gemm` | f64 | n=256 | 16777216 | faer | packed | 180658 | 205894 | 0.877 | 4% |
| output | 4 | `gemm` | c32 | n=256 | 16777216 | faer | packed | 355039 | 378997 | 0.937 | 2% |
| output | 4 | `gemm` | c64 | n=256 | 16777216 | packed | packed | 707789 | 708785 | 0.999 | 1% |
| output | 4 | `gemm_batched` | f32 | n=16 batch=16 | 4096 | faer | packed | 2134 | 11441 | 0.187 | 2% |
| output | 4 | `gemm_batched` | f64 | n=16 batch=16 | 4096 | faer | packed | 3557 | 9928 | 0.358 | 2% |
| output | 4 | `gemm_batched` | c32 | n=16 batch=16 | 4096 | faer | packed | 9137 | 24156 | 0.378 | 0% |
| output | 4 | `gemm_batched` | c64 | n=16 batch=16 | 4096 | faer | packed | 18164 | 21851 | 0.831 | 2% |
| output | 4 | `gemm_batched` | f32 | n=32 batch=16 | 32768 | faer | packed | 6742 | 15168 | 0.444 | 8% |
| output | 4 | `gemm_batched` | f64 | n=32 batch=16 | 32768 | faer | packed | 11401 | 20989 | 0.543 | 2% |
| output | 4 | `gemm_batched` | c32 | n=32 batch=16 | 32768 | faer | packed | 13716 | 24376 | 0.563 | 7% |
| output | 4 | `gemm_batched` | c64 | n=32 batch=16 | 32768 | faer | packed | 25167 | 37009 | 0.680 | 4% |
| output | 4 | `gemm_batched` | f32 | n=64 batch=16 | 262144 | faer | packed | 24135 | 47239 | 0.511 | 4% |
| output | 4 | `gemm_batched` | f64 | n=64 batch=16 | 262144 | faer | packed | 44904 | 63669 | 0.705 | 2% |
| output | 4 | `gemm_batched` | c32 | n=64 batch=16 | 262144 | faer | packed | 247501 | 120775 | 2.049 | 6% |
| output | 4 | `gemm_batched` | c64 | n=64 batch=16 | 262144 | packed | packed | 208258 | 208092 | 1.001 | 1% |
| output | 4 | `gemm_batched` | f32 | n=32 batch=64 | 32768 | faer | packed | 14857 | 38823 | 0.383 | 5% |
| output | 4 | `gemm_batched` | f64 | n=32 batch=64 | 32768 | faer | packed | 25618 | 50766 | 0.505 | 1% |
| output | 4 | `gemm_batched` | c32 | n=32 batch=64 | 32768 | faer | packed | 46647 | 84269 | 0.554 | 1% |
| output | 4 | `gemm_batched` | c64 | n=32 batch=64 | 32768 | faer | packed | 89878 | 134492 | 0.668 | 1% |
| output | 4 | `gemm_batched` | f32 | n=64 batch=64 | 262144 | faer | packed | 86692 | 177394 | 0.489 | 1% |
| output | 4 | `gemm_batched` | f64 | n=64 batch=64 | 262144 | faer | packed | 171572 | 248927 | 0.689 | 2% |
| output | 4 | `gemm_batched` | c32 | n=64 batch=64 | 262144 | faer | packed | 1022256 | 476448 | 2.146 | 2% |
| output | 4 | `gemm_batched` | c64 | n=64 batch=64 | 262144 | packed | packed | 998510 | 998191 | 1.000 | 1% |
| output | 4 | `mps_env` | f32 | chi=16 | 8192 | faer | packed | 340 | 1493 | 0.228 | 9% |
| output | 4 | `mps_env` | f64 | chi=16 | 8192 | faer | packed | 471 | 1543 | 0.305 | 4% |
| output | 4 | `mps_env` | c32 | chi=16 | 8192 | faer | packed | 821 | 2224 | 0.369 | 4% |
| output | 4 | `mps_env` | c64 | chi=16 | 8192 | faer | packed | 1503 | 3025 | 0.497 | 2% |
| output | 4 | `mps_env` | f32 | chi=24 | 27648 | faer | packed | 1012 | 1923 | 0.526 | 2% |
| output | 4 | `mps_env` | f64 | chi=24 | 27648 | faer | packed | 1252 | 2294 | 0.546 | 1% |
| output | 4 | `mps_env` | c32 | chi=24 | 27648 | faer | packed | 2374 | 5851 | 0.406 | 1% |
| output | 4 | `mps_env` | c64 | chi=24 | 27648 | faer | packed | 4728 | 8776 | 0.539 | 1% |
| output | 4 | `mps_env` | f32 | chi=32 | 65536 | faer | packed | 1443 | 4619 | 0.312 | 2% |
| output | 4 | `mps_env` | f64 | chi=32 | 65536 | faer | packed | 2766 | 5941 | 0.466 | 2% |
| output | 4 | `mps_env` | c32 | chi=32 | 65536 | faer | packed | 5380 | 10088 | 0.533 | 1% |
| output | 4 | `mps_env` | c64 | chi=32 | 65536 | faer | packed | 10730 | 15449 | 0.695 | 3% |
| output | 4 | `mps_env` | f32 | chi=48 | 221184 | faer | packed | 4508 | 7284 | 0.619 | 1% |
| output | 4 | `mps_env` | f64 | chi=48 | 221184 | faer | packed | 8896 | 12083 | 0.736 | 0% |
| output | 4 | `mps_env` | c32 | chi=48 | 221184 | faer | packed | 12513 | 16200 | 0.772 | 26% |
| output | 4 | `mps_env` | c64 | chi=48 | 221184 | packed | packed | 16481 | 16641 | 0.990 | 6% |
| output | 4 | `mps_env` | f32 | chi=64 | 524288 | faer | packed | 13235 | 13275 | 0.997 | 23% |
| output | 4 | `mps_env` | f64 | chi=64 | 524288 | faer | packed | 20708 | 18435 | 1.123 | 22% |
| output | 4 | `mps_env` | c32 | chi=64 | 524288 | faer | packed | 18294 | 23203 | 0.788 | 28% |
| output | 4 | `mps_env` | c64 | chi=64 | 524288 | packed | packed | 35256 | 35526 | 0.992 | 5% |
| output | 4 | `mps_env` | f32 | chi=96 | 1769472 | faer | packed | 33423 | 18435 | 1.813 | 78% |
| output | 4 | `mps_env` | f64 | chi=96 | 1769472 | faer | packed | 41799 | 28694 | 1.457 | 7% |
| output | 4 | `mps_env` | c32 | chi=96 | 1769472 | faer | packed | 67156 | 51756 | 1.298 | 44% |
| output | 4 | `mps_env` | c64 | chi=96 | 1769472 | packed | packed | 118563 | 118743 | 0.998 | 1% |
| output | 4 | `mps_env` | f32 | chi=128 | 4194304 | faer | packed | 28633 | 36337 | 0.788 | 55% |
| output | 4 | `mps_env` | f64 | chi=128 | 4194304 | faer | packed | 54182 | 77436 | 0.700 | 2% |
| output | 4 | `mps_env` | c32 | chi=128 | 4194304 | faer | packed | 112642 | 114977 | 0.980 | 11% |
| output | 4 | `mps_env` | c64 | chi=128 | 4194304 | packed | packed | 197247 | 197320 | 1.000 | 1% |
| output | 4 | `mps_site` | f32 | chi=16 | 8192 | faer | packed | 320 | 1663 | 0.192 | 6% |
| output | 4 | `mps_site` | f64 | chi=16 | 8192 | faer | packed | 481 | 1452 | 0.331 | 2% |
| output | 4 | `mps_site` | c32 | chi=16 | 8192 | faer | packed | 812 | 2925 | 0.278 | 2% |
| output | 4 | `mps_site` | c64 | chi=16 | 8192 | faer | packed | 1482 | 2795 | 0.530 | 1% |
| output | 4 | `mps_site` | f32 | chi=24 | 27648 | faer | packed | 952 | 2965 | 0.321 | 1% |
| output | 4 | `mps_site` | f64 | chi=24 | 27648 | faer | packed | 1282 | 2324 | 0.552 | 2% |
| output | 4 | `mps_site` | c32 | chi=24 | 27648 | faer | packed | 2374 | 5490 | 0.432 | 1% |
| output | 4 | `mps_site` | c64 | chi=24 | 27648 | faer | packed | 4688 | 8365 | 0.560 | 1% |
| output | 4 | `mps_site` | f32 | chi=32 | 65536 | faer | packed | 1462 | 4568 | 0.320 | 1% |
| output | 4 | `mps_site` | f64 | chi=32 | 65536 | faer | packed | 2836 | 6201 | 0.457 | 1% |
| output | 4 | `mps_site` | c32 | chi=32 | 65536 | faer | packed | 5380 | 9738 | 0.552 | 1% |
| output | 4 | `mps_site` | c64 | chi=32 | 65536 | faer | packed | 10850 | 15650 | 0.693 | 1% |
| output | 4 | `mps_site` | f32 | chi=48 | 221184 | faer | packed | 4609 | 7414 | 0.622 | 2% |
| output | 4 | `mps_site` | f64 | chi=48 | 221184 | faer | packed | 9267 | 12223 | 0.758 | 0% |
| output | 4 | `mps_site` | c32 | chi=48 | 221184 | faer | packed | 17653 | 15559 | 1.135 | 25% |
| output | 4 | `mps_site` | c64 | chi=48 | 221184 | packed | packed | 22231 | 22171 | 1.003 | 4% |
| output | 4 | `mps_site` | f32 | chi=64 | 524288 | faer | packed | 13265 | 15398 | 0.861 | 22% |
| output | 4 | `mps_site` | f64 | chi=64 | 524288 | faer | packed | 22222 | 17813 | 1.248 | 37% |
| output | 4 | `mps_site` | c32 | chi=64 | 524288 | faer | packed | 31789 | 23284 | 1.365 | 20% |
| output | 4 | `mps_site` | c64 | chi=64 | 524288 | packed | packed | 35456 | 35407 | 1.001 | 0% |
| output | 4 | `mps_site` | f32 | chi=96 | 1769472 | faer | packed | 29275 | 19627 | 1.492 | 29% |
| output | 4 | `mps_site` | f64 | chi=96 | 1769472 | faer | packed | 74218 | 29165 | 2.545 | 12% |
| output | 4 | `mps_site` | c32 | chi=96 | 1769472 | faer | packed | 89948 | 53070 | 1.695 | 18% |
| output | 4 | `mps_site` | c64 | chi=96 | 1769472 | packed | packed | 115417 | 115315 | 1.001 | 1% |
| output | 4 | `mps_site` | f32 | chi=128 | 4194304 | faer | packed | 61044 | 41948 | 1.455 | 19% |
| output | 4 | `mps_site` | f64 | chi=128 | 4194304 | faer | packed | 63640 | 77594 | 0.820 | 21% |
| output | 4 | `mps_site` | c32 | chi=128 | 4194304 | faer | packed | 108233 | 110898 | 0.976 | 12% |
| output | 4 | `mps_site` | c64 | chi=128 | 4194304 | packed | packed | 199114 | 197779 | 1.007 | 1% |

The second run's `absent` rows reproduce the earlier table within the session
spread. Over the 184 `(threads, class, dtype, params)` keys, the
median-over-sessions `output`/`absent` ns ratio is 0.65 / 1.01 / 1.32
(min / median / max) for the `default` arm and 0.53 / 1.04 / 1.30 for the
`packed` arm: for most rows the per-call `output` cost is within the session
noise of `absent`, and the largest shifts are on the small / 1T rows.

Across the three sessions of the second run the `absent` rows repeat within a
median 0.4% / p90 1.8% / max 63.9% spread at 1T and a median 1.4% / p90 22.6% /
max 94.4% at 4T; the `output` rows within a median 0.5% / p90 2.0% / max 59.9%
at 1T and a median 1.4% / p90 17.6% / max 78.4% at 4T. As before, 4T differences
below that band are not findings.

The `CHECK` lines in `results/cmode-session{1,2,3}.txt` (368 per session: one
per case, dtype, thread count and C mode) show both modes matching their
references. The worst `rel` over all cases and sessions is 1.13e-6 for `absent`
and 1.11e-6 for `output`, both on the `f32`/`c32` rows (tol 1e-5); the two arms
agree to at most 4.6e-7. No case failed.

### Third run (three arms, both C modes)

The third run adds the `faer_forced` arm
(`PlanConfig { faer_limit: FaerLimit::NONE, ..PlanConfig::default() }`, every
fusable problem to faer, the rule before #63) beside `default` and `packed`,
on the same cases, both C modes and both thread counts as the second run. It is
recorded as `results/triarm-session{1,2,3}.{csv,txt,guard}` (1104 rows per
session). `default` and `packed` are the same arms as in the second run; the
`faer_forced` arm is the only one that reaches faer above the dtype's
`FaerLimit`, so it is the c64 faer time the `default` route cannot show.

`faer_forced` reached faer for all 368 case/dtype/thread/mode rows in every
session (`faer_forced=faer`, `route_faer=fused`); no problem was refused. The
ratio column is `ns packed / ns faer_forced` (`> 1` means faer was faster),
which for a row where `default` routes to faer duplicates `packed/faer`; for a
row where `default` routes to packed it is the only faer/packed comparison.
For c64 above `mnk = 131072` every row's `default` column is `packed`, so
`default` and `packed` are the same engine there.

#### `absent` (overwrite)

| c_mode | threads | class | dtype | params | mnk | default | packed | faer_forced | ns default | ns packed | ns faer_forced | packed/faer_forced | spread |
|---|---|---|---|---|---:|---|---|---|---:|---:|---:|---:|---:|
| absent | 1 | `gemm` | f32 | n=32 | 32768 | faer | packed | faer | 751 | 2354 | 751 | 3.134 | 1% |
| absent | 1 | `gemm` | f64 | n=32 | 32768 | faer | packed | faer | 1402 | 8296 | 1402 | 5.917 | 3% |
| absent | 1 | `gemm` | c32 | n=32 | 32768 | faer | packed | faer | 2705 | 9318 | 2705 | 3.445 | 22% |
| absent | 1 | `gemm` | c64 | n=32 | 32768 | faer | packed | faer | 5300 | 7805 | 5300 | 1.473 | 2% |
| absent | 1 | `gemm` | f32 | n=64 | 262144 | faer | packed | faer | 5180 | 14266 | 5179 | 2.755 | 2% |
| absent | 1 | `gemm` | f64 | n=64 | 262144 | faer | packed | faer | 10279 | 23524 | 10289 | 2.286 | 1% |
| absent | 1 | `gemm` | c32 | n=64 | 262144 | faer | packed | faer | 20469 | 34264 | 20458 | 1.675 | 1% |
| absent | 1 | `gemm` | c64 | n=64 | 262144 | packed | packed | faer | 61926 | 61856 | 40846 | 1.514 | 1% |
| absent | 1 | `gemm` | f32 | n=128 | 2097152 | faer | packed | faer | 40355 | 69781 | 40364 | 1.729 | 1% |
| absent | 1 | `gemm` | f64 | n=128 | 2097152 | faer | packed | faer | 80579 | 129650 | 80529 | 1.610 | 0% |
| absent | 1 | `gemm` | c32 | n=128 | 2097152 | faer | packed | faer | 161161 | 206706 | 161131 | 1.283 | 1% |
| absent | 1 | `gemm` | c64 | n=128 | 2097152 | packed | packed | faer | 428214 | 428821 | 322884 | 1.328 | 1% |
| absent | 1 | `gemm` | f32 | n=256 | 16777216 | faer | packed | faer | 321245 | 398309 | 321331 | 1.240 | 1% |
| absent | 1 | `gemm` | f64 | n=256 | 16777216 | faer | packed | faer | 646007 | 861100 | 646768 | 1.331 | 1% |
| absent | 1 | `gemm` | c32 | n=256 | 16777216 | faer | packed | faer | 1287407 | 1352727 | 1287627 | 1.051 | 0% |
| absent | 1 | `gemm` | c64 | n=256 | 16777216 | packed | packed | faer | 2696220 | 2703674 | 2579372 | 1.048 | 1% |
| absent | 1 | `gemm_batched` | f32 | n=16 batch=16 | 4096 | faer | packed | faer | 2054 | 11081 | 2054 | 5.395 | 3% |
| absent | 1 | `gemm_batched` | f64 | n=16 batch=16 | 4096 | faer | packed | faer | 3537 | 9287 | 3546 | 2.619 | 3% |
| absent | 1 | `gemm_batched` | c32 | n=16 batch=16 | 4096 | faer | packed | faer | 9598 | 20639 | 9588 | 2.153 | 2% |
| absent | 1 | `gemm_batched` | c64 | n=16 batch=16 | 4096 | faer | packed | faer | 19366 | 19306 | 19367 | 0.997 | 2% |
| absent | 1 | `gemm_batched` | f32 | n=32 batch=16 | 32768 | faer | packed | faer | 11412 | 30597 | 11431 | 2.677 | 1% |
| absent | 1 | `gemm_batched` | f64 | n=32 batch=16 | 32768 | faer | packed | faer | 21921 | 46206 | 21891 | 2.111 | 0% |
| absent | 1 | `gemm_batched` | c32 | n=32 batch=16 | 32768 | faer | packed | faer | 42720 | 63549 | 42649 | 1.490 | 1% |
| absent | 1 | `gemm_batched` | c64 | n=32 batch=16 | 32768 | faer | packed | faer | 84548 | 118710 | 84598 | 1.403 | 2% |
| absent | 1 | `gemm_batched` | f32 | n=64 batch=16 | 262144 | faer | packed | faer | 82655 | 170626 | 82564 | 2.067 | 1% |
| absent | 1 | `gemm_batched` | f64 | n=64 batch=16 | 262144 | faer | packed | faer | 165128 | 235330 | 165116 | 1.425 | 1% |
| absent | 1 | `gemm_batched` | c32 | n=64 batch=16 | 262144 | faer | packed | faer | 328184 | 395293 | 328444 | 1.204 | 1% |
| absent | 1 | `gemm_batched` | c64 | n=64 batch=16 | 262144 | packed | packed | faer | 775249 | 774978 | 660084 | 1.174 | 1% |
| absent | 1 | `gemm_batched` | f32 | n=32 batch=64 | 32768 | faer | packed | faer | 45715 | 121878 | 45635 | 2.671 | 2% |
| absent | 1 | `gemm_batched` | f64 | n=32 batch=64 | 32768 | faer | packed | faer | 89086 | 186999 | 89015 | 2.101 | 2% |
| absent | 1 | `gemm_batched` | c32 | n=32 batch=64 | 32768 | faer | packed | faer | 172479 | 253430 | 172703 | 1.467 | 1% |
| absent | 1 | `gemm_batched` | c64 | n=32 batch=64 | 32768 | faer | packed | faer | 342816 | 481439 | 343166 | 1.403 | 2% |
| absent | 1 | `gemm_batched` | f32 | n=64 batch=64 | 262144 | faer | packed | faer | 332506 | 687775 | 332176 | 2.071 | 0% |
| absent | 1 | `gemm_batched` | f64 | n=64 batch=64 | 262144 | faer | packed | faer | 669831 | 975910 | 669261 | 1.458 | 9% |
| absent | 1 | `gemm_batched` | c32 | n=64 batch=64 | 262144 | faer | packed | faer | 1328161 | 1610841 | 1322894 | 1.218 | 1% |
| absent | 1 | `gemm_batched` | c64 | n=64 batch=64 | 262144 | packed | packed | faer | 3702588 | 3731120 | 2732535 | 1.365 | 6% |
| absent | 1 | `mps_env` | f32 | chi=16 | 8192 | faer | packed | faer | 301 | 1603 | 310 | 5.171 | 7% |
| absent | 1 | `mps_env` | f64 | chi=16 | 8192 | faer | packed | faer | 461 | 1523 | 461 | 3.304 | 2% |
| absent | 1 | `mps_env` | c32 | chi=16 | 8192 | faer | packed | faer | 801 | 1974 | 802 | 2.461 | 2% |
| absent | 1 | `mps_env` | c64 | chi=16 | 8192 | faer | packed | faer | 1493 | 2775 | 1493 | 1.859 | 3% |
| absent | 1 | `mps_env` | f32 | chi=24 | 27648 | faer | packed | faer | 901 | 1864 | 902 | 2.067 | 1% |
| absent | 1 | `mps_env` | f64 | chi=24 | 27648 | faer | packed | faer | 1262 | 2334 | 1262 | 1.849 | 2% |
| absent | 1 | `mps_env` | c32 | chi=24 | 27648 | faer | packed | faer | 2374 | 5270 | 2384 | 2.211 | 1% |
| absent | 1 | `mps_env` | c64 | chi=24 | 27648 | faer | packed | faer | 4659 | 8126 | 4658 | 1.745 | 2% |
| absent | 1 | `mps_env` | f32 | chi=32 | 65536 | faer | packed | faer | 1452 | 4328 | 1453 | 2.979 | 1% |
| absent | 1 | `mps_env` | f64 | chi=32 | 65536 | faer | packed | faer | 2776 | 5891 | 2785 | 2.115 | 2% |
| absent | 1 | `mps_env` | c32 | chi=32 | 65536 | faer | packed | faer | 5370 | 7985 | 5400 | 1.479 | 1% |
| absent | 1 | `mps_env` | c64 | chi=32 | 65536 | faer | packed | faer | 10770 | 14126 | 10750 | 1.314 | 1% |
| absent | 1 | `mps_env` | f32 | chi=48 | 221184 | faer | packed | faer | 4518 | 7284 | 4508 | 1.616 | 0% |
| absent | 1 | `mps_env` | f64 | chi=48 | 221184 | faer | packed | faer | 8966 | 11972 | 8956 | 1.337 | 2% |
| absent | 1 | `mps_env` | c32 | chi=48 | 221184 | faer | packed | faer | 17613 | 28513 | 17602 | 1.620 | 1% |
| absent | 1 | `mps_env` | c64 | chi=48 | 221184 | packed | packed | faer | 41548 | 41598 | 35145 | 1.184 | 1% |
| absent | 1 | `mps_env` | f32 | chi=64 | 524288 | faer | packed | faer | 10429 | 20989 | 10429 | 2.013 | 0% |
| absent | 1 | `mps_env` | f64 | chi=64 | 524288 | faer | packed | faer | 20769 | 28463 | 20778 | 1.370 | 1% |
| absent | 1 | `mps_env` | c32 | chi=64 | 524288 | faer | packed | faer | 41156 | 50474 | 41157 | 1.226 | 1% |
| absent | 1 | `mps_env` | c64 | chi=64 | 524288 | packed | packed | faer | 96190 | 96219 | 82354 | 1.168 | 1% |
| absent | 1 | `mps_env` | f32 | chi=96 | 1769472 | faer | packed | faer | 34564 | 43852 | 34584 | 1.268 | 0% |
| absent | 1 | `mps_env` | f64 | chi=96 | 1769472 | faer | packed | faer | 68949 | 79528 | 68958 | 1.153 | 1% |
| absent | 1 | `mps_env` | c32 | chi=96 | 1769472 | faer | packed | faer | 137767 | 153957 | 137778 | 1.117 | 0% |
| absent | 1 | `mps_env` | c64 | chi=96 | 1769472 | packed | packed | faer | 301209 | 301720 | 275515 | 1.095 | 1% |
| absent | 1 | `mps_env` | f32 | chi=128 | 4194304 | faer | packed | faer | 81722 | 108163 | 81693 | 1.324 | 0% |
| absent | 1 | `mps_env` | f64 | chi=128 | 4194304 | faer | packed | faer | 162983 | 204079 | 163014 | 1.252 | 0% |
| absent | 1 | `mps_env` | c32 | chi=128 | 4194304 | faer | packed | faer | 323991 | 357949 | 323975 | 1.105 | 0% |
| absent | 1 | `mps_env` | c64 | chi=128 | 4194304 | packed | packed | faer | 701823 | 701963 | 650997 | 1.078 | 0% |
| absent | 1 | `mps_site` | f32 | chi=16 | 8192 | faer | packed | faer | 300 | 1743 | 301 | 5.791 | 6% |
| absent | 1 | `mps_site` | f64 | chi=16 | 8192 | faer | packed | faer | 461 | 1493 | 461 | 3.239 | 5% |
| absent | 1 | `mps_site` | c32 | chi=16 | 8192 | faer | packed | faer | 791 | 2816 | 781 | 3.606 | 4% |
| absent | 1 | `mps_site` | c64 | chi=16 | 8192 | faer | packed | faer | 1493 | 2755 | 1492 | 1.847 | 1% |
| absent | 1 | `mps_site` | f32 | chi=24 | 27648 | faer | packed | faer | 892 | 2875 | 891 | 3.227 | 2% |
| absent | 1 | `mps_site` | f64 | chi=24 | 27648 | faer | packed | faer | 1282 | 2365 | 1282 | 1.845 | 2% |
| absent | 1 | `mps_site` | c32 | chi=24 | 27648 | faer | packed | faer | 2374 | 4979 | 2374 | 2.097 | 1% |
| absent | 1 | `mps_site` | c64 | chi=24 | 27648 | faer | packed | faer | 4629 | 8135 | 4639 | 1.754 | 1% |
| absent | 1 | `mps_site` | f32 | chi=32 | 65536 | faer | packed | faer | 1482 | 4178 | 1462 | 2.858 | 2% |
| absent | 1 | `mps_site` | f64 | chi=32 | 65536 | faer | packed | faer | 2855 | 6202 | 2855 | 2.172 | 0% |
| absent | 1 | `mps_site` | c32 | chi=32 | 65536 | faer | packed | faer | 5380 | 8746 | 5380 | 1.626 | 1% |
| absent | 1 | `mps_site` | c64 | chi=32 | 65536 | faer | packed | faer | 10770 | 15128 | 10810 | 1.399 | 1% |
| absent | 1 | `mps_site` | f32 | chi=48 | 221184 | faer | packed | faer | 4588 | 7524 | 4608 | 1.633 | 1% |
| absent | 1 | `mps_site` | f64 | chi=48 | 221184 | faer | packed | faer | 9227 | 12153 | 9237 | 1.316 | 1% |
| absent | 1 | `mps_site` | c32 | chi=48 | 221184 | faer | packed | faer | 17713 | 29234 | 17703 | 1.651 | 1% |
| absent | 1 | `mps_site` | c64 | chi=48 | 221184 | packed | packed | faer | 41127 | 41147 | 35375 | 1.163 | 1% |
| absent | 1 | `mps_site` | f32 | chi=64 | 524288 | faer | packed | faer | 10670 | 22142 | 10660 | 2.077 | 0% |
| absent | 1 | `mps_site` | f64 | chi=64 | 524288 | faer | packed | faer | 21489 | 29165 | 21520 | 1.355 | 1% |
| absent | 1 | `mps_site` | c32 | chi=64 | 524288 | faer | packed | faer | 41658 | 51736 | 41678 | 1.241 | 0% |
| absent | 1 | `mps_site` | c64 | chi=64 | 524288 | packed | packed | faer | 95436 | 95305 | 82684 | 1.153 | 1% |
| absent | 1 | `mps_site` | f32 | chi=96 | 1769472 | faer | packed | faer | 35095 | 45035 | 35045 | 1.285 | 1% |
| absent | 1 | `mps_site` | f64 | chi=96 | 1769472 | faer | packed | faer | 69731 | 80608 | 69699 | 1.157 | 1% |
| absent | 1 | `mps_site` | c32 | chi=96 | 1769472 | faer | packed | faer | 137533 | 156952 | 137738 | 1.139 | 0% |
| absent | 1 | `mps_site` | c64 | chi=96 | 1769472 | packed | packed | faer | 304401 | 304411 | 276217 | 1.102 | 0% |
| absent | 1 | `mps_site` | f32 | chi=128 | 4194304 | faer | packed | faer | 82061 | 112540 | 82293 | 1.368 | 1% |
| absent | 1 | `mps_site` | f64 | chi=128 | 4194304 | faer | packed | faer | 165016 | 211736 | 165450 | 1.280 | 2% |
| absent | 1 | `mps_site` | c32 | chi=128 | 4194304 | faer | packed | faer | 325679 | 379500 | 325769 | 1.165 | 1% |
| absent | 1 | `mps_site` | c64 | chi=128 | 4194304 | packed | packed | faer | 735815 | 734242 | 652191 | 1.126 | 1% |
| absent | 4 | `gemm` | f32 | n=32 | 32768 | faer | packed | faer | 752 | 2284 | 761 | 3.001 | 1% |
| absent | 4 | `gemm` | f64 | n=32 | 32768 | faer | packed | faer | 1412 | 3246 | 1412 | 2.299 | 1% |
| absent | 4 | `gemm` | c32 | n=32 | 32768 | faer | packed | faer | 2715 | 4338 | 2725 | 1.592 | 1% |
| absent | 4 | `gemm` | c64 | n=32 | 32768 | faer | packed | faer | 5300 | 7774 | 5300 | 1.467 | 0% |
| absent | 4 | `gemm` | f32 | n=64 | 262144 | faer | packed | faer | 5190 | 11101 | 5189 | 2.139 | 1% |
| absent | 4 | `gemm` | f64 | n=64 | 262144 | faer | packed | faer | 10309 | 14817 | 10309 | 1.437 | 1% |
| absent | 4 | `gemm` | c32 | n=64 | 262144 | faer | packed | faer | 15408 | 13375 | 15428 | 0.867 | 19% |
| absent | 4 | `gemm` | c64 | n=64 | 262144 | packed | packed | faer | 20719 | 19757 | 21911 | 0.902 | 14% |
| absent | 4 | `gemm` | f32 | n=128 | 2097152 | faer | packed | faer | 29265 | 19657 | 38571 | 0.510 | 87% |
| absent | 4 | `gemm` | f64 | n=128 | 2097152 | faer | packed | faer | 31959 | 42609 | 36368 | 1.172 | 20% |
| absent | 4 | `gemm` | c32 | n=128 | 2097152 | faer | packed | faer | 57366 | 53460 | 57036 | 0.937 | 3% |
| absent | 4 | `gemm` | c64 | n=128 | 2097152 | packed | packed | faer | 99266 | 99265 | 95960 | 1.034 | 1% |
| absent | 4 | `gemm` | f32 | n=256 | 16777216 | faer | packed | faer | 97992 | 141864 | 99455 | 1.426 | 43% |
| absent | 4 | `gemm` | f64 | n=256 | 16777216 | faer | packed | faer | 182028 | 201975 | 180302 | 1.120 | 4% |
| absent | 4 | `gemm` | c32 | n=256 | 16777216 | faer | packed | faer | 349184 | 349163 | 350301 | 0.997 | 2% |
| absent | 4 | `gemm` | c64 | n=256 | 16777216 | packed | packed | faer | 688870 | 689950 | 674721 | 1.023 | 1% |
| absent | 4 | `gemm_batched` | f32 | n=16 batch=16 | 4096 | faer | packed | faer | 2154 | 10990 | 2154 | 5.102 | 6% |
| absent | 4 | `gemm_batched` | f64 | n=16 batch=16 | 4096 | faer | packed | faer | 3486 | 9518 | 3486 | 2.730 | 6% |
| absent | 4 | `gemm_batched` | c32 | n=16 batch=16 | 4096 | faer | packed | faer | 9568 | 20578 | 9577 | 2.149 | 0% |
| absent | 4 | `gemm_batched` | c64 | n=16 batch=16 | 4096 | faer | packed | faer | 19366 | 19216 | 19306 | 0.995 | 2% |
| absent | 4 | `gemm_batched` | f32 | n=32 batch=16 | 32768 | faer | packed | faer | 6843 | 14968 | 6412 | 2.334 | 7% |
| absent | 4 | `gemm_batched` | f64 | n=32 batch=16 | 32768 | faer | packed | faer | 10420 | 20638 | 11451 | 1.802 | 13% |
| absent | 4 | `gemm_batched` | c32 | n=32 batch=16 | 32768 | faer | packed | faer | 14207 | 19877 | 14187 | 1.401 | 5% |
| absent | 4 | `gemm_batched` | c64 | n=32 batch=16 | 32768 | faer | packed | faer | 24746 | 34113 | 24666 | 1.383 | 2% |
| absent | 4 | `gemm_batched` | f32 | n=64 batch=16 | 262144 | faer | packed | faer | 23374 | 46868 | 24205 | 1.936 | 4% |
| absent | 4 | `gemm_batched` | f64 | n=64 batch=16 | 262144 | faer | packed | faer | 44713 | 61674 | 44522 | 1.385 | 1% |
| absent | 4 | `gemm_batched` | c32 | n=64 batch=16 | 262144 | faer | packed | faer | 250035 | 102499 | 251340 | 0.408 | 5% |
| absent | 4 | `gemm_batched` | c64 | n=64 batch=16 | 262144 | packed | packed | faer | 196555 | 197234 | 327702 | 0.602 | 3% |
| absent | 4 | `gemm_batched` | f32 | n=32 batch=64 | 32768 | faer | packed | faer | 14758 | 34774 | 14708 | 2.364 | 11% |
| absent | 4 | `gemm_batched` | f64 | n=32 batch=64 | 32768 | faer | packed | faer | 25608 | 49111 | 25838 | 1.901 | 10% |
| absent | 4 | `gemm_batched` | c32 | n=32 batch=64 | 32768 | faer | packed | faer | 45534 | 66655 | 46617 | 1.430 | 6% |
| absent | 4 | `gemm_batched` | c64 | n=32 batch=64 | 32768 | faer | packed | faer | 87854 | 122827 | 87984 | 1.396 | 8% |
| absent | 4 | `gemm_batched` | f32 | n=64 batch=64 | 262144 | faer | packed | faer | 86610 | 174883 | 86410 | 2.024 | 7% |
| absent | 4 | `gemm_batched` | f64 | n=64 batch=64 | 262144 | faer | packed | faer | 170018 | 242130 | 170178 | 1.423 | 1% |
| absent | 4 | `gemm_batched` | c32 | n=64 batch=64 | 262144 | faer | packed | faer | 1039924 | 401871 | 1042748 | 0.385 | 4% |
| absent | 4 | `gemm_batched` | c64 | n=64 batch=64 | 262144 | packed | packed | faer | 932483 | 922695 | 1406525 | 0.656 | 3% |
| absent | 4 | `mps_env` | f32 | chi=16 | 8192 | faer | packed | faer | 330 | 1393 | 321 | 4.340 | 3% |
| absent | 4 | `mps_env` | f64 | chi=16 | 8192 | faer | packed | faer | 461 | 1393 | 461 | 3.022 | 2% |
| absent | 4 | `mps_env` | c32 | chi=16 | 8192 | faer | packed | faer | 801 | 1904 | 792 | 2.404 | 2% |
| absent | 4 | `mps_env` | c64 | chi=16 | 8192 | faer | packed | faer | 1492 | 2654 | 1493 | 1.778 | 1% |
| absent | 4 | `mps_env` | f32 | chi=24 | 27648 | faer | packed | faer | 891 | 1773 | 901 | 1.968 | 1% |
| absent | 4 | `mps_env` | f64 | chi=24 | 27648 | faer | packed | faer | 1252 | 2234 | 1262 | 1.770 | 2% |
| absent | 4 | `mps_env` | c32 | chi=24 | 27648 | faer | packed | faer | 2364 | 5149 | 2374 | 2.169 | 1% |
| absent | 4 | `mps_env` | c64 | chi=24 | 27648 | faer | packed | faer | 4628 | 8055 | 4628 | 1.740 | 1% |
| absent | 4 | `mps_env` | f32 | chi=32 | 65536 | faer | packed | faer | 1443 | 4218 | 1442 | 2.925 | 1% |
| absent | 4 | `mps_env` | f64 | chi=32 | 65536 | faer | packed | faer | 2785 | 5801 | 2785 | 2.083 | 1% |
| absent | 4 | `mps_env` | c32 | chi=32 | 65536 | faer | packed | faer | 5370 | 7884 | 5370 | 1.468 | 1% |
| absent | 4 | `mps_env` | c64 | chi=32 | 65536 | faer | packed | faer | 10649 | 14006 | 10640 | 1.316 | 0% |
| absent | 4 | `mps_env` | f32 | chi=48 | 221184 | faer | packed | faer | 4509 | 7123 | 4518 | 1.577 | 0% |
| absent | 4 | `mps_env` | f64 | chi=48 | 221184 | faer | packed | faer | 8936 | 11792 | 8937 | 1.319 | 1% |
| absent | 4 | `mps_env` | c32 | chi=48 | 221184 | faer | packed | faer | 15579 | 14217 | 13275 | 1.071 | 34% |
| absent | 4 | `mps_env` | c64 | chi=48 | 221184 | packed | packed | faer | 14827 | 15609 | 25467 | 0.613 | 18% |
| absent | 4 | `mps_env` | f32 | chi=64 | 524288 | faer | packed | faer | 11782 | 12203 | 10580 | 1.153 | 40% |
| absent | 4 | `mps_env` | f64 | chi=64 | 524288 | faer | packed | faer | 20729 | 16711 | 22041 | 0.758 | 37% |
| absent | 4 | `mps_env` | c32 | chi=64 | 524288 | faer | packed | faer | 23654 | 19967 | 21460 | 0.930 | 42% |
| absent | 4 | `mps_env` | c64 | chi=64 | 524288 | packed | packed | faer | 33373 | 33142 | 32010 | 1.035 | 7% |
| absent | 4 | `mps_env` | f32 | chi=96 | 1769472 | faer | packed | faer | 33623 | 18685 | 28533 | 0.655 | 45% |
| absent | 4 | `mps_env` | f64 | chi=96 | 1769472 | faer | packed | faer | 41637 | 27181 | 40746 | 0.667 | 72% |
| absent | 4 | `mps_env` | c32 | chi=96 | 1769472 | faer | packed | faer | 80038 | 46356 | 62406 | 0.743 | 36% |
| absent | 4 | `mps_env` | c64 | chi=96 | 1769472 | packed | packed | faer | 110324 | 109395 | 78374 | 1.396 | 5% |
| absent | 4 | `mps_env` | f32 | chi=128 | 4194304 | faer | packed | faer | 32561 | 35295 | 44793 | 0.788 | 61% |
| absent | 4 | `mps_env` | f64 | chi=128 | 4194304 | faer | packed | faer | 54041 | 75670 | 53610 | 1.411 | 28% |
| absent | 4 | `mps_env` | c32 | chi=128 | 4194304 | faer | packed | faer | 99605 | 98352 | 98964 | 0.994 | 6% |
| absent | 4 | `mps_env` | c64 | chi=128 | 4194304 | packed | packed | faer | 186184 | 185566 | 180335 | 1.029 | 1% |
| absent | 4 | `mps_site` | f32 | chi=16 | 8192 | faer | packed | faer | 311 | 1563 | 311 | 5.026 | 6% |
| absent | 4 | `mps_site` | f64 | chi=16 | 8192 | faer | packed | faer | 471 | 1322 | 471 | 2.807 | 7% |
| absent | 4 | `mps_site` | c32 | chi=16 | 8192 | faer | packed | faer | 801 | 2775 | 811 | 3.422 | 4% |
| absent | 4 | `mps_site` | c64 | chi=16 | 8192 | faer | packed | faer | 1493 | 2604 | 1483 | 1.756 | 1% |
| absent | 4 | `mps_site` | f32 | chi=24 | 27648 | faer | packed | faer | 901 | 2695 | 892 | 3.021 | 2% |
| absent | 4 | `mps_site` | f64 | chi=24 | 27648 | faer | packed | faer | 1283 | 2284 | 1292 | 1.768 | 2% |
| absent | 4 | `mps_site` | c32 | chi=24 | 27648 | faer | packed | faer | 2374 | 4929 | 2374 | 2.076 | 2% |
| absent | 4 | `mps_site` | c64 | chi=24 | 27648 | faer | packed | faer | 4668 | 7984 | 4659 | 1.714 | 1% |
| absent | 4 | `mps_site` | f32 | chi=32 | 65536 | faer | packed | faer | 1472 | 4048 | 1482 | 2.731 | 2% |
| absent | 4 | `mps_site` | f64 | chi=32 | 65536 | faer | packed | faer | 2885 | 6102 | 2885 | 2.115 | 1% |
| absent | 4 | `mps_site` | c32 | chi=32 | 65536 | faer | packed | faer | 5390 | 8696 | 5380 | 1.616 | 0% |
| absent | 4 | `mps_site` | c64 | chi=32 | 65536 | faer | packed | faer | 10800 | 14958 | 10810 | 1.384 | 1% |
| absent | 4 | `mps_site` | f32 | chi=48 | 221184 | faer | packed | faer | 4609 | 7354 | 4598 | 1.599 | 1% |
| absent | 4 | `mps_site` | f64 | chi=48 | 221184 | faer | packed | faer | 9257 | 12062 | 9267 | 1.302 | 1% |
| absent | 4 | `mps_site` | c32 | chi=48 | 221184 | faer | packed | faer | 21610 | 14788 | 17823 | 0.830 | 24% |
| absent | 4 | `mps_site` | c64 | chi=48 | 221184 | packed | packed | faer | 21941 | 21800 | 39834 | 0.547 | 8% |
| absent | 4 | `mps_site` | f32 | chi=64 | 524288 | faer | packed | faer | 14497 | 13975 | 11101 | 1.259 | 29% |
| absent | 4 | `mps_site` | f64 | chi=64 | 524288 | faer | packed | faer | 21579 | 17613 | 24696 | 0.713 | 33% |
| absent | 4 | `mps_site` | c32 | chi=64 | 524288 | faer | packed | faer | 34895 | 25788 | 27391 | 0.941 | 38% |
| absent | 4 | `mps_site` | c64 | chi=64 | 524288 | packed | packed | faer | 32581 | 32921 | 43280 | 0.761 | 14% |
| absent | 4 | `mps_site` | f32 | chi=96 | 1769472 | faer | packed | faer | 23163 | 19316 | 30918 | 0.625 | 55% |
| absent | 4 | `mps_site` | f64 | chi=96 | 1769472 | faer | packed | faer | 42348 | 27612 | 73818 | 0.374 | 86% |
| absent | 4 | `mps_site` | c32 | chi=96 | 1769472 | faer | packed | faer | 98484 | 50283 | 80900 | 0.622 | 42% |
| absent | 4 | `mps_site` | c64 | chi=96 | 1769472 | packed | packed | faer | 110666 | 110465 | 100798 | 1.096 | 19% |
| absent | 4 | `mps_site` | f32 | chi=128 | 4194304 | faer | packed | faer | 90139 | 40435 | 61444 | 0.658 | 55% |
| absent | 4 | `mps_site` | f64 | chi=128 | 4194304 | faer | packed | faer | 81382 | 77555 | 70942 | 1.093 | 16% |
| absent | 4 | `mps_site` | c32 | chi=128 | 4194304 | faer | packed | faer | 108391 | 102541 | 110296 | 0.930 | 8% |
| absent | 4 | `mps_site` | c64 | chi=128 | 4194304 | packed | packed | faer | 194542 | 195374 | 195463 | 1.000 | 1% |

#### `output` (in-place accumulation)

| c_mode | threads | class | dtype | params | mnk | default | packed | faer_forced | ns default | ns packed | ns faer_forced | packed/faer_forced | spread |
|---|---|---|---|---|---:|---|---|---|---:|---:|---:|---:|---:|
| output | 1 | `gemm` | f32 | n=32 | 32768 | faer | packed | faer | 742 | 2375 | 742 | 3.201 | 1% |
| output | 1 | `gemm` | f64 | n=32 | 32768 | faer | packed | faer | 1402 | 3417 | 1402 | 2.437 | 3% |
| output | 1 | `gemm` | c32 | n=32 | 32768 | faer | packed | faer | 2715 | 8686 | 2715 | 3.199 | 14% |
| output | 1 | `gemm` | c64 | n=32 | 32768 | faer | packed | faer | 5390 | 8546 | 5380 | 1.588 | 1% |
| output | 1 | `gemm` | f32 | n=64 | 262144 | faer | packed | faer | 5190 | 11261 | 5189 | 2.170 | 1% |
| output | 1 | `gemm` | f64 | n=64 | 262144 | faer | packed | faer | 10389 | 15118 | 10380 | 1.456 | 1% |
| output | 1 | `gemm` | c32 | n=64 | 262144 | faer | packed | faer | 20628 | 29785 | 20618 | 1.445 | 1% |
| output | 1 | `gemm` | c64 | n=64 | 262144 | packed | packed | faer | 64720 | 64810 | 41146 | 1.575 | 1% |
| output | 1 | `gemm` | f32 | n=128 | 2097152 | faer | packed | faer | 40575 | 55574 | 40585 | 1.369 | 1% |
| output | 1 | `gemm` | f64 | n=128 | 2097152 | faer | packed | faer | 80991 | 102261 | 80851 | 1.265 | 1% |
| output | 1 | `gemm` | c32 | n=128 | 2097152 | faer | packed | faer | 162384 | 198541 | 162393 | 1.223 | 1% |
| output | 1 | `gemm` | c64 | n=128 | 2097152 | packed | packed | faer | 439441 | 438119 | 325519 | 1.346 | 1% |
| output | 1 | `gemm` | f32 | n=256 | 16777216 | faer | packed | faer | 321982 | 400011 | 322187 | 1.242 | 1% |
| output | 1 | `gemm` | f64 | n=256 | 16777216 | faer | packed | faer | 648482 | 709817 | 648672 | 1.094 | 2% |
| output | 1 | `gemm` | c32 | n=256 | 16777216 | faer | packed | faer | 1291633 | 1427437 | 1291834 | 1.105 | 0% |
| output | 1 | `gemm` | c64 | n=256 | 16777216 | packed | packed | faer | 2735094 | 2734880 | 2587414 | 1.057 | 1% |
| output | 1 | `gemm_batched` | f32 | n=16 batch=16 | 4096 | faer | packed | faer | 2144 | 11241 | 2154 | 5.219 | 2% |
| output | 1 | `gemm_batched` | f64 | n=16 batch=16 | 4096 | faer | packed | faer | 3587 | 9527 | 3587 | 2.656 | 1% |
| output | 1 | `gemm_batched` | c32 | n=16 batch=16 | 4096 | faer | packed | faer | 9127 | 25046 | 9137 | 2.741 | 3% |
| output | 1 | `gemm_batched` | c64 | n=16 batch=16 | 4096 | faer | packed | faer | 18164 | 21790 | 18154 | 1.200 | 1% |
| output | 1 | `gemm_batched` | f32 | n=32 batch=16 | 32768 | faer | packed | faer | 11391 | 31288 | 11411 | 2.742 | 1% |
| output | 1 | `gemm_batched` | f64 | n=32 batch=16 | 32768 | faer | packed | faer | 21981 | 47679 | 22011 | 2.166 | 3% |
| output | 1 | `gemm_batched` | c32 | n=32 batch=16 | 32768 | faer | packed | faer | 42991 | 81833 | 43151 | 1.896 | 2% |
| output | 1 | `gemm_batched` | c64 | n=32 batch=16 | 32768 | faer | packed | faer | 85859 | 130624 | 85749 | 1.523 | 2% |
| output | 1 | `gemm_batched` | f32 | n=64 batch=16 | 262144 | faer | packed | faer | 82523 | 172773 | 82603 | 2.092 | 2% |
| output | 1 | `gemm_batched` | f64 | n=64 batch=16 | 262144 | faer | packed | faer | 166621 | 238873 | 166522 | 1.434 | 1% |
| output | 1 | `gemm_batched` | c32 | n=64 batch=16 | 262144 | faer | packed | faer | 331615 | 470790 | 331705 | 1.419 | 1% |
| output | 1 | `gemm_batched` | c64 | n=64 batch=16 | 262144 | packed | packed | faer | 817006 | 817088 | 664863 | 1.229 | 1% |
| output | 1 | `gemm_batched` | f32 | n=32 batch=64 | 32768 | faer | packed | faer | 45625 | 123900 | 45696 | 2.711 | 1% |
| output | 1 | `gemm_batched` | f64 | n=32 batch=64 | 32768 | faer | packed | faer | 89358 | 192369 | 90168 | 2.133 | 3% |
| output | 1 | `gemm_batched` | c32 | n=32 batch=64 | 32768 | faer | packed | faer | 173264 | 328209 | 173214 | 1.895 | 2% |
| output | 1 | `gemm_batched` | c64 | n=32 batch=64 | 32768 | faer | packed | faer | 348526 | 526965 | 347484 | 1.517 | 1% |
| output | 1 | `gemm_batched` | f32 | n=64 batch=64 | 262144 | faer | packed | faer | 334551 | 692534 | 334004 | 2.073 | 2% |
| output | 1 | `gemm_batched` | f64 | n=64 batch=64 | 262144 | faer | packed | faer | 678488 | 1019014 | 679330 | 1.500 | 5% |
| output | 1 | `gemm_batched` | c32 | n=64 batch=64 | 262144 | faer | packed | faer | 1345355 | 1930056 | 1345365 | 1.435 | 1% |
| output | 1 | `gemm_batched` | c64 | n=64 batch=64 | 262144 | packed | packed | faer | 4073164 | 4027719 | 3081991 | 1.307 | 1% |
| output | 1 | `mps_env` | f32 | chi=16 | 8192 | faer | packed | faer | 330 | 1563 | 330 | 4.736 | 3% |
| output | 1 | `mps_env` | f64 | chi=16 | 8192 | faer | packed | faer | 471 | 1643 | 471 | 3.488 | 4% |
| output | 1 | `mps_env` | c32 | chi=16 | 8192 | faer | packed | faer | 811 | 2375 | 811 | 2.928 | 3% |
| output | 1 | `mps_env` | c64 | chi=16 | 8192 | faer | packed | faer | 1503 | 3136 | 1512 | 2.074 | 3% |
| output | 1 | `mps_env` | f32 | chi=24 | 27648 | faer | packed | faer | 1032 | 2063 | 1022 | 2.019 | 2% |
| output | 1 | `mps_env` | f64 | chi=24 | 27648 | faer | packed | faer | 1262 | 2414 | 1272 | 1.898 | 2% |
| output | 1 | `mps_env` | c32 | chi=24 | 27648 | faer | packed | faer | 2384 | 6072 | 2374 | 2.558 | 2% |
| output | 1 | `mps_env` | c64 | chi=24 | 27648 | faer | packed | faer | 4789 | 8846 | 4738 | 1.867 | 1% |
| output | 1 | `mps_env` | f32 | chi=32 | 65536 | faer | packed | faer | 1452 | 4719 | 1452 | 3.250 | 1% |
| output | 1 | `mps_env` | f64 | chi=32 | 65536 | faer | packed | faer | 2785 | 6092 | 2785 | 2.187 | 4% |
| output | 1 | `mps_env` | c32 | chi=32 | 65536 | faer | packed | faer | 5390 | 10310 | 5391 | 1.912 | 2% |
| output | 1 | `mps_env` | c64 | chi=32 | 65536 | faer | packed | faer | 10670 | 15569 | 10690 | 1.456 | 0% |
| output | 1 | `mps_env` | f32 | chi=48 | 221184 | faer | packed | faer | 4528 | 7404 | 4518 | 1.639 | 0% |
| output | 1 | `mps_env` | f64 | chi=48 | 221184 | faer | packed | faer | 8927 | 12173 | 8927 | 1.364 | 1% |
| output | 1 | `mps_env` | c32 | chi=48 | 221184 | faer | packed | faer | 17653 | 33452 | 17663 | 1.894 | 0% |
| output | 1 | `mps_env` | c64 | chi=48 | 221184 | packed | packed | faer | 44502 | 44492 | 35617 | 1.249 | 1% |
| output | 1 | `mps_env` | f32 | chi=64 | 524288 | faer | packed | faer | 10360 | 21239 | 10369 | 2.048 | 1% |
| output | 1 | `mps_env` | f64 | chi=64 | 524288 | faer | packed | faer | 21019 | 29435 | 20999 | 1.402 | 1% |
| output | 1 | `mps_env` | c32 | chi=64 | 524288 | faer | packed | faer | 41658 | 59842 | 41658 | 1.437 | 0% |
| output | 1 | `mps_env` | c64 | chi=64 | 524288 | packed | packed | faer | 101169 | 101199 | 82864 | 1.221 | 1% |
| output | 1 | `mps_env` | f32 | chi=96 | 1769472 | faer | packed | faer | 34594 | 44293 | 34605 | 1.280 | 0% |
| output | 1 | `mps_env` | f64 | chi=96 | 1769472 | faer | packed | faer | 69139 | 80219 | 69198 | 1.159 | 1% |
| output | 1 | `mps_env` | c32 | chi=96 | 1769472 | faer | packed | faer | 138449 | 174987 | 138507 | 1.263 | 0% |
| output | 1 | `mps_env` | c64 | chi=96 | 1769472 | packed | packed | faer | 312881 | 312470 | 277649 | 1.125 | 0% |
| output | 1 | `mps_env` | f32 | chi=128 | 4194304 | faer | packed | faer | 81670 | 109435 | 81763 | 1.338 | 1% |
| output | 1 | `mps_env` | f64 | chi=128 | 4194304 | faer | packed | faer | 163521 | 205217 | 163702 | 1.254 | 1% |
| output | 1 | `mps_env` | c32 | chi=128 | 4194304 | faer | packed | faer | 326761 | 395434 | 325949 | 1.213 | 1% |
| output | 1 | `mps_env` | c64 | chi=128 | 4194304 | packed | packed | faer | 723443 | 724204 | 653193 | 1.109 | 1% |
| output | 1 | `mps_site` | f32 | chi=16 | 8192 | faer | packed | faer | 310 | 1763 | 311 | 5.669 | 6% |
| output | 1 | `mps_site` | f64 | chi=16 | 8192 | faer | packed | faer | 471 | 1563 | 471 | 3.318 | 3% |
| output | 1 | `mps_site` | c32 | chi=16 | 8192 | faer | packed | faer | 801 | 3156 | 791 | 3.990 | 10% |
| output | 1 | `mps_site` | c64 | chi=16 | 8192 | faer | packed | faer | 1493 | 2925 | 1492 | 1.960 | 2% |
| output | 1 | `mps_site` | f32 | chi=24 | 27648 | faer | packed | faer | 951 | 2885 | 951 | 3.034 | 3% |
| output | 1 | `mps_site` | f64 | chi=24 | 27648 | faer | packed | faer | 1292 | 2435 | 1292 | 1.885 | 2% |
| output | 1 | `mps_site` | c32 | chi=24 | 27648 | faer | packed | faer | 2374 | 5620 | 2374 | 2.367 | 2% |
| output | 1 | `mps_site` | c64 | chi=24 | 27648 | faer | packed | faer | 4709 | 8485 | 4718 | 1.798 | 1% |
| output | 1 | `mps_site` | f32 | chi=32 | 65536 | faer | packed | faer | 1482 | 4198 | 1473 | 2.850 | 1% |
| output | 1 | `mps_site` | f64 | chi=32 | 65536 | faer | packed | faer | 2885 | 6351 | 2876 | 2.208 | 1% |
| output | 1 | `mps_site` | c32 | chi=32 | 65536 | faer | packed | faer | 5400 | 9978 | 5400 | 1.848 | 1% |
| output | 1 | `mps_site` | c64 | chi=32 | 65536 | faer | packed | faer | 10860 | 15759 | 10870 | 1.450 | 0% |
| output | 1 | `mps_site` | f32 | chi=48 | 221184 | faer | packed | faer | 4599 | 7534 | 4588 | 1.642 | 1% |
| output | 1 | `mps_site` | f64 | chi=48 | 221184 | faer | packed | faer | 9247 | 12322 | 9237 | 1.334 | 1% |
| output | 1 | `mps_site` | c32 | chi=48 | 221184 | faer | packed | faer | 17833 | 31789 | 17852 | 1.781 | 1% |
| output | 1 | `mps_site` | c64 | chi=48 | 221184 | packed | packed | faer | 42768 | 42769 | 35776 | 1.195 | 0% |
| output | 1 | `mps_site` | f32 | chi=64 | 524288 | faer | packed | faer | 10709 | 22292 | 10690 | 2.085 | 1% |
| output | 1 | `mps_site` | f64 | chi=64 | 524288 | faer | packed | faer | 21550 | 29384 | 21530 | 1.365 | 1% |
| output | 1 | `mps_site` | c32 | chi=64 | 524288 | faer | packed | faer | 41827 | 56275 | 41778 | 1.347 | 0% |
| output | 1 | `mps_site` | c64 | chi=64 | 524288 | packed | packed | faer | 98083 | 98252 | 83205 | 1.181 | 1% |
| output | 1 | `mps_site` | f32 | chi=96 | 1769472 | faer | packed | faer | 35025 | 45184 | 35035 | 1.290 | 0% |
| output | 1 | `mps_site` | f64 | chi=96 | 1769472 | faer | packed | faer | 69710 | 80861 | 69669 | 1.161 | 0% |
| output | 1 | `mps_site` | c32 | chi=96 | 1769472 | faer | packed | faer | 137984 | 167278 | 137834 | 1.214 | 0% |
| output | 1 | `mps_site` | c64 | chi=96 | 1769472 | packed | packed | faer | 310971 | 309700 | 276487 | 1.120 | 1% |
| output | 1 | `mps_site` | f32 | chi=128 | 4194304 | faer | packed | faer | 81932 | 112769 | 82023 | 1.375 | 0% |
| output | 1 | `mps_site` | f64 | chi=128 | 4194304 | faer | packed | faer | 165177 | 211034 | 165176 | 1.278 | 2% |
| output | 1 | `mps_site` | c32 | chi=128 | 4194304 | faer | packed | faer | 325695 | 397903 | 325525 | 1.222 | 1% |
| output | 1 | `mps_site` | c64 | chi=128 | 4194304 | packed | packed | faer | 744202 | 743361 | 654553 | 1.136 | 2% |
| output | 4 | `gemm` | f32 | n=32 | 32768 | faer | packed | faer | 751 | 2334 | 751 | 3.108 | 1% |
| output | 4 | `gemm` | f64 | n=32 | 32768 | faer | packed | faer | 1412 | 3366 | 1412 | 2.384 | 1% |
| output | 4 | `gemm` | c32 | n=32 | 32768 | faer | packed | faer | 2725 | 5560 | 2715 | 2.048 | 1% |
| output | 4 | `gemm` | c64 | n=32 | 32768 | faer | packed | faer | 5390 | 8446 | 5390 | 1.567 | 1% |
| output | 4 | `gemm` | f32 | n=64 | 262144 | faer | packed | faer | 5199 | 11291 | 5200 | 2.171 | 1% |
| output | 4 | `gemm` | f64 | n=64 | 262144 | faer | packed | faer | 10279 | 15058 | 10279 | 1.465 | 1% |
| output | 4 | `gemm` | c32 | n=64 | 262144 | faer | packed | faer | 14958 | 15660 | 16661 | 0.940 | 14% |
| output | 4 | `gemm` | c64 | n=64 | 262144 | packed | packed | faer | 21570 | 21370 | 22271 | 0.960 | 23% |
| output | 4 | `gemm` | f32 | n=128 | 2097152 | faer | packed | faer | 26449 | 20778 | 45455 | 0.457 | 43% |
| output | 4 | `gemm` | f64 | n=128 | 2097152 | faer | packed | faer | 33042 | 42318 | 32851 | 1.288 | 52% |
| output | 4 | `gemm` | c32 | n=128 | 2097152 | faer | packed | faer | 58329 | 62716 | 59370 | 1.056 | 4% |
| output | 4 | `gemm` | c64 | n=128 | 2097152 | packed | packed | faer | 106236 | 105114 | 97920 | 1.073 | 3% |
| output | 4 | `gemm` | f32 | n=256 | 16777216 | faer | packed | faer | 104533 | 142376 | 111397 | 1.278 | 18% |
| output | 4 | `gemm` | f64 | n=256 | 16777216 | faer | packed | faer | 184422 | 205109 | 180846 | 1.134 | 6% |
| output | 4 | `gemm` | c32 | n=256 | 16777216 | faer | packed | faer | 353878 | 378337 | 354474 | 1.067 | 2% |
| output | 4 | `gemm` | c64 | n=256 | 16777216 | packed | packed | faer | 708795 | 711059 | 676367 | 1.051 | 2% |
| output | 4 | `gemm_batched` | f32 | n=16 batch=16 | 4096 | faer | packed | faer | 2124 | 11151 | 2114 | 5.275 | 5% |
| output | 4 | `gemm_batched` | f64 | n=16 batch=16 | 4096 | faer | packed | faer | 3556 | 9267 | 3546 | 2.613 | 10% |
| output | 4 | `gemm_batched` | c32 | n=16 batch=16 | 4096 | faer | packed | faer | 9116 | 24866 | 9117 | 2.727 | 1% |
| output | 4 | `gemm_batched` | c64 | n=16 batch=16 | 4096 | faer | packed | faer | 18144 | 21680 | 18143 | 1.195 | 1% |
| output | 4 | `gemm_batched` | f32 | n=32 batch=16 | 32768 | faer | packed | faer | 6763 | 15178 | 6762 | 2.245 | 3% |
| output | 4 | `gemm_batched` | f64 | n=32 batch=16 | 32768 | faer | packed | faer | 11662 | 21289 | 11482 | 1.854 | 2% |
| output | 4 | `gemm_batched` | c32 | n=32 batch=16 | 32768 | faer | packed | faer | 14216 | 23885 | 14297 | 1.671 | 4% |
| output | 4 | `gemm_batched` | c64 | n=32 batch=16 | 32768 | faer | packed | faer | 25046 | 36628 | 24255 | 1.510 | 3% |
| output | 4 | `gemm_batched` | f32 | n=64 batch=16 | 262144 | faer | packed | faer | 24205 | 47239 | 24216 | 1.951 | 4% |
| output | 4 | `gemm_batched` | f64 | n=64 batch=16 | 262144 | faer | packed | faer | 44703 | 63297 | 45093 | 1.404 | 9% |
| output | 4 | `gemm_batched` | c32 | n=64 batch=16 | 262144 | faer | packed | faer | 259443 | 121434 | 255407 | 0.475 | 8% |
| output | 4 | `gemm_batched` | c64 | n=64 batch=16 | 262144 | packed | packed | faer | 207133 | 207405 | 332000 | 0.625 | 4% |
| output | 4 | `gemm_batched` | f32 | n=32 batch=64 | 32768 | faer | packed | faer | 14717 | 35155 | 14938 | 2.353 | 91% |
| output | 4 | `gemm_batched` | f64 | n=32 batch=64 | 32768 | faer | packed | faer | 25497 | 51536 | 25607 | 2.013 | 4% |
| output | 4 | `gemm_batched` | c32 | n=32 batch=64 | 32768 | faer | packed | faer | 46637 | 85189 | 46637 | 1.827 | 1% |
| output | 4 | `gemm_batched` | c64 | n=32 batch=64 | 32768 | faer | packed | faer | 90026 | 134970 | 89906 | 1.501 | 0% |
| output | 4 | `gemm_batched` | f32 | n=64 batch=64 | 262144 | faer | packed | faer | 87003 | 175839 | 85899 | 2.047 | 1% |
| output | 4 | `gemm_batched` | f64 | n=64 batch=64 | 262144 | faer | packed | faer | 173079 | 248412 | 173130 | 1.435 | 2% |
| output | 4 | `gemm_batched` | c32 | n=64 batch=64 | 262144 | faer | packed | faer | 1033091 | 479211 | 1032227 | 0.464 | 2% |
| output | 4 | `gemm_batched` | c64 | n=64 batch=64 | 262144 | packed | packed | faer | 1013303 | 1000150 | 1567584 | 0.638 | 5% |
| output | 4 | `mps_env` | f32 | chi=16 | 8192 | faer | packed | faer | 330 | 1453 | 330 | 4.403 | 3% |
| output | 4 | `mps_env` | f64 | chi=16 | 8192 | faer | packed | faer | 471 | 1522 | 470 | 3.238 | 3% |
| output | 4 | `mps_env` | c32 | chi=16 | 8192 | faer | packed | faer | 811 | 2294 | 802 | 2.860 | 2% |
| output | 4 | `mps_env` | c64 | chi=16 | 8192 | faer | packed | faer | 1502 | 3015 | 1502 | 2.007 | 1% |
| output | 4 | `mps_env` | f32 | chi=24 | 27648 | faer | packed | faer | 1022 | 1934 | 1021 | 1.894 | 3% |
| output | 4 | `mps_env` | f64 | chi=24 | 27648 | faer | packed | faer | 1262 | 2305 | 1262 | 1.826 | 1% |
| output | 4 | `mps_env` | c32 | chi=24 | 27648 | faer | packed | faer | 2374 | 5941 | 2374 | 2.503 | 1% |
| output | 4 | `mps_env` | c64 | chi=24 | 27648 | faer | packed | faer | 4749 | 8727 | 4749 | 1.838 | 2% |
| output | 4 | `mps_env` | f32 | chi=32 | 65536 | faer | packed | faer | 1452 | 4538 | 1453 | 3.123 | 1% |
| output | 4 | `mps_env` | f64 | chi=32 | 65536 | faer | packed | faer | 2786 | 6001 | 2785 | 2.155 | 0% |
| output | 4 | `mps_env` | c32 | chi=32 | 65536 | faer | packed | faer | 5400 | 10189 | 5400 | 1.887 | 1% |
| output | 4 | `mps_env` | c64 | chi=32 | 65536 | faer | packed | faer | 10820 | 15359 | 10820 | 1.420 | 1% |
| output | 4 | `mps_env` | f32 | chi=48 | 221184 | faer | packed | faer | 4528 | 7273 | 4519 | 1.609 | 1% |
| output | 4 | `mps_env` | f64 | chi=48 | 221184 | faer | packed | faer | 8947 | 12062 | 8926 | 1.351 | 1% |
| output | 4 | `mps_env` | c32 | chi=48 | 221184 | faer | packed | faer | 12513 | 17212 | 14006 | 1.229 | 14% |
| output | 4 | `mps_env` | c64 | chi=48 | 221184 | packed | packed | faer | 16501 | 16421 | 24816 | 0.662 | 23% |
| output | 4 | `mps_env` | f32 | chi=64 | 524288 | faer | packed | faer | 13625 | 12343 | 13595 | 0.908 | 21% |
| output | 4 | `mps_env` | f64 | chi=64 | 524288 | faer | packed | faer | 21279 | 17162 | 17142 | 1.001 | 30% |
| output | 4 | `mps_env` | c32 | chi=64 | 524288 | faer | packed | faer | 23824 | 23614 | 19707 | 1.198 | 62% |
| output | 4 | `mps_env` | c64 | chi=64 | 524288 | packed | packed | faer | 36788 | 35837 | 32099 | 1.116 | 5% |
| output | 4 | `mps_env` | f32 | chi=96 | 1769472 | faer | packed | faer | 30708 | 18575 | 19647 | 0.945 | 58% |
| output | 4 | `mps_env` | f64 | chi=96 | 1769472 | faer | packed | faer | 41197 | 28984 | 53419 | 0.543 | 53% |
| output | 4 | `mps_env` | c32 | chi=96 | 1769472 | faer | packed | faer | 57016 | 51836 | 63388 | 0.818 | 17% |
| output | 4 | `mps_env` | c64 | chi=96 | 1769472 | packed | packed | faer | 119292 | 119623 | 77184 | 1.550 | 8% |
| output | 4 | `mps_env` | f32 | chi=128 | 4194304 | faer | packed | faer | 36579 | 35656 | 43321 | 0.823 | 71% |
| output | 4 | `mps_env` | f64 | chi=128 | 4194304 | faer | packed | faer | 57898 | 76963 | 54081 | 1.423 | 13% |
| output | 4 | `mps_env` | c32 | chi=128 | 4194304 | faer | packed | faer | 102479 | 113922 | 99254 | 1.148 | 15% |
| output | 4 | `mps_env` | c64 | chi=128 | 4194304 | packed | packed | faer | 197035 | 197789 | 183771 | 1.076 | 2% |
| output | 4 | `mps_site` | f32 | chi=16 | 8192 | faer | packed | faer | 330 | 1593 | 320 | 4.978 | 7% |
| output | 4 | `mps_site` | f64 | chi=16 | 8192 | faer | packed | faer | 471 | 1382 | 471 | 2.934 | 4% |
| output | 4 | `mps_site` | c32 | chi=16 | 8192 | faer | packed | faer | 811 | 3065 | 801 | 3.826 | 2% |
| output | 4 | `mps_site` | c64 | chi=16 | 8192 | faer | packed | faer | 1502 | 2785 | 1493 | 1.865 | 1% |
| output | 4 | `mps_site` | f32 | chi=24 | 27648 | faer | packed | faer | 961 | 2745 | 962 | 2.853 | 2% |
| output | 4 | `mps_site` | f64 | chi=24 | 27648 | faer | packed | faer | 1292 | 2345 | 1292 | 1.815 | 2% |
| output | 4 | `mps_site` | c32 | chi=24 | 27648 | faer | packed | faer | 2384 | 5480 | 2384 | 2.299 | 0% |
| output | 4 | `mps_site` | c64 | chi=24 | 27648 | faer | packed | faer | 4709 | 8385 | 4718 | 1.777 | 1% |
| output | 4 | `mps_site` | f32 | chi=32 | 65536 | faer | packed | faer | 1483 | 4118 | 1482 | 2.779 | 2% |
| output | 4 | `mps_site` | f64 | chi=32 | 65536 | faer | packed | faer | 2886 | 6191 | 2895 | 2.139 | 1% |
| output | 4 | `mps_site` | c32 | chi=32 | 65536 | faer | packed | faer | 5400 | 9888 | 5400 | 1.831 | 1% |
| output | 4 | `mps_site` | c64 | chi=32 | 65536 | faer | packed | faer | 10880 | 15589 | 10870 | 1.434 | 1% |
| output | 4 | `mps_site` | f32 | chi=48 | 221184 | faer | packed | faer | 4599 | 7444 | 4608 | 1.615 | 1% |
| output | 4 | `mps_site` | f64 | chi=48 | 221184 | faer | packed | faer | 9227 | 12213 | 9247 | 1.321 | 1% |
| output | 4 | `mps_site` | c32 | chi=48 | 221184 | faer | packed | faer | 18795 | 15589 | 20979 | 0.743 | 21% |
| output | 4 | `mps_site` | c64 | chi=48 | 221184 | packed | packed | faer | 22040 | 22111 | 34474 | 0.641 | 19% |
| output | 4 | `mps_site` | f32 | chi=64 | 524288 | faer | packed | faer | 12844 | 13716 | 13055 | 1.051 | 17% |
| output | 4 | `mps_site` | f64 | chi=64 | 524288 | faer | packed | faer | 23764 | 18434 | 20798 | 0.886 | 31% |
| output | 4 | `mps_site` | c32 | chi=64 | 524288 | faer | packed | faer | 31498 | 23784 | 33442 | 0.711 | 27% |
| output | 4 | `mps_site` | c64 | chi=64 | 524288 | packed | packed | faer | 34193 | 34504 | 40936 | 0.843 | 16% |
| output | 4 | `mps_site` | f32 | chi=96 | 1769472 | faer | packed | faer | 34594 | 19236 | 40987 | 0.469 | 60% |
| output | 4 | `mps_site` | f64 | chi=96 | 1769472 | faer | packed | faer | 73747 | 29244 | 76572 | 0.382 | 50% |
| output | 4 | `mps_site` | c32 | chi=96 | 1769472 | faer | packed | faer | 64179 | 53049 | 64960 | 0.817 | 54% |
| output | 4 | `mps_site` | c64 | chi=96 | 1769472 | packed | packed | faer | 115885 | 115566 | 104053 | 1.111 | 19% |
| output | 4 | `mps_site` | f32 | chi=128 | 4194304 | faer | packed | faer | 45915 | 40395 | 42850 | 0.943 | 69% |
| output | 4 | `mps_site` | f64 | chi=128 | 4194304 | faer | packed | faer | 60022 | 78366 | 59632 | 1.314 | 34% |
| output | 4 | `mps_site` | c32 | chi=128 | 4194304 | faer | packed | faer | 118500 | 111195 | 108852 | 1.022 | 11% |
| output | 4 | `mps_site` | c64 | chi=128 | 4194304 | packed | packed | faer | 200973 | 200504 | 196737 | 1.019 | 8% |

Across the three sessions the third run's rows repeat within a median 0.5% /
p90 2.0% / max 21.8% spread at 1T and a median 1.4% / p90 25.6% / max 86.8% at
4T in the `absent` mode, and a median 0.6% / p90 2.2% / max 13.6% at 1T and a
median 1.6% / p90 19.5% / max 91.0% at 4T in the `output` mode (the larger of
the three arms' spreads per row). The `CHECK` lines in
`results/triarm-session{1,2,3}.txt` (368 per session) show every arm matching
its reference: the worst `rel` over all cases and sessions is 1.13e-6 for
`absent` and 1.11e-6 for `output`, both on the `f32`/`c32` rows (tol 1e-5);
`default` and `packed` agree to at most 4.6e-7, and `faer_forced` matches the
same oracle. No case failed and no problem was refused.

## What this does not cover

- One host, one `rustc`, two library revisions, one session order. This is a
  laptop APU under the `powersave` governor: no fixed frequency and no isolated
  cores, which is where most of the 4T spread comes from. The first run sits on
  a different library revision from the second and third (`results/manifest.txt`),
  so a ratio read across the first and the later tables mixes that change; the
  second and third run share the library revision, and each run's own rows are
  same-revision and directly comparable.
- Two C modes are measured: `absent` (overwrite) and `output` (in-place
  accumulation with `CSpec::Output(Op::Identity)` and `beta = 1`). The third
  mode, `CSpec::Separate` (a separately described `C`), is not measured.
- `output` is `beta = 1`, `alpha = 1` and `Op::Identity` only; other `beta`
  values, `op_C` conjugation, and `alpha != 1` are not measured.
- In the `output` mode the priming and timed calls accumulate into the same `D`
  without a reset between calls (as a sequence of MPS steps does). The per-call
  cost is the same, but `D` grows across the priming window; a reset outside the
  clock would measure the same call.
- Whole-shape timings only: no split between packing, the micro-kernel and
  write-back inside either engine, and no counter-based attribution.
- **For `c64` above the provisional limit the `default` and `packed` arms are
  the same engine.** The default route already sends `c64` to the packed driver
  at `mnk = 262144` (`FaerLimit::default()` bounds `c64` at `2^17`), so there
  the `default` column reads `packed` and its ratio to `packed` is ~1.00. The
  second run can see the c64 faer/packed crossover only below that volume; the
  third run's `faer_forced` arm reaches faer at every volume and is what to read
  for c64 above it.
- Routing actually exercised: in every C mode the default arm chose `faer` with
  reason `fused` for every `f32`/`f64`/`c32` case and for `c64` below the limit,
  and `packed` with reason `above_faer_limit` above it. The `faer_forced` arm
  chose `faer`/`fused` for all 368 case/dtype/thread/mode rows of every
  session. `NotFusable`, `AllBatch` and the elementwise route never appeared
  (every case is a copy-free fusion).
- Nothing here chooses a `FaerLimit`; the per-dtype values and the
  decision-log row are the main agent's to write from these numbers.
