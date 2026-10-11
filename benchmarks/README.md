# tcontract-bench

Benchmarks for the tcontract stack. The package started as an import of
[strided-rs-benchmark-suite](https://github.com/tensor4all/strided-rs-benchmark-suite)
(upstream `0550611`, with history); the strided-rs kernel benchmarks moved
back there when strided-rs became an external dependency again, and this
package now holds only the tcontract benchmarks.

| Binary | Page |
| --- | --- |
| `exec_entry` | [exec_entry](benchmarks/tcontract/exec_entry/README.md) |
| `tcbench` | [49-case TCCG comparison](benchmarks/tcbench/README.md): `run`, `verify`, `info`, `--threads`, `--stress`; optional `blas` baseline |
| `lukbench` | [per-shape corpus of tcontract-rs#61](benchmarks/lukbench/README.md): the same `run`/`verify`/`info` shape over fixed extents; optional `tblis` baseline |
| `contract` | [contract](benchmarks/tcontract/contract/README.md) |
| `capi_rust` | [C ABI comparison](c/README.md) |

## Build and run

The package is a member of the root workspace, so binaries land in the
repository's `target/`:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release -p tcontract-bench --bins   # jobs from CARGO_BUILD_JOBS
cpus=$(python3 scripts/idle_cpus.py pick 4)          # idle CPUs of one L3 domain
scripts/pinned.sh "${cpus%%,*}" -- ../target/release/contract --threads 1
scripts/pinned.sh "$cpus"       -- ../target/release/contract --threads 4
```

The full procedure is the `tcontract-benchmark` skill
(`../.agents/skills/tcontract-benchmark/SKILL.md`).

## This package and the campaign repository

This package is the **library-level harness**: the binaries, their per-operation
rows at 1T and 4T, the protocol scripts under `scripts/`, and the skill. It stays
here because the harness must be built out of the same checkout as the library it
measures, so that a result's recorded commit describes both.

The **campaign** — which commits were measured on which machines, the run
manifests, the published reports and their staleness — lives in
[tensor4all/tcontract-benchmark](https://github.com/tensor4all/tcontract-benchmark).
Its `scripts/build_for_tcontract_rev.sh` builds `tcbench` from a pinned checkout of
this repository with `--manifest-path benchmarks/Cargo.toml`, keyed by revision.
Publish campaign results there, not here.

`benchmarks/benchmarks/tcbench/results/` is **historical**: it holds the raw runs
cited as evidence by the PRs that produced them. It is frozen — do not add to it,
and do not treat the newest file in it as the current state of anything.

## Rules

Measure every tensor-sized case at 1 and 4 threads in the same run, verify the
effective thread count at startup, pin with `taskset` inside one L3 domain,
never run two benchmarks at once, and record the tcontract-rs commit, CPU, core
set and profile beside every published table. The fixed 1T/4T/8T/12T
three-provider procedure is in the skill; its explicitly requested 12T arm
may span L3 domains and is labelled separately (no SMT). See the root
[`PERFORMANCE_TIPS.md`](../PERFORMANCE_TIPS.md) and [`AGENTS.md`](AGENTS.md).
