# Project guidance

Before acting, read the shared tensor4all agent rules, starting from
`https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md`
(fallback: `../tensor4all-agent-rules/rules/index.md`); load only the files
relevant to the task and do not vendor them here. Then read
`REPOSITORY_RULES.md`. Read `PERFORMANCE_TIPS.md` in full before implementing
or reviewing kernels, planning, caches, execution/threading code or
benchmarks, and before creating a PR that touches them. Work inside
`benchmarks/` also follows that directory's own `AGENTS.md`; where it conflicts with this file, this file and
`PERFORMANCE_TIPS.md` win for cross-cutting execution and benchmark rules.

This is a research repository, not a production tensor-algebra library. Read `README.md`, `docs/research-map.md`, `docs/experiments.md`, and `docs/provenance.md` before adding an experiment.

- Keep BLIS-style GEMM, TBLIS-style contraction and executor/FFI choices provisional until comparable measurements support a decision.
- Before timing a numerical path, check known values and reconstruction or residuals. Record the provider version, build flags, CPU, layout, dtype, shape, batch size, thread count, and timed boundary with results.
- Cite the original paper and any implementation consulted in the source file when code is written. Clearly label a port or close translation, preserve upstream notices, and review imported tests file by file.
- Keep host-controlled threading explicit. Do not introduce an ambient global pool in a library experiment without identifying it as the behavior under test.
- Do not publish a package or present benchmark claims from unrecorded measurements.

## Layout

- `crates/`: `tcontract-exec`, `tcontract-kernel`, `tcontract`, `tcontract-testkit`
  and `tcontract-capi` (libtcontract_capi). Much of the kernel, driver and C ABI code is
  tensorprimitives-rs by Lukas Devos, moved here with history and authorship
  (`docs/provenance.md`); the old tree is archived in
  `docs/archive/tensorprimitives/`.
- `benchmarks/`: package `tcontract-bench` (started as an import of
  strided-rs-benchmark-suite). Every new operation adds rows here at 1T and 4T.
- `experiments/`: standalone measurement probes, excluded from the workspace.
- `.agents/skills/` (canonical, read by Codex and pi), mirrored byte for byte
  in `.claude/skills/`, with OpenCode commands in `.opencode/commands/`;
  `python3 scripts/check-agent-skills.py` checks the mirrors. Run benchmarks
  with the `tcontract-benchmark` skill.
- strided-rs is an external git dependency pinned to a published release with
  all four crates at one rev (`Cargo.toml` carries the exact requirement);
  consumers coordinate that source to share view types. strided changes and
  strided benchmarks go to tensor4all/strided-rs and
  tensor4all/strided-rs-benchmark-suite.

## Build

- The build job count depends on the host: set `CARGO_BUILD_JOBS` (16 on the
  shared 64-core EPYC workstation); scripts and CI do not hardcode `-j`.
- Local gate before a PR: `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo test --workspace`, `cargo test -p tcontract-exec` (without the `strided`
  feature the workspace unifies on), `cargo test -p tcontract --release`,
  `cargo check --workspace --all-targets --target aarch64-apple-darwin` and a
  build on the MSRV (`cargo +1.89.0 build --workspace --all-targets`). Build
  `cargo build -p tcontract-capi` before the workspace tests (the C ABI test links
  the built `libtcontract_capi.so`). CI uses stable clippy, which may be newer than a
  local toolchain.
- Before a PR, also check README and `docs/` against the implementation
  (REPOSITORY_RULES.md, Public Surface Drift): diagrams and dependency tables
  against `cargo tree`, names, features, commands and status against the code.
