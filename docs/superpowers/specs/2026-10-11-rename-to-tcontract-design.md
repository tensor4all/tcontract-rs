# Rename tprims to tcontract — design (issue #105)

Short design for the mechanical rename of the tensor4all/tprims-rs repository
and its public surface to `tcontract` (tensor contraction), per issue #105.
Independent pre-review (gpt-6.1-sol) corrections are folded in. The strided-rs
name and the standard TAPP interface names keep their names.

## Surface mapping (all "Today"→"After")

| Today | After | Kind |
| --- | --- | --- |
| repo `tensor4all/tprims-rs` | `tensor4all/tcontract-rs` | GitHub rename |
| crate `tprims-contract` | `tcontract` (core contraction library) | crate rename |
| crates `tprims-exec`, `tprims-kernel`, `tprims-capi`, `tprims-testkit` | `tcontract-exec`, `tcontract-kernel`, `tcontract-capi`, `tcontract-testkit` | crate rename |
| package `tprims-bench` | `tcontract-bench` | package rename |
| dirs `crates/tprims-*` → `crates/tcontract-*`; core dir `crates/tprims-contract` → `crates/tcontract` | | `git mv` |
| library `libtprims` → `libtcontract` (`.so`, `.dylib`, `.dll`, `.dll.a`) | | C lib rename |
| own headers `crates/tprims-capi/include/tprims/{tprims,core,tapp_ext}.h` → `include/tcontract/{tcontract,core,tapp_ext}.h`; guards `TPRIMS_CORE_H` → `TCONTRACT_CORE_H` | | header rename |
| C symbols/types: `tprims_abi_version`, `tprims_status`, `tprims_tensor`, `tprims_tensor_borrow_raw/versioned`, `tprims_tapp_executor_create_rayon/get_threads/set_budget`, `tprims_last_error`, `tprims_rayon_opts`, `tprims_has_part` | same names with `tcontract_` prefix | C rename |
| C macros/consts: `TPRIMS_ABI_VERSION` (100), `TPRIMS_OK`, `TPRIMS_ERR_*` (values 1-15), `TPRIMS_BUSY` + Rust `tprims_contract_status`/`abi` constants | `TCONTRACT_*`, numeric values preserved | macro/const rename |
| Rust types: `TprimsBackend` | `TcontractBackend` | type rename |
| Rust imports `tprims_contract::…`, `tprims_exec::…`, `tprims_kernel::…`, `tprims_testkit::…` | `tcontract::…`, `tcontract_exec::…`, … | import rename |
| `TAPP_implementation_name` string `"tprims-rs: tprims-contract (…)"` | `"tcontract-rs: tcontract (…)"`; function name unchanged | string |
| active env knob `TPRIMS_PHASE` (phase-timing) | `TCONTRACT_PHASE` | env rename |
| pkg-config `tprims.pc` (`Name: tprims`, `-ltprims`), CMake/Corrosion target `tprims`, `pkg_check_modules … tprims` | `tcontract.pc`, `-ltcontract`, target `tcontract` | packaging rename |
| bench suite dir `benchmarks/benchmarks/tprims/` | `benchmarks/benchmarks/tcontract/` | `git mv` |
| skill `.agents/skills/tprims-benchmark/` + `.claude/skills/` mirror + `.opencode/commands/tprims-benchmark.md` | `tcontract-benchmark` (byte-identical mirror) | skill rename |
| fixture `headers_tprims_first.c` | `headers_tcontract_first.c` | fixture rename |

**Preserved**: upstream vendored TAPP headers (`include/tapp/`, `tapp.h`),
`TAPP_*` exported names incl. `TAPP_implementation_name/version` function
names, `tcbench` harness name, status **numbers**, layouts, signatures, and
the `strided-*` crates.

**Naming rationale**: `tprims-contract` → `tcontract` (not
`tcontract-contract`): the crate IS the contraction library; a
`tcontract-contract` name is redundant. Consumers keep their qualification
pattern (`cpueinsum::tcontract::…`) — no new/redundant direct dependency is
added to make a replacement compile.

**Kept as history (not renamed)**: `docs/archive/`; recorded measurement data
(`**/results/**`, `*.csv`, `*.guard`, `manifest.txt`); dated
`docs/superpowers/plans/` + `docs/superpowers/specs/` (incl. this spec);
`docs/decision-log.md` historical rows and the removed-ABI measurement text in
`benchmarks/c/README.md`; `docs/provenance.md` historical statements.

**Historical guards kept verbatim**: `REMOVED_ENV_PREFIXES` stays
`["TENSORCONTRACT_", "TPRIMS_GEMM_"]` — the harness deliberately rejects those
*historically removed* namespaces; renaming the guard would stop detecting
them. This is not an alias. The `packed_explicit_config` test that sets
`TPRIMS_GEMM_KERNEL` to expect a refusal stays as-is.

**Navigational repairs**: relative links inside preserved historical docs that
point at moved evidence paths (`../benchmarks/benchmarks/tprims/…`) are
updated to the new path so the links resolve; the surrounding historical text
is not rewritten.

## Execution

1. `git mv` (order-independent): `crates/tprims-contract` → `crates/tcontract`,
   `crates/tprims-{exec,kernel,capi,testkit}` → `crates/tcontract-*`,
   `benchmarks/benchmarks/tprims` → `benchmarks/benchmarks/tcontract`,
   `.agents/skills/tprims-benchmark` + `.claude/skills/tprims-benchmark` →
   `tcontract-benchmark`, `.opencode/commands/tprims-benchmark.md` →
   `tcontract-benchmark.md`, `include/tprims/tprims.h` →
   `include/tprims/tcontract.h` (then dir `include/tprims` → `include/tcontract`),
   `crates/tprims-capi/tprims.pc.in` → `tcontract.pc.in`, fixture
   `headers_tprims_first.c` → `headers_tcontract_first.c`.
2. One bounded text pass over live files (exclusions: `target/`, `.git/`,
   `docs/archive/`, `docs/superpowers/`, `**/results/**`, `*.csv`, `*.guard`,
   `manifest.txt`), in this order:
   a. `tprims-contract` → `tcontract`, then other hyphen names
      `tprims-exec|-kernel|-capi|-testkit|-bench|-rs` → `tcontract-*`,
      then any remaining `tprims-` → `tcontract-`;
   b. `tprims_contract` → `tcontract`, then any remaining `tprims_` → `tcontract_`;
   c. `TprimsBackend` → `TcontractBackend` (any other `Tprims` camel → `Tcontract`);
   d. `TPRIMS_` → `TCONTRACT_` EXCEPT the file
      `benchmarks/benchmarks/tcbench/knobs.rs` + `benchmarks/src/threads.rs`
      rejection lists (`TENSORCONTRACT_`, `TPRIMS_GEMM_`) and
      `crates/tprims-contract/tests/packed_explicit_config.rs`
      (`TPRIMS_GEMM_KERNEL` refusal test);
   e. remaining bare `tprims` → `tcontract` (libtprims, tprims.h, prose).
   Also update the `TAPP_implementation_name` string.
3. Update `.github/workflows/ci.yml` (crate/lib/symbol/consumer refs) and
   `scripts/check-agent-skills.py` expectations if it names the skill.
4. Add `docs/migration-tprims-to-tcontract.md`, a decision-log row (current
   position: renamed), and a provenance note.
5. **Local gate** (repo AGENTS.md + REPOSITORY_RULES.md): fmt; clippy
   `-D warnings` workspace all-targets; workspace tests; `tprims-exec` without
   `strided`; `tprims-contract` release tests; C ABI build then `nm` audit:
   no `tprims_*`/`TPRIMS_` remains and `tcontract_abi_version` is present;
   the existing installed-prefix `examples/c-consumer` CMake lane and docs
   checks (README/docs against implementation); aarch64 check; MSRV 1.89 build.
6. After merge: rename GitHub repo; downstream updates (below).

## Downstream

- **cpueinsum-rs** (only cargo git-pin consumer): deps
  `tprims-contract`→`tcontract`, `tprims-exec`→`tcontract-exec` at the new
  main rev + URL; imports `cpueinsum::tprims_contract::…`→`cpueinsum::tcontract::…`
  (no new dependency; qualification preserved); rename its own
  `backend::Tprims` public type → `Tcontract`; docs (AGENTS.md, README.md,
  worklog references to the pinned crate names). Run cpueinsum's own checks.
- **tprims-benchmark** (campaign repo): rename repo → `tcontract-benchmark`;
  update `scripts/build_for_tprims_rev.sh`, `scripts/setup_extern_deps.sh`
  (`extern/tprims-rs` → `extern/tcontract-rs`, `crates/tprims-contract` →
  `crates/tcontract`), `pins/tprims-rs.rev` → `pins/tcontract-rs.rev`,
  README; recorded results stay.
- **tenferro-rs / tlinalg-rs / tlinalg-benchmark**: no cargo dep; update only
  current-doc prose naming the library (`tprims` → `tcontract`); leave
  historical/provenance mentions. Tensor4all.jl / tensor4all-rs / BubbleTeaCI:
  no references; no change.

## Open items for review

1. Breaking migration is intentional; GitHub code search shows no external
   users of `libtprims`/`tprims_abi_version` outside this repo. BinaryBuilder
   accommodation in `install.sh` is updated in place; a live recipe was not
   found.
2. `tprims-benchmark` repo rename to `tcontract-benchmark` is proposed for
   consistency; recorded results untouched.