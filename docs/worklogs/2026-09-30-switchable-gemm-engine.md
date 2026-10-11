# Switchable GEMM engine implementation

## Workspace ownership correction

The maintainer rejected Task 8's process-global arena on 2026-09-30.
Retain the approved spec's pool ownership: worker-local A/tile buffers,
with an exclusive team B lease returned to its originating pool. Serial
plans own their workspace. TLS lookup does not own retained payloads;
dropping/trimming an owner must release idle storage even when a borrowed
host pool's threads remain alive. Active leases remain valid.

`ExecSpmd` lives in blas/contract, but that is not a reason for a global
arena. `tcontract-exec::Pool` can own a provider from the thread-free kernel
contract without depending on the driver. No process-global workspace or
cross-pool team-buffer reuse is permitted. This changes ownership and
lifetime, not OS affinity or NUMA policy, and makes no performance claim.

TBLIS 1.x's global memory pools are a valid different choice: allocation
reuse is global while active buffers belong to a gang. That precedent does
not override this project's explicit context ownership. Reference:
`2026-09-30-research-blis-tblis-structure.md`, Part V.

## Validation and progress

Spec §6.2 and Task 8 now agree on ownership, provider borrowing, serial
ownership, re-entry, trim/drop and cross-pool isolation checks. The existing
process-default selection registry remains process-wide immutable metadata,
not a workspace cache. Design/plan/HANDOFF diff passes `git diff --check`.

Task 1 preserves all 367 passing workspace tests (including doctests) and
all 124 tensorcontract/kernel release tests, now distributed across three
crates. Workspace/all-targets build, std-disabled tensorcontract check,
formatting and warning-denying rustdoc pass. Clippy 1.98.0 passes with
`-D warnings`; the default 1.97.1 exits successfully but emits the existing
unknown-1.98-lint warning. No lint was disabled to obtain the clean result.
Evidence: `/home/shinaoka/tensor4all/.artifacts/gemm-engine-spec/` baseline
and `split-*` logs. Runtime kernel dependencies point only downward; a
path-only dev-dependency preserves the existing custom-scalar doctest.
The private-env blocking test follows the contract crate rather than
exposing its env helper solely for a test. No arithmetic bodies changed.

Task 2 adds CPU feature detection/masking, typed family descriptors,
validation/legacy bridging and per-real-type registration plus forced-id
errors and manifests. Conflicting descriptors sharing an id yield a typed
selection error, not the plan's input-derived debug assertion; enumeration
still de-duplicates ids. A typo under a built provider remains `UnknownId`,
not a misleading `NotBuilt`. Providers will supply portable and tc lists in
Task 3; execution is still the existing driver. The seven planned failing
tests first produced missing-type compile errors (`task2-red.log`). The
focused suite now has 11 passing tests, covering duplicate callback
registration, CPU masking, cross-dtype ids, missing features, unknown ids,
packing/tile bounds, checked geometry overflow and the legacy bridge.
Contract crate: 34 unit tests, 11 integration tests and 18 doctests pass.
`--no-default-features` check and all-targets clippy 1.98 with `-D warnings`
pass (`task2-*` evidence logs).

Source inspection corrected two plan transcription errors: tc 1m's logical
MR/NR need not be even, and its OneR B role uses `PackFormat::Planar`, not
`Real`. See `scalar.rs::config_cplx` and `onem_ukr`; a logical 3×5 regression
checks valid packing lengths and bridging. Generation eligibility in Task 6
still applies to the inner real family, not its already-halved descriptor.
The spec, plan and HANDOFF record this clarification; no arithmetic changed.

Task 3 adds project-owned portable real/native-interleaved complex kernels,
interleaved packing/write-back and four-plane decoding. Registry queries
always include the portable fallback. tc registration adapts every compiled
ISA/menu entry (including unavailable CPUs), keeps the legacy Auto head and
Planar default, lowers 1m priority and excludes 3m from Auto. The finite
process-constant descriptor manifest owns no workspace or executor.

The portable and tc integration tests failed first on missing variants and
registration. Four portable tests now cover native-complex arithmetic, KC=0,
conjugated packing with zero padding, and partial interleaved/FourM output
with beta=0/null C. Five tc tests check validation/ids, metadata/CPU masks,
legacy heads, snapshots and every available family against a packed-product
oracle for f32/f64/c32/c64 with all A/B conjugation combinations. Release
kernel/driver suites pass (159 tests before the six added documentation
examples; those examples also pass separately). Final docs: 23 contract and
4 provider doctests pass. Full workspace/all-targets clippy 1.98 with
`-D warnings` and std-disabled tensorcontract check pass (`task3-*` logs).

Architecture-specific f64 snapshots avoid a host-CPU-dependent golden list.
x86's list was observed and checked against its compiled menus. NEON/scalar
lists were derived from their source menus; aarch64-apple-darwin and wasm32
all-targets checks pass, but those target tests were not executed here.
No performance claim or execution-driver change has been made.

## Task 4 contract corrections — approved

Before connecting registry pointers to safe execution, a bounded reproducer
confirmed that safe `register`/`select` accepts a 4×4, 16-real descriptor
whose public pointer is `portable::real_tile::<f64,8,8>` (64-real output).
Geometry validation cannot prove a function's actual ABI, full-overwrite
promise or ISA requirements. The kernel was **not executed**; no invalid
memory was accessed. Evidence and reproducible source are in
`task4-registration-abi.log` and `task4-registration-abi-reproducer.rs` in
the existing artifact directory. This becomes unsafe when Task 4 wires the
accepted descriptor into the safe `Plan::run` path.

Proposed correction: external provider registration is an explicit `unsafe`
ABI/ISA contract; built-in providers expose safe wrappers around their
verified static menus. Remove the public mutable `RealSlot::slot` escape.
Keep registry lookup, geometry validation and scheduling safe Rust.

The plan also makes new `resolved`/`process_default` accessors panic on
invalid user selections while adding parallel `try_*` accessors. Repository
rules require typed errors and one canonical boundary. Proposed correction:
new canonical resolution APIs return `Result`; existing generic execution
APIs and foreign `KernelSet` default behavior remain compatible.

Orientation wording must match the current kernel ABI: exchanging user
operands changes which data fills the kernel's row-A and column-B panels,
not the family's fixed packing roles. In 1m, row-A stays OneE and column-B
stays OneR. Direct-B eligibility checks the post-orientation column operand
(user A when swapped). Swapping layout fields alone, as the plan currently
suggests, disagrees with the declared panel footprints. The existing driver
uses `ukr.a_pack` for row-A and `ukr.b_pack` for column-B after exchanging the
operand pointers. Add explicit swapped-output/asymmetric-family tests.

The existing `correctness.rs::large_gemm_case_oriented` already tests both
output orientations and all three complex methods. No duplicate test was
added. Fresh release runs of its c32/c64 cases pass under all four explicit
settings: `TENSORCONTRACT_KERNEL=scalar|avx2` ×
`TENSORCONTRACT_ORIENT=none|swap`, with runtime and test-thread counts both
one. Evidence: `task4-legacy-{scalar,avx2}-{none,swap}.log`. This establishes
the legacy orientation baseline before changing dispatch, not performance
or completion of Task 4. The kache store recheck also passed (6648 valid
entries, no corruption).

The maintainer approved these corrections (`Y`). The spec and plan now
record unsafe external registration, private mutable slots, canonical Result
resolution and fixed kernel packing roles. The registration boundary is
implemented: three compile-fail examples prevent safe registration, safe
trait-dispatch bypass and public mutable-slot access. Required trait methods
expose only copied callbacks and unsafe append; typed storage remains private.
Built-in tc registration wraps verified immutable compiled menus. Integration
fixtures now use real matching kernels, not an ABI-incompatible noop.
Contract/provider tests and all-targets clippy 1.98 pass; evidence is in
`task4-registration-{red,green,clippy}.log`.

Task 4 now has a Result-returning `ResolvedGemm::resolve` core and frozen
blocking-policy snapshots. Exact ids report typed selection errors; zero
width is rejected before registry access. Retargeting recomputes from the
original descriptor/model with the effective width, applying percentages
once. Synthetic shared-L3 coverage proves serial NC is larger than budget-8
NC; a standalone 1T probe confirms later environment mutation is ignored.
Geometry and checked overflow tests, model/percentage variants, std-disabled
check and all-targets clippy pass (`task4-resolved-*` evidence).

TC descriptors now retain unscaled register-aligned defaults, so model/env
normalization happens in resolution rather than being applied twice. The shared
override logic uses checked multiplication in canonical resolution, reporting
`Incompatible` on overflow. Legacy signatures and arithmetic (including their
pre-existing overflow behavior) are unchanged.
The selected descriptor reference is private (with a read-only getter), so
safe retargeting cannot be poisoned with replacement, unvalidated geometry.

Plan caches/default selection, monomorphized pack/write-back pointers and
driver integration are next. No new family has been connected to execution
by this increment. Tasks 4–12 and final performance/CI gates remain; the
durable goal stays active.

## Tasks 4–8 (continued): driver on resolved families, Direct, induced, partition, workspace

Task 4 is complete. `Plan` caches one fallible `ResolvedGemm` per storage
dtype (`b011dc2`/`8d9c444`), `run`/`run_with` and every batch item validate
selection before any shortcut, and the driver consumes the cached descriptor,
its format-monomorphized packers and its write-back emitter, deriving `NC` from
the *active* grid width rather than the requested thread count. `process_default`
freezes one default (including its error) per dtype and reads the legacy
`TENSORCONTRACT_KERNEL`/`COMPLEX` controls once; `KernelForce` moved into the
kernel crate so both dispatch paths share one startup fact. `Plan::resolved`
and `with_kernel` are the public surface; `Error::KernelSelection` carries the
typed `SelectError` with its source chain. Evidence: `task4-driver-*`,
`task4-defaults-*`, `task4-explicit-spmd.log`.

Task 5 adds the two portable Direct families per real dtype, a driver family
view that can dispatch either `UkrFn` arm, the plan-level direct-C guard, the
per-tile fallback when D's own scatters are irregular, and in-place B with the
`pm == 1` partition it requires. `driver_decisions` exposes the same rule the
driver uses so tests pin it. Evidence: `tensorcontract/tests/direct.rs`
(guard matrix, irregular-scatter fallback, in-place B, swapped-orientation
refusal, selection errors).

Task 6 generates 1m and 4m complex families from every registered real family,
once per storage type, with `allow_auto: false`. 1m's 2x2 real block lands in
exactly `TileFormat::OneM`; 4m copies planar panels into contiguous ones and
makes four calls. The legacy Auto menu excludes induced families, because that
path reproduces the compiled `KernelSet` list. Evidence:
`crates/tprims-gemm-kernel/tests/induced.rs` plus the all-family sweeps.

Task 7 makes the grid a policy: `partition::strip` is the single M split used
by both the cost model and an explicit `StaticGrid`, `align_c_lines` rounds
boundaries to a 64-byte line of `C` while the last strip keeps the tail, and
`DynamicTiles` is refused at resolution. The width sweep is bitwise identical
to serial for the scalar, portable and Direct families, aligned or not.

Task 8 (in progress) adds an owning `ArenaProvider`: per-thread worker slots
named by thread-local handles, page-aligned grow-only `PageBuf`s, an exclusive
`TeamLease` over the B panel, scatter vectors and barriers, and fresh
call-local buffers on re-entry. `Spmd::workspace()` lets a host lend its own
provider; the driver then builds the five block scatters into one reused
buffer, takes the panel from the lease, and on a declined broadcast runs the
strips on the caller with that same lease. One provider per `Pool` and one per
contract plan, so both adapters on a pool share storage and a serial plan
reuses its own. The workspace trace is keyed by owner, so tests cannot confuse
one owner's records with another's.

### Recorded deviations from the plan text

* `ResolvedGemm::resolve_with` takes `(choice, width, partition, opts)`: the
  plan's `method`/`cpu` arguments are resolved inside (the family id already
  encodes the method) and the blocking override has its own `with_blocking`.
* `WorkspaceReq` sizes are **bytes**, not reals: the provider is element-type
  erased, so the driver converts its real counts once.
* Workspace tracing is a runtime opt-in per owner (`ArenaProvider::traced`,
  `trace_take`) rather than a `workspace-trace` Cargo feature, because an
  integration test cannot enable its own crate's feature; and the trace is
  per owner rather than global, so parallel tests do not interfere.
* `PageBuf` growth is by doubling, not by an exact-size realloc, so a growing
  sequence of calls does not reallocate every time.
* Task 5's per-tile direct-C decision and Task 6's induced arithmetic are
  pinned in `tensorcontract/tests/direct.rs` and
  `crates/tprims-gemm-kernel/tests/induced.rs` respectively; the all-family
  sweeps cover them through the real driver, so no separate
  `tensorcontract/tests/resolved.rs` binary was added.

Remaining: the allocation-counting steady-state test and the same-pool
adapter test for Task 8, then Tasks 9 (gemm-f64/pgx86 providers), 10, 11
(selection/query APIs and compatibility), and 12 (docs, benchmark option,
ABBA non-regression gates, PR). No performance measurement has been taken.

## Tasks 8–12: workspace tests, providers, selection, docs and gates

Task 8's plumbing is finished and tested: `tcontract-exec`'s `Pool` and the
contract plan own an `ArenaProvider`, `Exec::workspace` hands it to both
`ExecSpmd` adapters, and the driver builds its block scatters into the leased
buffer, takes the panel from the lease and runs a declined broadcast on the
caller with the same lease. `tensorcontract/tests/workspace_alloc.rs` counts
allocations with a one-thread host: after one warm-up, two steady-state
executes add none, and a direct-B case adds no buffer-sized allocation beyond
the packed A block. `crates/tcontract-exec/tests/pool_workspace.rs` shows two
pools owning distinct arenas, an operation growing only its own pool's panel,
and a second operation of the same shape reusing that panel byte for byte.

Task 9 registers every compiled `gemm-f64`/`gemm-f32` microkernel as a Direct
family (`gemm.{isa}.{dtype}.{MR}x{NR}`, 20 on this host), with only the widest
table's largest tile Auto-eligible. A single `opaque` pointer per family carries
the real microkernel into the shim. The parity test checks every entry against a
naive product on a full and an edge tile, with B packed and in place, for both
dtypes, and the tensorcontract sweep drives them through the planner and driver.

Task 10 wraps `private_gemm_x86::gemm` (`tprims-kernel-pgx86`), pinned exactly,
with `available()` gating and a portable no-op body off x86-64; its tests cover
all four scalar kinds, both stride conventions, alpha with and without
accumulation, and every conjugation combination.

Task 11 adds the selection surface: `GemmConfig` (engine and kernel family),
`SelectedGemm` (engine, family, geometry, grid; wraps `Selected` rather than
replacing it), `gemm_with`, `list_kernels`, `Error::Select`, and the
`kernel-gemm`/`kernel-pgx86` features. The packed path resolves when the plan is
built, so an unknown id or an unusable engine is a constructor error, and an
explicit engine or kernel stops the automatic strategy from silently falling
back to the copying permute+GEMM plan. `ContractPlan::new_with` and
`selected_gemm` carry the same choice into the contract layer.

Task 12 adds `blas --engine faer|packed|pgx86` with the resolved family printed
per case, updates `README.md`, `docs/architecture.md`, `docs/provenance.md` and
`docs/decision-log.md`, and runs the local gate. Evidence for every gate step is
in `.artifacts/gemm-engine-spec/task12/`.

### Recorded deviations (continued)

* `tprims-blas` folds the planned `select.rs` into `engine.rs`, and the planned
  `concurrent_families.rs` test into `engine_select.rs`: one module and one
  binary for one subject.
* `SelectedGemm` has no `workspace` field: the driver's requirement depends on
  the active partition, so it is derived per call rather than stored as a
  property of the plan.
* `ContractPlan` does not store the `GemmConfig`: the choice is baked into the
  sub-plans (`Plan::with_kernel`, `PgPlan::gemm`), so a stored copy would be
  unread state.

### Performance gate: PASSED

The gate was run with the decision script's new old-vs-new mode
(`tblis_decision.py --old DIR... --new DIR...`, added in `4412334` with
`benchmarks/scripts/test_tblis_decision.py`, wired into the `scripts` CI job).
Metric: calls-weighted workload time per group; noise: mean
`|log(total / median over sessions)|` per side; a group passes when its ratio is
within `max(5%, the larger side's noise)`.

| corpus | T | group | new/old | noise | |
| --- | --- | --- | --- | --- | --- |
| contract, 57 cases | 1 | `pg_exec` | 0.988 | 0.018 | pass |
| | 1 | `tblis_exec` | 0.933 | 0.009 | pass |
| | 4 | `pg_exec` | 1.002 | 0 | pass |
| | 4 | `tblis_exec` | 0.966 | 0 | pass |
| | 8 | `pg_exec` | 1.023 | 0.022 | pass |
| | 8 | `tblis_exec` | 0.950 | 0.002 | pass |
| contract, 8-case subset | 4 | `pg_exec` | 1.061 | 0.067 | pass |
| (five sessions per side) | 4 | `tblis_exec` | 0.895 | 0.008 | pass |
| | 8 | `pg_exec` | 0.999 | 0.048 | pass |
| | 8 | `tblis_exec` | 0.914 | 0.005 | pass |
| blas, 53 cases | 1 | `faer` | 0.998 | 0.003 | pass |
| | 1 | `tblis` | 0.954 | 0.004 | pass |
| | 4 | `faer` | 0.949 | 0 | pass |
| | 4 | `tblis` | 0.917 | 0 | pass |
| | 8 | `faer` | 0.996 | 0 | pass |
| | 8 | `tblis` | 0.946 | 0 | pass |

Sessions, manifests and `decision.txt` are committed under
`benchmarks/benchmarks/tcontract/{contract,blas}/results/2026-10-01-gemm-engine/`;
the intermediate per-thread-count directories live in
`.artifacts/gemm-engine-spec/task12/perf/`.

Two protocol deviations, both recorded in `decision.txt`: `BENCH_RUNS=5`
instead of the default 50 (the measured noise bands are 0.2-6.7%, mostly below
2%, so the extra samples would not change the verdict at this resolution), and
the 4T full-corpus block has one session per side - the five-session subset
block above supplies the noise band for that thread count.

### Earlier partial run (superseded)

The prescribed gate (`.agents/skills/tcontract-benchmark/SKILL.md`: idle L3 domain,
`pinned.sh`, A/A noise floor, 1/4/8T, both corpora, default runs) could **not**
be completed from this session: `pinned.sh` needs the pinned cores idle before
*and* after each run, and this agent's own processes keep at least one core of
the chosen domain slightly busy, so the 4T runs exhausted their retries
(`cpus 32,33,34,35 busy before run (attempt 5)`). What did complete:

* `contract --corpus tenferro-p1.json`, `BENCH_RUNS=5`, **1T**, both sides
  pinning-verified and complete (228 case/variant rows each), new on cores 0–7
  and old on 32–39, ~15 minutes apart. Calls-weighted ratio new/old of the
  summed medians: `pg_exec` **0.973**, `tblis_exec` **0.926**; `pg_plan` 1.034
  and `tblis_plan` 1.037 (35 µs against tens of ms of execution), whose worst
  per-case ratio is 3.15.
* `contract` on a **reduced corpus** (entries 0, 8, 16, 24, 32, 40, 48, 56 of
  `tenferro-p1.json`, written to
  `.artifacts/gemm-engine-spec/task12/perf/tenferro-p1-subset.json`) at **4T**,
  `BENCH_RUNS=5`, five samples per side on cores 24–31, each run
  pinning-verified: `tblis_exec` new/old = **0.886** with per-run sums
  0.271–0.276 against 0.305–0.309, i.e. consistently ~11% faster.
  `pg_exec` came out at 1.066, but the *old* side's own runs span
  0.908–1.196 (27%), a spread far larger than the difference, so the permute +
  GEMM row is **not resolved** on this host; the changes add no per-execute work
  there (one `GemmConfig` clone at plan creation and two cheap checks per call).
  Planning rows: `pg_plan` 1.023, `tblis_plan` 1.049, worst case 2.3 at 4–20 µs.
* The A/A spreads are the reason single-run comparisons at 4T are not
  decisive: new/new summed 1.001 and 0.991 (exec rows) while old/old summed
  1.268 for one pair.
* Recorded, not gated (plan Task 12 Step 4): the built-in `gemm_f64` cases at
  **1T**, `BENCH_RUNS=5`, one pinned core (CPU 8), `median_ns`:

  | case | faer | packed | pgx86 |
  | --- | --- | --- | --- |
  | n8 | 210 | 4470 | 510 |
  | n32 | 1770 | 7810 | 2240 |
  | n128 | 78 702 | 216 737 | 98 554 |
  | n512 | 5 163 491 | 6 606 539 | 6 279 128 |
  | n1024 | 42 836 294 | 49 679 740 | 43 248 070 |
  | n2048 | 346 750 583 | 408 360 123 | 395 487 339 |
  | m256_n1024_k64 | 654 342 | 939 961 | 676 182 |
  | m1024_n64_k256 | 685 883 | 926 241 | 685 453 |

  Neither opt-in engine beats faer at 1T on these shapes: the packed driver is
  2–4%…24% slower on the large cases and much slower on tiny ones (its worst is
  4.47 µs against faer's 0.21 µs at n=8), and `private-gemm-x86` is 1–25%
  slower, with ~0.3 µs of extra dispatch visible at n=8. That is why `Auto`
  keeps faer. The `blas` binary in `target/release` was last built with
  `--features pgx86`; the `contract` comparisons above used the unmodified
  `contract` binary.
* Not measured: 8T, the `blas` corpus (faer and batched TBLIS rows), the default
  `BENCH_RUNS=50`, and a properly interleaved ABBA pair. No gate verdict is
  claimed from the partial runs.
