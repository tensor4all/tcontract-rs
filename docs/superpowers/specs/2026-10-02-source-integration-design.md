# Source integration and clean-slate crate redesign

Status: **accepted** (2026-10-02, after the Fable 5.1 review). Date: 2026-10-02. Baseline: `origin/main` `fbc83f5`.

Audited against issue [#37](https://github.com/tensor4all/tprims-rs/issues/37)
and the baseline source on 2026-10-02; audit record:
[`docs/worklogs/2026-10-02-source-integration-audit.md`](../../worklogs/2026-10-02-source-integration-audit.md).
The clarifications below preserve D1–D10. This remains a draft, not a claim
that the implementation or Phase 2 performance gates have passed.
The companion [detailed design](2026-10-02-source-integration-detail-design.md)
defines data structures, call boundaries and the staged source moves.

This spec records the maintainer decisions taken on 2026-10-01/02. It covers
**Phase 1**: dissolve the `tensorprimitives/` subtree into a newly designed
crate set and remove the parts that leave the repository. **Phase 2**
removes faer. It is summarised in §10 and gets its own spec later.

## 1. Goals and decisions

The goal is one coherent library: each responsibility exists in exactly one
place and every numerical test of a retained feature still passes or is
ported. A smoke run screens for gross regressions; it cannot establish
performance parity.

| # | Decision |
|---|---|
| D1 | `tensorprimitives/`, imported from lkdvos/tensorprimitives-rs (Lukas Devos), is dissolved into tprims. No upstream sync or separate product continues: no crates.io `tensorcontract`, no JLL/Yggdrasil, no Julia wrapper. Authorship and git history are preserved (§9). |
| D2 | The public Rust API may break freely. tenferro gets a migration guide; the `ext/tenferro-cpu-tprims` update is separate work. |
| D3 | Removed: the public BLAS API (`gemm`, `gemm_batched`, `gemm_grouped`, `trsm`), the BLAS C ABI (`tprims-blas-capi`), the engine layer (`Engine`, `EngineChoice`, `TPRIMS_GEMM_ENGINE`), the private-gemm-x86 engine (`tprims-kernel-pgx86`), the gemm-f64/f32 adapter (`tprims-kernel-gemm`) and `tprims-linalg`. BLAS and linalg may return later as separate repositories. |
| D4 | Phase 1 keeps faer only as an internal contraction strategy, for copy-free permute+GEMM and the batched faer loop. It lives in one module. Phase 2 deletes it after the packed driver is optimized. |
| D5 | **Amended 2026-10-03:** contractions that fuse to a strided (batched) GEMM without a copy run on **faer**, of any size; contractions that would need a copy run on the **packed driver** (kernel families); all-batch (no M, N or K) problems are delegated to **strided-rs**. faer stays a permanent dependency. This supersedes the earlier "one route, delete faer" target; see the [Phase 2 spec](2026-10-03-phase2-faer-routing-and-packed-driver-design.md). |
| D6 | The central contraction representation is a lowered, role-grouped problem (§4.1). Label-based (TAPP) and `DotGeneral` (Rust/tenferro) inputs are thin front ends lowered at plan time. Execution cost is identical: the driver already runs on precomputed offset tables (`tensorcontract/src/plan.rs:275`), not on labels. |
| D7 | Kernel family IDs are renamed to one scheme (§6.2). |
| D8 | The library reads **no environment variables**. Every tuning knob becomes explicit configuration; benchmarks translate env/flags into it. |
| D9 | Benchmarks are trimmed to the remaining surface, but the Hadamard corpus (`hadamard.json`) and the GEMM-route corpora are kept, rewritten as contractions where needed. |
| D10 | Heavy performance protocols are not part of Phase 1. Verification is correctness tests, the local gate and one 1T smoke run (§11). |

## 2. Current state (facts at `fbc83f5`)

- 19 workspace members, including `benchmarks`. Lukas Devos's crates are `tensorprimitives/crates/{tensorcontract, tensorprimitives-tapp, tensorprimitives-bench}`. The kernel layer was already moved out of `tensorcontract` in #27.
- Duplicated responsibilities:
  - **Executors and SPMD.** tensorcontract `pool.rs`/`spmd.rs`, `tprims-exec`, the `tprims-core` executor and `tprims-contract-traits::HostExecution`. `Par` is defined twice, and an identical `ExecSpmd` adapter exists three times (blas/tblis.rs, contract/tblis.rs, tapp lib.rs).
  - **Validation.** tensorcontract plan/resolve validate labels, while `DotGeneral::validate`/`validate_layouts` validate axis lists. `is_injective_layout` exists twice, and there are three overlap checks (`core::tensor::spans_overlap`, tapp `overlap`, blas-capi `check_no_overlap`).
  - **TBLIS bridge.** It is re-implemented in tprims-blas and in tprims-contract.
  - **Errors and oracles.** There are seven or more error types and five or more reference oracles.
  - **Thread spawning in tensorcontract.** It still spawns its own threads: a driver fallback using `std::thread::scope`, `batch.rs`, and `pool.rs`.
- Environment variables read in source:
  - In gemm-kernel: `TENSORCONTRACT_{THREADS,KERNEL,COMPLEX,MC,KC,NC,MC_PCT,NC_PCT,KC_COUPLE,BLOCKMODEL,L3_DOMAINS,WRITEBACK}`.
  - In tensorcontract plan/pool: `TENSORCONTRACT_{ORIENT,ROWBLOCK,PARTITION,POOL}`.
  - `TPRIMS_GEMM_ENGINE` and `TPRIMS_GEMM_KERNEL`.
- Files over 800 lines:
  - `tensorcontract/src/plan.rs` (1850)
  - `driver.rs` (1548)
  - `tprims-gemm-kernel/src/cache.rs` (1622)
  - `tensorprimitives-tapp/src/lib.rs` (1239)
- External consumer: tenferro `ext/tenferro-cpu-tprims` pins rev `d8e565a`. It uses:
  - `tprims_exec::{Exec, Pool}`;
  - `tprims_blas::{gemm, gemm_batched, gemm_grouped, trsm, ...}`;
  - `tprims_contract::{ContractPlan, DotGeneral, Flags, Strategy}`;
  - `tprims_linalg` decompositions.

## 3. Target crate set

The final **normal/build dependency** graph is:

| Consumer | Project dependencies |
|---|---|
| `tprims-exec` | none; optional `strided-basic` adapter |
| `tprims-kernel` | none; no dependency on exec, contract or capi |
| `tprims-contract` | exec, kernel, strided-view/basic; internal faer in Phase 1 |
| `tprims-capi` | contract, exec |
| `tprims-testkit` | contract, strided-view; unpublished test support |
| `tprims-bench` | contract, exec, kernel, capi, testkit as needed; unpublished |

Exec and kernel are siblings; neither depends on the other. Contract joins
kernel descriptors to exec's dtype-independent workspace. Testkit's normal
dependency on contract and contract's **path-only dev-dependency** on testkit
are deliberate: no production dependency on the oracle, no versioned
publication cycle. A dependency-direction test checks the final graph.

| Crate | Contents (from) | Public surface |
|---|---|---|
| `tprims-exec` | `tprims-exec`; tensorcontract `spmd.rs`; the workspace from gemm-kernel `workspace.rs` | `Pool`, `Exec`, `install`, `for_each_partition`, `broadcast`, `WidthPolicy`, workspace providers, `strided::run_with_exec` (feature `strided`) |
| `tprims-kernel` | gemm-kernel; kernel-tensorcontract; kernel-cplx; portable | `KernelFamily`, `KernelCatalog`/`KernelHandle`, `KernelChoice`, `list_kernels`, `unsafe register`, `SelectError` |
| `tprims-contract` | contract-traits; contract; tensorcontract plan/driver/dynamic/batch/select/resolve/layout/buffer/scatter/element | `Problem`, `Labels`, `DotGeneral`, `Plan`, `PlanConfig`, `ContractionBackend`/`BoxedPlan`, `Error`, `add`, `permute` |
| `tprims-capi` | tprims-core; tensorprimitives-tapp; tprims-bundle | TAPP standard ABI, `tprims_tapp_executor_*`, DLPack, status; builds `libtprims` (cdylib/staticlib/rlib) |
| `tprims-testkit` | tensorcontract `reference.rs`; contract-testkit; the per-test naive helpers | the reference oracle, seeded input generators, a naive second backend (`publish = false`) |
| `tprims-bench` | `benchmarks/`; tensorprimitives-bench | contract, exec_entry, capi_rust, tcbench (TCCG corpus; TBLIS FFI behind a feature) |

Deleted crates:
- `tprims-blas`
- `tprims-blas-capi`
- `tprims-linalg`
- `tprims-kernel-pgx86`
- `tprims-kernel-gemm`
- `tprims-bundle`
- `tprims-custom-kernel-test` (its tests move into kernel/contract)
- `tprims-contract-testkit` (moves into testkit)
- `tprims-contract-traits` (moves into the `tprims-contract::api` module)

The unused workspace dependencies `strided-perm` and `strided-kernel` are removed. The `strided-view`/`strided-basic` pin stays at the tenferro rev.
Package names, workspace dependencies, lockfile, features, test fixtures and
CI commands move together.

## 4. `tprims-contract`

### 4.1 Central representation

`Problem` is the single validated metadata description of
`D = op_D(alpha * sum_K(op_A(A) * op_B(B)) + beta * op_C(C))`:

- The supported storage types are f32, f64, c32 and c64, all operands of one
  type and accumulation at storage precision. Mixed precision and foreign
  scalar extension are not retained in Phase 1.
- Extents, signed element strides and logical offsets for A, B, D, and
  optionally a separate C. No tensor payload, pointer, executor, alpha or
  beta is stored in the problem. Alpha and beta are execution arguments.
- The C mode is explicit: overwrite (no C term), accumulate from D, or
  accumulate from a separately described C. Omitting C never implicitly
  changes overwrite into accumulation.
- Axes grouped by role, exactly as the driver uses them today:

  | Role | Present in |
  |---|---|
  | batch | A, B, D (and C) |
  | M | A, C, D |
  | N | B, C, D |
  | K | A and/or B, absent from C/D; a missing operand has stride zero |

  The table describes the reduced logical axes. Labels repeated within an
  operand select a diagonal: validate equal extents, then sum their signed
  strides with checked arithmetic. Apply this to C and D too; an output
  diagonal writes only that diagonal. A label isolated in A or B is a
  reduction, represented as K with zero stride in the other input. A label
  appearing only in C/D remains `Unsupported` (TAPP case 5).
- Per-operand conjugation for A, B, C and D. `op_D` applies to the **whole**
  result, including the beta term. It is not just conjugation of A*B.
- Keep the original operand metadata for execution layout matching and
  conservative addressed-span checks, plus the normalized role metadata
  used for planning. Fields are private; constructors return typed errors.

Metadata validation happens when the `Problem` is built, before scatter
allocation, selection or a zero-size shortcut:

- label count/rank, repeated-label extent agreement, and DotGeneral axis
  bounds, uniqueness and disjoint contracting/batch sets;
- extent agreement;
- checked role products, repeated-stride sums, folding, scatter sizes,
  signed reachable offsets, byte spans and FFI integer conversions;
- output injectivity on the reduced logical D domain (conservative rejection
  is documented); zero input strides and valid negative strides are allowed;
- C and D have the same logical labels/extents, but may have different
  orders and strides; precompute whether their mappings can be identical.

Pointer-dependent validation happens on **every execution**, before any
write: match the prepared shapes/strides/offsets, check Rust backing-slice
bounds, and reject D/A or D/B span overlap. Separate C/D overlap is allowed
only for identical logical address mappings; every other overlap is rejected,
even at beta zero. Address additions are checked as well as relative spans.
TAPP has no allocation lengths: its unsafe caller contract guarantees valid
storage, alignment, lifetime and exclusive output access; runtime checks do
not pretend to prove that arbitrary pointers belong to live allocations.

This replaces the duplicated injectivity and overlap checks listed in §2.

### 4.2 Front ends

- **`Labels`.** Per-operand i64 labels, with free output order and separate C and D. Used by TAPP.
- **`DotGeneral`.** Paired contracting and batch axes. Output order is `[lhs_free, rhs_free, batch]` and C = D. Used by Rust and tenferro.

Free axes keep their original operand order; batch axes use the supplied
paired-axis order. Repeated *axis indices* in DotGeneral are invalid, unlike
repeated *labels* in Labels. Equivalent front ends produce the same reduced
problem, validation categories and numerical result; the label front end
also supports diagonals and isolated reductions that DotGeneral cannot name.

Both lower to `Problem` at plan time. No front end is visible to the driver.

### 4.3 Plan

`Plan::<T>::new(&problem, &PlanConfig) -> Result<Plan<T>>` snapshots metadata
and config and owns the selected kernel handle. A selector or catalog need
not outlive construction; any required provider state must outlive the plan
through its handle (static descriptors remain static). Plans are reusable,
`Send + Sync`, and executable concurrently on independent outputs. No input
payload is retained. Replanning is explicit when metadata/config changes.

`PlanConfig` holds:

- `kernel`: Auto or an owned ID. The typed catalog/selector (#28) is a
  borrowed `new_with_selector` construction argument, not a closure embedded
  in the dtype-independent PlanConfig;
- `partition`: optional explicit StaticGrid or DynamicTiles (#29); absence
  allows strategy selection, with StaticGrid as the packed default;
- `no_materialize`;
- explicit tuning (§6.4), which replaces the removed environment variables.

The width policy and cost estimate are internal defaults (§5.2), not
configuration.

On a packed plan the family is resolved once, at plan time, after built-ins
are available. The selector runs at most once, sees the documented advisory
planning budget, and is not retained. Execution never reselects, reads the
environment or registers anything. It works correctly at every Exec budget;
the planning budget is not a runtime requirement. Fixed blocking and family
remain unchanged while scheduling and scratch requirements may adapt to the
actual width. A non-packed plan reports no selected family.

In Phase 1 the plan picks one internal strategy, using today's rule:

1. Explicit kernel/selector, partition, or packed-driver tuning requests
   select the packed driver, including for all-batch problems. A request is
   applied or rejected at plan time; never silently ignored.
2. Otherwise **Elementwise** when every axis is batch: strided-basic kernels
   with the full semantics (op_C, op_D, a separate C and conjugation), where
   `op_D` is folded into the other terms and one dispatch per execution picks a
   monomorphized, flag-free closure. No runtime mode flag is tested inside the
   per-element loop. This strategy is permanent (D5).
3. Otherwise **faer permute+GEMM** only when it fuses every operand without a
   copy and implements the full C/D/conjugation semantics. If it cannot,
   select packed. Distinct C/D must not cause a hidden copy into D.
4. **Packed driver** otherwise. This is the default for non-fusable layouts.

Strategies live in `src/strategy/{faer.rs, elementwise.rs}`. No public enum names them; `plan.report()` only displays the choice, together with the family, blocking, partition and materialization. `strategy/faer.rs` is the only file that depends on faer. Phase 2 deletes `faer.rs`; `elementwise.rs` stays, delegating to strided-rs.

Execution uses `StridedView`/`StridedViewMut`. The overwrite operation is
`execute_into(&exec, alpha, a, b, d)` and never reads previous D. The update
operation is `execute_into_accum(&exec, alpha, a, b, beta, source, d)`, with
an explicit accumulation source: D itself or a separate C view. Reading D
goes through its mutable view; callers must not construct simultaneous
immutable C and mutable D references to the same allocation. Prepared C
metadata must match the chosen source. An unsafe raw-pointer variant serves
TAPP, and `contract_batched` executes many items (§5.2).

`no_materialize` forbids full-operand normalization and temporary-result
copy-back. Bounded panels/tiles required by the packed driver are workspace,
not full-tensor materialization, and remain allowed. Report these separately,
including scratch bytes and the reason for any fallback from direct B/C.
Direct-C uses the selected family's existing scratch fallback for separate
C, incompatible layouts or conjugation; it never rereads beta*C for every
KC block. Phase 1 preserves the baseline direct-C applicability rules.

After validation, zero output size writes nothing; K=0 or alpha=0 reads no
A/B payload and computes `op_D(beta * op_C(C))`, including for a separate
C. The baseline `util::scale(c, beta)` shortcut does not do this, and the
output-update helper must. beta=0 reads no C or old D.
An empty K role set has product 1 (outer/Hadamard/scalar cases); a K extent
of zero has product 0. Check configuration even on these shortcuts. No
bitwise reproducibility across families is promised; tolerances and residuals
are dtype/method-specific, and 3m's accuracy tradeoff stays explicit opt-in.

### 4.4 Backend trait

`ContractionBackend<T>` / `PreparedContraction<T>` / `BoxedPlan<T>` (#31) move into the `tprims_contract::api` module.

They take `&Exec` directly. This removes the following #31 types:

- `HostExecution`
- `SerialHost`
- `NativeHost`
- `ExecHost`
- the `TypeId` downcast

`Plan<T>` implements `PreparedContraction<T>`; a separate `TprimsBackend`
factory owns PlanConfig and implements `ContractionBackend<T>::prepare`
with common Requirements and an advisory PlanningBudget. The methods remain
object-safe for fixed T and use the same prepared metadata, update naming,
typed errors and reporting as the concrete API. The naive second backend
lives in testkit and must work through this seam without private driver access.
This is a whole-backend seam; consolidation deliberately no longer offers
the dependency-isolated traits crate or arbitrary HostExecution callbacks.

### 4.5 Errors

There is one `tprims_contract::Error`, with these variants:

- `Config`
- `Shape`
- `Layout`
- `Alias`
- `Unsupported` (including an explicit would-materialize reason)
- `Select(tprims_kernel::SelectError)`
- `Exec(tprims_exec::ExecError)`
- `Backend` (a typed/source-preserving foreign-backend failure)
- `Internal` (an impossible invariant, never an input-validation catch-all)

Sources are typed, never stringified. Shape/config/layout/alias errors carry
structured operand, axis/label and expected/actual context. Lowering label
failures have a typed reason so the C boundary can preserve `LABELS` versus
`SHAPE`. Preparation and input-validation failures write nothing; backend
failures after compute starts may leave a partial output and never promise
rollback. `is_unsupported()` preserves #31's decline-before-writing contract.
The `tensorcontract::Error`, `contract-traits::Error`, `blas::Error` and
`linalg::Error` types disappear. `tprims-capi` maps Error in one exhaustive
table; FFI-only null/device/read-only/lifecycle and caught-panic statuses stay
in capi rather than leaking into contract.

### 4.6 File layout

```
src/
  lib.rs
  api/{problem.rs, labels.rs, dot_general.rs, validate.rs, backend.rs, error.rs}
  plan/{mod.rs, analysis.rs, orientation.rs, report.rs}   ← tensorcontract plan.rs (1850) split
  driver/{mod.rs, static_grid.rs, dynamic.rs, tile.rs}    ← tensorcontract driver.rs (1548) split
  select.rs, resolve.rs, layout.rs, buffer.rs, scatter.rs, batch.rs
  strategy/{faer.rs, elementwise.rs}                      ← faer.rs Phase 1 only
  wrappers.rs                                             ← add, permute (strided-basic)
```

These paths mark responsibilities, not mandatory file counts. Analysis owns
label reduction/classification/folding; orientation owns role swaps; driver
modules share one validated call/context and one output-update implementation.
Cache probing and the blocking model have separate inputs and tests. Extract
large inline tests into module-local test directories. Do not duplicate
validation between `api/validate.rs`, layout helpers and C wrappers.

## 5. `tprims-exec` and the parallel strategy

### 5.1 Primitives

- `Pool<'p>` owns the SPMD gate, workspace and entry counters and borrows
  `&'p rayon::ThreadPool`; Pool::owned stays for the C host. The host creates
  one wrapper per raw pool and shares it across calls. This is the existing
  coordination contract, stated in docs/examples: do not add an exclusive
  mutable borrow, global identity registry or wrapper-deduplication machinery.
- `Exec` is either Serial or Rayon { pool, budget }.
- `install(k, op)`: width one runs inline; a wider region enters the pool once.
- `for_each_partition(k, f)` is barrier-free.
- `broadcast(width, f)` keeps the baseline semantics. It runs the whole team
  concurrently or nothing, is serialized per pool, and is refused when wider
  than the pool or budget, or when called from a worker of the same pool. Width one runs
  inline.
- Workspace: `ArenaProvider`, `TeamLease` and the rest move here from
  gemm-kernel unchanged, together with their lease, trim and reentry
  behaviour and their tests. There is one arena per Pool and a Plan-owned
  arena for serial reuse, with no process-global state. Requests describe
  bytes, alignment and barrier counts, never KernelFamily types. Contract
  computes them from the resolved family.
- `strided::run_with_exec` is kept for `add`/`permute`. The Phase 1 faer
  strategy does no operand/result normalization copies.

Deleted:

- tensorcontract `pool.rs` and `TENSORCONTRACT_POOL`;
- the `Spmd` trait;
- every `std::thread::scope` path;
- the three `ExecSpmd` adapters;
- the duplicate `Par`;
- `TENSORCONTRACT_THREADS`.

### 5.2 Parallel strategy

1. **One source of threads.** Threads come only from `Exec`; without one, execution is serial.
2. **Width.** It is chosen once per execution, from the Exec budget and the
   plan's work estimate, by the existing `WidthPolicy`. The two duplicated
   `NS_PER_FLOP` constants (tapp and blas) become one internal default, the
   baseline's provisional 0.05 ns per real flop. Work below the serial
   threshold runs inline.
3. **Inside one contraction.** Width one is serial. Wider work runs SPMD
   via `broadcast` with StaticGrid (MR/NR-aligned pm×pn) or opt-in
   DynamicTiles, as in the baseline. In-plan batch axes stay serial inside
   the team in Phase 1.
4. **Many items** (`contract_batched`). When there are at least as many
   items as the width, items are split barrier-free with
   `for_each_partition` and each item runs serially. Otherwise items run in
   sequence, each with inner SPMD. One budget governs both levels. In the
   safe Rust API the items are separate `&mut` outputs, so the borrow
   checker guarantees disjointness, as in the baseline `batch.rs`.
   `TAPP_execute_batched_product` keeps its baseline sequential item loop,
   so it needs no new cross-item check.
5. **Nesting.** A broadcast from a worker of the same pool is refused before
   work starts, and execution falls back to serial with the same family.
6. **faer strategy (Phase 1).** It enters with install and derives Par from
   the granted width (Seq at width one). The batched faer loop from the
   deleted tprims-blas moves into `strategy/faer.rs`; when outer batching
   fans out, each inner faer call is Seq. Its corpus coverage is retained;
   performance parity is not asserted before measurement.

The existing Exec counters cover the tests. They check that:
- 1T makes no pool entry;
- nested calls terminate;
- concurrent full-team calls serialize;
- one plan works at different budgets without reselecting its family.

## 6. `tprims-kernel`

### 6.1 Layout

```
src/
  lib.rs
  abi/{family.rs, types.rs, element.rs, cpu.rs}
  select/{registry.rs, resolve.rs, catalog.rs, partition.rs}
  pack/{pack.rs, scatter.rs, writeback.rs}
  blocking/{probe.rs, model.rs}                 ← cache.rs (1622) split
  kernels/
    mod.rs          built-in registration and priorities
    macros.rs       ISA-generic tile bodies (was simd.rs)
    reference/{portable.rs, scalar.rs, induced.rs}
    x86/{mod.rs, avx2.rs, avx512.rs, avx2_complex.rs}
    aarch64/neon.rs
```

There is one crate for all project-owned kernels. Lukas Devos's code and the project's own share MIT OR Apache-2.0, so per-origin crates are no longer needed. Only external-dependency adapters needed separate crates, and those are deleted (D3).

### 6.2 Family IDs

The scheme is `{isa}.{dtype}.{scheme}.{MR}x{NR}` for built-ins:

| Field | Values |
|---|---|
| isa | `ref`, `avx2`, `avx512`, `neon`; the full required CPU-feature mask stays in the descriptor |
| dtype | storage dtype `f32`/`f64`/`c32`/`c64` |
| scheme | `real`/`real-scalar` for real scratch families; for complex, `planar`, `native`, `1m`, `3m`, `4m`; `i1m`/`i4m` with a base-implementation discriminator where needed; `direct`/`direct-b` for packed-B/direct-B direct-C |

Storage dtype stays explicit. At fbc83f5 the descriptor adapter already uses
c32/c64 for complex `tc.*` IDs; the old real-type kernel names are not the
registered family IDs. Do not invent a f64-to-c64 rename for those descriptors.

| Old | New |
|---|---|
| `tc.avx2.f64.8x6` | `avx2.f64.real.8x6` |
| `tc.avx2.c64.planar.4x5` | `avx2.c64.planar.4x5` |
| `cplx.avx2.c64.native.4x4` | `avx2.c64.native.4x4` |
| `portable.c64.native.4x4` | `ref.c64.native.4x4` |
| `portable.f64.4x4` | `ref.f64.real.4x4` |
| `tc.scalar.f64.4x4` | `ref.f64.real-scalar.4x4` |
| `portable.f64.4x4.direct` | `ref.f64.direct.4x4` |
| `portable.f64.4x4.direct-b` | `ref.f64.direct-b.4x4` |
| `portable.f64.4x4.1m-induced` | `ref.c64.i1m.2x4` |
| `tc.scalar.f64.4x4.1m-induced` | `ref.c64.i1m-scalar.2x4` |

Provenance is carried by Origin metadata, not by the ID. Keep origin project
and author distinct from the new owning crate's name. The built-in scheme
must remain injective: two descriptors differing in packing, update mode or
feature requirements cannot share an ID. The baseline has both portable and
tc.scalar real 4x4 families, plus direct/direct-B variants; the scheme
discriminators above resolve these actual collisions and carry through to
their induced families (`i4m-scalar` analogously). MR/NR name the descriptor's
logical complex tile, so induced 1m uses half its base real MR. Induced
families are derived from every real family, registered external ones
included (`registry.rs:97-105`). Induced families of non-built-in bases keep
the baseline `<base>.1m-induced` / `.4m-induced` suffix. In built-in IDs,
`4m` appears only as `i4m`, and `native` comes only from the portable and
cplx sources (the tc menus emit planar, 1m and 3m). Audit the full descriptor
inventory before renaming and preserve every remaining descriptor. External catalog
IDs remain provider-owned opaque names; do not force third-party names into
the built-in namespace. The full per-target ID list is pinned by snapshot
tests, resolution rejects duplicates, and the migration guide contains the
complete old/new mapping (including induced families).

### 6.3 Selection

`Auto` is unchanged in Phase 1: among CPU-available families in descending priority, it takes the first with `allow_auto`. ID selection, the catalog/selector (#28) and DynamicTiles (#29) are kept.

Keep priorities, tie order and allow_auto flags from the baseline, including
the opt-in native c32/c64 families from #30 and 3m. CPU/dtype/capability
checks apply to forced IDs and external selections before an ISA callback is
called. Built-ins are present deterministically without any caller registration
step. Explicit unsafe external registration stays idempotent for the same
manifest. As in the baseline, duplicate IDs are rejected at catalog
construction and at resolution, not at registration. New plans snapshot the available catalog;
later registration cannot mutate an existing plan. Remove environment-derived
process-default resolution caches; immutable CPU/cache-topology probes may
still be shared, but every plan's choices come from its explicit config.

### 6.4 Configuration instead of environment variables

Every `TENSORCONTRACT_*` and `TPRIMS_GEMM_*` variable is deleted. The knobs become `PlanConfig` fields:

- kernel
- complex method
- blocking override (mc/kc/nc and percentages)
- orientation
- row block
- partition
- cache-model override (block model, L3 domains)
- write-back mode

Benchmarks parse env/flags into `PlanConfig`.

The prohibition is on **runtime library policy**: no `std::env::var`/OS env
lookup to decide execution, selection or tuning. Cargo/build-script variables
and benchmark-only provider setup are not library runtime configuration.
Benchmark parsers reject invalid/removed knobs rather than silently ignoring
them; do not reintroduce an engine env variable as an adapter.

Use typed fields with documented baseline defaults. Unset optional fields
preserve the environment-unset baseline; an explicitly set partition is
distinguishable from the packed default. Explicit mc/kc/nc sizes and percentage
overrides are mutually exclusive for the same dimension. Validate positivity,
finite percentages/costs, tile alignment, checked byte footprints and supported
method/write-back combinations. Forced ID plus selector, or conflicting
method/row-block requests, is Config/Select before output writes. New custom
families are not retargeted to another geometry behind the caller's back.
Cache model overrides are inputs to one blocking calculation; reports include
the effective values and their sources. No hidden reblocking at execute time.

## 7. `tprims-capi`, `tprims-testkit`, `tprims-bench`

- **`tprims-capi`.** Preserve #26's TAPP/core ABI, except for the removals
  listed in §7.1. The library `blas` feature is gone.
  - Headers in `include/`: the pinned upstream TAPP headers (BSD-3) and `tprims/{tprims,core,tapp_ext}.h`. Lukas Devos's second `tapp.h` is deleted.
  - `examples/c-consumer`, the pkg-config template and `install.sh` are adapted to `libtprims`.
  - The tapp `lib.rs` (1239 lines) is split by ownership: `tensor_info.rs`
    owns info handles/getters/setters, `product.rs` owns metadata snapshots
    and the TAPP-to-Labels bridge, `execute.rs` owns single/batch pointer
    adaptation, and `executor.rs` owns executor lifetime. Shared status/error,
    DLPack and handle/attribute functions have their own existing core modules.
    The root is exports and wiring. There is no second contraction
    `validate.rs`: role/layout/alias validation belongs to contract; capi
    validates raw ABI arguments and maps their errors.
- **`tprims-testkit`.**
  - One label-based reference oracle (from `tensorcontract::reference`),
    used by tests needing a general oracle. It independently enumerates the
    original labels, diagonals and reductions and does not call production
    lowering/scatter/selection. Keep small known-value and algebraic tests
    as independent checks; do not replace them with comparisons between two
    users of the same production lowering.
  - Seeded generators (rand_chacha).
  - The naive `ContractionBackend`.
- **`tprims-bench`.**
  - Contract benchmarks driven by the corpora: `tenferro-p1`, `hadamard`, `example`, and `tenferro-p1-gemm`/`large-batched-gemm` rewritten as contractions.
  - `exec_entry`, `capi_rust` and `tcbench`.
  - The blas and linalg benchmarks are deleted.

### 7.1 Preserved ABI

"Unchanged" covers the pinned TAPP prototypes, numeric enum/status values,
handle ownership and supported semantics from #26. The intentional ABI break
is deletion of `tprims_blas_*` and the blas header/feature, with
`tprims_has_part("blas") == 0`. TAPP remains present in the final default
library; no implicit feature disables it. Keep `tprims_tensor_borrow_*`,
DLPack structs, `tprims_last_error`, ABI-version and part-query helpers while
making clear that standard TAPP consumes raw pointers/tensor-info metadata,
not DLPack operands. This does not add a new DLPack contraction entry point.
Update the umbrella header and installation manifest so neither includes or
installs `tprims/blas.h`. Pin the final export allowlist with nm and C/C++
consumers; no removed crate name should survive in build/install commands.

Tensor products snapshot info metadata, hold no data pointer or executor,
and survive mutation or destruction of the original infos.
`TAPP_create_tensor_product` uses the default PlanConfig. Phase 1 adds no C
tuning API; one can be added when a C caller needs non-default settings.

Preserve explicit executor ownership: executor 0 and TAPP_create_executor
are serial; Rayon creation with nthreads=0 is invalid and nthreads=1 creates
no workers. Budget updates are snapshotted once per call. BUSY or
WOULD_DEADLOCK on destruction leave the handle live; successful destruction
joins all workers, including TLS teardown, and failed creation joins any
workers already started. Every exported function contains the panic boundary;
no unwind crosses C, and thread-local error-message lifetime stays documented.
Arbitrary/stale raw handles remain an unsafe caller-contract violation, not
a promise that casts can validate their liveness.

Preserve TAPP's C rules separately from Rust's explicit accumulation source:
C metadata is required at product creation; null C data is allowed only at
beta=0. Null C with beta!=0 is Unsupported, even on an empty output. Explicit
C=D uses matching logical addresses; separate nonoverlapping C/D may have
different layouts. Keep existing null A/B/D rules, storage-precision checks,
op_D semantics, initialized status output and repeated-label behavior.
The batched adapter keeps its baseline behaviour: it validates every item,
then runs the items in sequence.

### 7.2 Benchmark baselines and features

tcbench does not invoke the historical scripts to execute its Rust commands.
Its TBLIS and BLAS/TTGT comparison code is **benchmark-only** and may remain
behind optional `tblis`/`tblis13`, `blas` and `accelerate` features. D3 removes
the library BLAS API, not external numerical references. The feature-free
build links no system TBLIS/OpenBLAS. Preserve the version-dependent TBLIS
dtype-tag self-check, provider version/build reporting and externally pinned
provider threads. Build-script provider variables stay in the benchmark
package. Keep the TCCG corpus, stride-stress modes and correctness commands;
translate old method/selection knobs to explicit PlanConfig before timing.
Name feature-gated reference cases separately from tprims strategies.

## 8. Documents and assets

- `tensorprimitives/docs/` (design, decisions, refuted, measurement-rules, notebook) moves to `docs/archive/tensorprimitives/`.
- **README.md is rewritten from scratch in PR 4.** The current one describes
  the pre-integration product: BLAS and linear algebra parts, the
  per-part C ABI, `tensorprimitives/` and the strided `-perm`/`-kernel`
  dependencies.
  - **Content:** what tprims is (binary tensor contraction with explicit
    execution and a TAPP C ABI), the crate set and dependency graph (§3),
    a minimal Rust example (DotGeneral → Plan → execute), a minimal C/TAPP
    example, kernel selection and IDs, build and test commands for Linux
    and macOS, the status (Phase 1, faer still internal, Phase 2 goal), and
    provenance/authorship (Lukas Devos) with licenses.
  - **Interim PRs:** PRs 1–3 only remove or correct statements their own
    deletions make false.
  - **Checks:** the examples are compiled (README doctest or an example
    crate), and the dependency diagram is checked against `cargo tree`
    (REPOSITORY_RULES, Public Surface Drift).
- Root `docs/` (architecture, decision-log, provenance, design-principles, research-map) is rewritten for the new structure. A new `docs/migration-2026-10.md` maps old paths and names to new ones for tenferro.
- Deleted: `RELEASING.md` (crates.io/JLL/Julia procedure, obsolete by D1), `CHANGELOG.md`, `CONTRIBUTING.md`, `tensorprimitives/CLAUDE.md`.
- Removed from the tree: **`tensorprimitives/bench-results/`** (about 2400
  files) and **`tensorprimitives/scripts/`** (35 sbatch/analysis files),
  retaining history. This does not delete root `scripts/` (CI checks) or
  `benchmarks/scripts/` (current measurement support).
- When the deletion lands, record the parent commit's full SHA in
  `docs/provenance.md`. Archived docs link to that pinned commit, not to
  the deleted paths.
- Scripts that the supported tcbench workflow needs move to `benchmarks/`
  and are adapted to the new CLI. Historical experiment scripts are not
  kept.
- Add root `LICENSE-MIT` / `LICENSE-APACHE`, with copyright for Lukas Devos and tensor4all contributors.

## 9. History and authorship

- Moves use `git mv`, in commits separate from content edits, so `git log --follow` works.
- Files derived from tensorprimitives-rs keep their source/author headers.
- The `authors` of `tprims-kernel`, `tprims-contract` and `tprims-capi` include Lukas Devos.
- Commits that port his code carry `Co-authored-by: Lukas Devos <ldevos@flatironinstitute.org>`.
- Preserve imported ancestor commits with ordinary merge commits; do not
  squash or rewrite the subtree's imported history. A pure move commit and
  later responsibility splits retain derivation notices and license texts;
  git mv alone does not guarantee follow across a split. The move manifest
  records old/new paths and commits for those cases. Preserve BSD-3 notices
  for the pinned TAPP headers and Apache-2.0 notices for DLPack. Testkit also
  credits Lukas Devos for its moved oracle.

## 10. Phase 2 (separate spec, summary only)

Superseded on 2026-10-03 by the [Phase 2 spec](2026-10-03-phase2-faer-routing-and-packed-driver-design.md). faer is kept for every GEMM-fusable contraction. Phase 2 widens faer's coverage and optimizes the packed driver for the contractions it owns.

## 11. Delivery and verification (Phase 1)

There are four PRs in dependency order. Each is green on its current
workspace's local gate before it is pushed, with no manifest referring to a
deleted package. The order below is ownership order, not permission to leave
intermediate clients broken.

1. **exec consolidation.** First delete the crates that have no retained
   consumer:
   - `tprims-linalg`, with its benchmark and the dependency-direction test
     reference;
   - `tprims-blas-capi`, with the bundle `blas` feature, the BLAS C symbols
     and `blas.h`;
   - `tprims-kernel-gemm` and `tprims-kernel-pgx86`, with the matching
     tprims-blas features and engine variants, `register_known_prefix`,
     `SelectError::NotBuilt` and the bench `pgx86` feature.

   This avoids renaming code that is about to go. Then move workspace and its tests from gemm-kernel;
   remove exec's dependency on gemm-kernel and adapt **all** old driver,
   bridge and pool-workspace consumers to the exec-owned byte API in this
   PR. Replace Spmd and every own-thread fallback: the driver's
   `try_broadcast`/pool/`std::thread::scope` (driver.rs:1043-1066) and
   `contract_batched_with_threads` (batch.rs:174), which now takes `&Exec`.
   tensorcontract and tensorprimitives-tapp gain a normal dependency on
   tprims-exec. The existing library/bench/C callers still build on the
   interim crate set.
2. **kernel consolidation.** Create tprims-kernel, move the provider bodies,
   rename IDs and migrate imports/manifests/ID fixtures in every consumer,
   including the temporarily retained tensorcontract and TAPP crates.
   Removing kernel env reads requires adapting old planners and benchmark
   parsers to explicit config in the same PR; old helpers cannot depend on
   a removed from_env function. Preserve family contract, ISA and opt-in tests.
3. **contract consolidation.** Problem, front ends, plan, errors, driver and
   strategies. Rebase the retained TAPP wrapper onto the new Plan and move
   the oracle/second-backend support needed by the ported tests now, rather
   than delaying it to PR 4. Delete blas/linalg/pgx86/kernel-gemm and
   contract-traits only after migrating/removing their consumers, including
   bundle's blas dependency/feature, BLAS C symbols and obsolete benchmark
   targets. Trim CI commands for those targets in the same PR.
   - **tcbench port.** tcbench moves to `tprims-bench` in this PR and keeps
     only the corpus run, `verify`, `info` and `--stress`. The research
     subcommands `sweep`, `shapes`, `orient` and `premise` are deleted. They
     use tensorcontract internals (`Scatters`, `run_structure`, `Operand`,
     `plan_config`) that contract does not export, and no internals module
     is added for them.
   - **tensorcontract.** The crate is deleted in this PR, with no
     compatibility shim.
   - **GemmConfig.** `blas/engine.rs` `GemmConfig` (kernel, method,
     partition, partition_opts) becomes contract's `PlanConfig`.
     `Engine`/`EngineChoice` and `SelectError::EngineUnsupported` are
     dropped; the equivalent refusals become `Error::Config`.
4. **capi, testkit and bench completion.** Merge core/TAPP/bundle into
   tprims-capi, move the remaining testkit support, then remove all
   remaining tensorprimitives packages, docs/assets and the subtree. Finish
   headers, install/pkg-config, licenses and migration
   docs. Rewrite README.md from scratch (§8). Replace all old package/feature
   paths in CI, AGENTS.md, REPOSITORY_RULES.md, rustdoc/examples and bundled
   usage skills.

**CI platforms.** Hosted CI runs on Linux x86_64 and macOS arm64 for every
PR (macOS only daily and on manual `workflow_dispatch` since 2026-11-23,
#218). This is a prerequisite: the matrix lands on `main` before PR 1.
- **Both OSes:** clippy, workspace tests, release tests of the packed
  driver/kernels, and the C/C++ ABI consumers. The ABI consumers use
  `libtprims.so`/`nm -D` on Linux and `libtprims.dylib`/`nm -gU` on macOS.
- **Linux only:** fmt, scripts, MSRV and docs.
- **Platform-specific code:** each PR keeps x86-only families and tests
  platform-correct. Descriptors that exist only on x86 report themselves
  unavailable or unknown on aarch64, and the tests assert exactly that.
  NEON kernels run their numerical tests on the macOS lane.
- **Coverage claims:** a passing local Linux gate does not substitute for
  the macOS lane.

Implementation is mostly one sequential lane. testkit/bench and the docs archive can run in parallel. PR 3 is internally sequential: problem and front ends, then the driver move, then the strategies.

Each PR description lists the tests it removes, each with the deleted
feature it belonged to. A test is never removed merely because it fails.

Correctness and structural acceptance (reuse/port existing tests; do not
expand the bullets into a Cartesian product of configurations):

- every numerical test of a kept feature is ported and passes across all four
  dtypes: TAPP cases 1–4, output diagonal preservation, isolated reductions,
  arbitrary output order, separate C/D, all conjugations and zero/scalar cases;
- Labels/DotGeneral/role inputs agree where equivalent; static/dynamic,
  direct/scratch, elementwise/faer/packed and trait/concrete paths agree with
  an independent oracle or known values. Force packed families via IDs for
  coverage without adding a permanent public strategy enum;
- negative/offset/broadcast input layouts, invalid shapes/strides/labels,
  arithmetic overflow, noninjective output, stale execution layouts and
  single/whole-batch alias failures are tested, including at alpha/beta zero
  and empty size. Whole-batch failures preserve all outputs;
- 1T and 4T **correctness/control tests** cover partial budgets, nesting,
  concurrent plans, repeated execution and executor/workspace cleanup. These
  are not heavy paired throughput experiments;
- allocation/copy counters check no full-operand copy, trait/concrete parity,
  steady-state workspace reuse and bounded retention/clear/reentry behavior;
- tests of deleted features are removed together with the feature;
- the local gate passes: fmt, both clippy configurations with `-D warnings`,
  workspace tests, release tests of packed-driver/kernel contracts, runnable
  doctests and docs `-D warnings`, MSRV 1.89, C/C++ ABI consumers, the nm
  export allowlist and the scripts job. Build the current library artifact
  before ABI tests (`tprims-bundle` during transition, `tprims-capi` finally).
  Every compiling Cargo command uses `-j 16` (fmt does not accept a job flag).
  Test exec with/without its strided feature separately; remove stale
  strided workspace-only feature invocations. The final crates use std;
  old tensorcontract no-std/package-list lanes disappear with that package,
  not by weakening coverage while it is still supported;
- benchmark rows cover retained public operations (including add/permute),
  kernel paths and alternative strategies at 1T/4T. A missing comparison
  feature is explicit, not silently treated as a zero-time row;
- **Smoke run.** It happens once, at Phase 1 completion: release mode, 1T,
  on Linux. It covers `tenferro-p1`, `hadamard`, `tenferro-p1-gemm` and
  `large-batched-gemm`, against `fbc83f5`. A case more than 2x slower is
  investigated. Nothing else is benchmarked in Phase 1, and the smoke run
  makes no parity claim.

### 11.1 tenferro migration boundary

The migration guide must distinguish a rename from a removed capability:

| Old dependency/API | Phase 1 destination or disposition |
|---|---|
| Exec/Pool | tprims-exec; borrowed host pool, one shared wrapper (§5.1) |
| ContractPlan/DotGeneral/Flags | Plan/DotGeneral/PlanConfig and explicit accumulation source |
| gemm | DotGeneral or Labels with one K pair; preserve op/scaling/layout semantics |
| gemm_batched | batch axes or a validated independent item batch; document the distinction |
| gemm_grouped | per-shape plans, grouped by the host; no claim that one fixed-layout plan accepts heterogeneous items |
| trsm | removed; no contraction replacement |
| tprims-linalg decompositions/solves | removed; the consumer chooses another provider separately |
| whole-backend trait crate | tprims-contract::api; updated object-safe factory/prepared-plan contract |
| kernel names | complete per-target ID rename table, with old names rejected |
| explicit kernel/partition on an all-batch problem | **behaviour change:** the baseline rejects it (`EngineUnsupported`, plan.rs:266-280); Phase 1 runs the packed driver (D5). Tested with StaticGrid and DynamicTiles on an all-batch problem |

The pinned `ext/tenferro-cpu-tprims` consumer stays on its old revision until
its separate migration is complete; this repository cannot claim a drop-in
rev bump. Update #31's acceptance language for the new seam, and record that
trsm/linalg migration requires downstream provider work. No shims or new
replacement numerical operations are introduced here.

## 12. Issues

- **#22.** Add a note that the linalg split is obsolete (linalg is deleted); the repository rename remains.
- **#31.** Slice 4 (the tenferro bridge) stays open and is rebased onto the new API.
- **#37.** Tracking issue for this spec. Phase 2 gets its own issue when its
  spec is written. Keep this draft and the issue's readiness status explicit;
  no automatic implementation/merge authorization follows from this audit.

## 13. Out of scope

- The tenferro update.
- Phase 2 optimization.
- New kernels.
- Repository rename.
- Publishing.
