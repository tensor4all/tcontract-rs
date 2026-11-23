# Source integration: detailed Phase 1 design

Status: **accepted**, 2026-10-02 (after the Fable 5.1 review). Baseline: `fbc83f5`.
Tracking issue: [#37](https://github.com/tensor4all/tprims-rs/issues/37).
The [overview spec](2026-10-02-source-integration-design.md) owns scope,
maintainer decisions D1–D10 and acceptance. This document fixes the concrete
interfaces and source-move boundaries; it does not request implementation.

Keep the refactor small: reuse the existing descriptor, scatter, packing,
partition and workspace algorithms. Change their owners and inputs. No new
plugin framework, handle-liveness registry, exclusive host-pool borrow,
transactional rollback, cancellation API or optimization campaign is needed.

## 1. Ownership and dependency graph

| Owner | Owns | Must not own |
|---|---|---|
| exec | borrowed/owned pool, budget, entry/broadcast gate, byte workspace | dtype, kernel family, contraction labels |
| kernel | scalar/ISA leaf contracts, immutable families, catalog, packing/writeback, hardware/blocking helpers | threads, pool/workspace ownership, frontend labels |
| contract::api | validated problem metadata, front ends, common errors and backend traits | kernel loops, C handles, faer types |
| contract::plan | role folding/orientation, offsets, family/strategy resolution, fixed report | input buffers, executor lifetime |
| contract::driver | joins an existing plan, Exec and workspace; numerical execution | environment reads, kernel reselection, thread creation |
| capi | raw arguments/handles, status mapping, executor lifetime, installation ABI | another label/layout interpretation or contraction scheduler |
| testkit | independent label oracle, seeded fixtures, second backend | production fallback or runtime dependency of contract |
| bench | corpus adapters, timing boundaries, external-reference features | library tuning defaults or runtime provider registry |

Normal edges are contract → exec/kernel, capi → contract/exec, testkit →
contract, bench → the used packages. Exec and kernel do not depend on one
another. Any contract → testkit edge is path-only and dev-only. A test checks
the graph once after consolidation, rather than maintaining another package
graph by hand. Strided dependencies retain the existing pinned revision.

## 2. Public metadata and lowering

The following are proposed interfaces, not runnable examples of today's API.
Public structs have private fields, checked constructors, Debug and read-only
accessors. Scalar is the common sealed storage vocabulary for f32/f64/c32/c64;
the public backend trait does not require a KernelFamily or faer bound.

| Type | Data/contract |
|---|---|
| DType | F32, F64, C32, C64; one storage/accumulation type per problem |
| LayoutSpec | dims `Vec<usize>`, signed element strides `Vec<isize>`, logical offset `isize` |
| OperandSpec | LayoutSpec plus Identity/Conjugate |
| Labels | original per-operand `Vec<i64>` labels; optional separate C labels |
| DotGeneral | paired lhs/rhs contraction and batch axis lists |
| CSpec | Absent (overwrite), Output with op_C, or separately described C |
| Problem | original operand metadata plus normalized M/N/K/H role axes and C mode |
| RoleAxis | extent and signed A/B/C/D strides; missing input has stride zero |

Construction contracts:

```text
Problem::from_labels(dtype, a, b, c_spec, d, labels) -> Result<Problem>
Problem::from_dot_general(dtype, a, b, d, dot_general) -> Result<Problem>
```

- `c_spec` and `Labels.c` go together. `CSpec::Separate` requires
  `Labels.c`; `CSpec::Absent` and `CSpec::Output` forbid it.
- `Plan::<T>::new` with `T` different from `problem.dtype()` is
  `Error::Config`.

DotGeneral lowers to CSpec::Output with identity op_C; free axis order is
lhs_free, rhs_free, then the supplied paired batch order. Labels keeps the
specified output order and can describe distinct C/D. Both use one lowering
function, whose result carries enough read-only role metadata for a second
backend; public callers cannot forge a validated Problem by editing fields.

Lowering proceeds once:

1. Check rank/label counts, axis bounds/disjointness and matching extents.
2. For each labeled operand, collapse repeated labels to a diagonal by checked
   stride addition. Reject unequal repeated extents. Preserve original
   metadata for buffer checks and leave off-diagonal output elements untouched.
3. Classify unique labels as M, N, K or H. A-only/B-only labels enter K with
   the opposite input stride zero. C/D label sets must agree. Output-only
   labels are Unsupported.
4. Check signed address ranges, products and logical D injectivity. Store
   checked span endpoints and whether C can address D identically.
5. Expose normalized roles. Plan construction, not Problem construction,
   chooses ordering/folding/orientation and builds scatter offsets.

An empty role list has extent-product 1; a zero extent has product 0. Compute
shape/byte/scatter products before allocation, with conversions checked at
their consuming boundary. Do not reject negative input strides merely for
their sign. Reuse the baseline conservative injectivity/span algorithms;
exact arbitrary-stride set intersection is not added.

## 3. PlanConfig and typed selection

PlanConfig is owned, dtype-independent and contains no callback. All unset
tuning preserves the baseline with environment overrides unset.

| Field | Type/default | Resolution |
|---|---|---|
| kernel | KernelChoice::Auto or Id(String); Auto | resolve once after strategy selection |
| partition | Option<PartitionPolicy>; None | explicit Some forces packed; None means packed StaticGrid when packed is selected |
| no_materialize | bool; false | reject full-operand normalization/copy-back, not bounded packing |
| method | Option<tprims_kernel::Method> (Native/OneM/ThreeM/FourM, the descriptor vocabulary GemmConfig uses); None | baseline default; a forced family must agree. The planner's 3-variant ComplexMethod becomes internal |
| blocking | optional absolute/percentage overrides per dimension | mutually exclusive for one dimension; check tile alignment/bytes |
| orientation | optional baseline orientation choice | affects role orientation once, not tensor storage |
| row_block | optional baseline menu request | resolve against the selected descriptor; no silent geometry change |
| cache_model | optional BlockModel, hierarchy/L3-domain hints and KC coupling | invoke existing blocking model with explicit inputs |
| writeback | optional baseline mode | validate against family capabilities; keep direct/scratch fallback |

Preserve the existing config enum vocabulary instead of adding stringly typed
options to the library. Benchmark flags/env are parsed before Plan construction.
Bad options fail rather than becoming an implicit default. Runtime library
environment helpers and their cached defaults are removed; immutable hardware
probes remain useful inputs to the blocking model.

Construction:

```text
Plan<T>::new(&Problem, &PlanConfig) -> Result<Plan<T>>
Plan<T>::new_with_selector(&Problem, &PlanConfig,
                          &KernelCatalog<T>, &mut selector) -> Result<Plan<T>>
```

The second constructor is the #28 typed selection seam, not a new framework.
The selector sees the validated, oriented shape and the explicit method. It is
called at most once, and returns a catalog handle or a typed error.
ID plus selector is Config/Select. Store the handle, not the callback or a
borrow of the temporary catalog. Keep the current static-descriptor lifetime
contract; no dynamic kernel-loader lifetime model is introduced.

Plan construction performs:

```text
validated Problem + config
    -> explicit packed requirement?
    -> else all-batch elementwise that implements the full op_C/op_D/separate-C semantics?
    -> else full-semantics, copy-free faer fusion?
    -> else packed
packed -> orient/fold -> build scatter -> resolve family/blocking/partition
       -> immutable prepared plan/report
```

An explicit kernel, partition or packed tuning request bypasses faer and
elementwise; it is honored or refused. The faer path must implement distinct
C/D and output conjugation without full copies, or decline to packed. Remove
its old copying branches. Runtime pointers, alpha and beta may choose an
existing direct-B/direct-C applicability fallback, not another family.

## 4. Prepared plan and execution surface

The private plan representation is one immutable metadata record plus one
strategy payload:

```text
Plan<T>
  metadata: original layouts, normalized roles, ops, C mode, checked spans
  strategy: Packed(PackedPlan<T>) | Faer(FaerPlan) | Elementwise(ElementPlan)
  report: immutable PlanReport
  serial_workspace: ArenaProvider

PackedPlan<T>
  scatter: A_M/A_K, B_K/B_N, C_M/C_N, D_M/D_N, H_A/H_B/H_C/H_D
  family: KernelHandle<T>
  blocking, oriented roles, resolved partition policy, work estimate
```

Strategy is private. PlanReport has a stable algorithm name and optional
family ID, blocking/partition, explicit materialization and estimated scratch.
Non-packed reports do not invent a family. The common trait's Diagnostics
only requires backend/algorithm identity and materialization information;
other backends need not report MR/NR.

Safe execution contracts:

```text
execute_into(exec, alpha, a, b, d) -> Result<()>
execute_into_accum(exec, alpha, a, b, beta, source, d) -> Result<()>
source = AccumulationSource::Output | AccumulationSource::Separate(c_view)
```

Inputs are borrowed StridedView values; D is a borrowed StridedViewMut.
Separate borrows a C view, without owning/copying payload. Output reads through
the mutable D view, so safe callers never create an aliasing immutable C view
just to accumulate in D. Overwrite fixes beta=0 and reads no old output.
Accumulation requires prepared C metadata; Output is accepted only when that
metadata maps to D. Raw TAPP execution uses pointers directly after the same
semantic preflight, without manufacturing aliasing Rust references.

One operation boundary checks layout compatibility, slice/range bounds and
actual pointer spans; then it creates a private validated execution record.
Hot loops receive that record and precomputed offsets. Do not revalidate per
tile, rehash labels or reconstruct shape roles. A plan has no pointer to an
operand and works with different buffers and Exec budgets. Concurrent calls
have independent outputs and workspace leases; no mutable scheduling state
is retained in the immutable plan.

After preflight, output-empty returns; alpha=0/K=0 uses the output-update
helper; otherwise dispatch once to the prepared strategy. Beta=0 never reads
C/D. Output conjugation applies after the complete alpha/beta expression.
The first KC contribution uses beta*C and later contributions accumulate
into D, preserving op_D correctly; reuse the baseline writeback contract.
Input/configuration errors leave D untouched. A foreign-backend failure
after compute starts may leave partial output; do not add rollback copies.

## 5. Whole-backend trait

Preserve #31's object-safe shape with direct Exec and the new accumulation
source; only concrete tprims planning exposes typed kernel selection:

```text
ContractionBackend<T>: Debug + Send + Sync
  id() -> &'static str
  prepare(&Problem, &Requirements, &PlanningBudget) -> Result<BoxedPlan<T>>

PreparedContraction<T>: Send + Sync
  execute_into(&Exec, alpha, a, b, d) -> Result<()>
  execute_into_accum(&Exec, alpha, a, b, beta, source, d) -> Result<()>
  diagnostics() -> &Diagnostics

BoxedPlan<T> = Box<dyn PreparedContraction<T> + Send + Sync>
```

Requirements contains common no-materialize policy. TprimsBackend owns
PlanConfig; preparation applies Requirements without mutating the factory.
Effective no_materialize is the OR of the config and requirement flags.
PlanningBudget stays advisory, as in #31. Plan<T> implements
PreparedContraction, not the backend factory.
The box is allocated once at prepare. No generic trait method prevents object
safety and there is no per-tile virtual call.

Testkit's second backend uses read-only Problem roles and Exec directly. The
label reference oracle takes the original test case, not production scatter
or lowering output; keep known-value tests to check shared metadata semantics.
The trait is inside contract::api by maintainer decision: it is an extension
seam in this package, not a claim of dependency-isolated implementation choice.

## 6. Scheduling and workspace

Preserve Pool::borrow(&ThreadPool), Pool::owned and borrowed Exec. Document
one shared Pool wrapper per raw pool, since its SPMD gate is owner-local.
No global wrapper registry or exclusive mutable host-pool borrow is added.

Execution chooses the actual width once, from the work estimate and the
current budget, with the existing WidthPolicy and one internal ns-per-flop
default. A plan never reselects its family for a different budget. Only the
scheduling and the byte workspace requirements are rederived from the width.

| Situation | Route |
|---|---|
| width 1 / below threshold | caller thread; no pool entry |
| width > 1 | baseline gated broadcast, existing shared-panel StaticGrid/DynamicTiles |
| already on the same pool's worker | serial same-family fallback; no barrier gate wait |
| many independent items, item_count >= chosen width | outer partitions, inner serial |
| fewer items | sequential items using their allowed inner width |

Full-team callbacks do not allocate, validate or invoke fallible user code
between barriers. A refused broadcast runs no callback.

contract_batched has one fixed-layout plan and independent item buffers,
which are separate `&mut` outputs. As in the baseline, the borrow checker
guarantees their disjointness. Heterogeneous shapes use per-shape plans
grouped by the host. TAPP batched execution keeps its sequential item loop.

Move ArenaProvider/WorkspaceReq/TeamLease and tests to exec. Requests stay
dtype-independent: byte counts/alignment, scatter capacities and barrier
count. Contract computes requests from the resolved family. Retain owner-local
worker buffers, exclusive team leases, trim and retained-byte accounting;
use a guard to clear a worker's borrowed state on unwind. Reentry takes
call-local buffers rather than waiting for itself.


## 7. Kernel consolidation and identifiers

Move existing leaf bodies/descriptors with provenance intact. Keep priority,
allow_auto, CPU masks, packing/tile geometry and direct update capabilities.
External catalog names remain opaque. Built-ins are deterministically present
before a plan resolves a name; remove duplicate bridge registration calls and
environment-derived process-default caches. Keep explicit unsafe external
registration with its current immutable-descriptor obligations.

Generate names from **logical** descriptor geometry and storage dtype:

```text
{isa}.{dtype}.{scheme}.{logical_mr}x{logical_nr}
```

Portable real 4x4 becomes ref.f64.real.4x4; the distinct scalar implementation
becomes ref.f64.real-scalar.4x4. Direct-C variants use direct/direct-b, so
packed-B and direct-B remain distinct. Induced 1m halves base real MR:
portable.f64.4x4.1m-induced → ref.c64.i1m.2x4; scalar-derived uses i1m-scalar.
Induced 4m keeps geometry and uses i4m/i4m-scalar. Native/planar/1m/3m/4m
descriptor schemes remain distinct. f32/c32 follow the same rule.

The complete per-target manifest is snapshotted when implemented; each old
retained descriptor maps to one new ID. Renaming must not silently delete a
collision or change Auto ordering. Validate ISA/capability before calling any
leaf, including an explicitly chosen external handle. No new kernel bodies
or measured cost metadata are required in Phase 1.

## 8. C adapter and status boundary

Keep standard upstream TAPP headers verbatim. Product creation snapshots infos
and lowers Labels once; execution borrows the chosen executor and invokes the
same Plan. A product can outlive its source info handles; a call cannot race
destruction of its product/executor. Retain existing BUSY/WOULD_DEADLOCK,
budget snapshots, join/TLS cleanup and thread-local error-message contracts.

TAPP null C at beta!=0 remains Unsupported. Standard calls use default
PlanConfig; in-place C=D is recognized by logical mapping, not pointer equality
alone. Batched execution validates every item, then runs the items in sequence, as in the baseline. DLPack borrow
helpers remain available but do not become a new TAPP operand form.

Phase 1 adds no C tuning API: standard TAPP calls use the default
PlanConfig. One can be added when a C caller needs it.

| Error reason | Existing C status |
|---|---|
| Config / bad axis lists | INVALID_ARGUMENT |
| Shape/extent/checked-size mismatch | SHAPE |
| label/rank count or C/D label-set mismatch | LABELS |
| Layout range incompatibility | SHAPE (matching baseline adapter) |
| overlapping/noninjective output | ALIASED |
| refused full materialization | WOULD_MATERIALIZE |
| unsupported dtype / storage mismatch | DTYPE |
| output-only label, unsupported operation/precision/family capability | UNSUPPORTED |
| unknown/incompatible forced kernel | UNSUPPORTED; malformed descriptor remains INTERNAL |
| executor unavailable/deadlock/lifecycle | existing exec-specific status, without a second table |
| foreign backend/internal invariant | INTERNAL with preserved diagnostic source |
| caught Rust panic | PANIC |

Keep FFI-only null/device/read-only statuses in capi. Preserve #26's numeric
values and error source context; the exhaustive mapping has focused tests.
Export checking verifies TAPP/core/extensions remain and tprims_blas_* does
not. Remove blas.h from the umbrella header and installation manifest.

## 9. Source moves and responsibility splits

| Source | Destination / split |
|---|---|
| gemm-kernel/workspace.rs | exec/workspace; lease/arena helpers and their current tests |
| tensorcontract/spmd.rs and pool.rs | remove after all callers use Exec; keep no own-thread fallback |
| gemm-kernel + kernel-tensorcontract + kernel-cplx | kernel/{abi,select,pack,blocking,kernels}; no per-origin crate |
| tensorcontract/plan.rs | contract/plan analysis (classification/folding), orientation, report and mod wiring |
| tensorcontract/driver.rs + driver/dynamic.rs | contract/driver static_grid, dynamic and tile/update helpers with one shared validated context |
| gemm-kernel/cache.rs | kernel/blocking/probe (hardware facts) and model (pure explicit inputs); extract large tests |
| contract-traits | contract/api; drop HostExecution/NativeHost/TypeId hooks |
| contract/permute_gemm.rs + blas batched faer loop | contract/strategy/faer.rs; retain only copy-free behavior |
| blas/engine.rs GemmConfig (kernel, method, partition, partition_opts) | contract PlanConfig; Engine/EngineChoice and SelectError::EngineUnsupported dropped (the refusals become Error::Config) |
| gemm-kernel registry `register_known_prefix`, `SelectError::NotBuilt` | deleted with the gemm/pgx86 providers (PR 1) |
| tprims-custom-kernel-test | kernel and contract tests (PR 2) |
| tensorcontract scatter/layout/buffer/element | contract-owned role/offset helpers or existing kernel scalar leaf seam; one owner per definition |
| tapp/lib.rs | capi/tensor_info, product, execute; core executor/status/DLPack modules retained under capi |
| reference.rs + contract-testkit | testkit; independent fixture/oracle and trait-backend modules |
| tensorprimitives-bench | benchmarks/tcbench in PR 3: the corpus run, verify, info and `--stress`, with optional external-reference features; `sweep`/`shapes`/`orient`/`premise` are deleted |

The TAPP split is not fixed to four files. Do not create a new validate module
that copies contract's checks. Keep files coherent; extract on responsibility
boundaries, not arbitrary line counts. Pure moves use git mv before semantic
edits. Where a file splits, record old/new paths in the move worklog because
git log --follow alone cannot describe all descendants.

Remove only tensorprimitives/scripts and tensorprimitives/bench-results.
tcbench's Rust execution has no script dependency, but its supported workflow
may need build/analysis/placement helpers: keep the transitive closure of
helpers used by the final documented commands under benchmarks. Do not keep
all historical sbatch/phase experiments automatically. Record the final
keep/archive/delete manifest and recovery SHA in the removal PR; root CI
scripts and current benchmarks/scripts stay. Archived reports link to pinned
old paths; old experiment reproduction runs the old checkout/toolchain.

## 10. Four PRs and proportional verification

| PR | Completion boundary |
|---|---|
| 1 exec | workspace move, direct Exec callers and thread-source consolidation; all current clients build |
| 2 kernel | new owning crate, complete ID/import migration and explicit config inputs in current planners/harnesses |
| 3 contract | new metadata/API/driver/strategies; TAPP rebased, testkit support moved as needed; remove obsolete library/bench dependencies together |
| 4 capi/testkit/bench/docs | consolidate libtprims, install artifacts, archived docs/assets, subtree removal and a from-scratch README rewrite |

Do not add permanent re-export shims just to preserve the old product layout.
Move or delete a consumer in the same PR that removes its dependency. Bundle
and ABI tests cannot retain a blas dependency until PR 4 if blas disappears
in PR 3. tcbench is ported in PR 3, and tensorcontract is deleted in that PR.

Every PR must be green on both hosted CI platforms: Linux x86_64 and macOS
arm64 (overview §11). Since 2026-11-23 (#218) macOS runs only daily and on
manual `workflow_dispatch`, not per PR: a PR is gated on Linux, and its macOS
result comes from a dispatch on the branch or the next daily run.
An x86-only kernel or test is gated by target and has
an explicit aarch64 expectation. A local Linux run does not stand in for
the macOS lane.

Port retained tests with a short old/new-owner ledger. Add focused regression
tests for actual changed seams: diagonal/reduction lowering, C/D update,
batch dependency refusal, forced-ID collisions, reduced-budget dispatch and
workspace lease cleanup. Existing broad numerical/ISA tests keep their cases;
do not multiply them across every knob or add tests that only mirror code.
Each PR runs the applicable local correctness/build/docs gate. Fix obsolete
feature/package invocations when that surface changes; do not invent missing
features to make the old AGENTS command parse.

Keep benchmark rows for the retained strategies and public operations. Timing
happens **once at Phase 1 completion**: one sequential 1T smoke suite over the
four overview corpora, with correctness checked first and exact baseline/case
mapping recorded. There is no per-file, per-PR, per-reviewer or documentation
benchmark requirement, nor a new concrete-vs-trait timing campaign. Use
counters for entries/allocations/copies rather than deriving them from timing.
Only an observed failure justifies a focused rerun. Heavy paired 1T/4T
performance promotion belongs to Phase 2's separate spec. 1T/4T scheduling
correctness tests are cheap functional tests, not performance campaigns.

The final worklog records the completion commit, test migration, unavailable
hardware and the single smoke evidence for independent integration review.
Update tenferro's guide with the actual removed trsm/linalg capability; its
separate adapter migration retains the old pin until replacement providers
are chosen. This refactor does not claim a drop-in downstream revision bump.
