# Performance Tips

tcontract performance, layout, scratch, threading and benchmark contracts, kept
separate from `REPOSITORY_RULES.md` so they can be read on demand. These rules
apply on top of the shared `rules/rust/performance.md` and
`rules/rust/numerical.md` from `tensor4all-agent-rules`. They were adapted
from tenferro-rs `PERFORMANCE_TIPS.md` (commit `5a4e7fd`); the threading and
dense-layout contracts are rewritten for explicit `Exec` contexts and DLPack
strides.

Read this file in full:

- before implementing or reviewing performance-sensitive code: kernels,
  planning, caches, execution and threading code, benchmarks, and examples;
- before creating a PR that touches such code;
- when running a static performance audit.

The `Audit hints` under each section are for static audits: `Detect` lists
source patterns worth inspecting, `Fix` the expected direction. A hint match
is evidence to inspect, not a finding by itself.

## Audit Procedure

Human/process protocol.

1. Scope: audit the requested paths, or with `full` the crates under
   `crates/`, then
   `benchmarks/`, examples, and doc snippets. Skip
   `docs/superpowers/` and `docs/worklogs/`.
2. For each section below, search the scope for its `Detect` patterns and read
   the surrounding code. Respect an `// INVARIANT:` marker that explains the
   pattern unless the explanation is wrong.
3. Report each finding as `file:line`, the section title, the evidence, and the
   `Fix` direction. Group findings that share a root cause.
4. Findings are static rule violations. Never claim a speedup or slowdown
   without a measurement, and start any optimization through the
   Performance-Gated Experiment Protocol.
5. Cross-check the Public Boundary Safety Audits and Unsafe Code Boundary
   sections of `REPOSITORY_RULES.md` when a finding touches unsafe code.

## Performance-Sensitive Safety Contracts

- Potentially dangerous operations kept for performance (raw scratch-buffer
  acquisition, unchecked indexing after validation, raw pointer arithmetic,
  raw faer view construction) carry a nearby one-line `// INVARIANT:`
  comment (see Invariant Markers) explaining why they are valid, even when
  technically correct, so later agents and reviewers do not "fix" a false
  positive with hidden copies, repeated checks, or unconditional
  initialization in a hot path.
- Scratch-buffer APIs distinguish full-overwrite callers from
  read-before-write callers. Do not fix stale or uninitialized reads with
  unconditional zero-fill in shared hot-path acquisition; expose an explicit
  zeroed/initialized path, keep raw acquisition unsafe and documented for
  full-overwrite kernels only, and add regression coverage for both.

Audit hints:

- Detect: raw scratch acquisition, unchecked indexing, raw pointer arithmetic,
  or raw faer/DLPack view construction without a nearby `// INVARIANT:`;
  unconditional zero-fill added to a shared hot-path acquisition.
- Fix: add the marker with the proof, or route through the explicit
  zeroed/initialized path.

## Complexity Budget

- No accidental `O(n^2)` behavior in plan construction, index classification,
  scatter/fold analysis, or batch scheduling. An intentional quadratic
  algorithm documents the bound or tradeoff with an `// INVARIANT:` marker.
- Do not repeatedly clone, hash, format, or scan whole index lists, stride
  vectors or plan keys inside per-item or per-axis loops. Prefer stable IDs,
  precomputed metadata, or cached fingerprints with exact equality checks.
- When optimizing planning overhead, measure scaling across increasing rank
  and extent, not one fixed case.

Audit hints:

- Detect: clone, hash, format, `contains` on a `Vec`, or linear scan of index
  or stride lists inside per-item or per-axis loops; nested loops over axis
  counts without an `// INVARIANT:`.
- Fix: precomputed metadata, `HashMap`/`HashSet` lookups, or cached plans;
  measure scaling across sizes.

## Materialization And Copies

- Production paths must not silently materialize dense temporaries whose
  memory or time scales with an unconstrained product of tensor dimensions.
  Dense/reference implementations are allowed only when the API is explicitly
  named and documented as dense, reference, or debug behavior.
- Avoid dense copy-in/copy-out around operations that can consume strided
  views, borrowed slices, or metadata-only layout changes.
- When an output contiguity contract, provider limitation, or external ABI
  boundary requires a copy or materialization, make that boundary explicit in
  the implementation, report it as the selected strategy, and cover it with
  tests. Across the C ABI, DLPack operands are never copied merely to cross
  the boundary; a forced internal copy becomes an error under
  `TCONTRACT_NO_MATERIALIZE`.
- Audit compact, offset/strided, negative-stride, and caller-provided-output
  paths together. Allocating and `_into` operations share the output-writing
  kernel; they do not allocate a temporary result and copy it into the
  caller's destination.
- Regression checks for removed materializations must cover data movement as
  well as numerical results: use allocation/copy counters or pointer identity.
  Include compact views and a noncompact/offset case, input preservation, and
  `_into` where supported.

Audit hints:

- Detect: `to_vec`, `collect::<Vec<_>>`, or a fresh output allocation on a
  production path whose size is a product of tensor dimensions; layout
  normalization copies before a kernel that accepts strides; repeated
  canonicalization and work-buffer copies across adjacent layers.
- Fix: strided views, borrowed slices, metadata-only layout changes; make any
  required copy explicit, reported, and tested.

## Dense Layout And Linear Algebra

- Operands are strided views with signed element strides and an offset
  (`strided-view`, DLPack at the C ABI). Honour the given layout; never
  normalize to a canonical order behind the caller's back.
- Compact buffers the library allocates (packed operands, factors, scratch)
  are column-major, with compute dimensions on the left and batch dimensions
  on the right, so each batch item is contiguous.
- A kernel that needs a specific layout either consumes the strides directly
  (faer `MatRef`, TBLIS-style packing) or materializes through `strided-perm`
  explicitly, reporting it (see Materialization And Copies).
- Batched linear algebra runs as one tight loop per call: compute scratch
  requirements before execution, query provider workspace once per call,
  allocate scratch once and reuse it across the batch, and write each result
  directly into the batch output. No per-item `Vec` for pivots or
  permutations, no per-item allocation, no per-item workspace query. Small
  matrices at large batch (2x2, batch 1024) are the regression case, because
  per-matrix overhead dominates there.

Audit hints:

- Detect: layout normalization copies before a kernel that accepts strides;
  batch dimensions left of compute dimensions in library-allocated buffers;
  workspace queries, `Vec` allocation, or per-item scratch inside a batch loop.
- Fix: consume strides, hoist scratch and workspace queries out of the batch
  loop, write in place.

## Range Checks And Slicing

- Public indexing and slicing APIs validate rank, bounds, steps, output shape,
  and empty/singleton boundary behavior at the API or planning boundary.
- Views may use signed strides and negative steps when reachable-range
  validation proves every logical element maps inside the backing allocation.
  Zero step remains invalid. Do not reject negative strides solely for being
  negative. Narrower adapter APIs may document stricter limits, explicitly at
  the API boundary.
- After validation, hot loops and kernels should not repeat range checks per
  element; carry validated shape/stride/offset metadata inward.
- Prefer safe Rust patterns that let LLVM eliminate bounds checks: iterate over
  slices directly, slice once before the loop, use `chunks_exact` when the
  chunk size divides the length, and add pre-loop assertions for validated
  index ranges. Unchecked indexing is a last resort.
- Prefer metadata-only slices and strided views over dense copies. If a slice
  must allocate, document and test the reason.
- Slice, reshape, transpose and reverse preserve the shape/stride/offset
  semantics of the input view.
- Unchecked indexing or unsafe pointer access after validation keeps the
  invariant close to the unsafe block, with tests for full range, empty or
  singleton slices, lower and upper boundaries, out-of-range errors, rank
  mismatch, and non-contiguous slices.

Audit hints:

- Detect: per-element `assert!`, `checked_*`, or indexed `[]` access on
  validated data inside hot loops; unchecked access without a nearby
  invariant; slices that allocate without a documented reason; stride
  rejection based only on sign.
- Fix: validate once at the boundary, carry metadata inward, iterate slices
  directly, and keep the boundary tests listed above.

## Faer Integration

- Prefer zero-copy `faer::MatRef` / `faer::MatMut` views over packing into
  temporary dense matrices. Validate shape, bounds, alignment, and aliasing
  before constructing unsafe raw faer views. faer treats zero or overlapping
  strides in a `MatMut` as undefined behaviour: reject them first.
- Feed faer column-major-friendly layouts. For dense matrices, row stride `1` is
  the preferred contiguous layout; generic-stride inputs that trigger faer
  performance warnings must be justified or converted deliberately at an
  explicit boundary.
- Take faer parallelism from the `tcontract-exec` context passed to the
  operation only; never choose `Par::rayon` or thread counts inside operation
  helpers.
- Scratch and batching follow the batched-matrix rule in Dense Layout And
  Linear Algebra.

Audit hints:

- Detect: packing into temporary dense matrices before a faer call;
  `Par::rayon(...)`/`Par::Seq` chosen inside an op helper; `MemBuffer`
  scratch allocated per decomposition, solve, or batch iteration.
- Fix: zero-copy `MatRef`/`MatMut` after validation, parallelism from `Exec`,
  scratch sized once per operation.

## Performance Anti-Patterns

- Do not hand-copy near-identical dtype-specific operation bodies. Use generic
  helpers, sealed traits, or macros, and isolate unavoidable dtype dispatch at
  the outer boundary.
- Do not allocate dense buffers when strided access is available.
- Do not zero-initialize buffers that will be fully overwritten.
- Avoid per-element index multiplication in hot loops; use incremental pointer
  offsets or precomputed strides.
- Do not allocate `Vec` or other heap buffers inside hot loops; pre-allocate
  and reuse scratch.
- Do not build plans inside execution loops; pre-compute plans and pass them
  in.
- Resolve a runtime operation enum, dtype, or flag once per call or worker
  range, never per element. A closure that matches on a runtime op inside a
  tensor-sized loop defeats vectorization
  ([strided-rs#269](https://github.com/tensor4all/strided-rs/issues/269)).

Audit hints:

- Detect: near-identical dtype-specific bodies; `vec![0 ...]` or `Vec::new()`
  inside loops; zero-fill before a full overwrite; index multiplication per
  element; plan construction per execution; `match op` or dtype dispatch
  inside a per-element closure.
- Fix: generic helpers or macros with dispatch at the outer boundary, hoisted
  scratch, incremental offsets, and precomputed plans.

## Performance-Sensitive Tests And Benchmarks

- Small reference tests may materialize dense tensors, but should materialize
  each full result once and compare the whole result. No per-element
  re-contraction or re-evaluation as the comparison mechanism.
- Long regression tests should be sized so accidental dense materialization,
  `O(n^2)` planning, or unexpected copies fail quickly while the intended
  algorithm stays cheap.
- For approximate equality, report a useful residual such as absolute max
  error or relative norm error.
- Use release-mode benchmarks for performance claims, wrap inputs and outputs
  with `std::hint::black_box` where needed, and pin thread counts when
  comparing CPU behavior.
- Benchmark scaling across representative sizes, shapes, layouts, and thread
  counts. A single fixed-size speedup is not enough evidence.
- Benchmarks live in `tcontract-bench` (`benchmarks/`). Every public operation
  and every alternative implementation (for example faer-loop versus
  TBLIS-style batched GEMM, permute-plus-GEMM versus direct contraction) has
  rows, added in the same change as the operation. When an operation
  delegates to a strided kernel, that kernel has its own rows so a defect is
  caught at the strided level first.
- Every tensor-sized case is measured at one and four threads in the same
  run. The harness builds the `Exec` from the requested count (`Exec::Serial`
  for 1T, a bounded four-worker pool borrowed through `Exec` for 4T), asserts
  the effective width at startup, and fails when `RAYON_NUM_THREADS`,
  `OMP_NUM_THREADS` or `OPENBLAS_NUM_THREADS` conflict with it, or when any
  variable of the removed library knobs is set (nothing reads them; the `tcbench` harness knobs are `TCBENCH_*`). A four-thread time that is not faster than the one-thread
  time for a tensor-sized case is a finding, even when the one-thread row
  matches the reference. A `--threads 1` flag that left an ambient pool
  running multi-threaded once produced a 7x wrong one-thread row in
  tenferro-rs.
- Build parallelism and measured thread count are different settings.
  Compile with a job count chosen for the host (`CARGO_BUILD_JOBS`); the measured
  binary itself runs with a pinned thread count.
- Pin the measured process with `taskset` to cores within one L3 domain and
  check they are idle immediately before and after each measurement (a
  `/proc/stat` busy fraction over a few seconds is enough;
  `benchmarks/scripts/pinned.sh` does both). CPU affinity is Linux-only;
  elsewhere, record that the run was unpinned. Never run two
  benchmarks at once. Prefer short per-case runs over long target-wide runs.
- Warm a microbenchmark to a settled clock before timing it. On the measured
  AVX-512 host a kernel reads ~25% low for the first 1-2 s of sustained work
  after an idle gate (39.8 vs 52.7 GFLOP/s for the same kernel), which is
  large enough to be mistaken for a kernel defect. Warm up for **0.5-2 s of
  wall time**, not a fixed call count, and give every arm the same warm-up:
  a count-based warm-up biases the ratio when the arms differ in duration.
  Quoting an absolute rate without its warm-up and run order is invalid. A short
  call needs the top of that range: measured on this host with a 2 ms call, 500 ms
  of warm-up left the first-measured arm at 2.89 ms against 2.06 ms settled (40%
  higher latency, a 29% lower rate) while the arm measured after it read as if it
  were 29% faster; 1500 ms removed the dependence on position for those cases.
  Keep the scatter across repetitions, not only the best: a slow first repetition
  is how an unsettled arm shows up. The check to apply by hand is that two arms
  whose rows carry the same driver, blocking and partition policy must agree.
- Attribute a case's cost to a phase before touching anything. The packed driver has an
  opt-in instrument for it: build with `--features phase-timing` (a no-op otherwise, every
  call site is `#[cfg]`-gated) and set `TCONTRACT_PHASE=1`; each execution then prints one line
  at the entry of the next one, with `pack_a`, `pack_b`, `kernel`, `writeback` and the
  remainder as `setup`. On the campaign's worst losing TCCG case (`abjc-cbka-kj` f64 16 MiB,
  1T, one process per configuration, best of 5):

  | configuration | total | pack_a | pack_b | kernel | writeback |
  |---|---|---|---|---|---|
  | TCCG sizing | 43.3 ms | **27.6 (63.6%)** | 0.003 | 14.5 (33.5%) | 1.8 (4.3%) |
  | `--stress padded` | 25.2 | **9.7 (38.7%)** | 0.003 | 14.2 (56.3%) | 1.6 (6.5%) |
  | `--stress ragged` | 21.8 | **10.9 (50.1%)** | 0.003 | 10.0 (46.0%) | 1.1 (5.2%) |

  `pack_a` is two thirds of the call and the only phase that moves when the layout moves: the
  1.8x between the TCCG sizing and a padded leading dimension is entirely `pack_a` (27.6
  against 9.7 ms), while the kernel and the write-back stay within their own noise. `pack_b`
  is 3 us because that operand is 48x40 and contiguous. `setup` saturates to zero when the
  phases account for the whole call, which they do once the instrument's own `Instant` calls
  are included - read it as "nothing large is left over", not as a measurement.

  Three limits, all in `phase.rs`: measure at **one thread** (the counters sum worker time, so
  at 4T the same case reports `pack_a=42.8 ms (296%)` against a 14.5 ms call, and the
  percentages stop being wall-clock shares); **one execution at a time** (the counters are
  process-global and snapshotted at the next entry, so overlapping calls mix or drop each
  other); and `writeback` covers `emit_tile` only, so a Direct family - which stores `D` inside
  the tile call - shows that store in `kernel`. With the feature on, absolute times are ~13%
  worse than a plain build; read the shares, not the totals.

- Read the shape of a slowdown before claiming a cause. A constant absolute
  delta across sizes is a per-call or per-entry cost; a uniform multiplicative
  factor across cases the change cannot affect is host contention. A
  same-binary A/A run gives the noise floor.
- Keep before/after harness identity explicit: the arm list, the case names
  and the harness settings are part of a baseline's identity. When they
  change, recapture the baseline with the new harness rather than comparing
  across it, and record which commit produced each side.
- Record the tcontract-rs commit, CPU, `taskset` core set, profile, thread count
  and timed boundary beside every published table; raw output stays out of
  git except under a result page.

Audit hints:

- Detect: element-wise reference loops that re-run the operation per
  element; debug-mode timing claims; missing `black_box`; benchmarks at one
  size or thread count only; a thread flag parsed but not used to build the
  `Exec`; missing startup verification of the effective width; a public op or
  alternative implementation with no `tcontract-bench` row.
- Fix: materialize once and compare whole results; release-mode benchmarks
  with an enforced `Exec` at 1T and 4T across representative sizes.

## Performance-Gated Experiment Protocol

Human/process protocol.

- Performance candidates found by static/source audit alone pass a
  need-before-implementation gate before code changes start: measure the
  candidate path's share of an end-to-end workload or another predeclared
  representative workload. If the share is not meaningful under the
  predeclared threshold, record the measurement or argument in the issue and
  close or defer without implementation. A microbenchmark of the helper is
  useful after the need is established, but alone proves only effect.
- Before running a candidate, record the baseline commit, candidate commit,
  benchmark source, build profile, hardware and affinity configuration,
  provider/thread settings, complete case list, comparison statistic,
  acceptance threshold, repetition policy, and host-noise observables and
  thresholds. Candidate results must not influence these choices.
- Run the complete baseline/candidate suite as one paired experiment. Do not
  selectively retry, omit, replace, or promote individual favorable cases.
- If a predeclared host-noise or validity gate fails, classify the entire
  paired experiment as `INCONCLUSIVE`. Reconsideration requires a complete
  paired rerun under the same protocol.
- Summarize the decision, primary result, validity, and remaining limitations
  in the work log. Retain every measured case, validity observation,
  regression, and reproduction-critical setting in the experiment results,
  linked from the work log. A negative or inconclusive primary result is
  evidence and must not be rewritten as success because secondary cases
  improved.
- Promote a performance-gated change only when its predeclared primary gate
  and all required non-regression/correctness gates pass. Do not relax
  thresholds, redefine the primary metric, or add post-hoc exclusions after
  seeing the candidate.
- Record the host observables that decide validity next to the predeclared
  thresholds: the pinned cores, their idle observation before and after each
  measurement, and the load at start.
- When a claim is about a countable mechanism (pool entries, provider calls,
  allocations, copies), count it with explicit counters rather than inferring
  it from timing.

Audit hints:

- Detect: an optimization PR or issue without a recorded end-to-end share,
  predeclared thresholds, or a complete paired run; selective retries or
  post-hoc exclusions in the linked experiment evidence.
- Fix: record the need measurement and protocol in the issue before code
  changes; report negative or inconclusive results as such.

## Cache Ownership

- Long-lived plan, scratch and descriptor caches are owned by the plan or
  context object that uses them, not hidden in thread-local/global state or
  buried in kernel internals.
- Every cache has a bounded default, a user-facing way to configure that
  bound, and a user-facing way to clear it.
- Every cache exposes user-facing introspection for retained entries and
  retained bytes, reported as the cache's owned/logical payload estimate.
- Resource pools such as scratch pools need explicit limit/clear controls,
  stats APIs, and documentation.
- Do not add a cache without documenting its owner, lifetime, default
  capacity, memory behavior, entry/byte accounting, and
  clear/configuration/stats path.

Audit hints:

- Detect: `OnceLock`, `lazy_static`, `thread_local!`, or a `static` map used
  as a cache; a cache type without bound, clear, and stats APIs. (The
  `tcontract-kernel` CPU-feature and cache-hierarchy probes are cached in
  `OnceLock`s; those are process-constant facts, not data caches.)
- Fix: own the cache from the plan or context object with bound/clear/stats
  controls and documentation.

## CPU Threading Contract

- There is no ambient pool. Every operation that may run in parallel takes a
  `tcontract_exec::Exec` argument: `Serial`, a Rayon pool borrowed from the host,
  or a host broadcast executor. The global Rayon pool is never used
  implicitly, and no tcontract code calls `rayon::current_num_threads()` or
  builds a `ThreadPoolBuilder` to decide policy. (Imported strided code keeps
  its `AmbientRayon` mode for its existing callers; tcontract never selects it.)
- Serial work runs on the calling thread and never enters a pool. A kernel
  picks its width from its own work; only when that width is greater than one
  does it enter the pool, and it does not enter if the calling thread is
  already a worker of that pool. The calling thread drives everything else,
  so host thread-local state needs no propagation.
- Keep three widths distinct: the budget set by the host, the active width
  chosen from the work, and the dispatch width actually woken. SPMD kernels
  with barriers use a full-pool broadcast only; medium widths use
  barrier-free partitions; a partition larger than the budget is
  repartitioned, never served by extra threads. No production path creates
  threads beyond the pool (no scoped-thread fallback, no crate-private pool).
- An SPMD kernel called from inside a region of the same pool runs its
  barrier-free variant; a plan that can only run co-scheduled reports a typed
  unavailable route before any write instead of being silently serialized.
  Concurrent SPMD kernels on one pool are serialized by the context.
- faer parallelism is derived from `Exec`: `Par::Seq` for width one,
  `Par::rayon(k)` inside the borrowed pool's `install` otherwise. Never pick
  `Par` inside a helper.
- One budget governs batch-level and inner parallelism: a batch that fans out
  over items runs each item serially.
- Thresholds are measured per kernel and machine and live in one place per
  kernel family; they are policy values, not scattered constants.
- BLAS/OpenMP provider threading stays controlled by provider variables
  (`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `OMP_NUM_THREADS`,
  `VECLIB_MAXIMUM_THREADS`); tcontract makes no placement promise for threads a
  provider creates.

Audit hints:

- Detect: `rayon::current_num_threads`, `par_iter`/`rayon::join`/`scope`
  outside an `Exec`-entered region, `ThreadPoolBuilder` in library code,
  `std::thread::scope`/`spawn` in a production path, `Par::rayon` chosen in a
  helper, a pool entered for work below the kernel's threshold.
- Fix: thread `Exec` through, enter only in the parallel branch, derive
  widths from the work and the budget.
