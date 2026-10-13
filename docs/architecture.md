# Architecture design notes

[Back to the overview](../README.md). These notes retain the detailed rationale, numerical requirements, and primary sources of the current design. There is no stable API, ABI or performance claim yet. AI-assisted design and implementation are welcome; contributors remain responsible for correctness, measurements, and code provenance.

tcontract is a CPU library for dense binary tensor contraction, usable from Rust and, through the TAPP C ABI, from C, Julia or Python hosts. It is not tied to one consumer.

## Working hypotheses

- **Contraction:** one validated `Problem` is planned once and run by one of three strategies: a packed, block-scatter (TBLIS-style) driver that packs general strides straight into microkernel panels, faer on problems that fuse into one copy-free batched GEMM, and an elementwise pass for all-batch problems. Measurement, not assertion, decides which one a plan selects.
- **Phase 2 goal:** optimize the packed driver until it replaces faer, leaving one route for contractions with a K role that differs only by kernel family; all-batch problems stay delegated to strided-rs. Small, Hadamard-like and batched shapes (`hadamard.json`) must not regress while doing so.
- **Execution ownership:** the effective thread budget, pool and scratch lifetime are explicit. A Rust caller supplies an execution context. A C, Julia or Python caller creates, uses and closes a pool through the C ABI, without the library taking over the host's threads.
- **No configuration through the environment.** No library crate reads an environment variable for numerical, threading or routing policy; every such knob is an explicit input (`PlanConfig`, `Exec`). The only environment read is the opt-in `TCONTRACT_PHASE` per-call phase-timing toggle behind the `phase-timing` feature.

These are hypotheses until evidence exists. [The research map](research-map.md) distinguishes published evidence from project-specific hypotheses, and [the experiment plan](experiments.md) defines how to compare them.

## Design principles

The full statement and rationale are in [design principles](design-principles.md). In short:

1. **Few crates with real boundaries.** Each crate has a separate consumer or a stable interface; there is no facade.
2. **Short names under one prefix.** Crates are named `tcontract-<part>`.
3. **One C ABI crate.** `tcontract-capi` builds `libtcontract_capi` (`cdylib`, `staticlib`, `rlib`). The standard TAPP interface is the contraction ABI; tcontract adds named extensions.
4. **Zero copy at the boundary.** Data crosses the C ABI as DLPack descriptors; operands are borrowed.
5. **Caller-owned execution.** Every expensive operation receives an explicit execution context. No ambient global pool.

## Crates

Dependencies are those of `cargo tree` (normal and build edges):

| Crate | Responsibility | Depends on |
| --- | --- | --- |
| `tcontract-exec` | Execution context: serial, a host's Rayon pool through a borrowed, owned or shared-`Arc` wrapper, width chosen from work, kernel-level entry, SPMD `broadcast`, and the workspace provider (`ArenaProvider`, `WorkspaceReq`, `TeamLease`: pool-owned, reusable scratch); `strided::run_with_exec` bridges strided-rs kernels (feature `strided`). | rayon, thiserror; strided-basic (optional) |
| `strided-view`, `strided-basic` (external, strided-rs) | Checked borrowed strided views, scalar and conjugation contracts; copy, permutation and elementwise kernels. | none in tcontract |
| `tcontract-kernel` | The kernel crate: packed formats, kernel-family descriptors, CPU masks and validation, the registry with deterministic built-in families, resolution with a frozen blocking policy (explicit inputs, no environment reads), the partition policy, caller-scoped `KernelCatalog`/`KernelHandle` for downstream kernels, packing, scatter and write-back, cache blocking (`blocking::{probe,model}`), and the microkernel families: Lukas Devos's scalar, AVX2, AVX-512 and NEON register-tile kernels (`kernels::{reference,x86,aarch64}`), the portable reference kernels and the native interleaved complex kernels. Ids: `{isa}.{dtype}.{scheme}.{MR}x{NR}`. No executor and no ambient state. | num-complex |
| `tcontract` | Binary contraction with free, contracted and batch indices (`dot_general` and label semantics) over one validated `Problem` (`api`: `Labels` and `DotGeneral` front ends, one lowering, one `Error`, the object-safe `ContractionBackend<T>` / `PreparedContraction<T>` traits taking an `&Exec`), planned once into a `Plan<T>` (`plan`: role folding and orientation, offsets, family and strategy resolution, an immutable `PlanReport`) and executed by one of three strategies (`strategy`, `driver`): the packed driver (packing traversal, loop nest, write-back, static and dynamic partition), faer on a copy-free batched-GEMM fusion, and an elementwise pass. Threads and workspace come from a `tcontract_exec::Exec`. Thin permute / add wrappers, the labels-based unary update (`unary`: `Unary`, `add`, one operation for a permutation, a diagonal and a reduction, lowered to a rank-zero `B`), `contract_batched`; `check_raw` / `execute_raw` are the pointer-level entry of the C adapter; `execute_slices` runs a plan on plain slices without building views. | faer, num-complex, thiserror, `tcontract-kernel`, `tcontract-exec`, `strided-view`, `strided-basic` |
| `tcontract-capi` | `libtcontract_capi` (`cdylib`, `staticlib`, `rlib`): the TAPP C ABI (tensor infos, label-based products lowered once into a `tcontract::Problem`, run through `Plan<T>::execute_raw`, batched products as sequential items), DLPack types and operand borrowing, the one status table (behind `TAPP_check_success` / `TAPP_explain_error`), the thread-local last-error message, the `TAPP_executor` (including the Rayon extension) and ABI version queries. Contains no algorithm and no second validator. Modules: `abi`, `handle`, `tensor_info`, `product`, `execute`, `executor`, `status`, `dlpack`, `tensor`. | `tcontract`, `tcontract-exec`, num-complex, rayon, strided-view |
| `tcontract-testkit` | Test support: an independent label oracle, seeded fixtures, a naive second backend (naive loop nest over the problem's roles) proving the trait seam, and the downstream-kernel fixture `custom_kernels`. Not a production fallback; never published or selected by default. | `tcontract`, `tcontract-kernel`, `tcontract-exec`, strided-view, rand |
| `tcontract-bench` (`benchmarks/`) | The benchmark harness: `tcbench` over the contraction corpus, Rust and C ABI rows at 1 and 4 threads. Not part of the library. | `tcontract-capi`, `tcontract`, `tcontract-kernel`, `tcontract-exec`, strided-rs |

The name `contract` was chosen over `tensordot` because NumPy, PyTorch and JAX `tensordot` has no batch indices; the operation here does, as in cuTENSOR's contraction entry point.

`tcontract-kernel` has no C ABI: the packed format is an internal contract between the driver and the kernels. `tcontract-testkit` is a dev-dependency of `tcontract` and `tcontract-capi`; it is linked only into tests, so it creates no cycle.

### Dependency rules

- `tcontract-exec` and `tcontract-kernel` are the bottom of the tcontract graph and depend on no other tcontract crate. strided-rs depends on nothing in tcontract.
- The kernel layer takes no execution context of its own; drivers take a `tcontract_exec::Exec` for parallelism and, through it, a workspace by borrowing the host's provider.
- `tcontract` sits on `tcontract-kernel` and `tcontract-exec` and takes an explicit `Exec` for every expensive operation.
- `tcontract-capi` is the only crate with C symbols and depends on `tcontract` and `tcontract-exec`.
- No cycle between crates.

### What is excluded

- **N-ary einsum.** Index notation and contraction-order planning stay above the stack (the published `strided-opteinsum` releases, or the consumer) and call `tcontract` for each binary step. The stack exposes no index-string API.
- **AD, traced execution, device transfer, GPU backends**, and adapters for `ndarray` / `mdarray`. These sit above the stack.
- **Matrix-level libraries.** GEMM and batched GEMM are contractions; dense linear algebra and Krylov solvers are not provided.

> **Removed in #37.** The former BLAS crate and its C ABI, the engine and adapter layers, and the dense linear-algebra crate were deleted in the source integration ([#37](https://github.com/tensor4all/tcontract-rs/issues/37)). What replaces each, and the old-to-new names, are in the [migration guide](migration-2026-10.md).

## The C library

`tcontract-capi` is one crate and one shared (or static) library, so every handle, in particular an executor, is valid in every call of the library: there is one copy of the execution runtime, one Rayon runtime and one status table. Separate shared libraries per part are not supported: an opaque handle is a Rust type, two libraries would each carry their own copy of `tcontract-exec` (so a pool created in one is not a valid object in the other), each with its own Rayon runtime, and two Rust static libraries linked into one C program duplicate Rust standard library symbols.

Headers live in `crates/tcontract-capi/include/`: the pinned upstream TAPP headers verbatim (`tapp.h`, `tapp/*.h`, BSD-3-Clause, commit and checksums in `tapp/README.md`), DLPack (`dlpack/`, Apache-2.0), and tcontract' own `tcontract/tcontract.h` (umbrella), `tcontract/core.h` (DLPack operands, status codes, library identity) and `tcontract/tapp_ext.h` (the Rayon executor). `crates/tcontract-capi/install.sh` installs the header tree, the library and a `tcontract_capi.pc` file.

### ABI conventions

- Symbols: the contraction and executor API keeps the standard `TAPP_*` names; tcontract extensions to it are `tcontract_tapp_<object>_<operation>` (for example `tcontract_tapp_executor_create_rayon`); library-level symbols are `tcontract_<object>_<operation>`.
- Every entry point catches panics and returns a status (a `TAPP_error` is a `tcontract_status`: zero is success and the other values are provider-defined, so callers test success with `TAPP_check_success`); `tcontract_last_error()` returns a thread-local message.
- `tcontract_abi_version()` and `TCONTRACT_ABI_VERSION` (header) state the ABI number; `TAPP_implementation_version()` states the crate version of the loaded library; `tcontract_has_part("tapp")` answers what the library contains. A header and a library from different versions are detectable.
- There is no C tuning API: kernel and blocking choices are Rust `PlanConfig` inputs.

## Zero-copy data exchange

The C ABI uses [DLPack](https://dmlc.github.io/dlpack/latest/) so that NumPy, PyTorch, JAX, CuPy and Julia arrays pass without copying.

- **Operand descriptor.** Every `tcontract_*` tensor argument is a borrowed `tcontract_tensor`, a view plus the DLPack flags that `DLTensor` alone does not carry (`DLPACK_FLAG_BITMASK_READ_ONLY` lives in `DLManagedTensorVersioned.flags`):

  ```c
  typedef struct {
      DLTensor *view;      /* borrowed; never freed by tcontract */
      uint64_t  flags;     /* DLPACK_FLAG_BITMASK_* */
  } tcontract_tensor;

  /* Borrow a versioned tensor: copies view pointer and flags; takes no ownership, never calls the deleter. */
  tcontract_tensor tcontract_tensor_borrow_versioned(DLManagedTensorVersioned *m);
  /* Raw DLTensor (DLPack 0.x producers): the caller asserts the memory is writable if used as an output. */
  tcontract_tensor tcontract_tensor_borrow_raw(DLTensor *t, uint64_t flags);
  ```

  The descriptor is valid only for the duration of the call; ownership and lifetime stay with the caller. Standard TAPP products take raw pointers described by `TAPP_tensor_info` (extents and element strides) instead.
- **Inputs** are read through `view`; the READ_ONLY flag is permitted and no copy is made. The library honours `byte_offset`, arbitrary element strides (column-major and negative strides included) and a NULL `strides` meaning compact row-major. `lanes` must be 1. Accepted devices are `kDLCPU` and `kDLCUDAHost`.
- **Outputs** are written in place with `D = alpha * op(A, B) + beta * op(C)` semantics and defined zero-size behaviour. An output whose span overlaps an input's span is rejected (`TCONTRACT_ERR_ALIASED`), even when the element sets are disjoint; a `C == D` in-place update is accepted only when both describe the same elements.
- **Conjugation** is a per-operand argument, since DLPack has no conjugation flag. It maps to the lazy conjugation of `strided-view`.
- **Read-only outputs** are rejected with `TCONTRACT_ERR_READ_ONLY` during validation, before any write. For a descriptor made with `tcontract_tensor_borrow_raw`, the check sees only the flags the caller passed; writability of the memory is the caller's precondition.
- **Materialization** is never hidden: no strategy copies a whole operand (`PlanReport::materialized` is all false), and the bounded packing inside the packed driver is not a materialization.
- **Dtypes** in ABI v1: `f32`, `f64`, `complex64`, `complex128`; others (`bf16`) are rejected with a datatype error.

The DLPack header is Apache-2.0; `tcontract-capi` mirrors its `#[repr(C)]` layout and records the upstream version.

## Execution context

`tcontract-exec` defines the Rust contract and `tcontract-capi` its C face. A context is one of:

- **Serial:** work runs on the calling thread.
- **Rayon pool:** a pool borrowed from the host (for example the pool of tenferro-rs) for the duration of a call; created through `tcontract_tapp_executor_create_rayon` by a C host that has none; or created once by a C host through `tcontract_pool_create_rayon` and lent to an executor with `tcontract_tapp_executor_borrow_pool`, so the host keeps one pool instead of two. The global Rayon pool is never used implicitly.
- **Host callbacks:** a vtable through which a Julia, Python or C host schedules tasks on its own threads. Not part of the TAPP redesign; if added later it starts with barrier-free outer batches and serial inner operations, since an arbitrary task-submission callback does not guarantee SPMD co-scheduling.

The C face is the standard `TAPP_executor` (`intptr_t`; the signatures are those of the pinned upstream `executor.h`) plus the Rayon extensions and the host-owned pool handle. See [decision-log](decision-log.md#execution) for the choice.

```c
TAPP_error TAPP_create_executor(TAPP_executor *exec);   /* serial executor, no workers */
TAPP_error TAPP_destroy_executor(TAPP_executor exec);   /* owned pool: stops and joins it; borrowed pool: releases the borrow */
TAPP_error tcontract_tapp_executor_create_rayon(TAPP_executor *out, size_t nthreads, const tcontract_rayon_opts *opts); /* stack size */
TAPP_error tcontract_tapp_executor_set_budget(TAPP_executor exec, size_t budget);
TAPP_error tcontract_tapp_executor_get_threads(TAPP_executor exec, size_t *pool_size, size_t *budget);
TAPP_error tcontract_pool_create_rayon(tcontract_pool *out, size_t nthreads, const tcontract_rayon_opts *opts);
TAPP_error tcontract_pool_destroy(tcontract_pool pool);
TAPP_error tcontract_tapp_executor_borrow_pool(TAPP_executor *out, tcontract_pool pool, size_t budget);
```

- **Default and width:** executor `0` is the default serial executor (a tcontract policy). `nthreads == 0` is an error; for `tcontract_tapp_executor_create_rayon`, `nthreads == 1` is a serial executor and creates no workers, while `tcontract_pool_create_rayon` builds a real one-worker pool (the handle is a pool, not an executor); the width is never inferred from `RAYON_NUM_THREADS` or the CPU count. The pool width is fixed at creation. The budget is positive, clamped to the pool width and snapshotted at the start of each call. A serial executor reports `pool_size == 0`, `budget == 1`. The query promises neither an active width nor an affinity.
- **Lifetime:** the C host owns the executor; an owned executor owns its pool. Plans do not bind an executor, so one plan runs on serial and 4T executors. There is no retain/release, and a bindings layer that wants a shared pool creates one `tcontract_pool` handle, one borrowed executor on it, and keeps both alive for every plan. Sharing one executor across plans shares its pool.
- **A host-owned pool** (`tcontract_pool_create_rayon`) is the shareable case: the handle owns the pool, `tcontract_tapp_executor_borrow_pool` builds an executor on it without creating a second one, and the pool must outlive every executor built from it and every call through one. `TAPP_destroy_executor` on such an executor only releases the borrow. `tcontract_pool_destroy` stops and joins the pool, and refuses with `TCONTRACT_BUSY` while a borrowed executor is live (`TCONTRACT_ERR_WOULD_DEADLOCK` from one of its own workers); both leave the handle live. Each borrowed executor carries its own workspace and SPMD gate for the shared pool, so one executor per pool is the intended configuration.
- **Destruction of an owned pool is synchronous and joins the threads.** A pool created by `tcontract_tapp_executor_create_rayon` is built with `ThreadPoolBuilder::spawn_handler`, which spawns each worker with `std::thread::Builder` and keeps its `JoinHandle`. `TAPP_destroy_executor` returns `TCONTRACT_ERR_WOULD_DEADLOCK` from a worker of the same pool and `TCONTRACT_BUSY` if calls are in flight (the handle stays live in both cases and the caller may retry); otherwise it drops the pool to start shutdown, joins every handle and frees the executor. An exit-handler notification is not sufficient: in rayon-core 1.13 it runs inside the worker's main loop, before thread-local destructors ([probe](../experiments/pool-close/README.md)). After a successful destroy no tcontract worker code, including TLS teardown, is running. A failed creation joins the workers it had started. Destroying the default executor `0` succeeds as a no-op. A live handle is destroyed successfully once: double destruction and stale or foreign handles are unsupported, and the caller synchronizes destruction with the start of new calls (BUSY detection does not make a race between raw-handle use and destruction safe).
- **Borrowed pools** (the Rust `Exec::Rayon` case, and the C `tcontract_pool` handle behind a borrowed executor) are governed by their owner's lifetime; tcontract never stops the pool of a borrowed executor (the pool handle's own destroy does). A Rust host shares one `Pool` wrapper per `ThreadPool`, because the SPMD gate and retained workspace belong to the wrapper; the C entry allows several borrowed executors on one pool handle, and each gets its own wrapper, so one borrowed executor per pool is the configuration that keeps that rule.
- **Tests:** a TLS-destructor handshake with a bounded wait proving destroy does not return early; busy, self-worker and failed-creation cases; bounded-time nested and concurrent SPMD on one executor.
- **Thread budget:** one budget controls batch-level and inner-matrix parallelism so nested parallelism does not oversubscribe.
- **Scratch:** `PackedReport::scratch_bytes` reports the serial packed estimate. The execution workspace is owned by the context, not the plan: a pool or a serial caller lends an `ArenaProvider`, which reports its retained and leased bytes and can release the idle half.

**Entry happens inside kernels, only for parallel work.** The calling thread drives every call. A kernel chooses its width from the amount of work; at width one it runs on the calling thread and never touches the pool. Only a parallel kernel enters the pool, and not at all if the calling thread is already one of its workers. With this rule an entry cost of about 10 µs is acceptable ([measurement](../experiments/rayon-entry/README.md), [decision](decision-log.md#execution)), and faer can serve as the initial backend: `Par::Seq` for serial work, `install` followed by `Par::rayon(n)` for parallel work.

```rust
pub enum Exec<'a> {
    Serial,                                        // no storage owner
    SerialWithWorkspace(&'a dyn WorkspaceProvider), // caller-owned serial scratch
    Rayon { pool: &'a Pool<'a>, budget: NonZeroUsize }, // pool borrowed from the host
    // Host(&'a dyn BroadcastExecutor): host scheduler with guaranteed width, Phase 2
}
```

Implemented in Phase 1a as `crates/tcontract-exec` (`install`, `for_each_partition`, `broadcast`, `width_for`); see the [decision log](decision-log.md#execution).

**Three widths, kept distinct.**

- *Budget* `b`: the most threads a kernel may occupy, set by the host (`tcontract_tapp_executor_set_budget`, never above the pool size). tcontract never creates threads beyond the pool; there is no scoped-thread fallback.
- *Active width* `k`: the number of workers doing arithmetic, chosen from the work, `k ≤ b`.
- *Dispatch width* `d`: the number of workers woken. It sets the entry cost. For Rayon `ThreadPool::broadcast`, `d` is always the whole pool, whatever `k` is: on a fixed 18-worker pool, a broadcast with one active worker costs as much as one with 18 (about 165 µs from idle, [measurement](../experiments/rayon-entry/README.md#active-width-on-a-fixed-pool)).

Two execution shapes follow:

- **Barrier-free partition** (batches, independent output tiles, faer's own parallel loops): `install` plus `k` tasks, so `d ≈ k`. Entry scales with `k` (about 17 µs for one task, 50 µs for four, from idle). Tasks are not guaranteed to run concurrently, so no barrier may be used.
- **SPMD with barriers** (the TBLIS inner driver): needs `k` workers guaranteed to run at once. On Rayon this is `broadcast`, so the active width is `k <= budget <= pool width` while the dispatch width `d` is the whole pool, and it is chosen only when the kernel is large enough to amortize the full-pool entry. A narrower SPMD team on a borrowed Rayon pool would need a subset-broadcast primitive that Rayon lacks; that is a separate prototype, not an assumption.

**Insufficient width.** If a plan's partition needs more workers than `b`, the planner repartitions to at most `b` (down to serial) before execution. It never spawns extra threads and never enters a barrier with fewer participants than the barrier counts.

**Nesting.** An SPMD kernel called from a worker that is already inside an SPMD region, or inside any job of the same pool, runs its barrier-free variant; a partition that can only run co-scheduled is refused with a typed error before any write, never silently serialized: a worker blocked at an outer barrier could not take its share of an inner broadcast. Concurrent SPMD kernels on one pool from different host threads are serialized by the context. Both rules are tested before the adapter is used as a general borrowed-pool solution.

### Cost of parallel execution

A parallel kernel pays a fixed entry cost before any work is shared. A simple model for work `T` split over `n` threads is

```text
T_par(n) ≈ L(n, state) + T / n
```

where `L` depends on the width and on whether the workers are awake. The [rayon-entry measurement](../experiments/rayon-entry/README.md) (M5 Max, 2026-09-29) splits `L` into three parts:

| Part | Cost | Cause |
| --- | --- | --- |
| Caller wake-up | about 3 µs | The caller outside the pool blocks on a lock latch (a `Condvar`) and must be woken when the job finishes. Same as a raw two-thread `Condvar` round trip. |
| Worker wake-up | 6 to 8 µs | Idle workers sleep after 32 `yield_now` rounds, within 100 µs; Rayon wakes one sleeping worker per injected job. |
| Fan-out | 55 to 200 µs at 18 threads | Waking every worker; the heterogeneous efficiency cores likely contribute. About 8 to 17 µs at 4 threads. Grows with the dispatch width, not the active width. |
| Already inside the pool | 8 to 33 ns | No handoff: the job runs on the current worker. |

Consequences for the design:

- **Break-even.** Parallelism gains only if `T - T/n > L`. With `L` about 10 µs at small width, work below roughly 50 to 100 µs should run serially on the calling thread; an `N = 100` matrix-vector product (about 1 µs) is always serial. Full width pays off only for work of the order of milliseconds.
- **Width from work.** A kernel picks `n` from its flop and byte count, not from the pool size, so a medium kernel wakes a few workers rather than all of them. This holds for barrier-free partitions only; an SPMD kernel on Rayon always pays full-pool dispatch (see *Three widths* above), so its threshold is higher. The thresholds are per kernel and per machine and must be measured; they are not fixed constants of the API.
- **No per-call handoff for serial work.** This is the difference from tenferro-rs today, which enters its pool once per session and so pays 8 to 14 µs on every FFI call, including serial ones ([tenferro #1945](https://github.com/tensor4all/tenferro-rs/issues/1945)).
- **OpenMP is not intrinsically cheaper.** A default libomp parallel region costs 10 to 12 µs at 4 threads and 53 to 94 µs at 18 threads, the same range as Rayon. Sub-microsecond OpenMP entry comes from active waiting (`KMP_BLOCKTIME`), which burns cores; at 18 threads (all cores) its median rose to 59 µs and p90 to 1.1 ms after a 10 ms idle gap. A spin policy is therefore deferred ([decision](decision-log.md#execution)).
- **Chains of small parallel kernels** are the case the model penalizes most: each pays `L`. The remedy is fusing them into one parallel region (a batch, or an SPMD kernel with barriers), not lowering `L`.

Not yet measured: real GEMM and contraction break-even against width, barrier cost inside one SPMD call, a subset-broadcast primitive, a homogeneous x86 CPU, and the fixed cost through the C ABI ([Prototype 4](experiments.md#prototype-4-c-abi-slice)).

TBLIS also uses cooperating threads and barriers inside a blocked contraction. An arbitrary task-submission interface, including host callbacks, does not automatically provide that contract. Start with outer-batch parallelism and serial inner contractions, then prototype an explicitly synchronized inner driver on a Rayon context if large contractions need it.

## Selection: strategy, then family

A contraction is chosen in two steps, and both happen when a plan is built,
never during a call.

* **Strategy** — by rule, in order: an explicit kernel, selector, partition,
  complex method, blocking, cache model or write-back request in `PlanConfig`
  forces the **packed** driver (all-batch problems included); otherwise an
  all-batch problem runs the **elementwise** pass; otherwise a problem that
  fuses copy-free to one strided batched GEMM with full `op_C` / `op_D` /
  separate-C semantics runs on **faer** (a separately described C only when
  its output pass is cheap: at most 2^16 outputs, K >= 512 with a column-major
  A, or a matrix-vector shape; otherwise the packed driver serves it at every
  beta); everything else runs **packed**.
  Nothing copies a whole operand.
* **Kernel family** (packed only) — `PlanConfig::kernel`: a registered family
  by id, or the default menu (the built-in families of the preferred ISA and
  complex scheme, default first, with an optional row-block shape). Families
  carry their own tile, packed layout, complex scheme, CPU requirements and
  cache blocking, and resolution freezes the blocking policy and the partition
  so a later environment change cannot move it (the library reads none).

`Plan::report()` says what was chosen: the algorithm, and for the packed
strategy the family id, geometry, grid, orientation and observed regularity. A
choice that cannot be honoured is an error from the constructor — an unknown
id, a CPU without the required instructions, a blocking override that is not
positive or exclusive. `list_kernels` returns the registry for diagnostics.

The packed driver is told where its scratch lives: a host that owns threads
passes a `tcontract_exec::Exec` that lends a workspace (per-thread A block and tile,
a page-aligned shared B panel, scatter vectors and barriers). A pool owns one for
every operation on it; a serial caller that wants reuse passes an
`Exec::serial_with_workspace`; a plan owns none. Implementations not given an
owner get per-call buffers. Nothing is process-global, the owner reports its
retained and leased bytes and can `trim` the idle half, and a lease returns to
the owner that issued it.

### Opt-in dynamic output assignment (`DynamicTiles`)

`Partition::DynamicTiles { job_m, job_n }` replaces the static
`pm x pn` grid with dynamic claiming for the packed driver. It assigns output
work only: K is never split, no accumulation is atomic, every tile's K slabs
run in order, and for a fixed blocking the result is bitwise identical to the
static and serial runs.

* **Epochs.** All active workers traverse the same `(batch, NC panel, KC
  slab)` sequence. Per epoch worker 0 resets a claim counter owned by the
  invocation, all workers pack disjoint `NR` slivers of the shared `B` panel,
  a *publication* barrier makes `B` and the reset visible, workers claim and
  compute jobs, and a *completion* barrier (taken by every worker, claimed
  work or not) frees `B` and the counter for the next epoch. A single barrier
  cannot do both jobs. `fetch_add(Relaxed)` only makes claims unique.
* **Assignment.** With at least as many `job_m` row bands as workers a worker
  claims a whole band and reuses each packed `MC x KC` chunk of `A` across the
  band's columns. Otherwise it claims `(band, column subtile)` jobs in
  row-major order and packs `A` per job, so `A` is packed up to
  `column jobs` times (the documented 2-D ceiling). The rule depends only on
  validated geometry and the active width.
* **Width.** The team is the host budget capped by the jobs that exist; width
  one runs on the caller with no claims or barriers. A team that the caller
  cannot co-schedule is refused with a typed error before any write.
* **Direct-B** keeps `b_bytes == 0` but still takes both barriers, an explicit
  tradeoff against the static barrier-free split.
* **Validation.** Job extents are positive multiples of the family's logical
  MR/NR; invalid values and job-count overflow fail when the plan is built, for
  empty problems too (`DynamicTiles` carries no `align_c_lines` option). Never
  rounded.
* **Surface.** `PlanConfig::partition` (`Partition`). `PlanReport::packed`
  (`partition`, `dynamic`) reports the resolved policy, job extents, active
  width and assignment. `DynStats` is opt-in instrumentation.

The Stage-0 need measurement and the paired static-vs-dynamic suite were not
run (maintainer decision); the policy stays opt-in and makes no speed claim.

### Custom kernels with a safe selector

A downstream crate can supply its own packed microkernels *and* its own
selection policy without editing tcontract or touching the process defaults.

1. **Admit once, `unsafe`.** `KernelCatalog::<T>::from_static_families(&'static
   [&'static KernelFamily<..>])` is the single place the provider promises that
   its code is correct: the declared panel footprints, complete tile overwrite
   or the direct-update ABI, a truthful ISA mask, immutable state, concurrent
   calls, and no panic or unwind (a worker lost inside a barrier-bearing region
   deadlocks its team). It validates geometry, formats, dtype and id
   uniqueness, then mints `KernelHandle<T>` values. The catalog is an immutable
   caller-owned list, not a second registry; `KernelCatalog::builtin()` gives a safe snapshot of the built-in families
   and `union` combines catalogs explicitly, so a built-in fallback is a
   handle the selector chooses, never an implicit default.
2. **Select, safe.** `KernelHandle<T>` has private fields, is typed by the
   storage dtype (so `c64` is not `f64`), carries its catalog's identity and
   exposes read-only metadata only. A selector is
   `FnOnce(&SelectionContext, &[KernelCandidate<T>]) -> Result<KernelHandle<T>, SelectError>`
   and need not be `Send`, `Sync` or `'static`. The context is problem metadata
   (dtype, folded M/N/K and batch, original extents and strides, conjugation,
   requested complex method, CPU mask; no thread budget, since a plan never
   reselects its family for another budget); each candidate carries
   the facts that depend on *its* geometry — whether the driver swaps the
   operands for its `MR` and whether it reads B in place —
   computed with the driver's own rules. Pointer-dependent facts (whether `C`
   and `D` are one buffer, bounds) stay execution guards.
3. **Resolve once, execute frozen.** The selection runs in planning
   (`Plan::<T>::new_with_selector` in `tcontract`), on the caller,
   outside registry/workspace locks and worker broadcasts, and before any
   empty-problem shortcut. A homogeneous batch selects once; execution never calls the selector or
   looks anything up. The plan keeps the chosen `'static` descriptor, so the
   selector and catalog may be dropped. Membership and admissibility (CPU mask,
   conjugation, complex method) are checked after the callback; there is no
   implicit fallback on `Err`.
4. **Refuse, do not ignore.** A selector is a requirement that forces the
   packed driver (an all-batch problem included). A forced
   `KernelChoice::Id` alongside a selector is ambiguous and a typed error.

Errors are `SelectError` variants (`DuplicateId`, `ForeignHandle`,
`NotACandidate`, `NoCandidates`, `SelectorFailed`, plus the existing
`CpuUnsupported`, `DtypeMismatch`, `Incompatible`); the
contract crate preserves the typed `SelectError` as the source of `Error::Backend` (downcast it). `PlanReport`
reports the chosen family, its geometry and its provenance (`origin`);
downstream kernels report `Origin::External { crate_name, license }`.

The issue's paired 1T/4T tensor-sized benchmark protocol (with A/A noise runs)
was **deferred** for this slice, by maintainer decision; correctness, compile-fail,
selector-call-count, steady-state-allocation and concurrency tests are in
`tcontract_testkit::custom_kernels`. Nothing here claims that a custom selector
speeds anything up.

## Contraction

A contraction is one lowered, role-grouped `Problem`; `Labels` and `DotGeneral` are front ends over one lowering. Repeated labels on one operand select a diagonal (strides add), a label on only one input is a reduction (a K axis with the other input's stride zero), and the output must be injective. The `Problem` keeps the original layouts plus normalized M/N/K/H role axes with signed A/B/C/D strides. Planning is `O(M + N + K)` and happens once; execution uses offset tables and no labels.

| Strategy | Source | Idea |
| --- | --- | --- |
| Packed (block-scatter) | the imported upstream project (see [provenance](provenance.md)), by Lukas Devos; [Matthews, TBLIS](https://arxiv.org/abs/1607.00291) | Pack tensor panels with general strides directly into the `tcontract-kernel` format, run the microkernels, and scatter bounded output tiles. No full operand transpose. |
| faer | tenferro-rs `dot_general` (`tenferro-cpu/src/dot_runtime.rs`, `gemm/`), MIT OR Apache-2.0 | Fold compatible strides into a batched matrix view without copying; run faer's GEMM per batch item. Declined when it would need a copy, a reduction over an axis one input lacks, or, at `beta != 0`, a separate C whose output pass would dominate (large output, K < 512). A nonzero `beta` and a separate C are written into D by one output-sized strided pass before faer accumulates; no operand is copied. |
| Elementwise | project code | An all-batch problem is one strided-rs pass (`map_into`, `zip_map2_into`, `zip_map3_into`, with `axpy`, `fma`, `mul_into` and `copy_scale` for unit-scalar forms; in-place accumulation reads D through the destination, so it runs on strided-rs's public `execution` contract (fused plan, blocked walk, threaded map-reduce) with a small raw-pointer inner loop rather than a second view of D) with full `op_C`, `op_D` and separate-C semantics. `op_D` is folded into the other conjugations and one dispatch per execution picks a monomorphized closure, so no flag is tested per element. |

`alpha == 0` or an empty contraction computes `op_D(beta * op_C(C))` in one output pass for every strategy, reading no input; beta zero reads neither C nor D.

The packed driver runs cooperating workers with barriers inside one contraction. It therefore needs guaranteed concurrent width from `tcontract-exec`, not arbitrary task submission. The Rayon `ThreadPool::broadcast` provides it at full pool width only: workers with index at or above the active width return immediately but are still dispatched and awaited. tcontract therefore uses it only for contractions large enough to amortize full-pool entry, repartitions to the budget instead of using a scoped-thread fallback, and otherwise runs a barrier-free partition (independent output tiles, each worker packing its own panels). See [three widths](#execution-context).

## Relationship to `strided-rs`

[strided-rs](https://github.com/tensor4all/strided-rs) provides the views, basic and fused kernels and HPTT-inspired permutation this stack builds on. It is an external dependency, pinned to the same commit as tenferro-rs so both share one `StridedView` type. It was imported in Phase 0 and removed again on 2026-09-30. Binary contraction is `tcontract`; N-ary planning stays above the stack.

| Component | Role here |
| --- | --- |
| `strided-view` | Reused unchanged as the shared view contract. |
| `strided-basic` (and `strided-perm`, `strided-kernel` through it) | Copies, permutations and every elementwise pass (the all-batch strategy and the `alpha == 0` / empty-`K` output update). `tcontract_exec::strided::run_with_exec` bridges an `Exec` to strided's `ExecContext`. |

## Implementation order

Phase 1 put being usable as the tenferro-rs CPU backend first, with a thin C ABI slice and its benchmarks to find out early whether the design holds across the C boundary.

| Phase | Content | Status |
| --- | --- | --- |
| 0 | Import strided-rs, the upstream contraction project and strided-rs-benchmark-suite with history; one workspace; rules ported from tenferro-rs; root CI. (strided-rs and its benchmarks were made external again on 2026-09-30.) | Done. |
| 1a | `tcontract-exec`: borrowed Rayon pool, width chosen from work, kernel-level entry, `broadcast(n, f)`. | Done. |
| 1b | GEMM and batched GEMM entry points. | Done, then removed in #37: GEMM is a contraction. |
| 1c | `tcontract`: permute plus batched GEMM and the packed driver compared under one plan API; one planner over the packed, faer and elementwise strategies after the source integration. | Done. |
| 1d | Dense linear algebra (faer per item plus batched loops). | Done, then removed in #37 (no retained consumer). |
| 1e | tenferro-rs integration behind a feature, with an explicit per-op fallback to the current backend, A/B correctness and a same-run performance gate. | The injection points and optional providers are merged in tenferro-rs and selectable in tenferro-benchmark; acceptance runs are deferred until Phase 2 optimization. |
| 1f | A thin C ABI slice and C benchmarks; the contraction part is the standard TAPP interface ([#26](https://github.com/tensor4all/tcontract-rs/issues/26)), consolidated into `tcontract-capi` in #37. | Done. |
| 2 | Optimize the packed driver (small, Hadamard-like and batched shapes included) until faer can be deleted: one route for contractions with a K role that differs only by kernel family (all-batch problems stay on strided-rs). Then wider C ABI coverage and Windows. | Goal; `hadamard.json` must not regress. |

Crates are published only after an interface and a consumer exist, consistent with [tenferro #1927](https://github.com/tensor4all/tenferro-rs/issues/1927).

AI-assisted contributions may include algorithms, implementations, benchmarks, counterexamples, and design proposals. Acceptance rests on attributable sources, numerical tests, reproducible performance evidence for optimization claims, and maintainer review.

## Why this project exists

[tenferro-rs #1945](https://github.com/tensor4all/tenferro-rs/issues/1945) documents a concrete FFI problem with an ambient Rayon pool: entering a CPU session costs roughly 8 to 14 µs on one measured AMD EPYC configuration, against about 1 µs for one small GEMM inside the session. The numbers are machine- and configuration-specific. [faer #319](https://codeberg.org/sarah-quinones/faer/issues/319) requests an explicit caller-owned Rayon pool. This repository explores that interface independently and for any host; it does not imply that tenferro will adopt a new backend.

## Sources and provenance

[Research map](research-map.md) links primary papers, official API documentation, project decisions, and upstream licenses. [Provenance policy](provenance.md) describes how to record an independently implemented algorithm, a code port, or a reused test.

## Status

Phase 1 is done; faer is an internal strategy. No stable API, ABI or package publication has been approved. The code is MIT OR Apache-2.0; imported files keep their own notices.
