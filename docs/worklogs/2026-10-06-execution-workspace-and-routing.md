# Execution workspace/lease boundary and worker-context routing (#70)

tcontract-side prerequisite for tensor4all/tenferro-rs#2004. Design:
`docs/superpowers/specs/2026-10-06-execution-workspace-and-routing-design.md`.

## What changed

**The plan owns no scratch.** `Plan<T>` loses its `ArenaProvider` field and the
`exec.workspace().unwrap_or(&self.workspace)` fallback. The workspace travels on the
execution resource instead: `Exec::Rayon` lends the pool's arena,
`Exec::serial_with_workspace(&provider)` lends a caller's, and a bare `Exec::Serial` runs
with call-local buffers (the driver's existing `workspace == None` path). The public
`Plan::execute_*`, `PreparedContraction`, `contract_batched` and `NaivePlan` signatures do
not change. `Exec` keeps `Copy`/`Send` and gets a hand-written `Debug` (a
`&dyn WorkspaceProvider` is not `Debug`).

**The arena is race-free and its accounting is exact at quiescence.**
`with_worker` used to hand out a raw pointer to a slot and then mutate `borrowed`,
`PageBuf::ptr` and `PageBuf::cap` outside the map lock while `retained_bytes`/`trim` read
and wrote the same fields under it — a data race reachable from safe code. A slot is now
*checked out* of the map for the duration of a call and returned by a panic-safe guard, so
a slot is either in the map (touched only under the lock) or owned by exactly one thread.
`ArenaProvider::stats()` reports `WorkspaceStats { retained_bytes, leased_bytes }` from one
accounting lock, settled in both directions at checkout/return/take/drop/trim;
`retained_panel_bytes()` stays for shape-comparable comparisons. `TeamSet`/`PageBuf`/`PAGE`
stop being public and `TeamLease` loses its `Deref`, so panel growth can only go through
`panel()` and the accounting bypass that could underflow `leased` is no longer
expressible.

**Route selection classifies the existing policy.** `batch::lanes` loses the worker gate
(a worker is a legitimate barrier-free caller); everything else — `width_for`,
`partition_with`, the explicit-grid clamp, the batch balance rule, blocking — is
unchanged. After the geometry is final and before the workspace is taken, the driver
classifies: batch lanes → barrier-free lanes; `p == 1` → serial; `pm == 1` → barrier-free
cells (no column group shares a panel slice, so there is no barrier); `pm > 1` from a
worker → `ExecError::Unavailable` with nothing written; otherwise SPMD `broadcast`. The
discovery-time "broadcast declined, retry serially" path is gone. The private driver is
now fallible and `Plan::run` propagates the error; the `PreparedContraction` trait already
documented `Error::Exec` "before any write".

The pool's SPMD mutex stays a blocking execution-time lock. There is no reservation and no
"coordination busy" error: a first revision proposed adding one, and the pre-implementation
review showed that checking availability at resolution and locking at execution cannot both
hold without a lease spanning the two.

## Alternatives rejected

- **`Plan::workspace_req()`** — one execution's requirement depends on the resolved route
  (lanes, effective grid, active-width blocking, `beta`), not on the plan, so a plan-time
  accessor could only be a misleading upper bound.
- **A `NoWorkspace` provider** — `None` already means "this call has no owner".
- **A public `Route` report** — the pool's existing `PoolStats` counters (`entries`,
  `broadcasts`, `inline_runs`) already identify the route, and the acceptance tests use
  them.
- **Threading an explicit `workspace` parameter through every entry point** — the
  `Exec`-carries-the-resource form keeps the public signatures and the trait unchanged.
- **A planned serial route for a worker caller that needs SPMD** — the maintainer-approved
  contract calls that case a typed error; silently serializing is what #70 removes.

## Verification

- `cargo test -p tcontract-exec`: checkout/return reuse, re-entry, exclusivity, panic
  recovery, both-direction accounting, trim-versus-live-lease, and a two-thread
  `stats`/`trim`-against-a-running-worker stress test.
- `cargo test -p tcontract`: worker-context refusals now assert the typed error, an
  untouched sentinel output and unmoved counters (`packed_exec_pool.rs`,
  `packed_exec_seam.rs`, `packed_exec_pin.rs`, `custom_concurrency.rs`, the DynamicTiles
  refusal in `driver/tests/dynamic.rs`); the batch worker case asserts barrier-free lanes
  through `PoolStats`; `packed_workspace_alloc.rs` measures the caller-owned warm serial
  steady state; `packed_exec_pool.rs` adds a `pm == 1` direct-B case that runs barrier-free
  in place from a worker.
- C ABI: the nonzero serial executor now owns an arena, so its 1T rows keep a warm steady
  state; the zero handle keeps call-local scratch.
- Benchmarks: `BenchThreads` owns the 1T arena, so a one-thread row measures the same
  steady state a pooled row does.

## Review rounds

The first read-only post-review of this diff (`openai/gpt-6.1-sol`) found one blocker and
five smaller findings, all of which are fixed here:

- **dangling-panel pointer arithmetic.** A direct-B call has `b_bytes == 0`, so its panel
  pointer is a zero-size dangling pointer, but `run_strip` offset it by the column
  group's slice (`bpart.panel * b_group`) and `compute_block` by the sliver offset before
  branching on `direct_b`. A non-zero `ptr.add` on a dangling pointer is undefined even
  when the result is never read. Both offsets now happen only on the packed branch.
  Reachable before this change too, with any multi-cell direct-B call, but this change
  made it a normal route (`pm == 1` cells).
- **a panicking `take_team` preparation left capacity counted.** `prepare` allocates, so it
  can panic; the popped set is then dropped with its pages while `owned`/`panel` still
  counted them, breaking quiescent exactness. A teardown guard now arms before `prepare`
  and settles the accounting on unwind (the state lock is no longer held across
  `prepare` either, so a panic cannot poison it). `a_panicking_team_prepare_counts_nothing`
  reproduces this and fails without the guard.
- **the `tcbench` 1T rows still used a bare `Exec::serial()`**, so after the plan stopped
  owning scratch they allocated per call inside the timed loop; the engines now own one
  arena outside it, as `BenchThreads` does.
- `retained_bytes` is documented as an unsettled estimate during a call, not a lower
  bound: a callback can shrink its scratch too.
- `execute_slices`' "allocates nothing" is limited to layout metadata, since packed
  scratch still comes from the `Exec`.
- the accounting tests now cover a capacity-reducing geometry change and a zero-byte
  request on a warmed panel, and the DynamicTiles refusal compares the whole `DynStats`
  snapshot before and after (ordering before `note_call`).

A second read-only pass on the fixed revision found two more:

- **`WorkspaceProvider` was safely implementable outside the crate.** Adding
  `Exec::serial_with_workspace(&dyn WorkspaceProvider)` made a custom provider reachable
  from a safe public entry point, and the driver writes through the pointers `with_worker`
  hands out, so a provider that returns null or misaligned buffers would have been
  undefined behaviour from safe code. The trait is now sealed (`ArenaProvider` is the only
  implementation), which is what the design already assumed: the owner is the arena.
- **A pinned `StaticGrid` whose `pm * pn` overflows was accepted** and then multiplied
  unchecked by the driver, panicking (or wrapping) on a safe public call. A product
  overflow is now rejected by `ResolvedGemm::with_partition`, which every configuration
  path goes through, with regression tests at both the kernel and the `Plan::new` level.

It also caught documentation drift that predates this change: the benchmark README still
called the 1T context `Exec::Serial`, `docs/architecture.md` claimed a scratch-size query
per operation and described barrier-bearing SPMD as `k = d = pool width` (the active width
is `k <= budget`; the dispatch width is the pool). All three are corrected.

## Remaining constraints

- One map entry per caller thread that has ever used a provider, kept after `trim` (which
  releases the buffers, not the entries). Reclaiming them needs a thread-exit hook.
- A checked-out worker slot is accounted at its checkout capacity while the call runs;
  `stats()` is exact whenever nothing is in flight.
- `stats()` exactness and `pm == 1`/worker routing depend on `Plan::run` normalizing the
  context through `with_budget`; a hand-built `Exec::Rayon` with a budget above its pool is
  not covered.
- `contract_batched` keeps its validate-all-then-execute preflight, but a route error in a
  later item is not rolled back; the all-or-nothing claim now says so.
- cpueinsum's pin (`fed859ac`) and the tenferro `ext/tenferro-cpu-tcontract` extension still
  predate this change and are migrated by their own issues (#4, #2004).
