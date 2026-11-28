# Changelog

All notable changes to the tcontract crates are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Each crate is versioned on its own line; a section names the crates that were
released together and one tag (`vX.Y.Z`) marks the commit it was published
from. Releases before 0.13.0 are recorded in their annotated tags
(`git tag -l -n1 'v0.*'`) and release commits.

## [Unreleased]

### Changed — `tcontract`: fixed-size `2x2` / `3x3` batched products (tenferro-rs#2745)

- A faer-route batch whose fused `m = n = k` is 2 or 3 now runs a fixed-size
  kernel instead of one general GEMM per item. It reads both operands through
  the fused strides, so conjugation, a transposed operand and a batch-first
  layout need no copy; it allocates and packs nothing, and covers overwrite,
  `alpha`, and accumulation in D for the four storage types. The route report
  is unchanged (`Algorithm::Faer`); no API change. Other sizes keep the general
  route. Serial, one core, i9-13900HX, c64 `[N, N, 32769]`: 2x2 67.2 -> 11.3
  ns per product, 3x3 76.6 -> 47.9 ns.

### Changed — `tcontract`: the tiny square kernels run across the batch (tenferro-rs#2745)

- The fixed-size kernels now run several items of a batch per SIMD register
  for `f64` and `c64` (`2x2`, `3x3`), and for `f64` also `4x4` and `5x5`
  (previously the general route). Back-to-back matrices are read and written
  through register transposes; other layouts gather lane by lane. Dispatch is
  `pulp` at run time (256-bit level on x86-64, `Arch::new()` elsewhere), never
  the build target. The result is bit-identical to the scalar fixed-size
  kernel on every dispatch level: separate multiply and add (no FMA), the same
  operation order per item. No API change.

### Added — `tcontract`: the fused grouped-GEMM route (#208)

- `FusedGemmPlan<T>`: prepares a `t4a_desc::gemm::GemmBatch` once (one
  forced-packed serial resolution per distinct geometry, kept for those that
  take work items, and one LPT-ordered work list) and runs every job's `(job, MC band, NC panel)` items in one
  barrier-free traversal claimed from one counter. Each job's result is bitwise
  the job alone through the packed driver on one thread, at every width.
  Entries: `execute_into`, `execute_into_accum`, `execute_uninit`,
  `execution_route`, `workspace_req`, `work_items`, `batch`.
- `ExecutionRoute::FusedTiles { width }` and `Error::Gemm(GemmError)` (both
  enums are `#[non_exhaustive]`).
- `tcontract` now depends on `t4a-desc`, so `t4a-desc` 0.1.0 has to be
  published with or before the next `tcontract` release.

## [0.13.0] - 2026-11-20

Released together: `t4a-cost` 0.1.0 (first release), `tcontract-exec` 0.13.0 and
`tcontract` 0.13.0. Not released here: `tcontract-kernel` 0.3.1 (unchanged since
its release; this cycle only moved its inline test modules out of `src`),
`t4a-desc` 0.1.0 (no dependent in the workspace yet), and `tcontract-capi` /
`tcontract-testkit` (not published).

### Breaking — dependency train: threadteam 0.6, strided 0.15, colmat 0.34

A `threadteam::Exec` token is a parameter of every fan-out entry here, and
`colmat`'s `Par` appears in `tcontract_exec::with_faer_par`, so moving either to
another 0.x minor breaks callers that carry a token across this boundary: this
is a 0.x minor bump of both crates, the rule that already made the 0.32 and 0.33
moves 0.11.0 and 0.12.0.

- `threadteam` 0.6, `strided-basic`/`-perm`/`-kernel` 0.15, `strided-view` 0.8.1.
- The faer strategy's dependency is now published as `colmat` 0.34 (same crate,
  new package name). No source change beyond the requirement.

### Added

- `t4a-cost` 0.1.0: `host::CpuClass` (the CPU class an offline-fitted row is
  keyed by, with `CpuClass::detect()`), `ItemTime` (a provider's time for one
  item at a width) and `ItemFit` (the generic fitted row behind it). It has no
  dependency and no scheduler, pool or scratch type (#205), so libraries can
  share a cost model without depending on each other.
- `tcontract-exec` re-exports `host::CpuClass` and `ItemTime` from `t4a-cost`,
  so its own API and doctests keep resolving.
- `tcontract` 0.13.0: `Plan::execution_width(&Exec, budget, alpha, OutputContract)`
  reports the participants the resolved plan occupies on a lane, answering from
  the same decisions execution takes (#176).
- `tcontract-exec` 0.13.0: a batch query may carry an `ItemTime`, so a provider
  whose parallel kernel does not scale by ideal division states its own time; a
  query without one decides exactly as before.

### Changed

- The faer strategy chooses the item width on a host class with a fitted row
  from the argmin of faer's time at a width, in rounds of the kernel's largest
  task, and keeps width one unless the predicted gain clears the row's own error
  margin. Other hosts keep the ideal-division rule, so a large single GEMM keeps
  its team speed-up where nothing was measured. The f64 row ships on the fitted
  classes in this release (#194, #203, #207).

### Fixed

- The faer item width is resolved on the budget execution actually leaves faer,
  so a plan with a non-default `WorkModel` no longer reports lanes that never
  run (#176).
