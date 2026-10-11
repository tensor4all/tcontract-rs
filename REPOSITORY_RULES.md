# Repository Rules

tcontract-specific rules, applied on top of the shared tensor4all rules from
`tensor4all-agent-rules` (see the shared `rules/index.md`). Performance,
layout, scratch, threading and benchmark contracts live in
`PERFORMANCE_TIPS.md`; read it before performance-sensitive work and reviews.
Adapted from tenferro-rs `REPOSITORY_RULES.md` (commit `5a4e7fd`). Keep this
file minimal.

## Public Surface Drift

- `README`, rustdoc, and examples must not claim capabilities beyond the
  current public surface.
- When the public API changes, check `README`, rustdoc, and examples for stale
  names, stale capability claims, and deleted paths before considering the
  work complete.
- Before every PR, check the README and `docs/` against the implementation,
  not only against the diff: crate lists, diagrams and dependency tables
  against the manifests (`cargo tree -e normal --depth 1 -p <crate>`); names,
  features, commands and status lines against the code; planned items
  labelled as planned. The Phase 0 README diagram kept showing
  `tcontract` depending on a sibling crate after the code settled
  otherwise.

## Public Surface Discipline

- Keep the public API small. Prefer `pub(crate)` for types, functions, traits,
  modules, fields, and helper constructors unless external users are expected
  to call them directly.
- Public types implement `Debug`. Prefer `derive(Debug)` when cheap, useful,
  and free of unstable internals. Hand-write a summary for plan, context,
  cache, or FFI wrapper types when derived output would dump large buffers,
  leak representation details, or constrain future changes.
- Do not make implementation details public because another module needs
  them; first consider moving the code or adding a narrower crate-private
  helper.
- Items mainly for tests, benchmarks, internal planning, or dispatch should
  normally be private or `pub(crate)`. `#[doc(hidden)] pub` is not a
  substitute for privacy; use it only for supported macro output or required
  trait contracts.
- When another crate of the stack needs access, expose the narrowest
  owner-scoped API that preserves its invariants. Every part must make sense
  used alone (design principle 1).

## Public Boundary Safety Audits

- User-reachable Rust and C APIs validate input-derived shape, stride, offset,
  axis, dtype and configuration before no-op shortcuts, allocation, planning,
  or provider calls. Review fast paths and zero-size returns like the main
  path.
- Shape products, byte lengths, strides, offsets and FFI dimensions use
  checked arithmetic before conversion to `usize`, `isize`, `i32`, pointer
  offsets, or allocation sizes. Audits search for `iter().product`,
  `* size_of`, `as usize`, `as isize`, `as i32`, `stride *=`, and unchecked
  `+`/`*` on shape-derived values.
- Pointer-offset loops over batches, matrix blocks, or tensor strides check
  both the per-item extent and the `batch * stride` offset.
- Publicly reachable library paths must not turn invalid user input into
  `panic`, `unwrap`, `expect`, unchecked indexing, poisoned-lock unwraps, or
  debug-only assertions. Keep truly internal invariants close to their proof;
  otherwise return a typed error. Every C ABI entry point also catches panics.
- When a bug exposes a public API design mismatch, fix the canonical API
  contract rather than adding a parallel `try_*` escape.
- Public validation APIs return crate error types, not `String`. Translate
  into another layer's error type explicitly, preserving the original in the
  `source()` chain.
- Validation helpers return typed prepared metadata when downstream code would
  otherwise repeat indexing, shape-product, or dimension-role calculations.
  Pass validated metadata to kernels instead of re-validating and recomputing
  unchecked offsets.
- Do not add public errors, helpers, or validation branches for states
  unreachable through supported APIs.

## Invariant Markers

- Use one canonical marker, `// INVARIANT: <why this is valid, bounded, or
  intentional>`, for non-obvious invariants kept for performance or by design:
  rank-bounded quadratic loops, checked-by-construction arithmetic,
  ownership-required copies, semantic zero-fill, and similar audit
  false-positive hotspots. Keep `// SAFETY:` for unsafe blocks.
- The marker states the invariant concretely enough to re-verify: name the
  validation site, the bound, or the owning contract.
- `#[allow(...)]` suppressions carry an adjacent `// INVARIANT:` line, except
  the workspace-wide lint allowances in the root `Cargo.toml`, which carry
  their reason there.
- Audits must not flag a site governed by an `// INVARIANT:` marker; they
  check whether the stated invariant still holds.
- Parallel operation surfaces (Rust and C ABI, serial and parallel paths,
  alternative strategies) keep validation semantics in parity. A bug in one
  surface triggers an audit of the others before the fix is complete.

## Work Logs And Design Records

- Keep a lightweight record for nontrivial multi-phase changes, non-obvious
  design choices, or performance experiments under `docs/worklogs/`. Record
  decisions and reasons, important alternatives, verification conclusions,
  and remaining constraints, not command histories.
- Specs and plans live in `docs/superpowers/specs/` and
  `docs/superpowers/plans/`; `docs/decision-log.md` records the current
  position on each open question and is updated in the PR that changes it.
- When a bug report or audit finding is a false positive because of an
  intentional invariant, add the `// INVARIANT:` comment or a test at the
  site so the non-bug is not rediscovered.

## CI Cost Discipline

Human/process protocol.

- Reproduce CI lanes locally (fmt, clippy with `-D warnings`, workspace
  tests, `tcontract-exec` without the `strided` feature, release-profile
  `tcontract` tests, the aarch64 check, the C ABI and c-consumer lanes,
  MSRV build, docs) before pushing.
- Batch fix-ups into one push; do not push a stream of small CI-fix commits to
  an open PR.

## Publication And Publish-Safety

Maintainer protocol.

- No crate is published from this repository without the maintainer's
  explicit approval for that specific package; a general request to finish a
  PR does not authorize it.
- strided-rs crates are published from their own repository; the tcontract crates
  are not published yet.
- Before any `cargo publish`, inspect the packaged files (`cargo package
  --list`) and the metadata: name, version, description, license, repository,
  homepage, documentation, README, `rust-version`, keywords, categories.

## No Ad Hoc Fixes

- Do not add ad hoc fixes that violate DRY, KISS, or layering.
- Do not introduce compatibility shims, duplicated logic, or reach-through
  into lower layers when the correct fix belongs in an existing seam.

## Unsafe Code Boundary

- `unsafe` is confined to FFI (C ABI, DLPack), raw strided/faer view
  construction, SIMD micro-kernels and packing. Planning, validation and
  scheduling code is unsafe-free.
- Keep the validation invariant next to each `unsafe` block (`// SAFETY:`)
  and test the boundary conditions.

## File Organization

Keep source files small and focused, but do not split solely to reduce line
count. ~1000 lines is a soft review trigger, not a mechanical limit. A split
must follow a clear boundary such as validation, planning, execution,
dispatch, public API, or provider glue. No `part1` / `part2` splits, and no
tiny files that scatter one concept across many modules.

## Unit Test Organization

Keep production source files focused on production code. No inline
`#[cfg(test)]` blocks in normal modules unless the file is a genuinely tiny
leaf module and the test is trivially small. Prefer module-local test
directories such as `src/<module>/tests/*.rs`, leaving only
`#[cfg(test)] mod tests;` in the source file. Reserve crate-root `tests/` for
integration tests. Do not use `include!` to inject test files. Imported code
keeps its existing layout until it is otherwise changed.

## Generic Over Scalar Type

- Use generic functions over sealed scalar traits instead of per-type
  functions (`gemm::<T>` rather than `dgemm`/`zgemm` in Rust).
- For dtype-polymorphic operations, prefer one typed generic implementation
  plus outer dtype dispatch (the C ABI) over per-dtype copies.
- If Rust generics cannot express the shared structure cleanly, use a local
  macro for the repetitive dispatch.

## PR Content Hygiene

- Do not include AI-generated analysis, task, or verification reports as
  standalone files in PRs. Durable session records belong in
  `docs/worklogs/`; specs and plans in `docs/superpowers/`.
- Do not commit new top-level directories or dot-directories without an
  explicit maintainer decision recorded in the PR.

## Imported Code

- tensorprimitives-rs (Lukas Devos) and `benchmarks/` were imported with
  `git subtree` (no squash) and keep upstream history; the tensorprimitives
  tree was then moved into `crates/` and dissolved (its docs are archived in
  `docs/archive/tensorprimitives/`); see `docs/provenance.md`.
  strided-rs was imported the same way and later removed again; it is an
  external dependency, and strided changes go to tensor4all/strided-rs.
- Files there keep their upstream copyright and license notices. A change to
  imported code is an ordinary commit here; say in the message when it
  diverges from upstream behaviour.
- Authorship is never dropped. Imported history is never rewritten
  (no squash merges of import branches, no history filtering). Imported
  crates keep their upstream `authors` (Lukas Devos for `tcontract-kernel`,
  `tcontract` and `tcontract-capi`; Satoshi Terasaki and Hiroshi Shinaoka for `strided-*`).
- Code moved or ported out of an imported crate into a `tcontract-*` crate (for
  example the packing and micro-kernels of
  tensorprimitives-rs's driver into `tcontract-kernel`) keeps the original authors in that crate's `authors`,
  keeps the upstream copyright and license notice in each moved file's
  header, moves with `git mv` where possible so history follows, and names
  the original author with a `Co-authored-by:` trailer in the commit.
- Crates and documents that build on imported work credit it: the TBLIS-style
  strategies credit Lukas Devos's tensorprimitives-rs and Matthews's TBLIS
  paper in rustdoc and README.
