> Historical note: a stale session handoff from 2026-10-01, kept as a record; it names crates, features and paths removed in the source integration (#37, see docs/migration-2026-10.md).

# HANDOFF — switchable GEMM engine (issue #23)

Date: 2026-09-30. Branch: `gemm-engine-spec` (spec pushed earlier;
implementation commits local, no PR yet).
State: **Tasks 1–3 complete; Task 4 resumed after maintainer approval of
public-contract safety corrections.** Registration is now an unsafe ABI/ISA
boundary with private mutable slots; canonical Result resolution and driver
integration are next. Task 1 commits are `d5776b8` (pure renames) and `6822d7d`
(wiring); its workspace tests remained 367 and combined release tests 124.
Task 2 adds 11 focused descriptor/registry tests; contract tests,
std-disabled check and clippy 1.98 pass. Task 3 adds portable native
complex, interleaved/FourM formats and all compiled tc menu descriptors.
Available-family packed-product oracles pass for all four dtypes/conjugations;
workspace clippy, release tests, docs and cross-target compilation pass.
Evidence is linked from
`docs/worklogs/2026-09-30-switchable-gemm-engine.md`.
On 2026-09-30 the maintainer rejected the process-global workspace ruling
and approved pool-owned team buffers plus worker-local A/tile buffers.
The spec and Task 8 now reflect that correction.

## Read first (in this order)

1. `docs/superpowers/specs/2026-09-30-switchable-gemm-engine-design.md`: the
   approved spec, v3. It is the binding authority. Appendix A lists the
   independent review of v1 and how each finding was resolved.
2. `docs/superpowers/plans/2026-09-30-switchable-gemm-engine.md`: the
   implementation plan, 12 tasks, TDD, with Review Focus.
3. Research notes the spec is built on:
   - `docs/worklogs/2026-09-30-research-blis-tblis-structure.md`: BLIS
     `cntx_t`/cntl, TBLIS 1.x/2.x, partitioning, pack-buffer ownership.
   - `docs/worklogs/2026-09-30-research-tensorcontract-faer-dispatch.md`:
     tensorcontract's current dispatch, faer/gemm-common/private-gemm-x86
     structure, threading boundaries, buffer locality problems.
   - `docs/worklogs/2026-09-30-gemm-strategy-faer-vs-tensorcontract.md`:
     why faer wins on copy-free GEMM on this EPYC.
4. `AGENTS.md`, `REPOSITORY_RULES.md`, `PERFORMANCE_TIPS.md` (mandatory
   before kernel, threading or benchmark work).

## How to continue

- Execute the plan sequentially in the main session, with tests first and
  a final integrated self-review. Do not delegate unless requested. The
  superpowers skills named in the original plan are not installed here.
- The maintainer approved Task 4's safety corrections: unsafe external ABI
  registration (without mutable public slots), canonical `Result` resolution
  APIs, and packing roles fixed to kernel operands after orientation swaps.
  Existing generic/foreign-scalar execution must remain compatible.
- Continue Task 4 with resolution/driver tests first. Task 1 rebased on `origin/main` (already
  up to date); the whole plan remains one PR on this branch.
- Stop after Task 12: merge on green CI plus the non-regression gate, and
  update #23. Optimization (DynamicTiles, native SIMD complex kernels,
  ports, per-node team pools, C ABI) is a separate session.

## Maintainer decisions (2026-09-30), all reflected in the spec

- Build the switching mechanism first, then stop; optimize in another
  session.
- Things that must be switchable:
  - the complex packed layout (interleaved, planar, 1e/1r);
  - the complex method (native, 1m, 3m, 4m);
  - SIMD vs generic kernels;
  - optimized native complex kernels, usable when they exist;
  - the kernel and block sizes, chosen at startup through function-pointer
    descriptors, with per-plan override.
- Kernels are **ported** (BLIS, OpenBLAS, faer), not taken as external deps,
  with copyright preserved. Each origin gets its own crate so licenses don't
  mix. faer's MPL-2.0 parts need a separate decision.
- private-gemm-x86 and gemm-common/gemm-* are **called** where possible, not
  reimplemented. This was the plan's `tprims-kernel-gemm` and
  `tprims-kernel-pgx86` (both removed in #37).
- The kernel layer is split out of `tensorcontract` into its own crates
  (spec v3 §3.1). Lukas Devos's files move with `git mv`, keeping authorship
  and history. This ends `git subtree pull` for the moved files, which is
  accepted.
- Parallel partition boundaries and buffer memory locality (first touch by
  the consuming thread) are part of the design. If B is not packed, no B
  buffer is allocated.
- DynamicTiles (faer-style) is designed but **not implemented** in this
  phase.
- The spec was reviewed by Fable before approval.
- P3 (tenferro-benchmark acceptance) stays cancelled until tcontract is
  optimized standalone.

## Rulings already made in the plan (flag them in the PR)

- `KernelFamily<R>` is keyed by the real scalar, not by the element type,
  because kernels operate on `T::Real` today.
- Workspace ownership correction: no process-global `ARENA`. Each
  `tcontract_exec::Pool` owns its provider; both blas/contract `ExecSpmd` (since removed in #37)
  adapters borrow it. `tcontract-exec` may depend on the thread-free
  `tprims-gemm-kernel` contract (now `tcontract-kernel`, #37) without a cycle. A/tile buffers are
  worker-local and owner-keyed; B/team sets are leased exclusively and
  returned only to their originating provider, never another pool. Serial
  typed plans own their workspace rather than using a hidden global cache.
  Owner drop/trim releases idle payloads even for borrowed host pools.
- Block-scatter vectors are filled by the caller into the leased arena. The
  spec's cooperative fill is deferred.
- `SelectedGemm` wraps `Selected` (additive) so tenferro-cpu-tcontract, pinned
  at `d8e565a`, does not break.
- Custom packers (`pack_override`) are declared but have no exerciser until
  a port needs them. Packers are otherwise layout-selected and
  monomorphized.
- Direct kernels are real-only this phase; `validate` rejects complex
  Direct families.
- Task 2 contract clarification: logical complex 1m MR/NR may be odd;
  evenness refers to the expanded inner real axis, not the logical tile.
  Existing 1m B packing is `Planar` (the OneR role), not `Real`. The spec,
  plan and regression test now agree with the existing kernel source.

## Facts verified in this session (don't re-derive)

- `private_gemm_x86::gemm(dtype, itype, instr, nrows, ncols, depth, dst,
  dst_rs, dst_cs, dst_row_idx, dst_col_idx, dst_kind, beta: Accum, lhs,
  lhs_rs, lhs_cs, conj_lhs, real_diag, diag_stride, rhs, rhs_rs, rhs_cs,
  conj_rhs, alpha, n_threads)` is public (`lib.rs:1154`).
  - Its threading goes through spindle (`with_lock` →
    `rayon::current_num_threads()` of the installed pool).
  - Enums: `InstrSet::{Avx256, Avx512}` and `Accum::{Replace, Add}`.
- `gemm_common::microkernel::MicroKernelFn<T>` is public, and `gemm-f64`
  exposes `microkernel::{scalar, fma, avx512f}::f64::UKR[mr_div_n][nr]`.
  - Semantics: `dst = alpha·dst + beta·(lhs·rhs)`; `alpha_status` 0 means
    dst is not read.
  - It handles partial m/n, and reads B in place via `rhs_rs`/`rhs_cs`.
  - `gemm-c64` microkernels are private.
- This host: EPYC 7713P, 1 NUMA node, 8 L3 domains × 8 cores. Build with
  `CARGO_BUILD_JOBS=16`. While measuring on CCD 0, build with `taskset -c
  16-63`.
- tcontract-rs main is at `690794c` (README faer citation, PR #25, merged).

## Loose ends outside this plan

- Worktrees `~/tensor4all/tenferro-rs-main` and
  `~/tensor4all/tenferro-benchmark-p1` are leftovers from Phase 1e and can be
  removed.
- tenferro's `ext/tenferro-cpu-tcontract` pins tcontract at `d8e565a`, before the
  copy-aware Auto change. Bump it when tcontract next changes behaviour
  tenferro should see.

## State at the end of the switchable-engine session (2026-10-01)

Tasks 1–12 of the plan are implemented and committed on `gemm-engine-spec`,
which is 15 commits ahead of `main` at `690794c`. Everything that can be
checked without a quiet host is green:

- `cargo fmt --all --check`;
- `cargo +1.98.0 clippy --workspace --all-targets -- -D warnings`;
- `cargo test --workspace --release` (89 green test binaries);
- `cargo test -p tcontract-exec --release --no-default-features`;
- `cargo test -p tprims-blas --release --features kernel-gemm,kernel-pgx86` (those features were removed in #37);
- the kernel crates for `wasm32-unknown-unknown` and `aarch64-apple-darwin`,
  and `tprims-kernel-pgx86` on both (it is inert off x86-64);
- `cargo build -p tprims-bundle --release`, and the capi crates' tests.

Logs: `.artifacts/gemm-engine-spec/task12/`. (`tprims-blas` for
`wasm32-unknown-unknown` fails inside `faer`'s `atomic-wait` dependency; that is
pre-existing and unrelated.)

## What is left before this can be merged

The performance gate now **passes** (see below and
`benchmarks/benchmarks/tcontract/{contract,blas}/results/2026-10-01-gemm-engine/decision.txt`);
what remains is the merge itself.

1. ~~**The performance gate.**~~ **Done.** Contract corpus: `tblis_exec`
   new/old 0.933/0.966/0.950 at 1/4/8T (noise 0.009/0/0.002) and `pg_exec`
   0.988/1.002/1.023 (noise 0.018/0/0.022); the 8-case subset with five
   sessions per side gives 4T `pg_exec` 1.061 against a 0.067 noise band and
   `tblis_exec` 0.895. Blas corpus: `tblis` 0.954/0.917/0.946 and `faer`
   0.998/0.949/0.996. All groups pass the `max(5%, noise)` rule with
   `BENCH_RUNS=5`.

2. **PR, CI and merge.** The branch is pushed and PR #27 is green; merge it
   when the maintainer is satisfied. Issue #23 stays open for the follow-ups
   the plan lists (DynamicTiles, per-node team pools, native SIMD complex
   kernels, ports, the C ABI).

### The original instructions for the gate (kept for reference)

1. **The performance gate.** It needs an *idle* host with an idle L3 domain:
   `pinned.sh` verifies the pinned cores are idle before and after every run,
   and this agent's own processes keep a core slightly busy, so the 4T runs
   exhausted their retries. A complete 1T comparison of the contract corpus
   (`BENCH_RUNS=5`) shows the reworked driver **faster**: summed medians new/old
   are 0.973 (`pg_exec`) and 0.926 (`tblis_exec`), with planning 1.034/1.037 at
   tens of microseconds. To finish the gate, run, on an otherwise idle host,
   with the default `BENCH_RUNS`:

   ```
   CORPUS=benchmarks/benchmarks/tcontract/corpus/tenferro-p1.json \
     benchmarks/scripts/paired.sh <old contract> OUT-old CPUS 1 4 8
   CORPUS=benchmarks/benchmarks/tcontract/corpus/tenferro-p1.json \
     benchmarks/scripts/paired.sh <new contract> OUT-new CPUS 1 4 8
   CORPUS=benchmarks/benchmarks/tcontract/corpus/tenferro-p1-gemm.json \
     benchmarks/scripts/paired.sh <old blas> OUT-old-blas CPUS 1 4 8
   CORPUS=benchmarks/benchmarks/tcontract/corpus/tenferro-p1-gemm.json \
     benchmarks/scripts/paired.sh <new blas> OUT-new-blas CPUS 1 4 8
   ```

   with an A/A repeat of each side for the noise floor, and apply the 5%
   per-group rule to the calls-weighted times.

   Measured so far (details in
   `docs/worklogs/2026-09-30-switchable-gemm-engine.md`): the packed driver is
   consistently faster — `tblis_exec` new/old 0.926 on the full corpus at 1T and
   0.886 on an 8-case subset at 4T — while `pg_exec` (permute + GEMM, faer) is
   not resolved here: its new/old ratio is 0.973 at 1T and 1.066 at 4T, and the
   baseline's own run-to-run spread reaches 27% on this host. The baseline binaries are built
   from the merge-base `690794c` (a worktree still exists at
   `/tmp/tprims-baseline`, target `/tmp/tprims-baseline-target`).

2. **PR, CI and merge.** The branch is pushed and a PR is open for review; the
   merge waits on the gate above and on CI. Issue #23 stays open for the
   follow-ups the plan lists (DynamicTiles, per-node team pools, native SIMD
   complex kernels, ports, the C ABI).
