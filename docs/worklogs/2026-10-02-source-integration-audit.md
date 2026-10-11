# Source integration design audit, 2026-10-02

Scope: issue [#37](https://github.com/tensor4all/tcontract-rs/issues/37), initial
spec commit `d6df16a`, implementation baseline `fbc83f5`. This revision edits
design documents only. It does not implement the consolidation or mark the
draft accepted. The user requested a detailed design, branch push and issue
post, and explicitly asked to avoid excessive safety machinery and benchmark
frequency.

The workspace size check was 86,768,512 512-byte blocks, about 41 GiB. It did
not cross the 100 GB threshold. The existing checkout was clean; the audit
continued on integration-spec. CodeGraph was synced to the baseline before
source review; Markdown/config files were read directly where it does not
index them.

## Findings and disposition

| Finding | Evidence at baseline | Change in the draft |
|---|---|---|
| The dependency arrow contradicts "kernel does not depend on exec". | Overview §3; exec currently imports kernel workspace types. | Explicit final dependency table; exec/kernel siblings, contract integrates them. |
| Rejecting repeated axes loses supported label diagonals; K described only as A/B omits isolated reductions. | tensorcontract/plan.rs:23–38, 478–556; TAPP conformance cases 3/4 and output diagonal tests. | Separate DotGeneral axis uniqueness from Labels normalization; sum checked strides and use zero opposite-input strides for isolated K. |
| Problem construction cannot validate real execution pointers, and alpha/beta ownership is contradictory. | TAPP Product snapshots metadata at lib.rs:700–707, checks each execution at 935–971; existing Rust prepared-plan API. | Metadata validation at lowering/plan boundaries; bounds/layout/alias preflight per call; scalars stay execution arguments. |
| Absent C, in-place C=D, separate C and op_D need distinct semantics. | TAPP lib.rs:1101–1113, aliasing/conformance tests; tensorcontract Plan::new docs describe overwrite without C. | Explicit overwrite/accumulation source; whole-result conjugation; keep null-C/nonzero-beta refusal in TAPP. |
| Plan-time selection has no stated relationship to runtime width. | contract/plan.rs custom selection; contract-traits/backend.rs advisory PlanningBudget; exec/width.rs. | One selected family, advisory planning budget, one centralized provisional cost estimate; runtime only schedules the fixed plan. Typed selector is a construction argument, not a generic closure in PlanConfig. |
| Partial-width broadcast wakes the full pool, and new outer batching introduces cross-item races without additional preflight. | exec/exec.rs:182–204; TAPP batch currently validates items then runs them sequentially at lib.rs:1024–1055. | Full-pool-only barrier broadcast; medium widths reuse barrier-free serial tile workers; check output/input cross-item dependencies before parallel batching. |
| "One arena plus a serial arena" lacks an owner. | workspace.rs module docs, Pool arena and contract TbPlan arena. | Move byte workspace to exec; Pool and serial Plan own it; retain lease/trim API, RAII reentry cleanup and one retained-byte limit. |
| ID renaming is not injective and describes stale complex dtype names. | families.rs:58–74 already emits c32/c64; tc.scalar and portable real 4x4 differ; portable.rs has direct/direct-B; induced.rs halves 1m MR. | Distinguish real-scalar/direct/direct-B/induced bases; logical induced geometry; full per-target manifest required. No Auto priority promotion. |
| Removing environment policy leaves C callers without a non-default configuration path. | Standard TAPP_create_tensor_product has no config argument. | Keep standard defaults; add a narrow opaque-config extension with typed setters and snapshots, leaving upstream prototypes untouched. |
| Delaying all consumer/testkit/bench changes to PR 4 breaks PRs 1–3. | Bundle depends on BLAS C ABI; bench depends on BLAS/linalg/kernel crates; testkit depends on contract-traits. | Consumer manifests, imports, fixtures and CI move with their dependency; migrate required testkit and TAPP glue in PR 3. |
| Deleted scripts/results paths are ambiguous and the recovery commit is called the last snapshot without proof. | Current root scripts and benchmarks/scripts are CI inputs; imported scripts reference each other; tcbench has no script-launch dependency. Both listed old snapshots contain the directories. | Scope deletion to tensorprimitives/*; preserve workflow helpers transitively and record an actual deletion-parent SHA, pinned archive links and restoration command. |
| File splitting risks recreating duplicated validation. | TAPP lib.rs contains info/product/execution/handle/status duties; plan/driver/cache symbol maps show distinct analysis/scheduling/probe/model responsibilities. | Responsibility-based source move table; contract owns contraction validation, C owns only raw-argument adaptation. No mandatory four-file TAPP count. |

## Choices and limits

The maintainer's D1–D10 choices remain. The detailed design adds draft choices
for explicit C configuration and a small workspace-retention bound. These are
reviewable design proposals, not claims that the maintainer already accepted
an implemented interface.

The final draft keeps Pool::borrow(&ThreadPool) and the existing one-shared-
wrapper host contract. An exclusive mutable borrow/global deduplication
registry was considered and discarded as unnecessary scope. Exact strided
alias-set intersection, transactional output rollback, handle-liveness
registries, cancellation/resource-manager frameworks and cross-pool lock
tracking are also excluded. Necessary checks remain at their owning operation
boundary and are not repeated in numerical loops.

Verification policy is proportional: port existing tests, add focused seam
regressions, and run ordinary local gates. Keep the benchmark rows, but run
one four-corpus 1T smoke only at Phase 1 completion. No benchmark per move,
PR, auditor or documentation edit; no paired performance campaign in this
change. Existing performance evidence is not recaptured just to rename files.
Heavy promotion experiments remain Phase 2 work.

The tenferro migration is a separate task. trsm/linalg removal has no
contraction replacement and cannot be described as a drop-in revision bump.
Issue #31's old dependency-isolated interface criterion is superseded by #37's
in-package backend seam; #22's repository rename remains outside this task.
Only #37 is posted to in this session.

## Source and document review

Read the root AGENTS/REPOSITORY_RULES/PERFORMANCE_TIPS and relevant shared
repository/docs/provenance/Rust rules. Inspected Cargo manifests, CI, decision
log and provenance sections; CodeGraph source/symbol maps for exec pool,
width and broadcast, kernel families/registry/portable/induced/workspace,
tensorcontract plan/driver/cache boundaries, Rust contract/trait/permute
bridges, TAPP product/execution and tcbench. Reviewed the historical scripts'
references and current benchmark helper ownership; did not claim a completed
35-file deletion manifest before an actual removal PR.

No new third-party algorithm or reference implementation was introduced.
The source-move design retains existing tensorprimitives/strided attribution,
TAPP BSD-3 and DLPack Apache-2.0 notices and imported Git ancestry.

## Verification of this documentation revision

- PASS: git diff --check; Markdown relative-link/section checks.
- PASS: cargo fmt --all -- --check.
- PASS: cargo clippy -j 16 --workspace --all-targets -- -D warnings, with
  the pre-existing unknown clippy::chunks_exact_to_as_chunks lint warning on
  this local stable toolchain; exit 0 does not mean it was warning-free.
- PASS: cargo build -j 16 -p tprims-bundle.
- PASS: RUSTDOCFLAGS='-D warnings' cargo doc -j 16 --workspace --no-deps.
- PASS: check-agent-skills.py, test_idle_cpus.py (6 cases),
  test_tblis_decision.py (6 cases), and CI's shell syntax-check list.
- FAIL (existing source, macOS/aarch64): cargo test -j 16 --workspace stops
  at tprims-blas/tests/cplx_native.rs; c32/c64 forced-ID cases and the family
  listing case fail. The source/manifest/lockfile are unchanged in this diff.
  This is not a passing full workspace gate.
- UNAVAILABLE AS WRITTEN: the root AGENTS additional clippy feature command
  refers to strided-kernel/parallel, strided-perm/parallel and
  tcontract-bench/parallel that are not selectable in this current workspace.
  The detailed design requires updating stale gate commands during migration,
  rather than adding fictitious features.
- PASS: cargo test -j 16 -p tensorprimitives-tapp -p tcontract-exec
  -p tprims-contract-traits -p tprims-bundle, including doctests and the
  bundle's C/C++ ABI consumers.
- Not run: MSRV 1.89 (not installed), release/hardware matrix, Linux nm lane,
  or timing benchmarks. This is a documentation audit on aarch64 macOS;
  future numerical, ISA and performance acceptance is not claimed.
