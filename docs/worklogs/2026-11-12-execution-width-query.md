# The width a resolved plan uses on a lane ([#176](https://github.com/tensor4all/tcontract-rs/issues/176))

> Requested by Ryo (`Ryo-wtnb11`) for a block-sparse tensor library that places a
> few hundred independent GEMMs per call on lanes of a 96-wide engine and
> currently credits each a linear speedup at an estimated width. The design was
> agreed in the issue thread (2026-11-06 and 2026-11-12); this record covers the
> implementation and the choices made inside that decision.

## What was decided before this change

`Plan::execution_width(&Exec, budget, alpha, OutputContract) -> Result<Option<NonZeroUsize>>`
is the only placement answer. `budget` has `batch_decision_within`'s semantics —
the width the plan takes on `min(budget, exec.budget())` slots of that token, with
the token's dispatch costs, team kind and batch policy and the plan's own estimate —
and it is callable from `&Exec`, because the consumer scores many candidate lanes
and `Exec::limit` consumes the token. `Plan::estimate` stays advisory permanently
and is never a placement input (tcontract#163).

## Choices made here

- **One route decision, two queries.** `execution_route` is now
  `route_within(exec, exec.budget(), ..)`, and `execution_width` is
  `route_within(exec, budget, ..)` plus the route's width. The packed arm
  therefore answers from the same `execution_geometry` call execution uses,
  including the barrier-free fallback (tcontract#178). A test pins the two
  queries to the same number on the same narrowed lane, so they cannot drift.
- **The `Auto` budget narrowing moved into `batch_budget(exec, budget)`.** It was
  `exec.width_for(serial_ns, PLAN_SECTIONS)`; it is now
  `exec.width_for_within(budget, ..)`, which is the same expression when `budget`
  is the context's own (`width_for_within` is `width_for` with the budget
  clamped). Both live callers pass `exec.budget()`, so the executed budget is
  unchanged; the hypothetical-lane query reuses one expression instead of
  restating the rule.
- **The Faer width comes from the decision `run` makes.** `FaerPlan::query` and
  `FaerPlan::decision` are now shared by `run` and by `width_within`, which
  returns `lanes * (carries_team ? item_threads : 1)`. The consumer's complaint
  was a restatement of `FAER_GEMM_SECTIONS`, the item-parallel flag and the
  section count in their own code; keeping the query on top of the executed
  decision is the answer to it.
- **`None` means "not the plan's to decide", not "serial".** `Empty` and
  `OutputOnly` run an output-sized pass; `Elementwise` and `InnerProduct` hand
  the token to strided, which splits by its own loop-length rule. Reporting `1`
  for those would be wrong exactly for the callers that schedule sub-teams, and
  a caller that gets `None` must reserve the width it asks for.
- **`RankKReference` reports `Some(1)`.** The reference kernel is serial on the
  calling thread, one participant; `Plan::estimate` already clamps a rank-k plan
  to one for the same reason. The route postdates the issue thread, which named
  only `Packed`/`Faer` and the output-only routes.
- **Rejected: `ExecutionRoute::Faer { width }`.** The route of a resolved plan
  under the context as handed in must not change, and one route cannot answer
  the many budgets the consumer scores. Kept as a separate, budget-taking query.
- **Rejected: a `Plan::estimate`-shaped time-at-width in this change.** The
  consumer prices items with its own rates; a second rate model would be a
  scheduling contract this crate cannot calibrate on the caller's host.

## Verification

`crates/tcontract/tests/execution_width.rs`:

- packed, `64³` (small, one participant at every lane) and `512³` (large,
  1/2/4/8 at 1/4/8 slots) at budgets 1/4/8 on an 8-slot pool with the reference
  cost table: `execution_width` equals the width `execution_route` reports on
  `exec.limit(budget)` exactly, never exceeds the lane, is `1` at one slot, and
  clamps a wider candidate lane to the context.
- faer, `512³` single item and `8³ × 4096` items at 1/4/8: `Some(1)` on a serial
  context and for the single item on a plain token (faer is `Par::Seq` there),
  the lanes the batch decision forks for the small items (6 at 8 slots with the
  reference costs — not the budget, which is why the query is not
  `min(budget, items)`), `<= budget`, and a ledger cross-check that no execution
  ran wider than the reported width.
- `Empty`, `OutputOnly` (`alpha == 0`), `Elementwise` and `InnerProduct` report
  `None` on a context where the packed plan reports a width.

Full gates: `cargo test -p tcontract`, `cargo test -p tcontract --release`,
`cargo test --workspace`, `cargo fmt --all -- --check`,
`cargo clippy --workspace --all-targets -- -D warnings`, `cargo doc -p tcontract
--no-deps`.

## Remaining constraints

- The query is Rust-only. The C ABI keeps `execution_route` for its preflight and
  grows no symbol; a C host that schedules lanes has no query yet.
- `Auto` on the faer route can still report fewer participants than the budget
  (6 of 8 for the small-item batch): the number is the cost model's, not
  `lanes = budget`, which is the point of asking instead of assuming.
- A `PartitionPolicy::StaticGrid` or `DynamicTiles` plan keeps its legacy budget
  for `Auto` (`driver::execution_geometry`), so its reported width follows that
  rule, not the narrowing.

## Review

Plan gate and post-implementation review: an independent reviewer subagent on
`ollama-cloud/glm-5.3-flash:high` (read-only). `VERDICT: APPROVE` on the plan, then
`VERDICT: APPROVE` on the finished diff. Findings taken: the `None` cases in the test now assert
the route they are about, so a mis-routed plan cannot pass; the packed width's positive invariant is
stated at the `NonZeroUsize::new` site. No finding was dismissed.
