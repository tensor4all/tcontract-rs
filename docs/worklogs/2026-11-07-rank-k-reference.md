# Rank-k update, step 1: the plan kind and the reference kernel

> tensor4all/tcontract-rs#175 (`herk`/`syrk`, requested by Ryo, `Ryo-wtnb11`). The
> shape of the feature was decided in the issue thread on 2026-11-04; this
> record covers step 1 only — the plan surface, a reference kernel and the
> oracle — and the choices made inside that decision.

## What was decided before this change

A new plan kind, `RankK { triangle, hermitian }`, resolved to its own kernel
family, not a flag on the general product. Only the requested triangle is
written, and the other is neither written nor read, also with `beta != 0`.
`alpha`/`beta` are real for the Hermitian form and complex for the symmetric
one; the Hermitian diagonal is validated, not trusted. `op(A)` is `A` or
`A^H`.

## Choices made here

- **A separate `RankKProblem`, not a field on `Problem`.** `Problem` is the
  description every `ContractionBackend` receives. A rank-k field on it would
  change what that type means for every existing backend: one that ignored the
  field (the testkit's `NaiveBackend`, any downstream backend) would compute the
  full product and write both triangles. `RankKProblem` lowers to a general
  `Problem` internally (`B` laid out as `A`, `C` from `D`), which supplies the
  spans and overflow checks; `Plan::problem()` on a rank-k plan returns that
  lowering, documented as the full product rather than the update.
- **The plan kind lives in `Plan`.** `Plan::new_rank_k` builds a `Plan<T>`
  whose strategy is `Strategy::RankK`; `report()`, `muladds()`, `estimate()`,
  `execution_route()` and `workspace_req()` answer for it. Every general-product
  entry (`execute_into*`, `execute_slices*`, `execute_uninit_slices`,
  `execute_raw*`, `contract_batched`, `PreparedContraction`) refuses a rank-k
  plan with `Unsupported` before reading or writing, because each of them would
  write both triangles. `Plan::execute_rank_k` refuses a general plan.
- **`C` is updated in place, as in BLAS.** No separate output and no separate
  `C`. `beta == 0` reads no `C`; `alpha == 0` reads no `A`.
- **Value validation is an error, not a repair.** A complex `alpha`/`beta` for
  the Hermitian form is `ValueError::ComplexScalar`; with `beta != 0` a
  diagonal element of `C` with a nonzero or NaN imaginary part is
  `ValueError::HermitianDiagonal`. Both are checked before any write, so a
  refused call leaves `C` unchanged. Reference BLAS instead drops the imaginary
  part silently; the issue asked for validation. The written diagonal has an
  imaginary part of exactly zero.
- **Matrices only.** `A` and `C` are rank-2 layouts (`ShapeError::NotMatrix`
  otherwise); a batched or multi-mode rank-k update is not part of step 1.
- **Honest route and cost.** `ExecutionRoute::RankKReference` for every
  nonempty call (including `alpha == 0`, which the kernel handles itself:
  the general output-only pass would write the whole matrix);
  `Algorithm::RankKReference`, `Reason::RankK`; `muladds()` is
  `n (n + 1) / 2 · k` (×4 for complex), the half product; `estimate()` ignores
  the width, since the kernel is serial; `workspace_req()` is zero. A
  `PlanConfig` that forces the packed driver is refused, not ignored.

## The `Exec` question

The reference kernel needs no decision about `Exec`: it runs serially on the
calling thread, takes no workspace and starts no task, which is what the
threading contract already prescribes for serial work. `execute_rank_k` takes
an `&mut Exec` anyway so the register-tile family can use it without a
signature change; until then it is unused, and the route says so. The open
questions belong to step 2: how the triangle is partitioned over a team (the
work per block column is triangular, so an even column split is unbalanced),
and whether a batched rank-k has a batch-lane policy at all.

## Verification

`crates/tcontract/tests/rank_k.rs` compares the requested triangle with
`tcontract_testkit::oracle::rank_k_reference` — `contract_reference` with `A`
passed as both operands and the contracted label repeated — over f32, f64,
c32 and c64, both triangles, both `op(A)`, both forms, `(n, k)` in
`(5,3) (3,7) (4,4) (1,1) (6,1) (0,3) (4,0)`, and four `(alpha, beta)` pairs.
`A` and `C` carry padding and offsets; every storage element of `C` outside
the requested triangle (other triangle and padding) is filled with NaN
payloads, infinities, `MAX` and subnormals and must come back bit-identical.
With `beta == 0` the requested triangle is garbage too. Separate tests cover
the Gram form for non-square `A`, rejection of an imaginary Hermitian diagonal
(with `C` unchanged), its acceptance at `beta == 0`, complex `alpha`/`beta`
rejection, `alpha == 0` with an all-NaN `A`, the route/cost/report, refusal by
the general entries, and construction errors.

Mutation check (each applied alone to the kernel, then reverted): writing both
triangles, reading the other triangle with `beta != 0`, reading `C` at
`beta == 0`, reading `A` at `alpha == 0`, skipping the diagonal check, and
accepting a complex `alpha` each fail at least one test. Forcing the diagonal
imaginary part to zero is not separately observable: for finite inputs
`conj(x) x` already has an exactly zero imaginary part, so the explicit store
is a guarantee for non-finite accumulations rather than a tested difference.

## Not in this step

No benchmark rows: a serial reference kernel has no 4T story and makes no
performance claim; rows come with the first register-tile family, together
with the half-multiply-add comparison against the general product. No C ABI
entry. No faer route (upstream `gemm` has no rank-k entry).
