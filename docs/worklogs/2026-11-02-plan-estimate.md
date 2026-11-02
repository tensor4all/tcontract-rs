# Plan work and advisory time estimates

Date: 2026-11-02
Issue: [tensor4all/tcontract-rs#163](https://github.com/tensor4all/tcontract-rs/issues/163), the
tcontract half of the decision recorded in
[tensor4all/tlinalg-rs#88](https://github.com/tensor4all/tlinalg-rs/issues/88)

## What a caller needed

A caller that runs many prepared `Plan`s of different geometries on one `Exec` has to split threads
between them, and to do that without mirroring tcontract's own quantity it needs the plan's work and
its time. The request was `Plan::estimate(&self, width, &Exec) -> Duration`, from the same model the
crate uses for its own decisions, reusing the plan's `work`.

## Where the work actually is

The issue names `JobMeta.work` / `max_job_work` from `GroupedPlan::new_with_ops`; those do not exist
in this tree. What exists is one quantity the plan's own width rule already prices —
`Problem::macs()`, used by `Plan::width` as `2 * macs * (4 if complex) * ns_per_flop()` and declared
to the token's rule with one section per execution. That is the single source this change reuses:
`Plan::muladds()` is `Problem::macs()` weighted for the scalar (4 for a complex multiply-add), in
`f64` so it can be divided by a rate. It is a fact about the problem, so it does not depend on the
strategy, the width, the budget or the machine — a test plans the same problem on faer and force-packed
and asserts one number.

## What the time is

`Plan::estimate(width, muladds_per_ns, &Exec)` is the width rule's model with the width given instead
of chosen:

```text
muladds / muladds_per_ns / width + dispatch_cost(width)
```

which is the expression `Exec::width_for` minimises (`sections * D(k) + estimate / k`), so estimates at
two widths relate the way the rule already assumes. The division is **ideal**, not a fitted speedup
curve; a contraction's work is GEMM-shaped, which is why that assumption is the crate's model and not
an addition to it.

## The rate is the caller's

Two deviations from the issue's signature, both deliberate:

* **The rate is an argument, not a default from the `Exec`.** A rate is a measurement of a host in a
  clock state, and which measurement describes the machine the estimate is about is the caller's
  decision. The token's `BatchPolicy` does carry `WorkModel.muladds_per_ns`, but that field is what
  this crate's own batch and width rules price work at; deriving the published number from it would
  make the "second opinion" the caller asked for a copy of the first. `tcontract_exec::host::muladds_per_ns()`
  (the host-class table the width rule is calibrated against) stays the natural default for a caller,
  and a measured rate of the same unit — `tlinalg::estimate::Rates::gemm()` — can be passed instead,
  which is what makes a contraction and a factorization comparable in one unit.
* **No tlinalg dependency.** `tlinalg` publishes the calibrated `Rates` value as a plain `f64` per
  nanosecond, so sharing the number needs no dependency and no layering change: a contraction library
  depending on a linear-algebra provider would be the wrong direction, and `tlinalg`'s `Rates` is a
  caller-held value there for the same reason it is one here.

## Advisory, and enforced as such

Nothing in the crate reads the estimate: `Plan::width` asks the token's rule with `Plan::serial_ns()`
and the estimate divides the same work by a caller-supplied rate, so the two share one count and one
unit rather than two formulas. That is now literal — `width` and `estimate` both go through
`Plan::muladds()`, and `serial_ns()` is the only place the host rate enters — so there is no second
expression to drift. One test checks the magnitude against the *shared rule* rather than against a
retyped formula: with synthetic dispatch costs, the published estimate is fed to
`Exec::width_for` and must land on the right side of the payback boundary at 16-cube and 512-cube,
which a seconds/microseconds confusion would not.

## Verification

Local gate: `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -D warnings`,
`cargo test --workspace`, `cargo test -p tcontract --release`, `cargo deny check bans`,
`scripts/check-no-public-rayon.sh`, rustdoc with `-D warnings`, and the doctests.
`crates/tcontract/tests/plan_estimate.rs` holds the six model tests; none asserts an absolute
wall-clock value — the rate is supplied by the test, so every expectation is a ratio or an identity.
A rate that is not a positive finite number is refused with `Duration::MAX` rather than panicking, and
`width` is clamped to `1..=exec.budget()` so the number is never about a width the token cannot run.

## Not in scope

No performance claim: the estimate is not validated against measured wall time here, and a measured
`s(p)` comparison is the natural next step on a quiet host. The shape-dependent serial estimate
(a single rate cannot follow blocking and cache behaviour across shapes) is unchanged from the width
rule, which has the same limitation today.
