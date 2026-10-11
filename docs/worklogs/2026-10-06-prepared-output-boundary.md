# Prepared output boundary validation (2026-10-06)

Base: merged #71 (`bb91172`); branch `prepared-output-boundary`. This is the
owning-layer prerequisite of cpueinsum-rs#4. tenferro's paths are untouched.

The accepted cross-repository pre-review used gpt-6.1-sol and required
MaybeUninit elementwise leaves, a prepared reference-free alternative for Faer,
reduced-label coverage, consistent C/beta semantics and actual route reporting.
A focused pre-review accepted default-off `fresh_output` after unconditional
preparation broke the existing <=2 planning-allocation gate (59 allocations).
The gate was preserved, not relaxed.

The completed gpt-6.1-sol post-review traced production leaves and found:

1. Shared-derived destination pointers at the two new safe slice boundaries.
   Fixed with `as_mut_ptr`; raw const address validation preserves provenance.
2. Invalid unused terminal pointer advances in pinned strided map leaves.
   Fixed in the owning repository, strided-rs#290, now pinned consistently to
   `722dc6bece0c845ee612fa21242521c2e5ea83ac`.
3. A C batch could write an initialized item before a later fresh route failed
   same-pool SPMD admission. All actual per-item routes now preflight inside the
   executor closure, before the first write. Mixed-source regression reproduces
   the earlier write without that loop and passes with it (observed Unsupported
   status 15 and both output sentinels unchanged).

Main audit also corrected unused beta-zero packed C pointer arithmetic: both
its base and H offsets now use D. A direct guard regression fails without the
normalization and passes with it; the safe fresh boundary exercises absent C
with an otherwise large C batch gap. Negative zero normalizes; NaN/nonzero beta
retains the actual C mapping.

The completed focused gpt-6.1-sol corrective review closed all three findings,
accepted the owning strided fix, found no new findings or justified deletions,
and required corrected-pin integration and deterministic gates before publish.
Those gates passed on the actual published strided revision (no path patches):

- workspace fmt check, C API build;
- workspace/all-target clippy `-D warnings`;
- workspace debug tests and standalone `tcontract-exec` tests;
- release `tcontract` tests, including the original planning allocation gate;
- workspace/all-target `aarch64-apple-darwin` check;
- Rust 1.89.0 workspace/all-target check;
- workspace docs with `RUSTDOCFLAGS=-D warnings`.

Commands used `CARGO_BUILD_JOBS=16 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`;
compile jobs are not a numerical thread baseline. C tests explicitly lend
serial/pooled executors. Full logs are `/tmp/tprims-stage2-pinned-*.log` in the
implementation environment. Hosted CI remains authoritative for its matrix.
No Miri/sanitizer run, general safety proof, or speedup claim is made.

Numerical downstream tests additionally exercise all four dtypes, all sixteen
A/B/C/D conjugation masks, all C modes, initialized/fresh output, K=3/513,
zero-scale NaNs, negative strides and pooled broadcast/reversal. cpueinsum's
final gate is separately rerun against this prerequisite's published revision.

Provenance: local changes reusing existing tcontract and strided APIs; no external
bodies copied or notices removed. Existing tensorprimitives/Lukas Devos and
strided lineage remains applicable. No extra bridge, ambient pool, hidden fill,
result temporary, execution-time planning, provider probe or serial retry.
