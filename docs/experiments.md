# Experiment plan

The experiments are separate prototypes. None is a production crate. Keep raw measurements, source revisions, build settings, and correctness checks with each result. Decide whether to extract a reusable library only after the interface and at least one non-tenferro consumer are concrete, consistent with [tenferro #1927](https://github.com/tensor4all/tenferro-rs/issues/1927).

## Shared rules for comparisons

- Pin the exact commit/version and build configuration of every provider (especially BLIS's chosen ISA/configuration, OpenBLAS thread settings, and TBLIS's linked BLIS). Never compare numbers collected on different machines as if they were a paired experiment.
- Record CPU model, ISA, cache topology, core affinity, SMT, compiler, release profile, matrix shape, strides, dtype, batch size, thread count, warm/cold state, and whether packing and allocation are inside the timed region.
- Run a correctness gate before timing: exact small examples, LU/Cholesky/QR reconstruction residuals, solve residuals, singular or non-positive-definite inputs, empty and rectangular cases, complex values where supported, and noncontiguous views.
- Keep operation-only, preparation/scratch, executor-entry, and public/FFI call costs separate. Also report end-to-end latency; an impressive kernel number cannot substitute for it.
- Predeclare the representative case list and comparison statistic. Use paired runs and a within-session noise estimate for small differences. Retain unfavorable and inconclusive results.
- Test at least serial and several requested thread counts. Compare a caller-owned outer pool with provider-owned inner threading and measure oversubscription rather than assuming it is harmless.

These rules follow the [archived tensorprimitives-rs measurement discussion](archive/tensorprimitives/results.md) and the [shared tensor4all performance protocol](https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/common/performance.md). They are a starting protocol, not a substitute for a pre-registered case list for each comparison.

## Prototype 1: execution and FFI boundary

**Question:** Can a caller keep an executor/session alive while making many small calls, without a per-call worker handoff or hidden global pool?

Implement minimal `f64` work that can be called through (a) direct Rust, (b) an explicit caller-owned Rayon pool, and (c) a C ABI from a real C program. Measure an empty call, a small GEMM-like operation, and a batch with fixed total arithmetic but varying item count. Include direct serial execution, one pool entry per item, and one entry per batch. Keep pool construction outside the timed operation unless reporting construction cost separately. Check that the host thread identity and thread-local behavior match the API contract.

The roughly 8 to 14 µs session entry in [tenferro #1945](https://github.com/tensor4all/tenferro-rs/issues/1945) is a motivating observation, **not** a predicted result here. Its sub-1 µs fixed-cost target is a design target to test, not an achieved property.

**Preliminary result (2026-09-29):** [Rayon entry latency](../experiments/rayon-entry/README.md) on an M5 Max separates worker wake-up (6 to 8 µs), caller wake-up (about 3 µs) and full-width fan-out (55 to 200 µs at 18 threads), and compares OpenMP. It led to the entry-cost decision in the [decision log](decision-log.md#execution). Still to do: C host, real small kernels, a homogeneous x86 CPU.

**Decision gate:** A viable API makes pool ownership, thread budget, session lifetime, and C error behavior explicit. It must show its measured fixed cost and avoid nested oversubscription.

## Prototype 2: batched GEMM and factorizations

**Question:** For which matrix and batch sizes do serial-per-matrix kernels, specialized batched kernels, and provider calls win?

Start with a correct scalar reference for small GEMM and Cholesky, then LU and QR. Try an array of independent matrices versus an interleaved batch layout; measure packing and scratch reuse. Compare the Phase 1 implementations (faer plus a loop over items, TBLIS-style batched GEMM), serial outer loop versus caller-owned Rayon batch scheduling, and provider threading. Use BLIS/OpenBLAS, faer, and gemmkit as external baselines where the operation and dtype are available. Do not copy their tests into this repository without a separate provenance review.

Cover at least small square and rectangular cases, batch sizes from one to many independent matrices, `f32` and `f64`, and a complex follow-up. Report latency per batch as well as aggregate throughput. [Haidar et al. §4.1](https://www.netlib.org/utk/people/JackDongarra/PAPERS/batched-matrix-comp.pdf) motivates the serial-per-matrix baseline; [the CPU batched-GEMM paper](https://arxiv.org/abs/2311.07602) motivates cache- and shape-sensitive alternatives. Neither fixes a universal crossover.

**Decision gate:** Correct reconstruction and residuals, clear threading ownership, and a measured region where a custom batch path earns its complexity. A provider is acceptable if it wins end to end.

## Prototype 3: direct tensor contraction

**Question:** When does TBLIS-style packing of general strides beat reshape/transpose-then-GEMM, and can the execution contract be shared with Prototype 2?

Build a small fixed corpus of labeled binary contractions with contiguous, permuted, and irregular strides. Include real and complex cases and report input/output bytes, conversion bytes, scratch, and end-to-end time. Compare the two strategies ported in Phase 1 (tenferro-rs permute plus batched GEMM, the TBLIS-style direct driver from tensorprimitives-rs) with TBLIS itself. Include tiny contractions where executor entry dominates and larger contractions where packing and cache behavior dominate.

Use [Matthews's TBLIS paper](https://arxiv.org/abs/1607.00291) for the algorithmic idea and the archived [tensorprimitives-rs notes](archive/tensorprimitives/README.md) as an independent point of comparison. Its D49/D50 ([decisions](archive/tensorprimitives/decisions.md)) make an important distinction: a barrier-driven inner path has different scheduler needs from a barrier-free batch.

**Decision gate:** A reproducible win on a predeclared shape family, or a measured reason to use an existing provider. If direct contraction adds no meaningful value, do not build another general engine merely to have one.

## Prototype 4: C ABI slice

**Question:** Does the design hold across the C boundary: an explicit executor (`tcontract_exec*` then, `TAPP_executor` since 2026-10-01) per call, DLPack views, one shared library?

Build the Phase 1f slice (a C ABI crate; it is now `tcontract-capi`, whose contraction part is TAPP) and a C benchmark program. Measure per-call fixed cost of an empty call and a small contraction through C against the same calls from Rust, both serial and with a pool created from C. Verify zero copy for strided and column-major `DLTensor` inputs by pointer identity, pool create, use and close (including `TCONTRACT_BUSY`), and that an executor handle is accepted by every plan of the library. Record per-call validation cost separately from kernel time.

**Decision gate:** C-side fixed cost within a small, recorded margin of the Rust call, no hidden copies, and no ABI change forced by the benchmark. Otherwise revise the design before Phase 2.

## Architecture decision after the prototypes

Record the evidence for: provider vs own kernel; Rayon vs specialized worker pool for each parallel axis; explicit Rust executor API vs C ABI handle/callback; per-call scratch vs caller-owned scratch; and separate library vs code remaining in tenferro. The result may be multiple narrow crates or no new production crate.
