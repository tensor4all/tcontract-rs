# 2026-11-22 `FusedGemmPlan`: the fused grouped-GEMM route (#208)

The route that runs every job of a `t4a_desc::gemm::GemmBatch` in one packed
traversal. Design context: tensor4all-meta
`docs/design/symmetric-tensors-and-symtn.md` §3.1. The public surface was
proposed on #208 first; the positions are in the
[decision log](../decision-log.md#2026-11-22-the-fused-grouped-gemm-route-208).

## Shape of the implementation

- **No second loop nest.** An item runs the static driver's own walk
  (`static_grid::run_strip`) restricted to its rows and to one loop-5 panel;
  `run_strip` gained a `[jc_lo, jc_hi)` range for that, and every existing
  caller passes `0..n`. Packing, the micro-tile loops, the direct-B/direct-C
  branches, the first-slab rule and the write-back are the existing code.
- **Per-group state built once.** `driver::fused::FusedGroup` holds the
  group's serial call shape and its five block scatters (C's included, used
  only when `beta != 0`), so an execute builds no scatter and allocates nothing
  per job or per item. The per-call shape rule (`call_shape`) and the operand
  orientation (`Oriented`) were pulled out of `execute_once` so the packed
  driver and the fused route share one copy.
- **Per-worker buffers.** Each worker takes one `A + B` region and one tile
  region from the `Exec` workspace (`with_worker`) or call-local panels, sized
  to the largest group, and reuses them for every item it claims. `B` sits in
  the worker's own region after `A`, aligned to a line, because nothing is
  shared between workers.
- **Output-only work.** Jobs with `k == 0` (and every job when `alpha == 0`)
  take `C ← beta·C` on the calling thread, zero when `beta == 0`, unchanged
  when `beta == 1`, as `Plan`'s output pass does.

## Why the result is the per-job result

An item starts on a band boundary (a multiple of `MC`, itself whole `MR`
panels) and a panel boundary (a multiple of `NC`), and uses the group's serial
`KC`, so every output element sees the same micro-tile, the same K slabs in the
same order and the same write-back as the job alone on one thread. When the
blocked outer traversal applies (it permutes rows across the whole role), the
item is the whole job. The tests compare against the per-job `Plan` bitwise
and against a naive loop exactly, on small-integer inputs, for f32/f64/c32/c64,
ops N/T/C on A and B, `beta` zero and nonzero, padded leading dimensions, empty
`k`, empty outputs, and a blocking override small enough to split jobs into
several bands, panels and slabs.

## Verification notes

- Comparisons against the naive loop use exact value equality: its summation
  order differs from the kernels' only in the sign of a zero.
- Benchmark rows: `tcontract-bench` bin `fused_gemm` (fused against per-job
  `contract_batched` on the issue's shapes, 1T and 4T, with a bitwise check
  against the per-job packed result). Only smoke runs were made (on a laptop,
  3 samples) to check the harness; nothing here claims a speed-up.
- Review fixes (round 1): a direct-B family needs `b_group` sized as
  `execute_once` sizes it even though no panel is written, because
  `run_strip` checks every epoch's slivers against it; and geometries without
  work items are planned too, so kernel errors are not skipped.
