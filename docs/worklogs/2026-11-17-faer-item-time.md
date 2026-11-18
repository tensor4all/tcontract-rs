# faer item width from a provider time model ([#194](https://github.com/tensor4all/tcontract-rs/issues/194))

> The direction was set in the issue thread: the provider-owned time at a width
> goes into the one batch decision, the row is data fitted offline per host
> class, a host without a row keeps today's ideal-division rule (the maintainer's
> second reply; an earlier draft made it serial), `Plan::estimate` stays advisory
> and `Plan::execution_width` reports the same argmin. The ship gate is "not
> slower than width one": no shape above about 1.25x width 1 outside noise.
> This record covers the implementation and the choices made inside that
> direction.

## Choices made here

- **The hook is a data type, not a closure.** `BatchQuery` stays `Copy` and
  `PartialEq` without a lifetime: `ItemTime` holds `fixed_ns`, `work_ns`,
  `parallel_ns`, `split`, `per_width_ns`, `slow_below`, `slow_ns` and
  `slow_per_width_ns`, and
  `ItemTime::at(w)` evaluates them. Nothing in it is faer-specific; a provider
  with a setup cost, a bounded split and a slow regime below some width can
  describe itself.
- **One place decides one item's width.** `item_width` in `batch.rs` is used by
  every `Auto` arm that asked the token for an item width (the faer rule, the
  generic rule, the packed driver's *tiny* test). Without an `ItemTime` it is the
  token's rule unchanged, payback gate included; with one it is the argmin of
  `at(w) + sections * D(w)`, ties to the narrower width, and no payback gate,
  because the fitted row already prices the overheads the gate hedges. The lanes
  over items still divide ideally.
- **The model form.** The probe measured every width at 1331 shapes. faer's time
  is not a smooth function of the width: on a shape like `64 x 256 x 32` it is
  3.5-4x the serial time at widths 6 to 16 and half of it at 24. That band is the
  regime in which `t4a-private-gemm-x86` packs the left operand (`n > 3 * nr *
  n_threads`); the fit found its threshold at 12 columns per participant for
  every dtype (`nr = 4`). The model therefore charges a share of the serial time
  below `ceil(n / 12)`, on top of a bounded-split division (`ceil(m / 64) *
  ceil(n / 24..48)` tiles), a parallel setup cost of 2 to 15 us and a
  per-participant cost. A NUMA-domain term, a row-block factor on the threshold
  and a cap at the split were tried and not selected.
- **The tail.** The first acceptance run left shapes up to 3.89x width 1
  (c32 `16 x 320 x 160` at width 16, budget 16). They were not microsecond
  shapes: 20 to 90 us serial, 100 to 170 us at the chosen width, far above the
  3-4% noise, so a small-shape rule would not have reached them. They were one
  mispriced regime: inside the packing band the measured cost grows with the
  width (that shape: 1.3x at width 4, 3.3x at 8, 3.9x at 16) while the model
  charged a constant share, so at budget 16 it chose the top of the band. The
  band now has a share per participant beyond the first (`slow_per_width`), and
  the fit penalises any choice above 1.25x width 1 thirty times more than a pass.
- **The serial rate is faer's, per dtype.** 41.7 (f64), 41.3 (c64), 83.8 (f32)
  and 84.7 (c32) multiply-adds per ns, against the host table's 15.6 for the
  class; the table (packed driver, `Plan::estimate`) is unchanged.
- **The class key.** CPU vendor, family, model and logical CPU count from
  `/proc/cpuinfo`, read once. Not `threadteam::HostClass` (ISA-level), and not
  CPUID through `tcontract-kernel`'s tuning-class decode: that is crate-private
  there, and std's CPU count follows the affinity mask, so a job given 16 CPUs of
  the fitted machine would read as another class.
- **The stage-one narrowing.** `Plan::batch_budget` narrowed every `Auto` plan by
  the ideal rule over the whole plan before the faer decision. On a team that cap
  excluded exactly the widths past the slow regime (fit: c32 at budget 96, worst
  regression 3.08x with the cap, 1.16x without), so a Faer plan on a team is no
  longer narrowed. Without a team it still is: it bounds the lanes, and
  `tests/execution_width.rs` pins that.
- **Warm-up in the probe.** A first run read width 1 1.37x slower when it
  followed width 96: the members of the previous wide call were still spinning
  (threadteam's 300k-PAUSE budget) and pulled the clock down. Every point now
  starts with at least 40 ms of untimed calls; the round-to-round noise fell
  from a median 7-9% to 3-4%.

- **f64 is withheld.** The second acceptance run met the 1.25x gate for f32,
  c32 and c64 but left four f64 products above it, the worst a single slow
  width of faer's partition (`80 x 512 x 40` at width 16). The maintainer kept
  f64 on the ideal-division rule; its row stays in the table behind
  `XEON_9242_2S_SHIPPED`, so enabling it is one line.

## Verification

Results, the acceptance table and the remaining limitations are in the pull
request and in `experiments/faer-item-width/README.md`.
