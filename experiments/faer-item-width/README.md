# faer-item-width

The time per call of a rank-2 Faer plan at each item width on a team token, and
the width the plan chooses
([tcontract-rs#194](https://github.com/tensor4all/tcontract-rs/issues/194)).
The rows of `crates/tcontract/src/strategy/faer_time.rs` are fitted from these
measurements, and the acceptance sweep checks the decision that uses them.

## Timed boundary

One `D[i,j] += sum_k A[i,k] B[k,j]` plan per shape, column-major, planned with
`FaerLimit::NONE` (every shape is a Faer plan) and `CSpec::Output(Identity)`.
The timed call is `Plan::execute_slices_accum` with `alpha = beta = 1` and
`SliceAccumulationSource::Output`; the identity output pass is skipped, so only
the GEMM and its dispatch are timed. Planning, allocation and the team are
outside the clock.

- `forced` rows: the plan on `team.exec(w)` with `BatchStrategy::Inner`, so faer
  gets exactly `w` threads.
- `auto` rows: the plan on `team.exec(budget)` with the default policy. The row
  carries `Plan::execution_width` at that budget (`chosen`), and the chosen
  width is also timed as a `forced` row.
- `DISPATCH` rows: the team token's `dispatch_cost(w)`, which the decision adds.

Per point: at least 40 ms of untimed warm-up calls (a wider previous point leaves
its members spinning for about 300k PAUSEs and the clock pulled down; without
this, width 1 read 1.37x slower when it followed width 96), then 7 to 21 samples
of back-to-back calls of at least 100 us each; the row is the median sample.
Before any timing every width's result from a zero `D` is checked against the
width-1 result (relative `1e-10`, `1e-4` for `f32`/`c32`); a mismatch aborts.

## Host and build

`lg` cluster nodes, exclusive (`place=excl`, `select=1:ncpus=96`): 2 x Intel Xeon
Platinum 9242 (`GenuineIntel` family 6 model 85), 96 CPUs, one thread per core,
4 NUMA nodes of 24 CPUs; `threadteam::HostClass::X86_64Avx512`. Release profile
of the workspace (thin LTO, one codegen unit), rustc 1.96.0. A team of 96
(`into_team(96)`). Two stacks were measured:

- **stack 1** (the `fit-`, `accept-base-`, `accept-` and `accept2-` runs, #194):
  threadteam 0.5.1, t4a-faer 0.33.0, t4a-private-gemm-x86 0.1.28;
- **stack 2** (the `fit2-` and `accept3*-` runs, #203, the stack of this tree):
  threadteam 0.6.0, colmat 0.34.0, t4a-private-gemm-x86 0.1.28.

The serial times are the same on both, but on stack 2 a parallel faer call of a
small product costs 5-7 us more (median `t(w) - t(1)` over the products with
`t(1) < 20 us`, f64: 0.9-1.8 us at widths 2 to 16 on `accept2-f64`, 5.6-8.9 us on
`accept3-old-f64`; c64 the same), so stack-1 rows do not describe stack 2. The
shipped rows are fitted on stack 2.
Each `job.log` records the node, `lscpu`, the binary's checksum and revision and
the load at start and end. Two rounds per job on one node, the second with the
shape, dtype and width order reversed.

## Runs

| directory | stack | binary | shapes | widths | budgets |
| --- | --- | --- | --- | --- | --- |
| `results/fit-<dtype>` | 1 | base `807183f` | `m, n, k` in {16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512} (1331) | 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96 | 16, 96 |
| `results/accept-base-<dtype>` | 1 | base `807183f` | `m, n, k` in {16, 20, 40, 80, 160, 320, 448, 512} (512) | 1 (+ chosen) | 16, 96 |
| `results/accept-<dtype>` | 1 | #194 candidate, first rows (constant slow share) | the same 512 | 1..=16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96 (+ chosen) | 16, 96 |
| `results/accept2-<dtype>` | 1 | #194 rows (main `20afa9d`) | the same 512 | the same | 16, 96 |
| `results/fit2-<dtype>` | 2 | `f7814e7` (forced rows only) | the 1331 of `fit-` | the 13 of `fit-` | 16, 96 |
| `results/accept3-base-<dtype>` | 2 | main `20afa9d` (#194 rows, f64 ideal) | the 512 | 1..=16, 20, ..., 96 (+ chosen) | 16, 96 |
| `results/accept3-old-<dtype>` | 2 | `f7814e7`: the #203 model with rows fitted on `fit-` (stack 1), f64 shipped | the same | the same | 16, 96 |
| `results/accept3-<dtype>` | 2 | `ef1c4d7`: the #203 model, rows fitted on `fit2-`, no gain margin, f64 shipped | the same | the same | 16, 96 |
| `results/accept3m-<dtype>` | 2 | `0cd8071`: the same rows with the gain margin, f64 shipped | the same | the same | 16, 96 |

The acceptance shapes are off the fitting grid except for 16 and 512. Forced
rows do not depend on the model, so `fit2-` was measured with the first
candidate binary. `f7814e7`, `ef1c4d7` and `0cd8071` are measurement builds
(their revisions are in each `job.log`; they are not in the merged history):
`ef1c4d7` and `0cd8071` are the two #203 commits with `DType::F64` added to
`XEON_9242_2S_SHIPPED`, and `f7814e7` the first of them with rows fitted on
`fit-`.

## Fit

`fit.py` fits, per dtype, `F` and `R` by least squares on the relative error of
the width-1 times, then grid-searches `O, C, E, Q0, Q1` over the (shape, budget)
pairs, the model's argmin taken over the measured widths <= budget of the model
plus the measured dispatch cost. The score counts the pairs within 5% of width 1
and of the best measured width, minus 3 per pair above 1.05 x width 1 and 30 more
per pair above 1.25 x width 1. The model and its mapping onto `ItemTime` are in
the module documentation of `strategy/faer_time.rs`: the task count, the largest
task's share and the packing band are faer's own partition rule
(`t4a-private-gemm-x86` 0.1.28 `lib.rs`, `gemm_impl` and `MilliPar::call`), not
fitted. `E` is the per-element cost of moving a task's operands, and `Q1` the
packing band's cost per participant in ns. After the fit, `item_fit.py` prints
the row's gain margin: the median `|model / measured - 1|` at the widths above
one it chooses (both budgets), which the decision uses as is.

Tried before: a NUMA-domain term, a row-block factor on the slow-regime
threshold, capping the width at the split (#194), and fitted output tiles
`TM, TN` with a packing band priced as a share of the work (the #194 rows).

The fit itself is provider-neutral: `item_fit.py` fits any `ItemTime` row from
rows in this probe's format, and `fit.py` supplies only faer GEMM's features
(its multiply-adds, its task partition, its per-task operand elements and its
packing band). A provider with another kernel passes its own features object to
`item_fit.fit`.

```bash
uv run --with numpy python fit.py f64 results/fit2-f64/round*.csv.gz
python3 accept.py results/accept3m-f64          # the acceptance table
python3 compare.py results/accept3-base-f64 results/accept3m-f64
```

`fit.txt` in each `results/fit2-<dtype>` is that output; the `fit.txt` in
`results/fit-<dtype>` is the #194 fit. The within-round noise floor (round 1
against round 2, same node): the median `|t1/t2 - 1|` over all points is 0.029
to 0.040 per dtype on stack 1 and 0.040 to 0.063 on stack 2 (`fit2-`).

## Acceptance (stack 2, #203)

Gate (maintainer, #203): on the 512-shape sweep no product above about 1.25 x
width 1 outside noise, and an f64 geo-mean t(chosen)/t(1) not worse than 0.592
(budget 16) / 0.549 (budget 96). Medians over the two rounds.

With the gain margin (`results/accept3m-<dtype>`, the rows as shipped; f64 measured
with its row enabled):

| dtype | budget | shapes | pass both | > 1.05 x width 1 | > 1.05 x best | chose width 1 | max t/t(1) | max t/best | median t/best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| c32 | 16 | 512 | 339 | 0 | 173 | 328 | 1.00 | 3.90 | 1.007 |
| c32 | 96 | 512 | 418 | 0 | 94 | 230 | 1.00 | 1.94 | 1.000 |
| c64 | 16 | 512 | 316 | 0 | 196 | 261 | 1.00 | 4.06 | 1.013 |
| c64 | 96 | 512 | 430 | 0 | 82 | 172 | 1.00 | 3.36 | 1.001 |
| f32 | 16 | 512 | 435 | 0 | 77 | 422 | 1.00 | 2.98 | 1.000 |
| f32 | 96 | 512 | 457 | 2 | 55 | 337 | 1.07 | 2.79 | 1.000 |
| f64 | 16 | 512 | 370 | 2 | 142 | 356 | 1.17 | 2.90 | 1.000 |
| f64 | 96 | 512 | 447 | 2 | 65 | 271 | 1.17 | 2.17 | 1.000 |

Without the margin (`results/accept3-<dtype>`, the first commit's rows):

| dtype | budget | shapes | pass both | > 1.05 x width 1 | > 1.05 x best | chose width 1 | max t/t(1) | max t/best | median t/best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| c32 | 16 | 512 | 354 | 2 | 158 | 281 | 1.45 | 2.51 | 1.002 |
| c32 | 96 | 512 | 452 | 0 | 60 | 208 | 1.00 | 1.87 | 1.000 |
| c64 | 16 | 512 | 328 | 5 | 184 | 193 | 1.21 | 2.04 | 1.013 |
| c64 | 96 | 512 | 398 | 5 | 114 | 147 | 1.21 | 2.85 | 1.003 |
| f32 | 16 | 512 | 440 | 0 | 72 | 393 | 1.00 | 1.84 | 1.000 |
| f32 | 96 | 512 | 472 | 1 | 40 | 318 | 1.11 | 2.24 | 1.000 |
| f64 | 16 | 512 | 386 | 3 | 126 | 312 | 1.28 | 1.80 | 1.000 |
| f64 | 96 | 512 | 464 | 4 | 48 | 248 | 1.28 | 1.80 | 1.000 |

Against main on the same stack (`python3 compare.py results/accept3-base-<dtype> results/<run>-<dtype>`;
main ships the #194 rows for f32, c32 and c64 and the ideal rule for f64):

| run | dtype | budget | shapes | main: > 1.05 x w1 | main: max t/t(1) | main: geo-mean t/t(1) | main: est. > 1.05 x best | run: > 1.05 x w1 | run: max t/t(1) | run: geo-mean t/t(1) | run: > 1.05 x best |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `accept3m` | f64 | 16 | 512 | 226 | 12.56 | 0.965 | 269 | 2 | 1.17 | 0.714 | 142 |
| `accept3m` | f64 | 96 | 512 | 207 | 12.56 | 0.767 | 347 | 2 | 1.17 | 0.541 | 65 |
| `accept3m` | c64 | 16 | 512 | 52 | 2.76 | 0.568 | 246 | 0 | 1.00 | 0.487 | 196 |
| `accept3m` | c64 | 96 | 512 | 35 | 1.65 | 0.330 | 130 | 0 | 1.00 | 0.321 | 82 |
| `accept3m` | f32 | 16 | 512 | 24 | 2.07 | 0.940 | 122 | 0 | 1.00 | 0.851 | 77 |
| `accept3m` | f32 | 96 | 512 | 44 | 2.07 | 0.706 | 129 | 2 | 1.07 | 0.682 | 55 |
| `accept3m` | c32 | 16 | 512 | 81 | 3.67 | 0.702 | 271 | 0 | 1.00 | 0.624 | 173 |
| `accept3m` | c32 | 96 | 512 | 51 | 1.86 | 0.445 | 223 | 0 | 1.00 | 0.434 | 94 |
| `accept3` | f64 | 16 | 512 | 226 | 12.56 | 0.965 | 283 | 3 | 1.28 | 0.684 | 126 |
| `accept3` | f64 | 96 | 512 | 207 | 12.56 | 0.767 | 369 | 4 | 1.28 | 0.524 | 48 |
| `accept3` | c64 | 16 | 512 | 52 | 2.76 | 0.568 | 248 | 5 | 1.21 | 0.459 | 184 |
| `accept3` | c64 | 96 | 512 | 35 | 1.65 | 0.330 | 120 | 5 | 1.21 | 0.334 | 114 |
| `accept3` | f32 | 16 | 512 | 24 | 2.07 | 0.940 | 128 | 0 | 1.00 | 0.829 | 72 |
| `accept3` | f32 | 96 | 512 | 44 | 2.07 | 0.706 | 149 | 1 | 1.11 | 0.667 | 40 |
| `accept3` | c32 | 16 | 512 | 81 | 3.67 | 0.702 | 273 | 2 | 1.45 | 0.590 | 158 |
| `accept3` | c32 | 96 | 512 | 51 | 1.86 | 0.445 | 263 | 0 | 1.00 | 0.414 | 60 |
| `accept3-old` | f64 | 16 | 512 | 226 | 12.56 | 0.965 | 265 | 73 | 2.42 | 0.716 | 149 |
| `accept3-old` | f64 | 96 | 512 | 207 | 12.56 | 0.767 | 333 | 67 | 2.42 | 0.570 | 102 |
| `accept3-old` | c64 | 16 | 512 | 52 | 2.76 | 0.568 | 237 | 64 | 4.66 | 0.477 | 190 |
| `accept3-old` | c64 | 96 | 512 | 35 | 1.65 | 0.330 | 111 | 52 | 2.38 | 0.352 | 127 |
| `accept3-old` | f32 | 16 | 512 | 24 | 2.07 | 0.940 | 123 | 51 | 3.09 | 0.842 | 80 |
| `accept3-old` | f32 | 96 | 512 | 44 | 2.07 | 0.706 | 112 | 57 | 11.62 | 0.697 | 70 |
| `accept3-old` | c32 | 16 | 512 | 81 | 3.67 | 0.702 | 246 | 64 | 4.18 | 0.608 | 176 |
| `accept3-old` | c32 | 96 | 512 | 51 | 1.86 | 0.445 | 182 | 50 | 4.47 | 0.448 | 102 |

The best measured width per shape bounds what any width choice can reach. Its
f64 geo-mean t(best)/t(1) is 0.508 / 0.434 (budget 16 / 96) on stack 1
(`accept2-f64`) and 0.646-0.657 / 0.513-0.533 on stack 2 (`accept3-old-f64`,
`accept3-f64`, `accept3m-f64`), so the issue's 0.592 at budget 16 cannot be met on
stack 2 by any rule. f64 therefore stays withheld (`XEON_9242_2S_SHIPPED`) until
the gate is restated for this stack; with its row it passes the 1.25 x bound and
the budget-96 number and is 0.714 / 0.541 against main's 0.965 / 0.767.

Products above 1.10 x width 1 (all runs on stack 2 with the #203 model; ratio is
the median of the rounds):

| run | dtype | budget | shape m x n x k | chosen width | t(1) us | t(chosen) us | ratio | ratio round 1 / round 2 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `accept3m` | f64 | 16 | 448 x 160 x 16 | 14 | 37.1 | 43.3 | 1.17 | 1.16 / 1.17 |
| `accept3m` | f64 | 96 | 448 x 160 x 16 | 14 | 37.1 | 43.3 | 1.17 | 1.16 / 1.17 |
| `accept3` | f64 | 16 | 320 x 160 x 20 | 14 | 28.4 | 32.2 | 1.13 | 1.26 / 1.02 |
| `accept3` | f64 | 96 | 320 x 160 x 20 | 14 | 28.4 | 32.2 | 1.13 | 1.26 / 1.02 |
| `accept3` | f64 | 16 | 448 x 160 x 16 | 14 | 37.6 | 41.8 | 1.11 | 1.19 / 1.03 |
| `accept3` | f64 | 96 | 448 x 160 x 16 | 14 | 37.6 | 41.8 | 1.11 | 1.19 / 1.03 |
| `accept3` | f64 | 16 | 512 x 80 x 20 | 9 | 25.7 | 32.9 | 1.28 | 1.41 / 1.15 |
| `accept3` | f64 | 96 | 512 x 80 x 20 | 9 | 25.7 | 32.9 | 1.28 | 1.41 / 1.15 |
| `accept3` | c64 | 16 | 320 x 40 x 16 | 14 | 24.2 | 29.3 | 1.21 | 1.13 / 1.29 |
| `accept3` | c64 | 96 | 320 x 40 x 16 | 14 | 24.2 | 29.3 | 1.21 | 1.13 / 1.29 |
| `accept3` | c64 | 16 | 512 x 20 x 16 | 11 | 20.3 | 23.7 | 1.17 | 1.15 / 1.18 |
| `accept3` | c64 | 96 | 512 x 20 x 16 | 11 | 20.3 | 23.7 | 1.17 | 1.15 / 1.18 |
| `accept3` | f32 | 96 | 448 x 320 x 16 | 30 | 34.4 | 38.2 | 1.11 | 1.18 / 1.05 |
| `accept3` | c32 | 16 | 40 x 320 x 320 | 5 | 178.0 | 207.4 | 1.17 | 1.19 / 1.14 |
| `accept3` | c32 | 16 | 80 x 320 x 160 | 5 | 177.2 | 256.2 | 1.45 | 1.60 / 1.29 |

The `accept3-old` rows (the #203 model with stack-1 rows) leave 51-73 products per dtype above 1.05 x width 1 on stack 2 (max 2.4-11.6 x), which is what the refit removes.


## Acceptance (stack 1, #194)

The #194 rows on their own stack. Gate (maintainer, #194): no shape above about
1.25 x width 1 outside noise. f32, c32 and c64 met it and shipped; f64 did not
(the four shapes below), which #203 prices.
"Within 5% of the best width" is reported as the model's target. Medians over
the two rounds; `python3 accept.py results/accept2-<dtype>`.

| dtype | budget | shapes | pass both | > 1.05 x width 1 | > 1.05 x best | chose width 1 | max t/t(1) | max t/best | median t/best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f64 | 16 | 512 | 357 | 4 | 155 | 251 | 2.34 | 4.18 | 1.002 |
| f64 | 96 | 512 | 327 | 2 | 185 | 235 | 1.27 | 2.87 | 1.009 |
| c64 | 16 | 512 | 220 | 0 | 292 | 123 | 1.00 | 2.92 | 1.176 |
| c64 | 96 | 512 | 362 | 1 | 150 | 94 | 1.07 | 1.97 | 1.004 |
| f32 | 16 | 512 | 313 | 0 | 199 | 425 | 1.00 | 6.98 | 1.000 |
| f32 | 96 | 512 | 409 | 0 | 103 | 272 | 1.00 | 3.94 | 1.000 |
| c32 | 16 | 512 | 244 | 7 | 268 | 159 | 1.12 | 2.85 | 1.058 |
| c32 | 96 | 512 | 385 | 7 | 127 | 126 | 1.12 | 1.54 | 1.004 |

Against the base on the same shapes (`python3 compare.py results/accept-base-<dtype> results/accept2-<dtype>`; geo-mean of t(chosen)/t(1), lower is better; the base's distance to the best is estimated through the candidate run's width-1 and best times):

| dtype | budget | shapes | base: > 1.05 x w1 | base: max t/t(1) | base: geo-mean t/t(1) | base: est. > 1.05 x best | candidate: > 1.05 x w1 | candidate: max t/t(1) | candidate: geo-mean t/t(1) | candidate: > 1.05 x best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f64 | 16 | 512 | 145 | 3.77 | 0.592 | 280 | 4 | 2.34 | 0.560 | 155 |
| f64 | 96 | 512 | 147 | 3.77 | 0.549 | 357 | 2 | 1.27 | 0.482 | 185 |
| c64 | 16 | 512 | 106 | 3.84 | 0.399 | 364 | 0 | 1.00 | 0.441 | 292 |
| c64 | 96 | 512 | 105 | 3.83 | 0.342 | 406 | 1 | 1.07 | 0.277 | 150 |
| f32 | 16 | 512 | 212 | 8.67 | 0.956 | 369 | 0 | 1.00 | 0.869 | 199 |
| f32 | 96 | 512 | 192 | 9.26 | 0.822 | 405 | 0 | 1.00 | 0.581 | 103 |
| c32 | 16 | 512 | 136 | 3.36 | 0.523 | 331 | 7 | 1.12 | 0.533 | 268 |
| c32 | 96 | 512 | 127 | 3.81 | 0.458 | 401 | 7 | 1.12 | 0.355 | 127 |

Remaining shapes above 1.15 x width 1 (all f64; f32, c64 and c32 stay at or below 1.12):

| dtype | budget | shape m x n x k | chosen width | t(1) us | t(chosen) us | ratio | ratio round 1 / round 2 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| f64 | 16 | 80 x 512 x 40 | 16 | 38.6 | 90.5 | 2.34 | 1.70 / 2.99 |
| f64 | 16, 96 | 80 x 40 x 448 | 4 | 32.0 | 40.7 | 1.27 | 1.26 / 1.29 |
| f64 | 16, 96 | 80 x 40 x 512 | 4 | 37.1 | 46.6 | 1.26 | 1.25 / 1.27 |
| f64 | 16 | 80 x 320 x 80 | 14 | 43.4 | 54.2 | 1.25 | 1.25 / 1.25 |

- `80 x 512 x 40` at width 16 is one width of faer's partition, not a regime:
  widths 4 to 14 run at 0.80 to 0.99 x width 1, width 15 at 1.08, width 16 at
  1.70 and 2.99 in the two rounds. The model has no term for it.
- `80 x 40 x 448/512` are narrow-`n` products where no width beats width 1
  (width 2: 1.24-1.27). The model splits them over two row tiles. Removing the
  split for outputs narrower than one column tile clears them, but it leaves
  large-`m` narrow-`n` products serial, up to 5x slower than their best width,
  so it was not kept.

### First rows (`results/accept-<dtype>`)

Before the slow band's cost grew with the width, the tail above 1.5 x width 1
was this. These are 20 to 90 us products, so it is not a small-shape effect.
At budget 16 the model chose the top of the packing band (widths 15 and 16)
because it priced the band at a constant half of the serial time, while the
measured cost grows with the width:

| dtype | budget | shape m x n x k | chosen width | t(1) us | t(chosen) us | ratio | ratio round 1 / round 2 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| f64 | 16 | 40 x 512 x 80 | 16 | 35.9 | 62.4 | 1.74 | 1.89 / 1.60 |
| f64 | 16 | 20 x 512 x 160 | 16 | 42.7 | 66.4 | 1.55 | 1.34 / 1.76 |
| c64 | 16 | 16 x 448 x 80 | 16 | 52.2 | 124.3 | 2.38 | 2.75 / 2.02 |
| c64 | 16 | 20 x 320 x 80 | 15 | 46.2 | 105.3 | 2.28 | 1.51 / 3.04 |
| c64 | 16 | 16 x 512 x 80 | 16 | 60.1 | 122.8 | 2.04 | 1.81 / 2.27 |
| c64 | 16 | 20 x 448 x 80 | 16 | 64.7 | 111.1 | 1.72 | 1.33 / 2.10 |
| c32 | 16 | 16 x 320 x 160 | 16 | 36.9 | 143.8 | 3.89 | 3.89 / 3.89 |
| c32 | 16 | 20 x 512 x 80 | 16 | 47.1 | 151.9 | 3.23 | 2.99 / 3.46 |
| c32 | 16 | 20 x 320 x 160 | 16 | 56.0 | 168.1 | 3.00 | 2.42 / 3.58 |
| c32 | 16 | 16 x 448 x 160 | 16 | 51.8 | 135.3 | 2.61 | 3.34 / 1.89 |
| c32 | 16 | 40 x 512 x 40 | 16 | 39.7 | 99.4 | 2.51 | 2.43 / 2.58 |
| c32 | 16 | 16 x 512 x 160 | 16 | 59.5 | 147.9 | 2.49 | 2.86 / 2.11 |
| c32 | 16 | 40 x 320 x 80 | 16 | 46.2 | 103.6 | 2.24 | 2.05 / 2.43 |
| c32 | 16 | 40 x 448 x 80 | 16 | 64.5 | 138.4 | 2.14 | 2.34 / 1.95 |
| c32 | 16 | 20 x 448 x 160 | 16 | 78.3 | 164.1 | 2.10 | 2.21 / 1.98 |
| c32 | 16 | 20 x 512 x 160 | 16 | 89.5 | 166.1 | 1.85 | 1.45 / 2.25 |

## Reproduce

```bash
cd experiments/faer-item-width && cargo build --release --locked
./target/release/faer-item-width --dtypes f64 --extents 16,24,32,48,64,96,128,192,256,384,512 \
    --widths 1,2,3,4,6,8,12,16,24,32,48,64,96 --budgets 16,96 --threads 96 --target-ms 100
```

`pbs/build.pbs` and `pbs/run.pbs` are the PBS jobs used (build once per
revision in a short non-exclusive job, then measure from the cached binary on an
exclusive node; `run.pbs` runs the two rounds).
