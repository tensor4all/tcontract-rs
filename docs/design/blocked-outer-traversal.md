# Matching the reference on the transposed-output contractions

Status: consolidated design, four pre-reviews taken into account. Nothing here is
implemented. Every claim marked *measured* was taken on this host with the pinned
binary and the opt-in phase instrument (`--features phase-timing`,
`TCONTRACT_PHASE=1`); everything else is a design statement and is labelled as one.

## 1. The problem, measured

`abjc-cbka-kj` f64 16 MiB 1T: 43.1 ms per call, of which `pack_a` 27.6 ms (64%),
`kernel` 14.3, output stores 1.8. TBLIS, through its own packm/gemm kernels
registered into BLIS, does 24.8 ms. Layouts from the harness's own `la`/`lc`:
`A` is `(a s1, b s48, j s1920, c s92160)`; the output is `(c s1, b s48, n s1920,
a s76800)`. So in the m group the operand and the output are **transposed**: `A`
is `a`-fastest, the output is `c`-fastest.

Measured alternatives, all dead ends:

| what was tried | result |
| --- | --- |
| m ordered by the **input** stride | `pack_a` 27.6 -> 4.5 ms, but output stores 1.8 -> 30.2, total 54.2 |
| order the pack's visit within the tile (row sort) | identity permutation on this case; nothing changes |
| one-step prefetch of the next k/sliver | 29.4 against 28.1 ms |
| rotate the lane start per k step | `pack_a` 45.6 against 27.6 |
| blocking model, MC/KC/NC, orientation, partition, `align_c_lines`, MR sweep | no effect (mr=8: +5%) |
| the same case with a one-element-padded leading dimension | 23.3 ms, `pack_a` 9.7 - the phase, not the kernel |

An offline address model (per k-step, per panel, resolved MR=24/KC=48) counts 1.0
useful element per touched 64-byte line today against the 8.0 a contiguous read
allows, and 4.42M pages against 98k for an `48a x 8c` block. The measured
`--stress padded` column shows what recovering that is worth.

## 2. What the four pre-reviews established

1. **A pack-side change cannot fix it.** With output-ordered rows the pack reads
   one element per line; with input-ordered rows the output stores do. The panel
   layout (`out + (s*kc + p)*vr*r + t*r`) is consistent with the packer writing
   the row order it reads, so source-ordered packing is fine for every
   `PackFormat` and needs **no** `PackFn` change. The transposition therefore has
   to happen downstream, in the output path.
2. **A per-tile permutation cannot fix it.** A tile is 24 rows of one axis run;
   the earlier drafts' intra-tile row sort degenerates to the identity. The same
   argument kills a per-`MC`-block permutation too: for this case a source-linear
   `MC`-row block (input order `a,b,c` with extents 48,40,48) contains `c` only
   once per 1920 rows, so it cannot contain a destination-contiguous `c` run.
3. **Membership, not order, is the lever.** A block must span enough of **both**
   axes to hold a contiguous source run *and* a contiguous destination run: e.g.
   all 48 `c` for a few `(a,b)` pairs. That is a *multi-dimensional* (axis-aware)
   block, which is what the reference's outer blocking does and what this
   implementation does not have today: `static_grid.rs` walks consecutive slices
   of one flattened row scatter.
4. **Cross-K persistence forces an ownership design.** Today the static traversal
   is `jc -> pc -> all ic blocks`, each slab producing an overwrite or accumulate
   tile, with barriers placed in identical `(h,jc,pc)` loops. Accumulating a whole
   block before emitting means either keeping every owned row block across K
   (roughly `owned_rows x owned_columns`), or moving K inside a block round, or
   restricting eligibility (e.g. a single K slab). Dynamic scheduling compounds it:
   the claim counter resets each epoch, so the same job can move between workers
   between slabs.
5. **Workspace and admission are not free.** The arena rounds to powers of two, so
   an 83 KiB block costs 128 KiB of retained capacity; an out-of-place permutation
   needs a second region; the traversal of 349 blocks costs about 58 MB of extra
   traffic, not 29 MB; and the existing seam test proves *route* refusal, not a
   capacity refusal, which does not exist yet.
6. **Numerical claims must stay within the rules.** The project requires exact
   serial/static/dynamic/width equivalence for a fixed arithmetic configuration;
   the corpus and harness comparisons are tolerance-based (1e-4..1e-12) with an
   exact known-value check. Delaying `alpha`, `beta`, complex recombination and
   conjugation to a final emission changes rounding and signed-zero behaviour, so
   it must be defined, not assumed away.
7. **No promotion on a regression.** Input-priority ordering alone is 26% slower
   end to end; it can only be an experimental switch until the full path wins.

## 3. The design that follows

**Axis-aware outer blocks, with the transpose moved to a block-level output
scratch.**

- **Membership.** A block is `{ per-axis ranges over the oriented m role's axes,
  per-axis ranges over the n role's axes, batch, K slab }`, chosen so that it
  contains whole runs of the packed operand's fastest axis **and** the output's
  fastest axis - for the measured case, all 48 `c` and 10 `(a,b)` pairs
  (`MC = 480` rows, an `a`-run of 10 elements per `(b,c)`, a `c`-run of 48). The
  offline model is used to pick the shape, and eligibility is a conservative
  predicate with a fallback to today's path.
- **Panel rows** stay in the **output** order, so the kernel, the emitter and the
  write-back contract are untouched.
- **Packing** reads the operand in the operand's order and writes the panel in the
  panel's order; the existing scatter and block-scatter machinery does this
  unchanged (the pack side is already efficient once the row order is settled),
  which is why no pack descriptor and no ABI change are needed.
- **Output.** The kernel accumulates the block's K slabs into an `MC x NC`
  scratch; a block-level transpose reorders that scratch into the output's order,
  and the existing emitter writes `D` contiguously - the same work it does today,
  fed a block instead of a tile. Workspace: one extra block region, checked and
  retained at the next power of two.
- **Ownership.** One of the three options from §2.4 must be chosen and written
  down; my preference is the narrowest that can win: a block round with K inside
  it, so a worker keeps one scratch per owned block, with the barriers restated on
  common rounds including idle participants, and direct-B decided for the whole
  call before any allocation (with the new N path disabling it on doubt).

## 4. Gates

Correctness first: the local gate; then old/new equivalence established by an
independent oracle (tolerance) with the partition-bitwise property and
width/serial equality retained exactly for a fixed configuration; all formats,
dtypes, conjugations, separate/in-place C, signed scatters, per-axis tails,
multiple K slabs, batches, and every route. Mechanism gates: destination run
lengths after the block permutation, exactly-once accumulation across K and across
dynamic job migration, unequal static row strips with empty column groups,
fresh-output/no-read-before-write, beta-zero C, and full checked scratch
accounting. Non-regression: the 49-case TCCG suite at 1T/4T, both dtypes, `plan`
and forced `packed`, none/padded/ragged, plus the per-shape corpus. Attribution
with disjoint phase scopes and an uninstrumented headline, thresholds declared
beforehand. Then both campaign cells are re-recorded.

## 5. Order of work

1. The offline geometry model, to fix the block shape and the expected runs, and
   to reject shapes whose destination runs are not contiguous (this is what the
   last four reviews asked for before any code).
2. The axis-aware block enumerator and block-local scatters, behind an
   eligibility switch, with the existing suites green.
3. The block scratch and the block-level transpose with postponed emission, one
   K slab first (the narrowest eligibility), then the ownership round.
4. The gates, then the campaign re-measurement.

## 6. What this costs, honestly

Steps 2-4 touch the driver's enumerator, both schedulers, ownership and barriers,
capacity and admission, the emitter interface and the reporting fields. That is an
architectural change of the kind the project writes a design for, not a patch, and
it will span more than one session. The cheap alternative that is already measured
is the caller-side layout: the campaign's TCCG sizing rounds stride-1 extents to
multiples of 24, which makes this case's source stride exactly 180 pages; the
one-element-padded variant of the same corpus is 23.3 ms against the reference's
20.0 ms, with no library change at all.

## 7. Narrow eligibility, and why it removes the hardest half

The fourth review's heaviest findings were about cross-K persistence, ownership and
workspace: accumulating a block over all K slabs means either keeping every owned row
block across K, or moving K inside a block round with restated barriers and a
specified B-publication lifetime.

The measured case needs neither. Its `K` is 48 against `kc = 256` (one slab) and its
`n` is 40 against `nc = 1536` (one NC panel), so a block's accumulator is complete
inside the single `pc` iteration the driver already performs, and the block's whole
output is one contiguous write. Restricting the blocked path to `k <= kc && n <= nc`
therefore removes the cross-slab, ownership, barrier and B-lifetime work from the
change entirely. The price is scope: it applies only where those conditions hold.
That is still the class the campaign measured - 16 of 196 TCCG rows lose to the
reference - and every other shape, including every multi-slab contraction, keeps
today's path unchanged by construction.

Eligibility is therefore computed where the resolved blocking is in scope (execution
initialization), from the plan's oriented role axes plus `mc`/`kc`/`nc`:
`driver/block.rs::blocked_eligibility` returns `Some` only under those conditions,
with `role` mapping the oriented axes to their operand and output strides. Both are
implemented and tested (including the swapped-role case, which has a single-axis row
role here and is therefore never eligible).

## 8. Wiring plan, with signatures

The mechanism is complete and tested: `driver/block.rs` (membership, eligibility, output
order, gather) and `tcontract_kernel::pack::permute` (format-exact row permutation across a
tile grid). What remains is the driver, in four mechanical steps, each gated on its own.

1. **Row views instead of global intervals.** `pack_a_rows` and `compute_block` currently
   index the epoch's global scatter vectors by `ic..ic + ic_len` and `i0 / mr`. Give them
   the rows as slices with a live count:

   - `pack_a_rows(cx, ep, a_m: &[i64], a_m_bs: &[i64], live: usize, ap)`
   - `compute_block(cx, ep, bufs, cm: &[i64], dm: &[i64], m_bs: &[i64], live: usize, jr_lo, jr_hi)`

   and index them from zero (`ir`, `ir / mr`). `static_grid.rs` slices the global vectors
   exactly as today, so this step changes no behaviour, and the driver's own suite -
   `partition_bitwise` above all, which pins serial/static/dynamic/width equality - is the
   gate. This is the "identity view" step the reviews approved.
2. **Collect instead of emit.** `compute_block`'s non-direct branch writes each finished
   tile into a tile-grid slot instead of calling `emit_tile`, under a flag only the blocked
   path sets. A tile copy is a straight copy of `reals * mr * nr` values - same format,
   same geometry, no recombination - so collecting needs no format knowledge and cannot
   change a value.
3. **The blocked branch.** In `static_grid.rs`, when the plan is eligible
   (`block::blocked_eligibility`, computed in `driver/mod.rs` where the axes and the
   resolved blocking meet) and the K slab and NC panel are single, the `ic` loop becomes a
   loop over blocks: gather the block's row scatters (`block::rows`, `block::gather`),
   rebuild its block-scatter vector from the gathered operand scatter, pack and collect all
   its tiles, then permute the grid (`block::output_order` + `permute::permute_grid_rows`)
   and emit each output-ordered `mr`-row tile with the permuted row scatters, reusing
   `emit_tile`. Direct-C and direct-B are disabled on this path, which is why eligibility
   is decided for the whole call before any allocation.
4. **Workspace.** One tile-grid region of `(MC / mr) * (NC / nr) * reals * mr * nr` values -
   120 KiB for the measured case, 128 KiB after the arena's power-of-two rounding -
   requested with the other panels, with a checked product and an admission that still
   leaves `D` untouched if it is refused.

Gates, in order: the mechanism tests already landed; byte-identical or
oracle-established output against today's path for a fixed family, orientation, blocking
and K order; then the 49-case TCCG suite at 1T and 4T, both dtypes, `plan` and forced
`packed`, none/padded/ragged, plus the per-shape corpus; then the phase attribution with
disjoint scopes and an uninstrumented headline; then both campaign cells re-recorded.

## 9. The reference's own domain, measured

The question "where is the outer-blocked, block-scatter approach the right one" can
be asked of the reference itself, because the campaign's tcbench cell carries TBLIS
per case. Over its 196 rows (16 MiB, 1T and 4T, two dtypes), TBLIS beats this
library's planner in **16**, and they are not scattered:

| shape class | rows where TBLIS wins | median TBLIS/plan |
| --- | --- | --- |
| `n <= 64` (thin output) | 12 of 28 | 1.01 |
| `n > 64` | 4 of 168 | 1.21 |
| `k <= 48` **and** `n <= 64` | **12 of 24** | 0.99 |
| `k <= 48` and `n > 64` | 0 of 84 | 1.25 |

So the reference's domain is **a thin output together with a shallow K** - the
transposed-output rank-5 and rank-6 cases (`abjc-cbka-kj`, `adbjc-cbdka-kj`,
`ajbc-ckba-jk`, `ijk-il-jlk`), where the output's fastest axis is long while the
operand's is short and the pack read is the whole story. That is exactly the class
this design's mechanism targets, and the two classes agree for the same reason: it
is the class where the packed operand's reads touch one element per cache line.

It is *not* where the current implementation pays. With the switch on, over the same
98 one-thread rows, the blocked path helps 2 and hurts 94 (median 25% slower), and the
two it helps are the two thinnest, most strided cases (`adbjc-cbdka-kj` f64 1.13x,
`abjc-cbka-kj` f64 1.08x). The mechanism is measured - `pack_a` 27.6 -> 5.5 ms - but
the implementation's own costs (a per-block `Vec` churn worth ~9 ms of a 23 ms call,
and the block-level transpose) eat it everywhere else. Both are known and bounded:
the allocations become stack arrays bounded by `BLOCK_MC_BUDGET`, and the transpose is
one pass over a block.

Therefore the rule for promotion is the intersection of the two domains: the
eligibility must select the thin-output, shallow-K class *and* the implementation must
first stop allocating per block. Until both hold, the switch stays off.

## 10. The allocation fix lands; promotion still does not

Moving `run_block`'s local metadata from per-block `Vec`s to stack arrays bounded by
`BLOCK_ROWS_MAX` (= `BLOCK_MC_BUDGET`, 480) changed the picture from "the mechanism is
right but the implementation loses" to "the implementation is nearly free":

| 1T, 16 MiB, `--engines packed`, same session | helped | hurt | neutral | median |
| --- | --- | --- | --- | --- |
| before the fix | 2 | 94 | 2 | ×0.80 |
| after the fix | 8 | 15 | 75 | ×1.00 |

and the measured case went 43.4 -> 31.3 ms (11.2 GF/s against 8.2 off). Two bugs came
out of the same work and are now fixed: the row map must be sorted over `live` rows
only and padded after (a `live % mr != 0` shape indexed past the end), and the grid
slot must be the *global* column tile (`jr_lo + jt * nr`) or two column-group workers
overwrite each other's tiles.

But promotion needs a profitability test, and the geometry does not give one. Taking
`BlockShape` per case, a win and a loss can be identical in every field this design
records:

| case | dtype | `span` | `line` | `line_len` | `grid_elems` | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| `adbjc-cbdka-kj` | f64 | 0 | 3 | 1 | 230400 | **×1.22** |
| `abjcd-dkbac-jk` | f64 | 0 | 3 | 1 | 230400 | **×0.74** |
| `abjc-cbka-kj` | f64 | 0 | 2 | 1 | 76800 | ×1.15 |
| `abjc-kbac-jk` | f64 | 0 | 1 | 1 | 92160 | ×0.73 |
| `abj-bka-kj` | f64 | 0 | 1 | 3 | 55296 | ×0.77 |

`line_len == 1` (the configuration the design is written for) holds on both sides, and
`line >= 2` holds on both sides too. The remaining difference has to be in how the
*operand* is laid out inside the block - the scatter regularity the pack loop walks -
which this design does not yet record and which is therefore the next measurement, not
a guess to be baked into an eligibility rule.

So the switch stays off, and per the standing instruction this investigation is not a
PR: it does not improve any published number yet. What it does deliver is the
mechanism, measured and bounded (`pack_a` 27.6 -> 5.5 ms), a cost-neutral
implementation, two fixed memory bugs with their gate, and the reference's own
applicability range (§9) to aim at.

## 11. Where the loss is: the per-block fixed cost, still in `setup`

The role axes explain the two indistinguishable cases after all, but not through the
four `BlockShape` numbers: they differ in *strides*.

| case | role axes `(extent, src, dst)` | ratio |
| --- | --- | --- |
| `adbjc-cbdka-kj` f64 | (24, 230400, 1) (20, 480, 24) (20, 24, 480) (24, 1, 192000) | ×1.22 |
| `abjcd-dkbac-jk` f64 | (24, 9600, 1) (20, 480, 24) (20, 230400, 11520) (24, 1, 230400) | ×0.74 |

The phase instrument then says where the difference lands. With the switch on, the
unmeasured remainder (`setup` = wall minus the four phases) is the whole story:

| case | switch | total | pack_a | kernel | writeback | setup |
| --- | --- | --- | --- | --- | --- | --- |
| `adbjc-cbdka-kj` | off | 51.7 | 35.3 (68%) | 14.4 | 2.5 | 0.0 |
| `adbjc-cbdka-kj` | on | 39.5 | 11.2 (28%) | 7.7 | 0.8 | **19.8 (50%)** |
| `abjcd-dkbac-jk` | off | 23.2 | 11.2 (48%) | 10.5 | 1.9 | 0.0 |
| `abjcd-dkbac-jk` | on | 33.0 | 4.7 (14%) | 6.8 | 0.8 | **20.6 (62%)** |
| `abjc-cbka-kj` | off | 38.1 | 26.1 (68%) | 11.2 | 1.4 | 0.0 |
| `abjc-cbka-kj` | on | 33.9 | 15.8 (47%) | 8.5 | 0.6 | **9.1 (27%)** |
| `abjc-kbac-jk` | off | 16.0 | 4.5 (28%) | 10.6 | 1.3 | 0.0 |
| `abjc-kbac-jk` | on | 23.6 | 5.6 (24%) | 8.4 | 0.5 | **9.1 (38%)** |

`pack_a` falls exactly as the mechanism predicts (35.3 -> 11.2, 26.1 -> 15.8), and the
loss is not there. It is in `setup`: the block's own fixed work - the three gathers,
the scatters and above all the grid -> grid2 permutation, which is a second pass over
the block's output - is 9 ms on the shapes that win and 20 ms on the four-axis shapes
that lose. So profitability is a question about **work per block** (block rows times
`n`, against a fixed per-block cost), not about the line geometry, and the descriptor
to record next is the block count and the block's output size - the two the design's
`BlockShape` still does not carry.

This is the blocker, stated plainly: the mechanism works and the implementation is
nearly free, but the per-block fixed cost has to be paid down (a permutation that
does not copy the whole block, or fewer, larger blocks) before any eligibility rule
can be a profitability rule. The switch stays off and this stays out of a PR.

## 12. Why it is a trade, and why that ends the promotion

The `setup` remainder is not an artifact (it is 15 ms at `--prime-ms 0` and at
`--reps 20` alike), it is not the permutation (removing the grid -> grid2 copy
entirely, by emitting each panel tile with its destination tile's own rows and
scatters, changed `setup` by nothing: 14.7 -> 15.0 ms), and it is not the per-block
metadata (the block geometry that had ten times too many blocks - a `per_block` that
divided by the other axes as well as the span axis, giving `line_len = 1` instead of
the design's 8 - was fixed, and `setup` *grew*, because the blocks only got bigger).

What is left is the work the instrument does not cover, and there is exactly one
candidate that scales the way `setup` scales. The blocked path moves the line problem
from the pack to the write-back: the block's rows are gathered in the operand's order,
so the *loads* become contiguous (`pack_a` 35.3 -> 5.3 ms, a five-fold fall) and the
*stores* become the strided side instead. The two are the same problem with the roles
exchanged, so the path is a trade of strided loads against strided stores, and it only
pays when there are more operand elements than output ones - `m*k` against `m*n`, i.e.
roughly `k` against `n`. That is also exactly the domain TBLIS's own advantage lives
in (§9: thin output, shallow K), for the same reason.

Fitting that to the corpus gives a rule that is real but not worth having:

| eligibility | selected | helped | hurt | median |
| --- | --- | --- | --- | --- |
| all rows | 98 | 8 | 15 | ×0.998 |
| `k > n` | 32 | 3 | 3 | ×0.998 |
| `n <= 64` and `k > n` | 6 | 3 | 1 | **×1.024** |

That last row is a *first* fit, and it is wrong in a way `§14` records: its population was
selected from the CSV's `m`/`n`/`k`, which are the unoriented contraction, while the
eligibility sees the *oriented role* dimensions.

Two and a half percent in the best-selected class, with a case that still loses, is
not a promotion: the switch stays off, this stays out of a PR, and the objective it
was written for - update both repositories with a TBLIS-equivalent algorithm that
improves the published numbers - is not met. What the work does establish is the
reason: an outer-blocked traversal with a block-level output ordering buys contiguous
operand reads with strided output stores, so it cannot be a general win, and the
remaining idea that could change that (a write-back path that absorbs the scatter
instead of paying it) is a design-level change, not a tuning step.

## 13. The instrument now adds up, and the cost has two names

The earlier attribution was wrong because the instrument was: `phase::scope(3)` sits
inside the *unblocked* write-back only, so the blocked path's emit loop was in no phase
at all, and per-block `Instant::now()` calls - the obvious fix - are expensive enough
on this host to swamp what they measure. The instrument now has `block_meta` and
`permute` phases of its own, taken once per call, and the parts add up to the call
(38.6 measured against a total of 38.1 ms):

| part | blocked (on) | today (off) |
| --- | --- | --- |
| `pack_a` | **6.3** | 28.8 |
| `kernel` (own) | 0.8 | 11.7 |
| `writeback` (stores) | 11.0 | 10.6 |
| **`block_meta`** | **15.0** | - |
| **`permute`** | **5.5** | - |
| total | 38.1 | 52.5 |

`pack_a` falls by 22.5 ms and the kernel by 10.9, against 20.5 ms of added block work:
15.0 for the per-block gathers, scatters and row order, and 5.5 for the permutation.
The machine was ~35% slower when this was taken than when the earlier tables were
(only same-minute ratios are valid), and both cases still win: 38.1 against 52.5.

So the remaining work is not a mystery any more, and it is bounded:

* **`block_meta`, 15 ms.** 240 blocks x ~62 us to gather 384 rows, scatter six times
  and sort the block's rows - which is ~100x what that work costs when the tables are
  hot. The gathers read the *global* scatter tables (736 KB each) at the block's own
  row indices, one cache miss per row; the block's offsets are also computable
  arithmetically from the axis strides the role already carries.
* **`permute`, 5.5 ms.** A 30 MB gather at about 5.5 GB/s, slow for the volume because it
  is inherent to the design (a panel tile spans several operand line-runs, so it is
  never a whole destination tile - which is why the direct-emit variant never
  triggers), so removing it means letting the emit write a per-row offset list instead
  of one stride per tile: a design-level change, not a tuning step.

## 14. Corrections after the independent review

An independent review (`gpt-6.1-sol`, the finished diff and the sources) found two
blockers, and they were right. Both are fixed, and with them the path narrowed:

* **A Direct family wrote `D` instead of the tile.** `direct_tile` did not test whether the
  call was collecting, so an eligible blocked call could compute straight into `D`, leave
  its grid slot unwritten, and then have `run_block` emit that unwritten slot. The guard now
  tests `collect`, which is the invariant, and the eligibility additionally refuses a
  Direct family - as this design always claimed it did.
* **The permutation read grid slots nobody had written.** Each column-group worker fills
  only its own column tiles, while `permute_grid_rows` walks every column of the grid. The
  eligibility now requires `pm == 1 && pn == 1`: the blocked path is serial, which is where
  it was measured, and partitioning the block itself stays a separate change. A Direct
  family that fell back to scratch also wrote only its live lanes while the permutation read
  a whole `nr`; refusing Direct families removes that too.
* **The row budget was not enforced.** `(mc_budget / span).max(1)` admits a `span` larger
  than the budget, and the panel, the tile grid and the block's row map are sized from the
  block *rounded up to whole `MR` tiles*, which can be `MR - 1` rows larger than the block.
  Both are now conditions in the eligibility rather than panics in `run_block`, with tests.
* **The traversal allocated.** `role`, `block_origins` (a `Vec<Vec<usize>>`, one allocation
  per block) and `rows` each allocated per execute, which the workspace's no-allocation
  contract forbids. They are now fixed-size scratch (`role_into`, `Origins`, `rows_into`),
  and the incremental origins are checked against the materialized list. **Not verified:**
  `tests/packed_workspace_alloc` exercises an ordinary matrix multiplication, whose single
  folded row axis can never select this path, so the blocked path's zero-allocation claim
  rests on reading the code, not on that test.
* **The phase account was not disjoint.** `block_meta` contained `pack_a` and `kernel`, the
  scopes were taken per block rather than per call, and clock reads per block on this host
  cost more than the intervals they measure. Those two phases were removed again; the
  instrument is back to four phases plus `setup`, which is a remainder and not an exclusive
  measurement. The blocked path's own work is in `setup` and in `writeback`, and the numbers
  in §13 are inclusive, not a partition. The derived "kernel (own)" and "3.1 ms of metadata"
  figures are withdrawn.
* **Provenance.** The block-scatter helpers adapt Matthews' scheme, which
  `docs/provenance.md` says a citation covers for an independently written implementation;
  no TBLIS code was translated. Upstream correspondence was not verified by the reviewer and
  is not claimed here.

What survives the review, measured again after the fixes with the eligibility narrowed, is
the improvement itself: over the same 98 tcbench rows at 1T, `abjc-cbka-kj` f64 goes
42.0 -> 24.7 ms (x1.70), `adbjc-cbdka-kj` f64 46.2 -> 28.4 ms (x1.63), and the c64 rows of
both 1.21x and 1.15x, with no eligible row slower. The four rows that read 3-7% worse are
all ineligible, take the same code path in both versions, and are therefore drift.
