# qcbench: unique selection — the df_tdhf backfill and the K vocabulary note

> Follow-up to `2026-11-10-qcbench.md`, executed per the benchmark job's
> `docs/real_bench_mod_plan.md` (the parent plan's §4 uniqueness rule, added
> 2026-11-15). Campaign-side consequences — pin bump, suite regeneration,
> re-recordings — are tracked there, not here.

## The audit finding

Two of the 24 selected cases were the same contraction. df_tdhf's
sequence-3 `response.exchange.metric` (`PQ,ijQ->ijP`, P/Q: naux, i/j: o) is
canonically identical to ri_rhf_gradient's selected `gradient.X` — same
letters, domains and shapes, so the pair survives relabeling and
operand-swap canonicalization. Two of the 72 timed families measured one
workload.

## What changed

- **The rule** (the parent plan's §4, quoted in the PR): no two selected
  cases may be the same contraction up to index relabeling and A/B operand
  swap, with matching index-domain roles. On collision, drop the
  later-`sequence` case and backfill with that corpus's next `sequence`
  row, re-checking uniqueness.
- **The swap:** df_tdhf's pick becomes sequence-4
  `response.exchange.b_trial_pair` (`ivP,vj->ijP`, i/j: o, v: ao, P: naux,
  `ao*o^2*naux` — rung 3 of that corpus, legal only through the backfill
  clause). The case count is unchanged: 24 cases, 72 families, 288 declared
  cells.
- **The executable rule:** `corpus/tests.rs` gains
  `selected_contractions_are_pairwise_unique`, which canonicalizes every
  case per ratio (letters relabeled to `<domain><first-appearance>`,
  counted per domain so `o`/`oroot` cannot collide; operands sorted;
  summed-domain set) and asserts 24 distinct keys. Because that labeling
  depends on the operand scan order, the key is the lexicographic minimum
  over both operand orders, so a duplicate written with A and B swapped
  gets the same key as its twin (adversarial-review finding against the
  single-order version: swapped forms passed). With the swap the pass is
  exact: the only corpus-level twin of `b_trial_pair` is
  ri_rhf_gradient's sequence-5 `gradient.pair_factor` (rung 3, unselected),
  so a future widening of that corpus's selection re-triggers the rule.
- **Vocabulary notes only:** `K.aux`/`K.matrix` keep the Neese–Olbrich `K`
  ids — frozen per the corpus format's cross-corpus register
  (`cases_csv_format.md` §2.1: `K` ≡ `exchange`, likewise `J` ≡ `coulomb`
  for `J.fit`/`J.matrix`, recorded in that corpus's `CORPUS.md`). No id
  changes anywhere.

## What did not change

`EXTENTS`, the engine rows, dtypes, every shared module, the `[[bin]]`
block, and the no-skip/8 GiB policy: the binding footprint case is still
`T.connected.virtual` (6.87/7.53/7.16 GiB A+B+C) and the new case is far
smaller (largest operand ~0.003 GiB). Census delta for the swapped row:
0.024→0.006 / 0.008→0.003 / 0.005→0.002 G MACs at the three ratios; the
per-pass sums move by < 0.02 G per ratio and the `gamma.abcd` ov_20 ceiling
cell is untouched.

## Verification

`cargo test --release -p tcontract-bench --bin qcbench` (now 9 corpus tests
+ 2 knobs tests) plus the full AGENTS gate. Cross-checkout smoke in
tcontract-benchmark at this branch: the corpus tests,
`build_for_tprims_rev.sh` (tblis) and `gen_suite_from_census.py` — the
emitted census carries `response.exchange.b_trial_pair` and no
`response.exchange.metric`.
