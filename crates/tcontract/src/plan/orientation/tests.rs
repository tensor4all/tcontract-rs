use super::super::test_support::{build, lay, laying};
use super::*;

/// The domain-aware gate's whole truth table, machine-independently.
///
/// Every one of the three conditions is here because measurement put it
/// there, so each gets a case that turns it off on its own — a gate that
/// quietly stopped consulting one of them would still look right on the
/// corpus's `abcijk` family, which satisfies all three.
#[test]
fn domain_gate_needs_all_three_conditions() {
    // The measured population: 64 threads over 16 L3 domains, 1024 column
    // blocks against 768 row panels, `k = 24`. Worth up to 4.3x (A36).
    assert!(columns_beat_rows(1024, 24, 64, 16, false));
    // One L3 domain: the early return is *correct* here, at any thread count.
    // Ice Lake reaches 32 threads on one domain and reads 1.014 — the point
    // that separates domain count from thread count — so it is pinned at both
    // ends of the thread range.
    assert!(!columns_beat_rows(1024, 24, 32, 1, false));
    assert!(!columns_beat_rows(1024, 24, 4, 1, false));
    // Too few column blocks to feed the threads: swapping would run the case
    // on a fraction of its cores, which the corpus's narrow half pays 2-5x for.
    assert!(!columns_beat_rows(63, 24, 64, 16, false));
    assert!(columns_beat_rows(64, 24, 64, 16, false));
    // Deep enough to be compute-bound: the cross-domain penalty is a
    // bandwidth cost and cannot dominate here. `ijkl-*` and `ij-ik-kj` sit on
    // this side and were measured *losing* 25% in the complex methods.
    assert!(!columns_beat_rows(1024, 2704, 64, 16, false));
    assert!(columns_beat_rows(1024, BANDWIDTH_BOUND_K, 64, 16, false));
    assert!(!columns_beat_rows(
        1024,
        BANDWIDTH_BOUND_K + 1,
        64,
        16,
        false
    ));
    // Serial is never a partition question.
    assert!(!columns_beat_rows(1024, 24, 1, 1, false));
}

/// `D[i,j] = A[i,k] B[k,j]` with `D` stored in the given strides.
fn gemm_plan(m: i64, n: i64, d_strides: [i64; 2]) -> PackedPlan {
    let a = lay(&[m, 7]);
    let b = lay(&[7, n]);
    let d = laying(&[m, n], &d_strides);
    build(&a, &[0, 2], &b, &[2, 1], &d, &[0, 1])
}

#[test]
fn column_major_output_is_not_transposed() {
    // Rows already have the unit stride: nothing to gain.
    assert!(!gemm_plan(64, 64, [1, 64]).transposes_gemm_rule(16));
}

#[test]
fn row_major_output_is_transposed() {
    // Columns have the unit stride and the run is long enough to cover MR.
    assert!(gemm_plan(64, 64, [64, 1]).transposes_gemm_rule(16));
}

#[test]
fn transpose_taken_when_the_row_block_fits_exactly() {
    let p = gemm_plan(64, 16, [16, 1]);
    assert!(p.transposes_gemm_rule(16), "run == MR fits");
}

/// `D` with `M` rows in runs of `m_run` at stride 24 and `N` columns in one
/// contiguous run of 24 — the shape of the whole `abcijk` family, where the
/// two arms are mirror images and neither can fit a 48-row block.
fn mirrored_plan(m_run: i64, outer: i64) -> PackedPlan {
    // `outer` and `outer2` break the folds, so both directions keep short
    // runs while staying long enough overall to fill a micro-tile.
    let d = laying(&[24, m_run, 4, 8], &[1, 24, outer, outer * 7 + 3]);
    let a = lay(&[m_run, 4, 7]);
    let b = lay(&[7, 24, 8]);
    build(&a, &[1, 2, 4], &b, &[4, 0, 3], &d, &[0, 1, 2, 3])
}

#[test]
fn orientation_falls_back_to_the_shorter_run_when_neither_arm_fits() {
    // At MR = 48 no arm can hold a row block inside a run: the columns run
    // 24 and the rows run 16 or 256. The tie is then broken by putting the
    // *shorter*-run direction in the row role — which is what separates the
    // `-mb` family (where swapping measured -14% in `f32`) from `e*bc`
    // (where it measured +40%), the distinction the previous rule missed.
    let short = mirrored_plan(16, 100_000);
    assert_eq!(short.d_m_run.0, 16);
    assert_eq!(short.d_n_run, (24, 1));
    assert!(
        !short.transposes_gemm_rule(48),
        "rows already have the shorter run"
    );

    let long = mirrored_plan(256, 1_000_000);
    assert_eq!(long.d_m_run.0, 256);
    assert!(
        long.transposes_gemm_rule(48),
        "columns have the shorter run"
    );

    // And step 1 still dominates: give it an `MR` the column run can hold
    // and both plans swap for that reason instead.
    assert!(short.transposes_gemm_rule(16));
    assert!(long.transposes_gemm_rule(16));
}

#[test]
fn transpose_declined_when_columns_are_not_contiguous() {
    // Both directions strided: the swap cannot make the rows contiguous,
    // so the smaller stride alone does not justify it.
    assert!(!gemm_plan(64, 64, [512, 2]).transposes_gemm_rule(16));
}

/// `D[a,c,j] = A[a,c,k] B[k,j]` with `a` contiguous in `D` (extent 24) and
/// `c` far away, so `D`'s rows come in runs of 24 — the shape the whole
/// TCCG corpus has, since it rounds stride-1 extents up to multiples of 24.
fn run24_plan() -> PackedPlan {
    let d = laying(&[24, 4, 8], &[1, 200, 4000]);
    let a = lay(&[24, 4, 7]);
    let b = lay(&[7, 8]);
    build(&a, &[0, 1, 3], &b, &[3, 2], &d, &[0, 1, 2])
}

/// The row-block rule over a menu written as bare `MR`s, answering with the
/// `MR` it chose rather than the menu position.
///
/// The menu is positional since A35, so the rule returns an index — but
/// every expectation below is about *which shape* is picked, and an index
/// would make them say that less clearly while also going stale whenever an
/// entry is inserted. `NR` is a placeholder here because the rule does not
/// read it; `row_block_may_reach_an_nr_only_alternate` is the test that does.
fn pick(p: &PackedPlan, mrs: &[usize]) -> Option<usize> {
    let menu: Vec<(usize, usize)> = mrs.iter().map(|&mr| (mr, 6)).collect();
    p.preferred_row_block(&menu).map(|i| menu[i].0)
}

#[test]
fn row_block_scores_follow_the_output_runs() {
    let p = run24_plan();
    assert_eq!(p.stats.m, 96, "two unfolded M axes");
    assert_eq!(p.row_block_score(24), 1.0);
    assert_eq!(p.row_block_score(8), 1.0);
    assert!((p.row_block_score(16) - 2.0 / 3.0).abs() < 1e-12);
    assert_eq!(p.row_block_score(48), 0.0);
}

#[test]
fn row_block_picks_a_shape_that_tiles_the_run() {
    let p = run24_plan();
    // `c64` planar's menu: the default straddles a third of its blocks,
    // the first alternate none, so the rule moves. This is the case worth
    // 1.09-1.26x on the corpus.
    assert_eq!(pick(&p, &[16, 24, 8]), Some(24));
    // The default is already perfect: never trade kernel peak for nothing.
    assert_eq!(pick(&p, &[24, 16, 8]), None);
    // Ties keep the default, which is the fastest kernel.
    assert_eq!(pick(&p, &[8, 24]), None);
    assert_eq!(pick(&p, &[]), None);
}

/// The point of keying the menu by position (A35): two entries of the same
/// height differing only in `NR`.
///
/// `planar` `f32`/`c32` ships `32x6` where Phase 3's own sweep names `32x5`
/// as 7.8% faster, and under the old `MR`-keyed menu that shape could not be
/// put on the menu at all — the second entry would have been unreachable, so
/// there was no way to A/B it at run time and the finding stayed
/// untestable. This pins the three things that has to mean.
#[test]
fn row_block_may_reach_an_nr_only_alternate() {
    let p = run24_plan();
    let menu = [(24, 6), (24, 5)];
    // 1. The rule cannot tell them apart — it reads `MR` — and ties go to
    //    the earlier entry, so the measured default stays in front.
    assert_eq!(p.preferred_row_block(&menu), None);
    // 2. `idx=` reaches the second one, which is what makes it measurable.
    assert_eq!(menu.get(1), Some(&(24, 5)));
    // 3. `mr=` cannot distinguish them and resolves to the first, which is
    //    the documented limitation rather than a silent surprise.
    assert_eq!(menu.iter().position(|&(m, _)| m == 24), Some(0));
}

#[test]
fn row_block_refuses_a_partial_improvement() {
    // 48 straddles everything and 16 fixes two blocks in three, but a
    // shape that does not clear the gather path outright cannot repay its
    // own cost: this is the `f32` menu, and taking it measured 0.88-0.95.
    let p = run24_plan();
    assert_eq!(pick(&p, &[48, 16]), None);
}

#[test]
fn row_block_may_change_the_orientation() {
    // Until Phase 4.1d a shape change was forbidden from flipping the
    // orientation, because the orientation rule of the day picked the wrong
    // arm on one family and the shape change would hand it the decision.
    // With that rule fixed the veto only blocked wins: the six `c32` 1m
    // cases it had been suppressing measured 1.19-1.24x once it was gone.
    // So a shape is judged on its own regularity, in whatever orientation
    // it implies.
    let p = run24_plan();
    assert_eq!(pick(&p, &[16, 24, 8]), Some(24));
}

#[test]
fn row_block_leaves_a_deep_contraction_alone() {
    // Same output structure, but `k` large enough that the write-back is
    // amortised: the shape change would cost and buy nothing.
    let d = laying(&[24, 4, 8], &[1, 200, 4000]);
    let a = lay(&[24, 4, 512]);
    let b = lay(&[512, 8]);
    let p = build(&a, &[0, 1, 3], &b, &[3, 2], &d, &[0, 1, 2]);
    assert_eq!(p.stats.k, 512);
    assert!(
        (p.row_block_score(16) - 2.0 / 3.0).abs() < 1e-12,
        "would fire"
    );
    assert_eq!(pick(&p, &[16, 24, 8]), None);
}

#[test]
fn row_block_leaves_a_fully_regular_output_alone() {
    // Column-major `D`: one run, so no shape can straddle anything.
    let p = gemm_plan(64, 64, [1, 64]);
    assert_eq!(pick(&p, &[16, 24, 8]), None);
}
