//! Orientation and partition rules of the packed plan.
//!
//! Which operand plays the GEMM row role, which register-tile shape suits the
//! output's stride pattern, and how many row strips by column groups the
//! threads are cut into. All three are decisions about a [`PackedPlan`] that
//! never change the result, only where the time goes.

use super::analysis::{Axis, PackedPlan};
use tcontract_kernel::scatter::unbroken_fraction;

/// How the contraction is oriented into a matrix product: which operand plays
/// the GEMM row role. Orientation never changes the result, only which
/// register tile shape suits the output strides.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Orient {
    /// The planner's orientation rule (`transposes_gemm`). The default.
    #[default]
    Rule,
    /// A pinned arm, for forced-arm measurement of both.
    Force(bool),
    /// The Phase 4.1 rule, for measuring the current one against it.
    Legacy,
}

impl Orient {
    /// Parse `rule | none | ab | swap | ba | legacy | phase41`
    /// (case-insensitively); `None` for any other spelling.
    pub fn parse(s: &str) -> Option<Orient> {
        match s.trim().to_ascii_lowercase().as_str() {
            "rule" | "auto" => Some(Orient::Rule),
            "none" | "ab" => Some(Orient::Force(false)),
            "swap" | "ba" => Some(Orient::Force(true)),
            "legacy" | "phase41" => Some(Orient::Legacy),
            _ => None,
        }
    }
}

/// Which micro-tile row block a plan asks for.
///
/// The default is `Auto`; `Base` pins the Phase 3 shape, which is what the rule
/// was measured against. Both arms stay reachable so the comparison can be
/// repeated in one session rather than as a diff between two builds (A15), and
/// `Index` re-runs the whole grid the rule was derived from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RowBlock {
    /// The kernel set's compiled default shape — the Phase 3 choice.
    Base,
    /// The planner's row-block rule (`row_block`). The default. Auto starts from
    /// the host tuning class's ordered menu, so the class's measured shape is its
    /// default and its alternates keep the class's order.
    #[default]
    Auto,
    /// A specific logical `MR`, where the kernel set has one.
    Pin(usize),
    /// A **compiled** menu position: the ISA module's order, default first. It is
    /// comparable across element types and methods in a way a bare `MR` is not —
    /// `mr=16` names different shapes in `f32` and `f64`, `idx=1` names "the first
    /// alternate" in both. The tuning class's ordering is deliberately not
    /// visible here (issue #156): `idx=0` is the ISA module's compiled default even
    /// on a host whose class prefers another shape.
    Index(usize),
}

impl RowBlock {
    /// Parse `base | default | auto | mr=<n> | idx=<i>` (case-insensitively);
    /// `None` for any other spelling.
    pub fn parse(s: &str) -> Option<RowBlock> {
        let v = s.trim().to_ascii_lowercase();
        match v.as_str() {
            "auto" => Some(RowBlock::Auto),
            "base" | "default" => Some(RowBlock::Base),
            _ => {
                let num = |p: &str| v.strip_prefix(p)?.parse::<usize>().ok();
                num("mr=")
                    .map(RowBlock::Pin)
                    .or_else(|| num("idx=").map(RowBlock::Index))
            }
        }
    }
}

/// The depth below which a contraction is bandwidth-bound enough for the
/// cross-domain `B` replication to be what limits it. The project's existing
/// memory-bound criterion is `min(n, k) <= 64` (see `CLAUDE.md`) and this is the
/// same 64; the corpus puts `k = 24` on one side of it and `k >= 204` on the
/// other, so nothing in it is near the boundary and the threshold is a
/// separation, not a tuned constant.
///
/// Named at length because `preferred_row_block` has its own `SHALLOW_K = 32`
/// meaning something else — there the question is whether the write-back is
/// amortised, here it is whether the case is bandwidth-bound. Two thresholds
/// that both mean "shallow" and are not the same number; keep them
/// distinguishable at the point of use.
const BANDWIDTH_BOUND_K: usize = 64;

/// The domain-aware gate: in the regime where the row axis alone fills the
/// threads (`panels >= p`, which the caller has already established), should the
/// column axis take them instead?
///
/// The original multi-domain arm has three measured conditions:
///
/// 1. **`domains > 1`** — the thread set spans more than one L3. This is the
///    mechanism, and the only quantity A36 separates from thread count: Ice Lake
///    runs 32 threads over one domain and reads 1.014, Zen2 runs 4 over one and
///    reads 1.031, while three multi-domain points rise monotonically to 2.38x.
///    Passing `1` is how [`Plan::partition`] expresses "no multi-domain
///    argument" — which is the legacy decision only while the local exception
///    below is false (unknown geometry, or a thread set that does not saturate
///    the detected L3).
/// 2. **`blocks >= p`** — the column axis can fill the threads by itself, so the
///    swap costs no parallelism. Without it the corpus's narrow cases lose 2–5x
///    by running on a fraction of their cores.
/// 3. **`k <= BANDWIDTH_BOUND_K`** — the penalty being dodged is bandwidth, so it can
///    only dominate where the case is bandwidth-bound, and `k` is this corpus's
///    knob for that. The whole effect was measured on the `k = 24` family; the
///    wide compute-bound families (`ijkl-*`, `ij-ik-kj`, `k` 2704–5184) were
///    measured *losing* 25% in the complex methods from the same swap. The guard
///    confines the rule to the population the evidence covers.
///
/// The local-column exception also requires a saturated known L3 and a long
/// contiguous output run; see `local_columns` and the scaling worklog.
/// Pure inputs let both gates be tested without depending on the host CPU.
fn columns_beat_rows(blocks: usize, k: usize, p: usize, domains: usize, local: bool) -> bool {
    (domains > 1 || local) && blocks >= p && k <= BANDWIDTH_BOUND_K
}

fn local_columns(
    rows: usize,
    cols: usize,
    mr: usize,
    p: usize,
    run: (usize, i64),
    cores: Option<usize>,
) -> bool {
    // ponytail: measured layout heuristic, not a full cost model; calibrate on
    // other CPUs before claiming architecture-independent optimality.
    //
    // INVARIANT: `cores` is `Some` only when the cache probe succeeded from a
    // real source, so an unknown or built-in geometry keeps the legacy rule
    // rather than guessing; `mr.max(1)` keeps the panel division total.
    cores == Some(p) && p > 1 && cols >= rows && run.1 == 1 && run.0 / mr.max(1) > p
}

impl PackedPlan {
    /// [`Plan::partition`] at a caller-supplied thread count.
    ///
    /// Exists because the driver may run at a width other than the requested
    /// one: `execute` takes a cap, and the batched entry point divides the
    /// available threads across items. `partition` is this at
    /// [`Plan::threads`].
    pub fn partition_with(&self, mr: usize, nr: usize, threads: usize) -> (usize, usize) {
        /// Cost of packing one `A` element relative to one micro-kernel lane-FMA.
        /// See [`Plan::partition`]; deliberately at the conservative end.
        const PACK_WEIGHT: usize = 8;

        let (rows, cols, run) = if self.transposes_gemm(mr) {
            (self.b_n.len(), self.a_m.len(), self.d_n_run)
        } else {
            (self.a_m.len(), self.b_n.len(), self.d_m_run)
        };
        let panels = rows.div_ceil(mr.max(1)).max(1);
        let blocks = cols.div_ceil(nr.max(1)).max(1);
        let p = threads.max(1);

        if panels >= p {
            let domains = tcontract_kernel::blocking::l3_domains(p, self.l3_domains);
            let local = if domains == 1 {
                let h = tcontract_kernel::blocking::hierarchy();
                let cores = (h.source != tcontract_kernel::blocking::CacheSource::Builtin)
                    .then(|| h.l3.map(|l3| h.cores_sharing(&l3)))
                    .flatten();
                local_columns(rows, cols, mr, p, run, cores)
            } else {
                false
            };
            if columns_beat_rows(blocks, self.a_k.len(), p, domains, local) {
                return (1, p.min(blocks));
            }
            return (p, 1);
        }
        let mut best = (1, 1);
        let mut best_cost = usize::MAX;
        for pm in 1..=panels {
            let pn = (p / pm).min(blocks);
            let cost = panels.div_ceil(pm) * (nr.max(1) * blocks.div_ceil(pn) + PACK_WEIGHT);
            // `<=`, so among equal-cost partitions the largest `pm` wins: the
            // `M` split needs no duplicated packing and is the measured one.
            if cost <= best_cost {
                best = (pm, pn);
                best_cost = cost;
            }
        }
        best
    }

    /// Whether execution will compute `D^T = B^T A^T` rather than `D = A B`.
    ///
    /// The engine is symmetric under exchanging `(A, M)` with `(B, N)`: doing
    /// so transposes the matrix view of `C` and `D` and changes nothing else.
    /// It is worth doing when it makes the `MR` rows of a micro-tile a single
    /// contiguous run of `D`, because the write-back runs under the innermost
    /// loop and is the one access packing cannot hide. On the nine corpus cases
    /// Phase 3 profiled as a 2x defect this is worth up to 2.7x.
    ///
    /// The rule is a preference with a fallback, not a test with a veto:
    ///
    /// 1. **Prefer the arm whose micro-tile row block lands inside a single run
    ///    of `D`** — unit stride and a run of at least `MR`. If exactly one arm
    ///    manages that, take it. If both do, stay put; there is nothing to buy.
    /// 2. **Otherwise put the direction with the *shorter* run in the row
    ///    role**, whichever operand that is.
    ///
    /// Step 2 is what the Phase 4.1 rule was missing: it treated "the row block
    /// would be shattered either way" as a reason never to swap, and all nine of
    /// its known misses lived in that case (A14). The rule has to be
    /// antisymmetric under exchanging the two directions, because the `abcijk`
    /// families are exact mirror images of each other.
    ///
    /// Step 1 dominates step 2, and must: it is why `c64` (`MR = 16` against a
    /// run of 24) takes the opposite arm from `f32` (`MR = 48`) on the same
    /// shapes. That element-type dependence is why the choice is not a property
    /// of the plan alone — hence the `mr` argument.
    ///
    /// This is a **tier-2** answer: stable signature, tuning-output value.
    /// [`Plan::with_orientation`] can disable the swap or force it; neither
    /// affects correctness. The mirror-family table the rule was derived
    /// from, the scoring against both forced arms of all 392 corpus
    /// case-dtype-methods, and the 21 cases still on the slower arm are in
    /// `docs/archive/tensorprimitives/notebook/` — the write-back chapter, Phase 4.1d.
    pub fn transposes_gemm(&self, mr: usize) -> bool {
        match self.orient {
            Orient::Rule => self.transposes_gemm_rule(mr),
            Orient::Force(v) => v,
            Orient::Legacy => self.transposes_gemm_legacy(mr),
        }
    }

    /// [`Plan::transposes_gemm`]'s rule with no environment override.
    ///
    /// Separate so the unit tests can assert what the *rule* decides even for a
    /// plan whose orientation is pinned.
    fn transposes_gemm_rule(&self, mr: usize) -> bool {
        // A row block lands inside one run when the rows are unit-stride and
        // the run is at least `MR` long.
        let fits = |(run, stride): (usize, i64)| stride == 1 && run >= mr;
        let (ab, ba) = (fits(self.d_m_run), fits(self.d_n_run));
        if ab != ba {
            return ba;
        }
        if ab {
            return false;
        }
        // A tie-break between two imperfect arms is not a tie when one of them
        // cannot fill a micro-tile: most of every `MR x NR` block would be
        // padding, which is arithmetic rather than a cache effect. The corpus
        // cannot test this — its shortest direction is 32 against a largest
        // `MR` of 48, and neither of the two cases where that bites reaches the
        // fallback — so this guard is inert on every measurement quoted here.
        let (m_rows, n_rows) = (self.d_m.len(), self.d_n.len());
        if (m_rows >= mr) != (n_rows >= mr) {
            return n_rows >= mr;
        }
        self.d_n_run.0 < self.d_m_run.0
    }

    /// The Phase 4.1 orientation rule, kept reachable as
    /// `Orient::Legacy` so that the current one can be measured
    /// against it as an A/B rather than a diff between two builds (A15).
    ///
    /// Swap only when `D`'s column direction is strictly more contiguous than
    /// its row direction *and* the swap leaves the row block unbroken. The
    /// second condition is a veto with no fallback, which is where all nine of
    /// its known misses live; see [`Plan::transposes_gemm`].
    fn transposes_gemm_legacy(&self, mr: usize) -> bool {
        let lead = |axes: &[Axis]| axes.first().map_or(u64::MAX, |a| a.sd.unsigned_abs());
        if lead(&self.stats.n_axes) >= lead(&self.stats.m_axes) {
            return false;
        }
        match self.stats.n_axes.first() {
            Some(ax) => ax.sd == 1 && ax.extent >= mr as i64,
            None => false,
        }
    }

    /// Choose the micro-tile row block `MR` from a menu of shapes the kernel
    /// set actually has, ordered fastest-in-isolation first.
    ///
    /// Returns `None` for "use the menu's default", which is also what
    /// `RowBlock::Base` gives.
    ///
    /// # Why `MR` is not just a kernel-tuning constant
    ///
    /// `MR` is the granularity at which the *output's* row scatter is blocked,
    /// so it decides which of the write-back's three paths each block
    /// takes. When `D`'s rows come in contiguous runs of `r` elements, an
    /// aligned `MR`-block lies inside one run — and so gets the unit-stride
    /// path — only when `MR` divides into the run pattern; otherwise it
    /// straddles a discontinuity, the block scatter reports [`IRREGULAR`], and
    /// the whole block falls back to a gather. That is a code-path change, not
    /// a tuning delta, and the shapes it wants are not the shapes peak
    /// throughput wants. The TCCG corpus rounds every stride-1 index up to a
    /// multiple of 24, while `MR` at `L = 16` lanes is 16, 32 or 48.
    ///
    /// [`IRREGULAR`]: crate::scatter::IRREGULAR
    ///
    /// # The rule, and the two guards it needs
    ///
    /// Take the first shape on the menu that makes *every* output row block a
    /// single run, but only when both of these hold. (A third guard existed and
    /// was removed in Phase 4.1d; see below.) Each guard is there
    /// because the grid in `tensorprimitives/bench-results/phase4c` measured what happens
    /// without it; none is a plausibility argument.
    ///
    /// 1. **The contraction is shallow** (`k <= 32`). The write-back costs a
    ///    constant per output element against `~4k` flops of kernel work, so
    ///    the path it takes only matters while `k` is small — and a shape off
    ///    the kernel's peak always costs something.
    /// 2. **The default is substantially broken** (`wb <= 0.75`). A shape change
    ///    is not free, so a marginal improvement cannot repay it.
    ///
    /// A third guard, against a shape change flipping the orientation as a side
    /// effect, was removed in Phase 4.1d once the orientation rule was fixed.
    /// The two rules are coupled and cannot be tuned separately.
    ///
    /// Unguarded, the rule is a **loss** — maximising the regular fraction alone
    /// scores 0.936 in `f32`. This is a **tier-2** answer: stable signature,
    /// tuning-output value. Both thresholds, what each guard is worth, the
    /// corpus firing count and the gain an oracle leaves on the table are in
    /// `docs/archive/tensorprimitives/notebook/` (the write-back chapter, Phase 4.1c and 4.1d); the grid
    /// they were scored against is `tensorprimitives/bench-results/phase4c`.
    ///
    /// # What `menu` is, and what comes back
    ///
    /// `menu` is the kernel set's `(MR, NR)` shapes, default first, and the
    /// answer is a **position in it** — not an `MR`. The distinction is not
    /// cosmetic: two entries may share an `MR` and differ only in `NR`, which is
    /// a shape the measurement asked for and an `MR`-keyed menu could not hold
    /// (A35). The rule below reads only the `MR`, so entries of equal height
    /// score alike and the earlier one wins, which keeps the measured default in
    /// front.
    pub fn row_block(&self, menu: &[(usize, usize)]) -> Option<usize> {
        match self.row_block_mode {
            RowBlock::Base => None,
            RowBlock::Auto => self.preferred_row_block(menu),
            RowBlock::Pin(mr) => menu.iter().position(|&(m, _)| m == mr),
            RowBlock::Index(i) => (i < menu.len()).then_some(i),
        }
    }

    /// [`Plan::row_block`]'s rule with no environment override, so that it can
    /// be scored offline against measured ground truth. Returns a menu position.
    pub fn preferred_row_block(&self, menu: &[(usize, usize)]) -> Option<usize> {
        /// Above this contraction depth the write-back is amortised and the
        /// shape's own cost is all that is left. See [`Plan::row_block`], and
        /// note this is a different threshold from [`BANDWIDTH_BOUND_K`], which
        /// asks a different question.
        const SHALLOW_K: usize = 32;
        /// A default this regular already is not worth paying a shape change
        /// to improve.
        const BROKEN_ENOUGH: f64 = 0.75;

        let (&(default, _), rest) = menu.split_first()?;
        if self.stats.k > SHALLOW_K || self.row_block_score(default) > BROKEN_ENOUGH {
            return None;
        }
        rest.iter()
            .position(|&(mr, _)| self.row_block_score(mr) >= 1.0 - 1e-9)
            .map(|i| i + 1)
    }

    /// Fraction of the output's row blocks that would stay off
    /// the write-back's gather path at row block `mr`, evaluated in the
    /// orientation `mr` itself selects.
    ///
    /// `0.0` when the output's rows have no uniform run structure — then `MR`
    /// has no predictable effect and every shape scores alike, which leaves the
    /// tie-break to keep the default.
    pub fn row_block_score(&self, mr: usize) -> f64 {
        let (rows, run) = if self.transposes_gemm(mr) {
            (&self.d_n, self.d_n_run)
        } else {
            (&self.d_m, self.d_m_run)
        };
        // A zero stride marks "no uniform run structure", where `MR` has no
        // predictable effect.
        if run.1 == 0 {
            return 0.0;
        }
        unbroken_fraction(rows.len(), run.0, mr)
    }
}

#[cfg(test)]
#[path = "tests/local_columns.rs"]
mod local_column_tests;

#[cfg(test)]
mod tests {
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
}
