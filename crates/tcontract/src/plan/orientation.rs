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
mod tests;
