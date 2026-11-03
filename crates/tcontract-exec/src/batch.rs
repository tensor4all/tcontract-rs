//! Batch lane policy: the shared vocabulary and the one decision function.
//!
//! Three libraries decide how a batch of independent problems is split over a
//! thread budget (tcontract's packed driver and faer strategy, cpueinsum's
//! grouped GEMM, tlinalg's lane plan). This module gives them one vocabulary
//! and one pure function, [`batch_decision`], so a caller can force a route and
//! two crates cannot drift on the same decision. The types are plain data with
//! no scheduler dependency; they live apart from [`Exec`] so moving
//! them later is a re-export, not a redesign.
//!
//! # Strategies
//!
//! [`BatchStrategy::Auto`] is the only heuristic. Every other variant forces a
//! route and never falls back: the route either honors it (clamped to the
//! items and the budget) or the library reports
//! [`crate::ExecError::StrategyUnavailable`].
//!
//! Forced strategies are evaluated against the **original host budget**. Only
//! `Auto` narrows, because the packed driver narrows its `Exec` twice (see
//! below).
//!
//! # What `Auto` computes
//!
//! `Auto` has two forms, chosen by the query. This is how one formula serves
//! both the shared work model and the packed driver's legacy rule:
//!
//! * Without `balanced_lanes`, the issue's generic formula: estimate
//!   `item_ns` and `total_ns`, cap the lanes at `total_ns /
//!   min_work_per_lane_ns`, and fan out only when `lanes >= 2`, `items >=
//!   min_items` and `items >= lanes * min_items_per_lane`. Each lane runs
//!   items serially; with one lane, `item_threads` is the width of one item, so
//!   a route that permits it (faer) can parallelize inside the item.
//! * With `balanced_lanes`, the packed two-stage rule below, which the default
//!   [`WorkModel`] reproduces to the lane.
//!
//! # Widths come from the token
//!
//! `Auto` never owns a width model. Every width it needs is the token's:
//! [`Exec::width_for`](crate::Exec::width_for) turns the estimated serial time of
//! the region and the number of sections (dispatches) it performs into the width at
//! which the region pays, using the dispatch cost of the token's kind (team, plain
//! token inside its pool, plain token outside it; `threadteam` owns the table and
//! the `PAYS` multiple). A batch has two such regions: the whole batch spread over
//! lanes (one section) and one item on a team ([`BatchQuery::item_sections`]
//! sections). The rules below combine them:
//!
//! | Rule | Source | Encoding here |
//! |---|---|---|
//! | packed balanced lanes | `crates/tcontract/src/driver/batch.rs:38`, `plan/mod.rs:641`, `driver/route.rs:49` | `Auto` + `balanced_lanes` + the default [`WorkModel`] (its item cost is `ns_per_flop()`) |
//! | faer item-width-1 | `crates/tcontract/src/strategy/faer.rs:243` | `Auto` + `item_parallel`; a decision with `item_threads > 1` is the inner route, `lanes > 1` with item threads one is the outer route |
//! | `contract_batched` item count | `crates/tcontract/src/batch.rs:140` | forced strategies, not `Auto`: [`BatchStrategy::Outer`] when the items fill the budget; below that one lane, the caller running each item on the whole budget (`Sequential`/`Inner`). It has no work threshold. |
//!
//! The packed rule is two-stage. `plan/mod.rs` first narrows the `Exec` to the
//! width the whole batch pays for, and `driver/batch.rs` recomputes the width
//! against that narrowed budget and only then applies its *tiny* (one item alone
//! does not pay a team) and *balanced* (every lane non-empty, the split near-even,
//! the batch alone fills the budget) tests. `Auto` reproduces exactly that: it
//! narrows `budget` with the token, then applies the tiny/balanced tests when the
//! query says `balanced_lanes`, and reports the narrowed width as `item_threads`
//! when it keeps one lane.
//!
//! # Examples
//!
//! ```
//! use tcontract_exec::batch::{batch_decision, BatchPolicy, BatchQuery, BatchStrategy};
//! use tcontract_exec::{Exec, Pool};
//!
//! let sp = threadteam::Pool::builder().threads(8).build().unwrap();
//! let pool = Pool::borrow(&sp);
//! let exec = Exec::new(&pool, 8).unwrap();
//! let query = BatchQuery::new(16, 64.0);
//!
//! // A forced route is clamped, never a fallback.
//! let outer = BatchPolicy::new(BatchStrategy::Outer);
//! let d = batch_decision(&outer, &exec, &query);
//! assert_eq!(d.lanes, exec.budget().min(8));
//! assert_eq!(d.item_threads, 1);
//! ```

use core::num::NonZeroUsize;

use crate::Exec;

/// How the items of a batch are split over a thread budget.
///
/// # Examples
///
/// ```
/// use tcontract_exec::batch::BatchStrategy;
/// assert_eq!(BatchStrategy::default(), BatchStrategy::Auto);
/// ```
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BatchStrategy {
    /// Choose from the [`WorkModel`] and the query's route capabilities.
    #[default]
    Auto,
    /// One lane, one item thread: no parallelism at all.
    Sequential,
    /// `lanes = min(budget, items)`, each item serial.
    Outer,
    /// One lane; the item kernel gets the whole budget.
    Inner,
    /// Exactly `lanes` lanes (clamped to the items and the budget), the rest
    /// of the budget split evenly as item threads. A route that cannot run item
    /// threads inside a lane refuses the resulting decision with
    /// [`crate::ExecError::StrategyUnavailable`] before any write; it never silently
    /// runs the items serially (see [`BatchDecision`]'s route contract).
    Split {
        /// Requested lane count.
        lanes: NonZeroUsize,
    },
}

/// The cost and fan-out model [`BatchStrategy::Auto`] applies.
///
/// The item rate is the serial GEMM throughput of the host class (`host::muladds_per_ns`: 27
/// multiply-adds per nanosecond on `aarch64-apple` and the `generic` row, 15.6 on `x86_64-avx512`;
/// see `crates/tcontract-exec/src/host.rs` and `docs/design/thresholds.md` of threadteam-rs);
/// the width a team pays for comes from the token, not from this model.
///
/// The three item-cost fields follow tenferro's restored
/// `CpuBatchThresholds`: one item costs roughly `item_overhead_ns` plus one
/// nanosecond per `muladds_per_ns` multiply-adds, a lane needs
/// `min_work_per_lane_ns` of estimated work, and a fan-out needs `min_items`
/// items, at least `min_items_per_lane` per lane.
///
/// # Examples
///
/// ```
/// use tcontract_exec::batch::WorkModel;
///
/// let model = WorkModel::default().with_min_items(4).with_muladds_per_ns(32.0);
/// assert_eq!(model.min_items, 4);
/// assert_eq!(model.muladds_per_ns, 32.0);
/// ```
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorkModel {
    /// Estimated fixed cost of one item, in nanoseconds.
    ///
    /// The default is zero so that [`Default`] reproduces the packed
    /// `ns_per_flop()` estimate.
    pub item_overhead_ns: f64,
    /// Estimated multiply-adds per nanosecond on one lane.
    ///
    /// `1.0 / ns_per_flop() / 2.0` (27 on `aarch64-apple` and the
    /// `generic` row, 15.6 on `x86_64-avx512`) reproduces
    /// `2 * macs * ns_per_flop()`: the packed driver counts two flops per
    /// multiply-add. A value `<= 0` is treated as one multiply-add per
    /// nanosecond.
    pub muladds_per_ns: f64,
    /// Least estimated work a lane must receive, in nanoseconds; `0` disables
    /// the work gate (the packed default).
    pub min_work_per_lane_ns: f64,
    /// Fewest items for which `Auto` fans out at all.
    pub min_items: usize,
    /// Fewest items each lane must receive.
    pub min_items_per_lane: usize,
}

impl Default for WorkModel {
    fn default() -> Self {
        Self {
            // Reproduce `driver/batch.rs`: `item_ns = 2 * m * n * k * ns_per_flop()`.
            item_overhead_ns: 0.0,
            muladds_per_ns: crate::host::muladds_per_ns(),
            min_work_per_lane_ns: 0.0,
            min_items: 2,
            min_items_per_lane: 1,
        }
    }
}

impl WorkModel {
    /// Return this model with a different per-item overhead.
    #[must_use]
    pub fn with_item_overhead_ns(mut self, nanoseconds: f64) -> Self {
        self.item_overhead_ns = nanoseconds;
        self
    }

    /// Return this model with a different lane throughput.
    ///
    /// # Examples
    ///
    /// ```
    /// use tcontract_exec::batch::WorkModel;
    /// assert_eq!(WorkModel::default().with_muladds_per_ns(16.0).muladds_per_ns, 16.0);
    /// ```
    #[must_use]
    pub fn with_muladds_per_ns(mut self, muladds_per_ns: f64) -> Self {
        self.muladds_per_ns = muladds_per_ns;
        self
    }

    /// Return this model with a different minimum work per lane.
    ///
    /// # Examples
    ///
    /// ```
    /// use tcontract_exec::batch::WorkModel;
    /// assert_eq!(WorkModel::default().with_min_work_per_lane_ns(8_000.0).min_work_per_lane_ns, 8_000.0);
    /// ```
    #[must_use]
    pub fn with_min_work_per_lane_ns(mut self, nanoseconds: f64) -> Self {
        self.min_work_per_lane_ns = nanoseconds;
        self
    }

    /// Return this model with a different minimum fan-out item count.
    #[must_use]
    pub fn with_min_items(mut self, items: usize) -> Self {
        self.min_items = items;
        self
    }

    /// Return this model with a different minimum item count per lane.
    #[must_use]
    pub fn with_min_items_per_lane(mut self, items: usize) -> Self {
        self.min_items_per_lane = items;
        self
    }
}

/// A strategy and the model its `Auto` arm applies.
///
/// # Examples
///
/// ```
/// use tcontract_exec::batch::{BatchPolicy, BatchStrategy};
/// assert_eq!(BatchPolicy::default().strategy, BatchStrategy::Auto);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BatchPolicy {
    /// The route selector.
    pub strategy: BatchStrategy,
    /// The cost model `Auto` applies.
    pub model: WorkModel,
}

impl BatchPolicy {
    /// A policy with `strategy` and the default model.
    #[must_use]
    pub fn new(strategy: BatchStrategy) -> Self {
        Self {
            strategy,
            model: WorkModel::default(),
        }
    }

    /// Return this policy with a different strategy.
    #[must_use]
    pub fn with_strategy(mut self, strategy: BatchStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Return this policy with a different model.
    #[must_use]
    pub fn with_model(mut self, model: WorkModel) -> Self {
        self.model = model;
        self
    }
}

/// What one batch of one route looks like to [`batch_decision`].
///
/// Work is described by an estimate, not by dimensions: each family supplies
/// the multiply-adds of **one** item (GEMM `mnk`, LU `n^3/3`, solve `n^2 k`,
/// QR `2mn^2/3`, with complex elements weighing four). The booleans and
/// `balanced_lanes` say what the route permits; they are how the legacy rules
/// above become inputs instead of hidden constants.
///
/// # Examples
///
/// ```
/// use tcontract_exec::batch::BatchQuery;
///
/// let query = BatchQuery::new(32, 4096.0).with_item_parallel(false);
/// assert_eq!(query.items, 32);
/// assert!(!query.item_parallel);
/// ```
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BatchQuery {
    /// Number of independent items; `0` means there is no work.
    pub items: usize,
    /// Multiply-adds of one item.
    pub item_muladds: f64,
    /// The route permits item-parallel width greater than one, i.e.
    /// `item_threads > 1` is meaningful. `Auto` keeps item threads one when
    /// this is false.
    pub item_parallel: bool,
    /// `Auto` may only keep lanes when every lane is non-empty, the split is
    /// near-even and the batch alone fills the narrowed budget: the packed
    /// driver's balanced-lanes rule.
    pub balanced_lanes: bool,
    /// `Auto` prefers the inner route (one lane, the item kernel gets a team) and
    /// fans out over items only when one item alone cannot use a team: the faer
    /// batched-GEMM rule. The item width is then the width of one item, not the
    /// whole budget.
    pub outer_only_when_item_serial: bool,
    /// Sections (dispatches) one item performs when it runs on a team: 1 for a
    /// kernel that publishes its region once (the packed driver's grid, one faer
    /// GEMM). A kernel that dispatches `s` times per item declares `s`, and the
    /// token charges `s` dispatch costs against the item's serial time.
    pub item_sections: u32,
}

impl BatchQuery {
    /// A query for `items` items of `item_muladds` multiply-adds each, on a
    /// route that permits item parallelism and has no vendor batch route.
    #[must_use]
    pub fn new(items: usize, item_muladds: f64) -> Self {
        Self {
            items,
            item_muladds,
            item_parallel: true,
            balanced_lanes: false,
            outer_only_when_item_serial: false,
            item_sections: 1,
        }
    }

    /// Return this query with the number of sections one item performs.
    #[must_use]
    pub fn with_item_sections(mut self, item_sections: u32) -> Self {
        self.item_sections = item_sections.max(1);
        self
    }

    /// Return this query with the item-parallel capability set.
    #[must_use]
    pub fn with_item_parallel(mut self, item_parallel: bool) -> Self {
        self.item_parallel = item_parallel;
        self
    }

    /// Return this query with the balanced-lanes capability set.
    #[must_use]
    pub fn with_balanced_lanes(mut self, balanced_lanes: bool) -> Self {
        self.balanced_lanes = balanced_lanes;
        self
    }

    /// Return this query with the "outer only when one item cannot use a team"
    /// capability set (the faer batched-GEMM rule).
    #[must_use]
    pub fn with_outer_only_when_item_serial(mut self, value: bool) -> Self {
        self.outer_only_when_item_serial = value;
        self
    }
}

/// The route [`batch_decision`] selected.
///
/// # The route contract
///
/// A decision is the **whole** concurrency of one call, and a route must never
/// discard half of it: `lanes` outer lanes run a contiguous share of the items,
/// each lane may run its items on `item_threads` threads, and `batch_decision`
/// keeps `lanes * item_threads <= budget`. A route that cannot run an item on a
/// sub-team inside a lane either honours both axes or refuses a decision that
/// asks for both — only a forced [`BatchStrategy::Split`] produces `lanes > 1`
/// together with `item_threads > 1` — with
/// [`crate::ExecError::StrategyUnavailable`], before it writes any output. Silently
/// running the items serially inside the lanes is not a route.
///
/// The routes in tcontract, and where each one stands:
///
/// * the faer strategy honours both axes: every item runs with
///   `item_threads` slots as the width it hands faer through `Exec::external`
///   inside its lane's share of the token, which nests there (`crates/tcontract/src/strategy/faer.rs`);
/// * the packed driver refuses it
///   (`crates/tcontract/src/driver/route.rs`): its inner parallelism is a
///   co-scheduled, barrier-bearing team, and a lane worker cannot join one
///   without blocking on an outer barrier;
/// * `contract_batched` refuses it too (`crates/tcontract/src/batch.rs`): its
///   schedule is one item per lane, or the items one after another on the whole
///   context, as the decision log records.
///
/// # Examples
///
/// ```
/// use tcontract_exec::batch::BatchDecision;
/// let d = BatchDecision { lanes: 2, item_threads: 1 };
/// assert_eq!(d.lanes * d.item_threads, 2);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchDecision {
    /// Outer lanes; each lane runs a contiguous share of the items.
    pub lanes: usize,
    /// Threads one item may use. `1` for a route that runs items serially inside
    /// each lane; a [`BatchStrategy::Split`] asks for both lanes and item threads,
    /// and a route that cannot run item threads inside lanes refuses it (see the
    /// type's route contract).
    pub item_threads: usize,
}

/// Decide the route for one batch on the budget and the dispatch costs of `exec`
/// under `policy`.
///
/// Forced strategies are clamped (`lanes <= min(items, budget)`,
/// `lanes * item_threads <= budget`) but never fall back. `Auto` narrows the
/// budget first (see the module docs) and uses the original budget only for the
/// forced strategies.
///
/// Whether a *route* can run the returned decision is the route's own
/// preflight: a decision with both lanes and item threads is honoured or
/// refused before any output write, never partially applied (see
/// [`BatchDecision`]'s route contract).
///
/// # Examples
///
/// ```
/// use std::num::NonZeroUsize;
/// use tcontract_exec::batch::{batch_decision, BatchPolicy, BatchQuery, BatchStrategy};
/// use tcontract_exec::{Exec, Pool};
///
/// let sp = threadteam::Pool::builder().threads(4).build().unwrap();
/// let pool = Pool::borrow(&sp);
/// let exec = Exec::new(&pool, 4).unwrap();
/// let query = BatchQuery::new(2, 1024.0);
/// let split = BatchPolicy::new(BatchStrategy::Split {
///     lanes: NonZeroUsize::new(8).unwrap(),
/// });
/// let d = batch_decision(&split, &exec, &query);
/// // Clamped to the items and the budget.
/// assert_eq!(d.lanes * d.item_threads, exec.budget());
/// assert_eq!(d.lanes, 2.min(exec.budget()));
/// ```
pub fn batch_decision(policy: &BatchPolicy, exec: &Exec<'_>, query: &BatchQuery) -> BatchDecision {
    batch_decision_within(policy, exec, exec.budget(), query)
}

/// [`batch_decision`] for a stage that was narrowed to `budget` slots (at most the
/// token's own) without narrowing the token: the widths minimise over `1..=budget`
/// with the token's dispatch costs ([`Exec::width_for_within`]).
///
/// # Examples
///
/// ```
/// use tcontract_exec::batch::{batch_decision_within, BatchPolicy, BatchQuery};
/// use tcontract_exec::Exec;
///
/// let d = batch_decision_within(&BatchPolicy::default(), &Exec::serial(), 4, &BatchQuery::new(8, 1e6));
/// assert_eq!((d.lanes, d.item_threads), (1, 1));
/// ```
pub fn batch_decision_within(
    policy: &BatchPolicy,
    exec: &Exec<'_>,
    budget: usize,
    query: &BatchQuery,
) -> BatchDecision {
    let budget = budget.max(1).min(exec.budget());
    decide(
        policy,
        budget,
        &|ns, sections| exec.width_for_within(budget, ns, sections),
        query,
    )
}

/// The width in `1..=budget` at which `serial_ns` of work in `sections` sections pays.
type Widths<'a> = &'a dyn Fn(f64, u32) -> usize;

/// [`batch_decision`] over an explicit width rule: the pure body, so a test can
/// give it a rule of its own.
pub(crate) fn decide(
    policy: &BatchPolicy,
    budget: usize,
    widths: Widths<'_>,
    query: &BatchQuery,
) -> BatchDecision {
    match policy.strategy {
        BatchStrategy::Auto => auto_decision(&policy.model, budget, widths, query),
        BatchStrategy::Sequential => BatchDecision {
            lanes: 1,
            item_threads: 1,
        },
        BatchStrategy::Outer => BatchDecision {
            lanes: clamp_lanes(budget, query.items),
            item_threads: 1,
        },
        BatchStrategy::Inner => BatchDecision {
            lanes: 1,
            item_threads: budget.max(1),
        },
        BatchStrategy::Split { lanes } => {
            let lanes = clamp_lanes(budget, query.items).min(lanes.get());
            BatchDecision {
                lanes,
                item_threads: budget / lanes,
            }
        }
    }
}

/// `lanes = min(budget, items)`, never below one.
fn clamp_lanes(budget: usize, items: usize) -> usize {
    items.clamp(1, budget.max(1))
}

/// The `Auto` arm.
///
/// Without `balanced_lanes` this is the issue's generic formula: estimate the
/// work, cap the lanes by the work model, and fan out when the `min_items` and
/// `min_items_per_lane` gates pass. With `balanced_lanes` it is the packed
/// driver's two-stage narrowing, which is what the default [`WorkModel`]
/// reproduces to the lane.
fn auto_decision(
    model: &WorkModel,
    budget: usize,
    widths: Widths<'_>,
    query: &BatchQuery,
) -> BatchDecision {
    let items = query.items;
    if items == 0 || budget < 2 {
        return one_lane(1);
    }
    let item_ns = item_ns(model, query.item_muladds);
    let total_ns = item_ns * items as f64;
    if query.outer_only_when_item_serial {
        faer_auto(widths, budget, query, item_ns, total_ns)
    } else if query.balanced_lanes {
        packed_auto(model, widths, query, item_ns, total_ns)
    } else {
        generic_auto(model, widths, budget, query, item_ns, total_ns)
    }
}

/// The faer batched-GEMM rule: prefer the inner route, and fan out over items
/// only when one item alone is too small for a team (`width_for(item_ns) == 1`)
/// and there are at least two items with work enough to leave the caller.
///
/// This is the item-width-1 rule of `strategy/faer.rs`: a large item gets the
/// whole team, a tiny one gets lanes so the team would not pay barriers for
/// nothing.
fn faer_auto(
    widths: Widths<'_>,
    budget: usize,
    query: &BatchQuery,
    item_ns: f64,
    total_ns: f64,
) -> BatchDecision {
    // A route without item parallelism (a faer item on a token without a team)
    // has no inner width: the items take the lanes.
    let inner = if query.item_parallel {
        widths(item_ns, query.item_sections)
    } else {
        1
    };
    if inner > 1 {
        return one_lane(inner);
    }
    // The lanes are one fork over the items: one section.
    let total = widths(total_ns, 1);
    if total > 1 && query.items > 1 {
        let lanes = budget.min(query.items).min(total);
        return BatchDecision {
            lanes,
            item_threads: 1,
        };
    }
    one_lane(1)
}

/// The issue's generic `auto_lanes`: lanes from the work model alone.
fn generic_auto(
    model: &WorkModel,
    widths: Widths<'_>,
    budget: usize,
    query: &BatchQuery,
    item_ns: f64,
    total_ns: f64,
) -> BatchDecision {
    let items = query.items;
    let by_work = work_lane_cap(model, total_ns);
    let lanes = budget.min(items).min(by_work);
    if lanes >= 2
        && items >= model.min_items
        && items >= lanes.saturating_mul(model.min_items_per_lane)
    {
        return BatchDecision {
            lanes,
            item_threads: 1,
        };
    }
    // One lane: the inner route is the width of one item, when the route
    // permits item parallelism (the faer rule).
    let threads = if query.item_parallel {
        widths(item_ns, query.item_sections)
    } else {
        1
    };
    one_lane(threads)
}

/// The packed driver's two-stage narrowing: `plan/mod.rs:641` narrows the
/// budget over the whole batch, `driver/batch.rs:38` then applies the
/// tiny/balanced rule against the narrowed budget.
fn packed_auto(
    model: &WorkModel,
    widths: Widths<'_>,
    query: &BatchQuery,
    item_ns: f64,
    total_ns: f64,
) -> BatchDecision {
    let items = query.items;
    let narrowed = widths(total_ns, query.item_sections);
    let one_lane_threads = if query.item_parallel { narrowed } else { 1 };
    if narrowed < 2 {
        return one_lane(one_lane_threads);
    }
    let mut lanes = items.min(narrowed).min(work_lane_cap(model, total_ns));
    let gates_pass = lanes >= 2
        && items >= model.min_items
        && items >= lanes.saturating_mul(model.min_items_per_lane);
    if !gates_pass {
        return one_lane(one_lane_threads);
    }
    // Tiny items keep lanes (a team would pay barriers for nothing); otherwise
    // the batch alone must fill the narrowed budget and split near-evenly.
    let tiny = widths(item_ns, query.item_sections) < 2;
    let balanced = items.saturating_mul(10)
        >= 9usize
            .saturating_mul(lanes)
            .saturating_mul(items.div_ceil(lanes));
    if !(tiny || (items >= narrowed && balanced)) {
        lanes = 1;
    }
    BatchDecision {
        lanes,
        item_threads: if lanes > 1 { 1 } else { one_lane_threads },
    }
}

/// Most lanes the estimated total work can fill, given
/// [`WorkModel::min_work_per_lane_ns`]. Zero disables the gate.
fn work_lane_cap(model: &WorkModel, total_ns: f64) -> usize {
    if model.min_work_per_lane_ns > 0.0 {
        let raw = total_ns / model.min_work_per_lane_ns;
        if raw.is_finite() && raw >= 1.0 {
            raw.floor() as usize
        } else {
            1
        }
    } else {
        usize::MAX
    }
}

/// One lane carrying `threads` item threads.
fn one_lane(threads: usize) -> BatchDecision {
    BatchDecision {
        lanes: 1,
        item_threads: threads.max(1),
    }
}

/// Estimated cost of one item in nanoseconds.
fn item_ns(model: &WorkModel, item_muladds: f64) -> f64 {
    let rate = if model.muladds_per_ns > 0.0 {
        model.muladds_per_ns
    } else {
        1.0
    };
    model.item_overhead_ns + item_muladds / rate
}

#[cfg(test)]
mod tests;
