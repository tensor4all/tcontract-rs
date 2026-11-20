//! The generic item-time fit, as the Rust contract between a fitted row and
//! the [`ItemTime`] the batch decision minimises.
//!
//! `experiments/faer-item-width/item_fit.py` (tcontract-rs) fits one row per
//! host class ([`crate::CpuClass`]) and dtype for every provider the same way;
//! only the work, the tasks and the slow band are provider-specific features.
//! [`ItemFit`] is that row as data, and [`ItemFit::item_time`] is the one
//! row-to-time rule; the fit itself stays in the experiment, so two providers
//! cannot drift on the mapping and a provider does not copy it.

use crate::ItemTime;

/// One fitted row of the generic item-time fit: the six parameters
/// `experiments/faer-item-width/item_fit.py` fits for every provider the same
/// way, keyed by a host class ([`crate::CpuClass`]) and a dtype. The fit itself
/// stays an experiment; this row is the contract between a provider's fitted
/// data and the [`ItemTime`] the batch decision minimises, so two providers
/// cannot drift on the mapping and neither copies it.
///
/// The row prices one item of `W` multiply-adds exactly as the fit's model
/// does:
///
/// ```text
/// at(1) = fixed_ns + work,  work = W / muladds_per_ns
/// at(w) = fixed_ns + parallel_ns + work * min(1, r * L) + r * task_ns
///         + per_width_ns * (w - 1)
///         + (slow * work + slow_per_width_ns * (w - 1)  if w < slow_below)
/// r     = ceil(P / w)
/// ```
///
/// The item's work `W` (its multiply-adds), its `P` tasks, the share `L` of the
/// largest, what a round of tasks adds (`task_ns`) and the widths of its slower
/// path (`slow_below`) are the provider's *features*, supplied per item to
/// [`ItemFit::item_time`]; the fields are the fitted parameters `F`, `R`, `O`,
/// `C`, `Q0` and `Q1` of `item_fit.py`: the slow band's fixed part a share of
/// the work, its per-participant part in nanoseconds, as the fit fits them,
/// and the margin the fit derives from its own error.
///
/// # Examples
///
/// ```
/// use t4a_cost::ItemFit;
///
/// // An item of 4e5 multiply-adds on a row of 50 muladds/ns: 8 µs of work,
/// // as 4 equal tasks, with a slower path below width 6.
/// let row = ItemFit {
///     fixed_ns: 100.0,
///     muladds_per_ns: 50.0,
///     parallel_ns: 1_000.0,
///     per_width_ns: 10.0,
///     slow: 0.5,
///     slow_per_width_ns: 2_000.0,
///     gain_margin: 0.25,
/// };
/// let t = row.item_time(4e5, 4, 0.25, 0.0, 6);
/// assert_eq!(t.at(1), 8_100.0);
/// // Width 6 leaves the slow band; width 8 no longer divides the work.
/// assert!(t.at(6) < t.at(4));
/// assert!(t.at(8) > t.at(6));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ItemFit {
    /// `F`: nanoseconds one item costs at every width.
    pub fixed_ns: f64,
    /// `R`: the provider's serial multiply-adds per nanosecond on that class.
    pub muladds_per_ns: f64,
    /// `O`: nanoseconds a width above one adds once.
    pub parallel_ns: f64,
    /// `C`: nanoseconds each participant beyond the first adds.
    pub per_width_ns: f64,
    /// `Q0`: the slower path's share of the work, once.
    pub slow: f64,
    /// `Q1`: nanoseconds the slower path adds per participant beyond the
    /// first.
    pub slow_per_width_ns: f64,
    /// The row's own error, derived after the fit: the median
    /// `|model / measured - 1|` at the widths above one it chooses on the fit
    /// grid. A width above one must beat width one by more than it.
    pub gain_margin: f64,
}

impl ItemFit {
    /// The provider's [`ItemTime`] for one item of `muladds` multiply-adds run
    /// as `tasks` tasks, the largest a share `largest_share` of the work, each
    /// round of tasks adding `task_ns`, with a slower path at the widths below
    /// `slow_below` (zero: none).
    ///
    /// This is the one row-to-time rule: a provider computes its features
    /// (the tasks from its partition, the slow band from its kernel's rule) and
    /// hands them to it with the item's work.
    #[must_use]
    pub fn item_time(
        &self,
        muladds: f64,
        tasks: usize,
        largest_share: f64,
        task_ns: f64,
        slow_below: usize,
    ) -> ItemTime {
        let work = muladds / self.muladds_per_ns;
        ItemTime::new(self.fixed_ns, work)
            .with_parallel(self.parallel_ns, tasks, self.per_width_ns)
            .with_largest_task(largest_share, task_ns)
            .with_slow_below(slow_below, self.slow * work, self.slow_per_width_ns)
            .with_gain_margin(self.gain_margin)
    }
}

#[cfg(test)]
mod tests;
