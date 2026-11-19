//! A provider's time of one item at a width.

/// A provider's time for one item at a width, without the token's dispatch cost
/// (tensor4all/tcontract-rs#194, #203).
///
/// ```text
/// at(1) = fixed_ns + work_ns
/// at(w) = fixed_ns + parallel_ns + work_ns * min(1, r * task_share) + r * task_ns
///         + per_width_ns * (w - 1)
///         + (slow_ns + slow_per_width_ns * (w - 1) if w < slow_below)
/// r     = ceil(split / w)
/// ```
///
/// `work_ns` is the item's serial compute time. It runs as `split` tasks that
/// the `w` participants take in turn, so width `w` finishes after `r` rounds of
/// at most the largest task, a share `task_share` of the work (`1 / split` when
/// the tasks are equal); each round also adds `task_ns`, what one task costs
/// besides its share (moving its operands to the participant, say).
/// `parallel_ns` is the setup a parallel call adds once, `per_width_ns` what
/// each further participant adds, and `slow_ns` plus `slow_per_width_ns` per
/// further participant what a width below `slow_below` adds when the provider
/// takes a slower path there. `gain_margin` is the time's own relative error: a
/// width above one is taken only when its time plus dispatch, raised by that
/// share, is still below width one's. A provider fits these per host class offline as
/// an [`crate::ItemFit`] row; the batch decision of the crate that consumes this
/// type only minimises them. A provider without a fitted time supplies none,
/// and the query keeps ideal division.
///
/// # Examples
///
/// ```
/// use t4a_cost::ItemTime;
///
/// let t = ItemTime::new(100.0, 8_000.0).with_parallel(1_000.0, 4, 50.0);
/// assert_eq!(t.at(1), 8_100.0);
/// assert_eq!(t.at(4), 100.0 + 1_000.0 + 2_000.0 + 150.0);
/// // Three participants still take two rounds of four equal tasks.
/// assert_eq!(t.at(3), 100.0 + 1_000.0 + 4_000.0 + 100.0);
/// // Beyond `split` a participant only adds its cost.
/// assert!(t.at(8) > t.at(4));
/// // The largest task is half the work and each round adds 200 ns.
/// let uneven = t.with_largest_task(0.5, 200.0);
/// assert_eq!(uneven.at(4), 100.0 + 1_000.0 + 4_000.0 + 200.0 + 150.0);
/// // Below width 8 a slower path adds 1 µs and 100 ns per further participant.
/// let slow = t.with_slow_below(8, 1_000.0, 100.0);
/// assert_eq!(slow.at(4), t.at(4) + 1_000.0 + 300.0);
/// assert_eq!(slow.at(8), t.at(8));
/// ```
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ItemTime {
    /// Nanoseconds one item costs at every width.
    pub fixed_ns: f64,
    /// Serial compute nanoseconds of one item, the part a width divides.
    pub work_ns: f64,
    /// Nanoseconds a width above one adds once.
    pub parallel_ns: f64,
    /// Tasks `work_ns` runs as; at least one.
    pub split: usize,
    /// Share of `work_ns` in the largest task, in `(0, 1]`.
    pub task_share: f64,
    /// Nanoseconds each round of tasks adds besides its share of `work_ns`.
    pub task_ns: f64,
    /// Nanoseconds each participant beyond the first adds.
    pub per_width_ns: f64,
    /// Widths `2..slow_below` take the provider's slower path.
    pub slow_below: usize,
    /// Nanoseconds that slower path adds once.
    pub slow_ns: f64,
    /// Nanoseconds that slower path adds per participant beyond the first.
    pub slow_per_width_ns: f64,
    /// Relative error of this time; a width above one must beat width one by
    /// more than it. Zero by default.
    pub gain_margin: f64,
}

impl ItemTime {
    /// An item of `fixed_ns + work_ns` at width one that no width speeds up,
    /// until [`ItemTime::with_parallel`] says how it divides.
    #[must_use]
    pub fn new(fixed_ns: f64, work_ns: f64) -> Self {
        Self {
            fixed_ns,
            work_ns,
            parallel_ns: 0.0,
            split: 1,
            task_share: 1.0,
            task_ns: 0.0,
            per_width_ns: 0.0,
            slow_below: 0,
            slow_ns: 0.0,
            slow_per_width_ns: 0.0,
            gain_margin: 0.0,
        }
    }

    /// Return this time with its parallel terms: `parallel_ns` once above width
    /// one, the work as `split` equal tasks taken in turn, and `per_width_ns`
    /// for each participant beyond the first.
    #[must_use]
    pub fn with_parallel(mut self, parallel_ns: f64, split: usize, per_width_ns: f64) -> Self {
        self.parallel_ns = parallel_ns;
        self.split = split.max(1);
        self.task_share = 1.0 / self.split as f64;
        self.per_width_ns = per_width_ns;
        self
    }

    /// Return this time with the largest task a share `task_share` of the work
    /// instead of `1 / split`, and `task_ns` added per round of tasks.
    #[must_use]
    pub fn with_largest_task(mut self, task_share: f64, task_ns: f64) -> Self {
        self.task_share = task_share;
        self.task_ns = task_ns;
        self
    }

    /// Return this time with `slow_ns + slow_per_width_ns * (w - 1)` added at the
    /// widths `w` in `2..slow_below`.
    #[must_use]
    pub fn with_slow_below(
        mut self,
        slow_below: usize,
        slow_ns: f64,
        slow_per_width_ns: f64,
    ) -> Self {
        self.slow_below = slow_below;
        self.slow_ns = slow_ns;
        self.slow_per_width_ns = slow_per_width_ns;
        self
    }

    /// Return this time with a width above one taken only when its time plus
    /// dispatch, times `1 + gain_margin`, is below width one's.
    #[must_use]
    pub fn with_gain_margin(mut self, gain_margin: f64) -> Self {
        self.gain_margin = gain_margin;
        self
    }

    /// Nanoseconds of one item at `width` participants (zero is width one),
    /// without the token's dispatch cost.
    pub fn at(&self, width: usize) -> f64 {
        if width <= 1 {
            return self.fixed_ns + self.work_ns;
        }
        let slow = if width < self.slow_below {
            self.slow_ns + self.slow_per_width_ns * (width - 1) as f64
        } else {
            0.0
        };
        let rounds = self.split.max(1).div_ceil(width) as f64;
        self.fixed_ns
            + self.parallel_ns
            + self.work_ns * (rounds * self.task_share).min(1.0)
            + rounds * self.task_ns
            + self.per_width_ns * (width - 1) as f64
            + slow
    }
}
