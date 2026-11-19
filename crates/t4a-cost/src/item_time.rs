//! A provider's time of one item at a width.

/// A provider's time for one item at a width, without the token's dispatch cost
/// (tensor4all/tcontract-rs#194).
///
/// ```text
/// at(1) = fixed_ns + work_ns
/// at(w) = fixed_ns + parallel_ns + work_ns / min(w, split) + per_width_ns * (w - 1)
///         + (slow_ns + slow_per_width_ns * (w - 1) if w < slow_below)
/// ```
///
/// `work_ns` is the item's serial compute time, divided over at most `split`
/// participants; `parallel_ns` is the setup a parallel call adds once,
/// `per_width_ns` what each further participant adds, and `slow_ns` plus
/// `slow_per_width_ns` per further participant what a width below `slow_below`
/// adds when the provider takes a slower path there. A provider fits these per
/// host class offline as an [`crate::ItemFit`] row; the batch decision of the
/// crate that consumes this type only minimises them. A provider without a
/// fitted time supplies none, and the query keeps ideal division.
///
/// # Examples
///
/// ```
/// use t4a_cost::ItemTime;
///
/// let t = ItemTime::new(100.0, 8_000.0).with_parallel(1_000.0, 4, 50.0);
/// assert_eq!(t.at(1), 8_100.0);
/// assert_eq!(t.at(4), 100.0 + 1_000.0 + 2_000.0 + 150.0);
/// // Beyond `split` a participant only adds its cost.
/// assert!(t.at(8) > t.at(4));
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
    /// Most participants `work_ns` divides over; at least one.
    pub split: usize,
    /// Nanoseconds each participant beyond the first adds.
    pub per_width_ns: f64,
    /// Widths `2..slow_below` take the provider's slower path.
    pub slow_below: usize,
    /// Nanoseconds that slower path adds once.
    pub slow_ns: f64,
    /// Nanoseconds that slower path adds per participant beyond the first.
    pub slow_per_width_ns: f64,
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
            per_width_ns: 0.0,
            slow_below: 0,
            slow_ns: 0.0,
            slow_per_width_ns: 0.0,
        }
    }

    /// Return this time with its parallel terms: `parallel_ns` once above width
    /// one, the work divided over at most `split` participants, and
    /// `per_width_ns` for each participant beyond the first.
    #[must_use]
    pub fn with_parallel(mut self, parallel_ns: f64, split: usize, per_width_ns: f64) -> Self {
        self.parallel_ns = parallel_ns;
        self.split = split.max(1);
        self.per_width_ns = per_width_ns;
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
        self.fixed_ns
            + self.parallel_ns
            + self.work_ns / width.min(self.split.max(1)) as f64
            + self.per_width_ns * (width - 1) as f64
            + slow
    }
}
