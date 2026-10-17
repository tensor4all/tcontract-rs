/// Cost model for choosing a barrier-free partition width from the work.
///
/// `T_par(k) = entry_base_ns + entry_per_thread_ns * k + serial_ns / k` for
/// `k > 1`; work below `serial_below_ns` always runs serially. The defaults
/// are provisional values from `experiments/rayon-entry` (M5 Max,
/// 2026-09-29); kernels replace them with their own measured policy.
///
/// # Examples
///
/// ```
/// let p = tcontract_exec::WidthPolicy::default();
/// assert_eq!(p.serial_below_ns, 50_000.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WidthPolicy {
    /// Estimated serial time below which work never leaves the caller.
    pub serial_below_ns: f64,
    /// Fixed part of the pool entry cost.
    pub entry_base_ns: f64,
    /// Entry cost added per participating worker.
    pub entry_per_thread_ns: f64,
}

impl Default for WidthPolicy {
    fn default() -> Self {
        Self {
            serial_below_ns: 50_000.0,
            entry_base_ns: 5_000.0,
            entry_per_thread_ns: 5_000.0,
        }
    }
}

/// Partition width for work estimated at `serial_ns` on one thread,
/// minimizing the [`WidthPolicy`] cost model within `budget` threads.
///
/// This is the one body of the width rule: [`Exec::width_for`](crate::Exec::width_for)
/// and the pure batch policy module both call it, so a plan-selected policy and a
/// route decision cannot drift.
pub(crate) fn width_for(serial_ns: f64, policy: &WidthPolicy, budget: usize) -> usize {
    // NaN or small work stays serial.
    if budget == 1 || serial_ns.is_nan() || serial_ns < policy.serial_below_ns {
        return 1;
    }
    let cost = |k: usize| {
        if k == 1 {
            serial_ns
        } else {
            policy.entry_base_ns + policy.entry_per_thread_ns * k as f64 + serial_ns / k as f64
        }
    };
    (1..=budget)
        .min_by(|&a, &b| cost(a).total_cmp(&cost(b)))
        .unwrap_or(1)
}
