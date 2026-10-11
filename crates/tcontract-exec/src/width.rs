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
