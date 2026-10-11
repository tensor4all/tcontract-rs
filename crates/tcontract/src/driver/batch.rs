//! Batch-axis claiming: the width rule.
//!
//! The in-plan batch axis (`plan.stats.batch`, the Hadamard modes) used to run
//! serially inside every thread of the SPMD team, with two team barriers per
//! `(jc, pc)` epoch per entry. For a batch of tiny entries that is all
//! synchronisation: a team of eight ran 10-77x slower than one thread.
//!
//! When [`lanes`] answers more than one, the driver instead cuts the batch into
//! that many contiguous shares and runs each share on its own pool worker,
//! **barrier-free**. A lane runs whole entries with the serial blocking, its
//! own packed `A`, `B` and tile, so every entry's arithmetic is exactly the
//! serial path's and the result is bitwise the serial one at any width.

use tcontract_exec::{Exec, WidthPolicy};

use crate::plan::NS_PER_FLOP;

/// How many lanes the batch axis is cut into: `1` keeps the usual path (a
/// team across the entry's output, or serial), more than one selects the
/// barrier-free lanes. The default partition only; an explicit `StaticGrid` or
/// `DynamicTiles` request never asks.
///
/// The rule, from `h` entries of `m x n x k` on `exec` with the default
/// [`WidthPolicy`] and [`NS_PER_FLOP`]:
///
/// 1. one lane when there is one entry or a budget of one;
/// 2. one lane when the whole batch is too small to leave the caller
///    (`exec.width_for(h * item_ns) == 1`);
/// 3. otherwise lanes when the entry is *tiny*, i.e. alone below the policy's
///    serial threshold, so a team would pay its barriers for nothing, **or**
///    when there are at least as many entries as workers *and* they split
///    evenly over the lanes (the busiest lane takes at most 1/0.9 of the mean
///    share), so the batch alone fills the budget; large entries with fewer
///    entries than workers, or an uneven split (9 entries on 8 lanes leaves
///    one lane with two and the rest idle for half the time), keep the team;
/// 4. the count is the width the whole batch warrants, at most one lane per
///    entry.
pub(super) fn lanes(
    exec: &Exec<'_>,
    h: usize,
    (m, n, k): (usize, usize, usize),
    cplx: bool,
) -> usize {
    let budget = exec.budget();
    if h < 2 || budget < 2 {
        return 1;
    }
    let policy = WidthPolicy::default();
    let flops = 2.0 * (m as f64) * (n as f64) * (k as f64) * if cplx { 4.0 } else { 1.0 };
    let item_ns = flops * NS_PER_FLOP;
    let width = exec.width_for(item_ns * h as f64, &policy);
    if width < 2 {
        return 1;
    }
    let tiny = item_ns < policy.serial_below_ns;
    let lanes = h.min(width);
    // Busiest lane's entries against the mean: `h / (lanes * ceil(h / lanes))`.
    let balanced = 10 * h >= 9 * lanes * h.div_ceil(lanes);
    if tiny || (h >= budget && balanced) {
        lanes
    } else {
        1
    }
}
