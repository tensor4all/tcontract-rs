//! Optional per-phase timing of the packed driver.
//!
//! Off unless the `phase-timing` feature is on, and every call site is behind
//! `#[cfg(feature = "phase-timing")]`, so a default build pays nothing. It is an
//! instrument in the same spirit as `benchmarks/scripts/idle_cpus.py`: it answers
//! "which phase of the packed route absorbs this case's time", which nothing else
//! in the tree can on this host - `perf` events are blocked
//! (`kernel.perf_event_paranoid` is 4), `/proc/<pid>/stat` masks the user PC, and
//! Yama refuses the ptrace a sampler would need.
//!
//! Read it a call late: the counters are printed and cleared at the *start* of the
//! next `execute_raw`, so one line describes the execution that just finished. The
//! last four numbers are the phases; `setup` is that execution's wall time minus
//! those four, i.e. the loops, the epoch bookkeeping and everything between the
//! phases.
//!
//! # What it does and does not mean
//!
//! * **Measure at one thread.** The counters sum *worker* time, and with N workers
//!   the phases can add up to more than the call's wall time, so the percentages
//!   stop being wall-clock shares (and `setup` saturates at zero). At one thread
//!   they are shares of the call, which is the reading the campaign's design work
//!   needed.
//! * **One execution at a time.** The counters are process-global and the snapshot
//!   is taken at the next entry, so two overlapping calls mix or drop each other's
//!   numbers. Report from a serialized session.
//! * **`writeback` covers `emit_tile` only.** A Direct family writes `D` inside the
//!   tile call, so that store is charged to `kernel` and `writeback` stays 0 - which
//!   is why the instrument's first documented example reads 0 for a Direct case.
//! * **`pack_b`** is timed at both call sites (static and dynamic) but not on the
//!   direct-B path, where nothing is packed.
//! * **The phases are not disjoint on the blocked outer traversal.** Its own work (the
//!   per-block gathers, scatters and row order, the grid permutation, and the stores its
//!   emit loop makes) is charged to `writeback` for the stores and to `setup` for the
//!   rest; `setup` is a remainder, not an exclusive measurement. Per-block scopes were
//!   tried for this and removed: clock reads per block on this host cost more than the
//!   intervals they measure.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Phase names, indexed like the counters.
pub(crate) const NAMES: [&str; 5] = ["pack_a", "pack_b", "kernel", "writeback", "setup"];

static NANOS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
static TOTAL: AtomicU64 = AtomicU64::new(0);

/// Accumulates one phase while it is alive.
#[derive(Debug)]
pub(crate) struct Scope(usize, Instant);

impl Drop for Scope {
    fn drop(&mut self) {
        NANOS[self.0].fetch_add(self.1.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

/// Start timing phase `i` of [`NAMES`].
#[inline]
pub(crate) fn scope(i: usize) -> Scope {
    assert!(i < 4, "phase index");
    Scope(i, Instant::now())
}

/// Record a whole `execute_raw` call, so the remainder can be reported as setup.
#[inline]
pub(crate) fn add_total(nanos: u64) {
    TOTAL.fetch_add(nanos, Ordering::Relaxed);
}

/// Print the previous call's phase totals and clear them, when `TCONTRACT_PHASE` is set.
pub(crate) fn report_previous() {
    if std::env::var_os("TCONTRACT_PHASE").is_none() {
        return;
    }
    let total = TOTAL.swap(0, Ordering::Relaxed);
    let mut ns = [0u64; 5];
    let mut accounted = 0u64;
    for i in 0..4 {
        ns[i] = NANOS[i].swap(0, Ordering::Relaxed);
        accounted += ns[i];
    }
    ns[4] = total.saturating_sub(accounted);
    if total == 0 {
        return;
    }
    let parts: Vec<String> = NAMES
        .iter()
        .zip(ns)
        .map(|(n, v)| {
            format!(
                "{n}={:.3}ms({:>4.1}%)",
                v as f64 / 1e6,
                100.0 * v as f64 / total as f64
            )
        })
        .collect();
    eprintln!(
        "PHASE total={:.3}ms {}",
        total as f64 / 1e6,
        parts.join(" ")
    );
}
