//! Placement helpers: pools from `threadteam`, a worker-thread runner and
//! ledger readers, so tests state placement claims through one mechanism.
#![allow(dead_code)]

use tcontract_exec::{Exec, Pool};
use threadteam::ledger;

/// A pool of `n` workers.
pub fn pool(n: usize) -> threadteam::Pool {
    threadteam::Pool::builder()
        .threads(n)
        .build()
        .expect("pool")
}

/// The Apple M5 dispatch costs (microseconds at widths 2, 4, 6: team 3.0, 2.3, 1.4; in pool 6.0,
/// 11, 5.5; outside 14, 18, 19) scaled by this host class's GEMM rate over the M5's
/// (`ns_per_flop() / 0.0185`). The tests that pin a route or a width were written against the
/// M5 pair of costs and rate; scaling the costs with the rate keeps the ratio of dispatch cost to
/// work the same on every host class, so they do not depend on which measured table the host has.
pub fn reference_costs() -> threadteam::DispatchCosts {
    use std::time::Duration;
    use threadteam::WidthCosts;
    let scale = tcontract_exec::host::ns_per_flop() / 0.0185;
    let row = |us: [f64; 3]| {
        WidthCosts::new(us.map(|x| Duration::from_nanos((x * 1000.0 * scale).round() as u64)))
    };
    threadteam::DispatchCosts::new(
        row([3.0, 2.3, 1.4]),
        row([6.0, 11.0, 5.5]),
        row([14.0, 18.0, 19.0]),
    )
}

/// A pool of `n` workers with the [`reference_costs`], independent of the host table.
pub fn pool_with_reference_costs(n: usize) -> threadteam::Pool {
    threadteam::Pool::builder()
        .threads(n)
        .dispatch_costs(reference_costs())
        .build()
        .expect("pool")
}

/// The ledger root of `exec` (every pooled token from `Exec::new` has one).
pub fn root(exec: &mut Exec<'_>) -> u64 {
    exec.token()
        .ledger_root()
        .expect("a token from Exec::new is a ledger root")
}

/// Most tasks of `root` that were live at once.
pub fn max_live(root: u64) -> usize {
    ledger::max_live(root)
}

/// Times work of `root` had to be installed into the pool from outside it.
pub fn installs(root: u64) -> usize {
    ledger::installs(root)
}

/// Run `f` on a worker of `pool`, the way a task of another operation on the
/// same pool would call: from there `Exec::broadcast` is refused
/// (`CallerInPool`) and entries run in place.
pub fn on_worker<R: Send>(pool: &Pool<'_>, f: impl FnOnce() -> R + Send) -> R {
    Exec::new(pool, pool.size())
        .expect("budget")
        .into_token()
        .external(|_| f())
}
