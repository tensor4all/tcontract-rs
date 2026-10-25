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
