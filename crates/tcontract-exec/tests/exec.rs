//! The resource bundle over `strided-exec`: budgets, ownership of the pool, the
//! placement invariants (checked through the `strided-exec` ledger) and the
//! global-pool rule for a serial `Exec`.
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use strided_exec::ledger;
use tcontract_exec::{BatchPolicy, BatchStrategy, Exec, ExecError, Pool};

fn pool(n: usize) -> strided_exec::Pool {
    strided_exec::Pool::builder().threads(n).build().unwrap()
}

fn nz(n: usize) -> std::num::NonZeroUsize {
    n.try_into().unwrap()
}

#[test]
fn pool_is_send_sync_and_shared_wrappers_keep_separate_workspaces() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Pool<'static>>();
    let raw = Arc::new(pool(2));
    let weak = Arc::downgrade(&raw);
    let a = Pool::shared(Arc::clone(&raw));
    let b = Pool::shared(Arc::clone(&raw));
    assert_eq!(a.size(), b.size());
    assert!(!std::ptr::eq(a.workspace(), b.workspace()));
    drop(raw);
    assert!(weak.upgrade().is_some(), "wrappers keep the pool alive");
    drop(a);
    assert!(weak.upgrade().is_some());
    drop(b);
    assert!(weak.upgrade().is_none(), "the last wrapper releases it");
}

#[test]
fn owned_pool_runs_work() {
    let owned = Pool::owned(pool(2));
    let mut exec = Exec::new(&owned, 2).unwrap();
    let n = AtomicUsize::new(0);
    exec.token().fork(4, |_, _| {
        n.fetch_add(1, Ordering::Relaxed);
    });
    assert_eq!(n.into_inner(), 4);
}

#[test]
fn budget_is_clamped_once_and_zero_is_an_error() {
    let sp = pool(4);
    let p = Pool::borrow(&sp);
    assert_eq!(Exec::new(&p, 2).unwrap().budget(), 2);
    assert_eq!(Exec::new(&p, 99).unwrap().budget(), p.size());
    assert!(matches!(
        Exec::new(&p, 0),
        Err(ExecError::Placement(strided_exec::Error::ZeroBudget))
    ));
    let narrowed = Exec::new(&p, 4).unwrap().limit(nz(2));
    assert_eq!(narrowed.budget(), 2);
    assert_eq!(
        Exec::new(&p, 2).unwrap().limit(nz(8)).budget(),
        2,
        "limit never raises"
    );
}

#[test]
fn bundle_keeps_policy_workspace_and_budget_through_reborrow_and_limit() {
    let sp = pool(4);
    let p = Pool::borrow(&sp);
    let mut exec = Exec::new(&p, 4)
        .unwrap()
        .with_batch_policy(BatchPolicy::new(BatchStrategy::Outer));
    {
        let r = exec.reborrow();
        assert_eq!(r.budget(), 4);
        assert_eq!(r.batch_policy().strategy, BatchStrategy::Outer);
        let ws = r.workspace().expect("a pool lends its arena");
        assert!(std::ptr::addr_eq(
            ws as *const dyn tcontract_exec::WorkspaceProvider,
            p.workspace() as *const tcontract_exec::ArenaProvider
        ));
    }
    let narrowed = exec.limit(nz(2));
    assert_eq!(narrowed.budget(), 2);
    assert_eq!(narrowed.batch_policy().strategy, BatchStrategy::Outer);
    assert!(narrowed.workspace().is_some());
    assert!(Exec::serial().workspace().is_none());
    assert_eq!(Exec::serial().batch_policy().strategy, BatchStrategy::Auto);
}

#[test]
fn serial_runs_on_the_calling_thread() {
    let caller = std::thread::current().id();
    let mut exec = Exec::serial();
    assert_eq!(exec.budget(), 1);
    let seen = Mutex::new(Vec::new());
    exec.token().fork(3, |_, _| {
        seen.lock().unwrap().push(std::thread::current().id());
    });
    exec.token().for_ranges(10, nz(1), |_| {
        seen.lock().unwrap().push(std::thread::current().id());
    });
    let seen = seen.into_inner().unwrap();
    assert!(!seen.is_empty());
    assert!(seen.iter().all(|t| *t == caller));
}

#[test]
fn width_one_pool_token_stays_on_the_caller() {
    let sp = pool(4);
    let p = Pool::borrow(&sp);
    let mut exec = Exec::new(&p, 1).unwrap();
    let caller = std::thread::current().id();
    let ran = AtomicUsize::new(0);
    exec.token().fork(5, |_, _| {
        assert_eq!(std::thread::current().id(), caller);
        ran.fetch_add(1, Ordering::Relaxed);
    });
    assert_eq!(ran.into_inner(), 5);
}

#[test]
fn forked_bundles_split_the_budget_and_stay_within_it() {
    let sp = pool(4);
    let p = Pool::borrow(&sp);
    let exec = Exec::new(&p, 4).unwrap();
    let ws = exec.workspace();
    let policy = *exec.batch_policy();
    let mut token = exec.into_token();
    let root = token.ledger_root().unwrap();
    let share_sum = AtomicUsize::new(0);
    token.reborrow().fork(2, |_, child| {
        let child = Exec::from_token(child, ws, policy);
        assert!(child.workspace().is_some());
        share_sum.fetch_add(child.budget(), Ordering::Relaxed);
        // Nested split under the same root.
        let mut child = child;
        child.token().fork(4, |_, _| {
            std::hint::black_box(0);
        });
    });
    assert_eq!(share_sum.into_inner(), 4, "shares add up to the budget");
    assert!(ledger::max_live(root) <= 4, "{}", ledger::max_live(root));
}

#[test]
fn broadcast_through_the_bundle_is_concurrent_or_refused_never_partial() {
    let sp = pool(3);
    let p = Pool::borrow(&sp);
    let mut exec = Exec::new(&p, 3).unwrap();
    let barrier = Barrier::new(exec.budget());
    let ids = Mutex::new(HashSet::new());
    exec.token()
        .broadcast(|_| {
            barrier.wait();
            ids.lock().unwrap().insert(std::thread::current().id());
        })
        .expect("a root token called from a plain thread");
    assert_eq!(ids.into_inner().unwrap().len(), 3);

    // A forked child with a share above one is refused and nothing runs.
    let sp4 = pool(4);
    let p4 = Pool::borrow(&sp4);
    let mut exec = Exec::new(&p4, 4).unwrap();
    let refused = AtomicUsize::new(0);
    let ran = AtomicUsize::new(0);
    exec.token().fork(2, |_, child| {
        assert_eq!(child.budget().get(), 2);
        if child.broadcast(|_| _ = ran.fetch_add(1, Ordering::Relaxed))
            == Err(strided_exec::Refused::ForkChild)
        {
            refused.fetch_add(1, Ordering::Relaxed);
        }
    });
    assert_eq!(ran.into_inner(), 0);
    assert_eq!(refused.into_inner(), 2);
}

#[test]
fn prepare_for_is_a_noop_for_a_refused_broadcast() {
    use tcontract_exec::{Workspace, WorkspaceReq};
    let sp = pool(2);
    let p = Pool::borrow(&sp);
    let mut exec = Exec::new(&p, 2).unwrap();
    let req = WorkspaceReq {
        a_bytes: 4096,
        ..Default::default()
    };
    exec.token().fork(1, |_, child| {
        let mut ws = Workspace::with_req(&req);
        let mut child = Exec::from_token(child, None, BatchPolicy::default());
        // budget 2 on a fork child: the broadcast is refused, one slot stays.
        ws.prepare_for(&mut child);
        assert_eq!(ws.slots(), 1);
    });
}

#[test]
fn width_for_and_batch_decision_read_the_token_budget() {
    let sp = pool(4);
    let p = Pool::borrow(&sp);
    let exec = Exec::new(&p, 4).unwrap();
    let w = exec.width_for(1e9, &tcontract_exec::WidthPolicy::default());
    assert!((2..=4).contains(&w), "{w}");
    assert_eq!(
        Exec::serial().width_for(1e9, &tcontract_exec::WidthPolicy::default()),
        1
    );
}
