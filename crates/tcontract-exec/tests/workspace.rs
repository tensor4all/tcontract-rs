//! Workspace ownership: no allocation for a zero requirement, page-aligned
//! reuse, fresh buffers on re-entry, exclusive team sets, and per-worker
//! first touch.
use tcontract_exec::*;

#[test]
fn zero_requirement_never_allocates() {
    let arena = ArenaProvider::traced();
    let _ = arena.trace_take();
    arena.with_worker(&WorkspaceReq::default(), &mut |_, _, _| {});
    let _lease = arena.take_team(&WorkspaceReq::default(), 1, 1);
    assert!(arena.trace_take().is_empty());
}

#[test]
fn page_aligned_and_reused() {
    let arena = ArenaProvider::default();
    let req = WorkspaceReq {
        a_bytes: 1000,
        tile_bytes: 64,
        ..Default::default()
    };
    let (mut p1, mut t1) = (0usize, 0usize);
    arena.with_worker(&req, &mut |a, t, _| {
        p1 = a as usize;
        t1 = t as usize;
    });
    let (mut p2, mut t2) = (0usize, 0usize);
    arena.with_worker(&req, &mut |a, t, _| {
        p2 = a as usize;
        t2 = t as usize;
    });
    assert_eq!(p1 % 4096, 0);
    assert_eq!(t1 % 4096, 0);
    assert_eq!(p1, p2, "the second call must reuse the A block");
    assert_eq!(t1, t2, "and the tile");
}

#[test]
fn reentrant_execute_uses_fresh_buffers() {
    let arena = ArenaProvider::default();
    let req = WorkspaceReq {
        a_bytes: 64,
        ..Default::default()
    };
    arena.with_worker(&req, &mut |outer, _, _| {
        let mut inner = outer;
        arena.with_worker(&req, &mut |a, _, _| inner = a);
        assert_ne!(outer, inner);
    });
    // The outer slot is usable again afterwards.
    let (mut first, mut second) = (0usize, 0usize);
    arena.with_worker(&req, &mut |a, _, _| first = a as usize);
    arena.with_worker(&req, &mut |a, _, _| second = a as usize);
    assert_eq!(first, second);
}

#[test]
fn concurrent_team_sets_never_share_buffers() {
    let arena = ArenaProvider::default();
    let req = WorkspaceReq {
        b_bytes: 4096,
        barriers: 2,
        ..Default::default()
    };
    // The driver sizes the panel it needs, because only it knows the element
    // type; the lease provides the storage and the barriers.
    let mut a = arena.take_team(&req, 2, 2);
    let a_ptr = a.panel(req.b_bytes);
    let mut b = arena.take_team(&req, 2, 2);
    let b_ptr = b.panel(req.b_bytes);
    assert_ne!(a_ptr, b_ptr);
    assert_eq!(a.barriers().len(), 2);
    assert_eq!(b.barriers().len(), 2);
    drop(a);
    // The returned set is what the next lease reuses, not a fresh allocation.
    let mut c = arena.take_team(&req, 2, 2);
    assert_eq!(c.panel(req.b_bytes), a_ptr);
    assert_eq!(c.barriers().len(), 2);
}

#[test]
fn trimmed_storage_is_released_and_live_storage_is_not() {
    let arena = ArenaProvider::default();
    let req = WorkspaceReq {
        a_bytes: 1 << 16,
        b_bytes: 1 << 16,
        ..Default::default()
    };
    arena.with_worker(&req, &mut |_, _, _| {});
    let mut lease = arena.take_team(&req, 1, 1);
    lease.panel(req.b_bytes);
    let before = arena.retained_bytes();
    assert!(before >= 2 * (1 << 16), "retained {before}");
    arena.trim();
    // The worker slot was idle, so its storage is gone; the leased team set is
    // live and must survive.
    let after = arena.retained_bytes();
    assert!(after < before, "before {before}, after {after}");
    assert!(after >= 1 << 16, "the live lease was released");
    drop(lease);
    arena.trim();
    assert_eq!(arena.retained_bytes(), 0);
}

/// Worker buffers are allocated by the thread that writes them, which is the
/// property that makes their first touch node-local.
#[test]
fn worker_buffers_are_allocated_on_the_worker() {
    let arena = ArenaProvider::traced();
    let _ = arena.trace_take();
    let req = WorkspaceReq {
        a_bytes: 1 << 16,
        tile_bytes: 4096,
        ..Default::default()
    };
    let tid = std::thread::scope(|s| {
        s.spawn(|| {
            arena.with_worker(&req, &mut |_, _, _| {});
            std::thread::current().id()
        })
        .join()
        .unwrap()
    });
    let recorded = arena.trace_take();
    assert!(!recorded.is_empty(), "the worker's buffers were not grown");
    assert!(
        recorded.iter().all(|(t, _)| *t == tid),
        "storage was first touched off the worker: {recorded:?}"
    );
}

/// Two owners used from one caller never share a worker slot, even though the
/// thread-local handle is per thread.
#[test]
fn two_owners_do_not_share_a_worker_slot() {
    let a = ArenaProvider::default();
    let b = ArenaProvider::default();
    let req = WorkspaceReq {
        a_bytes: 4096,
        ..Default::default()
    };
    let (mut pa, mut pb) = (0usize, 0usize);
    a.with_worker(&req, &mut |p, _, _| pa = p as usize);
    b.with_worker(&req, &mut |p, _, _| pb = p as usize);
    assert_ne!(pa, pb);
}

/// A callback that unwinds must not leave its worker slot marked as borrowed:
/// the next call on the thread reuses (and grows) the slot, and `trim` can
/// release it, instead of every later call taking the fresh-buffer path.
#[test]
fn a_panicking_callback_releases_the_worker_slot() {
    let arena = ArenaProvider::default();
    let small = WorkspaceReq {
        a_bytes: 4096,
        ..WorkspaceReq::default()
    };
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        arena.with_worker(&small, &mut |_, _, _| panic!("kernel panicked"));
    }));
    assert!(unwound.is_err());
    let big = WorkspaceReq {
        a_bytes: 1 << 16,
        ..WorkspaceReq::default()
    };
    arena.with_worker(&big, &mut |_, _, _| {});
    assert!(
        arena.retained_bytes() >= 1 << 16,
        "the slot stayed borrowed after the panic: {} bytes retained",
        arena.retained_bytes()
    );
    arena.trim();
    assert_eq!(arena.retained_bytes(), 0);
}

/// A caller that wants a serial steady state keeps its own provider: the first
/// call allocates, the next reuses, `stats` reports both directions and `trim`
/// releases only the idle half.
#[test]
fn a_caller_owned_provider_reuses_and_accounts() {
    let arena = ArenaProvider::default();
    let req = WorkspaceReq {
        a_bytes: 1 << 16,
        tile_bytes: 4096,
        b_bytes: 1 << 16,
        team_scatter: 8,
        ..Default::default()
    };
    let warm = |arena: &ArenaProvider| {
        arena.with_worker(&req, &mut |_, _, _| {});
        let mut lease = arena.take_team(&req, 1, 1);
        lease.panel(req.b_bytes);
    };
    let cold = arena.stats();
    assert_eq!(
        cold,
        WorkspaceStats::default(),
        "an idle owner holds nothing"
    );
    warm(&arena);
    let hot = arena.stats();
    assert!(hot.retained_bytes >= 2 * (1 << 16), "{hot:?}");
    assert_eq!(hot.leased_bytes, 0, "nothing is leased between calls");
    // A second identical call reuses: the owner does not grow.
    warm(&arena);
    assert_eq!(arena.stats().retained_bytes, hot.retained_bytes);
    arena.trim();
    assert_eq!(
        arena.stats(),
        WorkspaceStats::default(),
        "trim kept idle storage"
    );
    // Still usable, and it grows again rather than holding the trimmed bytes.
    warm(&arena);
    assert!(arena.stats().retained_bytes > 0);
}

/// A live lease is accounted separately from the idle storage, and `trim` on
/// another thread never releases it.
#[test]
fn a_live_lease_is_accounted_and_survives_a_trim() {
    let arena = ArenaProvider::default();
    let req = WorkspaceReq {
        b_bytes: 1 << 16,
        barriers: 1,
        ..Default::default()
    };
    let mut lease = arena.take_team(&req, 2, 2);
    lease.panel(req.b_bytes);
    let live = arena.stats();
    assert!(live.leased_bytes >= 1 << 16, "{live:?}");
    assert!(live.leased_bytes <= live.retained_bytes, "{live:?}");
    arena.trim();
    assert!(
        arena.stats().leased_bytes >= 1 << 16,
        "trim released a live lease"
    );
    assert_eq!(arena.retained_panel_bytes(), 1 << 16);
    drop(lease);
    assert_eq!(arena.stats().leased_bytes, 0);
}

/// `stats` and `trim` run against a worker that is checking storage in and out.
/// Before the checkout/return rule, the snapshot and `trim` read and freed
/// buffers another thread was writing.
#[test]
fn stats_and_trim_race_neither_side_against_a_running_worker() {
    let arena = ArenaProvider::default();
    let req = WorkspaceReq {
        a_bytes: 1 << 15,
        tile_bytes: 4096,
        b_bytes: 1 << 15,
        team_scatter: 16,
        barriers: 1,
        ..Default::default()
    };
    std::thread::scope(|s| {
        s.spawn(|| {
            for _ in 0..2000 {
                arena.with_worker(&req, &mut |_, _, _| {});
                let mut lease = arena.take_team(&req, 2, 2);
                lease.panel(req.b_bytes);
                lease.scatter_mut().push(1);
            }
        });
        s.spawn(|| {
            for _ in 0..2000 {
                let st = arena.stats();
                assert!(
                    st.leased_bytes <= st.retained_bytes,
                    "leased exceeds retained: {st:?}"
                );
                arena.trim();
            }
        });
    });
    arena.trim();
    assert_eq!(arena.stats(), WorkspaceStats::default());
}

/// A panic while a popped team set is being prepared must not leave its
/// capacity counted: the set is dropped with its pages, so the owner has to
/// stop counting them.
#[test]
fn a_panicking_team_prepare_counts_nothing() {
    let arena = ArenaProvider::default();
    let req = WorkspaceReq {
        team_scatter: 64,
        barriers: 1,
        ..Default::default()
    };
    let lease = arena.take_team(&req, 2, 2);
    drop(lease);
    let warm = arena.stats().retained_bytes;
    let warm_panel = arena.retained_panel_bytes();
    assert!(warm > 0 || warm_panel > 0);

    // `Vec::reserve(usize::MAX)` overflows and panics inside `prepare`, after the
    // idle set has been popped.
    let impossible = WorkspaceReq {
        team_scatter: usize::MAX,
        barriers: 1,
        ..Default::default()
    };
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = arena.take_team(&impossible, 2, 2);
    }));
    assert!(unwound.is_err(), "the oversized request must panic");
    assert_eq!(arena.stats().retained_bytes, 0, "{:?}", arena.stats());
    assert_eq!(arena.retained_panel_bytes(), 0, "panel capacity survived");
    assert_eq!(arena.stats().leased_bytes, 0);
}

/// Re-preparing a set for a different geometry rebuilds the barrier vector, and
/// the accounting follows it down as well as up.
#[test]
fn a_geometry_change_settles_the_barrier_capacity() {
    let arena = ArenaProvider::default();
    let two = WorkspaceReq {
        barriers: 2,
        ..Default::default()
    };
    let one = WorkspaceReq {
        barriers: 1,
        ..Default::default()
    };
    let lease = arena.take_team(&two, 2, 2);
    drop(lease);
    let with_two = arena.stats().retained_bytes;
    assert!(with_two >= 2 * core::mem::size_of::<std::sync::Barrier>());
    // A different `pn` forces `prepare` to rebuild the barrier vector.
    let lease = arena.take_team(&one, 2, 3);
    drop(lease);
    let with_one = arena.stats().retained_bytes;
    assert!(with_one < with_two, "{with_one} is not below {with_two}");
}

/// Nothing is billed through `panel()` alone: a zero-byte request on a set that
/// already holds a panel still reports it, and `trim` releases it.
#[test]
fn a_zero_request_keeps_a_warmed_panel_counted() {
    let arena = ArenaProvider::default();
    let req = WorkspaceReq {
        b_bytes: 1 << 16,
        ..Default::default()
    };
    let mut lease = arena.take_team(&req, 1, 1);
    lease.panel(req.b_bytes);
    drop(lease);
    assert_eq!(arena.retained_panel_bytes(), 1 << 16);
    let lease = arena.take_team(&WorkspaceReq::default(), 1, 1);
    assert!(arena.stats().leased_bytes >= 1 << 16, "{:?}", arena.stats());
    drop(lease);
    assert_eq!(arena.retained_panel_bytes(), 1 << 16);
    arena.trim();
    assert_eq!(arena.retained_panel_bytes(), 0);
    assert_eq!(arena.stats(), WorkspaceStats::default());
}

/// Reserving a B panel after a large A block must not be swallowed by the byte
/// total: the bound is component-wise.
#[test]
fn reserve_keeps_component_wise_maxima() {
    let mut ws = Workspace::with_req(&WorkspaceReq {
        a_bytes: 128,
        ..Default::default()
    });
    assert_eq!(ws.req().a_bytes, 128);
    assert_eq!(ws.req().b_bytes, 0);
    ws.reserve(&WorkspaceReq {
        b_bytes: 64,
        ..Default::default()
    });
    assert_eq!(ws.req().a_bytes, 128, "the A block must be kept");
    assert_eq!(ws.req().b_bytes, 64, "the B panel must be added");
    // A same-total request does not shrink or replace a component.
    ws.reserve(&WorkspaceReq {
        a_bytes: 32,
        b_bytes: 16,
        ..Default::default()
    });
    assert_eq!(ws.req().a_bytes, 128);
    assert_eq!(ws.req().b_bytes, 64);
}

/// `prepare_for` warms one worker slot per worker the executor may use.
#[test]
fn prepare_for_warms_every_worker_slot() {
    let sp = threadteam::Pool::builder().threads(2).build().unwrap();
    let pool = Pool::borrow(&sp);
    let mut exec = Exec::new(&pool, 2).unwrap();
    let req = WorkspaceReq {
        a_bytes: 4096,
        ..Default::default()
    };
    let mut ws = Workspace::with_req(&req);
    assert_eq!(ws.slots(), 1);
    ws.prepare_for(&mut exec);
    assert_eq!(ws.slots(), 2, "both workers must be warmed");
    assert!(
        ws.retained_bytes() >= 2 * req.a_bytes,
        "two slots are retained: {}",
        ws.retained_bytes()
    );
    let before = ws.retained_bytes();
    ws.prepare_for(&mut exec);
    assert_eq!(ws.retained_bytes(), before, "a second prepare is a no-op");
}
