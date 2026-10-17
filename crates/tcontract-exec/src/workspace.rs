//! Owning workspace: page-aligned, grow-only, reused per worker.
//!
//! A workspace belongs to exactly one owner — a `tcontract_exec::Pool` or a caller
//! that wants reuse — and is *lent* to the driver for one call at a time. There
//! is deliberately no process-global arena: storage handed to a team is
//! returned to the provider that issued it, and a second owner can never see
//! it.
//!
//! Two payload shapes:
//!
//! * **worker buffers** — the packed A block, the accumulator tile and the
//!   scratch vector of one thread. They are allocated by the thread that will
//!   write them, which is what makes their first touch local to the node that
//!   keeps using them; a thread-local handle names its slot in the owner, and a
//!   handle outlives nothing: the storage is the owner's, the handle is only an
//!   index, and a slot that has been trimmed is simply re-created.
//! * **team sets** — the shared packed B panel, the team's scatter vectors and
//!   its barriers. They are taken exclusively for the duration of a call and
//!   returned on drop, so two concurrent executes can never share one.
//!
//! # Exclusivity
//!
//! A slot that is *in the owner's map* is touched only while the map lock is
//! held. A slot that is *checked out* is not in the map at all: the thread that
//! checked it out is its only owner until it is returned, so a re-entrant call
//! on that thread finds nothing and allocates fresh call-local buffers instead
//! of aliasing the ones its outer call is using (waiting for itself would
//! deadlock). `trim` and the accounting snapshot therefore never observe a slot
//! another thread is writing.
//!
//! `ArenaProvider::traced()` additionally records every growth's thread and
//! size, which is how the tests prove that a worker's buffers were allocated by
//! the worker. Tracing is off by default and only covers buffers the owner
//! keeps; the fresh re-entrant path is not traced.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Barrier, Mutex};

/// The 4096-byte alignment every buffer in the workspace has.
const PAGE: usize = 4096;

/// Bytes and counts one execute needs. Buffer sizes are bytes because the
/// provider is element-type erased: the driver knows the element type and
/// converts its real counts once.
///
/// A zero entry never allocates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkspaceReq {
    /// This thread's packed A block.
    pub a_bytes: usize,
    /// This thread's accumulator tile, including any induced-method scratch.
    pub tile_bytes: usize,
    /// Elements reserved for this thread's scatter scratch.
    pub worker_scatter: usize,
    /// The team's shared packed B panel.
    pub b_bytes: usize,
    /// Elements reserved for the team's block-scatter vectors.
    pub team_scatter: usize,
    /// Barriers the team needs, or zero when nothing is shared.
    pub barriers: usize,
}

impl WorkspaceReq {
    /// The payload this request needs, in bytes: the buffers, the `i64` scatter
    /// elements and the barriers. It is the value a [`Workspace`] is sized to and
    /// the amount [`Workspace::have`] reports.
    ///
    /// # Examples
    ///
    /// ```
    /// use tcontract_exec::WorkspaceReq;
    /// let req = WorkspaceReq { a_bytes: 16, team_scatter: 2, ..Default::default() };
    /// assert_eq!(req.bytes(), 16 + 2 * core::mem::size_of::<i64>());
    /// ```
    pub fn bytes(&self) -> usize {
        self.a_bytes
            + self.tile_bytes
            + self.b_bytes
            + (self.worker_scatter + self.team_scatter) * core::mem::size_of::<i64>()
            + self.barriers * core::mem::size_of::<Barrier>()
    }
}

/// What an owner currently holds and currently lends out.
///
/// Exactness: both values are settled at every checkout, return, team take,
/// lease drop and `trim`, so `ArenaProvider::stats` is exact whenever no
/// execution is in flight. While one is, a checked-out slot is counted at the
/// capacity it had when it was checked out, so `retained_bytes` is an unsettled
/// estimate that can lag growth or shrinkage until the call returns.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkspaceStats {
    /// Capacity the owner holds: worker A/tile/scatter and team
    /// panel/scatter/barriers, idle or in use.
    pub retained_bytes: usize,
    /// The part of `retained_bytes` lent to a call right now.
    pub leased_bytes: usize,
}

/// Page-aligned, grow-only, never zeroed storage.
///
/// The caller writes before reading, which is the contract the packing and
/// kernel code already has; zeroing would only double the traffic.
#[derive(Default, Debug)]
struct PageBuf {
    ptr: *mut u8,
    cap: usize,
    /// Owner key this buffer reports growths under, or zero when untraced.
    traced: u64,
}

// SAFETY: the buffer owns its allocation and hands out pointers only through
// `&mut self` or an explicit, contracted call.
unsafe impl Send for PageBuf {}

impl PageBuf {
    /// A buffer that reports its growths under `owner`, for tests that prove
    /// where storage was first touched. Zero means untraced.
    fn traced(owner: u64) -> Self {
        Self {
            traced: owner,
            ..Self::default()
        }
    }

    /// A pointer to at least `bytes` writable, page-aligned bytes, growing the
    /// allocation if needed and reusing it otherwise. `ensure(0)` returns a
    /// dangling but well-aligned pointer and allocates nothing.
    fn ensure(&mut self, bytes: usize) -> *mut u8 {
        if bytes == 0 {
            return std::ptr::NonNull::<u8>::dangling().as_ptr();
        }
        if bytes > self.cap {
            // Grow by doubling, so a sequence of slightly larger calls does not
            // reallocate every time.
            let want = bytes.next_power_of_two();
            let layout = std::alloc::Layout::from_size_align(want, PAGE).expect("workspace layout");
            // SAFETY: `want` is nonzero, and any old block was allocated with
            // this same alignment and a power-of-two size.
            let new = unsafe {
                match self.ptr.is_null() {
                    true => std::alloc::alloc(layout),
                    false => {
                        let old = std::alloc::Layout::from_size_align_unchecked(self.cap, PAGE);
                        std::alloc::realloc(self.ptr, old, want)
                    }
                }
            };
            assert!(
                !new.is_null(),
                "workspace allocation of {want} bytes failed"
            );
            self.ptr = new;
            self.cap = want;
            if self.traced != 0 {
                record(self.traced, want);
            }
        }
        self.ptr
    }

    /// Pointer to the current allocation, or a dangling pointer when empty.
    fn as_ptr(&self) -> *mut u8 {
        if self.ptr.is_null() {
            std::ptr::NonNull::<u8>::dangling().as_ptr()
        } else {
            self.ptr
        }
    }

    /// Bytes currently allocated.
    fn cap_bytes(&self) -> usize {
        self.cap
    }

    /// Release the allocation. Callers must hold the buffer exclusively.
    fn trim(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: allocated with this alignment and a power-of-two size.
            unsafe {
                std::alloc::dealloc(
                    self.ptr,
                    std::alloc::Layout::from_size_align_unchecked(self.cap, PAGE),
                )
            };
            self.ptr = std::ptr::null_mut();
            self.cap = 0;
        }
    }
}

impl Drop for PageBuf {
    fn drop(&mut self) {
        self.trim();
    }
}

/// One thread's buffers, as the owner keeps them between calls.
#[derive(Debug)]
struct WorkerSlot {
    a: PageBuf,
    tile: PageBuf,
    scratch: Vec<i64>,
}

impl WorkerSlot {
    /// `owner` is the tracing key, or zero for an untraced owner.
    fn new(owner: u64) -> Self {
        Self {
            a: PageBuf::traced(owner),
            tile: PageBuf::traced(owner),
            scratch: Vec::new(),
        }
    }

    fn bytes(&self) -> usize {
        self.a.cap_bytes()
            + self.tile.cap_bytes()
            + self.scratch.capacity() * core::mem::size_of::<i64>()
    }

    fn trim(&mut self) {
        self.a.trim();
        self.tile.trim();
        self.scratch = Vec::new();
    }
}

/// A team's shared storage: one packed B panel, the scatter vectors and the
/// barriers that publish the panel between its threads.
#[derive(Debug, Default)]
struct TeamSet {
    /// The packed B panel.
    b: PageBuf,
    /// Block-scatter vectors for this call, capacity-reused.
    scatter: Vec<i64>,
    /// One barrier per column group when `pm > 1`, else empty.
    barriers: Vec<Barrier>,
    pm: usize,
    pn: usize,
}

impl TeamSet {
    fn prepare(&mut self, req: &WorkspaceReq, pm: usize, pn: usize) {
        self.scatter.clear();
        self.scatter.reserve(req.team_scatter);
        if (self.pm, self.pn) != (pm, pn) {
            // A barrier cannot be reset, so a differently shaped team needs new
            // ones; that is why the free list keeps sets and rebuilds here.
            self.barriers = (0..req.barriers).map(|_| Barrier::new(pm)).collect();
            self.pm = pm;
            self.pn = pn;
        }
    }

    fn bytes(&self) -> usize {
        self.b.cap_bytes()
            + self.scatter.capacity() * core::mem::size_of::<i64>()
            + self.barriers.capacity() * core::mem::size_of::<Barrier>()
    }
}

/// An exclusive loan of a team set, returned to its owner on drop.
///
/// The set is deliberately opaque: the panel grows only through
/// [`TeamLease::panel`], so every byte the owner hands out is accounted.
#[derive(Debug)]
pub struct TeamLease<'a> {
    owner: &'a ArenaProvider,
    set: Option<TeamSet>,
    /// Bytes this lease has added to the owner's leased total.
    billed: usize,
}

impl<'a> TeamLease<'a> {
    /// Size the shared panel and return its pointer, accounting the growth to
    /// the owner. The driver sizes it because only the driver knows the element
    /// type; the provider keeps the storage.
    pub fn panel(&mut self, bytes: usize) -> *mut u8 {
        let set = self.set.as_mut().expect("team lease holds its set");
        let before = set.b.cap_bytes();
        let ptr = set.b.ensure(bytes);
        let after = set.b.cap_bytes();
        if after != before {
            let delta = after as isize - before as isize;
            let mut st = lock(&self.owner.state);
            st.owned = add_signed(st.owned, delta, "retained bytes");
            st.leased = add_signed(st.leased, delta, "leased bytes");
            st.panel = add_signed(st.panel, delta, "panel bytes");
            self.billed = add_signed(self.billed, delta, "billed bytes");
        }
        ptr
    }

    /// The team's block-scatter buffer, sized by the driver each call.
    pub fn scatter_mut(&mut self) -> &mut Vec<i64> {
        &mut self.set.as_mut().expect("team lease holds its set").scatter
    }

    /// The team's block-scatter buffer.
    pub fn scatter(&self) -> &[i64] {
        &self.set.as_ref().expect("team lease holds its set").scatter
    }

    /// One barrier per column group, empty when nothing is shared.
    pub fn barriers(&self) -> &[Barrier] {
        &self
            .set
            .as_ref()
            .expect("team lease holds its set")
            .barriers
    }
}

impl Drop for TeamLease<'_> {
    fn drop(&mut self) {
        if let Some(set) = self.set.take() {
            let mut st = lock(&self.owner.state);
            // The scatter vector may have been resized through `scatter_mut`
            // without going through `panel`, so settle both directions.
            let delta = set.bytes() as isize - self.billed as isize;
            st.owned = add_signed(st.owned, delta, "retained bytes");
            st.leased = sub(st.leased, self.billed, "leased bytes");
            st.teams.push(set);
        }
    }
}

mod sealed {
    /// Only the crate's own owner implements [`WorkspaceProvider`](super::WorkspaceProvider):
    /// the driver writes through the raw pointers it hands out, so an
    /// implementation must honour the arena's buffer and exclusivity contracts.
    pub trait Sealed {}
    impl Sealed for super::ArenaProvider {}
}

/// Storage an execution may borrow. Implemented by the owner; the driver only
/// ever sees this trait, so nothing here depends on the driver.
///
/// Sealed: [`ArenaProvider`] is the supported way to lend reusable storage, and
/// the trait is deliberately not implementable outside this crate.
pub trait WorkspaceProvider: sealed::Sealed + Sync {
    /// Run `f` with this thread's A block, tile and scatter scratch. A
    /// re-entrant call on the same thread gets fresh buffers instead of the
    /// ones its outer call is using.
    fn with_worker(&self, req: &WorkspaceReq, f: &mut dyn FnMut(*mut u8, *mut u8, &mut Vec<i64>));

    /// Take the team set for `pm x pn`, exclusively, until the lease drops.
    fn take_team(&self, req: &WorkspaceReq, pm: usize, pn: usize) -> TeamLease<'_>;

    /// Release idle storage. Live leases and checked-out worker slots are kept.
    fn trim(&self);
}

/// What one owner holds and lends. One lock covers the storage maps and the
/// counters, so a snapshot and an update share one synchronization boundary.
#[derive(Debug, Default)]
struct State {
    workers: HashMap<u64, WorkerSlot>,
    teams: Vec<TeamSet>,
    /// Capacity of everything owned, idle or in use.
    owned: usize,
    /// The part of `owned` lent to a call right now.
    leased: usize,
    /// Capacity of the team panels alone, idle or leased. Independent of which
    /// workers happened to take part, so a test can compare two runs of one
    /// shape.
    panel: usize,
}

/// A workspace owner: per-thread worker slots plus a free list of team sets.
///
/// One per owner — never process-global — so a lease always returns to the
/// provider that issued it and no second owner can borrow it.
#[derive(Debug)]
pub struct ArenaProvider {
    /// Process-unique and never recycled, so a thread-local handle can never be
    /// confused with another provider that happens to share an address.
    key: u64,
    state: Mutex<State>,
    next_worker: AtomicU64,
    traced: u64,
}

impl Default for ArenaProvider {
    fn default() -> Self {
        Self::new()
    }
}

thread_local! {
    /// `(provider key, worker slot)` pairs this thread has used. Thread-local
    /// by construction, so a slot is never handed to a different thread after
    /// its owner exits: the handle dies with the thread.
    static HANDLES: std::cell::RefCell<Vec<(u64, u64)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

static NEXT_PROVIDER: AtomicU64 = AtomicU64::new(1);
static TRACE: Mutex<Vec<(u64, std::thread::ThreadId, usize)>> = Mutex::new(Vec::new());

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// `value + delta`, rejecting a decrement below zero. Every caller establishes
/// the bound; this turns a mistake into a loud failure instead of a wrap.
fn add_signed(value: usize, delta: isize, what: &str) -> usize {
    value
        .checked_add_signed(delta)
        .unwrap_or_else(|| panic!("workspace {what} went out of range"))
}

/// `value - amount`, rejecting an underflow for the same reason.
fn sub(value: usize, amount: usize, what: &str) -> usize {
    value
        .checked_sub(amount)
        .unwrap_or_else(|| panic!("workspace {what} went out of range"))
}

fn record(owner: u64, bytes: usize) {
    lock(&TRACE).push((owner, std::thread::current().id(), bytes));
}

impl ArenaProvider {
    /// A fresh owner with no storage.
    pub fn new() -> Self {
        Self {
            key: NEXT_PROVIDER.fetch_add(1, Ordering::Relaxed),
            state: Mutex::new(State::default()),
            next_worker: AtomicU64::new(0),
            traced: 0,
        }
    }

    /// A fresh owner that records where its storage is first touched, readable
    /// through [`ArenaProvider::trace_take`].
    pub fn traced() -> Self {
        let mut arena = Self::new();
        arena.traced = arena.key;
        arena
    }

    /// Take this owner's recorded buffer growths, leaving its log empty.
    ///
    /// Empty unless the owner was built with [`ArenaProvider::traced`].
    pub fn trace_take(&self) -> Vec<(std::thread::ThreadId, usize)> {
        let mut mine = Vec::new();
        lock(&TRACE).retain(|(owner, thread, bytes)| {
            if *owner == self.key {
                mine.push((*thread, *bytes));
                false
            } else {
                true
            }
        });
        mine
    }

    /// Bytes currently retained and leased, for accounting by the host.
    pub fn stats(&self) -> WorkspaceStats {
        let st = lock(&self.state);
        WorkspaceStats {
            retained_bytes: st.owned,
            leased_bytes: st.leased,
        }
    }

    /// Bytes currently retained, for accounting in tests and diagnostics.
    pub fn retained_bytes(&self) -> usize {
        self.stats().retained_bytes
    }

    /// Panel bytes retained, idle or leased. Unlike
    /// [`ArenaProvider::retained_bytes`] this does not depend on which workers
    /// happened to take part, so it is what a test can compare between two runs
    /// of one shape.
    pub fn retained_panel_bytes(&self) -> usize {
        lock(&self.state).panel
    }

    /// Check this thread's slot out of the map, or `None` when the thread
    /// already holds it (a re-entrant call).
    fn checkout(&self) -> Option<(u64, WorkerSlot, usize)> {
        let known = HANDLES.with(|h| {
            h.borrow()
                .iter()
                .find(|(k, _)| *k == self.key)
                .map(|(_, i)| *i)
        });
        match known {
            // The handle exists; a missing slot means an outer call holds it.
            Some(index) => {
                let mut st = lock(&self.state);
                let slot = st.workers.remove(&index)?;
                let billed = slot.bytes();
                st.leased += billed;
                Some((index, slot, billed))
            }
            None => {
                let index = self.next_worker.fetch_add(1, Ordering::Relaxed);
                HANDLES.with(|h| h.borrow_mut().push((self.key, index)));
                // A fresh slot owns nothing yet, so the counters do not move.
                Some((index, WorkerSlot::new(self.traced), 0))
            }
        }
    }

    /// Return a checked-out slot, settling both accounting directions.
    fn give_back(&self, index: u64, slot: WorkerSlot, billed: usize) {
        let delta = slot.bytes() as isize - billed as isize;
        let mut st = lock(&self.state);
        // `owned` follows the slot's capacity; `leased` gives back exactly what
        // checkout added, since growth was never lent to anyone else.
        st.owned = add_signed(st.owned, delta, "retained bytes");
        st.leased = sub(st.leased, billed, "leased bytes");
        st.workers.insert(index, slot);
    }
}

/// Returns a checked-out slot even when the callback unwinds.
struct SlotGuard<'a> {
    owner: &'a ArenaProvider,
    index: u64,
    slot: Option<WorkerSlot>,
    billed: usize,
}

impl Drop for SlotGuard<'_> {
    fn drop(&mut self) {
        if let Some(slot) = self.slot.take() {
            self.owner.give_back(self.index, slot, self.billed);
        }
    }
}

impl WorkspaceProvider for ArenaProvider {
    fn with_worker(&self, req: &WorkspaceReq, f: &mut dyn FnMut(*mut u8, *mut u8, &mut Vec<i64>)) {
        let Some((index, slot, billed)) = self.checkout() else {
            return fresh_worker(req, f);
        };
        // Armed before any growth, so a preparation unwind returns the slot too.
        let mut guard = SlotGuard {
            owner: self,
            index,
            slot: Some(slot),
            billed,
        };
        let slot = guard.slot.as_mut().expect("guard holds its slot");
        slot.a.ensure(req.a_bytes);
        slot.tile.ensure(req.tile_bytes);
        slot.scratch.clear();
        slot.scratch.reserve(req.worker_scatter);
        let (a, tile) = (slot.a.as_ptr(), slot.tile.as_ptr());
        f(a, tile, &mut slot.scratch);
    }

    fn take_team(&self, req: &WorkspaceReq, pm: usize, pn: usize) -> TeamLease<'_> {
        let (mut set, before, before_panel) = {
            let mut st = lock(&self.state);
            let set = st.teams.pop().unwrap_or_default();
            let (bytes, panel) = (set.bytes(), set.b.cap_bytes());
            (set, bytes, panel)
        };
        // `prepare` allocates, so it can panic. A panic drops `set` and its
        // pages, and the owner has to stop counting the capacity it had when it
        // was popped; the guard arms before `prepare` for that reason.
        let mut teardown = Teardown {
            owner: self,
            before,
            before_panel,
            armed: true,
        };
        set.prepare(req, pm, pn);
        let after = set.bytes();
        teardown.armed = false;
        let mut st = lock(&self.state);
        st.owned = add_signed(st.owned, after as isize - before as isize, "retained bytes");
        // `prepare` touches the scatter vector and the barriers, never the panel.
        st.leased += after;
        drop(st);
        TeamLease {
            owner: self,
            set: Some(set),
            billed: after,
        }
    }

    fn trim(&self) {
        let mut st = lock(&self.state);
        let idle_slots: usize = st.workers.values().map(|s| s.bytes()).sum();
        let idle_teams: usize = st.teams.iter().map(|t| t.bytes()).sum();
        let idle_panels: usize = st.teams.iter().map(|t| t.b.cap_bytes()).sum();
        for slot in st.workers.values_mut() {
            slot.trim();
        }
        st.teams.clear();
        st.owned = sub(st.owned, idle_slots + idle_teams, "retained bytes");
        st.panel = sub(st.panel, idle_panels, "panel bytes");
    }
}

/// Returns a popped team set's accounting when `prepare` unwinds.
struct Teardown<'a> {
    owner: &'a ArenaProvider,
    before: usize,
    before_panel: usize,
    armed: bool,
}

impl Drop for Teardown<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let mut st = lock(&self.owner.state);
        st.owned = sub(st.owned, self.before, "retained bytes");
        st.panel = sub(st.panel, self.before_panel, "panel bytes");
    }
}

/// Buffers for one call that either re-entered its owner or has none. They are
/// freed when the call returns, which is the price of re-entrancy, and they are
/// not traced or accounted: the owner never holds them.
fn fresh_worker(req: &WorkspaceReq, f: &mut dyn FnMut(*mut u8, *mut u8, &mut Vec<i64>)) {
    let mut a = PageBuf::default();
    let mut tile = PageBuf::default();
    let mut scratch = Vec::new();
    a.ensure(req.a_bytes);
    tile.ensure(req.tile_bytes);
    scratch.reserve(req.worker_scatter);
    f(a.as_ptr(), tile.as_ptr(), &mut scratch);
}

/// A caller-owned, bounded workspace: an [`ArenaProvider`] pre-sized to one
/// [`WorkspaceReq`], so a steady-state execute whose requirement is within the
/// bound allocates nothing.
///
/// A plan computes a conservative requirement (`Plan::workspace_req` in the
/// owning crate) and `execute_raw_in` checks it against [`Workspace::have`]
/// before any write. The bound is grow-only: [`Workspace::reserve`] enlarges it
/// for a larger later requirement.
///
/// # Examples
///
/// ```
/// use tcontract_exec::{Workspace, WorkspaceReq};
/// let mut ws = Workspace::with_req(&WorkspaceReq { a_bytes: 64, ..Default::default() });
/// assert!(ws.have() >= 64);
/// ws.reserve(&WorkspaceReq { b_bytes: 128, ..Default::default() });
/// assert!(ws.have() >= 128);
/// let _provider = ws.as_provider();
/// ```
#[derive(Debug, Default)]
pub struct Workspace {
    arena: ArenaProvider,
    req: WorkspaceReq,
}

impl Workspace {
    /// A workspace sized and warmed for `req`: the worker slot of the calling
    /// thread and one team set are grown to hold it, so a later execute within
    /// the bound does not grow the arena.
    pub fn with_req(req: &WorkspaceReq) -> Self {
        let arena = ArenaProvider::new();
        Self::warm(&arena, req);
        Self { arena, req: *req }
    }

    /// Grow the bound to cover `req` as well, warming the arena for it. The bound
    /// never shrinks.
    pub fn reserve(&mut self, req: &WorkspaceReq) {
        if req.bytes() > self.req.bytes() {
            Self::warm(&self.arena, req);
            self.req = *req;
        }
    }

    /// The bound this workspace was sized to, in bytes.
    pub fn have(&self) -> usize {
        self.req.bytes()
    }

    /// The requirement this workspace was sized to.
    pub fn req(&self) -> &WorkspaceReq {
        &self.req
    }

    /// The provider a plan lends to the driver for one call.
    pub fn as_provider(&self) -> &dyn WorkspaceProvider {
        &self.arena
    }

    /// First-touch the worker buffers (this thread's slot) and one team set so
    /// the first execute within the bound allocates only the arena's bookkeeping.
    fn warm(arena: &ArenaProvider, req: &WorkspaceReq) {
        arena.with_worker(req, &mut |_, _, _| {});
        let mut lease = arena.take_team(req, 1, 1);
        if req.b_bytes > 0 {
            lease.panel(req.b_bytes);
        }
        drop(lease);
    }
}
