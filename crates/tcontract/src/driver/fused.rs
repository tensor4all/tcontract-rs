//! The fused grouped-GEMM route's half of the driver (tensor4all/tcontract-rs#208).
//!
//! [`crate::FusedGemmPlan`] owns one forced-packed, serial-width [`Plan`] per
//! distinct job geometry and one work list over every job. This module holds
//! what the driver needs for that: a [`FusedGroup`] with the group's serial
//! call shape and its block scatters, built once at prepare, and
//! [`run_item`], which runs one work item `(job, row band, column panel)` on
//! the calling worker through the static driver's own walk
//! ([`run_strip`](super::static_grid::run_strip)) restricted to the item's
//! rows and `NC` panel.
//!
//! Every item covers whole `MC` bands and whole `NC` panels of one job, and
//! runs all of its `KC` slabs in order on one worker with the group's serial
//! blocking. So each output element takes the same operations in the same
//! order as the job run alone by the packed driver on one thread: the result is
//! bitwise that run's, for every width. Items share no packed panel, so no
//! barrier is taken; each worker packs its own `A` and `B`.
//!
//! [`Plan`]: crate::Plan

use tcontract_kernel::pack::panel_len;
use tcontract_kernel::scatter::append_block_scatter;
use tcontract_kernel::{Element, ResolvedGemm};

use super::static_grid::{run_strip, BPart};
use super::{
    c_for_call, call_shape, direct_c_allowed, pack_b_needed, CallShape, Ctx, Oriented,
    ResolvedCall, ScatterRuns, Shared,
};
use crate::plan::PackedPlan;

/// One group's serial call geometry and block scatters, prepared once.
///
/// Built from the group's [`PackedPlan`] and its resolution at serial width,
/// with the same rule the driver applies to a one-thread call (`call_shape`
/// at `(pm, pn) = (1, 1)`), so an item walks exactly the loops that call walks.
#[derive(Debug)]
pub(crate) struct FusedGroup {
    swap: bool,
    direct_b: bool,
    /// The oriented problem: rows of the kernel's row operand, columns, depth.
    m: usize,
    n: usize,
    k: usize,
    mc: usize,
    kc: usize,
    nc: usize,
    blocked: Option<super::block::BlockShape>,
    grid_off: usize,
    grid_elems: usize,
    /// Rows and columns one work item covers: `(mc, nc)`, or the whole job
    /// when the blocked outer traversal applies (it walks every block of the
    /// role at once, and its eligibility already means one K slab and one NC
    /// panel).
    item_m: usize,
    item_n: usize,
    /// Reals of the packed `A` block, rounded up to a 64-byte line so the `B`
    /// panel after it in the same worker region starts on a line.
    ap_reals: usize,
    /// Reals of one column group's slice of the packed `B` panel, sized as
    /// `execute_once` sizes it, also for a direct-B group: `run_strip` checks
    /// every epoch's slivers against it, whether or not a panel is written.
    b_group: usize,
    /// Reals of one worker's packed `B` panel: `b_group`, or zero for a
    /// direct-B group, which reads `B` in place and allocates no panel.
    bp_reals: usize,
    /// Reals of one worker's tile region: tile, induced scratch, tile grids.
    tile_reals: usize,
    /// The five block scatters, laid out by `runs`; `C`'s is last.
    scatter: Vec<i64>,
    runs: ScatterRuns,
}

impl FusedGroup {
    /// The group's geometry for `plan` and its serial resolution `rg`.
    ///
    /// `plan` must have one batch entry (a 2-D GEMM) and a non-empty output
    /// and contraction; [`crate::FusedGemmPlan`] builds groups only for such
    /// jobs.
    pub(crate) fn new<T: Element>(plan: &PackedPlan, rg: &ResolvedGemm<T::Real>) -> Self {
        // INVARIANT: `Plan::new` validated the family ABI for this dtype, as it
        // does for the packed driver (`execute_once` makes the same call).
        let fam = rg
            .family()
            .driver_family()
            .expect("validated family ABI for this scheme");
        let (mr, nr) = (fam.mr, fam.nr);
        let swap = plan.transposes_gemm(mr);
        let o = Oriented::new(plan, swap);
        let (m, n, k) = (o.am.len(), o.bn.len(), o.ak.len());
        let direct_b = !pack_b_needed(fam.b_access, o.bk, o.bn, nr);
        // One participant: the resolution is already the serial one (the plan
        // retargets to width 1 when it is built), as `execute_once` uses at p = 1.
        let CallShape {
            mc,
            kc,
            nc,
            blocked,
            grid_elems,
            grid_off,
            kd,
            ap_len,
        } = call_shape::<T>(plan, &fam, rg, swap, (m, n, k), (1, 1));
        let (item_m, item_n) = match blocked {
            Some(_) => (m, n),
            None => (mc, nc),
        };
        let line = (64 / core::mem::size_of::<T::Real>()).max(1);
        // One column group at `pn = 1`: `execute_once`'s `b_group`, computed
        // whether or not the call reads `B` in place.
        let b_group = panel_len(nc.div_ceil(nr) * nr, nr, kd, fam.b_pack);
        let bp_reals = match direct_b {
            true => 0,
            false => b_group,
        };
        let mut scatter = Vec::new();
        let runs = ScatterRuns {
            a: append_block_scatter(&mut scatter, o.am, mr),
            b: append_block_scatter(&mut scatter, o.bn, nr),
            dm: append_block_scatter(&mut scatter, o.dm, mr),
            dn: append_block_scatter(&mut scatter, o.dn, nr),
            cm: append_block_scatter(&mut scatter, o.cm, mr),
        };
        Self {
            swap,
            direct_b,
            m,
            n,
            k,
            mc,
            kc,
            nc,
            blocked,
            grid_off,
            grid_elems,
            item_m,
            item_n,
            ap_reals: ap_len.next_multiple_of(line),
            b_group,
            bp_reals,
            tile_reals: fam.tile + fam.induced_scratch(kc) + 2 * grid_elems,
            scatter,
            runs,
        }
    }

    /// Row bands and column panels of one job: the item grid.
    pub(crate) fn grid(&self) -> (usize, usize) {
        (self.m.div_ceil(self.item_m), self.n.div_ceil(self.item_n))
    }

    /// Multiply-adds of item `(band, panel)`: its rows times columns times `k`.
    pub(crate) fn item_muladds(&self, band: usize, panel: usize) -> u128 {
        let rows = self.item_m.min(self.m - band * self.item_m);
        let cols = self.item_n.min(self.n - panel * self.item_n);
        rows as u128 * cols as u128 * self.k as u128
    }

    /// Reals of one worker's `(A + B region, tile region)`.
    pub(crate) fn worker_reals(&self) -> (usize, usize) {
        (self.ap_reals + self.bp_reals, self.tile_reals)
    }
}

/// Run item `(band, panel)` of one job of group `g`.
///
/// `a`, `b` and `c` are the job's origins (base pointer plus the job's
/// element offset); `c` is read (when `beta != 0`) and written in place, as
/// `C ← alpha·op(A)·op(B) + beta·C`. `ab` is the worker's `A + B` region and
/// `tile` its tile region, sized by [`FusedGroup::worker_reals`].
///
/// # Safety
///
/// * `a`, `b` and `c` address every offset `plan`'s scatters generate (reads
///   for `a`/`b`, reads and writes for `c`); `c` does not alias `a` or `b`.
/// * No other thread writes this item's output elements while it runs.
/// * `rg` is `plan`'s validated serial resolution and `g` was built from both.
/// * `ab` and `tile` are this worker's alone and at least as large as
///   `g.worker_reals()` asks.
#[allow(clippy::too_many_arguments)] // INVARIANT: the GEMM argument set plus the item and buffers.
pub(crate) unsafe fn run_item<T: Element>(
    plan: &PackedPlan,
    rg: &ResolvedGemm<T::Real>,
    g: &FusedGroup,
    alpha: T,
    beta: T,
    a: *const T,
    b: *const T,
    c: *mut T,
    (band, panel): (usize, usize),
    ab: *mut T::Real,
    tile: *mut T::Real,
) {
    // INVARIANT: as in `FusedGroup::new`.
    let fam = rg
        .family()
        .driver_family()
        .expect("validated family ABI for this scheme");
    let (mr, nr) = (fam.mr, fam.nr);
    let o = Oriented::new(plan, g.swap);
    let (ptr_a, ptr_b) = if g.swap { (b, a) } else { (a, b) };
    let (cp, hc) = c_for_call(plan, beta, c.cast_const(), c);
    // `C`'s block scatter is empty when `beta == 0`, which is the signal the
    // write-back reads (as the driver builds it per call).
    let runs = match beta == T::zero() {
        true => ScatterRuns {
            cm: (g.runs.cm.0, g.runs.cm.0),
            ..g.runs
        },
        false => g.runs,
    };
    let bp = match g.direct_b {
        // A direct-B group reads B in place and has no panel.
        true => core::ptr::NonNull::<T::Real>::dangling().as_ptr(),
        // SAFETY: the region holds `ap_reals + bp_reals` reals.
        false => unsafe { ab.add(g.ap_reals) },
    };
    let cx = Ctx::<T> {
        plan,
        fam,
        packers: rg.packers::<T>(),
        emitter: rg.emitter::<T>(),
        call: ResolvedCall {
            pack_b_needed: !g.direct_b,
            direct_c_allowed: direct_c_allowed(plan, cp, c, beta),
        },
        direct_b: g.direct_b,
        mr,
        nr,
        mc: g.mc,
        kc: g.kc,
        nc: g.nc,
        m: g.m,
        n: g.n,
        k: g.k,
        b_group: g.b_group,
        am: o.am,
        ak: o.ak,
        bk: o.bk,
        bn: o.bn,
        cm: o.cm,
        cn: o.cn,
        dm: o.dm,
        dn: o.dn,
        ha: o.ha,
        hb: o.hb,
        hc,
        scatter: &g.scatter,
        runs,
        blocked: g.blocked,
        grid_off: g.grid_off,
        grid_elems: g.grid_elems,
        conj_a: o.conj_a,
        conj_b: o.conj_b,
        alpha,
        beta,
        a: Shared(ptr_a as *mut T),
        b: Shared(ptr_b as *mut T),
        c: Shared(cp as *mut T),
        d: Shared(c),
        bp: Shared(bp),
    };
    // INVARIANT: `item_m` is `mc` (a multiple of `mr`) or the whole `m`, and
    // `item_n` is `nc` or the whole `n`, so the item starts on the band and
    // panel boundaries the serial walk uses and its micro-tiles are that walk's.
    let m_lo = band * g.item_m;
    let jc_lo = panel * g.item_n;
    let (m_hi, jc_hi) = ((m_lo + g.item_m).min(g.m), (jc_lo + g.item_n).min(g.n));
    // SAFETY: the caller's contract covers every access; `BPart::SERIAL` is one
    // cell with no barrier, and the B panel slot 0 of this worker's region is
    // its own. One batch entry: the group's problem is a 2-D GEMM.
    unsafe {
        run_strip::<T>(
            &cx,
            0..1,
            m_lo,
            m_hi,
            jc_lo,
            jc_hi,
            ab,
            tile,
            tile.add(fam.tile),
            BPart::SERIAL,
        )
    }
}

/// One work item of the fused list: rows `band` and `NC` panel `panel` of job
/// `job`, in group `group`'s item grid ([`FusedGroup::grid`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FusedItem {
    pub(crate) job: usize,
    pub(crate) group: usize,
    pub(crate) band: usize,
    pub(crate) panel: usize,
}

/// One group as the traversal sees it: its plan, resolution and geometry.
pub(crate) type GroupRef<'a, T> = (
    &'a PackedPlan,
    &'a ResolvedGemm<<T as Element>::Real>,
    &'a FusedGroup,
);

/// Run every item of `items` on `width` barrier-free workers that claim them
/// from one counter, in list order; `width == 1` runs them on the calling
/// thread with no claim.
///
/// Each worker takes its `A + B` region and tile region (`worker` reals each)
/// once, from `exec`'s workspace when it has one and from call-local panels
/// otherwise, and keeps them for every item it claims: nothing is allocated
/// per item. Late workers find fewer items left, so the call never waits for
/// co-scheduling and spawns no thread (`fork` runs on the token's own workers).
///
/// # Safety
///
/// * Every item's job addresses only elements of `a`, `b` and `c` through its
///   group's scatters from `job.{a,b,c}_offset` (the batch's `check_lens`), and
///   `c` does not alias `a` or `b`.
/// * The items' output elements are pairwise disjoint: jobs' outputs are
///   disjoint (`GemmBatch::new`) and one job's items partition its output.
/// * `group(i)` returns a group built from that plan and resolution, whose
///   `worker_reals()` is at most `worker`, for every `item.group` in `items`.
/// * `width <= exec.budget()`.
#[allow(clippy::too_many_arguments)] // INVARIANT: the GEMM argument set plus the work list.
pub(crate) unsafe fn traverse<'g, T, G>(
    exec: &mut tcontract_exec::Exec<'_>,
    width: usize,
    items: &[FusedItem],
    jobs: &[t4a_desc::gemm::GemmJob],
    group: G,
    worker: (usize, usize),
    alpha: T,
    beta: T,
    a: *const T,
    b: *const T,
    c: *mut T,
) where
    T: Element,
    G: Fn(usize) -> GroupRef<'g, T> + Sync,
{
    let element = core::mem::size_of::<T::Real>();
    let req = tcontract_exec::WorkspaceReq {
        a_bytes: worker.0 * element,
        tile_bytes: worker.1 * element,
        ..Default::default()
    };
    let workspace = exec.workspace();
    let (a, b, c) = (Shared(a as *mut T), Shared(b as *mut T), Shared(c));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let claim_all = |ab: *mut T::Real, tile: *mut T::Real| {
        // Bind the wrappers whole: a closure that named `a.0` would capture
        // the bare pointer, which is not `Send`.
        let (a, b, c) = (a, b, c);
        // INVARIANT: claims stop at the first index past the end, so the counter
        // ends at most `items.len() + width`; both are bounded by the batch's
        // validated footprints (each item owns at least one output element of a
        // C buffer that fits in `isize`), so it cannot wrap.
        loop {
            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let Some(item) = items.get(i) else { break };
            let job = &jobs[item.job];
            let (plan, rg, g) = group(item.group);
            // SAFETY: the caller's contract. The offsets are inside the
            // buffers (`check_lens`), and the claim gives this item to this
            // worker alone.
            unsafe {
                run_item::<T>(
                    plan,
                    rg,
                    g,
                    alpha,
                    beta,
                    a.0.add(job.a_offset),
                    b.0.add(job.b_offset),
                    c.0.add(job.c_offset),
                    (item.band, item.panel),
                    ab,
                    tile,
                )
            };
        }
    };
    let with_buffers = || match workspace {
        Some(ws) => ws.with_worker(&req, &mut |ab, tile, _| {
            claim_all(ab as *mut T::Real, tile as *mut T::Real)
        }),
        None => {
            let mut ab = crate::buffer::Panel::<T::Real>::new(worker.0);
            let mut tile = crate::buffer::Panel::<T::Real>::new(worker.1);
            claim_all(ab.as_mut_ptr(), tile.as_mut_ptr());
        }
    };
    match width {
        0 | 1 => with_buffers(),
        _ => exec.token().fork(width, |_, _| with_buffers()),
    }
}
