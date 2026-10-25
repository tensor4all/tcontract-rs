//! `PartitionPolicy::DynamicTiles`: dynamic assignment of output work
//! (tcontract addition, issue #29; spec §6.1 of the switchable-engine design).
//!
//! The team traverses one **epoch** per `(batch, NC panel, KC slab)` in
//! lockstep. Per epoch:
//!
//! 1. worker 0 resets the claim counter, and every worker packs a disjoint range
//!    of `NR` slivers of the shared `B` panel (none for a direct-B call);
//! 2. a **publication** barrier: `B` is complete and read-only, the reset is
//!    visible (`fetch_add(Relaxed)` only makes claims unique; the barrier
//!    publishes);
//! 3. workers claim jobs until none are left — row bands when there are at
//!    least as many bands as workers, `(band, column subtile)` jobs otherwise —
//!    pack their own `A` and run the shared tile arithmetic of
//!    [`compute_block`](super::compute_block);
//! 4. a **completion** barrier taken by every worker, whether or not it claimed
//!    anything, so the next epoch may overwrite `B` and reset the counter.
//!
//! K is never split and no accumulation is atomic: a tile's K slabs run in the
//! original order (a later slab starts only after the previous epoch's
//! completion barrier), so the result is bitwise identical to the static and
//! serial runs for a fixed blocking. A width-1 call never enters this module.
//!
//! Claiming is not work stealing: there are no deques, only one monotonic
//! counter per epoch. Its claims are bounded by `jobs + width` (every worker
//! stops at its first claim past the end), and `Plan` resolution proves that
//! sum fits in `usize` before any broadcast, so the counter cannot wrap.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering::Relaxed};
use std::sync::Barrier;

use super::{compute_block, pack_a_rows, pack_b_slivers, Bufs, Ctx, Epoch};
use tcontract_kernel::Element;

/// How an NC epoch's output jobs are assigned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Assignment {
    /// A worker claims a whole row band and reuses each packed `A` chunk across
    /// the band's columns.
    RowBands,
    /// Workers claim `(row band, column subtile)` jobs in row-major order; `A`
    /// is packed once per job, so it is packed up to `column jobs` times.
    Tiles2D,
}

/// Decide the mode of an epoch from its validated geometry alone, so every
/// worker (and the report) reaches the same answer.
pub(crate) fn assignment(bands: usize, workers: usize) -> Assignment {
    if bands >= workers {
        Assignment::RowBands
    } else {
        Assignment::Tiles2D
    }
}

/// Jobs at the widest NC block, which caps the useful team width.
pub(crate) fn job_count(
    m: usize,
    n: usize,
    nr: usize,
    nc_serial: usize,
    job_m: usize,
    job_n: usize,
) -> usize {
    let widest = nc_serial.min(n.next_multiple_of(nr)).min(n);
    m.div_ceil(job_m)
        .saturating_mul(widest.div_ceil(job_n))
        .max(1)
}

/// What a dynamic plan resolved to, for reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DynamicReport {
    /// Job rows (logical).
    pub job_m: usize,
    /// Job columns (logical).
    pub job_n: usize,
    /// Workers that will run: the width budget capped by the available jobs.
    /// One means the call runs serially on the caller with no claims.
    pub active_width: usize,
    /// Row bands in the output (oriented rows over `job_m`).
    pub row_bands: usize,
    /// Assignment mode of a full-width NC epoch; tail epochs with fewer column
    /// jobs may differ only in being narrower, never in the band rule.
    pub assignment: Assignment,
}

/// The dynamic assignment `plan` would use at host width `width` with `rg`, or
/// none when the resolution is not `DynamicTiles`.
pub(crate) fn dynamic_report<R: tcontract_kernel::Real>(
    plan: &crate::plan::PackedPlan,
    rg: &tcontract_kernel::ResolvedGemm<R>,
    width: usize,
) -> Option<DynamicReport> {
    let tcontract_kernel::PartitionPolicy::DynamicTiles { job_m, job_n } = rg.partition else {
        return None;
    };
    let (m, n) = if plan.transposes_gemm(rg.mr) {
        (plan.b_n.len(), plan.a_m.len())
    } else {
        (plan.a_m.len(), plan.b_n.len())
    };
    let nc_serial = rg.with_threads(1).map_or(n, |rg| rg.nc);
    let jobs = job_count(m, n, rg.nr, nc_serial, job_m, job_n);
    let active_width = width.max(1).min(jobs);
    let row_bands = m.div_ceil(job_m);
    Some(DynamicReport {
        job_m,
        job_n,
        active_width,
        row_bands,
        assignment: assignment(row_bands, active_width),
    })
}

/// Opt-in counters for tests and benchmarks. Pass to
/// `execute_resolved_instrumented`; ordinary execution carries none and pays
/// nothing.
#[derive(Debug)]
pub struct DynStats {
    calls: AtomicUsize,
    team_setups: AtomicUsize,
    width: AtomicUsize,
    b_bytes: AtomicUsize,
    epochs: AtomicUsize,
    band_epochs: AtomicUsize,
    tile_epochs: AtomicUsize,
    claims: AtomicUsize,
    a_elems: AtomicUsize,
    b_slivers: AtomicUsize,
    jobs: Vec<AtomicUsize>,
    visits: Vec<AtomicU32>,
    tiles_n: usize,
}

/// A copy of [`DynStats`] counters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DynSnapshot {
    /// Dynamic executions that ran with more than one worker.
    pub calls: usize,
    /// Executions that leased and filled the barrier-bearing team set.
    pub team_setups: usize,
    /// Width of the most recent such execution.
    pub width: usize,
    /// Bytes of shared `B` panel the most recent execution requested.
    pub b_bytes_requested: usize,
    /// Epochs executed (counted once, by worker 0).
    pub epochs: usize,
    /// Epochs assigned as row bands.
    pub band_epochs: usize,
    /// Epochs assigned as 2-D tiles.
    pub tile_epochs: usize,
    /// Successful job claims across all workers.
    pub claims: usize,
    /// `A` elements (rows x K) packed, summed over workers.
    pub a_elems_packed: usize,
    /// `B` slivers packed, summed over workers.
    pub b_slivers_packed: usize,
    /// Jobs completed per worker index.
    pub jobs_per_worker: Vec<usize>,
}

impl DynStats {
    /// Counters for up to `workers` workers.
    pub fn new(workers: usize) -> Self {
        Self::with_coverage(workers, 0, 0)
    }

    /// As [`new`](Self::new), also counting how often each `MR x NR` output
    /// tile (oriented, `tiles_m x tiles_n`) is computed. In an execution with
    /// `batch` batches and `kslabs` K slabs every tile is computed exactly
    /// `batch * kslabs` times if and only if coverage is exactly-once.
    pub fn with_coverage(workers: usize, tiles_m: usize, tiles_n: usize) -> Self {
        let z = || AtomicUsize::new(0);
        Self {
            calls: z(),
            team_setups: z(),
            width: z(),
            b_bytes: z(),
            epochs: z(),
            band_epochs: z(),
            tile_epochs: z(),
            claims: z(),
            a_elems: z(),
            b_slivers: z(),
            jobs: (0..workers).map(|_| z()).collect(),
            visits: (0..tiles_m * tiles_n).map(|_| AtomicU32::new(0)).collect(),
            tiles_n,
        }
    }

    /// Current counter values.
    pub fn snapshot(&self) -> DynSnapshot {
        DynSnapshot {
            calls: self.calls.load(Relaxed),
            team_setups: self.team_setups.load(Relaxed),
            width: self.width.load(Relaxed),
            b_bytes_requested: self.b_bytes.load(Relaxed),
            epochs: self.epochs.load(Relaxed),
            band_epochs: self.band_epochs.load(Relaxed),
            tile_epochs: self.tile_epochs.load(Relaxed),
            claims: self.claims.load(Relaxed),
            a_elems_packed: self.a_elems.load(Relaxed),
            b_slivers_packed: self.b_slivers.load(Relaxed),
            jobs_per_worker: self.jobs.iter().map(|j| j.load(Relaxed)).collect(),
        }
    }

    /// Per-tile visit counts, row-major `tiles_m x tiles_n`.
    pub fn tile_visits(&self) -> Vec<u32> {
        self.visits.iter().map(|v| v.load(Relaxed)).collect()
    }

    /// Zero every counter (not the coverage shape).
    pub fn reset(&self) {
        for c in [
            &self.calls,
            &self.team_setups,
            &self.width,
            &self.b_bytes,
            &self.epochs,
            &self.band_epochs,
            &self.tile_epochs,
            &self.claims,
            &self.a_elems,
            &self.b_slivers,
        ] {
            c.store(0, Relaxed);
        }
        self.jobs.iter().for_each(|j| j.store(0, Relaxed));
        self.visits.iter().for_each(|v| v.store(0, Relaxed));
    }

    pub(super) fn note_team_setup(&self) {
        self.team_setups.fetch_add(1, Relaxed);
    }

    pub(super) fn note_call(&self, width: usize, b_bytes: usize) {
        if width > 1 {
            self.calls.fetch_add(1, Relaxed);
        }
        self.width.store(width, Relaxed);
        self.b_bytes.store(b_bytes, Relaxed);
    }

    fn visit(&self, mr: usize, nr: usize, ic: usize, ic_len: usize, col0: usize, col1: usize) {
        if self.visits.is_empty() {
            return;
        }
        for i in (ic..ic + ic_len).step_by(mr) {
            for j in (col0..col1).step_by(nr) {
                self.visits[(i / mr) * self.tiles_n + j / nr].fetch_add(1, Relaxed);
            }
        }
    }
}

/// One worker's traversal of every epoch.
///
/// Every worker runs the same `(batch, NC, KC)` loops and takes both barriers
/// of every epoch, so barrier counts match without anyone counting; what a
/// worker *claims* inside an epoch never changes how many epochs it sees.
///
/// # Safety
/// As `execute_packed`: operand pointers valid for the plan's
/// offsets. `bufs` is this worker's alone; `bar` has exactly `p` participants
/// and `claim` is shared by exactly the `p` workers of this invocation; `p >=
/// 2`; `job_m`/`job_n` are positive multiples of `MR`/`NR`.
#[allow(clippy::too_many_arguments)] // INVARIANT: the loop-nest context.
pub(super) unsafe fn run_dynamic<T>(
    cx: &Ctx<'_, T>,
    t: usize,
    p: usize,
    job_m: usize,
    job_n: usize,
    claim: &AtomicUsize,
    bar: &Barrier,
    bufs: Bufs<T::Real>,
    stats: Option<&DynStats>,
) where
    T: Element,
{
    let (plan, fam, mr, nr, mc, kc, nc, m, n, k, direct_b) = (
        cx.plan,
        cx.fam,
        cx.mr,
        cx.nr,
        cx.mc,
        cx.kc,
        cx.nc,
        cx.m,
        cx.n,
        cx.k,
        cx.direct_b,
    );
    let (ptr_a, ptr_b, c, d) = (cx.a.0, cx.b.0, cx.c.0 as *const T, cx.d.0);
    let bands = m.div_ceil(job_m);

    for h in 0..plan.stats.batch {
        let ah = ptr_a.offset(cx.ha[h] as isize);
        let bh = ptr_b.offset(cx.hb[h] as isize);
        let ch = c.offset(cx.hc[h] as isize);
        let dh = d.offset(plan.h_d[h] as isize);
        let mut jc = 0;
        while jc < n {
            let jc_len = nc.min(n - jc);
            let mut pc = 0;
            while pc < k {
                let pc_len = kc.min(k - pc);
                let ep = Epoch::<T> {
                    ah,
                    bh,
                    ch,
                    dh,
                    pc,
                    pc_len,
                    first_k_block: pc == 0,
                    jc,
                    jc_len,
                    b_sliver: fam.b_per_k * pc_len,
                    // One group: sliver `s` of the block lives at `s`.
                    bp: cx.bp.0,
                    q0: 0,
                };
                let nsliv = jc_len.div_ceil(nr);
                // Every worker finished the previous epoch (its completion
                // barrier), so the counter and the panel are free to reuse.
                if t == 0 {
                    claim.store(0, Relaxed);
                    if let Some(st) = stats {
                        st.epochs.fetch_add(1, Relaxed);
                    }
                }
                if !direct_b {
                    // Packing is split among all workers, independent of who
                    // later claims output.
                    let (w0, w1) = (t * nsliv / p, (t + 1) * nsliv / p);
                    if w1 > w0 {
                        // SAFETY: slivers `w0..w1` are this worker's alone.
                        #[cfg(feature = "phase-timing")]
                        let _phase = crate::phase::scope(1);
                        unsafe { pack_b_slivers::<T>(cx, &ep, w0, w1) };
                        if let Some(st) = stats {
                            st.b_slivers.fetch_add(w1 - w0, Relaxed);
                        }
                    }
                }
                // Publication: B complete and read-only, counter reset visible.
                bar.wait();

                let mode = assignment(bands, p);
                if t == 0 {
                    if let Some(st) = stats {
                        match mode {
                            Assignment::RowBands => st.band_epochs.fetch_add(1, Relaxed),
                            Assignment::Tiles2D => st.tile_epochs.fetch_add(1, Relaxed),
                        };
                    }
                }
                let col_jobs = jc_len.div_ceil(job_n);
                let total = match mode {
                    Assignment::RowBands => bands,
                    Assignment::Tiles2D => bands * col_jobs,
                };
                loop {
                    let id = claim.fetch_add(1, Relaxed);
                    if id >= total {
                        break;
                    }
                    let (band, col) = match mode {
                        Assignment::RowBands => (id, None),
                        Assignment::Tiles2D => (id / col_jobs, Some(id % col_jobs)),
                    };
                    let (jr_lo, jr_hi) = match col {
                        None => (0, jc_len),
                        Some(cj) => (cj * job_n, ((cj + 1) * job_n).min(jc_len)),
                    };
                    let (r0, r1) = (band * job_m, ((band + 1) * job_m).min(m));
                    let mut ic = r0;
                    while ic < r1 {
                        let ic_len = mc.min(r1 - ic);
                        // SAFETY: this worker owns the claimed job's output
                        // tiles and its private `A`/tile buffers; the shared
                        // `B` panel was published by the barrier above.
                        // Today's rows are one interval of the epoch's scatters; the
                        // blocked path passes the same thing, gathered over its block.
                        let (a_m_bs, c_m_bs, d_m_bs) = (
                            cx.runs.slice(cx.scatter, cx.runs.a),
                            cx.runs.slice(cx.scatter, cx.runs.cm),
                            cx.runs.slice(cx.scatter, cx.runs.dm),
                        );
                        let (b0, b1) = (ic / cx.mr, (ic + ic_len).div_ceil(cx.mr));
                        let (a_m_bs, d_m_bs) = (&a_m_bs[b0..b1], &d_m_bs[b0..b1]);
                        // `C`'s block scatter is deliberately empty when `beta == 0`
                        // (`driver/mod.rs`), and the write-back reads it with `.get()`,
                        // so empty is the signal rather than a range to cut.
                        let c_m_bs = if c_m_bs.is_empty() {
                            c_m_bs
                        } else {
                            &c_m_bs[b0..b1]
                        };
                        unsafe {
                            pack_a_rows::<T>(
                                cx,
                                &ep,
                                &cx.am[ic..ic + ic_len],
                                a_m_bs,
                                ic_len,
                                bufs.ap,
                            );
                            compute_block::<T>(
                                cx,
                                &ep,
                                bufs,
                                &cx.cm[ic..ic + ic_len],
                                &cx.dm[ic..ic + ic_len],
                                c_m_bs,
                                d_m_bs,
                                ic_len,
                                jr_lo,
                                jr_hi,
                            );
                        }
                        if let Some(st) = stats {
                            st.a_elems.fetch_add(ic_len * pc_len, Relaxed);
                            st.visit(mr, nr, ic, ic_len, jc + jr_lo, jc + jr_hi);
                        }
                        ic += mc;
                    }
                    if let Some(st) = stats {
                        st.claims.fetch_add(1, Relaxed);
                        if let Some(j) = st.jobs.get(t) {
                            j.fetch_add(1, Relaxed);
                        }
                    }
                }
                // Completion: nobody still reads this epoch's B or claims from
                // its counter.
                bar.wait();
                pc += kc;
            }
            jc += nc;
        }
    }
}
