//! The static team: one thread's walk over its row strip of the five-loop nest,
//! and its place in the `pm x pn` partition (which slivers of the shared `B`
//! panel it packs and computes). The dynamic scheduler replaces this walk and
//! reuses [`super::tile`].

use std::sync::Barrier;

use tcontract_kernel::Element;

use super::block;
use super::tile::{compute_block, pack_a_rows, pack_b_slivers, run_block, Bufs, Epoch};
use super::Ctx;

/// Where one thread sits in the `pm x pn` partition, as far as loop 5 and the
/// shared packed-`B` panel are concerned.
///
/// Each `jc` block of `nsliv` slivers is cut into `pn` contiguous groups; this
/// thread computes over group `g`'s slivers and packs the fraction `r` of `pm`
/// of them. Both ranges come from [`BPart::ranges`], which is the only place the
/// arithmetic lives, so "which slivers do I own" and "which slivers do I pack"
/// cannot drift apart.
///
/// [`BPart::SERIAL`] is the degenerate `1 x 1` case and reproduces the
/// pre-threading driver exactly: one thread, all the slivers, no barrier.
#[derive(Clone, Copy)]
pub(super) struct BPart<'a> {
    /// This thread's column group, in `0..pn`.
    pub(super) g: usize,
    /// Which slice of the shared `B` panel buffer this thread packs into: its
    /// column group in a team, its lane under batch-axis claiming (see
    /// [`super::batch`]), where every lane owns one slice and nothing is shared.
    pub(super) panel: usize,
    pub(super) pn: usize,
    /// This thread's row strip, in `0..pm` — its index *within* the column
    /// group, which is what decides its share of the group's packing.
    pub(super) r: usize,
    pub(super) pm: usize,
    /// Barrier shared by the `pm` threads of this column group, and by nobody
    /// else: they are the only threads that touch the group's slice of the
    /// panel. `None` when the group has one thread, which then packs and reads
    /// only what it wrote and needs no synchronisation at all.
    pub(super) bar: Option<&'a Barrier>,
}

impl BPart<'_> {
    /// The serial partition: one cell covering everything, no barrier.
    pub(super) const SERIAL: BPart<'static> = BPart {
        g: 0,
        panel: 0,
        pn: 1,
        r: 0,
        pm: 1,
        bar: None,
    };

    /// `(compute, pack)` sliver ranges out of a `jc` block's `nsliv` slivers:
    /// this thread's whole column group, and its share of packing that group.
    ///
    /// Both are half-open and both partition exactly — the `pn` groups tile
    /// `0..nsliv` and the `pm` packing shares tile their group — which is what
    /// makes every output element owned once and every sliver packed once. A
    /// group can come out empty when a tail `jc` block has fewer slivers than
    /// there are groups; that thread then does no work for the block, but still
    /// takes its barriers.
    #[inline]
    pub(super) fn ranges(&self, nsliv: usize) -> ((usize, usize), (usize, usize)) {
        let q0 = self.g * nsliv / self.pn;
        let q1 = (self.g + 1) * nsliv / self.pn;
        let span = q1 - q0;
        let w0 = q0 + self.r * span / self.pm;
        let w1 = q0 + (self.r + 1) * span / self.pm;
        ((q0, q1), (w0, w1))
    }
}

/// Loops 5 through 1 over one cell of the `pm x pn` partition.
///
/// `[m_lo, m_hi)` are whole `MR` panels; `bpart` selects whole `NR` slivers
/// within each `jc` block.
///
/// **Loops 5 and 4 are traversed identically by every thread** — the same `h`,
/// the same `jc`, the same `pc`, over the whole of `N` and `K` — and the
/// barriers live between them and loop 3. That is what makes the barrier counts
/// match without anyone counting: a thread's cell affects only *how much work it
/// does inside* an iteration, never how many iterations there are. Nothing here
/// may make a barrier conditional on `m_lo`, `m_hi` or `bpart`; a thread with an
/// empty cell in some `jc` block still takes that block's barriers and then does
/// nothing, which is why the skip below sits after them and not before.
///
/// `items` is the range of in-plan batch entries to run: all of them for a team
/// or a serial call, one lane's contiguous share under batch-axis claiming. The
/// team's barrier-count argument above holds per call, so a lane (which has no
/// barrier) is free to take any range.
///
/// # Safety
/// As [`execute`], plus: `ap` and `tile` must be this thread's alone, and no
/// other thread may own an overlapping cell.
pub(super) unsafe fn run_strip<T>(
    cx: &Ctx<'_, T>,
    items: std::ops::Range<usize>,
    m_lo: usize,
    m_hi: usize,
    ap_ptr: *mut T::Real,
    tile_ptr: *mut T::Real,
    scratch_ptr: *mut T::Real,
    bpart: BPart<'_>,
) where
    T: Element,
{
    let Ctx {
        plan,
        fam,
        nr,
        mc,
        kc,
        nc,
        n,
        k,
        b_group,
        ha,
        hb,
        direct_b,
        ..
    } = *cx;
    let (ptr_a, ptr_b, c, d, bp_ptr) = (cx.a.0, cx.b.0, cx.c.0 as *const T, cx.d.0, cx.bp.0);

    for h in items {
        let ah = ptr_a.offset(ha[h] as isize);
        let bh = ptr_b.offset(hb[h] as isize);
        let ch = c.offset(cx.hc[h] as isize);
        let dh = d.offset(plan.h_d[h] as isize);

        // ---- loop 5: N blocking -------------------------------------------
        let mut jc = 0;
        while jc < n {
            let jc_len = nc.min(n - jc);

            // ---- loop 4: K blocking ---------------------------------------
            let mut pc = 0;
            while pc < k {
                let pc_len = kc.min(k - pc);
                let first_k_block = pc == 0;
                let b_sliver = fam.b_per_k * pc_len;

                // Pack the shared `B` panel. `(q0, q1)` are the slivers this
                // thread will compute over — its column group — and `(w0, w1)`
                // its share of packing them. The two barriers say "nobody is
                // still reading the previous panel" and "the new one is
                // complete", and they bracket only the group's own slice
                // because only the group's own threads touch it. Serially both
                // ranges are all the slivers and there are no barriers, which
                // is exactly the call the pre-threading driver made.
                let nsliv = jc_len.div_ceil(nr);
                let ((q0, q1), (w0, w1)) = bpart.ranges(nsliv);
                // This column group's private slice of the panel. Sliver `s` of
                // the group lives at `(s - q0)` within it, not at `s`: see
                // `execute` for why the panel is cut per group.
                debug_assert!(
                    (q1 - q0) * b_sliver <= b_group,
                    "column group overruns its slice of the packed B panel"
                );
                // A direct-B call reads `B` where it lies and has no panel at
                // all, so its pointer must stay unoffset: `add` requires a zero
                // offset on a dangling pointer even when nothing is read.
                let bp_ptr = if direct_b {
                    bp_ptr
                } else {
                    bp_ptr.add(bpart.panel * b_group)
                };
                let ep = Epoch::<T> {
                    ah,
                    bh,
                    ch,
                    dh,
                    pc,
                    pc_len,
                    first_k_block,
                    jc,
                    jc_len,
                    b_sliver,
                    bp: bp_ptr,
                    q0,
                };
                let bufs = Bufs {
                    ap: ap_ptr,
                    tile: tile_ptr,
                    scratch: scratch_ptr,
                    // Null unless the blocked path applies; then the block's tiles live
                    // in the first grid and the permuted copy in the second.
                    grid: super::grid_base(tile_ptr, cx.grid_off, cx.grid_elems, false),
                    grid2: super::grid_base(tile_ptr, cx.grid_off, cx.grid_elems, true),
                    ntiles: (n.div_ceil(nr)).max(1),
                };
                // A direct-B call reads B where it lies: no panel is written,
                // so neither barrier is taken. The decision is per call, so
                // every thread of a group skips the same pair.
                if !direct_b {
                    if let Some(bar) = bpart.bar {
                        bar.wait();
                    }
                    if w1 > w0 {
                        #[cfg(feature = "phase-timing")]
                        let _phase = crate::phase::scope(1);
                        // SAFETY: slivers `w0..w1` are this thread's alone.
                        unsafe { pack_b_slivers::<T>(cx, &ep, w0, w1) };
                    }
                    if let Some(bar) = bpart.bar {
                        bar.wait();
                    }
                }

                // This thread's slice of loop 2, in columns of the `jc` block.
                let (jr_lo, jr_hi) = (q0 * nr, (q1 * nr).min(jc_len));

                // ---- loop 3: M blocking -----------------------------------
                // Skipped wholesale when this thread's column group is empty in
                // this `jc` block — possible only in a tail block with fewer
                // slivers than groups — since there is no point packing an `A`
                // block no micro-kernel call will read. Both barriers above have
                // already been taken, which is what keeps their counts equal.
                if let Some(shape) = cx.blocked {
                    // ---- the blocked path: one pass over the role's blocks --------
                    // Eligibility already guarantees one K slab and one NC panel, so a
                    // block's accumulator is complete here and is emitted once, after
                    // its rows are permuted into the output's order.
                    if jr_lo < jr_hi {
                        // The role, the block's rows and the block origins are all walked in
                        // fixed-size scratch: a steady-state execute must not allocate, and
                        // materializing the origins as a `Vec<Vec<usize>>` allocated once per
                        // block.
                        let mut axes_buf = [block::Role::EMPTY; block::MAX_AXES];
                        let axes = block::role_into(
                            &cx.plan.stats,
                            cx.plan.transposes_gemm(cx.mr),
                            &mut axes_buf,
                        );
                        let mut origins = block::Origins::new();
                        let mut rows_buf = [0i64; super::BLOCK_MC_BUDGET];
                        while let Some(origin) = origins.next(axes, shape) {
                            let live = block::rows_into(axes, shape, origin, &mut rows_buf);
                            run_block::<T>(cx, &ep, bufs, &rows_buf[..live], jr_lo, jr_hi);
                        }
                    }
                } else {
                    let mut ic = if jr_lo < jr_hi { m_lo } else { m_hi };
                    while ic < m_hi {
                        let ic_len = mc.min(m_hi - ic);
                        // Today's rows are one interval of the epoch's scatters; the blocked path
                        // passes the same thing, gathered over its block.
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
                        pack_a_rows::<T>(cx, &ep, &cx.am[ic..ic + ic_len], a_m_bs, ic_len, ap_ptr);
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
                        ic += mc;
                    }
                }
                pc += kc;
            }
            jc += nc;
        }
    }
}
