//! The micro-tile half of the loop nest: packing of one epoch's panels, the two
//! innermost loops with their direct-path branches and first-slab rule, and the
//! write-back call. Shared by the static and the dynamic scheduler so there is one
//! copy of the arithmetic.

use tcontract_kernel::scatter::IRREGULAR;
use tcontract_kernel::{Element, Real, UkrAux, UkrFn};

use super::Ctx;

// Common write-back call for the first-K and accumulation paths. Numerical
// loops stay in the kernel layer; foreign types retain the legacy entry.
unsafe fn emit_tile<T: Element>(
    cx: &Ctx<'_, T>,
    ab: *const T::Real,
    mrem: usize,
    nrem: usize,
    beta: T,
    c: *const T,
    cr: &[i64],
    cc: &[i64],
    crs: i64,
    conj_c: bool,
    d: *mut T,
    dr: &[i64],
    dc: &[i64],
    drs: i64,
    conj_d: bool,
) {
    // SAFETY: run_strip supplies the same checked live tile/scatters and
    // initialized accumulator that the former writeback calls used.
    unsafe {
        (cx.emitter)(
            ab, cx.mr, cx.nr, mrem, nrem, cx.alpha, beta, c, cr, cc, crs, conj_c, d, dr, dc, drs,
            conj_d,
        )
    }
}

/// Everything the micro-tile loops need that is fixed for one
/// `(batch, NC panel, KC slab)` epoch: the batch-offset operand bases, the K
/// slab, the NC block and where the packed `B` sliver for a column lives.
/// Shared by the static and the dynamic scheduler so there is one copy of the
/// arithmetic and of the direct-path branches.
#[derive(Clone, Copy)]
pub(super) struct Epoch<T: Element> {
    pub(super) ah: *mut T,
    pub(super) bh: *mut T,
    pub(super) ch: *const T,
    pub(super) dh: *mut T,
    pub(super) pc: usize,
    pub(super) pc_len: usize,
    pub(super) first_k_block: bool,
    pub(super) jc: usize,
    pub(super) jc_len: usize,
    pub(super) b_sliver: usize,
    /// Packed `B` for sliver `q0` of this NC block (the first of the group).
    pub(super) bp: *mut T::Real,
    pub(super) q0: usize,
}

/// Rows a blocked outer traversal holds in one block: the driver's eligibility shrinks
/// the block to `BLOCK_MC_BUDGET`, and `run_block` keeps its local metadata on the
/// stack because a steady-state execute must not allocate.
const BLOCK_ROWS_MAX: usize = super::BLOCK_MC_BUDGET;

/// One worker's private buffers.
#[derive(Clone, Copy)]
pub(super) struct Bufs<R> {
    pub(super) ap: *mut R,
    pub(super) tile: *mut R,
    pub(super) scratch: *mut R,
    /// When not null, finished tiles are **collected** into this tile grid instead of
    /// being emitted: slot `(ir / mr) * ntiles + (jr / nr)`, each
    /// `tile_planes(fam.tile) * mr * nr` values. The blocked path emits them later,
    /// after permuting whole rows into the output's order - which is why it needs the
    /// tiles to survive past the micro-tile that produced them.
    pub(super) grid: *mut R,
    /// The permute destination, one whole grid as well; `grid` holds the block's tiles
    /// in its own row order and `grid2` the same rows in the output's.
    pub(super) grid2: *mut R,
    /// Column tiles per row tile in `grid`.
    pub(super) ntiles: usize,
}

/// Pack `B` slivers `[w0, w1)` of the epoch's NC block into the shared panel.
///
/// # Safety
/// As [`run_strip`]; `w0..w1` are slivers of the block owned by the caller
/// alone, and the panel slice has room for them.
pub(super) unsafe fn pack_b_slivers<T>(cx: &Ctx<'_, T>, ep: &Epoch<T>, w0: usize, w1: usize)
where
    T: Element,
{
    let (nr, bn, bk, conj_b) = (cx.nr, cx.bn, cx.bk, cx.conj_b);
    let b_n_bs = cx.runs.slice(cx.scatter, cx.runs.b);
    let (pc, pc_len, b_sliver) = (ep.pc, ep.pc_len, ep.b_sliver);
    let c0 = ep.jc + w0 * nr;
    let c1 = (ep.jc + w1 * nr).min(ep.jc + ep.jc_len);
    let out = ep.bp.add((w0 - ep.q0) * b_sliver);
    let (_, pack_b) = cx.packers;
    // SAFETY: the validated scatters and capacity of this epoch.
    unsafe {
        pack_b(
            ep.bh,
            &bn[c0..c1],
            &b_n_bs[c0 / nr..c1.div_ceil(nr)],
            &bk[pc..pc + pc_len],
            nr,
            conj_b,
            out,
        )
    }
}

/// Pack rows `[ic, ic + ic_len)` of the epoch's K slab into `ap`.
///
/// # Safety
/// As [`run_strip`]; `ap` is the caller's private panel with room for `ic_len`
/// rows of the slab.
pub(super) unsafe fn pack_a_rows<T>(
    cx: &Ctx<'_, T>,
    ep: &Epoch<T>,
    a_m: &[i64],
    a_m_bs: &[i64],
    live: usize,
    ap: *mut T::Real,
) where
    T: Element,
{
    let (mr, ak, conj_a) = (cx.mr, cx.ak, cx.conj_a);
    let (pc, pc_len) = (ep.pc, ep.pc_len);
    let (pack_a, _) = cx.packers;
    debug_assert_eq!(a_m.len(), live, "one row scatter entry per live row");
    #[cfg(feature = "phase-timing")]
    let _phase = crate::phase::scope(0);
    // SAFETY: the validated scatters and capacity of this epoch. `a_m` covers the
    // rows this call packs - the epoch's own interval today, a gathered block-local
    // list on the blocked path - and `a_m_bs` is its block-scatter twin.
    unsafe { pack_a(ep.ah, a_m, a_m_bs, &ak[pc..pc + pc_len], mr, conj_a, ap) }
}

/// Loops 2 and 1: every micro-tile of the packed `A` rows `[ic, ic + ic_len)`
/// against the columns `[jr_lo, jr_hi)` of the epoch's NC block, with the
/// write-back. The one copy of the tile arithmetic, the direct-C/direct-B
/// branches and the first-slab/accumulate rule.
///
/// # Safety
/// As [`run_strip`]; `bufs.ap` holds the packed rows, the packed `B` panel (or
/// the in-place operand) covers the columns, and the caller owns the output
/// tiles of the covered block.
#[inline(always)]
#[allow(clippy::too_many_arguments)] // INVARIANT: the block's row views and its geometry.
pub(super) unsafe fn compute_block<T>(
    cx: &Ctx<'_, T>,
    ep: &Epoch<T>,
    bufs: Bufs<T::Real>,
    cm_rows: &[i64],
    dm_rows: &[i64],
    c_m_bs: &[i64],
    d_m_bs: &[i64],
    live: usize,
    jr_lo: usize,
    jr_hi: usize,
) where
    T: Element,
{
    debug_assert_eq!(cm_rows.len(), live, "one C row per live row");
    debug_assert_eq!(dm_rows.len(), live, "one D row per live row");
    let Ctx {
        plan,
        fam,
        mr,
        nr,
        cn,
        dn,
        bn,
        bk,
        alpha,
        beta,
        direct_b,
        ..
    } = *cx;
    let Epoch {
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
        ..
    } = *ep;
    let (ap_ptr, scratch_ptr) = (bufs.ap, bufs.scratch);
    let collect = !bufs.grid.is_null();
    let tile_len = tcontract_kernel::tile_planes(fam.tile_fmt) * mr * nr;
    let runs = cx.runs;
    let scatter = cx.scatter;
    let b_n_bs = runs.slice(scatter, runs.b);
    let d_n_bs = runs.slice(scatter, runs.dn);
    let one = T::one();
    let a_sliver = fam.a_per_k * pc_len;
    let mut jr = jr_lo;
    while jr < jr_hi {
        let nrem = nr.min(jc_len - jr);
        let j0 = jc + jr;
        // Only the packed path offsets the panel: a direct-B call has none, and
        // `add` requires a zero offset on a dangling pointer. The direct-B arms
        // below read `b_base` instead, so the value is never used as a panel.
        let bpan = if direct_b {
            bp_ptr
        } else {
            bp_ptr.add((jr / nr - q0) * b_sliver)
        };
        // B's k steps are one apart and its columns one constant
        // stride apart, both checked by `pack_b_needed`; otherwise
        // the tile reads the packed panel (k stride `NR`, columns
        // adjacent).
        let (b_base, b_rs, b_cs) = if direct_b {
            // The base carries this k block's and this column
            // block's offsets, so the kernel's own k stride is one.
            (
                bh.offset((bn[j0] + bk[pc]) as isize) as *const T::Real,
                1,
                b_n_bs[j0 / nr] as isize,
            )
        } else {
            (bpan as *const T::Real, nr as isize, 1)
        };

        // ---- loop 1: MR -----------------------------------
        let mut ir = 0;
        while ir < live {
            let mrem = mr.min(live - ir);
            let apan = ap_ptr.add((ir / mr) * a_sliver);
            // Collecting: this micro-tile's home is its slot in the block's grid.
            let tile_ptr = if collect {
                bufs.grid
                    .add(((ir / mr) * bufs.ntiles + jr / nr) * tile_len)
            } else {
                bufs.tile
            };
            let d_rs = *d_m_bs.get_unchecked(ir / mr);
            // A Direct family writes D itself only where D's own
            // strides make that expressible: the guard was decided
            // once for the call, and both scatters must be regular.
            // Collecting: the kernel must write the *tile*, never `D` itself, or the
            // emission below would read a grid slot that no kernel touched.
            let direct_tile = !collect
                && matches!(fam.kernel, UkrFn::Direct(_))
                && cx.call.direct_c_allowed
                && d_rs != IRREGULAR
                && *d_n_bs.get_unchecked(j0 / nr) != IRREGULAR;

            #[cfg(feature = "phase-timing")]
            let _phase = crate::phase::scope(2);
            match fam.kernel {
                UkrFn::Tile(_) => {
                    // A partial micro-tile - the last row block or the last
                    // column block of a panel - runs the family's live-extent
                    // entry when it has one, so it does not pay for a whole
                    // register block: same packed panels, same declared tile,
                    // dead slots zeroed by the kernel. The entry is `None` for
                    // the induced 1m/4m wrappers, the native 1m/3m methods and
                    // the interleaved complex kernel, which keep the full-tile
                    // call below; a Direct family is a different arm entirely.
                    match fam.edge {
                        Some(edge) if mrem < mr || nrem < nr => {
                            // SAFETY: the packed panels and the declared
                            // `mr x nr` tile are the ones the family contract
                            // covers, and `mrem`/`nrem` are the live extent of
                            // this tile (`1..=mr`, `1..=nr`).
                            unsafe { edge(pc_len, apan, bpan, tile_ptr, mrem, nrem) }
                        }
                        _ => {
                            // SAFETY: full packed panels and tile per
                            // family contract; an induced family
                            // scales the inner kernel's k itself.
                            unsafe {
                                tcontract_kernel::induced::tile_call(
                                    &fam,
                                    pc_len,
                                    apan,
                                    bpan,
                                    tile_ptr,
                                    scratch_ptr,
                                )
                            }
                        }
                    }
                }
                UkrFn::Direct(func) => {
                    // A direct tile overlaps the accumulator it
                    // scales; a fallback tile is overwritten
                    // (`alpha_d = 0`), and the write-back below
                    // then applies alpha/beta exactly as the
                    // scratch path does.
                    let (d_base, rs_d, cs_d, alpha_d, beta_ab) = if direct_tile {
                        (
                            dh.offset((dm_rows[ir] + dn[j0]) as isize) as *mut T::Real,
                            d_rs as isize,
                            *d_n_bs.get_unchecked(j0 / nr) as isize,
                            if first_k_block {
                                if beta == T::zero() {
                                    T::Real::ZERO
                                } else {
                                    beta.re()
                                }
                            } else {
                                T::Real::ONE
                            },
                            alpha.re(),
                        )
                    } else {
                        (tile_ptr, 1, mr as isize, T::Real::ZERO, T::Real::ONE)
                    };
                    let aux = UkrAux {
                        a_next: if ir + mr < live {
                            apan.add(a_sliver)
                        } else {
                            apan
                        },
                        // In-place B has no panel to point past.
                        b_next: match (direct_b, jr + nr < jc_len) {
                            (true, _) => b_base,
                            (false, true) => bpan.add(b_sliver),
                            (false, false) => bpan,
                        },
                        inner: None,
                        opaque: fam.opaque,
                    };
                    // SAFETY: A is the packed panel with unit row
                    // stride; B and D follow the strides derived
                    // above, and `d_out` is the live tile extent.
                    unsafe {
                        (func)(
                            mrem,
                            nrem,
                            pc_len,
                            d_base,
                            rs_d,
                            cs_d,
                            apan as *const T::Real,
                            fam.a_per_k as isize,
                            b_base,
                            b_rs,
                            b_cs,
                            alpha_d,
                            beta_ab,
                            &aux,
                        )
                    };
                }
            }

            if direct_tile {
                ir += mr;
                continue;
            }
            #[cfg(feature = "phase-timing")]
            let _phase = crate::phase::scope(3);
            // A collected tile is emitted by the blocked path, once, after its rows
            // have been permuted into the output's order.
            if collect {
                ir += mr;
                continue;
            }
            if first_k_block {
                let c_rs = c_m_bs.get(ir / mr).copied().unwrap_or(IRREGULAR);
                emit_tile::<T>(
                    cx,
                    tile_ptr,
                    mrem,
                    nrem,
                    beta,
                    ch,
                    &cm_rows[ir..ir + mrem],
                    &cn[j0..j0 + nrem],
                    c_rs,
                    plan.conj_c,
                    dh,
                    &dm_rows[ir..ir + mrem],
                    &dn[j0..j0 + nrem],
                    d_rs,
                    plan.conj_d,
                );
            } else {
                // Accumulate: C := D, beta := 1, and conjugate
                // the readback exactly when op_D conjugates.
                emit_tile::<T>(
                    cx,
                    tile_ptr,
                    mrem,
                    nrem,
                    one,
                    dh as *const T,
                    &dm_rows[ir..ir + mrem],
                    &dn[j0..j0 + nrem],
                    d_rs,
                    plan.conj_d,
                    dh,
                    &dm_rows[ir..ir + mrem],
                    &dn[j0..j0 + nrem],
                    d_rs,
                    plan.conj_d,
                );
            }
            ir += mr;
        }
        jr += nr;
    }
}

/// One axis-aware block of the blocked outer traversal.
///
/// Gathers the block's rows into local scatters, packs the operand in the block's own
/// row order (the order that keeps its reads contiguous), computes its micro-tiles
/// into the tile grid, permutes whole rows into the output's order, and emits from
/// there - so the stores to `D` stay consecutive even though the block's rows are not.
///
/// # Safety
///
/// As [`run_strip`], plus: `bufs.grid` and `bufs.grid2` each cover a whole block's
/// tile grid (one slot per row tile and column tile), `rows` holds the block's global
/// row indices of this epoch's scatters, and the contraction fits one K slab and one
/// NC panel - which is what makes the single emission below correct.
pub(super) unsafe fn run_block<T>(
    cx: &Ctx<'_, T>,
    ep: &Epoch<T>,
    bufs: Bufs<T::Real>,
    rows: &[i64],
    jr_lo: usize,
    jr_hi: usize,
) where
    T: Element,
{
    let (mr, nr, cm, dm, beta) = (cx.mr, cx.nr, cx.cm, cx.dm, cx.beta);
    let live = rows.len();
    // Everything below is bounded by the block's row budget, so the local metadata is
    // on the stack: a steady-state execute must not allocate
    // (`tests/packed_workspace_alloc`), and the per-block `Vec`s this used to build
    // cost more than the blocked path saved (about 9 ms of a 23 ms call).
    assert!(
        live <= BLOCK_ROWS_MAX,
        "blocked path: {live} rows exceed the {BLOCK_ROWS_MAX}-row scratch"
    );
    let mtiles = live.div_ceil(mr).max(1);
    assert!(
        mtiles * mr <= BLOCK_ROWS_MAX,
        "blocked path: the row map does not fit the scratch"
    );
    let ntiles = bufs.ntiles;
    // The traversal's own per-block work, measured once per call rather than per block:
    // per-block `Instant::now()` calls are expensive enough on this host to swamp what
    // they are measuring.
    let mut a_m_buf = [0i64; BLOCK_ROWS_MAX];
    let mut dm_buf = [0i64; BLOCK_ROWS_MAX];
    let mut cm_buf = [0i64; BLOCK_ROWS_MAX];
    let mut a_bs_buf = [0i64; BLOCK_ROWS_MAX];
    let mut d_bs_buf = [0i64; BLOCK_ROWS_MAX];
    let mut c_bs_buf = [0i64; BLOCK_ROWS_MAX];
    let mut order_buf = [0u32; BLOCK_ROWS_MAX];
    let mut dm_out_buf = [0i64; BLOCK_ROWS_MAX];
    let mut cm_out_buf = [0i64; BLOCK_ROWS_MAX];
    let mut d_bs_out_buf = [0i64; BLOCK_ROWS_MAX];
    let mut c_bs_out_buf = [0i64; BLOCK_ROWS_MAX];
    // The block's own row order: the operand's runs together, the output's not.
    let (a_m, dm_rows) = (&mut a_m_buf[..live], &mut dm_buf[..live]);
    super::block::gather_into(cx.am, rows, a_m);
    super::block::gather_into(dm, rows, dm_rows);
    let has_c = !cm.is_empty();
    let cm_rows = &mut cm_buf[..if has_c { live } else { 0 }];
    if has_c {
        super::block::gather_into(cm, rows, cm_rows);
    }
    let n = tcontract_kernel::scatter::build_block_scatter_into(a_m, mr, &mut a_bs_buf);
    let a_m_bs = &a_bs_buf[..n];
    let n = tcontract_kernel::scatter::build_block_scatter_into(dm_rows, mr, &mut d_bs_buf);
    let d_m_bs = &d_bs_buf[..n];
    let n = if has_c {
        tcontract_kernel::scatter::build_block_scatter_into(cm_rows, mr, &mut c_bs_buf)
    } else {
        0
    };
    let c_m_bs = &c_bs_buf[..n];
    pack_a_rows::<T>(cx, ep, a_m, a_m_bs, live, bufs.ap);
    compute_block::<T>(
        cx, ep, bufs, cm_rows, dm_rows, c_m_bs, d_m_bs, live, jr_lo, jr_hi,
    );

    // Whole rows into the output's order, then emit from the permuted grid. The permutation
    // covers `mtiles * mr` rows because the grid holds a whole number of tiles; the lanes
    // past `live` are never emitted (`mrem` clips them) and only need the map to be in
    // bounds.
    let order = &mut order_buf[..mtiles * mr];
    super::block::output_order_into(&dm_rows[..live], &mut order[..live]);
    order[live..].fill(0);
    let planes = tcontract_kernel::tile_planes(cx.fam.tile_fmt);
    let tile_len = planes * mr * nr;
    // SAFETY: the caller sized both grids for a whole block; `order` is a permutation of
    // the block's rows.
    unsafe {
        tcontract_kernel::pack::permute::permute_grid_rows(
            bufs.grid,
            bufs.grid2,
            mr,
            nr,
            cx.fam.tile_fmt,
            mtiles,
            ntiles,
            order,
        )
    };
    let dm_out = &mut dm_out_buf[..live];
    for (o, &r) in dm_out.iter_mut().zip(&order[..live]) {
        *o = dm_rows[r as usize];
    }
    let n = tcontract_kernel::scatter::build_block_scatter_into(dm_out, mr, &mut d_bs_out_buf);
    let d_m_bs_out = &d_bs_out_buf[..n];
    let cm_out = &mut cm_out_buf[..if has_c { live } else { 0 }];
    if has_c {
        for (o, &r) in cm_out.iter_mut().zip(&order[..live]) {
            *o = cm_rows[r as usize];
        }
    }
    let n = if has_c {
        tcontract_kernel::scatter::build_block_scatter_into(cm_out, mr, &mut c_bs_out_buf)
    } else {
        0
    };
    let c_m_bs_out = &c_bs_out_buf[..n];
    let Epoch {
        ch, dh, jc, jc_len, ..
    } = *ep;
    // The blocked path's write-back: the stores are `emit_tile`'s, and there is no scope on
    // the unblocked path's call site that would cover them.
    #[cfg(feature = "phase-timing")]
    let _wb = crate::phase::scope(3);
    for t in 0..mtiles {
        let mrem = mr.min(live.saturating_sub(t * mr));
        if mrem == 0 {
            break;
        }
        for jt in 0..ntiles {
            let jr = jr_lo + jt * nr;
            if jr >= jr_hi {
                break;
            }
            let nrem = nr.min(jc_len.saturating_sub(jr));
            if nrem == 0 {
                break;
            }
            let j0 = jc + jr;
            let c_rs = c_m_bs_out.get(t).copied().unwrap_or(IRREGULAR);
            // SAFETY: the permuted grid holds this tile, the scatters are the block's own
            // rows, and `D`'s write-back obligations are `emit_tile`'s.
            unsafe {
                emit_tile::<T>(
                    cx,
                    bufs.grid2.add((t * ntiles + jr / nr) * tile_len),
                    mrem,
                    nrem,
                    beta,
                    ch,
                    &cm_out[t * mr..t * mr + mrem],
                    &cx.cn[j0..j0 + nrem],
                    c_rs,
                    cx.plan.conj_c,
                    dh,
                    &dm_out[t * mr..t * mr + mrem],
                    &cx.dn[j0..j0 + nrem],
                    d_m_bs_out[t],
                    cx.plan.conj_d,
                )
            };
        }
    }
}
