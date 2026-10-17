//! One geometry decision shared by reporting and execution.

use super::{block, dynamic, pack_b_needed, BLOCK_MC_BUDGET};
use crate::plan::{PackedPlan, PackedRoute, NS_PER_FLOP};
use tcontract_exec::{BatchQuery, Exec, ExecError, WorkspaceReq};
use tcontract_kernel::pack::panel_len;
use tcontract_kernel::{Element, PartitionPolicy, ResolvedGemm};

pub(crate) struct Geometry {
    pub lanes: usize,
    pub pm: usize,
    pub pn: usize,
    pub p: usize,
    pub dyn_jobs: Option<(usize, usize, usize)>,
    pub direct_b: bool,
}

impl Geometry {
    pub fn kind(&self) -> PackedRoute {
        if self.lanes > 1 {
            PackedRoute::BatchLanes
        } else if self.p == 1 {
            PackedRoute::Serial
        } else if self.pm == 1 {
            PackedRoute::Cells
        } else {
            PackedRoute::Spmd
        }
    }
    pub fn width(&self) -> usize {
        if self.lanes > 1 {
            self.lanes
        } else {
            self.p
        }
    }
}

/// The batch query one packed call presents to [`Exec::batch_decision`].
///
/// `item_muladds` is one item's multiply-add count, the packed driver's
/// `m * n * k` times four for complex (its kernels run four real GEMMs).
/// `balanced_lanes` asks `Auto` for the driver's own two-stage rule, which is
/// the one place the "a team across the entry" and "a lane per entry" routes
/// are weighed. `item_parallel` is true: with one lane the *team grid* uses the
/// budget, and `packed_auto` reports that narrowed width as `item_threads`.
fn batch_query(plan: &PackedPlan, m: usize, n: usize, k: usize, complex: bool) -> BatchQuery {
    let item_muladds = m as f64 * n as f64 * k as f64 * if complex { 4.0 } else { 1.0 };
    BatchQuery::new(plan.stats.batch, item_muladds)
        .with_balanced_lanes(true)
        .with_item_parallel(true)
}

pub(crate) fn execution_geometry<T: Element>(
    plan: &PackedPlan,
    rg: &ResolvedGemm<T::Real>,
    exec: &Exec<'_>,
) -> Result<Geometry, ExecError> {
    let fam = rg.family();
    let (mr, nr) = (fam.mr, fam.nr);
    let swap = plan.transposes_gemm(mr);
    let (am, ak, bk, bn) = if swap {
        (&plan.b_n, &plan.b_k, &plan.a_k, &plan.a_m)
    } else {
        (&plan.a_m, &plan.a_k, &plan.b_k, &plan.b_n)
    };
    let (m, n, k) = (am.len(), bn.len(), ak.len());
    let direct_b = !pack_b_needed(fam.b_access, bk, bn, nr);
    let budget = exec.budget();
    let explicit_grid = match rg.partition {
        PartitionPolicy::StaticGrid { pm, pn } if pm != 0 => Some((pm, pn)),
        _ => None,
    };
    let dyn_jobs = match rg.partition {
        PartitionPolicy::DynamicTiles { job_m, job_n } => {
            let nc_serial = rg.with_threads(1).map_or(n, |rg| rg.nc);
            Some((
                job_m,
                job_n,
                dynamic::job_count(m, n, nr, nc_serial, job_m, job_n),
            ))
        }
        _ => None,
    };
    // The route is decided once, before any allocation or write. With a pinned
    // grid or dynamic tiles the batch lanes are not asked for and the team
    // width is the packed width rule for the whole batch (`Auto`'s stage one);
    // otherwise [`Exec::batch_decision`] returns both the lanes and, when it
    // keeps one lane, the effective team width. Forced strategies are evaluated
    // against this context's budget, exactly as [`Exec::batch_decision`]
    // documents.
    let (lanes, one_lane_width) = if explicit_grid.is_none() && dyn_jobs.is_none() {
        let decision = exec.batch_decision(&batch_query(plan, m, n, k, T::IS_COMPLEX))?;
        (decision.lanes, decision.item_threads)
    } else {
        let total_ns =
            2.0 * plan.stats.macs() as f64 * if T::IS_COMPLEX { 4.0 } else { 1.0 } * NS_PER_FLOP;
        (
            1,
            exec.width_for(total_ns, &exec.batch_policy().model.width),
        )
    };
    // The grid is unused on the lane route: every lane owns whole entries. The
    // pinned-grid and dynamic-tile routes keep the legacy resolution (the budget
    // the driver was handed, which `Plan::batch_context` narrows for `Auto`).
    let want = if lanes > 1 || explicit_grid.is_some() || dyn_jobs.is_some() {
        budget
    } else {
        one_lane_width.min(budget)
    };
    let (mut pm, mut pn) = match (explicit_grid, dyn_jobs) {
        _ if lanes > 1 => (1, 1),
        (_, Some((_, _, jobs))) => (want.min(jobs), 1),
        (Some(grid), None) => grid,
        (None, None) => plan.partition_with(mr, nr, want),
    };
    if dyn_jobs.is_none() {
        // INVARIANT: resolution validates the grid product; clamp only reduces it.
        while pm * pn > want {
            if pn > 1 {
                pn -= 1;
            } else {
                pm -= 1;
            }
        }
    }
    let p = pm * pn;
    if direct_b && dyn_jobs.is_none() {
        pm = 1;
        pn = p;
    }
    if pm > 1 && exec.is_worker() {
        return Err(ExecError::Unavailable);
    }
    Ok(Geometry {
        lanes,
        pm,
        pn,
        p,
        dyn_jobs,
        direct_b,
    })
}

/// A conservative upper bound on the workspace one packed call needs, over every
/// route the driver can take on `exec` (#121).
///
/// It is computed from the same geometry the driver uses, widened to the worst
/// case: the *serial* blocking (the largest cache share, which the resolved
/// `rg` already carries), the whole budget as the widest grid and lane count,
/// every team scatter a nonzero-`beta` call can ask for, and the blocked
/// traversal's tile grids whenever that shape could apply. A caller that sizes a
/// [`Workspace`](tcontract_exec::Workspace) to this bound and executes within it
/// grows the arena for none of the buffers.
///
/// The bound is deliberately not the exact call requirement: that depends on the
/// runtime geometry the driver resolves at execution. It is an upper bound, which
/// is all the `workspace_req`/`execute_raw_in` contract needs.
pub(crate) fn call_workspace_req<T: Element>(
    plan: &PackedPlan,
    rg: &ResolvedGemm<T::Real>,
    exec: &Exec<'_>,
) -> WorkspaceReq {
    let fam = rg
        .family()
        .driver_family()
        .expect("validated family ABI for this scheme");
    let (mr, nr) = (fam.mr, fam.nr);
    let budget = exec.budget().max(1);
    let (mc, kc, nc) = (rg.mc, rg.kc, rg.nc);
    let swap = plan.transposes_gemm(mr);
    let (am, ak, bk, bn) = if swap {
        (&plan.b_n, &plan.b_k, &plan.a_k, &plan.a_m)
    } else {
        (&plan.a_m, &plan.a_k, &plan.b_k, &plan.b_n)
    };
    let (m, n, k) = (am.len(), bn.len(), ak.len());
    let direct_b = !pack_b_needed(fam.b_access, bk, bn, nr);
    let element = core::mem::size_of::<T::Real>();
    // The packed A block at the serial blocking.
    let a_bytes = panel_len(mc, mr, kc, fam.a_pack) * element;
    // The shared B panel at the widest grid: the largest slice is one column
    // group, and lanes round the slice up to a cache line.
    let group_cap = nc.div_ceil(nr);
    let b_group =
        panel_len(group_cap * nr, nr, kc, fam.b_pack).next_multiple_of((64 / element).max(1));
    let b_bytes = if direct_b {
        0
    } else {
        budget * b_group * element
    };
    // The blocked traversal's two tile grids, when the shape could take it.
    let grid_elems = match block::blocked_eligibility(
        &plan.stats,
        swap,
        k,
        n,
        kc,
        nc,
        (64 / element).max(1),
        BLOCK_MC_BUDGET,
        mr,
    ) {
        None => 0,
        Some(shape) => {
            let mut axes_buf = [block::Role::EMPTY; block::MAX_AXES];
            let axes = block::role_into(&plan.stats, swap, &mut axes_buf);
            let block_rows = axes[shape.span].extent.saturating_mul(shape.line_len);
            tcontract_kernel::tile_planes(fam.tile_fmt)
                .saturating_mul(mr)
                .saturating_mul(nr)
                .saturating_mul(block_rows.div_ceil(mr))
                .saturating_mul(n.div_ceil(nr))
        }
    };
    // Every block-scatter vector, including the nonzero-beta C one, and the
    // widest barrier count.
    let team_scatter = m.div_ceil(mr)
        + n.div_ceil(nr)
        + plan.d_m.len().div_ceil(mr)
        + plan.d_n.len().div_ceil(nr)
        + plan.c_m.len().div_ceil(mr);
    WorkspaceReq {
        a_bytes,
        tile_bytes: (fam.tile + fam.induced_scratch(kc)) * element + 2 * grid_elems * element,
        worker_scatter: 0,
        b_bytes,
        team_scatter,
        barriers: budget,
    }
}
