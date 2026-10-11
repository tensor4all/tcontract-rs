//! One geometry decision shared by reporting and execution.

use super::{batch, dynamic, pack_b_needed};
use crate::plan::{PackedPlan, PackedRoute};
use tcontract_exec::{Exec, ExecError};
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
    let want = exec.budget();
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
    let lanes = match (explicit_grid, dyn_jobs) {
        (None, None) => batch::lanes(exec, plan.stats.batch, (m, n, k), T::IS_COMPLEX),
        _ => 1,
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
