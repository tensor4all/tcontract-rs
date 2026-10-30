//! Planning-time family resolution for the packed driver.
//!
//! A plan resolves its kernel family, blocking and partition once, when it is
//! built; execution never selects again. No workspace or executor lives here.

use tcontract_kernel::{
    Blocking, Families, KernelChoice, KernelHandle, Method, ResolvedGemm, SelectError,
};

use crate::plan::{PackedPlan, PlanConfig, RowBlock};

/// The complex scheme the default menu uses when the config names none.
const DEFAULT_METHOD: Method = Method::Native;

/// The default family menu: the Lukas Devos built-in kernel families of the
/// preferred ISA and the requested complex scheme, default first. Its `(MR, NR)`
/// shapes are what [`PackedPlan::row_block`] chooses among.
///
/// The list is cached per element type and `(ISA, complex method)` inside
/// `tcontract-kernel`: it contains only the immutable built-ins, so registration
/// stays live and a provider is reached by explicit id. This is why the menu
/// cannot change under a running process.
fn default_menu<T: Families>(
    cfg: &PlanConfig,
) -> &'static [&'static tcontract_kernel::KernelFamily<T::Real>] {
    let method = cfg.method.unwrap_or(DEFAULT_METHOD);
    let isa = tcontract_kernel::menu_isa(cfg.isa);
    // Real types ignore the complex method, so one cache entry serves them.
    tcontract_kernel::builtin_menu::<T>(isa, T::IS_COMPLEX.then_some(method))
}

/// The same menu in its compiled order. An explicit row-block request
/// (`base`, `mr=`, `idx=`) names a compiled entry, so it must not see the
/// tuning class's permutation of the menu (issue #156).
fn default_compiled_menu<T: Families>(
    cfg: &PlanConfig,
) -> &'static [&'static tcontract_kernel::KernelFamily<T::Real>] {
    let method = cfg.method.unwrap_or(DEFAULT_METHOD);
    let isa = tcontract_kernel::menu_isa(cfg.isa);
    tcontract_kernel::builtin_compiled_menu::<T>(isa, T::IS_COMPLEX.then_some(method))
}

/// Which menu a row-block request reads: `Auto` picks among the tuning class's
/// ordered menu, while every explicit request addresses the compiled menu by
/// position or `MR`. The class order therefore changes only Auto's default and
/// its tie-break, never what `idx=0`/`base`/`mr=` mean.
fn menu_for_row_block<'a, M>(mode: RowBlock, ordered: &'a [M], compiled: &'a [M]) -> &'a [M] {
    match mode {
        RowBlock::Auto => ordered,
        RowBlock::Base | RowBlock::Pin(_) | RowBlock::Index(_) => compiled,
    }
}

/// Resolve the family, blocking and partition of `p` for storage type `T`, at
/// the serial width (the largest cache share); the driver retargets the
/// blocking to the active grid width without reselecting.
///
/// `handle` is a family a caller's selector chose: trusted, never looked up.
pub(crate) fn resolve<T: Families>(
    p: &PackedPlan,
    cfg: &PlanConfig,
    handle: Option<&KernelHandle<T>>,
) -> Result<ResolvedGemm<T::Real>, SelectError> {
    let tuning = cfg.tuning();
    if let Some(handle) = handle {
        let rg = ResolvedGemm::<T::Real>::resolve_handle::<T>(
            handle,
            1,
            Default::default(),
            Default::default(),
        )?
        .with_tuning(&tuning)?;
        check_family::<T>(p, cfg, &rg)?;
        return apply_partition::<T>(p, cfg, rg);
    }
    let auto = matches!(cfg.kernel, KernelChoice::Auto);
    let chosen;
    let choice = if auto {
        let ordered = default_menu::<T>(cfg);
        let compiled = default_compiled_menu::<T>(cfg);
        let menu = menu_for_row_block(p.row_block_mode, ordered, compiled);
        let shapes: Vec<(usize, usize)> = menu.iter().map(|f| (f.mr, f.nr)).collect();
        let shape = p.row_block(&shapes).map(|i| shapes[i]);
        let family = menu
            .iter()
            .find(|f| shape.is_none_or(|s| (f.mr, f.nr) == s))
            .or_else(|| menu.first())
            .ok_or(SelectError::Incompatible {
                id: "Auto".into(),
                reason: "the default family menu is unavailable for this ISA and complex scheme",
            })?;
        chosen = KernelChoice::Id(family.id.into());
        &chosen
    } else {
        &cfg.kernel
    };
    let mut rg = ResolvedGemm::<T::Real>::resolve::<T>(choice, 1)?.with_tuning(&tuning)?;
    check_family::<T>(p, cfg, &rg)?;
    if auto && tuning.block_model == tcontract_kernel::blocking::BlockModel::Legacy {
        // The default menu's legacy blocking: derived from the packed
        // footprint, overrides applied before register rounding.
        let f = rg.family();
        let (a_reals, b_reals) = (f.a_per_k / f.mr, f.b_per_k / f.nr);
        let real_bytes = core::mem::size_of::<T::Real>();
        let mut blk = match tuning.kc_couple {
            Some(kc) => Blocking::derive_at_depth(real_bytes, a_reals, b_reals, kc),
            None => Blocking::derive(real_bytes, a_reals, b_reals),
        };
        if tuning.blocking.is_set() {
            blk = tuning
                .blocking
                .apply_checked(blk)
                .ok_or(SelectError::Incompatible {
                    id: f.id.into(),
                    reason: "blocking arithmetic overflow",
                })?;
        }
        rg = rg.with_blocking(blk)?;
    }
    apply_partition::<T>(p, cfg, rg)
}

/// Apply the config's partition request to a resolution: validate it against
/// the family, and for `DynamicTiles` prove the job counts and the claim
/// counter's bound fit in `usize` for this shape (in either orientation), so
/// nothing can overflow once execution starts.
fn apply_partition<T: Families>(
    p: &PackedPlan,
    cfg: &PlanConfig,
    rg: ResolvedGemm<T::Real>,
) -> Result<ResolvedGemm<T::Real>, SelectError> {
    let Some(partition) = cfg.partition else {
        return Ok(rg);
    };
    let (policy, opts) = partition.policy();
    let rg = rg.with_partition(policy, opts)?;
    if let tcontract_kernel::PartitionPolicy::DynamicTiles { job_m, job_n } = policy {
        let (m, n) = (p.a_m.len(), p.b_n.len());
        let jobs = |m: usize, n: usize| {
            m.div_ceil(job_m)
                .checked_mul(n.div_ceil(job_n))
                // Every worker's terminal claim, with generous slack.
                .and_then(|j| j.checked_add(1 << 16))
                .filter(|&j| j < usize::MAX / 2)
        };
        if jobs(m, n).is_none() || jobs(n, m).is_none() {
            return Err(SelectError::Incompatible {
                id: rg.family().id.into(),
                reason: "DynamicTiles job count overflows for this shape",
            });
        }
    }
    Ok(rg)
}

/// The checks every resolved family passes before it may execute this plan.
fn check_family<T: Families>(
    p: &PackedPlan,
    cfg: &PlanConfig,
    rg: &ResolvedGemm<T::Real>,
) -> Result<(), SelectError> {
    let family = rg.family();
    if T::IS_COMPLEX
        && cfg
            .method
            .is_some_and(|m| family.complex.is_none_or(|s| s.method != m))
    {
        return Err(SelectError::Incompatible {
            id: family.id.into(),
            reason: "family implements a different complex method",
        });
    }
    // A family whose kernel arm the driver cannot express (unsupported complex
    // scheme) must never reach execution; validation rejects it the same way
    // unsupported descriptors are rejected at registration.
    if family.driver_family().is_none() {
        return Err(SelectError::Incompatible {
            id: family.id.into(),
            reason: "family has no driver-expressible kernel",
        });
    }
    if (p.conj_a && !family.caps.conj_a) || (p.conj_b && !family.caps.conj_b) {
        return Err(SelectError::Incompatible {
            id: family.id.into(),
            reason: "operand conjugation unsupported",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
