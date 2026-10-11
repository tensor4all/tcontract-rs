//! Safe user-defined kernel selection (tcontract addition, issue #28).
//!
//! [`Plan::new_with_selector`](crate::Plan::new_with_selector) lets a caller
//! pass their own [`KernelCatalog`] together with a selection callback. The
//! callback receives metadata only -- a [`SelectionContext`] and one
//! [`KernelCandidate`] per admissible family -- and returns an opaque
//! [`KernelHandle`]. The plan keeps that trusted handle; execution never calls
//! the callback, consults the catalog or looks an id up again.
//!
//! The candidate facts are computed here, with the same helpers the driver
//! uses at execution (`PackedPlan::transposes_gemm` and the direct-B rule), so
//! what a selector is shown is what the driver will do.

use tcontract_kernel::{CpuFeatures, Families, KernelHandle, Method, SelectError};

use crate::api::{OperandId, Problem};
use crate::driver;
use crate::plan::{PackedPlan, PlanConfig, PlanStats};

/// What a selector returns: the trusted handle it chose, or why it declined.
pub type Selection<T> = core::result::Result<KernelHandle<T>, SelectError>;

/// A selector as the planner holds it while building one plan.
pub type Chooser<'a, T> =
    dyn FnMut(&SelectionContext<'_>, &[KernelCandidate<T>]) -> Selection<T> + 'a;

/// One operand's layout as the caller described it, borrowed from the problem
/// (nothing tensor-sized is copied).
#[derive(Clone, Copy, Debug)]
pub struct OperandMeta<'a> {
    /// Original extents.
    pub extents: &'a [usize],
    /// Original signed element strides (zero strides are broadcasts).
    pub strides: &'a [isize],
    /// Logical offset.
    pub offset: isize,
    /// Whether the operand is read conjugated.
    pub conj: bool,
}

/// Problem-level facts shown to a selector. Candidate-specific consequences
/// (orientation, direct B) are in [`KernelCandidate`], because they depend on
/// each candidate's geometry.
///
/// Bounds and pointer-aliasing facts (for example whether `C` and `D` are the
/// same buffer) are not known when a plan is built; they stay execution
/// guards, not selector guarantees. Neither is the thread budget: a plan never
/// reselects its family for a different budget, so the selection cannot depend
/// on one.
#[derive(Clone, Copy, Debug)]
pub struct SelectionContext<'a> {
    /// Storage scalar name: `"f32"`, `"f64"`, `"c32"` or `"c64"`.
    pub dtype: &'static str,
    /// Whether the storage type is complex.
    pub is_complex: bool,
    /// Folded matrix shape, batch count and axis structure the index analysis
    /// reached (`m`, `n`, `k`, `batch`, `is_pure_gemm`, folded axes).
    pub stats: &'a PlanStats,
    /// A, B, C and D. A problem without a C operand reports D's layout as C.
    pub operands: [OperandMeta<'a>; 4],
    /// Requested complex method, or none for the family's own.
    pub method: Option<Method>,
    /// CPU/OS instruction sets available; candidates already satisfy it.
    pub cpu: CpuFeatures,
    /// Whether the problem has no work (a zero extent); the selection and its
    /// validation still run.
    pub is_empty: bool,
}

/// One admissible family with facts derived under *its own* geometry.
#[derive(Clone, Copy, Debug)]
pub struct KernelCandidate<T: Families> {
    /// The handle to return to select this family; also carries its metadata.
    pub handle: KernelHandle<T>,
    /// Whether the driver exchanges (A, M) with (B, N) for this family's `MR`.
    pub swapped: bool,
    /// Whether this family reads the column operand in place (no B buffer).
    pub direct_b: bool,
}

/// Why `handle` cannot serve `plan`, using the checks a built-in resolution
/// applies; `None` when it can.
fn inadmissible<T: Families>(
    plan: &PackedPlan,
    cfg: &PlanConfig,
    handle: &KernelHandle<T>,
) -> Option<&'static str> {
    if !handle.caps().scatter_pack {
        return Some("the packed driver needs scatter packing");
    }
    if (plan.conj_a && !handle.caps().conj_a) || (plan.conj_b && !handle.caps().conj_b) {
        return Some("operand conjugation unsupported");
    }
    if T::IS_COMPLEX {
        if let Some(m) = cfg.method {
            if handle.complex().is_none_or(|s| s.method != m) {
                return Some("family implements a different complex method");
            }
        }
    }
    None
}

fn candidate<T: Families>(plan: &PackedPlan, handle: KernelHandle<T>) -> KernelCandidate<T> {
    let (mr, nr) = (handle.mr(), handle.nr());
    let swapped = plan.transposes_gemm(mr);
    let (bk, bn) = if swapped {
        (&plan.a_k, &plan.a_m)
    } else {
        (&plan.b_k, &plan.b_n)
    };
    let direct_b = !driver::pack_b_needed(handle.b_access(), bk, bn, nr);
    KernelCandidate {
        handle,
        swapped,
        direct_b,
    }
}

fn operand_meta(p: &Problem, which: OperandId) -> OperandMeta<'_> {
    let (layout, conj) = match which {
        OperandId::A => (p.a().layout(), p.a().op().is_conj()),
        OperandId::B => (p.b().layout(), p.b().op().is_conj()),
        OperandId::D => (p.d().layout(), p.d().op().is_conj()),
        OperandId::C => match p.c_spec() {
            crate::api::CSpec::Separate(c) => (c.layout(), c.op().is_conj()),
            _ => (p.d().layout(), p.op_c().is_conj()),
        },
    };
    OperandMeta {
        extents: layout.dims(),
        strides: layout.strides(),
        offset: layout.offset(),
        conj,
    }
}

/// Run `selector` once over the admissible candidates of `catalog` and return
/// the trusted handle it chose.
///
/// # Errors
///
/// `NoCandidates` when the catalog has nothing admissible (CPU, conjugation,
/// complex method, scatter packing); whatever the selector returned in `Err`,
/// unchanged (no fallback is substituted); `ForeignHandle` for a handle another
/// catalog minted; `CpuUnsupported` or `NotACandidate` for a catalog member
/// that was not among the offered candidates.
pub(crate) fn choose<T: Families>(
    problem: &Problem,
    plan: &PackedPlan,
    cfg: &PlanConfig,
    catalog: &tcontract_kernel::KernelCatalog<T>,
    selector: &mut Chooser<'_, T>,
) -> Result<KernelHandle<T>, SelectError> {
    let cpu = CpuFeatures::detect();
    let candidates: Vec<KernelCandidate<T>> = catalog
        .handles()
        .filter(|h| cpu.contains(h.required()) && inadmissible(plan, cfg, h).is_none())
        .map(|h| candidate(plan, h))
        .collect();
    if candidates.is_empty() {
        return Err(SelectError::NoCandidates { dtype: T::DTYPE });
    }
    let context = SelectionContext {
        dtype: T::DTYPE,
        is_complex: T::IS_COMPLEX,
        stats: &plan.stats,
        operands: [OperandId::A, OperandId::B, OperandId::C, OperandId::D]
            .map(|o| operand_meta(problem, o)),
        method: cfg.method,
        cpu,
        is_empty: plan.is_empty(),
    };
    let chosen = selector(&context, &candidates)?;
    catalog.admit(&chosen, cpu)?;
    if let Some(reason) = inadmissible(plan, cfg, &chosen) {
        return Err(SelectError::NotACandidate {
            id: chosen.id().into(),
            reason,
        });
    }
    Ok(chosen)
}
