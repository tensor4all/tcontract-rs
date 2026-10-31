//! [`Plan`]: a prepared contraction.
//!
//! A [`Plan`] is built from a validated [`Problem`] and a [`PlanConfig`]. It
//! owns one immutable metadata record and one strategy payload:
//!
//! ```text
//! Plan<T>
//!   problem   original layouts, normalized roles, ops, C mode, checked spans
//!   strategy  Packed(PackedPlan + resolved family) | Faer(FaerPlan) | Elementwise(ElementPlan)
//!   report    immutable PlanReport
//! ```
//!
//! # Strategy selection
//!
//! Chosen once, at construction, by these rules in order:
//!
//! 1. An explicit kernel, selector, partition, complex method, blocking, cache
//!    model or write-back request forces the **packed** driver, including for
//!    an all-batch problem. It is honoured or refused.
//! 2. An all-batch problem runs the **elementwise** pass, which implements the
//!    full `op_C` / `op_D` / separate-`C` semantics.
//! 3. A problem that fuses to one strided batched GEMM without copying any
//!    operand, with full semantics, runs on **faer**, unless its GEMM volume
//!    `m * n * k` per batch item exceeds the dtype's [`FaerLimit`] (#63).
//! 4. Everything else runs on the packed driver.
//!
//! Runtime pointers, `alpha` and `beta` may pick the packed driver's existing
//! direct-B and direct-C applicability fallbacks, never another family.
//!
//! # Execution
//!
//! After one operation boundary checks layout compatibility and the
//! accumulation source, an empty output returns; `alpha == 0` or an empty
//! contraction computes `op_D(beta * op_C(C))` and reads no `A` or `B`;
//! otherwise the call dispatches once to the prepared strategy. `beta == 0`
//! never reads `C` or `D`. A plan holds no pointer to an operand and works with
//! different buffers and different executor budgets; the actual width is chosen
//! from the work estimate and the budget at each call.

mod analysis;
mod config;
mod execution;
mod orientation;
mod report;
#[cfg(test)]
pub(crate) mod test_support;

use core::marker::PhantomData;

use core::num::NonZeroUsize;
use strided_view::{StridedView, StridedViewMut};
use tcontract_exec::Exec;
use tcontract_kernel::{Element, KernelCatalog, ResolvedGemm, SelectError, TuningClass};

pub(crate) use analysis::PackedPlan;
pub use analysis::{Axis, PlanStats};
pub use config::{CacheModel, FaerLimit, Partition, PlanConfig, Writeback};
pub use execution::{
    execute_uninit_with, ExecutionRoute, OutputContract, PackedRoute, SliceAccumulationSource,
    UninitOutputError,
};
pub use orientation::{Orient, RowBlock};
pub use report::{Algorithm, PackedReport, PlanReport, Reason};

use crate::api::{
    AccumulationSource, AliasError, CSpec, ConfigError, Diagnostics, LayoutError, OperandId,
    PreparedContraction, Problem, Result, Scalar,
};
use crate::driver;
use crate::resolve;
use crate::select::Chooser;
use crate::strategy::elementwise::{CRead, ElementPlan, Expr, Inputs};
use crate::strategy::faer::{self as faer_strategy, FaerPlan, FaerWrite};
use tcontract_exec::host::ns_per_flop;

/// Dispatches one execution of a plan performs, declared to the token's width rule:
/// the packed driver publishes its grid once (a team or fork), one faer GEMM is one
/// parallel region, and an elementwise pass is one region over the output.
const PLAN_SECTIONS: u32 = 1;

#[derive(Debug)]
struct Packed<T: Scalar> {
    plan: PackedPlan,
    rg: ResolvedGemm<<T as Scalar>::Re>,
}

#[allow(clippy::large_enum_variant)]
// INVARIANT: `ElementPlan`'s axis metadata is
// inline (tinyvec) so an all-batch plan allocates nothing; boxing it would add
// one allocation to the very path #121 removes.
#[derive(Debug)]
enum Strategy<T: Scalar> {
    Packed(Box<Packed<T>>),
    Faer(FaerPlan),
    Elementwise(ElementPlan),
}

/// A prepared contraction for fixed layouts, reusable across executions.
///
/// # Examples
///
/// ```
/// use strided_view::{StridedView, StridedViewMut};
/// use tcontract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
/// use tcontract::{Plan, PlanConfig};
/// use tcontract_exec::Exec;
///
/// let l = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
/// // D[i,k] = sum_j A[i,j] B[j,k]
/// let problem = Problem::from_labels(
///     DType::F64,
///     l(&[2, 2], &[1, 2]),
///     l(&[2, 2], &[1, 2]),
///     CSpec::Absent,
///     l(&[2, 2], &[1, 2]),
///     &Labels::new(&[0, 1], &[1, 2], &[0, 2]),
/// ).unwrap();
/// let plan = Plan::<f64>::new(&problem, &PlanConfig::default()).unwrap();
/// let a = [1.0, 2.0, 3.0, 4.0];
/// let b = [1.0, 0.0, 0.0, 1.0];
/// let mut d = [0.0; 4];
/// let av = StridedView::new(&a, &[2, 2], &[1, 2], 0).unwrap();
/// let bv = StridedView::new(&b, &[2, 2], &[1, 2], 0).unwrap();
/// let mut dv = StridedViewMut::new(&mut d, &[2, 2], &[1, 2], 0).unwrap();
/// plan.execute_into(&mut Exec::serial(), 1.0, &av, &bv, &mut dv).unwrap();
/// assert_eq!(d, a);
/// ```
#[derive(Debug)]
pub struct Plan<T: Scalar> {
    problem: Problem,
    strategy: Strategy<T>,
    /// The pass over the output's elements: `alpha == 0` and empty `K`.
    output: ElementPlan,
    report: PlanReport,
    diagnostics: Diagnostics,
    /// [`Plan::retained_bytes`], counted once at construction.
    retained: usize,
    _t: PhantomData<fn() -> T>,
}

impl<T: Scalar> Plan<T> {
    /// Plan `problem` under `config`.
    ///
    /// # Errors
    ///
    /// [`ConfigError::DtypeMismatch`] when `T` is not the problem's dtype, any
    /// [`PlanConfig::validate`] error, [`Error::Select`](crate::api::Error::Select) for an unknown or
    /// unusable kernel, blocking or partition request, and
    /// [`ShapeError::Overflow`](crate::api::ShapeError) for a role whose scatter
    /// vector cannot be addressed.
    pub fn new(problem: &Problem, config: &PlanConfig) -> Result<Self> {
        Self::build(problem.clone(), config, None)
    }

    /// [`Plan::new`] taking the problem by value: the plan keeps it instead of
    /// a copy, which saves the copy's allocations on a per-call path.
    ///
    /// # Errors
    ///
    /// As [`Plan::new`].
    pub fn from_problem(problem: Problem, config: &PlanConfig) -> Result<Self> {
        Self::build(problem, config, None)
    }

    /// Compile a plan from a borrowed [`ProblemRef`](crate::api::ProblemRef): the
    /// construction inputs are borrowed, the plan owns the compiled metadata it
    /// lowers them to (#121).
    ///
    /// # Errors
    ///
    /// As [`Plan::new`], plus the lowering errors of
    /// [`Problem::from_ref`](crate::api::Problem::from_ref).
    ///
    /// # Examples
    ///
    /// ```
    /// use tcontract::api::{CRef, ContractionRef, DType, LayoutRef, Op, ProblemRef};
    /// use tcontract::{Plan, PlanConfig};
    /// let a = LayoutRef { dims: &[2, 3], strides: &[1, 2], offset: 0 };
    /// let b = LayoutRef { dims: &[3, 4], strides: &[1, 3], offset: 0 };
    /// let d = LayoutRef { dims: &[2, 4], strides: &[1, 2], offset: 0 };
    /// let pref = ProblemRef {
    ///     dtype: DType::F64,
    ///     a: (a, Op::Identity),
    ///     b: (b, Op::Identity),
    ///     c: CRef::Absent,
    ///     d: (d, Op::Identity),
    ///     contraction: ContractionRef::Labels { a: &[0, 2], b: &[2, 1], c: None, d: &[0, 1] },
    /// };
    /// let plan = Plan::<f64>::compile(&pref, &PlanConfig::default()).unwrap();
    /// assert_eq!(plan.problem().roles().k().len(), 1);
    /// ```
    pub fn compile(problem: &crate::api::ProblemRef<'_>, config: &PlanConfig) -> Result<Self> {
        Self::from_problem(crate::api::Problem::from_ref(problem)?, config)
    }

    /// [`Plan::new`] on the packed driver, choosing the kernel family with a
    /// caller-supplied selector over a caller-supplied [`KernelCatalog`]
    /// (issue #28).
    ///
    /// The selector sees metadata only -- the validated, oriented shape and the
    /// explicit method -- and returns an opaque handle. It is called at most
    /// once, here, outside every lock and worker broadcast; execution never
    /// calls it, and the plan keeps only the chosen trusted handle, so the
    /// selector and catalog may be dropped. A selector is a requirement: it
    /// forces the packed driver, including for an all-batch problem.
    ///
    /// # Errors
    ///
    /// As [`Plan::new`], plus [`Error::Select`](crate::api::Error::Select) for an explicit kernel id
    /// alongside the selector (ambiguous), no admissible candidate, the
    /// selector's own `Err` (unchanged) or a handle it should not have
    /// returned. Zero-size problems are selected and validated too.
    pub fn new_with_selector(
        problem: &Problem,
        config: &PlanConfig,
        catalog: &KernelCatalog<T>,
        selector: &mut Chooser<'_, T>,
    ) -> Result<Self> {
        if let tcontract_kernel::KernelChoice::Id(id) = &config.kernel {
            return Err(SelectError::Incompatible {
                id: id.clone(),
                reason: "a forced kernel id and a custom selector are ambiguous",
            }
            .into());
        }
        Self::build(problem.clone(), config, Some((catalog, selector)))
    }

    fn build(
        owned: Problem,
        config: &PlanConfig,
        selection: Option<(&KernelCatalog<T>, &mut Chooser<'_, T>)>,
    ) -> Result<Self> {
        let problem = &owned;
        if problem.dtype() != T::STORAGE {
            return Err(ConfigError::DtypeMismatch {
                plan: T::STORAGE.name(),
                problem: problem.dtype().name(),
            }
            .into());
        }
        config.validate()?;
        let packed = config.requires_packed() || selection.is_some();
        let faer = if packed || problem.all_batch() {
            None
        } else {
            faer_strategy::plan(problem)
        };
        let reason = if packed {
            Reason::Forced
        } else if problem.all_batch() {
            Reason::AllBatch
        } else {
            match &faer {
                None => Reason::NotFusable,
                Some(f) => {
                    let volume = f.volume();
                    match config.faer_limit.get(problem.dtype()) {
                        Some(limit) if volume > limit => Reason::AboveFaerLimit { volume, limit },
                        _ => Reason::Fused,
                    }
                }
            }
        };
        let strategy = match (reason, faer) {
            (Reason::Forced, _) => {
                Strategy::Packed(Box::new(Self::plan_packed(problem, config, selection)?))
            }
            (Reason::AllBatch, _) => Strategy::Elementwise(ElementPlan::product(problem)),
            (Reason::Fused, Some(f)) => Strategy::Faer(f),
            _ => Strategy::Packed(Box::new(Self::plan_packed(problem, config, None)?)),
        };
        let (algorithm, packed_report) = match &strategy {
            Strategy::Packed(p) => (Algorithm::Packed, Some(packed_report(p))),
            Strategy::Faer(_) => (Algorithm::Faer, None),
            Strategy::Elementwise(_) => (Algorithm::Elementwise, None),
        };
        let mut plan = Self {
            output: ElementPlan::output(problem),
            report: PlanReport {
                algorithm,
                reason,
                materialized: [false; 3],
                packed: packed_report,
            },
            diagnostics: Diagnostics::new("tcontract", algorithm.name()),
            problem: owned,
            strategy,
            retained: 0,
            _t: PhantomData,
        };
        plan.retained = plan.count_retained();
        Ok(plan)
    }

    /// The bytes this plan's construction left allocated (see
    /// [`Plan::retained_bytes`]).
    fn count_retained(&self) -> usize {
        let Self {
            problem,
            strategy,
            output,
            report,
            diagnostics: _,
            retained: _,
            _t,
        } = self;
        let strategy = match strategy {
            Strategy::Packed(p) => core::mem::size_of::<Packed<T>>() + p.plan.heap_bytes(),
            Strategy::Faer(_) => 0,
            Strategy::Elementwise(e) => e.heap_bytes(),
        };
        let report = report.packed.as_ref().map_or(0, |p| p.stats.heap_bytes());
        core::mem::size_of::<Self>()
            + problem.heap_bytes()
            + strategy
            + output.heap_bytes()
            + report
    }

    fn plan_packed(
        problem: &Problem,
        config: &PlanConfig,
        selection: Option<(&KernelCatalog<T>, &mut Chooser<'_, T>)>,
    ) -> Result<Packed<T>> {
        let plan = PackedPlan::from_problem(problem, config)?;
        let handle = match selection {
            Some((catalog, selector)) => Some(crate::select::choose::<T>(
                problem, &plan, config, catalog, selector,
            )?),
            None => None,
        };
        let rg = resolve::resolve::<T>(&plan, config, handle.as_ref())?;
        Ok(Packed { plan, rg })
    }

    /// What this plan decided: the algorithm and, for the packed strategy, the
    /// family, blocking and grid of initialized execution. A lookup, not a
    /// re-selection. Fresh Faer output may use a separately prepared packed
    /// route; `execution_route` reports that actual execution choice.
    pub fn report(&self) -> &PlanReport {
        &self.report
    }

    /// The memory this plan keeps alive, in bytes: `size_of::<Plan<T>>()` plus
    /// every heap allocation the plan owns -- the problem's spilled layouts and
    /// role tables, the packed strategy's box, scatter vectors and folded axes,
    /// the elementwise passes' spilled axis metadata and compiled output replay,
    /// and the report's folded axes. Each allocation is counted at its capacity,
    /// as the allocator sized it.
    ///
    /// Excluded: the operand and output buffers, which the caller owns, and the
    /// execution workspace (packing panels, scratch tiles), which belongs to the
    /// executor or pool that runs the plan. A budget that charges those
    /// separately does not count them twice.
    ///
    /// Counted once when the plan is built, so this is a field read: `O(1)` and
    /// allocation-free. A plan is immutable, so the count stays exact.
    ///
    /// # Examples
    ///
    /// ```
    /// use tcontract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
    /// use tcontract::{Plan, PlanConfig};
    /// let l = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
    /// let problem = Problem::from_labels(
    ///     DType::F64,
    ///     l(&[2, 3], &[1, 2]),
    ///     l(&[3, 4], &[1, 3]),
    ///     CSpec::Absent,
    ///     l(&[2, 4], &[1, 2]),
    ///     &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    /// ).unwrap();
    /// let plan = Plan::<f64>::new(&problem, &PlanConfig::default()).unwrap();
    /// assert!(plan.retained_bytes() >= core::mem::size_of::<Plan<f64>>());
    /// ```
    pub fn retained_bytes(&self) -> usize {
        self.retained
    }

    /// The validated problem this plan was built from.
    pub fn problem(&self) -> &Problem {
        &self.problem
    }

    fn check_layout(&self, which: OperandId, dims: &[usize], strides: &[isize]) -> Result<()> {
        let planned = match which {
            OperandId::A => self.problem.a().layout(),
            OperandId::B => self.problem.b().layout(),
            OperandId::D => self.problem.d().layout(),
            OperandId::C => match self.problem.c_spec() {
                CSpec::Separate(c) => c.layout(),
                _ => self.problem.d().layout(),
            },
        };
        // The offset is part of the problem's address-range check; the view
        // brings its own origin and has validated its own storage.
        if planned.dims() != dims || planned.strides() != strides {
            return Err(LayoutError::Mismatch { operand: which }.into());
        }
        Ok(())
    }

    /// `D = op_D(alpha * dot_general(op_A(A), op_B(B)))`: the overwrite form.
    /// No previous output value is read.
    ///
    /// # Errors
    ///
    /// [`LayoutError::Mismatch`] when a view differs from the plan (nothing is
    /// written); [`Error::Exec`](crate::api::Error::Exec) or [`Error::Backend`](crate::api::Error::Backend) from a lower layer.
    pub fn execute_into(
        &self,
        exec: &mut Exec<'_>,
        alpha: T,
        a: &StridedView<'_, T>,
        b: &StridedView<'_, T>,
        d: &mut StridedViewMut<'_, T>,
    ) -> Result<()> {
        self.validate_views(a, b, None, d)?;
        // SAFETY: the views' layouts equal the plan's validated layouts, so
        // their origins address every offset the plan generates; `d` is an
        // exclusive borrow and cannot alias `a` or `b`.
        unsafe {
            self.run(
                exec,
                alpha,
                a.ptr(),
                b.ptr(),
                <T as Element>::zero(),
                CRead::None,
                d.as_mut_ptr(),
            )
        }
    }

    /// `D = op_D(alpha * dot_general(op_A(A), op_B(B)) + beta * op_C(C))`.
    ///
    /// `beta == 0` reads no previous C value; `alpha == 0` or an empty
    /// contraction reads no A or B value. `source` must match the planned C
    /// mode: [`AccumulationSource::Output`] only when C maps to D (a problem
    /// built with [`CSpec::Output`]), and [`AccumulationSource::Separate`] only
    /// for a separately described C.
    ///
    /// # Errors
    ///
    /// As [`Plan::execute_into`], and [`LayoutError::CMode`] for a source that
    /// does not match the planned C mode, or nonzero beta without a C term.
    /// `AccumulationSource::Absent` with zero beta is an overwrite for any mode.
    pub fn execute_into_accum(
        &self,
        exec: &mut Exec<'_>,
        alpha: T,
        a: &StridedView<'_, T>,
        b: &StridedView<'_, T>,
        beta: T,
        source: AccumulationSource<'_, T>,
        d: &mut StridedViewMut<'_, T>,
    ) -> Result<()> {
        let c = self.validate_views(a, b, Some(source), d)?;
        self.validate_c_beta(beta, c)?;
        // SAFETY: as `execute_into`; a separate C is an immutable borrow
        // distinct from the exclusive `d`.
        unsafe { self.run(exec, alpha, a.ptr(), b.ptr(), beta, c, d.as_mut_ptr()) }
    }

    /// [`Plan::execute_into`] on plain slices. Each operand is a slice and the
    /// index of its origin (the element at logical index zero) in that slice,
    /// as [`StridedView::new`] takes them; the planned extents and strides
    /// address the rest.
    ///
    /// No view is built, so the call itself allocates no layout metadata: this
    /// is the entry for a caller that keeps layouts in the plan and holds only
    /// buffers, such as an einsum executing its intermediates out of one
    /// scratch buffer. A packed execution still takes its scratch from the
    /// `Exec`, so a warm serial steady state needs an
    /// [`Exec::serial_with_workspace`](tcontract_exec::Exec::serial_with_workspace).
    ///
    /// # Errors
    ///
    /// [`LayoutError::Bounds`] when a slice does not cover the range its
    /// layout addresses from the given origin (nothing is written);
    /// [`Error::Exec`](crate::api::Error::Exec) or
    /// [`Error::Backend`](crate::api::Error::Backend) from a lower layer.
    ///
    /// # Examples
    ///
    /// ```
    /// use tcontract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
    /// use tcontract::{Plan, PlanConfig};
    /// use tcontract_exec::Exec;
    /// let l = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
    /// // D[i,j] = sum_k A[i,k] B[k,j], column-major 2x2.
    /// let p = Problem::from_labels(
    ///     DType::F64,
    ///     l(&[2, 2], &[1, 2]),
    ///     l(&[2, 2], &[1, 2]),
    ///     CSpec::Absent,
    ///     l(&[2, 2], &[1, 2]),
    ///     &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    /// )
    /// .unwrap();
    /// let plan = Plan::<f64>::new(&p, &PlanConfig::default()).unwrap();
    /// // A is stored after one padding element.
    /// let (a, b) = ([9.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 1.0, 0.0]);
    /// let mut d = [0.0; 4];
    /// let mut exec = Exec::serial();
    /// plan.execute_slices(&mut exec, 1.0, (&a, 1), (&b, 0), (&mut d, 0)).unwrap();
    /// assert_eq!(d, [3.0, 4.0, 1.0, 2.0]);
    /// assert!(plan.execute_slices(&mut exec, 1.0, (&a, 2), (&b, 0), (&mut d, 0)).is_err());
    /// ```
    pub fn execute_slices(
        &self,
        exec: &mut Exec<'_>,
        alpha: T,
        (a, a_origin): (&[T], isize),
        (b, b_origin): (&[T], isize),
        (d, d_origin): (&mut [T], isize),
    ) -> Result<()> {
        let a = self.slice_origin(OperandId::A, a.as_ptr(), a.len(), a_origin)?;
        let b = self.slice_origin(OperandId::B, b.as_ptr(), b.len(), b_origin)?;
        let d = self.slice_origin(OperandId::D, d.as_mut_ptr(), d.len(), d_origin)?;
        // SAFETY: each origin addresses only elements of its slice (checked
        // above against the operand's span); `d` comes from an exclusive
        // borrow and cannot alias `a` or `b`.
        unsafe {
            self.run(
                exec,
                alpha,
                a,
                b,
                <T as Element>::zero(),
                CRead::None,
                d.cast_mut(),
            )
        }
    }

    /// The origin of operand `which` at index `origin` of a slice of `len`
    /// elements starting at `base`, once the slice is known to cover every
    /// element the operand's layout addresses from there.
    ///
    /// An operand with no elements is never read (its output is empty or its
    /// contraction has no terms), so `base` itself is returned for it.
    fn slice_origin(
        &self,
        which: OperandId,
        base: *const T,
        len: usize,
        origin: isize,
    ) -> Result<*const T> {
        let Some(span) = self.problem.span(which) else {
            return Ok(base);
        };
        let planned = match which {
            OperandId::A => self.problem.a().layout().offset(),
            OperandId::B => self.problem.b().layout().offset(),
            OperandId::C => match self.problem.c_spec() {
                CSpec::Separate(c) => c.layout().offset(),
                _ => self.problem.d().layout().offset(),
            },
            OperandId::D => self.problem.d().layout().offset(),
        } as i128;
        // The span is in the planned offset's coordinates; shift it to the
        // given origin.
        let shift = origin as i128 - planned;
        if span.lo() + shift < 0 || span.hi() + shift >= len as i128 {
            return Err(LayoutError::Bounds { operand: which }.into());
        }
        // INVARIANT: the origin lies in [lo, hi] shifted, inside the slice.
        Ok(base.wrapping_offset(origin))
    }

    /// Check the views of one operation against the plan, before any write,
    /// and say where the accumulation term is read from.
    ///
    /// `source` is `None` for the overwrite form.
    pub(crate) fn validate_views(
        &self,
        a: &StridedView<'_, T>,
        b: &StridedView<'_, T>,
        source: Option<AccumulationSource<'_, T>>,
        d: &StridedViewMut<'_, T>,
    ) -> Result<CRead<T>> {
        self.check_layout(OperandId::A, a.dims(), a.strides())?;
        self.check_layout(OperandId::B, b.dims(), b.strides())?;
        self.check_layout(OperandId::D, d.dims(), d.strides())?;
        Ok(match (self.problem.c_spec(), source) {
            (_, None | Some(AccumulationSource::Absent)) => CRead::None,
            (CSpec::Output(_), Some(AccumulationSource::Output)) => CRead::InPlace,
            (CSpec::Separate(_), Some(AccumulationSource::Separate(c))) => {
                self.check_layout(OperandId::C, c.dims(), c.strides())?;
                CRead::Separate(c.ptr())
            }
            _ => return Err(LayoutError::CMode.into()),
        })
    }

    /// Run one item whose views [`Plan::validate_views`] has accepted.
    ///
    /// # Safety
    ///
    /// The views are the ones that were validated, `d` is exclusive, and a
    /// separate C is distinct from `d`.
    pub(crate) unsafe fn run_validated(
        &self,
        exec: &mut Exec<'_>,
        alpha: T,
        a: &StridedView<'_, T>,
        b: &StridedView<'_, T>,
        beta: T,
        c: CRead<T>,
        d: &mut StridedViewMut<'_, T>,
    ) -> Result<()> {
        // SAFETY: forwarded.
        unsafe { self.run(exec, alpha, a.ptr(), b.ptr(), beta, c, d.as_mut_ptr()) }
    }

    /// The semantic preflight of a raw call, from the pointers' addresses alone
    /// (nothing is dereferenced): `D` overlapping `A` or `B`, a separate `C`
    /// overlapping `D` without being the same mapping at the same origin.
    ///
    /// Overlap is judged on the address range each operand's layout can reach,
    /// whether or not the call ends up reading it (`alpha == 0`, `beta == 0`),
    /// so the verdict depends on the arguments' placement only. A null pointer is
    /// treated as absent: null handling is the caller's (an FFI status, not a
    /// contraction error). For a problem built with [`CSpec::Output`] or
    /// [`CSpec::Absent`], `c` is ignored.
    ///
    /// # Errors
    ///
    /// [`AliasError::OutputOverlapsInput`] and [`AliasError::CDOverlap`].
    pub fn check_raw(&self, a: *const T, b: *const T, c: *const T, d: *const T) -> Result<()> {
        let p = &self.problem;
        if p.out_empty() || d.is_null() {
            return Ok(());
        }
        let range = |o: OperandId, origin: *const T| -> Option<(usize, usize)> {
            if origin.is_null() {
                return None;
            }
            let span = p.span(o)?;
            let off = match o {
                OperandId::A => p.a().layout().offset(),
                OperandId::B => p.b().layout().offset(),
                OperandId::D => p.d().layout().offset(),
                OperandId::C => match p.c_spec() {
                    CSpec::Separate(c) => c.layout().offset(),
                    _ => p.d().layout().offset(),
                },
            } as i128;
            let base = origin as usize as i128;
            let size = core::mem::size_of::<T>() as i128;
            let lo = base + (span.lo() - off) * size;
            let hi = base + (span.hi() - off + 1) * size;
            Some((usize::try_from(lo).ok()?, usize::try_from(hi).ok()?))
        };
        let overlap = |x: Option<(usize, usize)>, y: Option<(usize, usize)>| match (x, y) {
            (Some(x), Some(y)) => x.0 < y.1 && y.0 < x.1,
            _ => false,
        };
        let rd = range(OperandId::D, d);
        for (operand, ptr) in [(OperandId::A, a), (OperandId::B, b)] {
            if overlap(rd, range(operand, ptr)) {
                return Err(AliasError::OutputOverlapsInput { operand }.into());
            }
        }
        if matches!(p.c_spec(), CSpec::Separate(_)) && !c.is_null() {
            // A separate C may be D itself only as the same mapping at the
            // same origin (an in-place update).
            let same = core::ptr::eq(c, d) && p.c_matches_d();
            if !same && overlap(rd, range(OperandId::C, c)) {
                return Err(AliasError::CDOverlap.into());
            }
        }
        Ok(())
    }

    /// Execute on raw origins: the pointers are the elements at logical index
    /// zero of each operand. This is the entry of the C adapter, which holds
    /// pointers and no Rust references.
    ///
    /// [`Plan::check_raw`] runs first, so input errors leave `D` untouched. Null
    /// pointers are the caller's to refuse: `a` and `b` must be non-null unless
    /// the call reads neither (`alpha == 0` or an empty contraction), `c` unless
    /// it reads a separate C (`beta == 0` reads none), and `d` unless the output
    /// is empty.
    ///
    /// `c` is read only for a problem built with [`CSpec::Separate`] and a
    /// nonzero `beta`; for [`CSpec::Output`] the previous `D` is read through
    /// `d`.
    ///
    /// # Safety
    ///
    /// Every pointer the call reads or writes is the origin of a live allocation that covers the
    /// problem's address range for that operand (see
    /// [`Problem::span`](crate::api::Problem::span), relative to the operand's
    /// logical offset); `d` is valid for writes and nothing else accesses any of
    /// the memory for the duration of the call. D may be uninitialized when the
    /// call is write-only (`beta` zero, or no C read): the Faer route zero-fills
    /// it in one output pass before it forms references. A nonzero beta reading
    /// D always requires live D.
    #[allow(clippy::too_many_arguments)] // INVARIANT: the contraction argument set.
    pub unsafe fn execute_raw(
        &self,
        exec: &mut Exec<'_>,
        alpha: T,
        a: *const T,
        b: *const T,
        beta: T,
        c: *const T,
        d: *mut T,
    ) -> Result<()> {
        #[cfg(feature = "phase-timing")]
        crate::phase::report_previous();
        self.check_raw(a, b, c, d)?;
        // SAFETY: preflight passed; no caller workspace, so the context's own
        // storage (or call-local scratch) serves the call.
        unsafe { self.raw_dispatch(None, None, exec, alpha, a, b, beta, c, d) }
    }

    /// [`Plan::execute_raw`] with a caller-owned, bounded
    /// [`Workspace`](tcontract_exec::Workspace): the plan checks
    /// [`Plan::workspace_req`] against the workspace's bound before any write
    /// and returns [`Error::Workspace`](crate::api::Error::Workspace) when it is
    /// short, so a steady state within the bound allocates nothing.
    ///
    /// `output` is the storage guarantee of `D`, as in
    /// [`OutputContract`](crate::OutputContract).
    ///
    /// # Errors
    ///
    /// As [`Plan::execute_raw`], plus
    /// [`Error::Workspace`](crate::api::Error::Workspace) when `workspace` is
    /// smaller than the plan's conservative requirement.
    ///
    /// # Safety
    ///
    /// As [`Plan::execute_raw`].
    #[allow(clippy::too_many_arguments)] // INVARIANT: the contraction argument set.
    pub unsafe fn execute_raw_in(
        &self,
        workspace: &mut tcontract_exec::Workspace,
        exec: &mut Exec<'_>,
        alpha: T,
        a: *const T,
        b: *const T,
        beta: T,
        c: *const T,
        d: *mut T,
        output: OutputContract,
    ) -> Result<()> {
        #[cfg(feature = "phase-timing")]
        crate::phase::report_previous();
        self.check_raw(a, b, c, d)?;
        // Sizing and dispatch share one output-route decision: `workspace_req`
        // resolves `output` exactly as `raw_dispatch` will, so a plan that sizes
        // zero cannot dispatch its packed alternative.
        let required = self.workspace_req(exec, output);
        // A workspace not warmed for every worker of this executor would have a
        // worker allocate its own slot mid-execute: refuse instead. The executor
        // the driver uses is the narrowed one (`Auto`/`Sequential`).
        let needed_slots = self.batch_budget(exec);
        if !required.covered_by(workspace.req()) || workspace.slots() < needed_slots {
            return Err(crate::api::Error::Workspace {
                needed: required.bytes(),
                have: workspace.have(),
            });
        }
        // SAFETY: preflight passed and the bound was checked before any write;
        // the workspace outlives the borrow taken here.
        unsafe {
            self.raw_dispatch(
                Some(workspace.as_provider()),
                Some(output),
                exec,
                alpha,
                a,
                b,
                beta,
                c,
                d,
            )
        }
    }

    /// The raw semantics shared by [`Plan::execute_raw`] and
    /// [`Plan::execute_raw_in`]: the C read, the storage contract, then the one
    /// dispatch. `check_raw` and any workspace-bound check already ran.
    ///
    /// # Safety
    ///
    /// As [`Plan::execute_raw`], with the preflight already done.
    #[allow(clippy::too_many_arguments)] // INVARIANT: the contraction argument set.
    unsafe fn raw_dispatch(
        &self,
        workspace: Option<&dyn tcontract_exec::WorkspaceProvider>,
        output: Option<OutputContract>,
        exec: &mut Exec<'_>,
        alpha: T,
        a: *const T,
        b: *const T,
        beta: T,
        c: *const T,
        d: *mut T,
    ) -> Result<()> {
        let zero = <T as Element>::zero();
        let c_read = match self.problem.c_spec() {
            CSpec::Absent => CRead::None,
            CSpec::Output(_) => CRead::InPlace,
            CSpec::Separate(_) if beta == zero => CRead::None,
            CSpec::Separate(_) => CRead::Separate(c),
        };
        self.validate_c_beta(beta, c_read)?;
        let write_only = beta == zero
            || matches!(c_read, CRead::None)
            || matches!(c_read, CRead::Separate(cp) if !core::ptr::eq(cp, d));
        let fresh = match output {
            // `execute_raw` has no declared contract: keep its write-only
            // optimization (a fresh route is only safe when write-only).
            None => write_only,
            // An initialized destination uses the initialized route, which needs
            // no prepared fresh alternative and no workspace.
            Some(OutputContract::Initialized) => false,
            Some(OutputContract::Fresh) => {
                if !write_only {
                    // A fresh D would be read as the accumulation source.
                    return Err(LayoutError::CMode.into());
                }
                true
            }
        };
        #[cfg(feature = "phase-timing")]
        let whole = std::time::Instant::now();
        // SAFETY: the caller's contract and the preflight.
        let out =
            unsafe { self.run_storage_in(workspace, exec, alpha, a, b, beta, c_read, d, fresh) };
        #[cfg(feature = "phase-timing")]
        crate::phase::add_total(whole.elapsed().as_nanos() as u64);
        out
    }

    /// The width the work estimate asks for at the executor's budget: the token's
    /// rule (`Exec::width_for`) over the serial estimate and the declared sections.
    fn width(&self, exec: &Exec<'_>) -> usize {
        let flops = 2.0 * self.problem.macs() as f64 * if T::IS_COMPLEX { 4.0 } else { 1.0 };
        exec.width_for(flops * ns_per_flop(), PLAN_SECTIONS)
    }

    /// The context the driver sees: `Auto` narrows the budget over the whole
    /// batch first (the packed two-stage rule's stage one), a forced batch
    /// strategy is left on the original host budget, as
    /// [`tcontract_exec::Exec::batch_decision`] documents.
    fn batch_context<'e>(&self, exec: &'e mut Exec<'_>) -> Exec<'e> {
        let budget = NonZeroUsize::new(self.batch_budget(exec)).unwrap_or(NonZeroUsize::MIN);
        exec.reborrow().limit(budget)
    }

    /// The budget [`Plan::batch_context`] leaves the driver, as a number, for
    /// the reports that need it without a mutable context.
    fn batch_budget(&self, exec: &Exec<'_>) -> usize {
        match exec.batch_policy().strategy {
            // `Auto` narrows the budget over the whole batch (the packed
            // two-stage rule's stage one).
            tcontract_exec::BatchStrategy::Auto => self.width(exec),
            // A forced sequential route is one thread for the whole call,
            // including the faer output pass and any packed team.
            tcontract_exec::BatchStrategy::Sequential => 1,
            // Every other forced strategy is resolved on the original budget.
            _ => exec.budget(),
        }
    }

    /// After preflight: output-empty returns; `alpha == 0` / empty `K` uses the
    /// output-update helper; otherwise one dispatch to the prepared strategy.
    ///
    /// # Safety
    ///
    /// As [`Plan::execute_raw`], with the preflight already done.
    #[allow(clippy::too_many_arguments)] // INVARIANT: the contraction argument set.
    unsafe fn run(
        &self,
        exec: &mut Exec<'_>,
        alpha: T,
        a: *const T,
        b: *const T,
        beta: T,
        c: CRead<T>,
        d: *mut T,
    ) -> Result<()> {
        // SAFETY: the initialized execution entry's validated buffers.
        #[cfg(feature = "phase-timing")]
        let whole = std::time::Instant::now();
        let out = unsafe { self.run_storage(exec, alpha, a, b, beta, c, d, false) };
        #[cfg(feature = "phase-timing")]
        crate::phase::add_total(whole.elapsed().as_nanos() as u64);
        out
    }

    /// Execute with a storage contract fixed by the entry point. Fresh product
    /// execution never reaches a Faer leaf that forms references before writing.
    ///
    /// # Safety
    /// As `run`; when `fresh` is true no old D value may be required.
    #[allow(clippy::too_many_arguments)] // INVARIANT: the contraction argument set.
    unsafe fn run_storage(
        &self,
        exec: &mut Exec<'_>,
        alpha: T,
        a: *const T,
        b: *const T,
        beta: T,
        c: CRead<T>,
        d: *mut T,
        fresh: bool,
    ) -> Result<()> {
        // SAFETY: forwarded.
        unsafe { self.run_storage_in(None, exec, alpha, a, b, beta, c, d, fresh) }
    }

    /// [`Plan::run_storage`] with an explicit caller workspace. `None` keeps the
    /// current behaviour (the context's own provider, or call-local scratch).
    ///
    /// # Safety
    /// As `run_storage`.
    #[allow(clippy::too_many_arguments)] // INVARIANT: the contraction argument set.
    unsafe fn run_storage_in(
        &self,
        workspace: Option<&dyn tcontract_exec::WorkspaceProvider>,
        exec: &mut Exec<'_>,
        alpha: T,
        a: *const T,
        b: *const T,
        beta: T,
        c: CRead<T>,
        d: *mut T,
        fresh: bool,
    ) -> Result<()> {
        let p = &self.problem;
        if p.out_empty() {
            return Ok(());
        }
        let zero = <T as Element>::zero();
        let (conj_a, conj_b) = (p.a().op().is_conj(), p.b().op().is_conj());
        let (conj_c, conj_d) = (p.op_c().is_conj(), p.d().op().is_conj());
        // No C term means beta is zero, whatever the caller passed.
        let beta = if matches!(c, CRead::None) { zero } else { beta };
        let expr = Expr {
            alpha,
            beta,
            conj_a,
            conj_b,
            conj_c,
            conj_d,
        };
        if p.k_empty() || alpha == zero {
            // D = op_D(beta * op_C(C)); A and B are not referenced. An
            // in-place identity update changes nothing.
            if matches!(c, CRead::InPlace) && beta == <T as Element>::one() && !conj_c && !conj_d {
                return Ok(());
            }
            // SAFETY: the caller's contract.
            unsafe { self.output.run(exec, expr, Inputs::None, c, d) }?;
            return Ok(());
        }
        if let Strategy::Faer(f) = &self.strategy {
            let mut exec = self.batch_context(exec);
            // The C term goes into D first, in one parallel output-sized
            // pass (not an operand copy; see `strategy::faer`), and faer
            // accumulates the product. D itself (in place, or a separate C
            // that is the same mapping at D's origin) needs the pass only
            // when `beta` and the conjugations are not the identity;
            // `beta == 0` reads no C and faer overwrites D. The batch policy is
            // resolved first so a forced `Sequential` limits this pass too.
            //
            // A fresh (uninitialized) D is write-only by contract: a separate C
            // with a nonzero `beta` initializes it through the output pass, and
            // every other write-only form (no C, or any C at `beta == 0`) is
            // handed to faer's overwrite-only entry
            // (`matmul_with_conj_uninit`, t4a-faer 0.27), which never reads D's
            // old value and never forms a reference to it before writing. That
            // entry skips its own initialization pass on x86_64 with AVX2 or
            // AVX-512, so a fresh call no longer pays the serial output-sized
            // fill this branch used to run before the product (an 8-18 MB
            // output took 0.3-1.4 ms serially on a 96-core node, comparable to
            // the product itself at high thread counts; tcontract-rs#166). The
            // packed driver that served fresh outputs before was measured 4x
            // slower than faer on a small-M GEMM (tenferro-rs#2233 decode_proj,
            // M=8, K=N=1024: 2.4 ms against 0.57 ms), so faer still serves them.
            let write = match c {
                CRead::Separate(cp) if beta != zero && !core::ptr::eq(cp, d) => {
                    // SAFETY: the caller's contract (C disjoint from D).
                    unsafe { self.output.run(&mut exec, expr, Inputs::None, c, d) }?;
                    FaerWrite::Accumulate
                }
                _ if fresh => FaerWrite::Fresh,
                CRead::None => FaerWrite::Replace,
                _ if beta == zero => FaerWrite::Replace,
                _ => {
                    let identity = beta == <T as Element>::one() && !conj_c && !conj_d;
                    if !identity {
                        // SAFETY: the caller's contract; in-place update of D.
                        unsafe {
                            self.output
                                .run(&mut exec, expr, Inputs::None, CRead::InPlace, d)
                        }?;
                    }
                    FaerWrite::Accumulate
                }
            };
            // SAFETY: the caller's contract; the fusion was proven copy-free
            // over exactly this problem's layouts.
            unsafe { f.run(&mut exec, alpha, a, b, write, d)? };
            return Ok(());
        }
        match &self.strategy {
            Strategy::Packed(pk) => {
                let mut exec = self.batch_context(exec);
                // The caller owns the workspace: a pool lends its arena, a
                // caller that wants serial reuse lends its own provider, and a
                // bare serial context runs with call-local scratch.
                let cp = match c {
                    CRead::Separate(c) => c,
                    _ => d as *const T,
                };
                // SAFETY: the caller's contract; `pk.rg` was validated for this
                // plan when it was built.
                unsafe {
                    driver::execute_packed(
                        &pk.plan, &pk.rg, &mut exec, workspace, alpha, a, b, beta, cp, d,
                    )?
                };
            }
            Strategy::Faer(_) => unreachable!("served above"),
            Strategy::Elementwise(e) => {
                // SAFETY: the caller's contract.
                unsafe { e.run(exec, expr, Inputs::Product(a, b), c, d) }?;
            }
        }
        Ok(())
    }
}

fn packed_report<T: Scalar>(pk: &Packed<T>) -> PackedReport {
    let rg = &pk.rg;
    let fam = rg.family();
    let element = core::mem::size_of::<<T as Scalar>::Re>();
    let a_reals = fam.a_per_k / fam.mr;
    let b_reals = fam.b_per_k / fam.nr;
    let scratch_bytes = (rg.mc * rg.kc * a_reals + rg.nc * rg.kc * b_reals) * element;
    let swapped = pk.plan.transposes_gemm(rg.mr);
    let (rows, cols) = if swapped {
        (&pk.plan.b_n, &pk.plan.a_m)
    } else {
        (&pk.plan.a_m, &pk.plan.b_n)
    };
    let regular = |scatter: &[i64], block: usize| {
        tcontract_kernel::scatter::regular_fraction(
            &tcontract_kernel::scatter::build_block_scatter(scatter, block),
        )
    };
    PackedReport {
        swapped,
        regular_rows: regular(rows, rg.mr),
        regular_cols: regular(cols, rg.nr),
        family_id: fam.id,
        tuning_class: TuningClass::detect(),
        origin: fam.origin,
        complex: fam.complex,
        mr: rg.mr,
        nr: rg.nr,
        mc: rg.mc,
        nc: rg.nc,
        kc: rg.kc,
        partition: rg.partition,
        align_c_lines: rg.opts.align_c_lines,
        dynamic: driver::dynamic_report(&pk.plan, rg, usize::MAX),
        stats: pk.plan.stats.clone(),
        scratch_bytes,
    }
}

impl<T: Scalar> PreparedContraction<T> for Plan<T> {
    fn execute_into(
        &self,
        exec: &mut Exec<'_>,
        alpha: T,
        a: &StridedView<'_, T>,
        b: &StridedView<'_, T>,
        d: &mut StridedViewMut<'_, T>,
    ) -> Result<()> {
        Plan::execute_into(self, exec, alpha, a, b, d)
    }

    fn execute_into_accum(
        &self,
        exec: &mut Exec<'_>,
        alpha: T,
        a: &StridedView<'_, T>,
        b: &StridedView<'_, T>,
        beta: T,
        source: AccumulationSource<'_, T>,
        d: &mut StridedViewMut<'_, T>,
    ) -> Result<()> {
        Plan::execute_into_accum(self, exec, alpha, a, b, beta, source, d)
    }

    fn diagnostics(&self) -> &Diagnostics {
        &self.diagnostics
    }
}
