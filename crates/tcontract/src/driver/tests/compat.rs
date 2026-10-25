//! Test-only adapters for the ported driver tests.
//!
//! The driver tests were written against the old label-based `tensorcontract`
//! vocabulary (`Layout` with `i64` extents, `Operand`, a dtype-independent
//! `Plan` with builder methods, `contract_reference` over `Layout`s). The
//! library no longer has those types; this module rebuilds just enough of them
//! on top of the new ones that the numerical cases keep their shape: a
//! [`Plan`] holds the validated problem and a [`PlanConfig`], derives the
//! [`PackedPlan`] and the resolved family on demand, and runs the packed driver
//! directly. Nothing here is a second implementation of anything the library
//! offers.

use core::ops::Deref;

use tcontract_exec::{Exec, WorkspaceProvider};
use tcontract_kernel::{
    Blocking, ComplexMethod, KernelChoice, Method, PartitionOpts, PartitionPolicy, ResolvedGemm,
    SelectError,
};
use tcontract_testkit::oracle;

use crate::api::{
    CSpec, DType, Error, Labels, LayoutSpec, Op, OperandSpec, Problem, Result, Scalar,
};
use crate::driver::{self, ResolvedCall};
use crate::plan::{PackedPlan, Partition, PlanConfig};
use crate::resolve;

/// Extents and strides with the old `i64` vocabulary.
#[derive(Clone, PartialEq, Eq)]
pub struct Layout {
    extents: Vec<i64>,
    strides: Vec<i64>,
}

impl core::fmt::Debug for Layout {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Layout")
            .field("extents", &self.extents)
            .field("strides", &self.strides)
            .finish()
    }
}

impl Layout {
    /// A layout from one stride per extent.
    pub fn new(extents: Vec<i64>, strides: Vec<i64>) -> Result<Self> {
        assert_eq!(extents.len(), strides.len(), "one stride per extent");
        Ok(Self { extents, strides })
    }

    /// Column-major strides (the first axis is fastest).
    pub fn col_major(extents: &[i64]) -> Self {
        let mut strides = Vec::new();
        let mut acc = 1i64;
        for &e in extents {
            strides.push(acc);
            acc *= e.max(1);
        }
        Self {
            extents: extents.to_vec(),
            strides,
        }
    }

    /// Row-major strides (the last axis is fastest).
    pub fn row_major(extents: &[i64]) -> Self {
        let mut strides = vec![0; extents.len()];
        let mut acc = 1i64;
        for (k, &e) in extents.iter().enumerate().rev() {
            strides[k] = acc;
            acc *= e.max(1);
        }
        Self {
            extents: extents.to_vec(),
            strides,
        }
    }

    /// The smallest allocation that backs this layout (non-negative strides,
    /// base pointer at zero).
    pub fn storage_len(&self) -> i64 {
        let mut n = 1i64;
        for (&e, &s) in self.extents.iter().zip(&self.strides) {
            if e > 0 {
                n += (e - 1) * s.abs();
            }
        }
        n
    }

    fn dims(&self) -> Vec<usize> {
        self.extents.iter().map(|&e| e as usize).collect()
    }

    fn strides_isize(&self) -> Vec<isize> {
        self.strides.iter().map(|&s| s as isize).collect()
    }
}

/// The old per-operand element operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ElementOp {
    /// As stored.
    #[default]
    Identity,
    /// Conjugated.
    Conjugate,
}

impl ElementOp {
    /// Whether this conjugates.
    pub fn is_conj(self) -> bool {
        self == ElementOp::Conjugate
    }

    fn op(self) -> Op {
        if self.is_conj() {
            Op::Conjugate
        } else {
            Op::Identity
        }
    }
}

/// An operand: layout, labels and element operation.
#[derive(Clone, Copy, Debug)]
pub struct Operand<'a> {
    /// Extents and strides.
    pub layout: &'a Layout,
    /// One label per mode.
    pub idx: &'a [i64],
    /// The element operation.
    pub op: ElementOp,
}

impl<'a> Operand<'a> {
    /// An operand read as stored.
    pub fn new(layout: &'a Layout, idx: &'a [i64]) -> Self {
        Self {
            layout,
            idx,
            op: ElementOp::Identity,
        }
    }

    /// The same operand, conjugated.
    #[must_use]
    pub fn conj(mut self) -> Self {
        self.op = ElementOp::Conjugate;
        self
    }
}

#[derive(Clone)]
struct Owned {
    layout: Layout,
    idx: Vec<i64>,
    op: ElementOp,
}

impl Owned {
    fn of(o: Operand<'_>) -> Self {
        Self {
            layout: o.layout.clone(),
            idx: o.idx.to_vec(),
            op: o.op,
        }
    }

    fn spec(&self) -> OperandSpec {
        OperandSpec::new(
            LayoutSpec::new(&self.layout.dims(), &self.layout.strides_isize(), 0).unwrap(),
        )
        .with_op(self.op.op())
    }
}

/// A dtype-independent plan with builder methods, over the new machinery.
#[derive(Clone)]
pub struct Plan {
    a: Owned,
    b: Owned,
    c: Option<Owned>,
    d: Owned,
    cfg: PlanConfig,
    threads: usize,
    blocking: Option<Blocking>,
    packed: PackedPlan,
}

impl Deref for Plan {
    type Target = PackedPlan;
    fn deref(&self) -> &PackedPlan {
        &self.packed
    }
}

impl core::fmt::Debug for Plan {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Plan")
            .field("stats", &self.packed.stats)
            .finish()
    }
}

impl Plan {
    fn problem_for(&self, dtype: DType) -> Result<Problem> {
        let (a, b, d) = (self.a.spec(), self.b.spec(), self.d.spec());
        let (c_spec, labels) = match &self.c {
            // A C operand with D's layout and labels is the in-place source.
            Some(c) if c.layout == self.d.layout && c.idx == self.d.idx => (
                CSpec::Output(c.op.op()),
                Labels::new(&self.a.idx, &self.b.idx, &self.d.idx),
            ),
            Some(c) => (
                CSpec::Separate(c.spec()),
                Labels::new(&self.a.idx, &self.b.idx, &self.d.idx).with_c(&c.idx),
            ),
            None => (
                CSpec::Absent,
                Labels::new(&self.a.idx, &self.b.idx, &self.d.idx),
            ),
        };
        Problem::from_labels(dtype, a, b, c_spec, d, &labels)
    }

    /// Analyse a contraction (validated as for `f64`; the analysis does not
    /// depend on the element type).
    pub fn new(
        a: Operand<'_>,
        b: Operand<'_>,
        c: Option<Operand<'_>>,
        d: Operand<'_>,
    ) -> Result<Plan> {
        let cfg = PlanConfig::default();
        let mut p = Plan {
            a: Owned::of(a),
            b: Owned::of(b),
            c: c.map(Owned::of),
            d: Owned::of(d),
            cfg,
            threads: 1,
            blocking: None,
            // Replaced below, once the problem validated.
            packed: PackedPlan::from_problem(
                &Problem::from_labels(
                    DType::F64,
                    Owned::of(a).spec(),
                    Owned::of(b).spec(),
                    CSpec::Absent,
                    Owned::of(d).spec(),
                    &Labels::new(a.idx, b.idx, d.idx),
                )?,
                &PlanConfig::default(),
            )?,
        };
        let problem = p.problem_for(DType::F64)?;
        p.packed = PackedPlan::from_problem(&problem, &p.cfg)?;
        Ok(p)
    }

    /// Choose a kernel (checked when the plan is resolved).
    pub fn with_kernel(mut self, choice: KernelChoice) -> core::result::Result<Self, Error> {
        self.cfg.kernel = choice;
        Ok(self)
    }

    /// The width the plan is resolved at.
    #[must_use]
    pub fn with_threads(mut self, n: usize) -> Self {
        self.threads = n.max(1);
        self
    }

    /// The requested width.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Explicit cache blocking.
    #[must_use]
    pub fn with_blocking(mut self, blk: Blocking) -> Self {
        self.blocking = Some(blk);
        self
    }

    /// The complex scheme.
    #[must_use]
    pub fn with_complex_method(mut self, method: ComplexMethod) -> Self {
        self.cfg.method = Some(match method {
            ComplexMethod::Planar => Method::Native,
            ComplexMethod::OneM => Method::OneM,
            ComplexMethod::ThreeM => Method::ThreeM,
        });
        self
    }

    /// A grid policy.
    #[must_use]
    pub fn with_partition(mut self, policy: PartitionPolicy, opts: PartitionOpts) -> Self {
        self.cfg.partition = Some(match policy {
            PartitionPolicy::StaticGrid { pm, pn } => Partition::StaticGrid {
                pin: (pm != 0).then_some((pm, pn)),
                align_c_lines: opts.align_c_lines,
            },
            PartitionPolicy::DynamicTiles { job_m, job_n } => {
                Partition::DynamicTiles { job_m, job_n }
            }
            _ => unreachable!("non-exhaustive policy"),
        });
        self
    }

    /// The grid at the requested width.
    pub fn partition(&self, mr: usize, nr: usize) -> (usize, usize) {
        self.packed.partition_with(mr, nr, self.threads)
    }

    /// The resolved family at this plan's width and blocking.
    pub fn resolved<T: Scalar>(
        &self,
    ) -> core::result::Result<ResolvedGemm<<T as Scalar>::Re>, SelectError> {
        let problem = self
            .problem_for(T::STORAGE)
            .expect("validated at construction");
        let packed = PackedPlan::from_problem(&problem, &self.cfg).expect("role products fit");
        let mut rg = resolve::resolve::<T>(&packed, &self.cfg, None)?;
        if let Some(blk) = self.blocking {
            rg = rg.with_blocking(blk)?;
        }
        rg.with_threads(self.threads)
    }

    /// Run on raw origins, serially.
    ///
    /// # Safety
    ///
    /// Every pointer covers the plan's offsets; `d` is exclusive.
    pub unsafe fn run_raw<T: Scalar>(
        &self,
        alpha: T,
        a: *const T,
        b: *const T,
        beta: T,
        c: *const T,
        d: *mut T,
    ) {
        // SAFETY: forwarded.
        unsafe { self.run_raw_with(&mut Exec::serial(), None, alpha, a, b, beta, c, d) }
            .expect("a serial route is always available")
    }

    /// Run on raw origins on `exec`.
    ///
    /// # Safety
    ///
    /// As [`Plan::run_raw`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run_raw_with<T: Scalar>(
        &self,
        exec: &mut Exec<'_>,
        workspace: Option<&dyn WorkspaceProvider>,
        alpha: T,
        a: *const T,
        b: *const T,
        beta: T,
        c: *const T,
        d: *mut T,
    ) -> std::result::Result<(), tcontract_exec::ExecError> {
        let rg = self.resolved::<T>().expect("a valid resolution");
        let problem = self.problem_for(T::STORAGE).expect("validated");
        let packed = PackedPlan::from_problem(&problem, &self.cfg).expect("role products fit");
        // SAFETY: the caller's contract; the resolution was just validated.
        unsafe { driver::execute_packed(&packed, &rg, exec, workspace, alpha, a, b, beta, c, d) }
    }

    /// The driver's operand-dependent decisions for `T` on these pointers.
    pub fn decisions<T: Scalar>(
        &self,
        rg: &ResolvedGemm<<T as Scalar>::Re>,
        c: *const T,
        d: *mut T,
        beta: T,
    ) -> ResolvedCall {
        let problem = self.problem_for(T::STORAGE).expect("validated");
        let packed = PackedPlan::from_problem(&problem, &self.cfg).expect("role products fit");
        driver::driver_decisions::<T>(&packed, rg, c, d, beta)
    }

    /// The default family's register block and blocking for `T`: what the
    /// engine uses when nothing is forced.
    pub fn selected_config<T: Scalar>(&self) -> (usize, usize, Blocking) {
        let rg = self.resolved::<T>().expect("a valid resolution");
        (
            rg.mr,
            rg.nr,
            Blocking {
                mc: rg.mc,
                kc: rg.kc,
                nc: rg.nc,
            },
        )
    }
}

/// An operand of the oracle, with the old `Layout` vocabulary.
#[derive(Clone, Copy, Debug)]
pub struct RefOperand<'a, T> {
    /// The backing allocation.
    pub data: &'a [T],
    /// Extents and strides.
    pub layout: &'a Layout,
    /// One label per mode.
    pub idx: &'a [i64],
    /// The element operation.
    pub op: ElementOp,
}

/// The oracle, with the old signature.
#[allow(clippy::too_many_arguments)]
pub fn contract_reference<T: tcontract_kernel::Element>(
    alpha: T,
    a: &RefOperand<'_, T>,
    b: &RefOperand<'_, T>,
    beta: T,
    c: Option<&RefOperand<'_, T>>,
    d_data: &mut [T],
    d_layout: &Layout,
    idx_d: &[i64],
    op_d: ElementOp,
) -> core::result::Result<(), oracle::OracleError> {
    let conv = |o: &RefOperand<'_, T>| {
        (
            o.layout.dims(),
            o.layout.strides_isize(),
            o.idx.to_vec(),
            o.op.is_conj(),
        )
    };
    let (ad, as_, al, ac) = conv(a);
    let (bd, bs, bl, bc) = conv(b);
    let cc = c.map(conv);
    let ra = oracle::RefOperand {
        data: a.data,
        dims: &ad,
        strides: &as_,
        offset: 0,
        labels: &al,
        conj: ac,
    };
    let rb = oracle::RefOperand {
        data: b.data,
        dims: &bd,
        strides: &bs,
        offset: 0,
        labels: &bl,
        conj: bc,
    };
    let rc = c
        .zip(cc.as_ref())
        .map(|(c, (d, s, l, cj))| oracle::RefOperand {
            data: c.data,
            dims: d,
            strides: s,
            offset: 0,
            labels: l,
            conj: *cj,
        });
    let (dd, ds) = (d_layout.dims(), d_layout.strides_isize());
    oracle::contract_reference(
        alpha,
        &ra,
        &rb,
        beta,
        rc.as_ref(),
        &mut oracle::RefOutput {
            data: d_data,
            dims: &dd,
            strides: &ds,
            offset: 0,
            labels: idx_d,
            conj: op_d.is_conj(),
        },
    )
}

impl Plan {
    /// The packed plan (the analysis does not depend on the element type, but
    /// the problem is rebuilt for `T` so the dtype matches the resolution).
    pub fn packed_for<T: Scalar>(&self) -> PackedPlan {
        let problem = self.problem_for(T::STORAGE).expect("validated");
        PackedPlan::from_problem(&problem, &self.cfg).expect("role products fit")
    }
}

/// Run the packed driver with an already resolved family.
///
/// # Safety
///
/// As [`Plan::run_raw`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn execute_resolved<T: Scalar>(
    plan: &Plan,
    rg: &ResolvedGemm<<T as Scalar>::Re>,
    exec: &mut Exec<'_>,
    workspace: Option<&dyn WorkspaceProvider>,
    alpha: T,
    a: *const T,
    b: *const T,
    beta: T,
    c: *const T,
    d: *mut T,
) -> std::result::Result<(), tcontract_exec::ExecError> {
    // SAFETY: forwarded.
    unsafe {
        driver::execute_packed(
            &plan.packed_for::<T>(),
            rg,
            exec,
            workspace,
            alpha,
            a,
            b,
            beta,
            c,
            d,
        )
    }
}

/// [`execute_resolved`] with dynamic-tile counters.
///
/// # Safety
///
/// As [`Plan::run_raw`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn execute_resolved_instrumented<T: Scalar>(
    plan: &Plan,
    rg: &ResolvedGemm<<T as Scalar>::Re>,
    exec: &mut Exec<'_>,
    workspace: Option<&dyn WorkspaceProvider>,
    stats: &driver::DynStats,
    alpha: T,
    a: *const T,
    b: *const T,
    beta: T,
    c: *const T,
    d: *mut T,
) -> std::result::Result<(), tcontract_exec::ExecError> {
    // SAFETY: forwarded.
    unsafe {
        driver::execute_packed_instrumented(
            &plan.packed_for::<T>(),
            rg,
            exec,
            workspace,
            stats,
            alpha,
            a,
            b,
            beta,
            c,
            d,
        )
    }
}

/// The dynamic assignment `plan` would use at `width` with `rg`.
pub fn dynamic_report<R: tcontract_kernel::Real>(
    plan: &Plan,
    rg: &ResolvedGemm<R>,
    width: usize,
) -> Option<driver::DynamicReport> {
    driver::dynamic_report(&plan.packed, rg, width)
}
