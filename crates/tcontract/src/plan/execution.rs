//! Safe buffer boundaries and observable execution routes.

use core::mem::MaybeUninit;

use super::{Plan, Strategy};
use crate::api::{CSpec, LayoutError, OperandId, Result, Scalar, ShapeError, Unsupported};
use crate::strategy::elementwise::CRead;
use tcontract_exec::{Exec, WorkspaceReq};
use tcontract_kernel::Element;

/// Failure to initialize and publish a complete destination.
///
/// Errors never contain destination storage. See [`execute_uninit_with`] for
/// the initialization contract and an executable example.
///
/// # Examples
/// ```
/// use tcontract::UninitOutputError;
/// let error = UninitOutputError::Execution("kernel failed");
/// assert!(matches!(error, UninitOutputError::Execution("kernel failed")));
/// ```
#[derive(Debug, thiserror::Error)]
pub enum UninitOutputError<E> {
    /// The callback failed; some destination slots may have been written.
    #[error("fresh output execution failed: {0}")]
    Execution(#[source] E),
    /// Initialized sub-slices do not exactly cover the original destination.
    #[error("initialized output slices must exactly cover the destination")]
    Coverage,
}

/// Execute a fresh-output operation and publish its complete initialized slice.
///
/// `execute` receives only `MaybeUninit` storage. It must return initialized
/// destination sub-slices obtained from safe full-write operations, such as
/// [`Plan::execute_uninit_slices`], plus an arbitrary result (for example a
/// scheduler route). Sub-slices may arrive in any order; empty slices are ignored.
/// This helper neither fills nor copies the destination.
///
/// # Errors
///
/// Returns [`UninitOutputError::Execution`] when the callback fails, preserving
/// its error, or [`UninitOutputError::Coverage`] when the returned initialized
/// slices leave holes, overlap, or name other storage. Both errors contain no
/// output slice. Callback writes are not rolled back. Callers that promise
/// unchanged output on invalid geometry, beta, bounds or routes must validate
/// those conditions **before** invoking a writing callback.
///
/// # Safety
///
/// This is a safe API: documented promises alone cannot establish initialization.
/// Valid initialized mutable sub-slices, checked to cover exactly the original
/// allocation, prove that every destination slot holds a valid `T` on success.
/// No initialized reference to unwritten storage is constructed. The higher-ranked
/// callback borrow prevents destination references escaping through its result
/// or error. Panic unwinding also publishes no initialized destination.
///
/// # Examples
/// ```
/// use core::mem::MaybeUninit;
/// use tcontract::execute_uninit_with;
/// let mut storage = [MaybeUninit::<f64>::uninit(); 3];
/// let (out, route) = execute_uninit_with(&mut storage, |slots| {
///     let (left, right) = slots.split_at_mut(1);
///     // Safe writes stand in for independent full-write kernels.
///     let left = core::slice::from_mut(left[0].write(2.0));
///     let (r0, r1) = right.split_at_mut(1);
///     let r0 = core::slice::from_mut(r0[0].write(3.0));
///     let r1 = core::slice::from_mut(r1[0].write(4.0));
///     Ok::<_, core::convert::Infallible>((vec![r1, left, r0], "serial"))
/// })?;
/// assert_eq!(out, &[2.0, 3.0, 4.0]);
/// assert_eq!(route, "serial");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
/// Destination storage cannot escape through the auxiliary result:
/// ```compile_fail
/// use core::mem::MaybeUninit;
/// use tcontract::execute_uninit_with;
/// let mut storage = [MaybeUninit::<f64>::uninit()];
/// let _ = execute_uninit_with(&mut storage, |slots| {
///     Ok::<_, ()>((Vec::new(), slots))
/// });
/// ```
/// Nor through a callback error:
/// ```compile_fail
/// use core::mem::MaybeUninit;
/// use tcontract::execute_uninit_with;
/// let mut storage = [MaybeUninit::<f64>::uninit()];
/// let _ = execute_uninit_with(&mut storage, |slots| {
///     Err::<(Vec<&mut [f64]>, ()), _>(slots)
/// });
/// ```
pub fn execute_uninit_with<'d, T: Scalar, R, E>(
    out: &'d mut [MaybeUninit<T>],
    execute: impl for<'a> FnOnce(
        &'a mut [MaybeUninit<T>],
    ) -> core::result::Result<(Vec<&'a mut [T]>, R), E>,
) -> core::result::Result<(&'d mut [T], R), UninitOutputError<E>> {
    let ptr = out.as_mut_ptr().cast::<T>();
    let len = out.len();
    let (mut initialized, result) = execute(out).map_err(UninitOutputError::Execution)?;
    initialized.sort_unstable_by_key(|slice| slice.as_ptr());
    let mut covered = 0;
    for slice in &initialized {
        if slice.is_empty() {
            continue;
        }
        if slice.len() > len - covered || slice.as_ptr() != ptr.wrapping_add(covered) {
            return Err(UninitOutputError::Coverage);
        }
        covered += slice.len();
    }
    if covered != len {
        return Err(UninitOutputError::Coverage);
    }
    drop(initialized);
    // Reborrow after the callback: its mutable borrows invalidate the old tag.
    let ptr = out.as_mut_ptr().cast::<T>();
    // SAFETY: valid initialized sub-slices proved exact, nonoverlapping coverage
    // above. Their borrows have ended; this pointer is freshly derived from the
    // original unique output, including for an empty output.
    let out = unsafe { core::slice::from_raw_parts_mut(ptr, len) };
    Ok((out, result))
}

/// The storage guarantee of a destination, not a provider selection.
///
/// # Examples
/// ```
/// use tcontract::OutputContract;
/// assert_ne!(OutputContract::Initialized, OutputContract::Fresh);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputContract {
    /// Every destination value is already initialized.
    Initialized,
    /// No old destination value may be read or referenced before writing.
    Fresh,
}

/// A packed call's scheduling shape, chosen before writes.
///
/// # Examples
/// ```
/// use tcontract::PackedRoute;
/// assert_ne!(PackedRoute::Serial, PackedRoute::Spmd);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedRoute {
    /// One caller thread.
    Serial,
    /// Barrier-free lanes over the contraction's batch axis.
    BatchLanes,
    /// Barrier-free output cells.
    Cells,
    /// A co-scheduled, barrier-bearing team.
    Spmd,
}

/// The chosen numerical route. Packed widths are active widths, not budgets.
/// Faer and elementwise routes do not expose their internal partition width.
///
/// # Examples
/// ```
/// use tcontract::{ExecutionRoute, PackedRoute};
/// let route = ExecutionRoute::Packed { kind: PackedRoute::Serial, width: 1 };
/// assert_ne!(route, ExecutionRoute::Empty);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExecutionRoute {
    /// No output elements.
    Empty,
    /// Alpha zero or empty K; includes an in-place identity no-op.
    OutputOnly,
    /// Prepared elementwise product.
    Elementwise,
    /// Prepared copy-free Faer contraction, potentially with a C output pass.
    Faer,
    /// Prepared packed contraction and its exact scheduling width.
    Packed {
        /// Scheduling shape.
        kind: PackedRoute,
        /// Active participants (batch lanes for BatchLanes).
        width: usize,
    },
}

/// A slice-based accumulation source with runtime element origins.
///
/// # Examples
/// ```
/// use tcontract::SliceAccumulationSource;
/// let c = [3.0_f64];
/// let source = SliceAccumulationSource::Separate((&c, 0));
/// assert!(matches!(source, SliceAccumulationSource::Separate(_)));
/// ```
#[derive(Clone, Copy, Debug)]
pub enum SliceAccumulationSource<'a, T> {
    /// No C term; requires beta zero.
    Absent,
    /// Previous destination values; requires a prepared Output C mode.
    Output,
    /// Separate initialized C and its logical origin.
    Separate((&'a [T], isize)),
}

impl<T: Scalar> Plan<T> {
    /// A conservative upper bound on the workspace one execution through this
    /// plan needs, for a caller that owns a
    /// [`Workspace`](tcontract_exec::Workspace) and runs
    /// [`Plan::execute_raw_in`] (#121).
    ///
    /// It is computed from the same packed geometry the driver resolves, widened
    /// to the widest grid and lane count. A Faer plan's fresh-output alternative
    /// is included; an elementwise route and an initialized Faer route take no
    /// workspace, so their request is zero. A `Workspace` sized to this bound and
    /// used through `execute_raw_in` grows the arena for none of the buffers.
    ///
    /// # Examples
    /// ```
    /// use tcontract::{OutputContract, Plan, PlanConfig};
    /// use tcontract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
    /// use tcontract_exec::Exec;
    /// let l = OperandSpec::new(LayoutSpec::new(&[1], &[1], 0)?);
    /// let p = Problem::from_labels(DType::F64, l.clone(), l.clone(), CSpec::Absent,
    ///     l, &Labels::new(&[0], &[0], &[0]))?;
    /// // The packed driver owns a workspace; a fusable plan runs on faer and
    /// // needs none.
    /// let plan = Plan::<f64>::new(&p, &PlanConfig::packed())?;
    /// let req = plan.workspace_req(&mut Exec::serial(), OutputContract::Initialized);
    /// assert!(req.a_bytes > 0);
    /// # Ok::<(), tcontract::Error>(())
    /// ```
    pub fn workspace_req(&self, exec: &Exec<'_>, output: OutputContract) -> WorkspaceReq {
        let packed = match (&self.strategy, output) {
            (Strategy::Packed(pk), _) => Some(pk.as_ref()),
            _ => None,
        };
        match packed {
            Some(pk) => crate::driver::call_workspace_req::<T>(&pk.plan, &pk.rg, exec),
            None => WorkspaceReq::default(),
        }
    }

    /// Resolve the numerical route on the current calling thread without
    /// taking a workspace or writing. This is not a reservation. Buffer and
    /// C-mode validation belongs to execution; fresh storage cannot accumulate
    /// from the previous destination. Requery if caller/executor changes.
    ///
    /// The route is the one a root token gets: when the token cannot
    /// co-schedule a barrier team (it was produced by `fork`, or the caller is
    /// a worker of the pool), execution takes the barrier-free cells route
    /// instead; see [`PackedRoute`].
    ///
    /// # Errors
    /// Returns `Error::Exec` when a forced batch strategy cannot be honoured by
    /// the packed route, before any write.
    ///
    /// # Examples
    /// ```
    /// use tcontract::{ExecutionRoute, OutputContract, Plan, PlanConfig};
    /// use tcontract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
    /// use tcontract_exec::Exec;
    /// let l = OperandSpec::new(LayoutSpec::new(&[1], &[1], 0)?);
    /// let p = Problem::from_labels(DType::F64, l.clone(), l.clone(), CSpec::Absent,
    ///     l, &Labels::new(&[0], &[0], &[0]))?;
    /// let plan = Plan::<f64>::new(&p, &PlanConfig::default())?;
    /// assert_eq!(plan.execution_route(&mut Exec::serial(), 0.0, OutputContract::Fresh)?, ExecutionRoute::OutputOnly);
    /// # Ok::<(), tcontract::Error>(())
    /// ```
    pub fn execution_route(
        &self,
        exec: &Exec<'_>,
        alpha: T,
        output: OutputContract,
    ) -> Result<ExecutionRoute> {
        if self.problem.out_empty() {
            return Ok(ExecutionRoute::Empty);
        }
        if self.problem.k_empty() || alpha == <T as Element>::zero() {
            return Ok(ExecutionRoute::OutputOnly);
        }
        // Fresh and initialized outputs take the same route: a fresh Faer
        // output is zero-filled first, which needs no workspace.
        let pk = match (&self.strategy, output) {
            (Strategy::Packed(pk), _) => pk.as_ref(),
            (Strategy::Faer(_), _) => return Ok(ExecutionRoute::Faer),
            (Strategy::Elementwise(_), _) => return Ok(ExecutionRoute::Elementwise),
        };
        let geometry = crate::driver::execution_geometry::<T>(
            &pk.plan,
            &pk.rg,
            exec,
            self.batch_budget(exec),
            exec.batch_policy(),
            true,
        )?;
        Ok(ExecutionRoute::Packed {
            kind: geometry.kind(),
            width: geometry.width(),
        })
    }

    pub(crate) fn validate_c_beta(&self, beta: T, source: CRead<T>) -> Result<()> {
        if beta != <T as Element>::zero()
            && (matches!(self.problem.c_spec(), CSpec::Absent) || matches!(source, CRead::None))
        {
            return Err(LayoutError::CMode.into());
        }
        Ok(())
    }

    fn slice_c(&self, beta: T, source: SliceAccumulationSource<'_, T>) -> Result<CRead<T>> {
        let c = match (self.problem.c_spec(), source) {
            (_, SliceAccumulationSource::Absent) => CRead::None,
            (CSpec::Output(_), SliceAccumulationSource::Output) => CRead::InPlace,
            (CSpec::Separate(_), SliceAccumulationSource::Separate((c, origin))) => {
                CRead::Separate(self.slice_origin(OperandId::C, c.as_ptr(), c.len(), origin)?)
            }
            _ => return Err(LayoutError::CMode.into()),
        };
        self.validate_c_beta(beta, c)?;
        Ok(c)
    }

    /// Execute an initialized slice update using the prepared layouts.
    /// No per-operand view metadata is constructed. Explicit source metadata
    /// is validated even when beta is zero; zero beta reads no C value.
    ///
    /// # Errors
    /// `Error::Layout` for bounds/C-mode errors and `Error::Exec` for unavailable
    /// routes, before any write. Backend failure may partially write D.
    ///
    /// # Examples
    /// ```
    /// use tcontract::{Plan, PlanConfig, SliceAccumulationSource};
    /// use tcontract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
    /// use tcontract_exec::Exec;
    /// let l = OperandSpec::new(LayoutSpec::new(&[1], &[1], 0)?);
    /// let p = Problem::from_labels(DType::F64, l.clone(), l.clone(), CSpec::Absent,
    ///     l, &Labels::new(&[0], &[0], &[0]))?;
    /// let plan = Plan::<f64>::new(&p, &PlanConfig::default())?;
    /// let mut d = [0.0];
    /// plan.execute_slices_accum(&mut Exec::serial(), 2.0, (&[3.0], 0), (&[4.0], 0),
    ///     0.0, SliceAccumulationSource::Absent, (&mut d, 0))?;
    /// assert_eq!(d, [24.0]);
    /// # Ok::<(), tcontract::Error>(())
    /// ```
    pub fn execute_slices_accum(
        &self,
        exec: &mut Exec<'_>,
        alpha: T,
        (a, ao): (&[T], isize),
        (b, bo): (&[T], isize),
        beta: T,
        source: SliceAccumulationSource<'_, T>,
        (d, do_): (&mut [T], isize),
    ) -> Result<()> {
        let a = self.slice_origin(OperandId::A, a.as_ptr(), a.len(), ao)?;
        let b = self.slice_origin(OperandId::B, b.as_ptr(), b.len(), bo)?;
        let dp = self.slice_origin(OperandId::D, d.as_mut_ptr(), d.len(), do_)?;
        let c = self.slice_c(beta, source)?;
        self.execution_route(exec, alpha, OutputContract::Initialized)?;
        // SAFETY: validated slice spans and modes; the unique mutable D borrow
        // excludes aliasing inputs. D values are initialized.
        unsafe { self.run(exec, alpha, a, b, beta, c, dp.cast_mut()) }
    }

    /// Execute into fresh storage and return initialized values only on success.
    /// D must have dense exact physical coverage (no holes or extra backing
    /// elements). Reduced output labels, not original axis products, prove
    /// coverage. Negative strides and dense permutations are supported.
    /// Output/nonzero beta is forbidden; Separate C may be read instead.
    ///
    /// # Errors
    /// `Error::Layout` for bounds/C-mode errors, `Error::Unsupported` for
    /// incomplete physical coverage, `Error::Shape` on count overflow, and
    /// `Error::Exec` for unavailable execution, all before writes. Backend
    /// failures may leave some slots written but expose no initialized slice.
    ///
    /// # Examples
    /// ```
    /// use core::mem::MaybeUninit;
    /// use tcontract::{Plan, PlanConfig, SliceAccumulationSource};
    /// use tcontract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
    /// use tcontract_exec::Exec;
    /// let l = OperandSpec::new(LayoutSpec::new(&[1], &[1], 0)?);
    /// let p = Problem::from_labels(DType::F64, l.clone(), l.clone(), CSpec::Absent,
    ///     l, &Labels::new(&[0], &[0], &[0]))?;
    /// let plan = Plan::<f64>::new(&p, &PlanConfig::default())?;
    /// let mut d = [MaybeUninit::uninit()];
    /// let initialized = plan.execute_uninit_slices(&mut Exec::serial(), 1.0,
    ///     (&[3.0], 0), (&[4.0], 0), 0.0, SliceAccumulationSource::Absent,
    ///     (&mut d, 0))?;
    /// assert_eq!(initialized, &[12.0]);
    /// # Ok::<(), tcontract::Error>(())
    /// ```
    pub fn execute_uninit_slices<'d>(
        &self,
        exec: &mut Exec<'_>,
        alpha: T,
        (a, ao): (&[T], isize),
        (b, bo): (&[T], isize),
        beta: T,
        source: SliceAccumulationSource<'_, T>,
        (d, do_): (&'d mut [MaybeUninit<T>], isize),
    ) -> Result<&'d mut [T]> {
        let a = self.slice_origin(OperandId::A, a.as_ptr(), a.len(), ao)?;
        let b = self.slice_origin(OperandId::B, b.as_ptr(), b.len(), bo)?;
        let dp = self.slice_origin(OperandId::D, d.as_mut_ptr().cast::<T>(), d.len(), do_)?;
        let c = self.slice_c(beta, source)?;
        if beta != <T as Element>::zero() && matches!(c, CRead::InPlace) {
            return Err(LayoutError::CMode.into());
        }
        self.check_fresh_coverage(d.len(), do_)?;
        self.execution_route(exec, alpha, OutputContract::Fresh)?;
        // SAFETY: validated slices; old D is never required. The prepared fresh
        // strategy writes without forming initialized references first.
        unsafe {
            self.run_storage(exec, alpha, a, b, beta, c, dp.cast_mut(), true)?;
        }
        // SAFETY: check_fresh_coverage proves every slot is written once over
        // the reduced injective output domain; successful run proves completion.
        Ok(unsafe { core::slice::from_raw_parts_mut(d.as_mut_ptr().cast::<T>(), d.len()) })
    }

    fn check_fresh_coverage(&self, len: usize, origin: isize) -> Result<()> {
        let roles = self.problem.roles();
        let count = if self.problem.out_empty() {
            0
        } else {
            roles
                .m()
                .iter()
                .chain(roles.n())
                .chain(roles.h())
                .try_fold(1usize, |n, axis| {
                    n.checked_mul(axis.extent()).ok_or(ShapeError::Overflow {
                        what: "fresh output count",
                    })
                })?
        };
        let full_span = match self.problem.span(OperandId::D) {
            None => len == 0,
            Some(span) => {
                let shift = origin as i128 - self.problem.d().layout().offset() as i128;
                span.lo() + shift == 0 && span.hi() + shift + 1 == len as i128
            }
        };
        if count != len || !full_span {
            return Err(Unsupported::Reason("fresh output needs dense exact physical coverage; use initialized output for padding or diagonal-only writes").into());
        }
        Ok(())
    }
}
