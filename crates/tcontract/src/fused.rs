//! [`FusedGemmPlan`]: a grouped GEMM run as one packed traversal
//! (tensor4all/tcontract-rs#208).
//!
//! The input is the device-neutral [`GemmBatch`] of `t4a-desc`: independent
//! column-major jobs of different shapes, at element offsets into one buffer
//! per operand, with BLAS ops N/T/C per call. Every job's work items go into one
//! list, and the workers claim them from one counter. This is the "fused" route
//! of the two-route design in tensor4all-meta
//! `docs/design/symmetric-tensors-and-symtn.md` §3.1, and the CPU counterpart of
//! CUTLASS's grouped-GEMM tile scheduler: one index space over all problems,
//! each output tile owned by one worker for the whole of K, ops and scalars
//! uniform per call. The other route, one [`Plan`] per job on lanes over the
//! jobs, is [`crate::contract_batched`] (and cpueinsum's `GroupedPlan`); choosing
//! between the two is not this crate's decision.

use core::mem::MaybeUninit;

use t4a_desc::gemm::{GemmBatch, GemmJob, GemmShape, Trans};
use t4a_desc::ScalarKind;
use tcontract_exec::{BatchQuery, Exec, ExecError, WorkspaceReq};
use tcontract_kernel::Element;

use crate::api::{
    CSpec, ConfigError, DType, Error, Labels, LayoutSpec, Op, OperandSpec, Problem, Result, Scalar,
};
use crate::driver::fused::{traverse, FusedGroup, FusedItem};
use crate::{ExecutionRoute, Partition, Plan, PlanConfig};

/// One distinct geometry: the job alone as a forced-packed [`Plan`], and the
/// driver's serial geometry for it.
struct Group<T: Scalar> {
    plan: Plan<T>,
    geo: FusedGroup,
}

/// A validated grouped-GEMM job list, prepared once for one fused packed
/// traversal: every job's work items in one list that the workers claim from
/// one counter.
///
/// Each job computes `C ← alpha·op(A)·op(B) + beta·C` in place, with BLAS
/// meaning (see [`t4a_desc::gemm`]). Every distinct job geometry is resolved
/// (so a kernel or configuration error is reported for any of them); the plan
/// owns the [`GemmBatch`], the packed-driver resolutions of the geometries that
/// take work items (`m, n, k ≥ 1`), and the work list. It holds no buffer, and
/// the buffers are lent per call as slices.
///
/// # Prepare
///
/// [`FusedGemmPlan::new`] resolves, for every distinct geometry
/// ([`GemmBatch::groups`]), exactly what [`Plan::new`] resolves for that job
/// alone when the packed driver is forced, at serial width. Each non-empty job
/// (`m, n, k ≥ 1`) contributes work items `(job, row band, column panel)` of its
/// group's `MC × NC` blocking (a job that fits one block is one item). The list
/// is ordered by multiply-adds, largest first, ties by `(job, band, panel)`, so
/// the dynamic claim approximates longest-processing-time-first and the order
/// is deterministic.
///
/// # Results
///
/// An item runs all its `KC` slabs in order on one worker, so K is never split
/// and nothing accumulates atomically. Each job's result is bitwise the result
/// of that job alone through [`Plan`] with the packed driver forced, run on one
/// thread, and the result is bitwise the same for every executor width.
///
/// # Configuration
///
/// `kernel`, `isa`, `method`, `blocking`, `cache_model`, `orientation`,
/// `row_block` and `writeback` apply as in [`Plan`]. `faer_limit`,
/// `no_materialize` and `fresh_output` have no effect, because the route is
/// packed only. `partition` must be `None`: the route has its own claim.
///
/// # Examples
///
/// ```
/// use t4a_desc::gemm::{GemmBatch, GemmJob, GemmOps, Trans};
/// use t4a_desc::ScalarKind;
/// use tcontract::{ExecutionRoute, FusedGemmPlan, PlanConfig};
/// use tcontract_exec::Exec;
///
/// let ops = GemmOps { a: Trans::N, b: Trans::N };
/// // A 2×2 product at offset 0 and a 1×1 product after it, in shared buffers.
/// let jobs = vec![
///     GemmJob::compact(ops, [0, 0, 0], 2, 2, 2),
///     GemmJob::compact(ops, [4, 4, 4], 1, 1, 1),
/// ];
/// let batch = GemmBatch::new(ScalarKind::F64, ops, jobs)?;
/// let plan = FusedGemmPlan::<f64>::new(batch, &PlanConfig::default())?;
/// let a = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let b = [1.0, 0.0, 0.0, 1.0, 2.0]; // the identity, then 2
/// let mut c = [0.0; 5];
/// let route = plan.execute_into(&mut Exec::serial(), 1.0, &a, &b, &mut c)?;
/// assert_eq!(c, [1.0, 2.0, 3.0, 4.0, 10.0]);
/// assert_eq!(route, ExecutionRoute::FusedTiles { width: 1 });
/// # Ok::<(), tcontract::Error>(())
/// ```
pub struct FusedGemmPlan<T: Scalar> {
    batch: GemmBatch,
    groups: Vec<Group<T>>,
    items: Vec<FusedItem>,
    /// Non-empty outputs with `k == 0`: on the fused route they take the
    /// output-only update (`C ← beta·C`) instead of items.
    k_empty: Vec<usize>,
    /// Whether any job has a non-empty output.
    has_output: bool,
    /// Multiply-adds of all items.
    muladds: u128,
    /// Reals of one worker's `(A + B region, tile region)`: the largest group's.
    worker: (usize, usize),
}

impl<T: Scalar> core::fmt::Debug for FusedGemmPlan<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FusedGemmPlan")
            .field("scalar", &self.batch.scalar())
            .field("ops", &self.batch.ops())
            .field("jobs", &self.batch.jobs().len())
            .field("groups", &self.groups.len())
            .field("work_items", &self.items.len())
            .finish_non_exhaustive()
    }
}

impl<T: Scalar> FusedGemmPlan<T> {
    /// Prepare `batch` under `config`: the dtype check, one packed resolution
    /// per distinct geometry, and the work list. Writes nothing and needs no
    /// buffer.
    ///
    /// # Errors
    ///
    /// [`ConfigError::DtypeMismatch`] when `T` is not `batch.scalar()`, any
    /// [`PlanConfig::validate`] error, [`ConfigError::Option`] for
    /// `partition: Some(_)`, and the first group's [`Plan::new`] error, in
    /// order of the groups' lowest job id ([`Error::Select`] for an unusable
    /// kernel request).
    pub fn new(batch: GemmBatch, config: &PlanConfig) -> Result<Self> {
        let dtype = dtype_of(batch.scalar());
        if dtype != T::STORAGE {
            return Err(ConfigError::DtypeMismatch {
                plan: T::STORAGE.name(),
                problem: dtype.name(),
            }
            .into());
        }
        config.validate()?;
        if config.partition.is_some() {
            return Err(ConfigError::Option(
                "the fused grouped-GEMM route has its own claim; partition must be None",
            )
            .into());
        }
        // Exactly the per-job reference: the packed driver forced, the planner's
        // own grid (unused at one participant).
        let group_config = PlanConfig {
            partition: Some(Partition::StaticGrid {
                pin: None,
                align_c_lines: false,
            }),
            ..config.clone()
        };
        let ops = batch.ops();
        let mut groups = Vec::new();
        let mut costed = Vec::new();
        let mut k_empty = Vec::new();
        let mut has_output = false;
        let mut worker = (0, 0);
        for group in batch.groups() {
            let s = group.shape;
            // Every geometry is planned, including those that take no item, so a
            // kernel or configuration error is reported as the per-job `Plan`
            // reports it (zero-size problems are selected and validated too)
            // instead of being skipped on a no-op path.
            let plan = Plan::<T>::from_problem(job_problem(dtype, ops, s)?, &group_config)?;
            if s.m == 0 || s.n == 0 {
                continue;
            }
            has_output = true;
            if s.k == 0 {
                k_empty.extend_from_slice(group.ids);
                continue;
            }
            let Some((packed, rg)) = plan.packed_parts() else {
                return Err(Error::Internal(
                    "a forced-packed plan did not resolve to the packed driver",
                ));
            };
            let geo = FusedGroup::new::<T>(packed, rg);
            let (ab, tile) = geo.worker_reals();
            worker = (worker.0.max(ab), worker.1.max(tile));
            let (bands, panels) = geo.grid();
            let index = groups.len();
            for &job in group.ids {
                for band in 0..bands {
                    for panel in 0..panels {
                        let item = FusedItem {
                            job,
                            group: index,
                            band,
                            panel,
                        };
                        costed.push((geo.item_muladds(band, panel), item));
                    }
                }
            }
            groups.push(Group { plan, geo });
        }
        k_empty.sort_unstable();
        costed.sort_unstable_by(|(x, i), (y, j)| {
            y.cmp(x)
                .then(i.job.cmp(&j.job))
                .then(i.band.cmp(&j.band))
                .then(i.panel.cmp(&j.panel))
        });
        let muladds = costed.iter().map(|&(w, _)| w).sum();
        Ok(Self {
            batch,
            groups,
            items: costed.into_iter().map(|(_, item)| item).collect(),
            k_empty,
            has_output,
            muladds,
            worker,
        })
    }

    /// The batch the plan was prepared from: job ids, required buffer lengths,
    /// groups.
    pub fn batch(&self) -> &GemmBatch {
        &self.batch
    }

    /// Number of work items `(job, row band, column panel)` in the claim list.
    ///
    /// # Examples
    ///
    /// ```
    /// use t4a_desc::gemm::{GemmBatch, GemmJob, GemmOps, Trans};
    /// use t4a_desc::ScalarKind;
    /// use tcontract::{FusedGemmPlan, PlanConfig};
    ///
    /// let ops = GemmOps { a: Trans::N, b: Trans::N };
    /// let jobs = (0..3).map(|i| GemmJob::compact(ops, [4 * i; 3], 2, 2, 2)).collect();
    /// let batch = GemmBatch::new(ScalarKind::F32, ops, jobs)?;
    /// let plan = FusedGemmPlan::<f32>::new(batch, &PlanConfig::default())?;
    /// assert_eq!(plan.work_items(), 3); // each small job is one block
    /// # Ok::<(), tcontract::Error>(())
    /// ```
    pub fn work_items(&self) -> usize {
        self.items.len()
    }

    /// The route a call with this `alpha` takes on `exec`, decided as the
    /// execute entries decide it, before any write.
    ///
    /// [`ExecutionRoute::Empty`] when no job has an output element;
    /// [`ExecutionRoute::OutputOnly`] when `alpha == 0` or every non-empty job
    /// has `k == 0` (an output-sized `C ← beta·C`, reading no `A` or `B`);
    /// otherwise [`ExecutionRoute::FusedTiles`]. Its width is `exec`'s batch
    /// policy applied with the work items as the items, which cannot use
    /// threads themselves: `Auto` keeps a small call on the calling thread,
    /// `Sequential` is 1 and `Outer` is `min(budget, items)`.
    ///
    /// # Errors
    ///
    /// [`ExecError::StrategyUnavailable`] for a policy that asks for threads
    /// inside an item (`Inner` on a budget above one, or a `Split` that leaves
    /// more than one thread per lane), which this
    /// route cannot honour and does not silently serialize.
    pub fn execution_route(&self, exec: &Exec<'_>, alpha: T) -> Result<ExecutionRoute> {
        if !self.has_output {
            return Ok(ExecutionRoute::Empty);
        }
        if alpha == T::zero() || self.items.is_empty() {
            return Ok(ExecutionRoute::OutputOnly);
        }
        let per_item = self.muladds as f64 / self.items.len() as f64;
        let per_item = per_item * if T::STORAGE.is_complex() { 4.0 } else { 1.0 };
        let query = BatchQuery::new(self.items.len(), per_item).with_item_parallel(false);
        let decision = exec.batch_decision(&query);
        if decision.item_threads > 1 {
            return Err(ExecError::StrategyUnavailable {
                strategy: exec.batch_policy().strategy,
            }
            .into());
        }
        Ok(ExecutionRoute::FusedTiles {
            width: decision.lanes.clamp(1, self.items.len()),
        })
    }

    /// The workspace one worker of a call needs: its packed `A` and `B` and its
    /// tile, at the largest group's blocking. A call takes one such set per
    /// worker from the [`Exec`]'s workspace, or call-local buffers without one.
    pub fn workspace_req(&self) -> WorkspaceReq {
        let element = core::mem::size_of::<T::Re>();
        WorkspaceReq {
            a_bytes: self.worker.0 * element,
            tile_bytes: self.worker.1 * element,
            ..Default::default()
        }
    }

    /// `C ← alpha·op(A)·op(B)` for every job; no previous `C` value is read.
    ///
    /// Offsets are element offsets from the start of each slice. Everything is
    /// checked before the first write.
    ///
    /// # Errors
    ///
    /// [`Error::Gemm`] with [`GemmError::Bounds`](t4a_desc::gemm::GemmError::Bounds)
    /// naming the lowest job whose footprint the first short slice (checked
    /// in the order `A`, `B`, `C`) does not cover, and the
    /// [`FusedGemmPlan::execution_route`] errors; nothing is written.
    pub fn execute_into(
        &self,
        exec: &mut Exec<'_>,
        alpha: T,
        a: &[T],
        b: &[T],
        c: &mut [T],
    ) -> Result<ExecutionRoute> {
        self.batch.check_lens(a.len(), b.len(), c.len())?;
        let route = self.execution_route(exec, alpha)?;
        self.run(exec, route, alpha, a, b, T::zero(), Out::Init(c));
        Ok(route)
    }

    /// `C ← alpha·op(A)·op(B) + beta·C` for every job (cuBLAS semantics).
    ///
    /// `beta == 0` reads no `C` value; `alpha == 0` or `k == 0` reads no `A` or
    /// `B` value.
    ///
    /// # Errors
    ///
    /// As [`FusedGemmPlan::execute_into`].
    pub fn execute_into_accum(
        &self,
        exec: &mut Exec<'_>,
        alpha: T,
        a: &[T],
        b: &[T],
        beta: T,
        c: &mut [T],
    ) -> Result<ExecutionRoute> {
        self.batch.check_lens(a.len(), b.len(), c.len())?;
        let route = self.execution_route(exec, alpha)?;
        self.run(exec, route, alpha, a, b, beta, Out::Init(c));
        Ok(route)
    }

    /// Overwrite into uninitialized storage: `C ← alpha·op(A)·op(B)`. The
    /// non-empty outputs must tile `c` exactly with `ldc == m`
    /// ([`GemmBatch::check_fresh_cover`]), so every element is written; the
    /// returned slice is the whole `c`, and it exists only on success.
    ///
    /// There is no `beta`: no old value exists to scale.
    ///
    /// # Errors
    ///
    /// [`Error::Gemm`] with
    /// [`GemmError::FreshLayout`](t4a_desc::gemm::GemmError::FreshLayout) or
    /// [`GemmError::FreshCoverage`](t4a_desc::gemm::GemmError::FreshCoverage),
    /// then as [`FusedGemmPlan::execute_into`]; nothing is written.
    ///
    /// # Examples
    ///
    /// ```
    /// use core::mem::MaybeUninit;
    /// use t4a_desc::gemm::{GemmBatch, GemmJob, GemmOps, Trans};
    /// use t4a_desc::ScalarKind;
    /// use tcontract::{FusedGemmPlan, PlanConfig};
    /// use tcontract_exec::Exec;
    ///
    /// let ops = GemmOps { a: Trans::T, b: Trans::N };
    /// let jobs = vec![GemmJob::compact(ops, [0, 0, 0], 1, 2, 2)];
    /// let plan = FusedGemmPlan::<f64>::new(
    ///     GemmBatch::new(ScalarKind::F64, ops, jobs)?,
    ///     &PlanConfig::default(),
    /// )?;
    /// let mut c = [MaybeUninit::<f64>::uninit(); 2];
    /// let (c, _) = plan.execute_uninit(&mut Exec::serial(), 1.0, &[1.0, 2.0], &[3.0, 4.0, 5.0, 6.0], &mut c)?;
    /// assert_eq!(c, [11.0, 17.0]);
    /// # Ok::<(), tcontract::Error>(())
    /// ```
    pub fn execute_uninit<'c>(
        &self,
        exec: &mut Exec<'_>,
        alpha: T,
        a: &[T],
        b: &[T],
        c: &'c mut [MaybeUninit<T>],
    ) -> Result<(&'c mut [T], ExecutionRoute)> {
        self.batch.check_fresh_cover(c.len())?;
        self.batch.check_lens(a.len(), b.len(), c.len())?;
        let route = self.execution_route(exec, alpha)?;
        self.run(exec, route, alpha, a, b, T::zero(), Out::Fresh(&mut *c));
        // SAFETY: `check_fresh_cover` proved that the non-empty outputs tile
        // `c` exactly with `ldc == m`, and the run above wrote every element of
        // every non-empty output (the product on the fused route, zero on the
        // output-only pass; an empty `c` has nothing to write).
        let c = unsafe { &mut *(c as *mut [MaybeUninit<T>] as *mut [T]) };
        Ok((c, route))
    }

    /// One call after its checks: the output-only pass for the jobs that take
    /// it, then the fused traversal.
    #[allow(clippy::too_many_arguments)] // INVARIANT: the GEMM argument set plus the route.
    fn run(
        &self,
        exec: &mut Exec<'_>,
        route: ExecutionRoute,
        alpha: T,
        a: &[T],
        b: &[T],
        beta: T,
        mut c: Out<'_, T>,
    ) {
        let jobs = self.batch.jobs();
        match route {
            ExecutionRoute::FusedTiles { width } => {
                for &id in &self.k_empty {
                    c.scale(&jobs[id], beta);
                }
                // SAFETY: `check_lens` covered every job's footprint in each
                // slice; `c` is an exclusive borrow, so it aliases neither `a`
                // nor `b`; `GemmBatch::new` proved the jobs' outputs disjoint and
                // one job's items partition its output; every group was built
                // from its own plan's packed resolution and `self.worker` is the
                // largest `worker_reals`; `execution_route` keeps `width` within
                // the budget.
                unsafe {
                    traverse::<T, _>(
                        exec,
                        width,
                        &self.items,
                        jobs,
                        |g| {
                            let group = &self.groups[g];
                            // INVARIANT: `new` pushed only groups whose plan
                            // resolved to the packed driver.
                            let (plan, rg) = group
                                .plan
                                .packed_parts()
                                .expect("a fused group's plan is packed");
                            (plan, rg, &group.geo)
                        },
                        self.worker,
                        alpha,
                        beta,
                        a.as_ptr(),
                        b.as_ptr(),
                        c.as_mut_ptr(),
                    )
                };
            }
            ExecutionRoute::OutputOnly => {
                for job in jobs.iter().filter(|j| j.m > 0 && j.n > 0) {
                    c.scale(job, beta);
                }
            }
            _ => {}
        }
    }
}

/// The output of one call: initialized, or fresh (write-only).
enum Out<'c, T> {
    Init(&'c mut [T]),
    Fresh(&'c mut [MaybeUninit<T>]),
}

impl<T: Scalar> Out<'_, T> {
    /// `C ← beta·C` on `job`'s output: zeros when `beta == 0` (reading
    /// nothing), unchanged when `beta == 1`. A fresh output is only ever zeroed
    /// (its entries have no `beta`).
    fn scale(&mut self, job: &GemmJob, beta: T) {
        let (zero, one) = (<T as Element>::zero(), <T as Element>::one());
        for col in 0..job.n {
            // INVARIANT: `check_lens` proved `c_offset + (n-1)·ldc + m` within
            // the slice, with checked arithmetic in `GemmBatch::new`.
            let start = job.c_offset + col * job.ldc;
            let range = start..start + job.m;
            match self {
                Out::Init(c) if beta == zero => c[range].fill(zero),
                Out::Init(_) if beta == one => {}
                Out::Init(c) => c[range].iter_mut().for_each(|x| *x = beta * *x),
                Out::Fresh(c) => c[range].iter_mut().for_each(|x| {
                    x.write(zero);
                }),
            }
        }
    }

    fn as_mut_ptr(&mut self) -> *mut T {
        match self {
            Out::Init(c) => c.as_mut_ptr(),
            Out::Fresh(c) => c.as_mut_ptr().cast(),
        }
    }
}

/// The problem one job of geometry `s` is, alone: a 2-D product with the
/// stored operands' strides, `op(A)` `m×k` and `op(B)` `k×n` read through
/// `(1, ld)` for N and `(ld, 1)` for T/C, conjugated for C on a complex dtype,
/// and `C` read from and written to `D` in place.
fn job_problem(dtype: DType, ops: t4a_desc::gemm::GemmOps, s: GemmShape) -> Result<Problem> {
    let operand = |trans: Trans, rows: usize, cols: usize, ld: usize| -> Result<OperandSpec> {
        let ld = isize::try_from(ld).map_err(|_| crate::api::ShapeError::Overflow {
            what: "a leading dimension",
        })?;
        let strides = match trans {
            Trans::N => [1, ld],
            Trans::T | Trans::C => [ld, 1],
        };
        let spec = OperandSpec::new(LayoutSpec::new(&[rows, cols], &strides, 0)?);
        Ok(match trans {
            Trans::C if dtype.is_complex() => spec.conj(),
            _ => spec,
        })
    };
    let a = operand(ops.a, s.m, s.k, s.lda)?;
    let b = operand(ops.b, s.k, s.n, s.ldb)?;
    let d = operand(Trans::N, s.m, s.n, s.ldc)?;
    Problem::from_labels(
        dtype,
        a,
        b,
        CSpec::Output(Op::Identity),
        d,
        &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    )
}

/// The tcontract storage type of a descriptor scalar kind.
const fn dtype_of(kind: ScalarKind) -> DType {
    match kind {
        ScalarKind::F32 => DType::F32,
        ScalarKind::F64 => DType::F64,
        ScalarKind::C32 => DType::C32,
        ScalarKind::C64 => DType::C64,
    }
}
