//! The validated job list and the partitions computed from it once.

use core::fmt;

use super::job::span;
use super::{GemmError, GemmJob, GemmOps, GemmShape, Operand};
use crate::ScalarKind;

/// A validated, immutable grouped-GEMM job list with its partitions.
///
/// [`GemmBatch::new`] checks every job before any consumer writes and stores
/// what the consumers need: the required buffer lengths, the output order that
/// proves the outputs disjoint, the [groups](GemmBatch::groups) of equal
/// geometry and the strided [runs](GemmBatch::runs). Nothing is recomputed
/// later.
///
/// The batch is owned, `Clone + Send + Sync`, and has no lifetime parameter.
/// It holds no buffer, pointer, device, executor, plan or scalar value: a
/// consuming plan owns the batch and borrows its buffers per execution, and a
/// GPU backend builds its pointer arrays from the offsets at its own prepare
/// time.
///
/// # Examples
///
/// ```
/// use t4a_desc::gemm::{GemmBatch, GemmJob, GemmOps, Operand, Trans};
/// use t4a_desc::ScalarKind;
///
/// let ops = GemmOps { a: Trans::N, b: Trans::N };
/// let batch = GemmBatch::new(
///     ScalarKind::C64,
///     ops,
///     vec![GemmJob::compact(ops, [0, 0, 0], 2, 2, 3)],
/// )?;
/// assert_eq!(batch.required_len(Operand::A), 6);
/// batch.check_lens(6, 6, 4)?;
/// # Ok::<(), t4a_desc::gemm::GemmError>(())
/// ```
#[derive(Clone)]
pub struct GemmBatch {
    scalar: ScalarKind,
    ops: GemmOps,
    jobs: Vec<GemmJob>,
    /// Maximum footprint end over all jobs, per operand (`Operand::index`).
    required: [usize; 3],
    /// Ids of the non-empty outputs in increasing `c_offset`.
    output_order: Vec<usize>,
    /// The ids of every group, concatenated in group order.
    group_ids: Vec<usize>,
    groups: Vec<GroupRange>,
    runs: Vec<GemmRun>,
    work: u128,
    max_job_work: u128,
}

/// One group: its shape and its range in `GemmBatch::group_ids`.
#[derive(Clone, Copy)]
struct GroupRange {
    shape: GemmShape,
    start: usize,
    end: usize,
}

/// One group of a [`GemmBatch`]: jobs with equal [`GemmShape`].
///
/// These are cuBLAS's groups, and the jobs that share one child plan in
/// cpueinsum (tensor4all/cpueinsum-rs#51). Groups come in order of their
/// lowest job id, and `ids` ascend.
///
/// # Examples
///
/// ```
/// use t4a_desc::gemm::{GemmBatch, GemmJob, GemmOps, Trans};
/// use t4a_desc::ScalarKind;
///
/// let ops = GemmOps { a: Trans::N, b: Trans::N };
/// let jobs = vec![
///     GemmJob::compact(ops, [0, 0, 0], 1, 1, 1),
///     GemmJob::compact(ops, [0, 0, 1], 2, 1, 1),
///     GemmJob::compact(ops, [0, 0, 3], 1, 1, 1),
/// ];
/// let batch = GemmBatch::new(ScalarKind::F64, ops, jobs)?;
/// let ids: Vec<&[usize]> = batch.groups().map(|g| g.ids).collect();
/// assert_eq!(ids, [&[0, 2][..], &[1][..]]);
/// # Ok::<(), t4a_desc::gemm::GemmError>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct GemmGroup<'b> {
    /// The geometry every job of the group has.
    pub shape: GemmShape,
    /// The original ids of the group's jobs, ascending.
    pub ids: &'b [usize],
}

/// A maximal strided run of jobs (tensor4all/cpueinsum-rs#78).
///
/// Jobs `first..first + len` of the original list have the same
/// [`GemmShape`], and each of their offsets advances by one constant stride:
/// job `first + i` has offsets `offset(first) + i·stride` for `A`, `B` and `C`
/// in that order. The `C` stride is positive and at least the `C` footprint
/// `(n-1)·ldc + m`, so the run's outputs are disjoint by construction. The `A`
/// and `B` strides may be any value, zero included (a shared input).
///
/// Runs are found greedily from the left and cover every job; a job that
/// cannot extend a run is a run of length 1 with stride `[0, 0, 0]`.
///
/// # Examples
///
/// ```
/// use t4a_desc::gemm::{GemmBatch, GemmJob, GemmOps, GemmRun, Trans};
/// use t4a_desc::ScalarKind;
///
/// let ops = GemmOps { a: Trans::N, b: Trans::N };
/// // Three 2×2×2 jobs sharing B, with A and C advancing by 4.
/// let jobs = (0..3).map(|i| GemmJob::compact(ops, [4 * i, 0, 4 * i], 2, 2, 2)).collect();
/// let batch = GemmBatch::new(ScalarKind::F32, ops, jobs)?;
/// assert_eq!(batch.runs(), [GemmRun { first: 0, len: 3, stride: [4, 0, 4] }]);
/// # Ok::<(), t4a_desc::gemm::GemmError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmRun {
    /// Original id of the run's first job.
    pub first: usize,
    /// Number of jobs in the run, at least 1.
    pub len: usize,
    /// Offset strides of `A`, `B` and `C`, in elements.
    pub stride: [isize; 3],
}

const SPAN: [&str; 3] = ["A span", "B span", "C span"];
const END: [&str; 3] = ["A end", "B end", "C end"];
const BYTE_END: [&str; 3] = ["A byte end", "B byte end", "C byte end"];

impl GemmBatch {
    /// Validate `jobs` for one scalar kind and one pair of ops, and compute
    /// the partitions.
    ///
    /// The checks run before any consumer writes, in this order:
    ///
    /// 1. per job, in id order: every extent and leading dimension fits in
    ///    `isize`; the leading-dimension rules (`ld ≥ max(1, stored rows)`,
    ///    `A`, then `B`, then `C`); then, per operand, the footprint
    ///    `(cols-1)·ld + rows` (0 when the stored matrix is empty), its end
    ///    `offset + footprint`, and that end in bytes, all with checked
    ///    arithmetic and within `isize`. So `k = 0` gives `A` and `B` empty
    ///    footprints while `C` is still updated (`C ← beta·C`);
    /// 2. the non-empty outputs are pairwise disjoint, by the interval test on
    ///    `[c_offset, c_offset + (n-1)·ldc + m)`. The test is conservative:
    ///    interleaved outputs with `ldc > m` are rejected even when they share
    ///    no element.
    ///
    /// The footprint of a job with an empty operand still counts its offset
    /// towards [`required_len`](Self::required_len), so a consumer can slice
    /// every operand.
    ///
    /// # Errors
    ///
    /// The error names the lowest failing job id:
    ///
    /// - [`GemmError::Overflow`] when an extent or leading dimension does not
    ///   fit in `isize`, or a footprint, its end or its byte end overflows or
    ///   exceeds `isize::MAX`;
    /// - [`GemmError::LeadingDim`] when a leading dimension is below its
    ///   minimum;
    /// - [`GemmError::Overlap`] when two non-empty outputs overlap, naming the
    ///   lowest id that overlaps any other job.
    ///
    /// # Examples
    ///
    /// ```
    /// use t4a_desc::gemm::{GemmBatch, GemmError, GemmJob, GemmOps, Trans};
    /// use t4a_desc::ScalarKind;
    ///
    /// let ops = GemmOps { a: Trans::N, b: Trans::N };
    /// let job = |c| GemmJob::compact(ops, [0, 0, c], 2, 2, 1);
    /// assert!(GemmBatch::new(ScalarKind::F64, ops, vec![job(0), job(4)]).is_ok());
    /// let err = GemmBatch::new(ScalarKind::F64, ops, vec![job(4), job(0), job(3)]).unwrap_err();
    /// assert_eq!(err, GemmError::Overlap { job: 0, other: 2 });
    /// ```
    pub fn new(scalar: ScalarKind, ops: GemmOps, jobs: Vec<GemmJob>) -> Result<Self, GemmError> {
        let mut required = [0usize; 3];
        let mut work = 0u128;
        let mut max_job_work = 0u128;
        // (start, end, id) of every non-empty output.
        let mut outputs: Vec<(usize, usize, usize)> = Vec::with_capacity(jobs.len());

        for (id, job) in jobs.iter().enumerate() {
            check_isize(id, job)?;
            for operand in Operand::ALL {
                let (rows, _, ld) = job.stored(ops, operand);
                let min = rows.max(1);
                if ld < min {
                    return Err(GemmError::LeadingDim {
                        job: id,
                        operand,
                        ld,
                        min,
                    });
                }
            }
            for operand in Operand::ALL {
                let end = checked_end(id, job, ops, operand, scalar)?;
                let slot = &mut required[operand.index()];
                *slot = (*slot).max(end);
                if operand == Operand::C && job.m != 0 && job.n != 0 {
                    outputs.push((job.c_offset, end, id));
                }
            }
            // INVARIANT: once the outputs are proven disjoint below, the
            // non-empty outputs lie in one buffer of fewer than 2^61 elements
            // (its byte end is within `isize`), so Σ m·n < 2^61, and k < 2^63
            // (`check_isize`): Σ m·n·k < 2^124 fits in `u128`. A batch that
            // fails the overlap test is discarded with these values, so the
            // saturating forms never change a returned result.
            let job_work = (job.m as u128)
                .saturating_mul(job.n as u128)
                .saturating_mul(job.k as u128);
            work = work.saturating_add(job_work);
            max_job_work = max_job_work.max(job_work);
        }

        outputs.sort_unstable_by_key(|&(start, _, id)| (start, id));
        if let Some((job, other)) = lowest_overlap(&outputs) {
            return Err(GemmError::Overlap { job, other });
        }
        let output_order = outputs.iter().map(|&(_, _, id)| id).collect();
        let (group_ids, groups) = group(&jobs);
        let runs = strided_runs(&jobs);

        Ok(Self {
            scalar,
            ops,
            jobs,
            required,
            output_order,
            group_ids,
            groups,
            runs,
            work,
            max_job_work,
        })
    }

    /// The element type the batch was validated for.
    ///
    /// # Examples
    ///
    /// ```
    /// use t4a_desc::gemm::{GemmBatch, GemmOps, Trans};
    /// use t4a_desc::ScalarKind;
    ///
    /// let ops = GemmOps { a: Trans::N, b: Trans::N };
    /// let batch = GemmBatch::new(ScalarKind::C32, ops, Vec::new())?;
    /// assert_eq!(batch.scalar(), ScalarKind::C32);
    /// # Ok::<(), t4a_desc::gemm::GemmError>(())
    /// ```
    pub fn scalar(&self) -> ScalarKind {
        self.scalar
    }

    /// The ops of every job.
    ///
    /// # Examples
    ///
    /// ```
    /// use t4a_desc::gemm::{GemmBatch, GemmOps, Trans};
    /// use t4a_desc::ScalarKind;
    ///
    /// let ops = GemmOps { a: Trans::T, b: Trans::C };
    /// assert_eq!(GemmBatch::new(ScalarKind::C64, ops, Vec::new())?.ops(), ops);
    /// # Ok::<(), t4a_desc::gemm::GemmError>(())
    /// ```
    pub fn ops(&self) -> GemmOps {
        self.ops
    }

    /// The jobs in their original order: the index is the job id every error
    /// and partition refers to.
    ///
    /// # Examples
    ///
    /// ```
    /// use t4a_desc::gemm::{GemmBatch, GemmJob, GemmOps, Trans};
    /// use t4a_desc::ScalarKind;
    ///
    /// let ops = GemmOps { a: Trans::N, b: Trans::N };
    /// let jobs = vec![GemmJob::compact(ops, [0, 0, 1], 1, 1, 1), GemmJob::compact(ops, [0; 3], 1, 1, 1)];
    /// let batch = GemmBatch::new(ScalarKind::F64, ops, jobs.clone())?;
    /// assert_eq!(batch.jobs(), jobs);
    /// # Ok::<(), t4a_desc::gemm::GemmError>(())
    /// ```
    pub fn jobs(&self) -> &[GemmJob] {
        &self.jobs
    }

    /// The smallest buffer length, in elements, that holds every job's
    /// footprint in `operand`: the maximum `offset + footprint` over all jobs.
    ///
    /// # Examples
    ///
    /// ```
    /// use t4a_desc::gemm::{GemmBatch, GemmJob, GemmOps, Operand, Trans};
    /// use t4a_desc::ScalarKind;
    ///
    /// let ops = GemmOps { a: Trans::N, b: Trans::N };
    /// // C is 2×3 with ldc = 5: (3 - 1)·5 + 2 = 12 elements from offset 1.
    /// let job = GemmJob { ldc: 5, ..GemmJob::compact(ops, [0, 0, 1], 2, 3, 1) };
    /// let batch = GemmBatch::new(ScalarKind::F32, ops, vec![job])?;
    /// assert_eq!(batch.required_len(Operand::C), 13);
    /// # Ok::<(), t4a_desc::gemm::GemmError>(())
    /// ```
    pub fn required_len(&self, operand: Operand) -> usize {
        self.required[operand.index()]
    }

    /// Check buffer lengths, in elements, against every job's footprint.
    ///
    /// O(1) when the lengths suffice; on failure one scan finds the lowest
    /// job id that does not fit.
    ///
    /// # Errors
    ///
    /// [`GemmError::Bounds`] for the first of `A`, `B`, `C` that is too short,
    /// naming the lowest job whose footprint ends past it.
    ///
    /// # Examples
    ///
    /// ```
    /// use t4a_desc::gemm::{GemmBatch, GemmError, GemmJob, GemmOps, Operand, Trans};
    /// use t4a_desc::ScalarKind;
    ///
    /// let ops = GemmOps { a: Trans::N, b: Trans::N };
    /// let jobs = vec![GemmJob::compact(ops, [0, 0, 4], 2, 2, 2), GemmJob::compact(ops, [0; 3], 2, 2, 2)];
    /// let batch = GemmBatch::new(ScalarKind::F64, ops, jobs)?;
    /// assert!(batch.check_lens(4, 4, 8).is_ok());
    /// assert_eq!(
    ///     batch.check_lens(4, 4, 7),
    ///     Err(GemmError::Bounds { job: 0, operand: Operand::C, end: 8, available: 7 }),
    /// );
    /// # Ok::<(), t4a_desc::gemm::GemmError>(())
    /// ```
    pub fn check_lens(&self, a: usize, b: usize, c: usize) -> Result<(), GemmError> {
        for (operand, available) in Operand::ALL.into_iter().zip([a, b, c]) {
            if available >= self.required[operand.index()] {
                continue;
            }
            for (id, job) in self.jobs.iter().enumerate() {
                let end = self.end(job, operand);
                if end > available {
                    return Err(GemmError::Bounds {
                        job: id,
                        operand,
                        end,
                        available,
                    });
                }
            }
            // INVARIANT: `required` is the maximum `end` over the jobs
            // (`GemmBatch::new`), so the scan above has returned whenever
            // `available < required`; this point is unreachable.
        }
        Ok(())
    }

    /// Check the layout half of an uninitialized-output entry: every
    /// non-empty job has `ldc == m`, and the non-empty outputs tile
    /// `[0, c_len)` exactly once.
    ///
    /// Rejecting `beta ≠ 0` stays with the consumer, which owns the scalars.
    /// The consumer also calls [`check_lens`](Self::check_lens): an empty job's
    /// offset is not part of the coverage.
    ///
    /// # Errors
    ///
    /// - [`GemmError::FreshLayout`] naming the lowest non-empty job with
    ///   `ldc != m`;
    /// - [`GemmError::FreshCoverage`] naming the lowest job that does not start
    ///   where the previous output (in offset order) ends, or with `job: None`
    ///   when the outputs tile `[0, end)` for an `end != c_len`.
    ///
    /// # Examples
    ///
    /// ```
    /// use t4a_desc::gemm::{GemmBatch, GemmError, GemmJob, GemmOps, Trans};
    /// use t4a_desc::ScalarKind;
    ///
    /// let ops = GemmOps { a: Trans::N, b: Trans::N };
    /// let jobs = vec![GemmJob::compact(ops, [0, 0, 4], 2, 2, 1), GemmJob::compact(ops, [0; 3], 2, 2, 1)];
    /// let batch = GemmBatch::new(ScalarKind::F64, ops, jobs)?;
    /// assert!(batch.check_fresh_cover(8).is_ok());
    /// assert_eq!(batch.check_fresh_cover(9), Err(GemmError::FreshCoverage { job: None }));
    /// # Ok::<(), t4a_desc::gemm::GemmError>(())
    /// ```
    pub fn check_fresh_cover(&self, c_len: usize) -> Result<(), GemmError> {
        if let Some(job) = self
            .jobs
            .iter()
            .position(|j| j.m != 0 && j.n != 0 && j.ldc != j.m)
        {
            return Err(GemmError::FreshLayout { job });
        }
        let mut cursor = 0usize;
        let mut gap: Option<usize> = None;
        for &id in &self.output_order {
            let job = &self.jobs[id];
            if job.c_offset != cursor {
                gap = Some(gap.map_or(id, |g| g.min(id)));
            }
            cursor = self.end(job, Operand::C);
        }
        match gap {
            Some(job) => Err(GemmError::FreshCoverage { job: Some(job) }),
            None if cursor != c_len => Err(GemmError::FreshCoverage { job: None }),
            None => Ok(()),
        }
    }

    /// The ids of the non-empty outputs in increasing `c_offset`.
    ///
    /// Consecutive entries are disjoint and ordered, which is what a consumer
    /// needs to split one mutable output buffer into per-job pieces. Jobs with
    /// `m = 0` or `n = 0` are not listed.
    ///
    /// # Examples
    ///
    /// ```
    /// use t4a_desc::gemm::{GemmBatch, GemmJob, GemmOps, Trans};
    /// use t4a_desc::ScalarKind;
    ///
    /// let ops = GemmOps { a: Trans::N, b: Trans::N };
    /// let jobs = vec![
    ///     GemmJob::compact(ops, [0, 0, 5], 1, 1, 1),
    ///     GemmJob::compact(ops, [0, 0, 9], 0, 1, 1), // empty output
    ///     GemmJob::compact(ops, [0, 0, 2], 1, 1, 1),
    /// ];
    /// assert_eq!(GemmBatch::new(ScalarKind::F64, ops, jobs)?.output_order(), [2, 0]);
    /// # Ok::<(), t4a_desc::gemm::GemmError>(())
    /// ```
    pub fn output_order(&self) -> &[usize] {
        &self.output_order
    }

    /// The groups of jobs with equal [`GemmShape`], in order of their lowest
    /// id. See [`GemmGroup`].
    ///
    /// # Examples
    ///
    /// ```
    /// use t4a_desc::gemm::{GemmBatch, GemmJob, GemmOps, Trans};
    /// use t4a_desc::ScalarKind;
    ///
    /// let ops = GemmOps { a: Trans::N, b: Trans::N };
    /// let jobs = (0..4).map(|i| GemmJob::compact(ops, [0, 0, 4 * i], 1 + i % 2, 1, 1)).collect();
    /// let batch = GemmBatch::new(ScalarKind::F64, ops, jobs)?;
    /// let sizes: Vec<_> = batch.groups().map(|g| (g.shape.m, g.ids.len())).collect();
    /// assert_eq!(sizes, [(1, 2), (2, 2)]);
    /// # Ok::<(), t4a_desc::gemm::GemmError>(())
    /// ```
    pub fn groups(&self) -> impl ExactSizeIterator<Item = GemmGroup<'_>> + '_ {
        self.groups.iter().map(move |g| GemmGroup {
            shape: g.shape,
            ids: &self.group_ids[g.start..g.end],
        })
    }

    /// The strided runs, in original order, covering every job. See
    /// [`GemmRun`].
    ///
    /// # Examples
    ///
    /// ```
    /// use t4a_desc::gemm::{GemmBatch, GemmJob, GemmOps, Trans};
    /// use t4a_desc::ScalarKind;
    ///
    /// let ops = GemmOps { a: Trans::N, b: Trans::N };
    /// let jobs = [0, 1, 2, 4].map(|c| GemmJob::compact(ops, [0, 0, c], 1, 1, 1)).to_vec();
    /// let lens: Vec<_> = GemmBatch::new(ScalarKind::F64, ops, jobs)?.runs().iter().map(|r| r.len).collect();
    /// assert_eq!(lens, [3, 1]);
    /// # Ok::<(), t4a_desc::gemm::GemmError>(())
    /// ```
    pub fn runs(&self) -> &[GemmRun] {
        &self.runs
    }

    /// The total work `Σ m·n·k`, in multiply-adds.
    ///
    /// # Examples
    ///
    /// ```
    /// use t4a_desc::gemm::{GemmBatch, GemmJob, GemmOps, Trans};
    /// use t4a_desc::ScalarKind;
    ///
    /// let ops = GemmOps { a: Trans::N, b: Trans::N };
    /// let jobs = vec![GemmJob::compact(ops, [0; 3], 2, 3, 4), GemmJob::compact(ops, [0, 0, 6], 1, 1, 5)];
    /// assert_eq!(GemmBatch::new(ScalarKind::F64, ops, jobs)?.work(), 29);
    /// # Ok::<(), t4a_desc::gemm::GemmError>(())
    /// ```
    pub fn work(&self) -> u128 {
        self.work
    }

    /// The largest work `m·n·k` of a single job.
    ///
    /// # Examples
    ///
    /// ```
    /// use t4a_desc::gemm::{GemmBatch, GemmJob, GemmOps, Trans};
    /// use t4a_desc::ScalarKind;
    ///
    /// let ops = GemmOps { a: Trans::N, b: Trans::N };
    /// let jobs = vec![GemmJob::compact(ops, [0; 3], 2, 3, 4), GemmJob::compact(ops, [0, 0, 6], 1, 1, 5)];
    /// assert_eq!(GemmBatch::new(ScalarKind::F64, ops, jobs)?.max_job_work(), 24);
    /// # Ok::<(), t4a_desc::gemm::GemmError>(())
    /// ```
    pub fn max_job_work(&self) -> u128 {
        self.max_job_work
    }

    /// The exclusive end of `job`'s footprint in `operand`.
    fn end(&self, job: &GemmJob, operand: Operand) -> usize {
        let (rows, cols, ld) = job.stored(self.ops, operand);
        // INVARIANT: `GemmBatch::new` computed this end with checked
        // arithmetic for every job of the batch, so neither step saturates.
        span(rows, cols, ld)
            .unwrap_or(usize::MAX)
            .saturating_add(job.offset(operand))
    }
}

impl fmt::Debug for GemmBatch {
    /// A summary: counts and the first few jobs, not the whole list.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const SHOWN: usize = 4;
        f.debug_struct("GemmBatch")
            .field("scalar", &self.scalar)
            .field("ops", &self.ops)
            .field("jobs", &self.jobs.len())
            .field("groups", &self.groups.len())
            .field("runs", &self.runs.len())
            .field("required", &self.required)
            .field("first_jobs", &&self.jobs[..self.jobs.len().min(SHOWN)])
            .finish()
    }
}

/// Every extent and leading dimension of `job` fits in `isize`.
fn check_isize(id: usize, job: &GemmJob) -> Result<(), GemmError> {
    let fields = [
        (job.m, "m"),
        (job.n, "n"),
        (job.k, "k"),
        (job.lda, "lda"),
        (job.ldb, "ldb"),
        (job.ldc, "ldc"),
    ];
    for (value, what) in fields {
        if isize::try_from(value).is_err() {
            return Err(GemmError::Overflow { job: id, what });
        }
    }
    Ok(())
}

/// The checked end of `job`'s footprint in `operand`, with its byte end within
/// `isize`.
fn checked_end(
    id: usize,
    job: &GemmJob,
    ops: GemmOps,
    operand: Operand,
    scalar: ScalarKind,
) -> Result<usize, GemmError> {
    let i = operand.index();
    let overflow = |what| GemmError::Overflow { job: id, what };
    let (rows, cols, ld) = job.stored(ops, operand);
    let span = span(rows, cols, ld).ok_or(overflow(SPAN[i]))?;
    let end = job
        .offset(operand)
        .checked_add(span)
        .ok_or(overflow(END[i]))?;
    end.checked_mul(scalar.size_of())
        .filter(|&bytes| isize::try_from(bytes).is_ok())
        .ok_or(overflow(BYTE_END[i]))?;
    Ok(end)
}

/// The lowest id whose output overlaps another, and one job it overlaps.
///
/// `outputs` holds `(start, end, id)` of the non-empty outputs sorted by
/// `(start, id)`. A job overlaps an earlier one exactly when its start is
/// below the largest end before it, and a later one exactly when the next
/// start is below its end, so one sweep finds every overlapping job.
fn lowest_overlap(outputs: &[(usize, usize, usize)]) -> Option<(usize, usize)> {
    let mut found: Option<(usize, usize)> = None;
    // The largest end so far and the job that has it.
    let mut reach: Option<(usize, usize)> = None;
    for (i, &(start, end, id)) in outputs.iter().enumerate() {
        let earlier = reach.filter(|&(r, _)| r > start).map(|(_, other)| other);
        let later = outputs
            .get(i + 1)
            .filter(|&&(next, _, _)| next < end)
            .map(|&(_, _, other)| other);
        if let Some(other) = earlier.or(later) {
            if found.is_none_or(|(job, _)| id < job) {
                found = Some((id, other));
            }
        }
        if reach.is_none_or(|(r, _)| end > r) {
            reach = Some((end, id));
        }
    }
    found
}

/// The groups of equal shape: their ids concatenated, and one range per group,
/// in order of the lowest id.
fn group(jobs: &[GemmJob]) -> (Vec<usize>, Vec<GroupRange>) {
    let mut by_shape: Vec<usize> = (0..jobs.len()).collect();
    by_shape.sort_unstable_by_key(|&id| (jobs[id].shape(), id));
    // Each shape is now one contiguous segment with its lowest id first.
    let mut segments: Vec<(usize, usize)> = Vec::new();
    let mut start = 0;
    while start < by_shape.len() {
        let shape = jobs[by_shape[start]].shape();
        let mut end = start + 1;
        while end < by_shape.len() && jobs[by_shape[end]].shape() == shape {
            end += 1;
        }
        segments.push((start, end));
        start = end;
    }
    segments.sort_unstable_by_key(|&(start, _)| by_shape[start]);

    let mut ids = Vec::with_capacity(jobs.len());
    let mut groups = Vec::with_capacity(segments.len());
    for (start, end) in segments {
        let first = ids.len();
        ids.extend_from_slice(&by_shape[start..end]);
        groups.push(GroupRange {
            shape: jobs[by_shape[start]].shape(),
            start: first,
            end: ids.len(),
        });
    }
    (ids, groups)
}

/// The strided runs, greedily from the left (see [`GemmRun`]).
fn strided_runs(jobs: &[GemmJob]) -> Vec<GemmRun> {
    let mut runs = Vec::new();
    let mut first = 0;
    while first < jobs.len() {
        let shape = jobs[first].shape();
        let same = |id: usize| jobs.get(id).is_some_and(|j| j.shape() == shape);
        let mut run = GemmRun {
            first,
            len: 1,
            stride: [0; 3],
        };
        if same(first + 1) {
            let stride = delta(&jobs[first], &jobs[first + 1]);
            // INVARIANT: `GemmBatch::new` checked `span(m, n, ldc)` for every job.
            let footprint = span(shape.m, shape.n, shape.ldc).unwrap_or(usize::MAX);
            if stride[2] > 0 && stride[2].unsigned_abs() >= footprint {
                run.stride = stride;
                run.len = 2;
                while same(first + run.len)
                    && delta(&jobs[first + run.len - 1], &jobs[first + run.len]) == stride
                {
                    run.len += 1;
                }
            }
        }
        runs.push(run);
        first += run.len;
    }
    runs
}

/// The offset steps from `x` to `y`, for `A`, `B` and `C`.
fn delta(x: &GemmJob, y: &GemmJob) -> [isize; 3] {
    // INVARIANT: every offset is at most its byte end, which `GemmBatch::new`
    // bounded by `isize::MAX`, so the casts are exact and the difference of
    // two values in `0..=isize::MAX` cannot overflow.
    Operand::ALL.map(|op| y.offset(op) as isize - x.offset(op) as isize)
}
