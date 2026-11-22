//! `FusedGemmPlan` (tensor4all/tcontract-rs#208): one packed traversal over
//! every job of a `t4a_desc::gemm::GemmBatch`.
//!
//! Inputs are small integers, so every product is exact in every dtype and the
//! results are asserted exactly: bitwise against each job run alone through a
//! forced-packed `Plan` on one thread (the per-job oracle), exactly against a
//! naive triple loop, and bitwise across executor widths.

use core::mem::MaybeUninit;

use num_complex::{Complex32, Complex64};
use t4a_desc::gemm::{GemmBatch, GemmError, GemmJob, GemmOps, Operand, Trans};
use t4a_desc::ScalarKind;
use tcontract::api::{
    CSpec, ConfigError, DType, Labels, LayoutSpec, Op, OperandSpec, Problem, Scalar,
};
use tcontract::{
    Error, ExecutionRoute, FusedGemmPlan, Partition, Plan, PlanConfig, SliceAccumulationSource,
};
use tcontract_exec::{BatchPolicy, BatchStrategy, Exec, ExecError, Pool};
use tcontract_kernel::{BlockingOverride, Element, KernelChoice, Real};

mod common;

/// A small integer in `-3..=3`, with an independent imaginary part for a
/// complex type.
fn int<T: Element>(i: usize) -> T {
    let r = ((i * 5 + 3) % 7) as f64 - 3.0;
    let im = ((i * 3 + 1) % 5) as f64 - 2.0;
    T::from_parts(
        T::Real::from_f64(r),
        if T::IS_COMPLEX {
            T::Real::from_f64(im)
        } else {
            T::Real::ZERO
        },
    )
}

fn ints<T: Element>(len: usize, seed: usize) -> Vec<T> {
    (0..len).map(|i| int(i * 3 + seed)).collect()
}

#[track_caller]
fn same<T: Element>(got: &[T], want: &[T], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    if let Some(i) = (0..got.len()).find(|&i| bits(&got[i..=i]) != bits(&want[i..=i])) {
        panic!(
            "{what}: first difference at {i}: got ({}, {}), want ({}, {})",
            got[i].re().to_f64(),
            got[i].im().to_f64(),
            want[i].re().to_f64(),
            want[i].im().to_f64()
        );
    }
}

/// Exact value equality against the naive reference, whose summation order
/// differs from the kernels' only in the sign of a zero.
#[track_caller]
fn equal<T: Element>(got: &[T], want: &[T], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    let key = |x: &T| (x.re().to_f64(), x.im().to_f64());
    let differs = |i: &usize| {
        let ((a, b), (c, d)) = (key(&got[*i]), key(&want[*i]));
        // NaN marks an element neither side may write: it must stay NaN.
        !((a == c || a.is_nan() && c.is_nan()) && (b == d || b.is_nan() && d.is_nan()))
    };
    if let Some(i) = (0..got.len()).find(differs) {
        panic!(
            "{what}: first difference at {i}: got {:?}, want {:?}",
            key(&got[i]),
            key(&want[i])
        );
    }
}

fn bits<T: Element>(v: &[T]) -> Vec<(u64, u64)> {
    v.iter()
        .map(|x| (x.re().to_f64().to_bits(), x.im().to_f64().to_bits()))
        .collect()
}

/// `ld` elements per stored column, at least the stored rows.
fn job(ops: GemmOps, offsets: [usize; 3], (m, n, k): (usize, usize, usize), pad: usize) -> GemmJob {
    let mut j = GemmJob::compact(ops, offsets, m, n, k);
    j.lda += pad;
    j.ldb += pad;
    j.ldc += pad;
    j
}

/// Lay `shapes` out back to back in three buffers (each job's footprint after
/// the previous one, plus a gap), with `pad` extra elements per leading
/// dimension.
fn layout(ops: GemmOps, shapes: &[(usize, usize, usize)], pad: usize) -> Vec<GemmJob> {
    let mut at = [0usize; 3];
    let mut jobs = Vec::new();
    for &s in shapes {
        let j = job(ops, at, s, pad);
        let span = |rows: usize, cols: usize, ld: usize| match rows * cols {
            0 => 0,
            _ => (cols - 1) * ld + rows,
        };
        let (am, an) = match ops.a {
            Trans::N => (j.m, j.k),
            _ => (j.k, j.m),
        };
        let (bm, bn) = match ops.b {
            Trans::N => (j.k, j.n),
            _ => (j.n, j.k),
        };
        at[0] += span(am, an, j.lda) + 1;
        at[1] += span(bm, bn, j.ldb) + 2;
        at[2] += span(j.m, j.n, j.ldc) + 3;
        jobs.push(j);
    }
    jobs
}

fn kind<T: Scalar>() -> ScalarKind {
    match T::STORAGE {
        DType::F32 => ScalarKind::F32,
        DType::F64 => ScalarKind::F64,
        DType::C32 => ScalarKind::C32,
        DType::C64 => ScalarKind::C64,
    }
}

/// `op(X)[r, c]` of a stored operand at `off` with leading dimension `ld`.
fn op_at<T: Scalar>(x: &[T], trans: Trans, off: usize, ld: usize, r: usize, c: usize) -> T {
    match trans {
        Trans::N => x[off + r + c * ld],
        Trans::T => x[off + c + r * ld],
        Trans::C => <T as Element>::conj(x[off + c + r * ld]),
    }
}

/// `C ← alpha·op(A)·op(B) + beta·C` for every job, by the definition.
fn naive<T: Scalar>(
    ops: GemmOps,
    jobs: &[GemmJob],
    alpha: T,
    a: &[T],
    b: &[T],
    beta: T,
    c: &mut [T],
) {
    for j in jobs {
        for col in 0..j.n {
            for row in 0..j.m {
                let mut s = <T as Element>::zero();
                for p in 0..j.k {
                    s = s + op_at(a, ops.a, j.a_offset, j.lda, row, p)
                        * op_at(b, ops.b, j.b_offset, j.ldb, p, col);
                }
                let at = j.c_offset + row + col * j.ldc;
                let zero = <T as Element>::zero();
                let old = if beta == zero { zero } else { beta * c[at] };
                c[at] = alpha * s + old;
            }
        }
    }
}

/// Each job alone through `Plan` with the packed driver forced, on one
/// thread: the per-job oracle.
fn per_job_plan<T: Scalar>(
    ops: GemmOps,
    jobs: &[GemmJob],
    config: &PlanConfig,
    alpha: T,
    a: &[T],
    b: &[T],
    beta: T,
    c: &mut [T],
) {
    let spec = |trans: Trans, rows: usize, cols: usize, ld: usize| {
        let ld = ld as isize;
        let strides = match trans {
            Trans::N => [1, ld],
            _ => [ld, 1],
        };
        let s = OperandSpec::new(LayoutSpec::new(&[rows, cols], &strides, 0).unwrap());
        match trans {
            Trans::C if T::IS_COMPLEX => s.conj(),
            _ => s,
        }
    };
    let config = PlanConfig {
        partition: Some(Partition::StaticGrid {
            pin: None,
            align_c_lines: false,
        }),
        ..config.clone()
    };
    for j in jobs.iter().filter(|j| j.m > 0 && j.n > 0) {
        let problem = Problem::from_labels(
            T::STORAGE,
            spec(ops.a, j.m, j.k, j.lda),
            spec(ops.b, j.k, j.n, j.ldb),
            CSpec::Output(Op::Identity),
            spec(Trans::N, j.m, j.n, j.ldc),
            &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
        )
        .unwrap();
        let plan = Plan::<T>::new(&problem, &config).unwrap();
        let source = match beta == <T as Element>::zero() {
            true => SliceAccumulationSource::Absent,
            false => SliceAccumulationSource::Output,
        };
        plan.execute_slices_accum(
            &mut Exec::serial(),
            alpha,
            (a, j.a_offset as isize),
            (b, j.b_offset as isize),
            beta,
            source,
            (c, j.c_offset as isize),
        )
        .unwrap();
    }
}

/// Mixed sizes: one job, a few of one geometry, larger ones that span several
/// blocks under `small_blocks`, empty `k` and empty outputs.
const MIXED: &[(usize, usize, usize)] = &[
    (3, 2, 4),
    (17, 9, 13),
    (3, 2, 4),
    (0, 5, 3),
    (6, 7, 0),
    (1, 1, 1),
    (33, 21, 40),
    (4, 0, 2),
    (3, 2, 4),
    (8, 8, 8),
];

/// Blocking small enough that the larger `MIXED` jobs take several row bands,
/// column panels and K slabs.
fn small_blocks() -> PlanConfig {
    PlanConfig {
        blocking: BlockingOverride {
            mc: Some(16),
            kc: Some(8),
            nc: Some(16),
            ..Default::default()
        },
        ..PlanConfig::default()
    }
}

/// One fused call against both oracles; returns the fused `C`.
fn check<T: Scalar>(
    ops: GemmOps,
    shapes: &[(usize, usize, usize)],
    pad: usize,
    config: &PlanConfig,
    beta: T,
) -> Vec<T> {
    let jobs = layout(ops, shapes, pad);
    let batch = GemmBatch::new(kind::<T>(), ops, jobs.clone()).unwrap();
    let (la, lb, lc) = (
        batch.required_len(Operand::A),
        batch.required_len(Operand::B),
        batch.required_len(Operand::C),
    );
    let (a, b, c0) = (ints::<T>(la, 1), ints::<T>(lb, 2), ints::<T>(lc, 3));
    let alpha = int::<T>(4);
    let plan = FusedGemmPlan::<T>::new(batch, config).unwrap();

    let mut fused = c0.clone();
    let route = plan
        .execute_into_accum(&mut Exec::serial(), alpha, &a, &b, beta, &mut fused)
        .unwrap();
    assert_eq!(route, ExecutionRoute::FusedTiles { width: 1 });

    let mut oracle = c0.clone();
    per_job_plan(ops, &jobs, config, alpha, &a, &b, beta, &mut oracle);
    same(
        &fused,
        &oracle,
        &format!("per-job Plan oracle, ops {ops:?}"),
    );

    let mut want = c0.clone();
    naive(ops, &jobs, alpha, &a, &b, beta, &mut want);
    equal(&fused, &want, &format!("naive reference, ops {ops:?}"));
    fused
}

const N: Trans = Trans::N;
const T_: Trans = Trans::T;
const C_: Trans = Trans::C;

fn ops(a: Trans, b: Trans) -> GemmOps {
    GemmOps { a, b }
}

fn op_matrix<T: Scalar>() -> Vec<GemmOps> {
    let mut v = vec![ops(N, N), ops(T_, N), ops(N, T_), ops(T_, T_), ops(C_, N)];
    if T::IS_COMPLEX {
        v.extend([ops(C_, C_), ops(N, C_)]);
    }
    v
}

fn dtype_suite<T: Scalar>() {
    for o in op_matrix::<T>() {
        for config in [PlanConfig::default(), small_blocks()] {
            for pad in [0, 2] {
                check::<T>(o, MIXED, pad, &config, <T as Element>::zero());
                check::<T>(o, MIXED, pad, &config, int::<T>(6));
            }
        }
    }
}

#[test]
fn f64_matches_both_oracles() {
    dtype_suite::<f64>();
}

#[test]
fn f32_matches_both_oracles() {
    dtype_suite::<f32>();
}

#[test]
fn c64_matches_both_oracles() {
    dtype_suite::<Complex64>();
}

#[test]
fn c32_matches_both_oracles() {
    dtype_suite::<Complex32>();
}

#[test]
fn one_job_and_many_small_jobs() {
    check::<f64>(ops(N, N), &[(5, 4, 3)], 0, &PlanConfig::default(), 0.0);
    let many: Vec<_> = (0..64).map(|i| (1 + i % 5, 1 + i % 3, 1 + i % 4)).collect();
    check::<f64>(ops(T_, N), &many, 1, &PlanConfig::default(), 2.0);
}

#[test]
fn small_blocking_splits_jobs_into_items() {
    let ops = ops(N, N);
    let batch = GemmBatch::new(
        ScalarKind::F64,
        ops,
        layout(ops, &[(33, 21, 40), (3, 2, 4)], 0),
    )
    .unwrap();
    let plan = FusedGemmPlan::<f64>::new(batch, &small_blocks()).unwrap();
    // 33 rows in bands of 16 (or NR-oriented equivalents) and 21 columns in
    // panels of 16: more than one item for the large job, one for the small.
    assert!(plan.work_items() > 2, "{}", plan.work_items());
}

/// The result does not depend on the width, and a forced outer policy runs on
/// several workers.
#[test]
fn bitwise_identical_across_widths() {
    let sp = common::placement::pool(4);
    let pool = Pool::borrow(&sp);
    for config in [PlanConfig::default(), small_blocks()] {
        let o = ops(T_, N);
        let jobs = layout(o, MIXED, 1);
        let batch = GemmBatch::new(ScalarKind::F64, o, jobs).unwrap();
        let (la, lb, lc) = (
            batch.required_len(Operand::A),
            batch.required_len(Operand::B),
            batch.required_len(Operand::C),
        );
        let (a, b, c0) = (ints::<f64>(la, 1), ints::<f64>(lb, 2), ints::<f64>(lc, 3));
        let plan = FusedGemmPlan::<f64>::new(batch, &config).unwrap();
        let mut serial = c0.clone();
        plan.execute_into_accum(&mut Exec::serial(), 3.0, &a, &b, -2.0, &mut serial)
            .unwrap();
        for width in [1, 2, 4] {
            let mut exec = Exec::new(&pool, width)
                .unwrap()
                .with_batch_policy(BatchPolicy::new(BatchStrategy::Outer));
            let mut c = c0.clone();
            let route = plan
                .execute_into_accum(&mut exec, 3.0, &a, &b, -2.0, &mut c)
                .unwrap();
            let want = width.min(plan.work_items());
            assert_eq!(route, ExecutionRoute::FusedTiles { width: want });
            same(&c, &serial, &format!("width {width}"));
        }
        // The default `Auto` policy on the pool agrees too.
        let mut exec = Exec::new(&pool, 4).unwrap();
        let mut c = c0.clone();
        plan.execute_into_accum(&mut exec, 3.0, &a, &b, -2.0, &mut c)
            .unwrap();
        same(&c, &serial, "result");
    }
}

#[test]
fn overwrite_reads_no_old_output() {
    let o = ops(N, T_);
    let jobs = layout(o, MIXED, 0);
    let batch = GemmBatch::new(ScalarKind::F64, o, jobs.clone()).unwrap();
    let la = batch.required_len(Operand::A);
    let lb = batch.required_len(Operand::B);
    let lc = batch.required_len(Operand::C);
    let (a, b) = (ints::<f64>(la, 1), ints::<f64>(lb, 2));
    let plan = FusedGemmPlan::<f64>::new(batch, &PlanConfig::default()).unwrap();
    let mut c = vec![f64::NAN; lc];
    plan.execute_into(&mut Exec::serial(), 2.0, &a, &b, &mut c)
        .unwrap();
    let mut want = vec![f64::NAN; lc];
    naive(o, &jobs, 2.0, &a, &b, 0.0, &mut want);
    equal(&c, &want, "naive reference");
}

#[test]
fn alpha_zero_and_empty_k_are_output_only() {
    let o = ops(N, N);
    // Every non-empty job has k == 0, or alpha == 0: no A or B is read.
    let jobs = layout(o, &[(2, 3, 0), (4, 1, 0), (0, 2, 2)], 1);
    let batch = GemmBatch::new(ScalarKind::F64, o, jobs.clone()).unwrap();
    let (la, lb, lc) = (
        batch.required_len(Operand::A),
        batch.required_len(Operand::B),
        batch.required_len(Operand::C),
    );
    let plan = FusedGemmPlan::<f64>::new(batch, &PlanConfig::default()).unwrap();
    assert_eq!(plan.work_items(), 0);
    let c0 = ints::<f64>(lc, 3);
    let mut c = c0.clone();
    let (a, b) = (vec![f64::NAN; la], vec![f64::NAN; lb]);
    let route = plan
        .execute_into_accum(&mut Exec::serial(), 1.0, &a, &b, 3.0, &mut c)
        .unwrap();
    assert_eq!(route, ExecutionRoute::OutputOnly);
    let mut want = c0.clone();
    naive(o, &jobs, 1.0, &a, &b, 3.0, &mut want);
    equal(&c, &want, "naive reference");

    let jobs = layout(o, MIXED, 0);
    let batch = GemmBatch::new(ScalarKind::F64, o, jobs.clone()).unwrap();
    let (la, lb, lc) = (
        batch.required_len(Operand::A),
        batch.required_len(Operand::B),
        batch.required_len(Operand::C),
    );
    let plan = FusedGemmPlan::<f64>::new(batch, &PlanConfig::default()).unwrap();
    let c0 = ints::<f64>(lc, 3);
    let mut c = c0.clone();
    let a = vec![f64::NAN; la];
    let b = vec![f64::NAN; lb];
    let route = plan
        .execute_into_accum(&mut Exec::serial(), 0.0, &a, &b, -1.0, &mut c)
        .unwrap();
    assert_eq!(route, ExecutionRoute::OutputOnly);
    let mut want = c0;
    naive(
        o,
        &jobs,
        0.0,
        &vec![0.0; la],
        &vec![0.0; lb],
        -1.0,
        &mut want,
    );
    equal(&c, &want, "naive reference");
}

#[test]
fn empty_batches_are_empty() {
    let o = ops(N, N);
    for jobs in [vec![], layout(o, &[(0, 3, 2), (2, 0, 1)], 0)] {
        let batch = GemmBatch::new(ScalarKind::F32, o, jobs).unwrap();
        let a = vec![f32::NAN; batch.required_len(Operand::A)];
        let b = vec![f32::NAN; batch.required_len(Operand::B)];
        let plan = FusedGemmPlan::<f32>::new(batch, &PlanConfig::default()).unwrap();
        let mut c = [7.0f32; 4];
        let route = plan
            .execute_into(&mut Exec::serial(), 1.0, &a, &b, &mut c)
            .unwrap();
        assert_eq!(route, ExecutionRoute::Empty);
        assert_eq!(c, [7.0; 4]);
    }
}

#[test]
fn uninit_output_is_fully_written() {
    let o = ops(C_, N);
    let shapes = [(3, 2, 4), (5, 3, 0), (0, 2, 2), (4, 4, 4)];
    let jobs = layout(o, &shapes, 0);
    // A fresh output must be tiled exactly: rebuild C offsets back to back.
    let mut at = 0;
    let jobs: Vec<_> = jobs
        .into_iter()
        .map(|mut j| {
            j.c_offset = at;
            at += j.m * j.n;
            j
        })
        .collect();
    let batch = GemmBatch::new(ScalarKind::C64, o, jobs.clone()).unwrap();
    let la = batch.required_len(Operand::A);
    let lb = batch.required_len(Operand::B);
    let (a, b) = (ints::<Complex64>(la, 1), ints::<Complex64>(lb, 2));
    let plan = FusedGemmPlan::<Complex64>::new(batch, &PlanConfig::default()).unwrap();
    let alpha = Complex64::new(1.0, -2.0);
    let mut c = vec![MaybeUninit::<Complex64>::uninit(); at];
    let (got, route) = plan
        .execute_uninit(&mut Exec::serial(), alpha, &a, &b, &mut c)
        .unwrap();
    assert_eq!(route, ExecutionRoute::FusedTiles { width: 1 });
    let mut want = vec![Complex64::new(f64::NAN, f64::NAN); at];
    naive(o, &jobs, alpha, &a, &b, Complex64::new(0.0, 0.0), &mut want);
    equal(got, &want, "naive reference");
}

/// Every refused call leaves `C` as it was, and names the lowest job id.
#[test]
fn errors_are_reported_before_any_write() {
    let o = ops(N, N);
    let jobs = layout(o, MIXED, 0);
    let batch = GemmBatch::new(ScalarKind::F64, o, jobs.clone()).unwrap();
    let (la, lb, lc) = (
        batch.required_len(Operand::A),
        batch.required_len(Operand::B),
        batch.required_len(Operand::C),
    );
    let plan = FusedGemmPlan::<f64>::new(batch, &PlanConfig::default()).unwrap();
    let (a, b) = (ints::<f64>(la, 1), ints::<f64>(lb, 2));
    let sentinel = vec![5.0; lc];

    // Bounds: an A buffer one short fails on the job whose footprint is last,
    // and the error names the lowest job that does not fit.
    let mut c = sentinel.clone();
    let err = plan
        .execute_into(&mut Exec::serial(), 1.0, &a[..la - 1], &b, &mut c)
        .unwrap_err();
    assert!(
        matches!(
            err,
            Error::Gemm(GemmError::Bounds {
                operand: Operand::A,
                ..
            })
        ),
        "{err}"
    );
    assert_eq!(c, sentinel);
    // A short C names job 0 when every job overflows it.
    let mut c = sentinel.clone();
    let err = plan
        .execute_into(&mut Exec::serial(), 1.0, &a, &b, &mut c[..1])
        .unwrap_err();
    assert!(
        matches!(
            err,
            Error::Gemm(GemmError::Bounds {
                job: 0,
                operand: Operand::C,
                ..
            })
        ),
        "{err}"
    );
    assert_eq!(c, sentinel);

    // A fresh output with ldc > m names the lowest such job.
    let padded = layout(o, &[(2, 2, 2), (3, 3, 3), (4, 4, 4)], 1);
    let batch = GemmBatch::new(ScalarKind::F64, o, padded).unwrap();
    let plan = FusedGemmPlan::<f64>::new(batch, &PlanConfig::default()).unwrap();
    let mut fresh = vec![MaybeUninit::<f64>::uninit(); 64];
    let err = plan
        .execute_uninit(&mut Exec::serial(), 1.0, &[0.0; 64], &[0.0; 64], &mut fresh)
        .unwrap_err();
    assert!(
        matches!(err, Error::Gemm(GemmError::FreshLayout { job: 0 })),
        "{err}"
    );

    // Outputs that leave a gap in a fresh output.
    let gapped = vec![
        GemmJob::compact(o, [0, 0, 0], 2, 2, 1),
        GemmJob::compact(o, [0, 0, 5], 1, 1, 1),
    ];
    let batch = GemmBatch::new(ScalarKind::F64, o, gapped).unwrap();
    let plan = FusedGemmPlan::<f64>::new(batch, &PlanConfig::default()).unwrap();
    let mut fresh = vec![MaybeUninit::<f64>::uninit(); 6];
    let err = plan
        .execute_uninit(&mut Exec::serial(), 1.0, &[1.0; 4], &[1.0; 4], &mut fresh)
        .unwrap_err();
    assert!(
        matches!(err, Error::Gemm(GemmError::FreshCoverage { job: Some(1) })),
        "{err}"
    );
}

/// Overlap and overflow are rejected by `GemmBatch::new`, before a plan can
/// exist; through `tcontract::Result` they surface as `Error::Gemm`.
#[test]
fn overlap_and_overflow_are_rejected_before_the_plan() {
    fn prepare(jobs: Vec<GemmJob>) -> tcontract::Result<FusedGemmPlan<f64>> {
        let o = GemmOps {
            a: Trans::N,
            b: Trans::N,
        };
        let batch = GemmBatch::new(ScalarKind::F64, o, jobs)?;
        FusedGemmPlan::new(batch, &PlanConfig::default())
    }
    let o = ops(N, N);
    let overlap = vec![
        GemmJob::compact(o, [0, 0, 0], 1, 1, 1),
        GemmJob::compact(o, [0, 0, 10], 2, 2, 1),
        GemmJob::compact(o, [0, 0, 11], 2, 2, 1),
    ];
    let err = prepare(overlap).unwrap_err();
    assert!(
        matches!(err, Error::Gemm(GemmError::Overlap { job: 1, other: 2 })),
        "{err}"
    );

    let mut huge = GemmJob::compact(o, [0, 0, 0], 2, 2, 2);
    huge.c_offset = usize::MAX - 1;
    let jobs = vec![GemmJob::compact(o, [0, 0, 100], 1, 1, 1), huge, huge];
    let err = prepare(jobs).unwrap_err();
    assert!(
        matches!(err, Error::Gemm(GemmError::Overflow { job: 1, .. })),
        "{err}"
    );
}

#[test]
fn configuration_and_policy_refusals() {
    let o = ops(N, N);
    let batch = GemmBatch::new(ScalarKind::F64, o, layout(o, &[(4, 4, 4), (2, 2, 2)], 0)).unwrap();
    let err = FusedGemmPlan::<f32>::new(batch.clone(), &PlanConfig::default()).unwrap_err();
    assert!(
        matches!(err, Error::Config(ConfigError::DtypeMismatch { .. })),
        "{err}"
    );
    let partition = PlanConfig {
        partition: Some(Partition::StaticGrid {
            pin: None,
            align_c_lines: false,
        }),
        ..PlanConfig::default()
    };
    let err = FusedGemmPlan::<f64>::new(batch.clone(), &partition).unwrap_err();
    assert!(
        matches!(err, Error::Config(ConfigError::Option(_))),
        "{err}"
    );

    // An item cannot use threads: `Inner` on a budget above one is refused
    // before any write instead of being silently serialized.
    let plan = FusedGemmPlan::<f64>::new(batch, &PlanConfig::default()).unwrap();
    let sp = common::placement::pool(2);
    let pool = Pool::borrow(&sp);
    let mut exec = Exec::new(&pool, 2)
        .unwrap()
        .with_batch_policy(BatchPolicy::new(BatchStrategy::Inner));
    let mut c = vec![9.0; 32];
    let err = plan
        .execute_into(&mut exec, 1.0, &[1.0; 32], &[1.0; 32], &mut c)
        .unwrap_err();
    assert!(
        matches!(err, Error::Exec(ExecError::StrategyUnavailable { .. })),
        "{err}"
    );
    assert_eq!(c, vec![9.0; 32]);
    // `Sequential` is width one.
    let mut exec = Exec::new(&pool, 2)
        .unwrap()
        .with_batch_policy(BatchPolicy::new(BatchStrategy::Sequential));
    let route = plan
        .execute_into(&mut exec, 1.0, &[1.0; 32], &[1.0; 32], &mut c)
        .unwrap();
    assert_eq!(route, ExecutionRoute::FusedTiles { width: 1 });
}

#[test]
fn plan_is_send_sync_and_debug_summarizes() {
    fn assert_send_sync<X: Send + Sync>() {}
    assert_send_sync::<FusedGemmPlan<f64>>();
    assert_send_sync::<FusedGemmPlan<Complex32>>();
    let o = ops(N, N);
    let batch = GemmBatch::new(ScalarKind::F64, o, layout(o, &[(2, 2, 2)], 0)).unwrap();
    let plan = FusedGemmPlan::<f64>::new(batch, &PlanConfig::default()).unwrap();
    let text = format!("{plan:?}");
    assert!(text.contains("work_items: 1"), "{text}");
    assert!(plan.workspace_req().a_bytes > 0);
}

fn kernel(id: &str) -> PlanConfig {
    PlanConfig {
        kernel: KernelChoice::Id(id.into()),
        ..PlanConfig::default()
    }
}

/// The GLM review's reproducer on #216: a family that reads `B` in place
/// (`direct-b`) used to hand `run_strip` a zero `b_group` and trip its panel
/// check ("column group overruns its slice of the packed B panel").
#[test]
fn direct_b_family_runs_and_matches_the_per_job_plan() {
    let ops = GemmOps {
        a: Trans::N,
        b: Trans::N,
    };
    let jobs = vec![GemmJob::compact(ops, [0, 0, 0], 4, 4, 4)];
    let batch = GemmBatch::new(ScalarKind::F64, ops, jobs).unwrap();
    let plan = FusedGemmPlan::<f64>::new(batch, &kernel("ref.f64.direct-b.4x4")).unwrap();
    let (a, b) = (ints::<f64>(16, 1), ints::<f64>(16, 2));
    let mut c = vec![0.0; 16];
    plan.execute_into(&mut Exec::serial(), 1.0, &a, &b, &mut c)
        .unwrap();

    // The mixed batch, both oracles, with B read in place (op N on B keeps its
    // k stride one; op T does not, so that case packs B as before).
    for id in ["ref.f64.direct-b.4x4", "ref.f64.direct.4x4"] {
        for o in [
            ops,
            GemmOps {
                a: Trans::T,
                b: Trans::N,
            },
            GemmOps {
                a: Trans::N,
                b: Trans::T,
            },
        ] {
            for pad in [0, 2] {
                check::<f64>(o, MIXED, pad, &kernel(id), 0.0);
                check::<f64>(o, MIXED, pad, &kernel(id), -2.0);
            }
        }
    }
    let f32_direct_b = kernel("ref.f32.direct-b.4x4");
    check::<f32>(ops, MIXED, 1, &f32_direct_b, 3.0);
}

/// A geometry that takes no work item is still planned, so an unusable kernel
/// request is refused as the per-job `Plan` refuses it, not skipped.
#[test]
fn kernel_errors_are_reported_for_jobs_without_items() {
    let o = ops(N, N);
    let bogus = kernel("no.such.kernel");
    for shape in [(1, 1, 0), (0, 2, 3)] {
        let batch = GemmBatch::new(ScalarKind::F64, o, layout(o, &[shape], 0)).unwrap();
        let err = FusedGemmPlan::<f64>::new(batch, &bogus).unwrap_err();
        assert!(matches!(err, Error::Select(_)), "{shape:?}: {err}");
    }
}
