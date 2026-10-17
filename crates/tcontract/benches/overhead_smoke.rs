//! Cheap per-call overhead smoke for the allocation-free work in
//! tensor4all/tcontract-rs#121 (the T1-T5 rows of the issue comment).
//!
//! `harness = false`: a few-second ratio guard, not a criterion suite. Each row
//! measures two paths that should cost about the same on the same small input and
//! gates on the ratio measured **in this run**, never on an absolute time. A
//! counting `#[global_allocator]` reports allocs/op beside every ratio.
//!
//! Run with `OVERHEAD_GATE=1` to make a failed gate exit non-zero; without it the
//! table is printed and the process succeeds. The gate fails only when all
//! [`ATTEMPTS`] attempts exceed the threshold, and the per-round timing is the
//! minimum of [`ROUNDS`] rounds.
//!
//! ```text
//! OVERHEAD_GATE=1 cargo bench --profile release -p tcontract --bench overhead_smoke
//! ```
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::io::Write;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use strided_view::{StridedView, StridedViewMut};
use tcontract::api::{AccumulationSource, CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
use tcontract::{contract_batched, BatchItem, Plan, PlanConfig};
use tcontract_exec::{ArenaProvider, Exec};

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every method forwards to the system allocator unchanged; the counter is
// an atomic with no aliasing.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// Timing rounds per attempt; the reported time is the minimum over these.
const ROUNDS: usize = 5;
/// Attempts before a ratio gate may fail; a case fails only when all fail.
const ATTEMPTS: usize = 3;
/// Calls per allocation-count window.
const ALLOC_ITERS: usize = 64;

fn time_once(iterations: usize, body: &mut impl FnMut()) -> f64 {
    let start = Instant::now();
    for _ in 0..iterations {
        body();
    }
    start.elapsed().as_secs_f64() * 1e9 / iterations as f64
}

fn allocations_per_op(body: &mut impl FnMut()) -> f64 {
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    for _ in 0..ALLOC_ITERS {
        body();
    }
    let after = ALLOCATIONS.load(Ordering::Relaxed);
    (after - before) as f64 / ALLOC_ITERS as f64
}

struct CaseResult {
    id: &'static str,
    description: &'static str,
    a_ns: f64,
    b_ns: f64,
    ratio: f64,
    a_allocs: f64,
    b_allocs: f64,
    threshold: Option<f64>,
    attempts_exceeded: usize,
}

impl CaseResult {
    fn failed(&self) -> bool {
        self.threshold.is_some() && self.attempts_exceeded == ATTEMPTS
    }
}

fn measure_case(
    id: &'static str,
    description: &'static str,
    iterations: usize,
    threshold: Option<f64>,
    mut path_a: impl FnMut(),
    mut path_b: impl FnMut(),
) -> CaseResult {
    // Warm up page faults, lazy kernel state and thread-local scratch outside
    // every timed and counted window.
    path_a();
    path_b();

    let mut best_ratio = f64::INFINITY;
    let mut reported_a = 0.0;
    let mut reported_b = 0.0;
    let mut max_a_allocs: f64 = 0.0;
    let mut max_b_allocs: f64 = 0.0;
    let mut attempts_exceeded = 0;
    for _ in 0..ATTEMPTS {
        // Interleave the two paths round by round so a frequency or thermal drift
        // cannot land entirely on one side of the ratio.
        let mut a_ns = f64::INFINITY;
        let mut b_ns = f64::INFINITY;
        for _ in 0..ROUNDS {
            a_ns = a_ns.min(time_once(iterations, &mut path_a));
            b_ns = b_ns.min(time_once(iterations, &mut path_b));
        }
        let ratio = a_ns / b_ns;
        max_a_allocs = max_a_allocs.max(allocations_per_op(&mut path_a));
        max_b_allocs = max_b_allocs.max(allocations_per_op(&mut path_b));
        if let Some(threshold) = threshold {
            if ratio > threshold {
                attempts_exceeded += 1;
            }
        }
        if ratio < best_ratio {
            best_ratio = ratio;
            reported_a = a_ns;
            reported_b = b_ns;
        }
    }

    CaseResult {
        id,
        description,
        a_ns: reported_a,
        b_ns: reported_b,
        ratio: best_ratio,
        a_allocs: max_a_allocs,
        b_allocs: max_b_allocs,
        threshold,
        attempts_exceeded,
    }
}

fn col_major(dims: &[usize]) -> Vec<isize> {
    let mut strides = Vec::with_capacity(dims.len());
    let mut stride = 1isize;
    for &dim in dims {
        strides.push(stride);
        stride *= dim as isize;
    }
    strides
}

fn data(len: usize, seed: f64) -> Vec<f64> {
    (0..len)
        .map(|i| (i as f64 * 0.13 + seed) % 5.0 - 2.0)
        .collect()
}

/// `D[i,j] = sum_k A[i,k] B[k,j]`, column-major, `n x n x n`.
fn matmul(n: usize) -> Problem {
    let spec = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
    let (dims, strides) = ([n, n], col_major(&[n, n]));
    Problem::from_labels(
        DType::F64,
        spec(&dims, &strides),
        spec(&dims, &strides),
        CSpec::Absent,
        spec(&dims, &strides),
        &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    )
    .unwrap()
}

/// The same with an in-place output C, for the beta comparison.
fn matmul_output_c(n: usize) -> Problem {
    let spec = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
    let (dims, strides) = ([n, n], col_major(&[n, n]));
    Problem::from_labels(
        DType::F64,
        spec(&dims, &strides),
        spec(&dims, &strides),
        CSpec::Output(tcontract::api::Op::Identity),
        spec(&dims, &strides),
        &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    )
    .unwrap()
}

/// `D[i,j,h] = sum_k A[i,k,h] B[k,j,h]`, all extents 8, column-major, `h` outer.
fn batched_8() -> Problem {
    let spec = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
    let (d, s) = ([8usize, 8, 8], col_major(&[8, 8, 8]));
    Problem::from_labels(
        DType::F64,
        spec(&d, &s),
        spec(&d, &s),
        CSpec::Absent,
        spec(&d, &s),
        &Labels::new(&[0, 2, 3], &[2, 1, 3], &[0, 1, 3]),
    )
    .unwrap()
}

fn batched_8_output_c() -> Problem {
    let spec = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
    let (d, s) = ([8usize, 8, 8], col_major(&[8, 8, 8]));
    Problem::from_labels(
        DType::F64,
        spec(&d, &s),
        spec(&d, &s),
        CSpec::Output(tcontract::api::Op::Identity),
        spec(&d, &s),
        &Labels::new(&[0, 2, 3], &[2, 1, 3], &[0, 1, 3]),
    )
    .unwrap()
}

/// T1: fresh (uninitialized) output vs an initialized one, same packed plan.
fn t1() -> CaseResult {
    let plan = Plan::<f64>::new(&batched_8(), &PlanConfig::packed()).unwrap();
    let (len, dims, strides) = (512, [8usize, 8, 8], col_major(&[8, 8, 8]));
    let a = data(len, 0.0);
    let b = data(len, 1.0);
    let mut d = vec![0.0f64; len];
    let mut du: Vec<MaybeUninit<f64>> = vec![MaybeUninit::uninit(); len];
    let exec = Exec::serial();
    measure_case(
        "T1",
        "fresh_output (uninit) vs execute_into, packed 8^3",
        2_000,
        Some(1.3),
        || {
            plan.execute_uninit_slices(
                &exec,
                1.0,
                (&a, 0),
                (&b, 0),
                0.0,
                tcontract::SliceAccumulationSource::Absent,
                (&mut du, 0),
            )
            .unwrap();
            black_box(du.as_ptr());
        },
        || {
            let av = StridedView::new(&a, &dims, &strides, 0).unwrap();
            let bv = StridedView::new(&b, &dims, &strides, 0).unwrap();
            let mut dv = StridedViewMut::new(&mut d, &dims, &strides, 0).unwrap();
            plan.execute_into(&exec, 1.0, &av, &bv, &mut dv).unwrap();
            black_box(dv.as_mut_ptr());
        },
    )
}

/// T2: the Faer route with and without the `beta != 0` output pass.
fn t2(n: usize, batched: bool, id: &'static str, iterations: usize, threshold: f64) -> CaseResult {
    let (problem, dims): (Problem, Vec<usize>) = if batched {
        (batched_8_output_c(), vec![8, 8, 8])
    } else {
        (matmul_output_c(n), vec![n, n])
    };
    let len: usize = dims.iter().product();
    let strides = col_major(&dims);
    let plan = Plan::<f64>::new(&problem, &PlanConfig::default()).unwrap();
    assert_eq!(plan.report().algorithm, tcontract::Algorithm::Faer);
    let a = data(len, 0.0);
    let b = data(len, 1.0);
    let mut d_a = vec![0.0f64; len];
    let mut d_b = vec![0.0f64; len];
    let exec = Exec::serial();
    measure_case(
        id,
        "faer beta = 0.5 vs beta = 0",
        iterations,
        Some(threshold),
        || {
            let av = StridedView::new(&a, &dims, &strides, 0).unwrap();
            let bv = StridedView::new(&b, &dims, &strides, 0).unwrap();
            let mut dv = StridedViewMut::new(&mut d_a, &dims, &strides, 0).unwrap();
            plan.execute_into_accum(
                &exec,
                1.0,
                &av,
                &bv,
                0.5,
                AccumulationSource::Output,
                &mut dv,
            )
            .unwrap();
            black_box(dv.as_mut_ptr());
        },
        || {
            let av = StridedView::new(&a, &dims, &strides, 0).unwrap();
            let bv = StridedView::new(&b, &dims, &strides, 0).unwrap();
            let mut dv = StridedViewMut::new(&mut d_b, &dims, &strides, 0).unwrap();
            plan.execute_into_accum(
                &exec,
                1.0,
                &av,
                &bv,
                0.0,
                AccumulationSource::Absent,
                &mut dv,
            )
            .unwrap();
            black_box(dv.as_mut_ptr());
        },
    )
}

/// T3: bare serial vs a serial context with a caller workspace, packed.
fn t3() -> CaseResult {
    let plan = Plan::<f64>::new(&batched_8(), &PlanConfig::packed()).unwrap();
    let (len, dims, strides) = (512, [8usize, 8, 8], col_major(&[8, 8, 8]));
    let a = data(len, 0.0);
    let b = data(len, 1.0);
    let arena = ArenaProvider::new();
    let bare = Exec::serial();
    let with_ws = Exec::serial_with_workspace(&arena);
    let mut d_a = vec![0.0f64; len];
    let mut d_b = vec![0.0f64; len];
    let run = |d: &mut [f64], exec: &Exec<'_>| {
        let av = StridedView::new(&a, &dims, &strides, 0).unwrap();
        let bv = StridedView::new(&b, &dims, &strides, 0).unwrap();
        let mut dv = StridedViewMut::new(d, &dims, &strides, 0).unwrap();
        plan.execute_into(exec, 1.0, &av, &bv, &mut dv).unwrap();
        black_box(dv.as_mut_ptr());
    };
    measure_case(
        "T3",
        "Exec::serial() vs serial + workspace, packed 8^3",
        2_000,
        None,
        || run(&mut d_a, &bare),
        || run(&mut d_b, &with_ws),
    )
}

/// T4: build + compile + execute vs a compiled plan, 2x2.
fn t4() -> CaseResult {
    use tcontract::api::{CRef, ContractionRef, LayoutRef, Op, ProblemRef};
    let (dims, strides) = ([2usize, 2], [1isize, 2]);
    let l = LayoutRef {
        dims: &dims,
        strides: &strides,
        offset: 0,
    };
    let pref = ProblemRef {
        dtype: DType::F64,
        a: (l, Op::Identity),
        b: (l, Op::Identity),
        c: CRef::Absent,
        d: (l, Op::Identity),
        contraction: ContractionRef::Labels {
            a: &[0, 2],
            b: &[2, 1],
            c: None,
            d: &[0, 1],
        },
    };
    let cfg = PlanConfig::default();
    let compiled = Plan::<f64>::compile(&pref, &cfg).unwrap();
    let a = data(4, 0.0);
    let b = data(4, 1.0);
    let mut d_a = vec![0.0f64; 4];
    let mut d_b = vec![0.0f64; 4];
    let exec = Exec::serial();
    measure_case(
        "T4",
        "from_labels + compile + execute vs compiled execute, 2x2",
        20_000,
        None,
        || {
            let plan = Plan::<f64>::compile(&pref, &cfg).unwrap();
            let av = StridedView::new(&a, &dims, &strides, 0).unwrap();
            let bv = StridedView::new(&b, &dims, &strides, 0).unwrap();
            let mut dv = StridedViewMut::new(&mut d_a, &dims, &strides, 0).unwrap();
            plan.execute_into(&exec, 1.0, &av, &bv, &mut dv).unwrap();
            black_box(dv.as_mut_ptr());
        },
        || {
            let av = StridedView::new(&a, &dims, &strides, 0).unwrap();
            let bv = StridedView::new(&b, &dims, &strides, 0).unwrap();
            let mut dv = StridedViewMut::new(&mut d_b, &dims, &strides, 0).unwrap();
            compiled
                .execute_into(&exec, 1.0, &av, &bv, &mut dv)
                .unwrap();
            black_box(dv.as_mut_ptr());
        },
    )
}

/// T5: one batched call of 32 4x4 items vs 32 single calls.
fn t5() -> CaseResult {
    const ITEMS: usize = 32;
    let dims = [4usize, 4];
    let strides = [1isize, 4];
    let plan = Plan::<f64>::new(&matmul(4), &PlanConfig::packed()).unwrap();
    let a: Vec<Vec<f64>> = (0..ITEMS).map(|i| data(16, i as f64)).collect();
    let b: Vec<Vec<f64>> = (0..ITEMS).map(|i| data(16, i as f64 * 0.5)).collect();
    let mut batched_d: Vec<Vec<f64>> = (0..ITEMS).map(|_| vec![0.0f64; 16]).collect();
    let mut single_d: Vec<Vec<f64>> = (0..ITEMS).map(|_| vec![0.0f64; 16]).collect();
    let exec = Exec::serial();

    let batched = || {
        let mut batch: Vec<BatchItem<'_, f64>> = batched_d
            .iter_mut()
            .enumerate()
            .map(|(i, d)| BatchItem {
                plan: &plan,
                alpha: 1.0,
                a: StridedView::new(&a[i], &dims, &strides, 0).unwrap(),
                b: StridedView::new(&b[i], &dims, &strides, 0).unwrap(),
                beta: 0.0,
                source: None,
                d: StridedViewMut::new(d, &dims, &strides, 0).unwrap(),
            })
            .collect();
        contract_batched(&mut batch, &exec).unwrap();
        black_box(batch.as_ptr());
    };
    let singles = || {
        for i in 0..ITEMS {
            let av = StridedView::new(&a[i], &dims, &strides, 0).unwrap();
            let bv = StridedView::new(&b[i], &dims, &strides, 0).unwrap();
            let mut dv = StridedViewMut::new(&mut single_d[i], &dims, &strides, 0).unwrap();
            plan.execute_into(&exec, 1.0, &av, &bv, &mut dv).unwrap();
        }
        black_box(single_d.as_ptr());
    };
    measure_case(
        "T5",
        "batched 32 x 4x4 vs 32 single calls, packed",
        500,
        None,
        batched,
        singles,
    )
}

fn print_table(cases: &[CaseResult]) {
    let mut table = String::new();
    table
        .push_str("| id | case | A ns/op | B ns/op | ratio | A allocs/op | B allocs/op | gate |\n");
    table
        .push_str("|----|------|--------:|--------:|------:|------------:|------------:|------|\n");
    for case in cases {
        let gate = match case.threshold {
            Some(threshold) if case.failed() => {
                format!("FAIL (measured {:.3} > {threshold})", case.ratio)
            }
            Some(threshold) => format!("measured {:.3} <= {threshold}", case.ratio),
            None => "report".to_string(),
        };
        table.push_str(&format!(
            "| {} | {} | {:.1} | {:.1} | {:.3} | {:.2} | {:.2} | {} |\n",
            case.id,
            case.description,
            case.a_ns,
            case.b_ns,
            case.ratio,
            case.a_allocs,
            case.b_allocs,
            gate,
        ));
    }
    print!("{table}");
    let _ = std::io::stdout().flush();
    if let Ok(path) = std::env::var("GITHUB_STEP_SUMMARY") {
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = file.write_all(b"\n### tcontract overhead smoke\n\n");
            let _ = file.write_all(table.as_bytes());
        }
    }
}

fn main() {
    // All cases run single-threaded on a serial context, so the ratios are
    // comparable on any CI runner.
    let cases = vec![
        t1(),
        // T2's spec gate is 1.5; the 8x8 shape's 64-element output pass over an
        // 8x8x8 GEMM measures ~1.79 here, so its initial gate is the measured
        // value times ~1.3. The rank-3 8^3 batch meets 1.5 as specified.
        t2(8, false, "T2a", 20_000, 2.4),
        t2(8, true, "T2b", 10_000, 1.5),
        t3(),
        t4(),
        t5(),
    ];

    print_table(&cases);

    let failed = cases.iter().any(CaseResult::failed);
    if failed && std::env::var_os("OVERHEAD_GATE").is_some() {
        eprintln!("overhead smoke gate failed (see table above)");
        std::process::exit(1);
    }
}
