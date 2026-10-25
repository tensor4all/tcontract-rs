//! The plain `D = beta * C` output pass replays a traversal compiled at plan
//! time (#121 item 2): no per-execute strided planning, and a fresh destination
//! is written without being read.
//!
//! Its own binary with the counting allocator.
use std::alloc::{GlobalAlloc, Layout as AllocLayout, System};
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use tcontract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
use tcontract::{Plan, PlanConfig, SliceAccumulationSource};
use tcontract_exec::{ArenaProvider, Exec};

static COUNT: AtomicUsize = AtomicUsize::new(0);

struct Counting;

// SAFETY: every method forwards to the system allocator unchanged; the counter
// is an atomic with no aliasing.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: AllocLayout) -> *mut u8 {
        COUNT.fetch_add(1, Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: AllocLayout) {
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// The counter is process-global and the test harness runs tests on parallel
/// threads, so measured regions run one at a time.
static MEASURE: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn measured() -> std::sync::MutexGuard<'static, ()> {
    MEASURE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

const N: usize = 4;

/// `D[i,j] = sum_k A[i,k] B[k,j] + beta * C[i,j]`, column-major, C separate.
fn problem() -> Problem {
    let spec = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
    let (dims, strides) = ([N, N], [1isize, N as isize]);
    Problem::from_labels(
        DType::F64,
        spec(&dims, &strides),
        spec(&dims, &strides),
        CSpec::Separate(spec(&dims, &strides)),
        spec(&dims, &strides),
        &Labels::new(&[0, 2], &[2, 1], &[0, 1]).with_c(&[0, 1]),
    )
    .unwrap()
}

#[test]
fn beta_pass_is_allocation_free_and_fresh_d_is_not_read() {
    let _measured = measured();
    let config = PlanConfig {
        fresh_output: true,
        ..PlanConfig::default()
    };
    let plan = Plan::<f64>::new(&problem(), &config).unwrap();
    let arena = ArenaProvider::new();
    let mut exec = Exec::serial_with_workspace(&arena);

    let a = vec![1.0f64; N * N];
    let b = vec![1.0f64; N * N];
    let c: Vec<f64> = (0..N * N).map(|i| i as f64 * 0.5 - 1.0).collect();
    let mut d = vec![0.0f64; N * N];

    // alpha = 0 takes the output-only pass: D = beta * C. Warm the arena.
    let mut run = |d: &mut [f64]| {
        plan.execute_slices_accum(
            &mut exec,
            0.0,
            (&a, 0),
            (&b, 0),
            0.5,
            SliceAccumulationSource::Separate((&c, 0)),
            (d, 0),
        )
        .unwrap();
    };
    run(&mut d);
    let before = COUNT.load(Relaxed);
    run(&mut d);
    let allocs = COUNT.load(Relaxed) - before;
    assert_eq!(allocs, 0, "the output pass allocated {allocs} times");

    // A fresh (NaN-poisoned) destination is written, never read: D = 0.5 * C.
    let mut poison: [MaybeUninit<f64>; N * N] = [MaybeUninit::new(f64::NAN); N * N];
    let initialized = plan
        .execute_uninit_slices(
            &mut exec,
            0.0,
            (&a, 0),
            (&b, 0),
            0.5,
            SliceAccumulationSource::Separate((&c, 0)),
            (&mut poison, 0),
        )
        .unwrap();
    assert!(initialized.iter().all(|x| x.is_finite()), "{initialized:?}");
    for (got, want) in initialized.iter().zip(&c) {
        assert_eq!(*got, 0.5 * want);
    }
}

/// A fresh output on the faer route (tcontract 0.4.3) is zero-filled by the
/// output pass before the product; once warm, that pass allocates nothing, so
/// a repeated allocating contraction stays allocation-free (tenferro's eager
/// plan-reuse contract).
#[test]
fn fresh_faer_route_is_allocation_free_once_warm() {
    let _measured = measured();
    let spec = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
    let (dims, strides) = ([N, N], [1isize, N as isize]);
    let problem = Problem::from_labels(
        DType::F64,
        spec(&dims, &strides),
        spec(&dims, &strides),
        CSpec::Absent,
        spec(&dims, &strides),
        &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    )
    .unwrap();
    let plan = Plan::<f64>::new(&problem, &PlanConfig::default()).unwrap();
    assert_eq!(plan.report().algorithm, tcontract::Algorithm::Faer);
    let arena = ArenaProvider::new();
    let mut exec = Exec::serial_with_workspace(&arena);
    let a = vec![1.0f64; N * N];
    let b = vec![1.0f64; N * N];
    let mut d: [MaybeUninit<f64>; N * N] = [MaybeUninit::uninit(); N * N];
    let mut run = |d: &mut [MaybeUninit<f64>; N * N]| {
        let out = plan
            .execute_uninit_slices(
                &mut exec,
                1.0,
                (&a, 0),
                (&b, 0),
                0.0,
                SliceAccumulationSource::Absent,
                (d, 0),
            )
            .unwrap();
        assert!(out.iter().all(|&x| x == N as f64));
    };
    run(&mut d);
    let before = COUNT.load(Relaxed);
    run(&mut d);
    let allocs = COUNT.load(Relaxed) - before;
    assert_eq!(allocs, 0, "the fresh faer route allocated {allocs} times");
}
