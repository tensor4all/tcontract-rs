//! The faer route's per-execute allocation residual, pinned as an executable
//! upper bound (#121 item 3).
//!
//! The gemm-common packing scratch (`MemBuffer` and the `mc / MR` mask) is a
//! residual the faer fork owns; #121 item 3 defers removing it up the publish
//! chain. Until then this binary measures the count per execute and fails when
//! it grows past the pinned bound, so the deferral cannot quietly become a
//! regression.
use std::alloc::{GlobalAlloc, Layout as AllocLayout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use strided_view::{StridedView, StridedViewMut};
use tcontract::{Plan, PlanConfig};
use tcontract_exec::{ArenaProvider, Exec};

mod common;
use common::plans::matmul_problem;

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

/// The deferral's documented residual is two allocations per GEMM that packs
/// (gemm-common's `MemBuffer` and mask). Zero was measured on the host this
/// landed on, over 32^3 / 64^3 / 128x128x8 / 300^3; the bound keeps a small
/// ceiling so another host's faer still passes while a regression toward the
/// per-call packing scratch fails.
const FAER_ALLOC_UPPER_BOUND: usize = 2;

#[test]
fn faer_execute_stays_within_the_pinned_allocation_bound() {
    // A shape that fuses copy-free to faer (column-major, no separate C).
    let (m, n, k) = (32usize, 24, 16);
    let plan = Plan::<f64>::new(&matmul_problem(m, n, k), &PlanConfig::default()).unwrap();
    assert_eq!(plan.report().algorithm, tcontract::Algorithm::Faer);

    let arena = ArenaProvider::new();
    let mut exec = Exec::serial_with_workspace(&arena);
    let data = |len: usize, seed: f64| {
        (0..len)
            .map(|i| ((i as f64 * 0.13 + seed) % 4.0) - 1.5)
            .collect::<Vec<f64>>()
    };
    let a = data(m * k, 0.0);
    let b = data(k * n, 1.0);
    let mut d = vec![0.0f64; m * n];

    let mut run = |d: &mut [f64]| {
        let av = StridedView::new(&a, &[m, k], &[1, m as isize], 0).unwrap();
        let bv = StridedView::new(&b, &[k, n], &[1, k as isize], 0).unwrap();
        let mut dv = StridedViewMut::new(d, &[m, n], &[1, m as isize], 0).unwrap();
        plan.execute_into(&mut exec, 1.0, &av, &bv, &mut dv)
            .unwrap();
    };

    // Warm any one-time state outside the counted window.
    run(&mut d);
    let before = COUNT.load(Relaxed);
    run(&mut d);
    let allocs = COUNT.load(Relaxed) - before;
    assert!(
        allocs <= FAER_ALLOC_UPPER_BOUND,
        "faer execute allocated {allocs} times, above the pinned {FAER_ALLOC_UPPER_BOUND}"
    );
}
