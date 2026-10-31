//! The faer route's per-execute allocation residual, pinned as an executable
//! upper bound (#121 item 3).
//!
//! The gemm-common packing scratch (`MemBuffer` and the `mc / MR` mask) is a
//! residual of faer's non-SIMD fallback; #121 item 3 defers removing it up the
//! publish chain, and #162 accepted it as a known limitation of the targets
//! that take that fallback. This binary measures the count per execute, fails
//! when it grows past the pinned bound, and on x86_64 with AVX2 asserts the
//! stronger contract the native kernels give: zero.
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
    // The bound above is the fallback's residual. Where the native kernels run
    // there is no packing scratch at all, and a warm execute allocates nothing
    // (#162): that is the contract this test exists to keep from regressing.
    // The same feature test t4a-faer itself makes before it takes the native
    // kernels (`src/linalg/matmul/mod.rs`, AVX-512 or AVX2 + FMA).
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx512f")
        || (std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma"))
    {
        assert_eq!(
            allocs, 0,
            "faer execute on the x86_64 native kernels allocated {allocs} times, not zero"
        );
    }
}
