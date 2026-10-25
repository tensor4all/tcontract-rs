//! A parallel steady state is bounded, not zero: the Rayon dispatch itself
//! allocates a fixed number of bookkeeping objects, and that number must not
//! grow with the size or the batch of the contraction (#121).
//!
//! Its own binary because the counter is global and the measurement must not
//! race another test's allocations. The workspace is the pool's own arena, so
//! the count covers the driver's reuse plus whatever Rayon allocates to dispatch
//! the lanes; the assertion is that the larger case does not exceed the smaller
//! one, and that both stay under a fixed dispatch constant.
use std::alloc::{GlobalAlloc, Layout as AllocLayout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use strided_view::{StridedView, StridedViewMut};
use tcontract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
use tcontract::{Plan, PlanConfig};
use tcontract_exec::{Exec, Pool};

mod common;

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

/// `D[m,n,h] = sum_k A[m,k,h] B[k,n,h]`, column-major, `h` outermost.
fn batched(m: usize, n: usize, k: usize, h: usize) -> Problem {
    let spec = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
    let (mi, ni, ki) = (m as isize, n as isize, k as isize);
    Problem::from_labels(
        DType::F64,
        spec(&[m, k, h], &[1, mi, mi * ki]),
        spec(&[k, n, h], &[1, ki, ki * ni]),
        CSpec::Absent,
        spec(&[m, n, h], &[1, mi, mi * ni]),
        &Labels::new(&[0, 2, 3], &[2, 1, 3], &[0, 1, 3]),
    )
    .unwrap()
}

struct Case {
    plan: Plan<f64>,
    dims: [usize; 4],
    a: Vec<f64>,
    b: Vec<f64>,
    d: Vec<f64>,
}

impl Case {
    fn new(m: usize, n: usize, k: usize, h: usize) -> Self {
        let plan = Plan::<f64>::new(&batched(m, n, k, h), &PlanConfig::packed()).unwrap();
        let data = |len: usize, seed: f64| {
            (0..len)
                .map(|i| ((i as f64 * 0.31 + seed) % 5.0) - 2.0)
                .collect()
        };
        Self {
            plan,
            dims: [m, n, k, h],
            a: data(m * k * h, 0.0),
            b: data(k * n * h, 1.0),
            d: vec![0.0; m * n * h],
        }
    }

    /// One execute on `exec`, returning the allocations the library and its
    /// dispatch made. Views are built before the count starts.
    fn run(&mut self, exec: &mut Exec<'_>) -> usize {
        let [m, n, k, h] = self.dims;
        let (mi, ni, ki) = (m as isize, n as isize, k as isize);
        let av = StridedView::new(&self.a, &[m, k, h], &[1, mi, mi * ki], 0).unwrap();
        let bv = StridedView::new(&self.b, &[k, n, h], &[1, ki, ki * ni], 0).unwrap();
        let mut dv = StridedViewMut::new(&mut self.d, &[m, n, h], &[1, mi, mi * ni], 0).unwrap();
        let before = COUNT.load(Relaxed);
        self.plan
            .execute_into(exec, 1.0, &av, &bv, &mut dv)
            .unwrap();
        COUNT.load(Relaxed) - before
    }
}

/// A fixed ceiling for the per-call dispatch books Rayon keeps. It bounds the
/// count independent of the contraction, which is the property under test.
const DISPATCH_ALLOCS: usize = 24;

#[test]
fn parallel_allocations_do_not_grow_with_size_or_batch() {
    let sp = common::placement::pool(4);
    let pool = Pool::borrow(&sp);
    let mut exec = Exec::new(&pool, 4).unwrap();
    let root = common::placement::root(&mut exec);

    // Two batches at the same tiny item, four times apart. Both narrow to the
    // same full budget, so the lanes are equal and the only variable is the
    // batch size the dispatch does *not* see in its allocation count.
    let mut small = Case::new(8, 8, 8, 2048);
    let mut large = Case::new(8, 8, 8, 8192);
    // Warm the arena (worker slots, panel, scatter) outside the counted window.
    small.run(&mut exec);
    large.run(&mut exec);

    let small_allocs = small.run(&mut exec);
    let large_allocs = large.run(&mut exec);
    assert!(
        common::placement::installs(root) >= 1,
        "the batch never left the caller; this is not a parallel case"
    );
    assert!(
        small_allocs <= DISPATCH_ALLOCS,
        "a parallel steady state allocated {small_allocs} times"
    );
    assert!(
        large_allocs <= small_allocs,
        "allocations grew with the batch: {small_allocs} -> {large_allocs}"
    );

    // A larger item must not change the dispatch bound either.
    let mut bigger = Case::new(120, 120, 120, 8);
    bigger.run(&mut exec);
    let bigger_allocs = bigger.run(&mut exec);
    assert!(
        bigger_allocs <= DISPATCH_ALLOCS,
        "a larger item allocated {bigger_allocs} times"
    );
}
