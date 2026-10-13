//! A steady-state execute must not allocate.
//!
//! Its own binary, because the counter is global: another test running in
//! parallel would show up in the count. The `Exec` is serial and the test owns
//! the workspace it lends, so the measurement covers the driver's own reuse
//! path — the leased team set, worker buffers and scatter vectors, and the
//! caller-owned provider — rather than thread plumbing. A threaded host's first
//! touch and per-thread slot reuse are pinned in `tcontract-exec/tests/workspace.rs`.
use std::alloc::{GlobalAlloc, Layout as AllocLayout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use strided_view::{StridedView, StridedViewMut};
use tcontract::{Plan, PlanConfig};
use tcontract_exec::{ArenaProvider, Exec};
use tcontract_kernel::KernelChoice;

mod common;
use common::plans::matmul_problem;

static COUNT: AtomicUsize = AtomicUsize::new(0);
static BIG: AtomicUsize = AtomicUsize::new(0);
/// Allocations at or above this size are "a buffer", not incidental bookkeeping.
const BIG_BYTES: usize = 1 << 14;

struct Counting;

// SAFETY: every method forwards to the system allocator unchanged; the
// counters are atomics with no aliasing.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: AllocLayout) -> *mut u8 {
        COUNT.fetch_add(1, Relaxed);
        if l.size() >= BIG_BYTES {
            BIG.fetch_add(1, Relaxed);
        }
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: AllocLayout) {
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// One case's plan and operands, built before anything is measured.
struct Prepared {
    plan: Plan<f64>,
    /// The serial caller's own scratch owner: the plan owns none, so a warm
    /// steady state is the caller's to keep.
    arena: ArenaProvider,
    dims: [usize; 3],
    a: Vec<f64>,
    b: Vec<f64>,
    d: Vec<f64>,
}

impl Prepared {
    fn new(id: &str, m: usize, n: usize, k: usize) -> Self {
        let config = PlanConfig {
            kernel: KernelChoice::Id(id.into()),
            ..PlanConfig::default()
        };
        let plan = Plan::<f64>::new(&matmul_problem(m, n, k), &config).unwrap();
        let data = |len: usize, seed: f64| {
            (0..len)
                .map(|i| ((i as f64 * 0.37 + seed) % 3.0) - 1.0)
                .collect()
        };
        Self {
            plan,
            arena: ArenaProvider::new(),
            dims: [m, n, k],
            a: data(m * k, 0.0),
            b: data(k * n, 1.0),
            d: vec![0.0; m * n],
        }
    }

    /// One execute, returning the allocations the library itself made. The
    /// views borrow the layouts built here, so they allocate nothing and are
    /// built before the count starts either way.
    fn run(&mut self) -> usize {
        let [m, n, k] = self.dims;
        let (a_dims, a_strides) = ([m, k], [1isize, m as isize]);
        let (b_dims, b_strides) = ([k, n], [1isize, k as isize]);
        let (d_dims, d_strides) = ([m, n], [1isize, m as isize]);
        let av = StridedView::new(&self.a, &a_dims, &a_strides, 0).unwrap();
        let bv = StridedView::new(&self.b, &b_dims, &b_strides, 0).unwrap();
        let mut dv = StridedViewMut::new(&mut self.d, &d_dims, &d_strides, 0).unwrap();
        let before = COUNT.load(Relaxed);
        // The plan owns no scratch; a serial caller that wants a steady state
        // with no allocation lends its own provider.
        let exec = Exec::serial_with_workspace(&self.arena);
        self.plan
            .execute_into(&exec, 1.5, &av, &bv, &mut dv)
            .unwrap();
        COUNT.load(Relaxed) - before
    }
}

const SHAPE: (usize, usize, usize) = (300, 300, 300);

#[test]
fn steady_state_execute_allocates_nothing() {
    let (m, n, k) = SHAPE;

    let mut prepared = Prepared::new("ref.f64.real-scalar.4x4", m, n, k);
    prepared.run();
    assert_eq!(prepared.run(), 0, "a steady-state execute allocated");
    assert_eq!(prepared.run(), 0, "a steady-state execute allocated");

    // A direct-B family must not ask for a B-sized buffer even on its first run:
    // the only large allocation allowed there is the packed A block.
    let mut direct_b = Prepared::new("ref.f64.direct-b.4x4", m, n, k);
    let big = BIG.load(Relaxed);
    direct_b.run();
    assert!(
        BIG.load(Relaxed) - big <= 1,
        "direct-B allocated {} large buffers; only the A block is expected",
        BIG.load(Relaxed) - big
    );
    assert_eq!(direct_b.run(), 0, "steady state allocated");
    assert_eq!(direct_b.run(), 0, "steady state allocated");
}
