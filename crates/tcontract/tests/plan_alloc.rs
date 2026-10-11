//! The planning path allocates a bounded handful of buffers (cpueinsum-rs's
//! per-call gate, tcontract-rs#62).
//!
//! Its own binary with one test, and only the counting thread's allocations
//! are counted, so the harness's own threads do not leak in. A per-call einsum
//! builds a `Problem` and a `Plan` for every binary step, so the number of
//! heap allocations there is the planning cost that matters at small extents.
//! `Plan::from_problem` keeps the problem instead of copying it and must agree
//! with `Plan::new`.
use std::alloc::{GlobalAlloc, Layout as AllocLayout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use num_complex::Complex64;
use tcontract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
use tcontract::{Algorithm, Plan, PlanConfig};

static COUNT: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    // Const-initialised, so reading it from the allocator never allocates.
    static COUNTING: Cell<bool> = const { Cell::new(false) };
}

struct Counting;

// SAFETY: forwards to the system allocator unchanged; the counter is atomic.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: AllocLayout) -> *mut u8 {
        if COUNTING.with(Cell::get) {
            COUNT.fetch_add(1, Relaxed);
        }
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: AllocLayout) {
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

fn count<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let c0 = COUNT.load(Relaxed);
    COUNTING.with(|c| c.set(true));
    let r = f();
    COUNTING.with(|c| c.set(false));
    (r, COUNT.load(Relaxed) - c0)
}

fn col_major(dims: &[usize]) -> Vec<isize> {
    let mut s = Vec::new();
    let mut acc = 1isize;
    for &d in dims {
        s.push(acc);
        acc *= d as isize;
    }
    s
}

/// Extents of A, B and D, then their labels.
type Step = (
    &'static [usize],
    &'static [usize],
    &'static [usize],
    &'static [i64],
    &'static [i64],
    &'static [i64],
);

#[test]
fn planning_a_small_mps_step_allocates_a_bounded_handful() {
    // The two MPS steps `ab,asc->bsc` and `bsc,bsd->cd` at chi = 4.
    let steps: [Step; 2] = [
        (
            &[4, 4],
            &[4, 2, 4],
            &[4, 2, 4],
            &[0, 1],
            &[0, 2, 3],
            &[1, 2, 3],
        ),
        (
            &[4, 2, 4],
            &[4, 2, 4],
            &[4, 4],
            &[1, 2, 3],
            &[1, 2, 4],
            &[3, 4],
        ),
    ];
    // The first pass warms any one-time initialisation; the second is checked.
    for (pass, (da, db, dd, la, lb, ld)) in steps.iter().chain(steps.iter()).enumerate() {
        let (da, db, dd, la, lb, ld) = (*da, *db, *dd, *la, *lb, *ld);
        let (sa, sb, sd) = (col_major(da), col_major(db), col_major(dd));
        let labels = Labels::new(la, lb, ld);
        let spec = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
        let (a, b, d) = (spec(da, &sa), spec(db, &sb), spec(dd, &sd));
        let (problem, n_problem) =
            count(|| Problem::from_labels(DType::C64, a, b, CSpec::Absent, d, &labels).unwrap());
        let by_ref = Plan::<Complex64>::new(&problem, &PlanConfig::default()).unwrap();
        let (plan, n_plan) =
            count(|| Plan::<Complex64>::from_problem(problem, &PlanConfig::default()).unwrap());
        assert_eq!(plan.report().algorithm, Algorithm::Faer);
        assert_eq!(plan.report(), by_ref.report());
        assert_eq!(plan.problem(), by_ref.problem());
        // Measured 2026-10-05 on both steps: 7 for the problem (three reduced
        // operands, the label table, three roles) and 2 for the plan (the
        // output pass). Before the planning-cost change, 9 and 32 (`Plan::new`).
        if pass < steps.len() {
            continue;
        }
        assert!(n_problem <= 7, "problem: {n_problem} allocations");
        assert!(n_plan <= 2, "plan: {n_plan} allocations");
    }
}
