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
use tcontract::api::{
    CRef, CSpec, ContractionRef, DType, Labels, LayoutRef, LayoutSpec, Op, OperandSpec, Problem,
    ProblemRef,
};
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
    // The counter is process-global while the enable flag is thread-local, so two
    // tests measuring at once would leak into each other's window; serialize the
    // windows instead of running the binary single-threaded.
    let _guard = COUNT_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    let c0 = COUNT.load(Relaxed);
    COUNTING.with(|c| c.set(true));
    let r = f();
    COUNTING.with(|c| c.set(false));
    (r, COUNT.load(Relaxed) - c0)
}

static COUNT_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

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

/// Borrowed construction: a rank-3 fusable problem must lower and compile
/// without a heap allocation (#121 item 1). The plan's own packed scatter tables
/// are the one exception and need a packed route; a Faer plan builds none.
#[test]
fn borrowed_construction_and_compile_are_allocation_free() {
    // D[i,j,h] = sum_k A[i,k,h] B[k,j,h], rank 3, fusable to faer.
    let dims = [8usize; 3];
    let strides = col_major(&dims);
    let (la, lb, ld) = ([0i64, 2, 3], [2i64, 1, 3], [0i64, 1, 3]);
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
            a: &la,
            b: &lb,
            c: None,
            d: &ld,
        },
    };

    // The first pass warms any one-time initialisation.
    let _ = Problem::from_ref(&pref).unwrap();
    let _ = Plan::<f64>::compile(&pref, &PlanConfig::default()).unwrap();

    let (problem, n_ref) = count(|| Problem::from_ref(&pref).unwrap());
    let (_, n_clone) = count(|| problem.clone());
    let (plan, n_compile) = count(|| Plan::<f64>::compile(&pref, &PlanConfig::default()).unwrap());
    let (_, n_new) = count(|| Plan::<f64>::new(&problem, &PlanConfig::default()).unwrap());
    assert_eq!(plan.report().algorithm, Algorithm::Faer);
    assert_eq!(n_ref, 0, "Problem::from_ref allocated {n_ref} times");
    assert_eq!(n_clone, 0, "Problem::clone allocated {n_clone} times");
    assert_eq!(n_compile, 0, "Plan::compile allocated {n_compile} times");
    // `Plan::new` clones the problem and compiles it; both are inline now.
    assert_eq!(n_new, 0, "Plan::new allocated {n_new} times");

    // The one documented exception: a forced packed plan builds the scatter
    // tables, whose length is the product of the role extents, not the rank.
    let (_, n_packed) = count(|| Plan::<f64>::compile(&pref, &PlanConfig::packed()).unwrap());
    assert!(
        n_packed > 0,
        "a packed plan should allocate its scatter tables"
    );
}

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
        // Since the inline-metadata change (#121 item 1), both are zero: the
        // operands, labels, role table and output pass all fit their inline
        // storage at this rank. Before it, 7 for the problem and 2 for the plan.
        if pass < steps.len() {
            continue;
        }
        assert_eq!(n_problem, 0, "problem: {n_problem} allocations");
        assert_eq!(n_plan, 0, "plan: {n_plan} allocations");
    }
}
