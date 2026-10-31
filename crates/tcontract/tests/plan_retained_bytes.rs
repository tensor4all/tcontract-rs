//! `Plan::retained_bytes` equals what building the plan left allocated
//! (tcontract-rs#164), exactly, so a layout change that the query misses fails
//! here rather than drifting a caller's byte budget.
//!
//! A counting allocator tracks this thread's live bytes. A plan built inside
//! the window from inputs also built there (`Plan::from_problem`) or from a
//! problem built outside it (`Plan::new`, which copies the problem) leaves
//! exactly its own heap live; `retained_bytes` adds the inline header.
use std::alloc::{GlobalAlloc, Layout as AllocLayout, System};
use std::cell::Cell;

use num_complex::{Complex32, Complex64};
use tcontract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem, Scalar};
use tcontract::{Algorithm, Plan, PlanConfig};

thread_local! {
    // Const-initialised, so touching them from the allocator never allocates.
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static LIVE: Cell<isize> = const { Cell::new(0) };
}

struct Counting;

// SAFETY: forwards to the system allocator unchanged; the counters are
// thread-local cells.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: AllocLayout) -> *mut u8 {
        if COUNTING.with(Cell::get) {
            LIVE.with(|c| c.set(c.get() + l.size() as isize));
        }
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: AllocLayout) {
        if COUNTING.with(Cell::get) {
            LIVE.with(|c| c.set(c.get() - l.size() as isize));
        }
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// The bytes `f` leaves live on this thread.
fn live<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let l0 = LIVE.with(Cell::get);
    COUNTING.with(|c| c.set(true));
    let r = f();
    COUNTING.with(|c| c.set(false));
    let net = LIVE.with(Cell::get) - l0;
    (
        r,
        usize::try_from(net).expect("construction freed more than it allocated"),
    )
}

fn col_major(dims: &[usize]) -> Vec<isize> {
    dims.iter()
        .scan(1isize, |acc, &d| {
            let s = *acc;
            *acc *= d as isize;
            Some(s)
        })
        .collect()
}

/// One problem: extents, strides and labels of A, B and D, a separate C with
/// D's labels and the given strides, and the config.
struct Case {
    name: &'static str,
    a: (Vec<usize>, Vec<isize>, Vec<i64>),
    b: (Vec<usize>, Vec<isize>, Vec<i64>),
    d: (Vec<usize>, Vec<isize>, Vec<i64>),
    c: Option<Vec<isize>>,
    config: PlanConfig,
    /// The expected algorithm, as a function of the dtype.
    algorithm: fn(DType) -> Algorithm,
}

impl Case {
    fn problem<T: Scalar>(&self) -> Problem {
        let spec = |(d, s, _): &(Vec<usize>, Vec<isize>, Vec<i64>)| {
            OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap())
        };
        let mut labels = Labels::new(&self.a.2, &self.b.2, &self.d.2);
        let c = match &self.c {
            Some(sc) => {
                labels = labels.with_c(&self.d.2);
                CSpec::Separate(OperandSpec::new(LayoutSpec::new(&self.d.0, sc, 0).unwrap()))
            }
            None => CSpec::Absent,
        };
        Problem::from_labels(
            T::STORAGE,
            spec(&self.a),
            spec(&self.b),
            c,
            spec(&self.d),
            &labels,
        )
        .unwrap()
    }

    fn check<T: Scalar>(&self) {
        // Warm one-time initialisation (host detection, kernel registry).
        drop(Plan::<T>::new(&self.problem::<T>(), &self.config).unwrap());

        let (owned, bytes) =
            live(|| Plan::<T>::from_problem(self.problem::<T>(), &self.config).unwrap());
        assert_eq!(
            owned.report().algorithm,
            (self.algorithm)(T::STORAGE),
            "{}",
            self.name
        );
        assert_eq!(
            owned.retained_bytes(),
            size_of::<Plan<T>>() + bytes,
            "{} {:?} from_problem",
            self.name,
            T::STORAGE
        );

        let problem = self.problem::<T>();
        let (copied, bytes) = live(|| Plan::<T>::new(&problem, &self.config).unwrap());
        assert_eq!(
            copied.retained_bytes(),
            size_of::<Plan<T>>() + bytes,
            "{} {:?} new",
            self.name,
            T::STORAGE
        );
    }
}

fn operand(dims: &[usize], labels: &[i64]) -> (Vec<usize>, Vec<isize>, Vec<i64>) {
    (dims.to_vec(), col_major(dims), labels.to_vec())
}

/// faer, except that a c64 GEMM volume above the default `FaerLimit` runs packed.
fn faer_below_c64_limit(dtype: DType) -> Algorithm {
    match dtype {
        DType::C64 => Algorithm::Packed,
        _ => Algorithm::Faer,
    }
}

fn cases() -> Vec<Case> {
    let seq = |n: i64| (0..n).collect::<Vec<i64>>();
    // Rank 17, extent 2: every metadata vector spills its inline storage.
    let wide = [2usize; 17];
    // Strides 3^i: no two axes fuse, so the output replay's traversal spills too.
    let gapped: Vec<isize> = (0..17u32).map(|i| 3isize.pow(i)).collect();
    vec![
        Case {
            name: "rank-2 GEMM",
            a: operand(&[3, 4], &[0, 2]),
            b: operand(&[4, 5], &[2, 1]),
            d: operand(&[3, 5], &[0, 1]),
            c: None,
            config: PlanConfig::default(),
            algorithm: |_| Algorithm::Faer,
        },
        Case {
            name: "rank-3 batched GEMM",
            a: operand(&[8, 6, 3], &[0, 2, 3]),
            b: operand(&[6, 7, 3], &[2, 1, 3]),
            d: operand(&[8, 7, 3], &[0, 1, 3]),
            c: None,
            config: PlanConfig::default(),
            algorithm: |_| Algorithm::Faer,
        },
        Case {
            name: "rank-3 batched GEMM, packed",
            a: operand(&[8, 6, 3], &[0, 2, 3]),
            b: operand(&[6, 7, 3], &[2, 1, 3]),
            d: operand(&[8, 7, 3], &[0, 1, 3]),
            c: Some(col_major(&[8, 7, 3])),
            config: PlanConfig::packed(),
            algorithm: |_| Algorithm::Packed,
        },
        Case {
            // D's two M axes in the opposite order to A's: no copy-free fusion.
            name: "unfusable rank-3",
            a: operand(&[4, 3, 5], &[0, 1, 2]),
            b: operand(&[5, 6], &[2, 3]),
            d: operand(&[3, 4, 6], &[1, 0, 3]),
            c: None,
            config: PlanConfig::default(),
            algorithm: |_| Algorithm::Packed,
        },
        Case {
            name: "rank-3 all-batch",
            a: operand(&[4, 3, 5], &[0, 1, 2]),
            b: operand(&[4, 3, 5], &[0, 1, 2]),
            d: operand(&[4, 3, 5], &[0, 1, 2]),
            c: None,
            config: PlanConfig::default(),
            algorithm: |_| Algorithm::Elementwise,
        },
        Case {
            name: "compact rank-17 all-batch",
            a: operand(&wide, &seq(17)),
            b: operand(&wide, &seq(17)),
            d: operand(&wide, &seq(17)),
            c: None,
            config: PlanConfig::default(),
            algorithm: |_| Algorithm::Elementwise,
        },
        Case {
            name: "gapped rank-17 all-batch, separate C",
            a: operand(&wide, &seq(17)),
            b: operand(&wide, &seq(17)),
            d: (wide.to_vec(), gapped.clone(), seq(17)),
            c: Some(gapped.clone()),
            config: PlanConfig::default(),
            algorithm: |_| Algorithm::Elementwise,
        },
        Case {
            name: "gapped rank-17 all-batch, packed",
            a: operand(&wide, &seq(17)),
            b: operand(&wide, &seq(17)),
            d: (wide.to_vec(), gapped, seq(17)),
            c: None,
            config: PlanConfig::packed(),
            algorithm: |_| Algorithm::Packed,
        },
        Case {
            // Nine M and nine N axes: a rank-18 output and volume 2^19.
            name: "rank-18 output contraction",
            a: operand(&[2; 10], &(0..10).collect::<Vec<_>>()),
            b: operand(
                &[2; 10],
                &[[9i64].as_slice(), &(10..19).collect::<Vec<_>>()].concat(),
            ),
            d: operand(
                &[2; 18],
                &[(0..9).collect::<Vec<_>>(), (10..19).collect()].concat(),
            ),
            c: None,
            config: PlanConfig::default(),
            algorithm: faer_below_c64_limit,
        },
    ]
}

#[test]
fn retained_bytes_equals_live_heap_f32() {
    cases().iter().for_each(Case::check::<f32>);
}

#[test]
fn retained_bytes_equals_live_heap_f64() {
    cases().iter().for_each(Case::check::<f64>);
}

#[test]
fn retained_bytes_equals_live_heap_c32() {
    cases().iter().for_each(Case::check::<Complex32>);
}

#[test]
fn retained_bytes_equals_live_heap_c64() {
    cases().iter().for_each(Case::check::<Complex64>);
}
