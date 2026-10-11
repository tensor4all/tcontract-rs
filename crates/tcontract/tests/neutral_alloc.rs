//! Crossing the neutral interface adds no allocation and no copy.
//!
//! Its own binary with one test: the counters are global. The same plan is
//! executed through the concrete `ContractPlan::execute` and through the
//! trait object on a serial host; the allocation count and bytes must match
//! exactly, so the trait boundary (adapter, dynamic dispatch, view checks)
//! allocates nothing, in particular no tensor-sized copy. The tensors are
//! large enough that a stray operand copy would show as bytes.
use std::alloc::{GlobalAlloc, Layout as AllocLayout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use tcontract::api::{
    AccumulationSource, ContractionBackend, DType, DotGeneral, LayoutSpec, OperandSpec,
    PlanningBudget, Problem, Requirements,
};
use tcontract::{Plan, PlanConfig, TcontractBackend};
use tcontract_exec::Exec;

static COUNT: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

struct Counting;

// SAFETY: forwards to the system allocator unchanged; counters are atomics.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: AllocLayout) -> *mut u8 {
        COUNT.fetch_add(1, Relaxed);
        BYTES.fetch_add(l.size(), Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: AllocLayout) {
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

fn measure(f: impl FnOnce()) -> (usize, usize) {
    let (c0, b0) = (COUNT.load(Relaxed), BYTES.load(Relaxed));
    f();
    (COUNT.load(Relaxed) - c0, BYTES.load(Relaxed) - b0)
}

#[test]
fn the_trait_path_allocates_exactly_what_the_concrete_path_does() {
    // C[i,m] = sum_{j,k} A[i,j,k] B[k,m,j]: the contracted axes are in different
    // orders, so nothing fuses and the planner runs the packed driver.
    let cfg = DotGeneral::new(&[1, 2], &[2, 0], &[], &[]);
    let (da, db, dc) = ([48usize, 40, 44], [44usize, 36, 40], [48usize, 36]);
    let col = |d: &[usize]| -> Vec<isize> {
        let mut s = vec![1isize];
        for &x in &d[..d.len() - 1] {
            s.push(s.last().unwrap() * x as isize);
        }
        s
    };
    let (sa, sb, sc) = (col(&da), col(&db), col(&dc));
    let a: Vec<f64> = (0..da.iter().product::<usize>())
        .map(|i| (i % 7) as f64 - 3.0)
        .collect();
    let b: Vec<f64> = (0..db.iter().product::<usize>())
        .map(|i| (i % 5) as f64 - 2.0)
        .collect();
    let mut c1 = vec![0.0f64; dc.iter().product()];
    let mut c2 = c1.clone();
    let tensor_bytes = a.len() * 8;
    let spec = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
    let problem = Problem::from_dot_general(
        DType::F64,
        spec(&da, &sa),
        spec(&db, &sb),
        spec(&dc, &sc),
        &cfg,
    )
    .unwrap();

    let forced = PlanConfig::packed();
    for config in [PlanConfig::default(), forced] {
        let plan = Plan::<f64>::new(&problem, &config).unwrap();
        let backend = TcontractBackend {
            config: config.clone(),
        };
        let boxed = ContractionBackend::<f64>::prepare(
            &backend,
            &problem,
            &Requirements::new(),
            &PlanningBudget::serial(),
        )
        .unwrap();
        let (av, bv) = (
            strided_view::StridedView::new(&a, &da, &sa, 0).unwrap(),
            strided_view::StridedView::new(&b, &db, &sb, 0).unwrap(),
        );
        let exec = Exec::serial();
        let run_concrete = |c: &mut Vec<f64>| {
            let mut cv = strided_view::StridedViewMut::new(c, &dc, &sc, 0).unwrap();
            plan.execute_into(&exec, 1.0, &av, &bv, &mut cv).unwrap();
        };
        let run_trait = |c: &mut Vec<f64>| {
            let mut cv = strided_view::StridedViewMut::new(c, &dc, &sc, 0).unwrap();
            boxed
                .execute_into_accum(
                    &exec,
                    1.0,
                    &av,
                    &bv,
                    0.0,
                    AccumulationSource::Output,
                    &mut cv,
                )
                .unwrap();
        };
        run_concrete(&mut c1); // warm-up: lazy statics, first-use tables
        run_trait(&mut c2);
        let concrete = measure(|| run_concrete(&mut c1));
        let through_trait = measure(|| run_trait(&mut c2));
        assert_eq!(through_trait, concrete, "{config:?}");
        assert_eq!(c1, c2, "{config:?}");
        // Nothing is copied: no allocation approaches an operand's size.
        assert!(concrete.1 < tensor_bytes / 2, "{concrete:?}");
        assert_eq!(boxed.diagnostics().materialized, [false; 3]);
    }
}
