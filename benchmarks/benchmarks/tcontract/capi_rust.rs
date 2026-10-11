//! The Rust side of `benchmarks/c/bench.c`: the same calls made directly
//! from Rust (no ABI), per call, at an enforced thread count.
use std::hint::black_box;
use std::time::Instant;

use strided_view::{StridedView, StridedViewMut};
use tcontract::api::{DType, DotGeneral, LayoutSpec, OperandSpec, Problem};
use tcontract::{Plan, PlanConfig};
use tcontract_bench::threads::BenchThreads;

const INNER: usize = 1000;
const SAMPLES: usize = 101;

fn median(mut f: impl FnMut()) -> f64 {
    let mut s: Vec<f64> = (0..SAMPLES)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..INNER {
                f();
            }
            t.elapsed().as_nanos() as f64 / INNER as f64
        })
        .collect();
    s.sort_by(f64::total_cmp);
    s[SAMPLES / 2]
}

fn main() {
    let th = BenchThreads::from_args();
    th.verify();
    let t = th.requested;
    println!("case,variant,threads,median_ns,samples");
    th.with_exec(|exec, _| {
        let ns = median(|| {
            black_box(tcontract_capi::ABI_VERSION);
        });
        println!("empty_call,rust,{t},{ns:.1},{SAMPLES}");
        let x: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..8).map(|i| 8.0 - i as f64).collect();
        let mut z = vec![0.0f64; 16];
        let cfg = DotGeneral::new(&[2], &[0], &[], &[]);
        let spec =
            |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).expect("layout"));
        let problem = Problem::from_dot_general(
            DType::F64,
            spec(&[2, 2, 2], &[1, 2, 4]),
            spec(&[2, 2, 2], &[1, 2, 4]),
            spec(&[2, 2, 2, 2], &[1, 2, 4, 8]),
            &cfg,
        )
        .expect("problem");
        let plan = Plan::<f64>::new(&problem, &PlanConfig::default()).expect("plan");
        let xv = StridedView::new(&x, &[2, 2, 2], &[1, 2, 4], 0).expect("x");
        let yv = StridedView::new(&y, &[2, 2, 2], &[1, 2, 4], 0).expect("y");
        let ns = median(|| {
            let mut zv = StridedViewMut::new(&mut z, &[2, 2, 2, 2], &[1, 2, 4, 8], 0).expect("z");
            plan.execute_into(exec, 1.0, &xv, &yv, &mut zv)
                .expect("exec");
        });
        println!("contract_2x2x2,rust,{t},{ns:.1},{SAMPLES}");
    });
}
