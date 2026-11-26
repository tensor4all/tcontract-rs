//! Wall time of a packed contraction whose scatters are large and layout-driven
//! (tcontract-rs#221): a two-qubit gate on an n-qubit complex32 state, with the
//! state's operand in column-major and in row-major layout.
//!
//! `THREADS=16 cargo run --release -p tcontract --example gather_speed -- [qubits [reps]]`
//!
//! Prints plan build time, retained bytes and the median/min execute time over
//! `reps` runs (serial unless `THREADS` > 1). Run it on two revisions and compare; nothing is checked.
use num_complex::Complex32;
use std::time::Instant;
use strided_view::{StridedView, StridedViewMut};
use tcontract::api::{CSpec, Labels, LayoutSpec, OperandSpec, Problem, Scalar};
use tcontract::{Plan, PlanConfig};
use tcontract_exec::{Exec, Pool};

fn col_major(dims: &[usize]) -> Vec<isize> {
    let mut acc = 1isize;
    dims.iter()
        .map(|&d| {
            let s = acc;
            acc *= d as isize;
            s
        })
        .collect()
}

fn row_major(dims: &[usize]) -> Vec<isize> {
    let mut s = col_major(&dims.iter().rev().copied().collect::<Vec<_>>());
    s.reverse();
    s
}

fn main() {
    let n: usize = std::env::args().nth(1).map_or(24, |s| s.parse().unwrap());
    let reps: usize = std::env::args().nth(2).map_or(5, |s| s.parse().unwrap());
    let threads: usize = std::env::var("THREADS")
        .ok()
        .map_or(1, |s| s.parse().unwrap());
    let sp = threadteam::Pool::builder()
        .threads(threads)
        .build()
        .unwrap();
    let pool = Pool::borrow(&sp);
    let sd = vec![2usize; n];
    let gd = [2usize; 4];
    for (name, row_a) in [("state col-major", false), ("state row-major", true)] {
        for &(p, q) in &[(3i64, 17i64), (0, 13)] {
            let labels: Vec<i64> = (0..n as i64).collect();
            let out: Vec<i64> = labels
                .iter()
                .map(|&l| {
                    if l == p {
                        1000
                    } else if l == q {
                        1001
                    } else {
                        l
                    }
                })
                .collect();
            let st_a = if row_a {
                row_major(&sd)
            } else {
                col_major(&sd)
            };
            let (st_b, st_d) = (col_major(&gd), col_major(&sd));
            let spec =
                |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
            let problem = Problem::from_labels(
                <Complex32 as Scalar>::STORAGE,
                spec(&sd, &st_a),
                spec(&gd, &st_b),
                CSpec::Absent,
                spec(&sd, &st_d),
                &Labels::new(&labels, &[1000, 1001, p, q], &out),
            )
            .unwrap();
            let t = Instant::now();
            let plan = Plan::<Complex32>::new(&problem, &PlanConfig::packed()).unwrap();
            let build = t.elapsed();
            let len = 1usize << n;
            let a: Vec<Complex32> = (0..len)
                .map(|i| Complex32::new((i % 7) as f32, 1.0))
                .collect();
            let b: Vec<Complex32> = (0..16).map(|i| Complex32::new(i as f32, 0.5)).collect();
            let mut d = vec![Complex32::new(0.0, 0.0); len];
            let mut times = Vec::new();
            for _ in 0..reps {
                let va = StridedView::new(&a, &sd, &st_a, 0).unwrap();
                let vb = StridedView::new(&b, &gd, &st_b, 0).unwrap();
                let mut vd = StridedViewMut::new(&mut d, &sd, &st_d, 0).unwrap();
                let t = Instant::now();
                let mut ex = if threads > 1 {
                    Exec::new(&pool, threads).unwrap()
                } else {
                    Exec::serial()
                };
                plan.execute_into(&mut ex, Complex32::new(1.0, 0.0), &va, &vb, &mut vd)
                    .unwrap();
                times.push(t.elapsed().as_secs_f64());
            }
            times.sort_by(f64::total_cmp);
            let med = times[times.len() / 2];
            let checksum: f64 = d.iter().step_by(4099).map(|z| z.re as f64).sum();
            println!(
                "{name} gate({p},{q}) n={n}: build {:.3} ms, retained {} B, exec median {:.1} ms (min {:.1}), {:.2} ns/elem, checksum {checksum}",
                build.as_secs_f64() * 1e3,
                plan.retained_bytes(),
                med * 1e3,
                times[0] * 1e3,
                med * 1e9 / len as f64
            );
        }
    }
}
