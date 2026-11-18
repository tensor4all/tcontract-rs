//! Time per call of a rank-2 Faer plan at each item width, and the width the
//! plan chooses on a team token (tcontract-rs#194).
//!
//! One `D[i,j] += sum_k A[i,k] B[k,j]` plan per shape, column-major, planned
//! with `FaerLimit::NONE` so every shape is a Faer plan, with
//! `CSpec::Output(Identity)` and timed through `Plan::execute_slices_accum`
//! with `beta = 1` (the in-place accumulation a block-sparse sweep pays; the
//! identity output pass is skipped, so only the GEMM is timed).
//!
//! * `forced` rows: the plan on `team.exec(w)` with `BatchStrategy::Inner`, so
//!   faer gets exactly `w` threads.
//! * `auto` rows: the plan on `team.exec(budget)` with the default policy; the
//!   row carries `Plan::execution_width` at that budget (`chosen`), and the
//!   chosen width is also timed as a `forced` row.
//!
//! Before any timing every width's result (from a zero D) is checked against
//! the width-1 result; a mismatch aborts the run.
//!
//! Output: `DISPATCH,w,ns` (the team token's dispatch cost) and `ROW,kind,dtype,m,n,k,w,budget,chosen,ns_median,ns_min,calls,samples`.

use std::time::{Duration, Instant};

use num_complex::{Complex32 as C32, Complex64 as C64};
use tcontract::api::{CSpec, DType, Labels, LayoutSpec, Op, OperandSpec, Problem, Scalar};
use tcontract::{FaerLimit, OutputContract, Plan, PlanConfig, SliceAccumulationSource};
use tcontract_exec::{BatchPolicy, BatchStrategy, Exec};

trait P: Scalar + Copy {
    const DT: DType;
    const NAME: &'static str;
    fn mk(re: f64, im: f64) -> Self;
    fn parts(self) -> (f64, f64);
}
impl P for f64 {
    const DT: DType = DType::F64;
    const NAME: &'static str = "f64";
    fn mk(re: f64, _: f64) -> Self {
        re
    }
    fn parts(self) -> (f64, f64) {
        (self, 0.0)
    }
}
impl P for f32 {
    const DT: DType = DType::F32;
    const NAME: &'static str = "f32";
    fn mk(re: f64, _: f64) -> Self {
        re as f32
    }
    fn parts(self) -> (f64, f64) {
        (self as f64, 0.0)
    }
}
impl P for C32 {
    const DT: DType = DType::C32;
    const NAME: &'static str = "c32";
    fn mk(re: f64, im: f64) -> Self {
        C32::new(re as f32, im as f32)
    }
    fn parts(self) -> (f64, f64) {
        (self.re as f64, self.im as f64)
    }
}
impl P for C64 {
    const DT: DType = DType::C64;
    const NAME: &'static str = "c64";
    fn mk(re: f64, im: f64) -> Self {
        C64::new(re, im)
    }
    fn parts(self) -> (f64, f64) {
        (self.re, self.im)
    }
}

struct Args {
    dtypes: Vec<String>,
    extents: Vec<usize>,
    widths: Vec<usize>,
    budgets: Vec<usize>,
    threads: usize,
    reverse: bool,
    target: Duration,
}

fn list(s: &str) -> Vec<usize> {
    s.split(',').map(|x| x.trim().parse().expect("number")).collect()
}

fn args() -> Args {
    let mut a = Args {
        dtypes: vec!["f64".into(), "c64".into()],
        extents: vec![16, 32, 64, 128, 256, 512],
        widths: vec![1, 2, 4, 8, 16],
        budgets: vec![16],
        threads: 16,
        reverse: false,
        target: Duration::from_millis(200),
    };
    let v: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < v.len() {
        let val = v.get(i + 1).cloned().unwrap_or_default();
        match v[i].as_str() {
            "--dtypes" => a.dtypes = val.split(',').map(String::from).collect(),
            "--extents" => a.extents = list(&val),
            "--widths" => a.widths = list(&val),
            "--budgets" => a.budgets = list(&val),
            "--threads" => a.threads = val.parse().unwrap(),
            "--target-ms" => a.target = Duration::from_millis(val.parse().unwrap()),
            "--reverse" => {
                a.reverse = true;
                i += 1;
                continue;
            }
            o => panic!("unknown argument {o}"),
        }
        i += 2;
    }
    a
}

fn plan<T: P>(m: usize, n: usize, k: usize) -> Plan<T> {
    let l = |d: &[usize]| OperandSpec::new(LayoutSpec::new(d, &[1, d[0] as isize], 0).unwrap());
    let problem = Problem::from_labels(
        T::DT,
        l(&[m, k]),
        l(&[k, n]),
        CSpec::Output(Op::Identity),
        l(&[m, n]),
        &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    )
    .unwrap();
    let config = PlanConfig {
        faer_limit: FaerLimit::NONE,
        ..PlanConfig::default()
    };
    let plan = Plan::<T>::new(&problem, &config).unwrap();
    assert_eq!(plan.report().algorithm, tcontract::Algorithm::Faer);
    plan
}

fn fill<T: P>(len: usize, seed: usize) -> Vec<T> {
    (0..len)
        .map(|i| {
            let j = i + seed;
            T::mk(
                ((j * 37) % 101) as f64 / 101.0 - 0.5,
                ((j * 53) % 97) as f64 / 97.0 - 0.5,
            )
        })
        .collect()
}

fn call<T: P>(plan: &Plan<T>, exec: &mut Exec<'_>, a: &[T], b: &[T], d: &mut [T]) {
    plan.execute_slices_accum(
        exec,
        T::mk(1.0, 0.0),
        (a, 0),
        (b, 0),
        T::mk(1.0, 0.0),
        SliceAccumulationSource::Output,
        (d, 0),
    )
    .unwrap();
}

/// Median and minimum ns per call, after at least 40 ms of warm-up calls: samples of `r` back-to-back calls, `r` so a
/// sample lasts at least 100 us, 7..=21 samples within about `target`.
fn time<T: P>(
    plan: &Plan<T>,
    exec: &mut Exec<'_>,
    a: &[T],
    b: &[T],
    d: &mut [T],
    target: Duration,
) -> (f64, f64, usize, usize) {
    // Settle first: the members a wider previous point left spinning (threadteam spins
    // about 300k PAUSEs, some 20 ms here) and the clock they pulled down.
    let warm = Instant::now();
    let mut n = 0;
    while n < 3 || warm.elapsed() < Duration::from_millis(40) {
        call(plan, exec, a, b, d);
        n += 1;
    }
    let t0 = Instant::now();
    call(plan, exec, a, b, d);
    let one = t0.elapsed().as_secs_f64().max(1e-8);
    let r = ((100e-6 / one).ceil() as usize).max(1);
    let samples = ((target.as_secs_f64() / (one * r as f64)) as usize).clamp(7, 21);
    let mut v: Vec<f64> = (0..samples)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..r {
                call(plan, exec, a, b, d);
            }
            t.elapsed().as_secs_f64() * 1e9 / r as f64
        })
        .collect();
    v.sort_by(|x, y| x.partial_cmp(y).unwrap());
    (v[samples / 2], v[0], r, samples)
}

fn shape<T: P>(team: &threadteam::Team, args: &Args, (m, n, k): (usize, usize, usize)) {
    let plan = plan::<T>(m, n, k);
    let (a, b) = (fill::<T>(m * k, 0), fill::<T>(k * n, 7));
    let at = |w: usize, strategy| {
        Exec::from_token(team.exec(w), None, BatchPolicy::new(strategy))
    };
    let mut chosen = Vec::new();
    for &budget in &args.budgets {
        let exec = at(budget, BatchStrategy::Auto);
        let w = plan
            .execution_width(&exec, budget, T::mk(1.0, 0.0), OutputContract::Initialized)
            .unwrap()
            .map_or(0, |w| w.get());
        chosen.push((budget, w));
    }
    let mut widths: Vec<usize> = args.widths.clone();
    widths.extend(chosen.iter().map(|&(_, w)| w).filter(|&w| w > 0));
    widths.sort_unstable();
    widths.dedup();
    if args.reverse {
        widths.reverse();
    }
    // Correctness: every width from a zero D against width 1.
    let reference = {
        let mut d = vec![T::mk(0.0, 0.0); m * n];
        call(&plan, &mut at(1, BatchStrategy::Inner), &a, &b, &mut d);
        d
    };
    let scale = reference
        .iter()
        .map(|x| x.parts().0.abs() + x.parts().1.abs())
        .fold(0.0, f64::max)
        .max(1.0);
    for &w in &widths {
        let mut d = vec![T::mk(0.0, 0.0); m * n];
        call(&plan, &mut at(w, BatchStrategy::Inner), &a, &b, &mut d);
        let err = d
            .iter()
            .zip(&reference)
            .map(|(x, y)| {
                let (x, y) = (x.parts(), y.parts());
                (x.0 - y.0).abs() + (x.1 - y.1).abs()
            })
            .fold(0.0, f64::max);
        let tol = if matches!(T::DT, DType::F32 | DType::C32) { 1e-4 } else { 1e-10 };
        assert!(err / scale < tol, "mismatch {m}x{n}x{k} w={w}: {err}");
    }
    let mut d = fill::<T>(m * n, 3);
    let row = |kind: &str, w: usize, budget: usize, chosen: usize, t: (f64, f64, usize, usize)| {
        println!(
            "ROW,{kind},{},{m},{n},{k},{w},{budget},{chosen},{:.1},{:.1},{},{}",
            T::NAME,
            t.0,
            t.1,
            t.2,
            t.3
        );
    };
    for &w in &widths {
        let t = time(&plan, &mut at(w, BatchStrategy::Inner), &a, &b, &mut d, args.target);
        row("forced", w, w, w, t);
    }
    for &(budget, w) in &chosen {
        let t = time(&plan, &mut at(budget, BatchStrategy::Auto), &a, &b, &mut d, args.target);
        row("auto", w, budget, w, t);
    }
}

fn main() {
    let args = args();
    let width = args
        .widths
        .iter()
        .chain(&args.budgets)
        .copied()
        .max()
        .unwrap()
        .max(args.threads);
    let pool = threadteam::Pool::builder()
        .threads(width)
        .acquire_timeout(Duration::from_secs(30))
        .build()
        .expect("pool");
    let team = pool.into_team(width).expect("team");
    assert_eq!(team.size().get(), width, "team width");
    eprintln!(
        "team {width} host_class {:?} muladds_per_ns {}",
        tcontract_exec::host::HostClass::detect(),
        tcontract_exec::host::muladds_per_ns()
    );
    for w in 1..=width {
        let exec = Exec::from_token(team.exec(w), None, BatchPolicy::default());
        let d = exec
            .token_ref()
            .dispatch_cost(std::num::NonZeroUsize::new(w).unwrap());
        println!("DISPATCH,{w},{}", d.as_nanos());
    }
    let mut shapes = Vec::new();
    for &m in &args.extents {
        for &n in &args.extents {
            for &k in &args.extents {
                shapes.push((m, n, k));
            }
        }
    }
    let mut dtypes = args.dtypes.clone();
    if args.reverse {
        shapes.reverse();
        dtypes.reverse();
    }
    for dt in &dtypes {
        for &s in &shapes {
            match dt.as_str() {
                "f64" => shape::<f64>(&team, &args, s),
                "c64" => shape::<C64>(&team, &args, s),
                "f32" => shape::<f32>(&team, &args, s),
                "c32" => shape::<C32>(&team, &args, s),
                o => panic!("dtype {o}"),
            }
        }
    }
}
