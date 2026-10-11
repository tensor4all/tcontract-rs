//! What the neutral contraction interface costs per call.
//!
//! The deferred acceptance item of #31: a prepared plan run directly
//! (`Plan::execute_into`) against the same plan through the interface
//! (`Box<dyn ContractionBackend<T>>::prepare` -> `BoxedPlan<T>::execute_into`),
//! with **preparation and repeated execution measured separately**, at one and
//! four threads, on a fixed two-case corpus.
//!
//! Both arms run the same kernels by construction: `TcontractBackend::prepare` is
//! `Plan::<T>::new` with the trait's `no_materialize` OR-ed in, and the trait's
//! `execute_into` forwards to the same execution. So any difference here is
//! dispatch and boxing, not arithmetic. Correctness is checked against a naive
//! reference written in this file before anything is timed, and the two arms
//! are checked against each other.
//!
//! Allocations are not counted here: `tests/plan_alloc.rs` pins the concrete
//! path's allocation counts (7 for the problem, 2 for the plan on the chi = 4
//! MPS steps) and the interface adds one `Box` at preparation, which the same
//! test file's sibling `tests/neutral_alloc.rs` covers.

use std::time::{Duration, Instant};

use num_complex::Complex64 as C64;
use strided_view::{StridedView, StridedViewMut};
use tcontract::api::{
    ContractionBackend, DotGeneral, LayoutSpec, OperandSpec, PlanningBudget, Problem, Requirements,
    Scalar,
};
use tcontract::{Plan, PlanConfig, TcontractBackend};
use tcontract_exec::{Exec, Pool};
use tcontract_kernel::{Element, Real};

/// One case: the two operand extents, the output extents, the contracted axes
/// and a naive reference written directly from the axes.
struct Case {
    name: &'static str,
    a: Vec<usize>,
    b: Vec<usize>,
    d: Vec<usize>,
    contract: (Vec<usize>, Vec<usize>),
    reference: fn(&[f64], &[f64], &mut [f64]),
}

fn cases() -> Vec<Case> {
    vec![
        // D[i, k] = sum_j A[i, j] B[j, k]
        Case {
            name: "gemm_n64",
            a: vec![64, 64],
            b: vec![64, 64],
            d: vec![64, 64],
            contract: (vec![1], vec![0]),
            reference: |a, b, d| {
                for i in 0..64 {
                    for k in 0..64 {
                        let mut acc = 0.0;
                        for j in 0..64 {
                            acc += a[i + 64 * j] * b[j + 64 * k];
                        }
                        d[i + 64 * k] = acc;
                    }
                }
            },
        },
        // The #61 corpus's MPS environment step at chi = 32, `ab,asc->bsc`:
        // D[b, s, c] = sum_a A[a, b] B[a, s, c], with B's axes (a, s, c).
        Case {
            name: "mps_env_chi32",
            a: vec![32, 32],
            b: vec![32, 2, 32],
            d: vec![32, 2, 32],
            contract: (vec![0], vec![0]),
            reference: |a, b, d| {
                for bb in 0..32 {
                    for s in 0..2 {
                        for c in 0..32 {
                            let mut acc = 0.0;
                            for aa in 0..32 {
                                acc += a[aa + 32 * bb] * b[aa + 32 * s + 64 * c];
                            }
                            d[bb + 32 * s + 64 * c] = acc;
                        }
                    }
                }
            },
        },
    ]
}

fn col_major(dims: &[usize]) -> Vec<isize> {
    let mut s = Vec::with_capacity(dims.len());
    let mut acc = 1isize;
    for &d in dims {
        s.push(acc);
        acc *= d.max(1) as isize;
    }
    s
}

fn fill_f64(len: usize) -> Vec<f64> {
    (0..len).map(|i| ((i * 37) % 101) as f64 / 101.0 - 0.5).collect()
}

fn fill_c64(len: usize) -> Vec<C64> {
    (0..len)
        .map(|i| C64::new(((i * 37) % 101) as f64 / 101.0 - 0.5, ((i * 53) % 97) as f64 / 97.0 - 0.5))
        .collect()
}

/// Best of `reps` calls in nanoseconds, after `prime_ms` of untimed priming.
fn timed(reps: usize, prime_ms: u64, mut f: impl FnMut()) -> f64 {
    let until = Instant::now() + Duration::from_millis(prime_ms);
    while Instant::now() < until {
        f();
    }
    let mut best = f64::INFINITY;
    for _ in 0..reps.max(1) {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64() * 1e9);
    }
    best
}

fn rel_err<S: Scalar>(got: &[S], want: &[S]) -> f64 {
    let mag = |z: S| Real::to_f64(Element::re(z)).hypot(Real::to_f64(Element::im(z)));
    let scale = want.iter().map(|&z| mag(z)).fold(1.0f64, f64::max);
    got.iter()
        .zip(want)
        .map(|(&g, &w)| {
            mag(Element::add(
                g,
                Element::mul(
                    w,
                    <S as Element>::from_parts(Real::from_f64(-1.0), Real::from_f64(0.0)),
                ),
            ))
        })
        .fold(0.0f64, f64::max)
        / scale
}

fn operand<S: Scalar>(dims: &[usize], strides: &[isize]) -> OperandSpec {
    OperandSpec::new(LayoutSpec::new(dims, strides, 0).unwrap())
}

fn one_case<S: Scalar>(
    dtype: &str,
    c: &Case,
    threads: usize,
    pool: &Pool<'_>,
    reps: usize,
    prime_ms: u64,
    rows: &mut Vec<String>,
) {
    eprintln!("case {} {} {threads}T", c.name, dtype);
    let sa = col_major(&c.a);
    let sb = col_major(&c.b);
    let sd = col_major(&c.d);
    let a: Vec<S> = if dtype == "f64" { fill_f64(c.a.iter().product()).into_iter().map(real::<S>).collect() } else { fill_c64(c.a.iter().product()).into_iter().map(complex::<S>).collect() };
    let b: Vec<S> = if dtype == "f64" { fill_f64(c.b.iter().product()).into_iter().map(real::<S>).collect() } else { fill_c64(c.b.iter().product()).into_iter().map(complex::<S>).collect() };
    let mut d = vec![<S as Element>::zero(); c.d.iter().product()];

    let cfg = DotGeneral::new(&c.contract.0, &c.contract.1, &[], &[]);
    let problem = Problem::from_dot_general(
        <S as Scalar>::STORAGE,
        operand::<S>(&c.a, &sa),
        operand::<S>(&c.b, &sb),
        operand::<S>(&c.d, &sd),
        &cfg,
    )
    .unwrap();

    let exec = if threads == 1 { Exec::serial() } else { Exec::rayon(pool) };

    // --- correctness, before anything is timed ---------------------------------
    if dtype == "f64" {
        let mut want = vec![0.0f64; c.d.iter().product()];
        let af: Vec<f64> = a.iter().map(|&x| Real::to_f64(Element::re(x))).collect();
        let bf: Vec<f64> = b.iter().map(|&x| Real::to_f64(Element::re(x))).collect();
        (c.reference)(&af, &bf, &mut want);
        let want: Vec<S> = want.into_iter().map(real::<S>).collect();
        let mut got = vec![<S as Element>::zero(); c.d.iter().product()];
        {
            let av = StridedView::new(&a, &c.a, &sa, 0).expect("av");
            let bv = StridedView::new(&b, &c.b, &sb, 0).expect("bv");
            let mut dv = StridedViewMut::new(&mut got, &c.d, &sd, 0).expect("dv");
            let concrete = Plan::<S>::new(&problem, &PlanConfig::default()).expect("plan");
            concrete.execute_into(&exec, <S as Element>::one(), &av, &bv, &mut dv).unwrap();
        }
        let err = rel_err(&got, &want);
        println!("CHECK {} {} {threads}T concrete {err:.3e}", c.name, dtype);
        assert!(err < 1e-12, "{} {}: the concrete path disagrees with the reference", c.name, dtype);
        let mut got2 = vec![<S as Element>::zero(); c.d.iter().product()];
        {
            let av = StridedView::new(&a, &c.a, &sa, 0).expect("av");
            let bv = StridedView::new(&b, &c.b, &sb, 0).expect("bv");
            let mut dv = StridedViewMut::new(&mut got2, &c.d, &sd, 0).unwrap();
            let backend: Box<dyn ContractionBackend<S>> = Box::new(TcontractBackend::default());
            let plan = backend
                .prepare(&problem, &Requirements::new(), &PlanningBudget::serial())
                .unwrap();
            plan.execute_into(&exec, <S as Element>::one(), &av, &bv, &mut dv).unwrap();
        }
        let err = rel_err(&got2, &want);
        println!("CHECK {} {} {threads}T trait    {err:.3e}", c.name, dtype);
        assert!(err < 1e-12, "{} {}: the trait path disagrees with the reference", c.name, dtype);
    }

    // --- preparation, separately ----------------------------------------------
    let p_concrete = timed(reps, prime_ms, || {
        let p = Plan::<S>::new(&problem, &PlanConfig::default()).unwrap();
        std::hint::black_box(&p);
    });
    let backend: Box<dyn ContractionBackend<S>> = Box::new(TcontractBackend::default());
    let p_trait = timed(reps, prime_ms, || {
        let p = backend
            .prepare(&problem, &Requirements::new(), &PlanningBudget::serial())
            .unwrap();
        std::hint::black_box(&p);
    });
    rows.push(format!("{},{},{threads},prepare_concrete,{p_concrete:.1}", c.name, dtype));
    rows.push(format!("{},{},{threads},prepare_trait,{p_trait:.1}", c.name, dtype));

    // --- repeated execution, separately ---------------------------------------
    let concrete = Plan::<S>::new(&problem, &PlanConfig::default()).expect("plan");
    let trait_plan = backend
        .prepare(&problem, &Requirements::new(), &PlanningBudget::serial())
        .unwrap();
    let av = StridedView::new(&a, &c.a, &sa, 0).expect("av");
    let bv = StridedView::new(&b, &c.b, &sb, 0).expect("bv");
    let mut dv = StridedViewMut::new(&mut d, &c.d, &sd, 0).unwrap();
    let exec_c = timed(reps, prime_ms, || {
        concrete
            .execute_into(&exec, <S as Element>::one(), &av, &bv, &mut dv)
            .unwrap();
    });
    let exec_t = timed(reps, prime_ms, || {
        trait_plan
            .execute_into(&exec, <S as Element>::one(), &av, &bv, &mut dv)
            .unwrap();
    });
    rows.push(format!("{},{},{threads},exec_concrete,{exec_c:.1}", c.name, dtype));
    rows.push(format!("{},{},{threads},exec_trait,{exec_t:.1}", c.name, dtype));
}

fn real<S: Scalar>(x: f64) -> S {
    <S as Element>::from_parts(Real::from_f64(x), Real::from_f64(0.0))
}

fn complex<S: Scalar>(x: C64) -> S {
    <S as Element>::from_parts(Real::from_f64(x.re), Real::from_f64(x.im))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut reps = 5usize;
    let mut prime_ms = 500u64;
    let mut csv: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        let next = || args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--reps" => {
                reps = next().parse().unwrap_or(5);
                i += 1;
            }
            "--prime-ms" => {
                prime_ms = next().parse().unwrap_or(500);
                i += 1;
            }
            "--csv" => {
                csv = Some(next());
                i += 1;
            }
            other => eprintln!("warning: ignoring {other}"),
        }
        i += 1;
    }

    let tp = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    let pool = Pool::borrow(&tp);
    let mut rows = vec!["case,dtype,threads,arm,ns".to_string()];
    for c in cases() {
        for threads in [1usize, 4] {
            one_case::<f64>("f64", &c, threads, &pool, reps, prime_ms, &mut rows);
            one_case::<C64>("c64", &c, threads, &pool, reps, prime_ms, &mut rows);
        }
    }
    for row in &rows {
        println!("{row}");
    }
    if let Some(path) = csv {
        std::fs::write(&path, rows.join("\n") + "\n").expect("the CSV path is writable");
        eprintln!("wrote {path}");
    }
}
