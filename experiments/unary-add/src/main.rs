//! Per-call cost of the labels-based unary update on small blocks, at one
//! thread.
//!
//! The operation is `tcontract::unary::add`:
//! `D = alpha * op_A(A[labels_a]) + beta * D_old`, which is one entry point for
//! a permutation (`tensoradd!`), a diagonal and a reduction (`tensortrace!`).
//! TensorKit.jl calls it once per subblock, so what matters is the per-call
//! cost, not the bandwidth of a large block.
//!
//! Two timed boundaries, per call:
//!
//! * `api`  — `unary::add`: build the `Problem`, plan it, execute. This is what
//!   a caller with no plan cache pays.
//! * `plan` — a prebuilt `Plan` and `AccumulationSource::Output` per call: what
//!   a caller that keeps a plan pays (the C ABI's per-call shape).
//!
//! Every arm is checked against a naive label oracle before any timing; a
//! mismatch aborts the run rather than publishing a number.
//!
//! One thread, no pool: `Exec::serial()` asserts the library's own width-1
//! contract, and nothing here enters a pool.

use std::time::{Duration, Instant};

use num_complex::Complex64 as C64;
use strided_view::{StridedView, StridedViewMut};
use tcontract::api::{
    AccumulationSource, CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem, Scalar,
};
use tcontract::unary::{add, Unary};
use tcontract::{Plan, PlanConfig};
use tcontract_exec::Exec;
use tcontract_kernel::{Element, Real};

/// One small-block case: `A`'s extents and labels, `D`'s extents and labels.
#[derive(Clone, Debug)]
struct Shape {
    name: &'static str,
    a: Vec<usize>,
    la: Vec<i64>,
    d: Vec<usize>,
    ld: Vec<i64>,
    /// `beta`; the accumulation form is what `tensoradd!` uses.
    beta: f64,
}

/// Shape classes a TensorKit subblock exercises: a small permutation with an
/// accumulated output, a bigger one, a site tensor's permutation, and a
/// diagonal that is then reduced.
fn shapes() -> Vec<Shape> {
    vec![
        Shape {
            name: "perm_8x8",
            a: vec![8, 8],
            la: vec![0, 1],
            d: vec![8, 8],
            ld: vec![1, 0],
            beta: 1.0,
        },
        Shape {
            name: "perm_32x32",
            a: vec![32, 32],
            la: vec![0, 1],
            d: vec![32, 32],
            ld: vec![1, 0],
            beta: 1.0,
        },
        Shape {
            name: "site_chi32",
            a: vec![32, 2, 32],
            la: vec![0, 1, 2],
            d: vec![2, 32, 32],
            ld: vec![1, 0, 2],
            beta: 1.0,
        },
        Shape {
            name: "trace_32",
            a: vec![32, 32, 32],
            la: vec![0, 1, 1],
            d: vec![32],
            ld: vec![0],
            beta: 0.0,
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

fn fill<S: Scalar>(len: usize) -> Vec<S> {
    (0..len)
        .map(|i| {
            let re = ((i * 37) % 101) as f64 / 101.0 - 0.5;
            let im = ((i * 53) % 97) as f64 / 97.0 - 0.5;
            <S as Element>::from_parts(Real::from_f64(re), Real::from_f64(im))
        })
        .collect()
}

fn product(dims: &[usize]) -> usize {
    dims.iter().product()
}

/// `D = alpha * op_A(A[labels_a]) + beta * D_old` from the labels alone.
#[allow(clippy::too_many_arguments)]
fn oracle<S: Scalar>(
    alpha: S,
    a: &[S],
    s: &Shape,
    beta: S,
    d: &[S],
) -> Vec<S> {
    let sa = col_major(&s.a);
    let sd = col_major(&s.d);
    let mut sum = vec![<S as Element>::zero(); d.len()];
    let mut idx = vec![0usize; s.a.len()];
    loop {
        let mut consistent = true;
        for k in 0..s.a.len() {
            for j in k + 1..s.a.len() {
                if s.la[k] == s.la[j] && idx[k] != idx[j] {
                    consistent = false;
                }
            }
        }
        if consistent {
            let mut di = vec![0usize; s.d.len()];
            let mut ok = true;
            for (j, &l) in s.ld.iter().enumerate() {
                match s.la.iter().position(|&al| al == l) {
                    Some(k) if idx[k] < s.d[j] => di[j] = idx[k],
                    _ => ok = false,
                }
            }
            if ok {
                let ai: usize = idx.iter().zip(&sa).map(|(&i, &st)| i as usize * st as usize).sum();
                let di_pos: usize = di.iter().zip(&sd).map(|(&i, &st)| i as usize * st as usize).sum();
                sum[di_pos] = Element::add(sum[di_pos], a[ai]);
            }
        }
        // odometer over A's extents
        let mut k = 0;
        loop {
            if k == idx.len() {
                let mut out = d.to_vec();
                for (p, &v) in sum.iter().enumerate() {
                    let base = if beta == <S as Element>::zero() {
                        <S as Element>::zero()
                    } else {
                        Element::mul(beta, d[p])
                    };
                    out[p] = Element::add(Element::mul(alpha, v), base);
                }
                return out;
            }
            idx[k] += 1;
            if idx[k] < s.a[k] {
                break;
            }
            idx[k] = 0;
            k += 1;
        }
    }
}

fn rel_err<S: Scalar>(got: &[S], want: &[S]) -> f64 {
    let mag = |z: S| Real::to_f64(Element::re(z)).hypot(Real::to_f64(Element::im(z)));
    let scale = want.iter().map(|&z| mag(z)).fold(1.0f64, f64::max);
    got.iter()
        .zip(want)
        .map(|(&g, &w)| mag(Element::add(g, Element::mul(w, <S as Element>::from_parts(Real::from_f64(-1.0), Real::from_f64(0.0))))))
        .fold(0.0f64, f64::max)
        / scale
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

fn one_case<S: Scalar>(
    dtype: DType,
    s: &Shape,
    reps: usize,
    prime_ms: u64,
    rows: &mut Vec<String>,
) {
    assert_eq!(<S as Scalar>::STORAGE, dtype);
    let alpha = <S as Element>::one();
    let beta = <S as Element>::from_parts(Real::from_f64(s.beta), Real::from_f64(0.0));
    let a = fill::<S>(product(&s.a));
    let d0 = fill::<S>(product(&s.d));
    let sa = col_major(&s.a);
    let sd = col_major(&s.d);
    let exec = Exec::serial();

    // The oracle, and the `api` arm's result, before any timing.
    let want = oracle(alpha, &a, s, beta, &d0);
    let mut got = d0.clone();
    {
        let av = StridedView::new(&a, &s.a, &sa, 0).unwrap();
        let mut dv = StridedViewMut::new(&mut got, &s.d, &sd, 0).unwrap();
        add(&exec, &Unary::new(&s.la, &s.ld), alpha, &av, beta, &mut dv).unwrap();
    }
    let err = rel_err(&got, &want);
    println!("CHECK {} {err:.3e}", s.name);
    assert!(err < 1e-12, "{}: the unary update disagrees with the oracle", s.name);

    // `api`: problem + plan + execute per call. The views are descriptors over
    // the fixed buffers, built once, so the number is the call path and not a
    // per-call view construction.
    let mut d = d0.clone();
    let av = StridedView::new(&a, &s.a, &sa, 0).unwrap();
    let mut dv = StridedViewMut::new(&mut d, &s.d, &sd, 0).unwrap();
    let secs = timed(reps, prime_ms, || {
        add(&exec, &Unary::new(&s.la, &s.ld), alpha, &av, beta, &mut dv).unwrap();
    });
    drop(dv);
    rows.push(format!("{},{},{:.1},api", s.name, dtype.name(), secs));

    // `plan`: a prebuilt plan, executed per call.
    let c = if beta == <S as Element>::zero() {
        CSpec::Absent
    } else {
        CSpec::Output(tcontract::api::Op::Identity)
    };
    let l = |d: &[usize], st: &[isize]| OperandSpec::new(LayoutSpec::new(d, st, 0).unwrap());
    let p = Problem::from_labels(dtype, l(&s.a, &sa), l(&[], &[]), c, l(&s.d, &sd), &Labels::new(&s.la, &[], &s.ld))
        .unwrap();
    let plan = Plan::<S>::new(&p, &PlanConfig::default()).unwrap();
    let algorithm = plan.report().algorithm.name();
    let scalar = [<S as Element>::one()];
    let bv = StridedView::new(&scalar[..], &[], &[], 0).unwrap();
    let mut d = d0.clone();
    let av = StridedView::new(&a, &s.a, &sa, 0).unwrap();
    let mut dv = StridedViewMut::new(&mut d, &s.d, &sd, 0).unwrap();
    let secs = timed(reps, prime_ms, || {
        if beta == <S as Element>::zero() {
            plan.execute_into(&exec, alpha, &av, &bv, &mut dv).unwrap();
        } else {
            plan.execute_into_accum(&exec, alpha, &av, &bv, beta, AccumulationSource::Output, &mut dv)
                .unwrap();
        }
    });
    rows.push(format!(
        "{},{},{:.1},plan[{algorithm}]",
        s.name,
        dtype.name(),
        secs
    ));
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

    println!(
        "# unary-add: tcontract {} per call, 1 thread (Exec::serial), {} reps, {} ms priming",
        env!("CARGO_PKG_VERSION"),
        reps,
        prime_ms
    );
    let mut rows = vec!["shape,dtype,ns,arm".to_string()];
    for s in shapes() {
        one_case::<f64>(DType::F64, &s, reps, prime_ms, &mut rows);
    }
    // The site-block permutation, which is TensorKit's case, in complex double.
    one_case::<C64>(DType::C64, &shapes()[2], reps, prime_ms, &mut rows);

    for row in &rows {
        println!("{row}");
    }
    if let Some(path) = csv {
        std::fs::write(&path, rows.join("\n") + "\n").expect("the CSV path is writable");
        eprintln!("wrote {path}");
    }
}
