//! tcontract at an enforced thread count: a predeclared corpus of
//! binary contractions, or a recorded one (`--corpus FILE`, see
//! `tcontract_bench::corpus`), under the planner's own choice (`plan`) and the
//! packed driver forced (`packed`). Planning and execution are timed
//! separately.
//!
//! Usage: `contract --threads N [--corpus FILE] [--partition static|dynamic:JM,JN]
//! [--list]`. CSV `case,variant,threads,median_ns,samples` where `variant` is
//! `<engine>_<plan|exec>`; `# selected` lines report what each plan runs (the
//! algorithm and, for the packed driver, its family); `CHECK` lines compare the
//! engines' results. `--list` prints the case names and exits. Environment:
//! `BENCH_RUNS` (default 20), `BENCH_WARMUP` (3), `BENCH_CASE` (exact case
//! name), `BENCH_C_MODE` (`overwrite`, the default `dot_general` form, or the
//! TAPP-style separate-C forms `separate_b0` (beta 0, C unread), `separate_b1`
//! (beta 0.5, C a distinct buffer of D's layout) and `separate_same` (beta 0.5,
//! C is D itself); they run through `execute_raw`). The built-in corpus runs in f64 and c64; a corpus file sets each
//! entry's dtype.
use std::hint::black_box;

use num_complex::{Complex32, Complex64};
use strided_view::{StridedView, StridedViewMut};
use tcontract::api::{
    CSpec, DType, DotGeneral, Labels, LayoutSpec, Op, OperandSpec, Problem, Scalar,
};
use tcontract::{Partition, Plan, PlanConfig};
use tcontract_bench::corpus::{Corpus, DotGeneralEntry, Dtype, Entry, Operand};
use tcontract_bench::threads::BenchThreads;
use tcontract_bench::timing::{env_usize, median_ns};
use tcontract_exec::Exec;
use tcontract_kernel::{Element, Real};

/// One corpus entry. `a_order` / `b_order` give the storage order of the
/// logical axes (fastest first), so operands can be stored "transposed".
pub struct Case {
    pub name: &'static str,
    pub a: &'static [usize],
    pub b: &'static [usize],
    pub a_order: &'static [usize],
    pub b_order: &'static [usize],
    pub lc: &'static [usize],
    pub rc: &'static [usize],
    pub lb: &'static [usize],
    pub rb: &'static [usize],
}

/// The predeclared corpus (also listed in README.md).
pub const CORPUS: &[Case] = &[
    Case {
        name: "tiny_matmul",
        a: &[2, 2],
        b: &[2, 2],
        a_order: &[0, 1],
        b_order: &[0, 1],
        lc: &[1],
        rc: &[0],
        lb: &[],
        rb: &[],
    },
    Case {
        name: "matmul_256",
        a: &[256, 256],
        b: &[256, 256],
        a_order: &[0, 1],
        b_order: &[0, 1],
        lc: &[1],
        rc: &[0],
        lb: &[],
        rb: &[],
    },
    Case {
        name: "batched_64_b32",
        a: &[64, 64, 32],
        b: &[64, 64, 32],
        a_order: &[0, 1, 2],
        b_order: &[0, 1, 2],
        lc: &[1],
        rc: &[0],
        lb: &[2],
        rb: &[2],
    },
    Case {
        name: "permuted_fusable",
        a: &[64, 32, 32],
        b: &[32, 32, 64],
        a_order: &[2, 1, 0],
        b_order: &[2, 0, 1],
        lc: &[1, 2],
        rc: &[1, 0],
        lb: &[],
        rb: &[],
    },
    // A's contracted axes 1 and 2 are separated in storage by free axis 0: A must be copied.
    Case {
        name: "permuted_nonfusable",
        a: &[64, 32, 32],
        b: &[32, 32, 64],
        a_order: &[1, 0, 2],
        b_order: &[1, 0, 2],
        lc: &[1, 2],
        rc: &[1, 0],
        lb: &[],
        rb: &[],
    },
    Case {
        name: "network_ijkl_klmn",
        a: &[16, 16, 16, 16],
        b: &[16, 16, 16, 16],
        a_order: &[0, 1, 2, 3],
        b_order: &[0, 1, 2, 3],
        lc: &[2, 3],
        rc: &[0, 1],
        lb: &[],
        rb: &[],
    },
    Case {
        name: "large_ijk_jkl",
        a: &[256, 64, 64],
        b: &[64, 64, 256],
        a_order: &[0, 1, 2],
        b_order: &[0, 1, 2],
        lc: &[1, 2],
        rc: &[0, 1],
        lb: &[],
        rb: &[],
    },
];

fn fill<T: Scalar>(len: usize, seed: u64) -> Vec<T> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let re = (s % 1000) as f64 / 1000.0 - 0.5;
            let im = ((s >> 20) % 1000) as f64 / 1000.0 - 0.5;
            <T as Element>::from_parts(Real::from_f64(re), Real::from_f64(im))
        })
        .collect()
}

/// Strides for storing logical `dims` in `order` (fastest first).
fn strides(dims: &[usize], order: &[usize]) -> Vec<isize> {
    let mut s = vec![0isize; dims.len()];
    let mut acc = 1isize;
    for &a in order {
        s[a] = acc;
        acc *= dims[a] as isize;
    }
    s
}

/// A built-in case as a corpus entry: operands stored in `a_order` /
/// `b_order`, output compact column-major.
fn entry(case: &Case, dtype: Dtype) -> DotGeneralEntry {
    let cfg = DotGeneral::new(case.lc, case.rc, case.lb, case.rb);
    let cdims = cfg.validate(case.a, case.b).expect("config").out_dims;
    let order: Vec<usize> = (0..cdims.len()).collect();
    DotGeneralEntry {
        name: format!("{}_{}", case.name, dtype.tag()),
        dtype,
        a: Operand {
            dims: case.a.to_vec(),
            strides: strides(case.a, case.a_order),
        },
        b: Operand {
            dims: case.b.to_vec(),
            strides: strides(case.b, case.b_order),
        },
        c: Operand {
            strides: strides(&cdims, &order),
            dims: cdims,
        },
        lc: case.lc.to_vec(),
        rc: case.rc.to_vec(),
        lb: case.lb.to_vec(),
        rb: case.rb.to_vec(),
        conj: [false, false],
        calls: None,
        time_share: None,
    }
}

fn operand(o: &Operand, op: Op) -> OperandSpec {
    let (offset, _) = o.span();
    OperandSpec::new(LayoutSpec::new(&o.dims, &o.strides, offset as isize).expect("layout"))
        .with_op(op)
}

fn op(c: bool) -> Op {
    if c {
        Op::Conjugate
    } else {
        Op::Identity
    }
}

fn dtype_of(d: Dtype) -> DType {
    match d {
        Dtype::F32 => DType::F32,
        Dtype::F64 => DType::F64,
        Dtype::C32 => DType::C32,
        Dtype::C64 => DType::C64,
    }
}

/// How C is supplied (`BENCH_C_MODE`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum CMode {
    /// The `dot_general` overwrite form (`execute_into`).
    Overwrite,
    /// TAPP-style separate C, beta zero.
    SeparateB0,
    /// Separate C in its own buffer, beta 0.5.
    SeparateB1,
    /// Separate C that is D itself, beta 0.5.
    SeparateSame,
}

fn c_mode() -> CMode {
    match std::env::var("BENCH_C_MODE").as_deref() {
        Ok("separate_b0") => CMode::SeparateB0,
        Ok("separate_b1") => CMode::SeparateB1,
        Ok("separate_same") => CMode::SeparateSame,
        Ok("overwrite") | Ok("") | Err(_) => CMode::Overwrite,
        Ok(x) => panic!("unknown BENCH_C_MODE {x}"),
    }
}

/// The labels of a `dot_general` entry (A, B, D), as `Problem::from_dot_general`
/// numbers them: batch `0..nb`, contracted `nb..nb+nc`, free axes after.
fn dot_labels(e: &DotGeneralEntry) -> Labels {
    let dot = DotGeneral::new(&e.lc, &e.rc, &e.lb, &e.rb);
    let shape = dot.validate(&e.a.dims, &e.b.dims).expect("config");
    let (nb, nc) = (e.lb.len() as i64, e.lc.len() as i64);
    let (mut la, mut lb) = (vec![0i64; e.a.dims.len()], vec![0i64; e.b.dims.len()]);
    let mut ld = Vec::new();
    let mut next = nb + nc;
    for &x in &shape.lhs_free {
        la[x] = next;
        ld.push(next);
        next += 1;
    }
    for &x in &shape.rhs_free {
        lb[x] = next;
        ld.push(next);
        next += 1;
    }
    for (p, (&l, &r)) in e.lb.iter().zip(&e.rb).enumerate() {
        la[l] = p as i64;
        lb[r] = p as i64;
        ld.push(p as i64);
    }
    for (p, (&l, &r)) in e.lc.iter().zip(&e.rc).enumerate() {
        la[l] = nb + p as i64;
        lb[r] = nb + p as i64;
    }
    Labels::new(&la, &lb, &ld)
}

fn run<T: Scalar>(
    e: &DotGeneralEntry,
    exec: &Exec<'_>,
    threads: usize,
    warmup: usize,
    runs: usize,
) {
    let name = &e.name;
    let ((oa, la), (ob, lb), (oc, lc)) = (e.a.span(), e.b.span(), e.c.span());
    let ad: Vec<T> = fill(la, 1);
    let bd: Vec<T> = fill(lb, 2);
    let mode = c_mode();
    let problem = if mode == CMode::Overwrite {
        Problem::from_dot_general(
            dtype_of(e.dtype),
            operand(&e.a, op(e.conj[0])),
            operand(&e.b, op(e.conj[1])),
            operand(&e.c, Op::Identity),
            &DotGeneral::new(&e.lc, &e.rc, &e.lb, &e.rb),
        )
    } else {
        let labels = dot_labels(e);
        let ld = labels.d().to_vec();
        Problem::from_labels(
            dtype_of(e.dtype),
            operand(&e.a, op(e.conj[0])),
            operand(&e.b, op(e.conj[1])),
            CSpec::Separate(operand(&e.c, Op::Identity)),
            operand(&e.c, Op::Identity),
            &labels.with_c(&ld),
        )
    }
    .expect("problem");
    // The C buffer is only filled for the separate-C modes: the overwrite form
    // reads none, and filling an output-sized vector costs ~24 instructions per
    // element of whole-process time.
    let cd: Vec<T> = if mode == CMode::Overwrite {
        Vec::new()
    } else {
        fill(lc, 3)
    };
    let av = StridedView::new(&ad, &e.a.dims, &e.a.strides, oa as isize).expect("a");
    let bv = StridedView::new(&bd, &e.b.dims, &e.b.strides, ob as isize).expect("b");
    let mut outs = Vec::new();
    // `--partition dynamic:JM,JN` adds a separately labelled packed row that uses
    // the opt-in dynamic scheduler; the plan and static packed rows are
    // unchanged. Any explicit partition forces the packed driver.
    let dynamic = tcontract_bench::partition::from_args();
    let static_grid = Some(Partition::StaticGrid {
        pin: None,
        align_c_lines: false,
    });
    let mut variants = vec![
        ("plan".to_string(), None),
        ("packed".to_string(), static_grid),
    ];
    if dynamic.is_some() {
        variants.push((
            format!("packed{}", tcontract_bench::partition::suffix(dynamic)),
            dynamic,
        ));
    }
    for (tag, partition) in variants.iter().map(|(t, p)| (t.as_str(), *p)) {
        let config = tcontract_bench::partition::apply(PlanConfig::default(), partition);
        let mk = || Plan::<T>::new(&problem, &config).expect("plan");
        let ns = median_ns(warmup, runs, || {
            black_box(mk());
        });
        println!("{name},{tag}_plan,{threads},{ns:.0},{runs}");
        let plan = mk();
        let report = plan.report();
        if let Some(p) = &report.packed {
            println!("# selected {name} {tag}: packed {}", p.family_id);
        } else {
            println!("# selected {name} {tag}: {}", report.algorithm.name());
        }
        let mut c = if mode == CMode::Overwrite {
            vec![<T as Element>::zero(); lc]
        } else {
            cd.clone()
        };
        let one = <T as Element>::one();
        let half = <T as Element>::from_parts(Real::from_f64(0.5), Real::from_f64(0.0));
        let ns = median_ns(warmup, runs, || match mode {
            CMode::Overwrite => {
                let mut cv =
                    StridedViewMut::new(&mut c, &e.c.dims, &e.c.strides, oc as isize).expect("c");
                plan.execute_into(exec, one, &av, &bv, &mut cv)
                    .expect("exec");
            }
            // SAFETY: every origin addresses a buffer of its operand's span;
            // C is D itself only in `SeparateSame`, the same mapping.
            CMode::SeparateB0 => unsafe {
                let dp = c.as_mut_ptr().add(oc);
                let cp = cd.as_ptr().add(oc);
                plan.execute_raw(
                    exec,
                    one,
                    av.ptr(),
                    bv.ptr(),
                    <T as Element>::zero(),
                    cp,
                    dp,
                )
                .expect("exec");
            },
            CMode::SeparateB1 => unsafe {
                let dp = c.as_mut_ptr().add(oc);
                let cp = cd.as_ptr().add(oc);
                plan.execute_raw(exec, one, av.ptr(), bv.ptr(), half, cp, dp)
                    .expect("exec");
            },
            CMode::SeparateSame => unsafe {
                let dp = c.as_mut_ptr().add(oc);
                plan.execute_raw(exec, one, av.ptr(), bv.ptr(), half, dp as *const T, dp)
                    .expect("exec");
            },
        });
        println!("{name},{tag}_exec,{threads},{ns:.0},{runs}");
        outs.push(c);
    }
    let mag = |z: T| {
        let (re, im): (f64, f64) = (Real::to_f64(Element::re(z)), Real::to_f64(Element::im(z)));
        re.hypot(im)
    };
    let scale = outs[0].iter().map(|&z| mag(z)).fold(1.0, f64::max);
    let err = outs[0]
        .iter()
        .zip(&outs[1])
        .map(|(&x, &y)| mag(Element::sub(x, y)))
        .fold(0.0, f64::max)
        / scale;
    let tol = if matches!(e.dtype, Dtype::F32 | Dtype::C32) {
        1e-4
    } else {
        1e-12
    };
    println!(
        "CHECK {name} threads={threads} plan_vs_packed_rel={err:e} {}",
        if err < tol { "ok" } else { "MISMATCH" }
    );
}

fn dispatch(e: &DotGeneralEntry, exec: &Exec<'_>, threads: usize, warmup: usize, runs: usize) {
    match e.dtype {
        Dtype::F32 => run::<f32>(e, exec, threads, warmup, runs),
        Dtype::F64 => run::<f64>(e, exec, threads, warmup, runs),
        Dtype::C32 => run::<Complex32>(e, exec, threads, warmup, runs),
        Dtype::C64 => run::<Complex64>(e, exec, threads, warmup, runs),
    }
}

/// Built-in cases (f64 and c64) or the `dot_general` entries of `--corpus`.
fn cases() -> Vec<DotGeneralEntry> {
    let args: Vec<String> = std::env::args().collect();
    match args.iter().position(|a| a == "--corpus") {
        Some(i) => {
            let path = args
                .get(i + 1)
                .unwrap_or_else(|| panic!("--corpus needs a file"));
            let corpus = Corpus::load(path).unwrap_or_else(|e| panic!("{e}"));
            corpus
                .entries
                .into_iter()
                .map(|Entry::DotGeneral(d)| d)
                .collect()
        }
        None => CORPUS
            .iter()
            .flat_map(|c| [entry(c, Dtype::F64), entry(c, Dtype::C64)])
            .collect(),
    }
}

fn main() {
    let cases = cases();
    if std::env::args().any(|a| a == "--list") {
        for c in &cases {
            println!("{}", c.name);
        }
        return;
    }
    let threads = BenchThreads::from_args();
    threads.verify();
    let warmup = env_usize("BENCH_WARMUP", 3);
    let runs = env_usize("BENCH_RUNS", 20);
    let exact = std::env::var("BENCH_CASE").ok().filter(|f| !f.is_empty());
    println!("case,variant,threads,median_ns,samples");
    threads.with_exec(|exec, _| {
        for e in &cases {
            if exact.as_deref().is_some_and(|x| x != e.name) {
                continue;
            }
            let mults: usize = e.c.dims.iter().product::<usize>()
                * e.lc.iter().map(|&i| e.a.dims[i]).product::<usize>();
            let r = if mults >= 1 << 24 { runs.min(10) } else { runs };
            dispatch(e, exec, threads.requested, warmup, r);
        }
    });
}
