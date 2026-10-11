//! Attribute the cost of the campaign's worst losing case, `abjc-cbka-kj` f64 16
//! MiB (effective GEMM m=92160, n=40, k=48), by layout.
//!
//! Four rows, one contraction, one kernel, one blocking, one thread budget: the
//! only thing that changes between them is where the operands live in memory. The
//! "corpus" layouts are the harness's own `la`/`lb`/`lc` for that case, not a
//! reconstruction - an earlier version of this file rebuilt them from the plan's
//! axis record, swapped two axes of equal extent, and measured a different
//! contraction (see the README, "Traversal attribution", for that retraction).
//!
//! What it shows today, and what it does not: the four rows land within 4% of one
//! another, so this case's cost is not a property of the operand layouts; and all
//! four are 1.37x slower than the campaign's own run of the same case with the same
//! plan (52 against 38 ms), so it cannot attribute that cost either. The
//! difference is in the execution path - this binary calls `execute_slices`, which
//! builds views per call, the harness calls `execute_raw` on raw pointers.
//!
//! Run it under the `tcontract-benchmark` skill protocol (pinned idle cores, one L3
//! domain, an A/A noise floor); it does no pinning itself.
//!
//! Usage: `traversal [--threads N] [--reps N] [--prime-ms N]`

use std::hint::black_box;
use std::time::Instant;

use tcontract_bench::threads::BenchThreads;
use tcontract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
use tcontract::{Plan, PlanConfig};
const A_DIMS: [usize; 4] = [48, 40, 48, 48];
const D_DIMS: [usize; 4] = [48, 40, 40, 48];

/// The case's own layouts, read from the harness (`la`, `lb`, `lc` of the sized
/// case, not reconstructed): `A = (a,b,j,c)` at `[1, 48, 1920, 92160]`, so the
/// `m` axes are contiguous but the contracted axis `j` sits *inside* the `b`/`c`
/// plane at stride 1920; `D = (c,b,n,a)` at `[1, 48, 1920, 76800]`. The labels are
/// the harness's own ids.
const A_CORPUS: [isize; 4] = [1, 48, 1920, 92160];
const B_CORPUS: [isize; 2] = [1, 48];
const D_CORPUS: [isize; 4] = [1, 48, 1920, 76800];
/// The same contraction with `A`'s axes ordered so that the `(m, k)` matrix is
/// contiguous (`m` axes `a, b, c` then the contracted `j`), and `D`'s so that
/// `(m, n)` is.
const A_COMPACT: [isize; 4] = [1, 48, 1920, 76800];
const D_COMPACT: [isize; 4] = [1, 48, 1920, 92160];
const A_LABELS: [i64; 4] = [99, 98, 107, 97];
const B_LABELS: [i64; 2] = [107, 106];
const D_LABELS: [i64; 4] = [97, 98, 106, 99];

fn spec(dims: &[usize], strides: &[isize]) -> OperandSpec {
    OperandSpec::new(LayoutSpec::new(dims, strides, 0).unwrap())
}

fn span(dims: &[usize], strides: &[isize]) -> usize {
    dims.iter()
        .zip(strides)
        .fold(1usize, |n, (&e, &s)| n + (e - 1) * s.unsigned_abs())
}

fn problem(a: &[isize; 4], d: &[isize; 4]) -> Problem {
    Problem::from_labels(
        DType::F64,
        spec(&A_DIMS, a),
        spec(&[48, 40], &B_CORPUS),
        CSpec::Absent,
        spec(&D_DIMS, d),
        &Labels::new(&A_LABELS, &B_LABELS, &D_LABELS),
    )
    .unwrap()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let threads = BenchThreads::from_args();
    threads.verify();
    let wanted = tcontract_bench::threads::parse_threads(&args).unwrap_or(1);
    let reps = args
        .iter()
        .position(|a| a == "--reps")
        .map_or(5, |i| args[i + 1].parse::<usize>().unwrap());
    let prime_ms = args
        .iter()
        .position(|a| a == "--prime-ms")
        .map_or(1500, |i| args[i + 1].parse::<u64>().unwrap());

    let a_len = span(&A_DIMS, &A_CORPUS).max(span(&A_DIMS, &A_COMPACT));
    let d_len = span(&D_DIMS, &D_CORPUS).max(span(&D_DIMS, &D_COMPACT));
    let mut a = vec![0.0f64; a_len];
    let mut b = vec![0.0f64; 48 * 40];
    let mut rng = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng as f64 / u64::MAX as f64) * 2.0 - 1.0
    };
    for x in a.iter_mut().chain(b.iter_mut()) {
        *x = next();
    }
    let flops = 2.0 * 92160.0 * 40.0 * 48.0;

    let variants: [(&str, [isize; 4], [isize; 4]); 4] = [
        ("corpus-a-corpus-d", A_CORPUS, D_CORPUS),
        ("corpus-a-compact-d", A_CORPUS, D_COMPACT),
        ("compact-a-corpus-d", A_COMPACT, D_CORPUS),
        ("compact-a-compact-d", A_COMPACT, D_COMPACT),
    ];

    println!("variant,threads,best_s,gflops,residual");
    let mut reference: Option<Vec<f64>> = None;
    for (name, a_s, d_s) in variants {
        let p = problem(&a_s, &d_s);
        let plan = Plan::<f64>::new(&p, &PlanConfig::packed()).unwrap();
        let mut d = vec![0.0f64; d_len];
        eprintln!("PLAN {name} {:?}", plan.report());
        let (secs, residual) = threads.with_exec(|exec, _| {
            let until = Instant::now() + std::time::Duration::from_millis(prime_ms);
            while Instant::now() < until {
                plan.execute_slices(exec, 1.0, (&a, 0), (&b, 0), (&mut d, 0))
                    .unwrap();
            }
            let mut best = f64::INFINITY;
            for _ in 0..reps.max(1) {
                let t = Instant::now();
                plan.execute_slices(exec, 1.0, (&a, 0), (&b, 0), (&mut d, 0))
                    .unwrap();
                best = best.min(t.elapsed().as_secs_f64());
                black_box(&d);
            }
            // Every variant computes the same contraction, so the compact pair is
            // the oracle for the interleaved ones: a layout cannot change the
            // answer, and a nonzero residual here means it did.
            let residual = match &reference {
                None => {
                    reference = Some(d.clone());
                    0.0
                }
                Some(want) => {
                    let num: f64 = d.iter().zip(want).map(|(x, y)| (x - y).powi(2)).sum();
                    let den: f64 = want.iter().map(|x| x * x).sum();
                    (num / den).sqrt()
                }
            };
            (best, residual)
        });
        println!(
            "{name},{},{secs:.9},{:.4},{residual:.2e}",
            wanted,
            flops / secs / 1e9
        );
    }
}
