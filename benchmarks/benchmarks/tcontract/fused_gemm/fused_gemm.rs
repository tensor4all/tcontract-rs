//! The fused grouped-GEMM route (`tcontract::FusedGemmPlan`, tcontract-rs#208)
//! against the per-job route (`contract_batched`, one `Plan` per distinct
//! geometry), at an enforced thread count.
//!
//! Usage: `fused_gemm --threads N [--list]`. CSV
//! `case,variant,threads,median_ns,samples`, each row followed by a
//! `# spread case variant threads min p25 p75 max` line (ns). Variants: `fused`;
//! `per_job_packed` (each geometry's `Plan` with the packed driver forced, the
//! per-job reference the fused route must reproduce bitwise); and
//! `per_job_default` (the planner's own strategy per geometry). Before timing a
//! case, every variant's result is checked on the measured context (bitwise
//! for the two packed ones, exactly by value for the default one) and a
//! mismatch exits with status 1. `--list` prints the case names (for
//! `scripts/paired.sh`). Environment: `BENCH_RUNS` (default 50),
//! `BENCH_WARMUP_MS` (default 1500: untimed calls for that much wall time per
//! arm), `BENCH_WARMUP` (default 5 more calls), `BENCH_CASE` (exact case name),
//! `BENCH_FILTER` (substring).
use std::hint::black_box;
use std::time::Duration;

use strided_view::{StridedView, StridedViewMut};
use t4a_desc::gemm::{GemmBatch, GemmJob, GemmOps, Operand, Trans};
use t4a_desc::ScalarKind;
use tcontract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
use tcontract::{contract_batched, BatchItem, FusedGemmPlan, Partition, Plan, PlanConfig};
use tcontract_bench::threads::BenchThreads;
use tcontract_bench::timing::{env_usize, samples_ns};
use tcontract_exec::Exec;

struct Cfg {
    threads: usize,
    prime: Duration,
    warmup: usize,
    runs: usize,
}

impl Cfg {
    fn row(&self, case: &str, variant: &str, ns: f64) {
        println!("{case},{variant},{},{ns:.0},{}", self.threads, self.runs);
    }
}

/// `count` jobs of one `(m, n, k)`.
type Run = (usize, (usize, usize, usize));

/// The issue's cases: `(name, [(count, (m, n, k))])`, all `(N, N)`, compact
/// and back to back in each buffer.
fn cases() -> Vec<(String, Vec<Run>)> {
    let mut v = vec![
        ("16384x32^3".to_string(), vec![(16384, (32, 32, 32))]),
        ("4096x8^3".to_string(), vec![(4096, (8, 8, 8))]),
        (
            "issue46_1x1024x256x1024+8192x32^3".to_string(),
            vec![(1, (1024, 1024, 256)), (8192, (32, 32, 32))],
        ),
    ];
    // Uniform block sets: 46 blocks of degeneracy d (symtn's U(1) ledger range).
    for d in [2usize, 4, 8, 16] {
        v.push((format!("uniform46_d{d}"), vec![(46, (d, d, d))]));
    }
    v
}

fn jobs(spec: &[Run]) -> Vec<GemmJob> {
    let ops = GemmOps {
        a: Trans::N,
        b: Trans::N,
    };
    let mut at = [0usize; 3];
    let mut out = Vec::new();
    for &(count, (m, n, k)) in spec {
        for _ in 0..count {
            out.push(GemmJob::compact(ops, at, m, n, k));
            at[0] += m * k;
            at[1] += k * n;
            at[2] += m * n;
        }
    }
    out
}

/// One `Plan` per distinct geometry under `config`, and each job's plan index.
fn per_job_plans(jobs: &[GemmJob], config: &PlanConfig) -> (Vec<Plan<f64>>, Vec<usize>) {
    let mut shapes = Vec::new();
    let mut plans = Vec::new();
    let index = jobs
        .iter()
        .map(|j| {
            let s = (j.m, j.n, j.k);
            match shapes.iter().position(|x| *x == s) {
                Some(i) => i,
                None => {
                    let l = |d: [usize; 2]| {
                        OperandSpec::new(
                            LayoutSpec::new(&d, &[1, d[0] as isize], 0).expect("layout"),
                        )
                    };
                    let problem = Problem::from_labels(
                        DType::F64,
                        l([j.m, j.k]),
                        l([j.k, j.n]),
                        CSpec::Absent,
                        l([j.m, j.n]),
                        &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
                    )
                    .expect("problem");
                    shapes.push(s);
                    plans.push(Plan::<f64>::new(&problem, config).expect("plan"));
                    plans.len() - 1
                }
            }
        })
        .collect();
    (plans, index)
}

/// A column-major `rows x cols` view at `off` of `x`.
fn view(x: &[f64], off: usize, rows: usize, cols: usize) -> StridedView<'_, f64> {
    StridedView::new(
        &x[off..off + rows * cols],
        &[rows, cols],
        &[1, rows as isize],
        0,
    )
    .expect("view")
}

/// One `contract_batched` call over every job. The items (views of the shared
/// buffers) are built inside the call, as a caller of the per-job route does.
fn run_per_job(
    exec: &mut Exec<'_>,
    jobs: &[GemmJob],
    (plans, index): &(Vec<Plan<f64>>, Vec<usize>),
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
) {
    let mut rest = c;
    let mut items = Vec::with_capacity(jobs.len());
    for (j, &p) in jobs.iter().zip(index) {
        let (mine, tail) = rest.split_at_mut(j.m * j.n);
        rest = tail;
        items.push(BatchItem {
            plan: &plans[p],
            alpha: 1.0,
            a: view(a, j.a_offset, j.m, j.k),
            b: view(b, j.b_offset, j.k, j.n),
            beta: 0.0,
            source: None,
            d: StridedViewMut::new(mine, &[j.m, j.n], &[1, j.m as isize], 0).expect("view"),
        });
    }
    contract_batched(&mut items, exec).expect("contract_batched");
}

/// Time `f` (after `prime` of wall time and `warmup` calls) and print its CSV
/// row and its `# spread` line (min, p25, p75, max in ns), as `contract` does.
fn timed(cfg: &Cfg, case: &str, variant: &str, f: impl FnMut()) {
    let s = samples_ns(cfg.prime, cfg.warmup, cfg.runs, f);
    let q = |k: usize| s[(s.len() - 1) * k / 4];
    cfg.row(case, variant, s[s.len() / 2]);
    println!(
        "# spread {case} {variant} {} {:.0} {:.0} {:.0} {:.0}",
        cfg.threads,
        q(0),
        q(1),
        q(3),
        q(4)
    );
}

/// Abort before any timing unless `got` equals `want` exactly: bitwise when
/// `bitwise`, otherwise by value (the inputs are small multiples of 1/2, so every
/// route computes the exact sum and only the sign of a zero may differ).
fn check(name: &str, variant: &str, got: &[f64], want: &[f64], bitwise: bool) {
    let bad = got.iter().zip(want).position(|(x, y)| match bitwise {
        true => x.to_bits() != y.to_bits(),
        false => x != y,
    });
    if let Some(i) = bad {
        eprintln!(
            "CHECK {name} {variant} MISMATCH at {i}: {} vs {}",
            got[i], want[i]
        );
        std::process::exit(1);
    }
}

fn main() {
    let cases = cases();
    if std::env::args().any(|a| a == "--list") {
        for (name, _) in &cases {
            println!("{name}");
        }
        return;
    }
    let threads = BenchThreads::from_args();
    threads.verify();
    let cfg = Cfg {
        threads: threads.requested,
        prime: Duration::from_millis(env_usize("BENCH_WARMUP_MS", 1500) as u64),
        warmup: env_usize("BENCH_WARMUP", 5),
        runs: env_usize("BENCH_RUNS", 50),
    };
    let exact = std::env::var("BENCH_CASE").ok().filter(|f| !f.is_empty());
    let filter = std::env::var("BENCH_FILTER").ok().filter(|f| !f.is_empty());
    let packed = PlanConfig {
        partition: Some(Partition::StaticGrid {
            pin: None,
            align_c_lines: false,
        }),
        ..PlanConfig::default()
    };
    println!("case,variant,threads,median_ns,samples");
    threads.with_exec(|exec, _| {
        for (name, spec) in cases {
            let wanted = match (&exact, &filter) {
                (Some(e), _) => *e == name,
                (None, Some(f)) => name.contains(f.as_str()),
                (None, None) => true,
            };
            if !wanted {
                continue;
            }
            let jobs = jobs(&spec);
            let ops = GemmOps {
                a: Trans::N,
                b: Trans::N,
            };
            let batch = GemmBatch::new(ScalarKind::F64, ops, jobs.clone()).expect("batch");
            let (la, lb, lc) = (
                batch.required_len(Operand::A),
                batch.required_len(Operand::B),
                batch.required_len(Operand::C),
            );
            let a: Vec<f64> = (0..la).map(|x| (x % 13) as f64 - 6.0).collect();
            let b: Vec<f64> = (0..lb).map(|x| (x % 7) as f64 * 0.5).collect();
            let fused = FusedGemmPlan::<f64>::new(batch, &PlanConfig::default()).expect("fused");
            let packed_plans = per_job_plans(&jobs, &packed);
            let default_plans = per_job_plans(&jobs, &PlanConfig::default());

            // Correctness before timing, on the measured context: the fused
            // result is bitwise the per-job packed one, and every variant
            // computes the exact product. A mismatch aborts the run.
            let mut want = vec![0.0; lc];
            run_per_job(&mut Exec::serial(), &jobs, &packed_plans, &a, &b, &mut want);
            let mut c = vec![f64::NAN; lc];
            let route = fused
                .execute_into(exec, 1.0, &a, &b, &mut c)
                .expect("fused run");
            check(&name, "fused", &c, &want, true);
            for (variant, plans) in [
                ("per_job_packed", &packed_plans),
                ("per_job_default", &default_plans),
            ] {
                let mut got = vec![f64::NAN; lc];
                run_per_job(exec, &jobs, plans, &a, &b, &mut got);
                check(&name, variant, &got, &want, variant == "per_job_packed");
            }
            println!(
                "CHECK {name} threads={} route={route:?} items={} ok",
                cfg.threads,
                fused.work_items()
            );

            // Every arm gets the same wall-clock priming and call count.
            timed(&cfg, &name, "fused", || {
                fused
                    .execute_into(exec, 1.0, black_box(&a), black_box(&b), black_box(&mut c))
                    .expect("fused run");
            });
            for (variant, plans) in [
                ("per_job_packed", &packed_plans),
                ("per_job_default", &default_plans),
            ] {
                timed(&cfg, &name, variant, || {
                    run_per_job(
                        exec,
                        &jobs,
                        plans,
                        black_box(&a),
                        black_box(&b),
                        black_box(&mut c),
                    )
                });
            }
        }
    });
}
