//! `lukbench` -- Lukas Devos's per-shape corpus (tensor4all/tcontract-rs#61) with a
//! TBLIS arm, as the second suite the tcontract-benchmark campaign drives.
//!
//! Subcommands:
//!
//! * `run` -- every case in the corpus across engines, reporting GFLOP/s. One
//!   timed call is the whole program (a chain of pairwise steps), not one step.
//! * `verify` -- every case and every compiled-in arm once, untimed, against the
//!   naive label-loop reference. Prints `all comparisons within tolerance` and
//!   exits non-zero on a mismatch.
//! * `info` -- the machine, the baselines this binary was built with, and the
//!   corpus as this binary sees it (case, dtype, steps, MACs, `m`/`n`/`k`).
//!
//! Engines: `plan` (the planner's own choice, `PlanConfig::default()`), `packed`
//! (the packed driver forced, `PlanConfig::packed()`) and `tblis` (the C++ TBLIS
//! adapter, under `--features tblis` / `tblis13`). The default build needs no
//! external library.
//!
//! Defaults to one thread; `--threads N` selects an enforced host budget.
//! `--size` is accepted and ignored: unlike `tcbench`'s TCCG corpus, this corpus
//! has fixed extents and no size knob, and the flag exists so the campaign's
//! runner can pass the same command line to both harnesses.

mod corpus;
mod engines;
mod report;
#[cfg(feature = "tblis")]
#[path = "../tcbench/tblis.rs"]
mod tblis;

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("help");
    let opts = Options::parse(&args[2.min(args.len())..]);

    let threads = tcontract_bench::threads::BenchThreads::from_args();
    threads.verify();
    engines::configure_threads(threads.requested);
    match cmd {
        "verify" => threads.with_exec(|exec, _| engines::verify::run(&opts, exec)),
        "run" => threads.with_exec(|exec, _| engines::run::run(&opts, exec)),
        "info" => {
            report::print_environment();
            println!();
            report::print_corpus();
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!(
                "usage: lukbench <run|verify|info> [options]\n\
                 \n\
                 options:\n\
                 \x20 --threads <n>     host/provider thread budget (default 1)\n\
                 \x20 --size <MiB>      accepted and ignored: this corpus has fixed extents\n\
                 \x20 --reps <n>        timed repetitions per measurement (default 5)\n\
                 \x20 --prime-ms <n>    untimed time-based priming per arm, per engine (default 1500)\n\
                 \x20 --dtype <list>    comma separated: f64,c64 (default both; a dtype the\n\
                 \x20                   corpus does not use selects nothing)\n\
                 \x20 --case <substr>   only cases whose name contains this\n\
                 \x20 --engines <list>  comma separated, default all of:\n\
                 \x20                   plan,packed    (this library)\n\
                 \x20                   tblis          (optional external baseline, cargo feature)\n\
                 \x20 --csv <path>      also write machine-readable results\n"
            );
            ExitCode::FAILURE
        }
    }
}

/// Parsed command line.
pub struct Options {
    pub reps: usize,
    /// Untimed priming per arm, in milliseconds. Time-based rather than a call
    /// count: see [`engines::timed`].
    pub prime_ms: u64,
    pub dtypes: Vec<String>,
    pub case_filter: Option<String>,
    pub engines: Vec<String>,
    pub csv: Option<String>,
}

impl Options {
    fn parse(args: &[String]) -> Options {
        let mut o = Options {
            reps: 5,
            prime_ms: 1500,
            dtypes: vec!["f64".into(), "c64".into()],
            case_filter: None,
            engines: engines::run::ENGINE_ORDER
                .iter()
                .map(|s| s.to_string())
                .collect(),
            csv: None,
        };
        let mut i = 0;
        while i < args.len() {
            let next = |i: usize| args.get(i + 1).cloned().unwrap_or_default();
            match args[i].as_str() {
                "--threads" => i += 1, // validated by BenchThreads
                // Accepted and ignored: this corpus is a fixed set of shapes
                // (tensor4all/tcontract-rs#61) with no size knob. The campaign
                // runner passes `--size` to every harness.
                "--size" => i += 1,
                "--reps" => {
                    o.reps = next(i).parse().unwrap_or(5);
                    i += 1;
                }
                "--prime-ms" => {
                    o.prime_ms = next(i).parse().unwrap_or(1500);
                    i += 1;
                }
                "--dtype" => {
                    o.dtypes = next(i).split(',').map(|s| s.to_string()).collect();
                    i += 1;
                }
                "--case" => {
                    o.case_filter = Some(next(i));
                    i += 1;
                }
                "--engines" => {
                    o.engines = next(i).split(',').map(|s| s.to_string()).collect();
                    i += 1;
                }
                "--csv" => {
                    o.csv = Some(next(i));
                    i += 1;
                }
                other => eprintln!("warning: ignoring unknown option {other}"),
            }
            i += 1;
        }
        for e in &o.engines {
            if !engines::run::ENGINE_ORDER.contains(&e.as_str()) {
                eprintln!(
                    "warning: engine `{e}` is not built into this binary (it needs a cargo \
                     feature); it will be skipped. A suite that names it must be run against \
                     a binary built with that feature."
                );
            }
        }
        o
    }

    pub fn wants(&self, dtype: &str) -> bool {
        self.dtypes.iter().any(|d| d == dtype)
    }
    pub fn engine(&self, name: &str) -> bool {
        self.engines.iter().any(|e| e == name)
    }
}

/// The corpus, filtered by `--dtype` (each case fixes its own dtype) and
/// `--case`, in corpus order.
pub fn select_cases(opts: &Options) -> Vec<corpus::Program> {
    corpus::corpus()
        .into_iter()
        .filter(|p| opts.wants(p.dtype))
        .filter(|p| {
            opts.case_filter
                .as_ref()
                .map(|f| p.name.contains(f.as_str()))
                .unwrap_or(true)
        })
        .collect()
}
