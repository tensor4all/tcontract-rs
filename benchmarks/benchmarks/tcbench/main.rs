//! `tcbench` -- correctness and performance harness over the TCCG corpus.
//!
//! Subcommands:
//!
//! * `run` -- the corpus across engines and dtypes, reporting GFLOP/s and the
//!   per-case complex efficiency ratio. Engines: `plan` (the planner's choice
//!   under the `TCBENCH_*` knobs), `packed` (the packed driver, forced),
//!   `ttgt` and `tblis` (external baselines).
//! * `verify` -- the planner's choice, the packed driver, TTGT and TBLIS over the
//!   whole corpus at benchmark sizes, in every dtype and under every
//!   stride-stress mode.
//! * `info` -- the machine and the baselines this binary was built with.
//!
//! Defaults to one thread; `--threads N` selects an enforced host budget. The `sweep`, `shapes`, `orient`
//! and `premise` analyses of the original harness were retired with the move into
//! `tcontract-bench`; `docs/migration-2026-10.md` records where they went.

mod blas;
mod corpus;
mod engines;
mod knobs;
mod report;
#[cfg(feature = "tblis")]
mod tblis;
mod ttgt;

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("help");

    if let Err(e) = knobs::init_from_env() {
        eprintln!("tcbench: {e}");
        return ExitCode::from(2);
    }
    let opts = Options::parse(&args[2.min(args.len())..]);

    let threads = tcontract_bench::threads::BenchThreads::from_args();
    threads.verify();
    engines::configure_threads(threads.requested);
    match cmd {
        "verify" => threads.with_exec(|exec, _| engines::verify::run(&opts, exec)),
        "run" => threads.with_exec(|exec, _| engines::run::run(&opts, exec)),
        "info" => {
            report::print_environment();
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!(
                "usage: tcbench <run|verify|info> [options]\n\
                 \n\
                 options:\n\
                 \x20 --threads <n>     host/provider thread budget (default 1)\n\
                 \x20 --size <MiB>      tensor size target for TCCG sizing (default 200)\n\
                 \x20 --reps <n>        timed repetitions per measurement (default 5)\n\
                 \x20 --prime-ms <n>    untimed time-based priming per arm, per engine (default 1500)\n\
                 \x20 --dtype <list>    comma separated: f32,f64,c32,c64 (default all)\n\
                 \x20 --case <substr>   only cases whose name contains this\n\
                 \x20 --engines <list>  comma separated, default all of:\n\
                 \x20                   plan,packed   (this library)\n\
                 \x20                   ttgt,tblis     (optional external baselines)\n\
                 \x20 --csv <path>      also write machine-readable results\n\
                 \x20 --stress <mode>   none|ragged|padded: perturb TCCG extents/layouts\n\
                 \x20                   so the block-scatter gather path is exercised\n"
            );
            ExitCode::FAILURE
        }
    }
}

/// Parsed command line.
pub struct Options {
    pub size_mib: f64,
    pub reps: usize,
    /// Untimed priming per arm, in milliseconds. Time-based rather than a call
    /// count: see [`engines::timed`].
    pub prime_ms: u64,
    pub dtypes: Vec<String>,
    pub case_filter: Option<String>,
    pub engines: Vec<String>,
    pub csv: Option<String>,
    pub stress: corpus::Stress,
}

impl Options {
    fn parse(args: &[String]) -> Options {
        let mut o = Options {
            size_mib: 200.0,
            reps: 5,
            prime_ms: 1500,
            dtypes: vec!["f32".into(), "f64".into(), "c32".into(), "c64".into()],
            case_filter: None,
            engines: crate::engines::run::ENGINE_ORDER
                .iter()
                .map(|s| s.to_string())
                .collect(),
            csv: None,
            stress: corpus::Stress::None,
        };
        let mut i = 0;
        while i < args.len() {
            let next = |i: usize| args.get(i + 1).cloned().unwrap_or_default();
            match args[i].as_str() {
                "--threads" => i += 1, // validated by BenchThreads
                "--size" => {
                    o.size_mib = next(i).parse().unwrap_or(200.0);
                    i += 1;
                }
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
                "--stress" => {
                    o.stress = corpus::Stress::parse(&next(i)).unwrap_or_else(|| {
                        eprintln!("unknown --stress value; expected none|ragged|padded");
                        std::process::exit(2)
                    });
                    i += 1;
                }
                other => eprintln!("warning: ignoring unknown option {other}"),
            }
            i += 1;
        }
        // Fail closed. A requested arm this binary was not built with produces no
        // rows at all, and a run that measures nothing else still exits successfully,
        // which would let a caller publish a cell with an arm silently missing.
        let supported = crate::engines::run::ENGINE_ORDER;
        if let Some(missing) = o.engines.iter().find(|e| !supported.contains(&e.as_str())) {
            eprintln!(
                "unknown or unavailable engine {missing:?}; this binary has {} \
                 (baselines are cargo features: `--features tblis` adds tblis, \
                 `--features blas` adds ttgt)",
                supported.join(",")
            );
            std::process::exit(2);
        }
        o
    }

    pub fn wants(&self, dtype: &str) -> bool {
        self.dtypes.iter().any(|d| d == dtype)
    }
    pub fn engine(&self, name: &str) -> bool {
        self.engines.iter().any(|e| e == name)
    }
    pub fn tensor_bytes(&self) -> f64 {
        self.size_mib * 1024.0 * 1024.0
    }
}
