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

use crate::engines::Statistic;

/// Exit with status 2 and a clear message. A bad flag must not be ignored: the
/// evidence columns it would have set are read by a downstream toolkit.
fn die(msg: &str) -> ! {
    eprintln!("tcbench: {msg}");
    std::process::exit(2)
}

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
            threads.with_exec(|exec, _| {
                report::print_environment();
                report::print_evidence_keys(threads.requested, exec.budget());
            });
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!(
                "usage: tcbench <run|verify|info> [options]\n\
                 \n\
                 options:\n\
                 \x20 --threads <n>     host/provider thread budget (default 1)\n\
                 \x20 --size <MiB>      tensor size target for TCCG sizing (default 200)\n\
                 \x20 --n <spec>        declared size spec, echoed into the census; for\n\
                 \x20                   tcbench it is the tensor-size target in MiB, so\n\
                 \x20                   --n 16 behaves like --size 16 (default empty)\n\
                 \x20 --reps <n>        timed repetitions per measurement (default 5)\n\
                 \x20 --prime-ms <n>    untimed time-based priming per arm, per engine (default 1500)\n\
                 \x20 --dtype <list>    comma separated: f32,f64,c32,c64 (default all)\n\
                 \x20 --case <substr>   only cases whose name contains this\n\
                 \x20 --family <list>   comma separated exact case ids (default all),\n\
                 \x20                   intersected with --case\n\
                 \x20 --batch <list>    comma separated declared batch values; this\n\
                 \x20                   corpus has unit batch, so every value must be 1\n\
                 \x20 --regime <label>  regime label echoed into the evidence (default\n\
                 \x20                   `default`)\n\
                 \x20 --engines <list>  comma separated, default all of:\n\
                 \x20                   plan,packed   (this library)\n\
                 \x20                   ttgt,tblis     (optional external baselines)\n\
                 \x20 --rows <list>     comma separated engine rows to run, in order\n\
                 \x20                   (default the declared engine list), intersected\n\
                 \x20                   with --engines\n\
                 \x20 --rotate-rows <k> rotate the engine execution order by k positions\n\
                 \x20                   (default 0); the reported row name is unchanged\n\
                 \x20 --statistic <s>   best|median of the per-call samples (default best)\n\
                 \x20 --csv <path>      also write machine-readable results\n\
                 \x20 --census-csv <p>  also write the declared-cell census\n\
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
    /// The declared size spec, echoed verbatim into the census. `--n`'s value;
    /// empty when the flag was not given.
    pub n_spec: String,
    pub reps: usize,
    /// Untimed priming per arm, in milliseconds. Time-based rather than a call
    /// count: see [`engines::timed`].
    pub prime_ms: u64,
    pub dtypes: Vec<String>,
    pub case_filter: Option<String>,
    /// Exact case-id filter, intersected with `case_filter`.
    pub families: Vec<String>,
    /// The engine rows to measure, in the order `--rows`/`--engines` gave them.
    /// Rotation is applied only to execution order, not to the reported order.
    pub engines: Vec<String>,
    /// How far to rotate [`Options::exec_order`] by.
    pub rotate: usize,
    pub statistic: Statistic,
    /// The run's regime label, echoed into the evidence.
    pub regime: String,
    /// The effective unit batch, echoed into the evidence.
    pub batch: String,
    pub csv: Option<String>,
    pub census_csv: Option<String>,
    pub stress: corpus::Stress,
}

impl Options {
    fn parse(args: &[String]) -> Options {
        let mut o = Options {
            size_mib: 200.0,
            n_spec: String::new(),
            reps: 5,
            prime_ms: 1500,
            dtypes: vec!["f32".into(), "f64".into(), "c32".into(), "c64".into()],
            case_filter: None,
            families: Vec::new(),
            engines: crate::engines::run::ENGINE_ORDER
                .iter()
                .map(|s| s.to_string())
                .collect(),
            rotate: 0,
            statistic: Statistic::Best,
            regime: "default".into(),
            batch: "1".into(),
            csv: None,
            census_csv: None,
            stress: corpus::Stress::None,
        };
        let mut rows: Vec<String> = Vec::new();
        let mut size_arg: Option<f64> = None;
        let mut n_arg: Option<f64> = None;
        let mut engines_explicit = false;
        let mut i = 0;
        while i < args.len() {
            let next = |i: usize| args.get(i + 1).cloned().unwrap_or_default();
            match args[i].as_str() {
                "--threads" => i += 1, // validated by BenchThreads
                "--size" => {
                    size_arg = Some(next(i).parse().unwrap_or(200.0));
                    i += 1;
                }
                "--n" => {
                    let v = next(i);
                    n_arg = Some(v.parse().unwrap_or_else(|_| {
                        die(&format!(
                            "--n {v:?}: for tcbench the declared spec is the tensor-size \
                             target in MiB, so it must be a number"
                        ))
                    }));
                    o.n_spec = v;
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
                "--family" => {
                    o.families = split_list(&next(i));
                    i += 1;
                }
                "--batch" => {
                    let v = next(i);
                    if v.split(',').any(|b| b.trim() != "1") {
                        die(&format!(
                            "--batch {v:?}: this corpus has unit batch; every value must be 1"
                        ));
                    }
                    i += 1;
                }
                "--regime" => {
                    o.regime = next(i);
                    i += 1;
                }
                "--engines" => {
                    engines_explicit = true;
                    // The old flag keeps its exact parsing: `.split(',')` with no
                    // filtering, so a trailing comma is still an error.
                    o.engines = next(i).split(',').map(|s| s.to_string()).collect();
                    i += 1;
                }
                "--rows" => {
                    rows = split_list(&next(i));
                    i += 1;
                }
                "--rotate-rows" => {
                    let v = next(i);
                    o.rotate = v.parse().unwrap_or_else(|_| {
                        die(&format!("--rotate-rows {v:?}: expected a whole number"))
                    });
                    i += 1;
                }
                "--statistic" => {
                    let v = next(i);
                    o.statistic = Statistic::parse(&v).unwrap_or_else(|| {
                        die(&format!("--statistic {v:?}: expected best|median"))
                    });
                    i += 1;
                }
                "--csv" => {
                    o.csv = Some(next(i));
                    i += 1;
                }
                "--census-csv" => {
                    o.census_csv = Some(next(i));
                    i += 1;
                }
                "--stress" => {
                    o.stress = corpus::Stress::parse(&next(i)).unwrap_or_else(|| {
                        eprintln!("unknown --stress value; expected none|ragged|padded");
                        std::process::exit(2)
                    });
                    i += 1;
                }
                other => {
                    eprintln!("unknown option {other}");
                    std::process::exit(2);
                }
            }
            i += 1;
        }
        // `--n` and `--size` name the same knob; two disagreeing values are a
        // caller error rather than a last-flag-wins surprise.
        o.size_mib = match (n_arg, size_arg) {
            (Some(n), Some(sz)) if n != sz => die(&format!(
                "--n {n} conflicts with --size {sz} MiB; they name the same target"
            )),
            (Some(n), _) => n,
            (None, Some(sz)) => sz,
            (None, None) => 200.0,
        };
        // Fail closed. A requested arm this binary was not built with produces no
        // rows at all, and a run that measures nothing else still exits successfully,
        // which would let a caller publish a cell with an arm silently missing.
        let supported = crate::engines::run::ENGINE_ORDER;
        for value in rows.iter().chain(o.engines.iter()) {
            if !supported.contains(&value.as_str()) {
                die(&format!(
                    "unknown or unavailable engine {value:?}; this binary has {} \
                     (baselines are cargo features: `--features tblis` adds tblis, \
                     `--features blas` adds ttgt)",
                    supported.join(",")
                ));
            }
        }
        // `--rows` selects and orders the declared rows; `--engines` keeps
        // working as a filter on top of them.
        if !rows.is_empty() {
            o.engines = if engines_explicit {
                rows.into_iter().filter(|e| o.engines.contains(e)).collect()
            } else {
                rows
            };
        }
        o
    }

    pub fn wants(&self, dtype: &str) -> bool {
        self.dtypes.iter().any(|d| d == dtype)
    }
    pub fn engine(&self, name: &str) -> bool {
        self.engines.iter().any(|e| e == name)
    }
    /// Whether `--case`/`--family` select this case id.
    pub fn case_selected(&self, name: &str) -> bool {
        if !self.families.is_empty() && !self.families.iter().any(|f| f == name) {
            return false;
        }
        if let Some(f) = &self.case_filter {
            if !name.contains(f.as_str()) {
                return false;
            }
        }
        true
    }
    /// The engine rows in the order the run executes them: the reported order
    /// rotated by `--rotate-rows`. Names are unchanged; only the schedule moves.
    pub fn exec_order(&self) -> Vec<String> {
        if self.engines.is_empty() {
            return Vec::new();
        }
        let k = self.rotate % self.engines.len();
        let mut v = self.engines[k..].to_vec();
        v.extend_from_slice(&self.engines[..k]);
        v
    }
    pub fn tensor_bytes(&self) -> f64 {
        self.size_mib * 1024.0 * 1024.0
    }
}

/// A comma-separated flag value, dropping empty entries so a trailing comma is
/// not a phantom filter.
fn split_list(v: &str) -> Vec<String> {
    v.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}
