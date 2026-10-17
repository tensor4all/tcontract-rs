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
//!   corpus as this binary sees it (case, dtype, steps, MACs, `m`/`n`/`k`). The
//!   toolkit's `key=value` contract lines are appended after that.
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

use crate::engines::Statistic;

/// Exit with status 2 and a clear message. A bad flag must not be ignored: the
/// evidence columns it would have set are read by a downstream toolkit.
fn die(msg: &str) -> ! {
    eprintln!("lukbench: {msg}");
    std::process::exit(2)
}

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
            threads.with_exec(|exec, _| {
                report::print_environment();
                println!();
                report::print_corpus();
                report::print_evidence_keys(threads.requested, exec.budget());
            });
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!(
                "usage: lukbench <run|verify|info> [options]\n\
                 \n\
                 options:\n\
                 \x20 --threads <n>     host/provider thread budget (default 1)\n\
                 \x20 --size <MiB>      accepted and ignored: this corpus has fixed extents\n\
                 \x20 --n <spec>        declared size spec, echoed into the census; this\n\
                 \x20                   corpus has no size knob, so it selects and resizes\n\
                 \x20                   nothing (default empty)\n\
                 \x20 --reps <n>        timed repetitions per measurement (default 5)\n\
                 \x20 --prime-ms <n>    untimed time-based priming per arm, per engine (default 1500)\n\
                 \x20 --dtype <list>    comma separated: f64,c64 (default both); a case whose\n\
                 \x20                   own dtype is not requested is still declared, with\n\
                 \x20                   every cell skipped\n\
                 \x20 --case <substr>   only cases whose name contains this\n\
                 \x20 --family <list>   comma separated exact case ids (default all),\n\
                 \x20                   intersected with --case\n\
                 \x20 --batch <list>    comma separated declared batch values; this\n\
                 \x20                   corpus has unit batch, so every value must be 1\n\
                 \x20 --regime <label>  regime label echoed into the evidence (default\n\
                 \x20                   `default`)\n\
                 \x20 --engines <list>  comma separated, default all of:\n\
                 \x20                   plan,packed    (this library)\n\
                 \x20                   tblis          (optional external baseline, cargo feature)\n\
                 \x20 --rows <list>     comma separated engine rows to run, in order\n\
                 \x20                   (default the declared engine list), intersected\n\
                 \x20                   with --engines\n\
                 \x20 --rotate-rows <k> rotate the engine execution order by k positions\n\
                 \x20                   (default 0); the reported row name is unchanged\n\
                 \x20 --statistic <s>   best|median of the per-call samples (default best)\n\
                 \x20 --csv <path>      also write machine-readable results\n\
                 \x20 --census-csv <p>  also write the declared-cell census\n"
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
    /// Exact case-id filter, intersected with `case_filter`.
    pub families: Vec<String>,
    /// The engine rows to measure, in the order `--rows`/`--engines` gave them.
    /// Rotation is applied only to execution order, not to the reported order.
    pub engines: Vec<String>,
    /// How far to rotate [`Options::exec_order`] by.
    pub rotate: usize,
    /// Which summary of the chronological samples is reported.
    pub statistic: Statistic,
    /// The run's regime label, echoed into the evidence.
    pub regime: String,
    /// The effective unit batch, echoed into the evidence.
    pub batch: String,
    /// The declared size spec (`--n`), echoed verbatim into the census identity
    /// (`m = n = --n`, `k` empty). Empty when the flag was not given.
    pub n_spec: String,
    pub csv: Option<String>,
    pub census_csv: Option<String>,
}

impl Options {
    fn parse(args: &[String]) -> Options {
        let mut o = Options {
            reps: 5,
            prime_ms: 1500,
            dtypes: vec!["f64".into(), "c64".into()],
            case_filter: None,
            families: Vec::new(),
            engines: engines::run::ENGINE_ORDER
                .iter()
                .map(|s| s.to_string())
                .collect(),
            rotate: 0,
            statistic: Statistic::Best,
            regime: "default".into(),
            batch: "1".into(),
            n_spec: String::new(),
            csv: None,
            census_csv: None,
        };
        let mut rows: Vec<String> = Vec::new();
        let mut engines_explicit = false;
        let mut i = 0;
        while i < args.len() {
            let next = |i: usize| args.get(i + 1).cloned().unwrap_or_default();
            match args[i].as_str() {
                "--threads" => i += 1, // validated by BenchThreads
                // Accepted and ignored: this corpus is a fixed set of shapes
                // (tensor4all/tcontract-rs#61) with no size knob. The campaign
                // runner passes `--size` to every harness.
                "--size" => i += 1,
                "--n" => {
                    // Echoed verbatim: this corpus has no size knob, so the spec
                    // names the toolkit's declared grid and nothing else.
                    o.n_spec = next(i);
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
                other => die(&format!("unknown option {other}")),
            }
            i += 1;
        }
        // Fail closed. A requested arm this binary was not built with produces no
        // rows at all, and a run that measures nothing else still exits
        // successfully, which would let a caller publish a cell with an arm
        // silently missing.
        let supported = engines::run::ENGINE_ORDER;
        for value in rows.iter().chain(o.engines.iter()) {
            if !supported.contains(&value.as_str()) {
                die(&format!(
                    "unknown or unavailable engine {value:?}; this binary has {} \
                     (the baseline is a cargo feature: `--features tblis` adds tblis)",
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

/// The corpus, filtered by `--case`/`--family` (exact ids), in corpus order.
///
/// The dtype axis is *not* applied here. A case fixes its own dtype, and the
/// toolkit's declared grid multiplies every family by the suite's whole dtype
/// list, so a wrong-dtype cell has to be declared and skipped rather than the
/// case silently dropped. `run`/`verify` time only a case whose own dtype was
/// requested.
pub fn select_cases(opts: &Options) -> Vec<corpus::Program> {
    corpus::corpus()
        .into_iter()
        .filter(|p| opts.case_selected(p.name.as_str()))
        .collect()
}
