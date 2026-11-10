//! `qcbench` -- fixed-extent DF-RCCSD contractions with tcbench's engines.

#[path = "../tcbench/blas.rs"]
mod blas;
mod corpus;
#[path = "../tcbench/engines/mod.rs"]
mod engines;
#[path = "../tcbench/knobs.rs"]
mod knobs;
#[path = "../tcbench/report.rs"]
mod report;
#[cfg(feature = "tblis")]
#[path = "../tcbench/tblis.rs"]
mod tblis;
#[path = "../tcbench/ttgt.rs"]
mod ttgt;

use std::process::ExitCode;

use crate::engines::Statistic;

const ENGINE_ORDER: &[&str] = &[
    "plan",
    #[cfg(feature = "tblis")]
    "tblis",
];

fn die(msg: &str) -> ! {
    eprintln!("qcbench: {msg}");
    std::process::exit(2)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("help");

    if let Err(e) = knobs::init_from_env() {
        eprintln!("qcbench: {e}");
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
                "usage: qcbench <run|verify|info> [options]\n\
                 \n\
                 options:\n\
                 \x20 --threads <n>     host/provider thread budget (default 1)\n\
                 \x20 --n 1             timing-v2 identity sentinel; not an extent\n\
                 \x20 --reps <n>        timed repetitions per measurement (default 5)\n\
                 \x20 --prime-ms <n>    untimed time-based priming per arm (default 1500)\n\
                 \x20 --dtype f64       QC storage type (the only supported value)\n\
                 \x20 --case <substr>   only families whose name contains this\n\
                 \x20 --family <list>   exact family IDs, intersected with --case\n\
                 \x20 --batch 1         declared unit batch (the only supported value)\n\
                 \x20 --regime <label>  evidence regime (default qcbench)\n\
                 \x20 --engines <list>  plan,tblis, subject to build features\n\
                 \x20 --rows <list>     engine rows to run, in order\n\
                 \x20 --rotate-rows <k> rotate engine execution order (default 0)\n\
                 \x20 --statistic <s>   best|median of the timed samples (default best)\n\
                 \x20 --csv <path>      write machine-readable results\n\
                 \x20 --census-csv <p>  write the declared-cell census\n\
                 \x20 --stress none     fixed contiguous layouts (the only supported value)"
            );
            ExitCode::FAILURE
        }
    }
}

pub struct Options {
    pub size_mib: f64,
    pub n_spec: String,
    pub reps: usize,
    pub prime_ms: u64,
    pub dtypes: Vec<String>,
    pub case_filter: Option<String>,
    pub families: Vec<String>,
    pub engines: Vec<String>,
    pub rotate: usize,
    pub statistic: Statistic,
    pub regime: String,
    pub batch: String,
    pub csv: Option<String>,
    pub census_csv: Option<String>,
    pub stress: corpus::Stress,
}

impl Options {
    fn parse(args: &[String]) -> Options {
        let mut o = Options {
            size_mib: 0.0,
            n_spec: String::new(),
            reps: 5,
            prime_ms: 1500,
            dtypes: vec!["f64".into()],
            case_filter: None,
            families: Vec::new(),
            engines: ENGINE_ORDER.iter().map(|s| s.to_string()).collect(),
            rotate: 0,
            statistic: Statistic::Best,
            regime: "qcbench".into(),
            batch: "1".into(),
            csv: None,
            census_csv: None,
            stress: corpus::Stress::None,
        };
        let mut rows = Vec::new();
        let mut engines_explicit = false;
        let mut i = 0;
        while i < args.len() {
            let next = |i: usize| args.get(i + 1).cloned().unwrap_or_default();
            match args[i].as_str() {
                "--threads" => i += 1,
                "--n" => {
                    let v = next(i);
                    if v.split(',').any(|n| n.trim() != "1") {
                        die("--n is the identity sentinel 1 for this fixed-shape corpus");
                    }
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
                    o.dtypes = split_list(&next(i));
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
                        die("--batch must be 1 for this corpus");
                    }
                    i += 1;
                }
                "--regime" => {
                    o.regime = next(i);
                    i += 1;
                }
                "--engines" => {
                    engines_explicit = true;
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
                    let v = next(i);
                    o.stress = corpus::Stress::parse(&v).unwrap_or_else(|| {
                        die(&format!(
                            "--stress {v:?}: expected none for fixed QC layouts"
                        ))
                    });
                    i += 1;
                }
                "--size" => die("qcbench has fixed QC extents and does not accept --size"),
                other => die(&format!("unknown option {other}")),
            }
            i += 1;
        }
        if o.dtypes != ["f64"] {
            die("qcbench supports exactly --dtype f64");
        }
        for value in rows.iter().chain(o.engines.iter()) {
            if !crate::engines::run::ENGINE_ORDER.contains(&value.as_str())
                || !ENGINE_ORDER.contains(&value.as_str())
            {
                die(&format!(
                    "unknown or unavailable engine {value:?}; this binary has {} \
                     (`--features tblis` adds tblis)",
                    ENGINE_ORDER.join(",")
                ));
            }
        }
        if !rows.is_empty() {
            o.engines = if engines_explicit {
                rows.into_iter().filter(|e| o.engines.contains(e)).collect()
            } else {
                rows
            };
        }
        if o.engines.is_empty() {
            die("no engine rows selected");
        }
        o
    }

    pub fn wants(&self, dtype: &str) -> bool {
        self.dtypes.iter().any(|d| d == dtype)
    }

    pub fn engine(&self, name: &str) -> bool {
        self.engines.iter().any(|e| e == name)
    }

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
        self.size_mib
    }

    pub fn size_description(&self) -> String {
        "fixed QC extents at three occupied/virtual ratios (1/3, 1/10, 1/20)".to_string()
    }
}

fn split_list(v: &str) -> Vec<String> {
    v.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}
