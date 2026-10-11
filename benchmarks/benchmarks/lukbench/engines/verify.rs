//! Cross-implementation verification of the #61 corpus.
//!
//! Every arm runs one untimed pass over every case and is compared with the
//! naive label-loop reference of `experiments/three-engine-contract`; the
//! campaign's runner refuses to publish a cell whose harness did not report
//! `all comparisons within tolerance` here. `run` times these same arms, so
//! this is the correctness gate on the numbers it produces.

use std::process::ExitCode;

use num_complex::Complex;
use tcontract::PlanConfig;
use tcontract_exec::Exec;

use super::run::{self, ENGINE_ORDER};
use super::{max_rel_err, sample_inputs, tol, BenchElem};
use crate::corpus::{self, Program};
use crate::report::Table;
use crate::Options;

pub fn run(opts: &Options, exec: &Exec<'_>) -> ExitCode {
    crate::report::print_environment();
    let cases = crate::select_cases(opts);
    println!(
        "\nverifying {} cases, one untimed pass per arm (--size is accepted and ignored; \
         --reps and --prime-ms do not apply)\n",
        cases.len()
    );
    let mut t = Table::new(&[
        "case", "dtype", "steps", "m", "n", "k", "engine", "rel_err", "status",
    ]);
    let mut failures = 0usize;
    for p in &cases {
        match p.dtype {
            "f64" => failures += check::<f64>(p, opts, exec, &mut t),
            "c64" => failures += check::<Complex<f64>>(p, opts, exec, &mut t),
            other => eprintln!("{}: unknown dtype {other}", p.name),
        }
    }
    t.print();
    if failures == 0 {
        println!("\nall comparisons within tolerance");
        ExitCode::SUCCESS
    } else {
        println!("\n{failures} comparison(s) FAILED");
        ExitCode::FAILURE
    }
}

/// One row per (case, engine): the max relative error against the naive
/// reference. Returns the number of failed engines.
fn check<T: BenchElem>(p: &Program, opts: &Options, exec: &Exec<'_>, t: &mut Table) -> usize {
    let inputs = sample_inputs::<T>(p, 0xA11CE);
    let reference = corpus::naive_program::<T>(p, &inputs);
    let (m, n, k) = p.report_mnk();
    let mut fails = 0usize;

    let mut emit = |engine: &str, err: Result<f64, String>| {
        let (cell, status, bad) = match err {
            Ok(e) => {
                let bad = !e.is_finite() || e > tol::<T>();
                (format!("{e:.2e}"), if bad { "FAIL" } else { "ok" }, bad)
            }
            Err(msg) => (msg, "FAIL", true),
        };
        fails += usize::from(bad);
        t.row(vec![
            p.name.clone(),
            T::NAME.into(),
            p.steps.len().to_string(),
            m.to_string(),
            n.to_string(),
            k.to_string(),
            engine.into(),
            cell,
            status.into(),
        ]);
    };

    // `reps = 0, prime_ms = 0` is exactly one untimed pass through the same
    // builder `run` times.
    for &engine in ENGINE_ORDER {
        if !opts.engine(engine) {
            continue;
        }
        let mut slots = p.new_slots::<T>();
        match engine {
            "plan" => {
                match run::run_plans(p, &PlanConfig::default(), &inputs, &mut slots, exec, 0, 0) {
                    Ok(_) => emit(engine, Ok(max_rel_err(slots.last().unwrap(), &reference))),
                    Err(e) => emit(engine, Err(format!("PLAN ERROR: {e}"))),
                }
            }
            "packed" => {
                match run::run_plans(p, &PlanConfig::packed(), &inputs, &mut slots, exec, 0, 0) {
                    Ok(_) => emit(engine, Ok(max_rel_err(slots.last().unwrap(), &reference))),
                    Err(e) => emit(engine, Err(format!("PLAN ERROR: {e}"))),
                }
            }
            #[cfg(feature = "tblis")]
            "tblis" => {
                run::run_tblis(p, &inputs, &mut slots, 0, 0);
                emit(engine, Ok(max_rel_err(slots.last().unwrap(), &reference)));
            }
            _ => {}
        }
    }
    fails
}
