//! Cross-implementation verification of the #61 corpus.
//!
//! Every arm runs one untimed pass over every case and is compared with the
//! naive label-loop reference of `experiments/three-engine-contract`; the
//! campaign's runner refuses to publish a cell whose harness did not report
//! `all comparisons within tolerance` here. `run` times these same arms, so
//! this is the correctness gate on the numbers it produces.
//!
//! `verify` also writes the declared-cell census (to `--csv`, which is where the
//! recorder reads it, and to `--census-csv`), so the toolkit can tell a skipped
//! cell from a missing one.

use std::process::ExitCode;

use num_complex::Complex;
use tcontract::PlanConfig;
use tcontract_exec::Exec;

use super::run::{self, ENGINE_ORDER};
use super::{max_rel_err, sample_inputs, tol, BenchElem};
use crate::corpus::{self, Program};
use crate::report::{Census, Table};
use crate::Options;

pub fn run(opts: &Options, exec: &mut Exec<'_>) -> ExitCode {
    crate::report::print_environment();
    let cases = crate::select_cases(opts);
    // Only a case whose own dtype was requested is checked; the rest are still
    // declared, with every cell skipped, because the toolkit's grid multiplies
    // each family by the suite's whole dtype list.
    let checked = cases.iter().filter(|p| opts.wants(p.dtype)).count();
    println!(
        "\nverifying {} cases, one untimed pass per arm (--size is accepted and ignored; \
         --reps and --prime-ms do not apply)\n",
        checked
    );
    // The census covers `verify` as well as `run`: the same declared grid, in
    // the run's execution order. The toolkit's verify step passes `--csv`, so
    // that is where its census goes; `--census-csv` works too.
    let mut census = (opts.csv.is_some() || opts.census_csv.is_some()).then(|| {
        Census::declared(
            &opts.regime,
            &opts.n_spec,
            &opts.batch,
            exec.budget(),
            &cases,
            &opts.dtypes,
            &opts.exec_order(),
        )
    });
    let mut t = Table::new(&[
        "case", "dtype", "steps", "m", "n", "k", "engine", "rel_err", "status",
    ]);
    let mut failures = 0usize;
    for p in &cases {
        if !opts.wants(p.dtype) {
            continue;
        }
        match p.dtype {
            "f64" => failures += check::<f64>(p, opts, exec, &mut t, census.as_mut()),
            "c64" => failures += check::<Complex<f64>>(p, opts, exec, &mut t, census.as_mut()),
            other => eprintln!("{}: unknown dtype {other}", p.name),
        }
    }
    t.print();
    for path in [opts.csv.as_ref(), opts.census_csv.as_ref()]
        .into_iter()
        .flatten()
    {
        if let Some(c) = census.as_ref() {
            match c.write_csv(path) {
                Ok(()) => println!("wrote {path}"),
                Err(e) => eprintln!("failed to write {path}: {e}"),
            }
        }
    }
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
fn check<T: BenchElem>(
    p: &Program,
    opts: &Options,
    exec: &mut Exec<'_>,
    t: &mut Table,
    mut census: Option<&mut Census>,
) -> usize {
    let inputs = sample_inputs::<T>(p, 0xA11CE);
    let reference = corpus::naive_program::<T>(p, &inputs);
    let (m, n, k) = p.report_mnk();
    let mut fails = 0usize;

    // `reps = 0, prime_ms = 0` is exactly one untimed pass through the same
    // builder `run` times. The table keeps the declared engine order; the
    // census was declared in the rotated execution order and a mark addresses a
    // cell by name, so the two orders do not have to agree here.
    for &engine in ENGINE_ORDER {
        if !opts.engine(engine) {
            continue;
        }
        let mut slots = p.new_slots::<T>();
        let result: Result<f64, String> = match engine {
            "plan" => run::run_plans(p, &PlanConfig::default(), &inputs, &mut slots, exec, 0, 0)
                .map(|_| max_rel_err(slots.last().unwrap(), &reference))
                .map_err(|e| format!("PLAN ERROR: {e}")),
            "packed" => run::run_plans(p, &PlanConfig::packed(), &inputs, &mut slots, exec, 0, 0)
                .map(|_| max_rel_err(slots.last().unwrap(), &reference))
                .map_err(|e| format!("PLAN ERROR: {e}")),
            #[cfg(feature = "tblis")]
            "tblis" => {
                run::run_tblis(p, &inputs, &mut slots, 0, 0);
                Ok(max_rel_err(slots.last().unwrap(), &reference))
            }
            _ => continue,
        };
        let (cell, status, bad) = match &result {
            Ok(e) => {
                let bad = !e.is_finite() || *e > tol::<T>();
                (format!("{e:.2e}"), if bad { "FAIL" } else { "ok" }, bad)
            }
            Err(msg) => (msg.clone(), "FAIL", true),
        };
        fails += usize::from(bad);
        // A cell that was built and checked is `ok` (a FAIL is still measured;
        // the exit status carries the verdict); a cell the check could not
        // build stays `skipped` with the reason.
        if let Some(c) = census.as_deref_mut() {
            match &result {
                Ok(_) => c.mark_ok(&p.name, T::NAME, engine),
                Err(_) => c.mark_skipped(&p.name, T::NAME, engine, "planning failed"),
            }
        }
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
    }
    fails
}
