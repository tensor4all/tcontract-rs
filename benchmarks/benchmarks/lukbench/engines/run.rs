//! The #61 corpus run: every case, every requested engine, one dtype per case.
//!
//! Engines: `plan` (the planner's own choice), `packed` (the packed driver,
//! forced) and the external baseline `tblis`. Every arm runs a
//! prebuilt plan into a preallocated output, and one timed call is the whole
//! program.

use std::process::ExitCode;

use num_complex::Complex;
use strided_view::{StridedView, StridedViewMut};
use tcontract::{Plan, PlanConfig};
use tcontract_exec::Exec;
use tcontract_kernel::Element;

use super::{
    gflops, max_rel_err, problem_of, sample_inputs, steps_with, timed, timed_prepare, tol,
    BenchElem,
};
use crate::corpus::{self, col_major_strides, Program};
use crate::report::{Census, Results, Row, Table};
use crate::Options;

/// Column order for the report tables: this library's engines first, then the
/// external baselines.
pub const ENGINE_ORDER: &[&str] = &[
    "plan",
    "packed",
    #[cfg(feature = "tblis")]
    "tblis",
];

pub fn run(opts: &Options, exec: &mut Exec<'_>) -> ExitCode {
    if opts.reps == 0 {
        eprintln!(
            "lukbench: --reps 0 leaves nothing to time and skips priming for every arm; \
             use `lukbench verify` for a correctness pass without timing"
        );
        return ExitCode::from(2);
    }
    crate::report::print_environment();
    println!();

    let cases = crate::select_cases(opts);
    // Only a case whose own dtype was requested is timed; the rest are still
    // declared, with every cell skipped, because the toolkit's grid multiplies
    // each family by the suite's whole dtype list.
    let timed_cases = cases.iter().filter(|p| opts.wants(p.dtype)).count();
    println!(
        "running {} cases at fixed extents (--size is accepted and ignored: this corpus has \
         no size knob), {} reps, {} ms priming, engines={}\n",
        timed_cases,
        opts.reps,
        opts.prime_ms,
        opts.engines.join(",")
    );

    let mut results = Results {
        regime: opts.regime.clone(),
        batch: opts.batch.clone(),
        statistic: opts.statistic,
        n_spec: opts.n_spec.clone(),
        ..Default::default()
    };
    // The declared grid the census enumerates. `--rotate-rows` moves execution
    // order only, so the census follows `exec_order` while the human tables keep
    // the selection order. The timed CSV is driven from the same grid, so a
    // skipped cell appears in both.
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

    for p in &cases {
        if !opts.wants(p.dtype) {
            continue;
        }
        match p.dtype {
            "f64" => run_case::<f64>(p, opts, exec, &mut results, census.as_mut()),
            "c64" => run_case::<Complex<f64>>(p, opts, exec, &mut results, census.as_mut()),
            other => eprintln!("{}: unknown dtype {other}", p.name),
        }
    }

    print_tables(&results, opts);
    if let Some(path) = &opts.csv {
        let census = census
            .as_ref()
            .expect("a timed CSV needs the declared grid");
        match results.write_csv(path, census) {
            Ok(()) => println!("\nwrote {path}"),
            Err(e) => eprintln!("failed to write {path}: {e}"),
        }
    }
    if let (Some(c), Some(path)) = (census.as_ref(), opts.census_csv.as_ref()) {
        match c.write_csv(path) {
            Ok(()) => println!("wrote {path}"),
            Err(e) => eprintln!("failed to write {path}: {e}"),
        }
    }
    if results.rows.iter().any(|r| r.notes.contains("MISMATCH")) {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

/// Measure one case across the requested engines, in the order
/// [`Options::exec_order`] gives (the declared rows rotated per process start).
fn run_case<T: BenchElem>(
    p: &Program,
    opts: &Options,
    exec: &mut Exec<'_>,
    results: &mut Results,
    mut census: Option<&mut Census>,
) {
    let inputs = sample_inputs::<T>(p, 0x5EED);
    // The naive label loop is the only oracle for a program: the corpus has no
    // analytically known output, and `verify` runs the same comparison before
    // any timing.
    let reference = corpus::naive_program::<T>(p, &inputs);
    let (m, n, k) = p.report_mnk();
    let shape = p.shape_note();
    let macs = p.macs;

    // The max relative error against the naive reference goes into every row,
    // as `tcbench`'s check lines do; `MISMATCH` in a note fails the run.
    let check = |engine: &str, d: &[T]| -> String {
        let err = max_rel_err(d, &reference);
        let flag = if !err.is_finite() || err > tol::<T>() {
            format!("MISMATCH({engine}) ")
        } else {
            String::new()
        };
        format!("{flag}rel_err={err:.2e}")
    };

    // The budget is fixed for the run; read it once so the row closure below
    // does not hold the (mutably lent) context.
    let threads = exec.budget();
    let push = |engine: &str,
                samples: Vec<f64>,
                spread: f64,
                prepare: f64,
                notes: String,
                results: &mut Results,
                census: Option<&mut Census>| {
        // The arm's reported time is the requested statistic of its samples, so
        // the evidence's `total_ms` and the human table always agree.
        let secs = opts.statistic.of(&samples);
        results.push(Row {
            case: p.name.clone(),
            dtype: T::NAME.to_string(),
            engine: engine.to_string(),
            threads,
            m,
            n,
            k,
            macs,
            secs,
            spread,
            prepare_s: prepare,
            gflops: gflops::<T>(macs, secs),
            reg_a: 0.0,
            reg_b: 0.0,
            notes: format!("{notes} {shape}").trim().to_string(),
            samples,
        });
        if let Some(c) = census {
            c.mark_ok(&p.name, T::NAME, engine);
        }
    };

    // ---- every engine, in the requested execution order ---------------------
    for name in &opts.exec_order() {
        if !opts.engine(name) {
            continue;
        }
        match name.as_str() {
            // ---- this library: the planner's choice, and the packed driver ---
            "plan" | "packed" => {
                let config = if name == "packed" {
                    PlanConfig::packed()
                } else {
                    PlanConfig::default()
                };
                let mut slots = p.new_slots::<T>();
                match run_plans(
                    p,
                    &config,
                    &inputs,
                    &mut slots,
                    exec,
                    opts.reps,
                    opts.prime_ms,
                ) {
                    Ok((samples, scatter, prepare, strategy)) => {
                        let notes = format!("{strategy} {}", check(name, slots.last().unwrap()));
                        push(
                            name,
                            samples,
                            scatter,
                            prepare,
                            notes,
                            results,
                            census.as_deref_mut(),
                        );
                    }
                    Err(e) => {
                        eprintln!("{} [{}]: planning failed: {e}", p.name, T::NAME);
                        if let Some(c) = census.as_deref_mut() {
                            c.mark_skipped(&p.name, T::NAME, name, "planning failed");
                        }
                    }
                }
            }

            // ---- TBLIS ------------------------------------------------------
            #[cfg(feature = "tblis")]
            "tblis" => {
                let mut slots = p.new_slots::<T>();
                let (samples, scatter, prepare) =
                    run_tblis(p, &inputs, &mut slots, opts.reps, opts.prime_ms);
                let notes = check("tblis", slots.last().unwrap());
                push(
                    "tblis",
                    samples,
                    scatter,
                    prepare,
                    notes,
                    results,
                    census.as_deref_mut(),
                );
            }
            _ => {}
        }
    }
}

/// Build a plan per step, run the whole program into `slots` `reps` times
/// after `prime_ms` of priming, and return the chronological samples, the
/// scatter across the repetitions, the prepare time, and the per-step strategy
/// description.
///
/// `reps == 0` with `prime_ms == 0` is one untimed pass, which is what `verify`
/// wants: it then exercises exactly the code path `run` times.
pub fn run_plans<T: BenchElem>(
    p: &Program,
    config: &PlanConfig,
    inputs: &[Vec<T>],
    slots: &mut [Vec<T>],
    exec: &mut Exec<'_>,
    reps: usize,
    prime_ms: u64,
) -> Result<(Vec<f64>, f64, f64, String), tcontract::Error> {
    let (plans, prepare) = timed_prepare(3, || -> Result<Vec<Plan<T>>, tcontract::Error> {
        (0..p.steps.len())
            .map(|k| Plan::<T>::new(&problem_of::<T>(p, k)?, config))
            .collect::<Result<_, _>>()
    });
    let plans = plans?;
    let strategy = strategies(&plans);
    let strides: Vec<Vec<isize>> = p.shapes.iter().map(|d| col_major_strides(d)).collect();
    let n_in = p.n_in();
    let (_, scatter, samples) = timed(reps, prime_ms, || {
        steps_with(p, inputs, slots, |k, st, a, b, d| {
            let av = StridedView::new(a, &p.shapes[st.lhs], &strides[st.lhs], 0).expect("view");
            let bv = StridedView::new(b, &p.shapes[st.rhs], &strides[st.rhs], 0).expect("view");
            let mut dv =
                StridedViewMut::new(d, &p.shapes[n_in + k], &strides[n_in + k], 0).expect("view");
            // `beta = 0`, so no previous value in `D` is read.
            plans[k]
                .execute_into(exec, <T as Element>::one(), &av, &bv, &mut dv)
                .expect("a validated plan runs");
        });
    });
    Ok((samples, scatter, prepare, strategy))
}

/// The distinct per-step strategy descriptions of a program's plans.
fn strategies<T: BenchElem>(plans: &[Plan<T>]) -> String {
    let mut seen: Vec<String> = Vec::new();
    for pl in plans {
        let r = pl.report();
        let s = match &r.packed {
            Some(x) => format!(
                "{} {}x{} {} {}x{}x{} {}",
                x.family_id,
                x.mr,
                x.nr,
                if x.swapped { "BA" } else { "AB" },
                x.mc,
                x.kc,
                x.nc,
                tcontract_bench::partition::describe(x)
            ),
            None => r.algorithm.name().to_string(),
        };
        if !seen.contains(&s) {
            seen.push(s);
        }
    }
    seen.join(" | ")
}

/// The TBLIS baseline: one `tblis_tensor_mult` per step into preallocated slots.
///
/// TBLIS matches an index by the label *character* of the call, so the labels
/// come from the step's eigenequation while the extents and strides come from
/// the slot; the `Operand` triples are built outside the timed region and only
/// the four-field tensor headers are rebuilt per call, as `tcbench` did.
/// Thread count and the 1.3/2.x type-tag self-check are set up once by
/// `configure_threads`.
#[cfg(feature = "tblis")]
pub fn run_tblis<T: BenchElem>(
    p: &Program,
    inputs: &[Vec<T>],
    slots: &mut [Vec<T>],
    reps: usize,
    prime_ms: u64,
) -> (Vec<f64>, f64, f64) {
    use crate::tblis as tb;
    let n_in = p.n_in();
    let (mut ops, prepare) = timed_prepare(3, || {
        let ops: Vec<(tb::Operand, tb::Operand, tb::Operand)> = p
            .steps
            .iter()
            .enumerate()
            .map(|(k, st)| {
                let mk = |slot: usize, labels: &[char]| {
                    let (extents, strides) = corpus::dims_strides_i64(&p.shapes[slot]);
                    tb::Operand::new(&extents, &strides, &labels.iter().collect::<String>())
                };
                (mk(st.lhs, &st.a), mk(st.rhs, &st.b), mk(n_in + k, &st.d))
            })
            .collect();
        ops
    });
    let (_, scatter, samples) = timed(reps, prime_ms, || {
        steps_with(p, inputs, slots, |k, _st, a, b, d| {
            let (oa, ob, oc) = &mut ops[k];
            let ta = oa.tensor(
                T::TBLIS_TYPE,
                T::tblis_scalar(1.0),
                a.as_ptr() as *mut std::ffi::c_void,
            );
            let tbv = ob.tensor(
                T::TBLIS_TYPE,
                T::tblis_scalar(1.0),
                b.as_ptr() as *mut std::ffi::c_void,
            );
            let mut tc = oc.tensor(
                T::TBLIS_TYPE,
                T::tblis_scalar(0.0),
                d.as_mut_ptr() as *mut std::ffi::c_void,
            );
            // SAFETY: the operands' `len`/`stride` describe the buffers they
            // point at, `d` is exclusive, beta = 0 leaves C unread. The null
            // communicator and context select TBLIS's default single-node path.
            unsafe {
                tb::tblis_tensor_mult(
                    std::ptr::null(),
                    std::ptr::null(),
                    &ta,
                    oa.labels(),
                    &tbv,
                    ob.labels(),
                    &mut tc,
                    oc.labels(),
                )
            }
        });
    });
    (samples, scatter, prepare)
}

fn print_tables(results: &Results, opts: &Options) {
    if results.rows.is_empty() {
        println!("\nno rows: no case matched, or no requested engine is built into this binary");
        return;
    }
    let engines: Vec<&str> = ENGINE_ORDER
        .iter()
        .copied()
        .filter(|e| results.rows.iter().any(|r| &r.engine == e))
        .collect();

    let mut headers = vec![
        "case".to_string(),
        "dtype".into(),
        "steps".into(),
        "macs".into(),
    ];
    for e in &engines {
        headers.push(format!("{e} GF/s"));
    }
    let hrefs: Vec<&str> = headers.iter().map(|s| s.as_str()).collect();
    let mut t = Table::new(&hrefs);
    for p in corpus::corpus() {
        let Some(first) = results.get(&p.name, p.dtype, engines.first().copied().unwrap_or(""))
        else {
            continue;
        };
        let mut cells = vec![
            p.name.clone(),
            p.dtype.to_string(),
            p.steps.len().to_string(),
            first.macs.to_string(),
        ];
        for e in &engines {
            cells.push(
                results
                    .get(&p.name, p.dtype, e)
                    .map(|r| format!("{:.1}", r.gflops))
                    .unwrap_or_else(|| "-".into()),
            );
        }
        t.row(cells);
    }
    println!(
        "\n=== GFLOP/s, {} of {} calls after {} ms priming, whole program per call ===",
        opts.statistic.name(),
        opts.reps,
        opts.prime_ms
    );
    t.print();

    // The per-step strategy and the check that every row carries: on the
    // console because a MISMATCH must be visible in the run's own output, not
    // only in a CSV nobody opened.
    println!("\n=== notes ===");
    for r in &results.rows {
        println!(
            "{:>24} {:>4} {:<8} {:>10.4} ms  {}",
            r.case,
            r.dtype,
            r.engine,
            r.secs * 1e3,
            r.notes
        );
    }
}
