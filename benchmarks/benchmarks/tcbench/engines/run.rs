//! The TCCG corpus run: every case, every requested dtype, every engine.
//!
//! Engines: `plan` (the planner's own choice under the knobs), `packed` (the
//! packed driver, forced), and the external baselines `ttgt` and `tblis`.

use std::process::ExitCode;

use num_complex::Complex;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tcontract::{Plan, PlanConfig};
use tcontract_exec::Exec;
use tcontract_kernel::Element;

use super::{gflops, problem_of, rel_error, timed, timed_prepare, BenchElem};
use crate::corpus::{self, Sized};
use crate::report::{Results, Row, Table};
#[cfg(feature = "blas")]
use crate::ttgt::{ttgt, TtgtPlan, TtgtScratch};
use crate::Options;

/// Column order for the report tables: this library's engines first, then the
/// external baselines.
pub const ENGINE_ORDER: &[&str] = &[
    "plan",
    "packed",
    #[cfg(feature = "blas")]
    "ttgt",
    // Restored: this arm was retired while the baseline could not name its own
    // revision. It can now - a release tag, its commit, the bundled BLIS revision and
    // an artifact hash, through `benchmarks/scripts/build_tblis.sh` - which is what a
    // recorded ratio against it needs.
    #[cfg(feature = "tblis")]
    "tblis",
];

pub fn run(opts: &Options, exec: &mut Exec<'_>) -> ExitCode {
    if opts.reps == 0 {
        eprintln!(
            "tcbench: --reps 0 leaves nothing to time and skips priming for every arm; \
             use `tcbench verify` for a correctness pass without timing"
        );
        return ExitCode::from(2);
    }
    crate::report::print_environment();
    println!();

    let mut results = Results::default();
    let cases: Vec<_> = corpus::corpus()
        .into_iter()
        .filter(|c| opts.case_selected(c.name))
        .collect();

    // The evidence identity every row carries, and the declared grid the census
    // enumerates. `--rotate-rows` moves execution order only, so the census
    // follows `exec_order` while the human tables keep the selection order. The
    // timed CSV is driven from the same grid, so a skipped cell appears in both.
    results.regime = opts.regime.clone();
    results.batch = opts.batch.clone();
    results.statistic = opts.statistic;
    results.n_spec = opts.n_spec.clone();
    let mut census = (opts.csv.is_some() || opts.census_csv.is_some()).then(|| {
        crate::report::Census::declared(
            &opts.regime,
            &opts.n_spec,
            &opts.batch,
            exec.budget(),
            &cases,
            &opts.dtypes,
            &opts.exec_order(),
        )
    });

    println!(
        "running {} cases at {} MiB nominal tensor size, {} reps, {} ms priming, stress={}\n",
        cases.len(),
        opts.size_mib,
        opts.reps,
        opts.prime_ms,
        opts.stress.name()
    );

    for case in &cases {
        let s = corpus::size_case_stressed(case, opts.tensor_bytes(), opts.stress);
        if opts.wants("f32") {
            run_case::<f32>(&s, opts, exec, &mut results, census.as_mut());
        }
        if opts.wants("f64") {
            run_case::<f64>(&s, opts, exec, &mut results, census.as_mut());
        }
        if opts.wants("c32") {
            run_case::<Complex<f32>>(&s, opts, exec, &mut results, census.as_mut());
        }
        if opts.wants("c64") {
            run_case::<Complex<f64>>(&s, opts, exec, &mut results, census.as_mut());
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

/// Measure one case for one element type across all requested engines, in the
/// order [`Options::exec_order`] gives (the declared rows rotated per process
/// start).
pub fn run_case<T>(
    s: &Sized,
    opts: &Options,
    exec: &mut Exec<'_>,
    results: &mut Results,
    mut census: Option<&mut crate::report::Census>,
) where
    T: BenchElem,
{
    let order = opts.exec_order();
    let problem = match problem_of::<T>(s) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{} [{}]: description failed: {e}", s.case.name, T::NAME);
            if let Some(c) = census.as_deref_mut() {
                for name in &order {
                    c.mark_skipped(s.case.name, T::NAME, name, "description failed");
                }
            }
            return;
        }
    };
    let knobs = crate::knobs::get();
    let plan_for = |config: &PlanConfig| match Plan::<T>::new(&problem, config) {
        Ok(p) => Some(p),
        Err(e) => {
            eprintln!("{} [{}]: planning failed: {e}", s.case.name, T::NAME);
            None
        }
    };

    let (m, n, k) = s.mnk();
    let macs = s.macs();

    let mut rng = ChaCha8Rng::seed_from_u64(0x5EED);
    let a: Vec<T> = (0..s.elems_a()).map(|_| T::sample(&mut rng)).collect();
    let b: Vec<T> = (0..s.elems_b()).map(|_| T::sample(&mut rng)).collect();
    let mut d: Vec<T> = vec![<T as Element>::zero(); s.elems_c()];

    let mut reference: Option<Vec<T>> = None;
    // The first arm measured for a case becomes the reference the rest are compared
    // against, so a case measured by one arm alone has nothing checking its output;
    // this counts the comparisons that actually happened.
    let mut comparisons = 0usize;
    let mut check = |name: &str, d: &Vec<T>, reference: &mut Option<Vec<T>>| -> String {
        match reference {
            None => {
                *reference = Some(d.clone());
                String::new()
            }
            Some(r) => {
                comparisons += 1;
                let err = rel_error(d, r);
                let tol = if core::mem::size_of::<T::Real>() == 4 {
                    1e-3
                } else {
                    1e-10
                };
                if !err.is_finite() || err > tol {
                    format!("MISMATCH({name}) rel_err={err:.2e}")
                } else {
                    String::new()
                }
            }
        }
    };

    // The budget is fixed for the run; read it once so the row closure below
    // does not hold the (mutably lent) context.
    let threads = exec.budget();
    let push = |engine: &str,
                samples: Vec<f64>,
                spread: f64,
                prepare: f64,
                reg_a: f64,
                reg_b: f64,
                notes: String,
                results: &mut Results,
                census: Option<&mut crate::report::Census>| {
        // The arm's reported time is the requested statistic of its samples, so
        // the evidence's `total_ms` and the human table always agree.
        let secs = opts.statistic.of(&samples);
        results.push(Row {
            case: s.case.name.to_string(),
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
            reg_a,
            reg_b,
            notes,
            samples,
        });
        if let Some(c) = census {
            c.mark_ok(s.case.name, T::NAME, engine);
        }
    };

    // ---- every engine, in the requested execution order ---------------------
    //
    // Regularity is reported against the orientation the packed engine actually
    // executes in, which is not necessarily `A`-rows / `B`-columns; a strategy
    // that does not use the packed driver reports no regularity (0, 0).
    let mut regularity = (0.0f64, 0.0f64);
    // BenchThreads owns the warm arena/pool outside the timed region.
    for name in &order {
        if !opts.engine(name) {
            continue;
        }
        match name.as_str() {
            // ---- this library: the planner's choice, and the packed driver ---
            "plan" | "packed" => {
                let config = if name == "packed" {
                    knobs.packed_config()
                } else {
                    knobs.config()
                };
                // The plan is the work this arm does before the timed call, and the
                // policy that excludes it is why the harness builds one at all;
                // recording it keeps the exclusion legible, since the reference has
                // no equivalent to hoist.
                let (p, prepare) = timed_prepare(3, || plan_for(&config));
                let Some(p) = p else {
                    if let Some(c) = census.as_deref_mut() {
                        c.mark_skipped(s.case.name, T::NAME, name, "planning failed");
                    }
                    continue;
                };
                let report = p.report();
                let (reg_a, reg_b) = report
                    .packed
                    .as_ref()
                    .map_or((0.0, 0.0), |r| (r.regular_rows, r.regular_cols));
                if report.packed.is_some() && regularity == (0.0, 0.0) {
                    regularity = (reg_a, reg_b);
                }
                let (_, spread, samples) = timed(opts.reps, opts.prime_ms, || {
                    // SAFETY: the buffers are sized by the layouts, `D` is exclusive
                    // and `beta = 0` reads no previous value.
                    unsafe {
                        p.execute_raw(
                            exec,
                            <T as Element>::one(),
                            a.as_ptr(),
                            b.as_ptr(),
                            <T as Element>::zero(),
                            std::ptr::null(),
                            d.as_mut_ptr(),
                        )
                    }
                    .expect("a validated plan runs")
                });
                // `MR x NR`, the orientation and the blocking go in the notes because
                // they are not constants per dtype: a CSV without them cannot be
                // re-read later.
                let notes = match &report.packed {
                    Some(r) => format!(
                        "{} {}x{} {} {}x{}x{} {} {}",
                        r.family_id,
                        r.mr,
                        r.nr,
                        if r.swapped { "BA" } else { "AB" },
                        r.mc,
                        r.kc,
                        r.nc,
                        tcontract_bench::partition::describe(r),
                        check(name, &d, &mut reference)
                    ),
                    None => format!(
                        "{} {}",
                        report.algorithm.name(),
                        check(name, &d, &mut reference)
                    ),
                }
                .trim()
                .to_string();
                push(
                    name,
                    samples,
                    spread,
                    prepare,
                    reg_a,
                    reg_b,
                    notes,
                    results,
                    census.as_deref_mut(),
                );
            }

            // ---- TTGT ----------------------------------------------------
            #[cfg(feature = "blas")]
            "ttgt" => {
                let ((tp, mut scratch), prepare) = timed_prepare(3, || {
                    let tp = TtgtPlan::new(&problem);
                    let scratch = TtgtScratch::<T>::new(&tp);
                    (tp, scratch)
                });
                let mut dt: Vec<T> = vec![<T as Element>::zero(); s.elems_c()];
                let (_, spread, samples) = timed(opts.reps, opts.prime_ms, || {
                    ttgt(
                        &tp,
                        <T as Element>::one(),
                        &a,
                        &b,
                        <T as Element>::zero(),
                        &dt.clone(),
                        &mut dt,
                        &mut scratch,
                    )
                });
                let notes = check("ttgt", &dt, &mut reference);
                push(
                    "ttgt",
                    samples,
                    spread,
                    prepare,
                    regularity.0,
                    regularity.1,
                    notes,
                    results,
                    census.as_deref_mut(),
                );
            }

            // ---- TBLIS ---------------------------------------------------
            #[cfg(feature = "tblis")]
            "tblis" => {
                use crate::tblis as tb;
                let ((mut oa, mut ob, mut oc), prepare) = timed_prepare(3, || {
                    (
                        tb::Operand::new(s.la.extents(), s.la.strides(), s.case.a),
                        tb::Operand::new(s.lb.extents(), s.lb.strides(), s.case.b),
                        tb::Operand::new(s.lc.extents(), s.lc.strides(), s.case.c),
                    )
                });
                let mut dt: Vec<T> = vec![<T as Element>::zero(); s.elems_c()];
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
                let (_, spread, samples) = timed(opts.reps, opts.prime_ms, || {
                    let mut tc = oc.tensor(
                        T::TBLIS_TYPE,
                        T::tblis_scalar(0.0),
                        dt.as_mut_ptr() as *mut std::ffi::c_void,
                    );
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
                let notes = check("tblis", &dt, &mut reference);
                push(
                    "tblis",
                    samples,
                    spread,
                    prepare,
                    regularity.0,
                    regularity.1,
                    notes,
                    results,
                    census.as_deref_mut(),
                );
            }
            _ => {}
        }
    }

    if comparisons == 0 {
        eprintln!(
            "warning: {} {}: one arm was measured, so nothing cross-checked its output              (the first arm measured is the reference for the others)",
            s.case.name,
            T::NAME
        );
    }
}

fn print_tables(results: &Results, opts: &Options) {
    // The reported row order is the selection order (`--rows`/`--engines`);
    // `--rotate-rows` moves execution order only, not what is printed.
    let engines: Vec<&str> = opts
        .engines
        .iter()
        .map(String::as_str)
        .filter(|e| results.rows.iter().any(|r| &r.engine == e))
        .collect();

    for dtype in ["f32", "f64", "c32", "c64"] {
        if !opts.wants(dtype) {
            continue;
        }
        let mut headers = vec![
            "case".to_string(),
            "m".into(),
            "n".into(),
            "k".into(),
            "regA".into(),
            "regB".into(),
        ];
        for e in &engines {
            headers.push(format!("{e} GF/s"));
        }
        let hrefs: Vec<&str> = headers.iter().map(|s| s.as_str()).collect();
        let mut t = Table::new(&hrefs);
        let mut any = false;
        for case in crate::corpus::corpus() {
            let Some(first) = results.get(case.name, dtype, engines.first().copied().unwrap_or(""))
            else {
                continue;
            };
            any = true;
            let mut cells = vec![
                case.name.to_string(),
                first.m.to_string(),
                first.n.to_string(),
                first.k.to_string(),
                format!("{:.2}", first.reg_a),
                format!("{:.2}", first.reg_b),
            ];
            for e in &engines {
                cells.push(
                    results
                        .get(case.name, dtype, e)
                        .map(|r| format!("{:.1}", r.gflops))
                        .unwrap_or_else(|| "-".into()),
                );
            }
            t.row(cells);
        }
        if any {
            println!("\n=== {dtype} ===");
            t.print();
        }
    }

    print_ratio_table(results, opts, "f64", "c64");
    print_ratio_table(results, opts, "f32", "c32");
}

/// The headline metric: complex GFLOP/s divided by real GFLOP/s for the same
/// contraction. A value of 1.0 means the engine extracts the same fraction of
/// the machine's FMA throughput from complex data as from real data; anything
/// below that is the complex penalty.
fn print_ratio_table(results: &Results, opts: &Options, real: &str, cplx: &str) {
    let engines: Vec<&str> = opts
        .engines
        .iter()
        .map(String::as_str)
        .filter(|e| results.rows.iter().any(|r| &r.engine == e))
        .collect();
    if engines.is_empty() {
        return;
    }
    let mut headers = vec!["case".to_string()];
    for e in &engines {
        headers.push(format!("{e} {cplx}/{real}"));
    }
    let hrefs: Vec<&str> = headers.iter().map(|s| s.as_str()).collect();
    let mut t = Table::new(&hrefs);
    let mut sums = vec![(0.0f64, 0usize); engines.len()];
    let mut any = false;
    for case in crate::corpus::corpus() {
        let mut cells = vec![case.name.to_string()];
        let mut has = false;
        for (i, e) in engines.iter().enumerate() {
            let r = results.get(case.name, real, e);
            let c = results.get(case.name, cplx, e);
            match (r, c) {
                (Some(r), Some(c)) if r.gflops > 0.0 => {
                    let ratio = c.gflops / r.gflops;
                    sums[i].0 += ratio;
                    sums[i].1 += 1;
                    cells.push(format!("{ratio:.3}"));
                    has = true;
                }
                _ => cells.push("-".into()),
            }
        }
        if has {
            any = true;
            t.row(cells);
        }
    }
    if !any {
        return;
    }
    let mut mean = vec!["MEAN".to_string()];
    for (s, n) in &sums {
        mean.push(if *n > 0 {
            format!("{:.3}", s / *n as f64)
        } else {
            "-".into()
        });
    }
    t.row(mean);
    println!("\n=== complex efficiency ratio ({cplx} GF/s over {real} GF/s; 1.0 = no penalty) ===");
    t.print();
}
