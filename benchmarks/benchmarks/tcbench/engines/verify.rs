//! Cross-implementation verification at realistic sizes.
//!
//! The exhaustive semantic testing (repeated indices, reductions, batch
//! indices, conjugation, negative strides, every loop-nest edge case) lives in
//! `crates/tcontract` and runs against a brute-force oracle. This
//! subcommand does the complementary job: run the *whole TCCG corpus* at
//! benchmark sizes and confirm that the planner's choice, the packed driver,
//! TTGT and TBLIS all agree. That is what catches blocking and packing bugs that only appear once
//! a problem is bigger than one cache block.

use std::process::ExitCode;

use num_complex::Complex;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tcontract::{Plan, PlanConfig};
use tcontract_exec::Exec;
use tcontract_kernel::Element;

#[cfg(any(feature = "blas", feature = "tblis"))]
use super::rel_error;
use super::BenchElem;
use crate::corpus::{self, Sized};
use crate::report::Table;
use crate::Options;

pub fn run(opts: &Options, exec: &Exec<'_>) -> ExitCode {
    crate::report::print_environment();

    let cases: Vec<_> = corpus::corpus()
        .into_iter()
        .filter(|c| opts.case_selected(c.name))
        .collect();

    // The census covers `verify` as well as `run`: the same declared grid, in
    // the run's execution order. A cell the check reached is `ok`; a cell it
    // could not build is `skipped` with the reason. The toolkit's verify step
    // passes `--csv`, so that is where its census goes; `--census-csv` works too.
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

    // Small enough that every case is quick, large enough to span several
    // cache blocks in at least one dimension.
    println!(
        "\nverifying {} cases at {} MiB nominal size\n",
        cases.len(),
        opts.size_mib
    );

    let mut t = Table::new(&[
        "case", "dtype", "m", "n", "k", "plan", "packed", "vs ttgt", "vs tblis", "status",
    ]);
    let mut failures = 0usize;
    // Known-value GEMM: A=B=1 implies every D element is exactly K. Not a corpus
    // cell, so it is not in the census.
    let known = corpus::size_case(&corpus::corpus()[0], 256.0);
    failures += check::<f64>(&known, exec, &mut t, true, None, &[]);
    failures += check::<Complex<f64>>(&known, exec, &mut t, true, None, &[]);

    for case in &cases {
        let s = corpus::size_case_stressed(case, opts.tensor_bytes(), opts.stress);
        if opts.wants("f32") {
            failures += check::<f32>(&s, exec, &mut t, false, census.as_mut(), &opts.engines);
        }
        if opts.wants("f64") {
            failures += check::<f64>(&s, exec, &mut t, false, census.as_mut(), &opts.engines);
        }
        if opts.wants("c32") {
            failures +=
                check::<Complex<f32>>(&s, exec, &mut t, false, census.as_mut(), &opts.engines);
        }
        if opts.wants("c64") {
            failures +=
                check::<Complex<f64>>(&s, exec, &mut t, false, census.as_mut(), &opts.engines);
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

fn tol<T: Element>() -> f64 {
    if core::mem::size_of::<T::Real>() == 4 {
        2e-3
    } else {
        1e-10
    }
}

/// Run `plan` once on fresh data, overwriting `d`.
fn run_plan<T: BenchElem>(plan: &Plan<T>, exec: &Exec<'_>, a: &[T], b: &[T], d: &mut [T]) {
    // SAFETY: the buffers are sized by the layouts (`elems_*`), `D` is exclusive,
    // and `beta = 0` so no previous value is read.
    unsafe {
        plan.execute_raw(
            exec,
            <T as Element>::one(),
            a.as_ptr(),
            b.as_ptr(),
            <T as Element>::zero(),
            std::ptr::null(),
            d.as_mut_ptr(),
        )
    }
    .expect("a validated plan runs");
}

fn check<T>(
    s: &Sized,
    exec: &Exec<'_>,
    t: &mut Table,
    known: bool,
    mut census: Option<&mut crate::report::Census>,
    rows: &[String],
) -> usize
where
    T: BenchElem,
{
    let build = |config: &PlanConfig| -> Result<Plan<T>, String> {
        let problem = super::problem_of::<T>(s).map_err(|e| e.to_string())?;
        Plan::<T>::new(&problem, config).map_err(|e| e.to_string())
    };
    let knobs = crate::knobs::get();
    let (plan, packed) = match (build(&knobs.config()), build(&knobs.packed_config())) {
        (Ok(p), Ok(q)) => (p, q),
        (Err(e), _) | (_, Err(e)) => {
            if let Some(c) = census.as_deref_mut() {
                for row in rows {
                    c.mark_skipped(s.case.name, T::NAME, row, "planning failed");
                }
            }
            t.row(vec![
                s.case.name.into(),
                T::NAME.into(),
                "-".into(),
                "-".into(),
                "-".into(),
                "-".into(),
                "-".into(),
                "-".into(),
                "-".into(),
                "-".into(),
                format!("PLAN ERROR: {e}"),
            ]);
            return 1;
        }
    };
    let (m, n, k) = s.mnk();

    let mut rng = ChaCha8Rng::seed_from_u64(0xA11CE);
    let mut sample = || {
        if known {
            <T as Element>::one()
        } else {
            T::sample(&mut rng)
        }
    };
    let a: Vec<T> = (0..s.elems_a()).map(|_| sample()).collect();
    let b: Vec<T> = (0..s.elems_b()).map(|_| sample()).collect();
    let mut d: Vec<T> = vec![<T as Element>::zero(); s.elems_c()];
    run_plan(&plan, exec, &a, &b, &mut d);
    // The packed driver, forced, against the planner's own choice.
    let mut dp: Vec<T> = vec![<T as Element>::zero(); s.elems_c()];
    run_plan(&packed, exec, &a, &b, &mut dp);
    let packed_err = super::rel_error(&dp, &d);

    let mut fails = usize::from(known && d.iter().any(|&v| v != T::from_f64(k as f64)));

    let ttgt_err: Option<f64> = {
        #[cfg(feature = "blas")]
        {
            let tp = crate::ttgt::TtgtPlan::new(plan.problem());
            let mut scratch = crate::ttgt::TtgtScratch::<T>::new(&tp);
            let mut dt: Vec<T> = vec![<T as Element>::zero(); s.elems_c()];
            let empty: Vec<T> = Vec::new();
            crate::ttgt::ttgt(
                &tp,
                <T as Element>::one(),
                &a,
                &b,
                <T as Element>::zero(),
                &empty,
                &mut dt,
                &mut scratch,
            );
            Some(rel_error(&dt, &d))
        }
        #[cfg(not(feature = "blas"))]
        {
            None
        }
    };

    let tblis_err: Option<f64> = {
        #[cfg(feature = "tblis")]
        {
            use crate::tblis as tb;
            let mut oa = tb::Operand::new(s.la.extents(), s.la.strides(), s.case.a);
            let mut ob = tb::Operand::new(s.lb.extents(), s.lb.strides(), s.case.b);
            let mut oc = tb::Operand::new(s.lc.extents(), s.lc.strides(), s.case.c);
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
            };
            Some(rel_error(&dt, &d))
        }
        #[cfg(not(feature = "tblis"))]
        {
            None
        }
    };

    let fmt = |e: Option<f64>| e.map(|v| format!("{v:.1e}")).unwrap_or_else(|| "-".into());
    let limit = tol::<T>();
    let bad = [Some(packed_err), ttgt_err, tblis_err]
        .iter()
        .flatten()
        .any(|&e| !e.is_finite() || e > limit);
    if bad {
        fails += 1;
    }
    let bad = fails != 0;
    t.row(vec![
        s.case.name.into(),
        T::NAME.into(),
        m.to_string(),
        n.to_string(),
        k.to_string(),
        plan.report().algorithm.name().into(),
        fmt(Some(packed_err)),
        fmt(ttgt_err),
        fmt(tblis_err),
        if bad { "FAIL".into() } else { "ok".into() },
    ]);
    // The check exercised every requested engine row, so each is a measured
    // cell (a FAIL is still measured; the exit status carries the verdict).
    if let Some(c) = census {
        for row in rows {
            c.mark_ok(s.case.name, T::NAME, row);
        }
    }
    fails
}
