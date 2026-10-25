//! Where a call through an executor runs, read from the `strided-exec` ledger:
//! a pooled executor enters its pool once per large call and never for a small
//! one, a budget of one or a serial executor never enters a pool, and the
//! environment does not choose the width. The calls go through the Rust seam
//! (`Executor::with`) that the TAPP entry points use, so the token is the one a
//! C call would get.
use std::ffi::c_void;

use strided_exec::ledger;
use tcontract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
use tcontract::{Plan, PlanConfig};
use tcontract_capi::executor::{executor_ref, Executor};
use tcontract_capi::*;

fn plan(m: usize, n: usize, k: usize) -> Plan<f64> {
    let spec = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
    let problem = Problem::from_labels(
        DType::F64,
        spec(&[m, k], &[1, m as isize]),
        spec(&[k, n], &[1, k as isize]),
        CSpec::Absent,
        spec(&[m, n], &[1, m as isize]),
        &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    )
    .unwrap();
    Plan::<f64>::new(&problem, &PlanConfig::packed()).unwrap()
}

/// One contraction through `e`; returns `(installs, max_live)` of its token.
fn run(e: &Executor, m: usize, n: usize, k: usize) -> (usize, usize) {
    let (a, b) = (vec![1.0f64; m * k], vec![1.0f64; k * n]);
    let mut d = vec![0.0f64; m * n];
    let plan = plan(m, n, k);
    let root = e
        .with(|exec| {
            let root = exec.token().ledger_root();
            plan.execute_slices(exec, 1.0, (&a, 0), (&b, 0), (&mut d, 0))
                .map_err(|e| {
                    tcontract_capi::status::FfiError::new(
                        tcontract_capi::status::TCONTRACT_ERR_INTERNAL,
                        e.to_string(),
                    )
                })?;
            Ok(root)
        })
        .unwrap();
    assert!(d.iter().all(|&x| x == k as f64), "wrong product");
    root.map_or((0, 0), |r| (ledger::installs(r), ledger::max_live(r)))
}

unsafe fn rayon_exec(n: usize) -> isize {
    let mut e = 0isize;
    assert_eq!(
        tcontract_tapp_executor_create_rayon(&mut e, n, std::ptr::null()),
        TAPP_SUCCESS
    );
    e
}

#[test]
fn large_work_enters_the_pool_once_and_small_work_never() {
    unsafe {
        let four = rayon_exec(4);
        let e = executor_ref(four).unwrap();
        let (installs, live) = run(e, 256, 256, 256);
        assert_eq!(installs, 1, "a large contraction enters the pool once");
        assert!(live <= 4, "{live}");
        // Small work stays on the caller of an explicitly larger pool.
        for _ in 0..10 {
            assert_eq!(run(e, 4, 4, 4).0, 0);
        }
        // Each call mints its own token: two plans, two entries.
        assert_eq!(run(e, 240, 250, 260).0, 1);
        assert_eq!(TAPP_destroy_executor(four), TAPP_SUCCESS);

        // 1T and the default executor have no pool to enter.
        let one = rayon_exec(1);
        assert_eq!(run(executor_ref(one).unwrap(), 256, 256, 256).0, 0);
        assert_eq!(TAPP_destroy_executor(one), TAPP_SUCCESS);
    }
}

#[test]
fn width_is_the_executors_not_the_environment_and_budget_one_is_serial() {
    unsafe {
        // The environment asks for one thread; the 4T executor still decides.
        std::env::set_var("RAYON_NUM_THREADS", "1");
        let four = rayon_exec(4);
        let e = executor_ref(four).unwrap();
        let (installs, live) = run(e, 256, 256, 256);
        assert_eq!(installs, 1);
        assert!(live > 1 && live <= 4, "{live}");
        // A budget of one keeps the call serial however large the pool.
        assert_eq!(tcontract_tapp_executor_set_budget(four, 1), TAPP_SUCCESS);
        assert_eq!(
            run(e, 256, 256, 256).0,
            0,
            "budget 1 does not enter the pool"
        );
        std::env::remove_var("RAYON_NUM_THREADS");
        assert_eq!(TAPP_destroy_executor(four), TAPP_SUCCESS);
    }
}

#[test]
fn a_borrowed_executor_enters_the_hosts_pool_once() {
    unsafe {
        let mut p = 0;
        assert_eq!(
            tcontract_pool_create_rayon(&mut p, 4, std::ptr::null()),
            TAPP_SUCCESS
        );
        let mut four = 0;
        assert_eq!(
            tcontract_tapp_executor_borrow_pool(&mut four, p, 4),
            TAPP_SUCCESS
        );
        let (installs, live) = run(executor_ref(four).unwrap(), 256, 256, 256);
        assert_eq!(installs, 1, "the product entered the borrowed pool once");
        assert!(live <= 4);
        assert_eq!(TAPP_destroy_executor(four), TAPP_SUCCESS);
        assert_eq!(tcontract_pool_destroy(p), TAPP_SUCCESS);
        let _ = std::marker::PhantomData::<*const c_void>;
    }
}
