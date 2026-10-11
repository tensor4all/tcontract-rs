//! tcontract addition: threads come from a `tcontract_exec::Exec` alone. A wide
//! budget broadcasts on the pool, a width of one never does, and a team the
//! pool refuses is reported as a route error with nothing written.

use std::sync::atomic::{AtomicUsize, Ordering};

use strided_view::{StridedView, StridedViewMut};
use tcontract::api::Error;
use tcontract::{Plan, PlanConfig};
use tcontract_exec::{Exec, Pool};
use tcontract_kernel::KernelChoice;

mod common;
use common::plans::{matmul_problem, packed};

const M: usize = 256;
const N: usize = 240;
const K: usize = 200;

fn inputs() -> (Vec<f64>, Vec<f64>) {
    let a = (0..M * K).map(|x| (x % 13) as f64 - 6.0).collect();
    let b = (0..K * N).map(|x| (x % 7) as f64 * 0.5).collect();
    (a, b)
}

fn pool4() -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
}

/// One contraction into `out`.
fn fill(plan: &Plan<f64>, exec: &Exec<'_>, out: &mut [f64]) -> Result<(), Error> {
    let (av, bv) = inputs();
    plan.execute_into(
        exec,
        1.0,
        &StridedView::new(&av, &[M, K], &[1, M as isize], 0).unwrap(),
        &StridedView::new(&bv, &[K, N], &[1, K as isize], 0).unwrap(),
        &mut StridedViewMut::new(out, &[M, N], &[1, M as isize], 0).unwrap(),
    )
}

fn run_on(plan: &Plan<f64>, exec: &Exec<'_>) -> Vec<f64> {
    let mut out = vec![0.0; M * N];
    fill(plan, exec, &mut out).unwrap();
    out
}

#[test]
fn exec_matches_serial_bitwise_and_a_refused_team_reports_no_route() {
    let plan = Plan::<f64>::new(&matmul_problem(M, N, K), &packed()).unwrap();
    let serial = run_on(&plan, &Exec::serial());

    let tp = pool4();
    let pool = Pool::borrow(&tp);
    let exec = Exec::rayon(&pool);
    assert_eq!(run_on(&plan, &exec), serial);
    assert_eq!(pool.stats().broadcasts, 1);

    // Called from one of the pool's own workers, the team cannot be
    // co-scheduled; the call reports that instead of silently running serially,
    // and the output is untouched.
    pool.reset_stats();
    let mut out = vec![-7.0; M * N];
    let err = exec
        .install(2, |_| fill(&plan, &exec, &mut out))
        .unwrap_err();
    assert!(matches!(err, Error::Exec(_)), "{err}");
    assert!(out.iter().all(|&v| v == -7.0), "a refused route wrote");
    assert_eq!(pool.stats().broadcasts, 0);
}

#[test]
fn width_one_never_broadcasts_whatever_the_pool_could_do() {
    let plan = Plan::<f64>::new(&matmul_problem(M, N, K), &packed()).unwrap();
    let tp = pool4();
    let pool = Pool::borrow(&tp);
    let exec = Exec::rayon(&pool).with_budget(1).unwrap();
    let serial = run_on(&plan, &Exec::serial());
    assert_eq!(run_on(&plan, &exec), serial);
    assert_eq!(pool.stats().broadcasts, 0);
    assert_eq!(pool.stats().entries, 0);
}

#[test]
fn a_refused_route_runs_no_kernel_and_keeps_the_family() {
    use tcontract_kernel::{KernelFamily, UkrFn};
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    unsafe fn traced(k: usize, a: *const f64, b: *const f64, out: *mut f64) {
        CALLS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: identical panel/tile ABI to the descriptor we copy below.
        unsafe {
            tcontract_kernel::portable::real_tile::<f64, 4, 4>(k, a, b, out);
        }
    }
    fn manifest() -> &'static [&'static KernelFamily<f64>] {
        static LIST: std::sync::OnceLock<[&'static KernelFamily<f64>; 1]> =
            std::sync::OnceLock::new();
        LIST.get_or_init(|| {
            // Do not recurse through the registry while this callback's
            // OnceLock initializes: copy the immutable built-in menu directly.
            let mut f = **tcontract_kernel::portable::families_f64()
                .iter()
                .find(|f| f.id == "ref.f64.real.4x4")
                .unwrap();
            f.id = "test.traced.f64.4x4";
            f.allow_auto = false;
            f.ukr = UkrFn::Tile(traced);
            [Box::leak(Box::new(f))]
        })
    }
    // SAFETY: immutable manifest copies the validated portable 4x4 footprint,
    // ISA and overwrite contract; traced only counts then calls that same ABI.
    unsafe {
        tcontract_kernel::register::<f64>(manifest);
    }
    let config = PlanConfig {
        kernel: KernelChoice::Id("test.traced.f64.4x4".into()),
        ..PlanConfig::default()
    };
    let plan = Plan::<f64>::new(&matmul_problem(M, N, K), &config).unwrap();
    let _ = run_on(&plan, &Exec::serial());
    assert!(
        CALLS.load(Ordering::Relaxed) > 0,
        "the forced family never ran"
    );
    CALLS.store(0, Ordering::Relaxed);
    // The caller is a worker of the pool, so no team can be co-scheduled. The
    // refusal runs nothing at all: no other kernel may be selected in its
    // place, and the output stays untouched.
    let tp = pool4();
    let pool = Pool::borrow(&tp);
    let exec = Exec::rayon(&pool);
    let mut out = vec![-7.0; M * N];
    let err = exec
        .install(2, |_| fill(&plan, &exec, &mut out))
        .unwrap_err();
    assert!(matches!(err, Error::Exec(_)), "{err}");
    assert_eq!(
        CALLS.load(Ordering::Relaxed),
        0,
        "a refused route ran a kernel"
    );
    assert!(out.iter().all(|&v| v == -7.0), "a refused route wrote");
    assert_eq!(pool.stats().broadcasts, 0);
}
