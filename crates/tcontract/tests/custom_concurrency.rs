//! Independent catalogs, selectors and plans coexist: on one shared pool, on
//! separate pools, and next to the built-in defaults they never touch.
use strided_view::{StridedView, StridedViewMut};
use tcontract::api::{DType, DotGeneral, LayoutSpec, OperandSpec, Problem};
use tcontract::{Plan, PlanConfig};
use tcontract_exec::{Exec, Pool};
use tcontract_kernel::KernelCatalog;
use tcontract_testkit::custom_kernels as own;

const M: usize = 150;
const N: usize = 130;
const K: usize = 90;

fn catalog() -> KernelCatalog<f64> {
    // SAFETY: the testkit's families are immutable, `'static` descriptors that
    // meet the family contract (checked by their own tests).
    unsafe { KernelCatalog::<f64>::from_static_families(own::f64_families()) }.unwrap()
}

fn data(len: usize, seed: usize) -> Vec<f64> {
    (0..len)
        .map(|x| ((x * 7 + seed) % 11) as f64 - 5.0)
        .collect()
}

fn problem() -> Problem {
    let spec = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
    Problem::from_dot_general(
        DType::F64,
        spec(&[M, K], &[1, M as isize]),
        spec(&[K, N], &[1, K as isize]),
        spec(&[M, N], &[1, M as isize]),
        &DotGeneral::new(&[1], &[0], &[], &[]),
    )
    .unwrap()
}

fn plan_for(cat: &KernelCatalog<f64>, id: &'static str) -> Plan<f64> {
    Plan::<f64>::new_with_selector(&problem(), &PlanConfig::default(), cat, &mut |_, _| {
        Ok(cat.get(id).unwrap())
    })
    .unwrap()
}

fn run(plan: &Plan<f64>, exec: &Exec<'_>) -> Vec<f64> {
    let mut c = vec![0.0; M * N];
    fill(plan, exec, &mut c).unwrap();
    c
}

/// One contraction into `c`.
fn fill(plan: &Plan<f64>, exec: &Exec<'_>, c: &mut [f64]) -> Result<(), tcontract::api::Error> {
    let (a, b) = (data(M * K, 1), data(K * N, 2));
    let av = StridedView::new(&a, &[M, K], &[1, M as isize], 0).unwrap();
    let bv = StridedView::new(&b, &[K, N], &[1, K as isize], 0).unwrap();
    let mut cv = StridedViewMut::new(c, &[M, N], &[1, M as isize], 0).unwrap();
    plan.execute_into(exec, 1.0, &av, &bv, &mut cv)
}

/// `reps` contractions on a plan choosing `id` from a private catalog; returns
/// the product and the family the plan reports.
fn worker(exec: &Exec<'_>, id: &'static str, reps: usize) -> (Vec<f64>, &'static str) {
    let cat = catalog();
    let plan = plan_for(&cat, id);
    let mut c = vec![];
    for _ in 0..reps {
        c = run(&plan, exec);
    }
    (c, plan.report().packed.as_ref().unwrap().family_id)
}

#[test]
fn catalogs_run_concurrently_on_one_pool_and_on_separate_pools() {
    let default_id = || {
        tcontract_kernel::ResolvedGemm::<f64>::resolve::<f64>(
            &tcontract_kernel::KernelChoice::Auto,
            1,
        )
        .unwrap()
        .family()
        .id
    };
    let before = default_id();
    let tp = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let reference = worker(&Exec::serial(), "custom.f64.2x2", 1).0;

    // Same pool, two threads, two different custom kernels.
    let shared = Pool::borrow(&tp);
    let (x, y) = std::thread::scope(|s| {
        let x = s.spawn(|| worker(&Exec::rayon(&shared), "custom.f64.2x2", 6));
        let y = s.spawn(|| worker(&Exec::rayon(&shared), "custom.f64.3x4", 6));
        (x.join().unwrap(), y.join().unwrap())
    });
    assert_eq!((x.1, y.1), ("custom.f64.2x2", "custom.f64.3x4"));
    assert_eq!(x.0, reference);
    assert_eq!(y.0, reference);

    // Separate pools (each with its own workspace) on the same worker threads.
    let (pa, pb) = (Pool::borrow(&tp), Pool::borrow(&tp));
    let (x, y) = std::thread::scope(|s| {
        let x = s.spawn(|| worker(&Exec::rayon(&pa), "custom.f64.3x4", 4));
        let y = s.spawn(|| worker(&Exec::rayon(&pb), "custom.f64.2x2", 4));
        (x.join().unwrap(), y.join().unwrap())
    });
    assert_eq!(x.0, reference);
    assert_eq!(y.0, reference);

    // The built-in default is exactly what it was.
    let after = default_id();
    assert_eq!(before, after);
    assert!(tcontract_kernel::list_kernels::<f64>()
        .iter()
        .all(|k| !k.id.starts_with("custom.")));
}

#[test]
fn a_call_from_a_worker_is_refused_without_a_write_on_the_same_family() {
    let tp = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let pool = Pool::borrow(&tp);
    let exec = Exec::rayon(&pool);
    let cat = catalog();
    let plan = plan_for(&cat, "custom.f64.3x4");
    drop(cat);
    let reference = worker(&Exec::serial(), "custom.f64.2x2", 1).0;
    let outside = run(&plan, &exec);
    assert_eq!(outside, reference);
    // Inside a worker of the very pool the plan runs on, a barrier-bearing team
    // cannot be co-scheduled: the call reports the refused route with no write
    // and never switches to another family.
    let mut c = vec![-7.0; M * N];
    let err = tp.install(|| fill(&plan, &exec, &mut c)).unwrap_err();
    assert!(matches!(err, tcontract::api::Error::Exec(_)), "{err}");
    assert!(c.iter().all(|&v| v == -7.0), "a refused route wrote");
    assert_eq!(
        plan.report().packed.as_ref().unwrap().family_id,
        "custom.f64.3x4"
    );
}
