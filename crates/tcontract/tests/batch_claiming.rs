//! The in-plan batch axis runs barrier-free across the pool when the items are
//! tiny, or when there are at least as many items as workers, and the result is
//! bitwise the serial one at every width. A big item with few batch entries
//! keeps the SPMD team.
use std::time::Duration;

use num_complex::Complex64;
use strided_view::{StridedView, StridedViewMut};
use tcontract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem, Scalar};
use tcontract::{ExecutionRoute, OutputContract, PackedRoute, Plan, PlanConfig};
use tcontract_exec::{Exec, Pool};
use tcontract_kernel::{Element, Real};

mod common;
use common::placement::{installs, max_live, on_worker, pool, root};

/// `D[m,n,h] = sum_k A[m,k,h] B[k,n,h]`, column-major, `h` outermost.
fn batched_problem(dtype: DType, m: usize, n: usize, k: usize, h: usize) -> Problem {
    let spec = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
    let (mi, ni, ki) = (m as isize, n as isize, k as isize);
    Problem::from_labels(
        dtype,
        spec(&[m, k, h], &[1, mi, mi * ki]),
        spec(&[k, n, h], &[1, ki, ki * ni]),
        CSpec::Absent,
        spec(&[m, n, h], &[1, mi, mi * ni]),
        &Labels::new(&[0, 2, 3], &[2, 1, 3], &[0, 1, 3]),
    )
    .unwrap()
}

fn value<T: Element>(i: usize) -> T {
    let r = T::Real::from_f64(((i * 7 + 3) % 23) as f64 * 0.11 - 1.0);
    let im = T::Real::from_f64(((i * 5 + 1) % 17) as f64 * 0.07 - 0.5);
    T::from_parts(r, if T::IS_COMPLEX { im } else { T::Real::ZERO })
}

/// One execute of the packed default plan on `exec`, returning `D`.
fn run<T: Scalar>(dtype: DType, exec: &mut Exec<'_>, dims: [usize; 4]) -> Vec<T> {
    let [m, n, k, h] = dims;
    let a: Vec<T> = (0..m * k * h).map(value).collect();
    let b: Vec<T> = (0..k * n * h).map(|i| value(i + 5)).collect();
    let mut d = vec![<T as Element>::zero(); m * n * h];
    let plan = Plan::<T>::new(&batched_problem(dtype, m, n, k, h), &PlanConfig::packed()).unwrap();
    let (mi, ni, ki) = (m as isize, n as isize, k as isize);
    plan.execute_into(
        exec,
        <T as Element>::one(),
        &StridedView::new(&a, &[m, k, h], &[1, mi, mi * ki], 0).unwrap(),
        &StridedView::new(&b, &[k, n, h], &[1, ki, ki * ni], 0).unwrap(),
        &mut StridedViewMut::new(&mut d, &[m, n, h], &[1, mi, mi * ni], 0).unwrap(),
    )
    .unwrap();
    d
}

fn bits<T: Scalar>(v: &[T]) -> Vec<u64> {
    v.iter()
        .flat_map(|x| {
            let (re, im) = (x.re().to_f64(), x.im().to_f64());
            [re.to_bits(), im.to_bits()]
        })
        .collect()
}

fn pool8() -> strided_exec::Pool {
    pool(8)
}

/// The packed route the default plan takes for `dims` on `exec`, and the width
/// it runs at.
fn route_of(dtype: DType, exec: &Exec<'_>, dims: [usize; 4]) -> (PackedRoute, usize) {
    let [m, n, k, h] = dims;
    let plan =
        Plan::<f64>::new(&batched_problem(dtype, m, n, k, h), &PlanConfig::packed()).unwrap();
    match plan
        .execution_route(exec, 1.0, OutputContract::Initialized)
        .unwrap()
    {
        ExecutionRoute::Packed { kind, width } => (kind, width),
        other => panic!("not a packed route: {other:?}"),
    }
}

/// Item shapes: tiny (always batch-parallel), and ~1.7 MMAC (SPMD-sized).
const TINY: [usize; 3] = [3, 5, 7];
/// Tiny too (about 3 us), but wide enough that the team would split it.
const SPLITTABLE: [usize; 3] = [48, 40, 16];
const BIG: [usize; 3] = [120, 120, 120];

fn sweep<T: Scalar>(dtype: DType, item: [usize; 3], batches: &[usize]) {
    let sp = pool8();
    let pool = Pool::borrow(&sp);
    for &h in batches {
        let dims = [item[0], item[1], item[2], h];
        let serial = bits(&run::<T>(dtype, &mut Exec::serial(), dims));
        for w in [1usize, 2, 3, 4, 8] {
            let mut exec = Exec::new(&pool, w).unwrap();
            let root = root(&mut exec);
            let got = bits(&run::<T>(dtype, &mut exec, dims));
            assert_eq!(got, serial, "{dtype:?} item {item:?} h {h} width {w}");
            // Teams and lanes alike never run more tasks at once than the budget.
            assert!(max_live(root) <= w, "item {item:?} h {h} width {w}");
        }
    }
}

#[test]
fn tiny_items_are_bitwise_serial_at_every_width() {
    // 1 and 2 are below the width, 13 and 64 do not divide it evenly.
    let hs = [1, 2, 3, 13, 64, 257];
    for item in [TINY, SPLITTABLE] {
        sweep::<f32>(DType::F32, item, &hs);
        sweep::<f64>(DType::F64, item, &hs);
        sweep::<Complex64>(DType::C64, item, &hs);
    }
}

#[test]
fn large_items_are_bitwise_serial_at_every_width() {
    // 3 < 8 keeps the team; 8 and 11 take the batch lanes.
    let hs = [2, 3, 8, 11];
    sweep::<f32>(DType::F32, BIG, &hs);
    sweep::<f64>(DType::F64, BIG, &hs);
    sweep::<Complex64>(DType::C64, BIG, &hs);
}

#[test]
fn f64_matches_a_naive_contraction() {
    let [m, n, k, h] = [4usize, 3, 5, 9];
    let sp = pool8();
    let pool = Pool::borrow(&sp);
    let mut exec = Exec::new(&pool, pool.size()).unwrap();
    let d = run::<f64>(DType::F64, &mut exec, [m, n, k, h]);
    let a: Vec<f64> = (0..m * k * h).map(value).collect();
    let b: Vec<f64> = (0..k * n * h).map(|i| value(i + 5)).collect();
    for t in 0..h {
        for j in 0..n {
            for i in 0..m {
                let want: f64 = (0..k)
                    .map(|p| a[i + m * p + m * k * t] * b[p + k * j + k * n * t])
                    .sum();
                let got = d[i + m * j + m * n * t];
                assert!((got - want).abs() < 1e-12, "item {t} ({i},{j})");
            }
        }
    }
}

#[test]
fn the_mode_follows_the_width_rule() {
    let sp = pool8();
    let pool = Pool::borrow(&sp);
    // Each call mints its own token, so the ledger answers per call.
    let one = |dims: [usize; 4]| {
        let mut exec = Exec::new(&pool, pool.size()).unwrap();
        let (kind, width) = route_of(DType::F64, &exec, dims);
        let root = root(&mut exec);
        run::<f64>(DType::F64, &mut exec, dims);
        (kind, width, root)
    };
    // Tiny items: batch lanes, no SPMD team (and so no barrier).
    let (kind, _, root) = one([SPLITTABLE[0], SPLITTABLE[1], SPLITTABLE[2], 64]);
    assert_eq!(kind, PackedRoute::BatchLanes, "tiny items used the team");
    assert_eq!(installs(root), 1, "tiny items never left the caller");
    // A big item with fewer entries than workers keeps the team.
    let (kind, width, root) = one([BIG[0], BIG[1], BIG[2], 3]);
    assert_eq!(kind, PackedRoute::Spmd, "a big item lost its team");
    assert_eq!(max_live(root), width, "the team did not run at once");
    // As many entries as workers: batch lanes again.
    let (kind, _, _) = one([BIG[0], BIG[1], BIG[2], 8]);
    assert_eq!(
        kind,
        PackedRoute::BatchLanes,
        "H >= width should not use a team"
    );
    // 9 large entries on 8 lanes would leave one lane with two: keep the team.
    let (kind, width, root) = one([BIG[0], BIG[1], BIG[2], 9]);
    assert_eq!(kind, PackedRoute::Spmd, "an uneven split lost its team");
    assert_eq!(max_live(root), width);
}

#[test]
fn a_tiny_total_stays_on_the_caller() {
    let sp = pool8();
    let pool = Pool::borrow(&sp);
    let mut exec = Exec::new(&pool, pool.size()).unwrap();
    let root = root(&mut exec);
    run::<f64>(DType::F64, &mut exec, [2, 2, 2, 3]);
    assert_eq!(installs(root), 0);
}

#[test]
fn steady_state_reuses_the_pool_workspace() {
    let sp = pool8();
    let pool = Pool::borrow(&sp);
    let mut exec = Exec::new(&pool, pool.size()).unwrap();
    let dims = [SPLITTABLE[0], SPLITTABLE[1], SPLITTABLE[2], 500];
    run::<f64>(DType::F64, &mut exec, dims);
    let panel = pool.workspace().retained_panel_bytes();
    assert!(panel > 0, "the batch lanes never leased a panel");
    for _ in 0..3 {
        run::<f64>(DType::F64, &mut exec, dims);
    }
    assert_eq!(
        pool.workspace().retained_panel_bytes(),
        panel,
        "a steady-state batch run grew the panel"
    );
}

/// A worker caller is a legitimate barrier-free caller: the batch is cut into
/// lanes and run on the pool the caller already belongs to, with the same
/// result as the serial path.
#[test]
fn a_call_from_a_worker_runs_barrier_free_lanes_with_the_same_result() {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let sp = pool8();
        let pool = Pool::borrow(&sp);
        let dims = [SPLITTABLE[0], SPLITTABLE[1], SPLITTABLE[2], 64];
        let serial = bits(&run::<f64>(DType::F64, &mut Exec::serial(), dims));
        let (nested, root) = on_worker(&pool, || {
            let mut inner = Exec::new(&pool, pool.size()).unwrap();
            let root = root(&mut inner);
            (bits(&run::<f64>(DType::F64, &mut inner, dims)), root)
        });
        assert_eq!(nested, serial);
        // The lanes ran in place on the worker, which is what makes the batch
        // path barrier-free.
        assert_eq!(installs(root), 0);
        assert!(max_live(root) <= pool.size());
        let _ = tx.send(());
    });
    rx.recv_timeout(Duration::from_secs(60))
        .expect("nested batch execution deadlocked");
}
