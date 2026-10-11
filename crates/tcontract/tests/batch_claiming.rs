//! The in-plan batch axis runs barrier-free across the pool when the items are
//! tiny, or when there are at least as many items as workers, and the result is
//! bitwise the serial one at every width. A big item with few batch entries
//! keeps the SPMD team.
use std::time::Duration;

use num_complex::Complex64;
use strided_view::{StridedView, StridedViewMut};
use tcontract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem, Scalar};
use tcontract::{Plan, PlanConfig};
use tcontract_exec::{Exec, Pool};
use tcontract_kernel::{Element, Real};

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
fn run<T: Scalar>(dtype: DType, exec: &Exec<'_>, dims: [usize; 4]) -> Vec<T> {
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

fn pool8() -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap()
}

/// Item shapes: tiny (always batch-parallel), and ~1.7 MMAC (SPMD-sized).
const TINY: [usize; 3] = [3, 5, 7];
/// Tiny too (about 3 us), but wide enough that the team would split it.
const SPLITTABLE: [usize; 3] = [48, 40, 16];
const BIG: [usize; 3] = [120, 120, 120];

fn sweep<T: Scalar>(dtype: DType, item: [usize; 3], batches: &[usize]) {
    let tp = pool8();
    let pool = Pool::borrow(&tp);
    for &h in batches {
        let dims = [item[0], item[1], item[2], h];
        let serial = bits(&run::<T>(dtype, &Exec::serial(), dims));
        for w in [1usize, 2, 3, 4, 8] {
            let exec = Exec::rayon(&pool).with_budget(w).unwrap();
            let got = bits(&run::<T>(dtype, &exec, dims));
            assert_eq!(got, serial, "{dtype:?} item {item:?} h {h} width {w}");
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
    let tp = pool8();
    let pool = Pool::borrow(&tp);
    let exec = Exec::rayon(&pool);
    let d = run::<f64>(DType::F64, &exec, [m, n, k, h]);
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
    let tp = pool8();
    let pool = Pool::borrow(&tp);
    let exec = Exec::rayon(&pool);
    // Tiny items: batch lanes, no SPMD broadcast (and so no barrier).
    run::<f64>(
        DType::F64,
        &exec,
        [SPLITTABLE[0], SPLITTABLE[1], SPLITTABLE[2], 64],
    );
    assert_eq!(pool.stats().broadcasts, 0, "tiny items used the team");
    assert!(
        pool.stats().entries >= 1,
        "tiny items never left the caller"
    );
    // A big item with fewer entries than workers keeps the team.
    run::<f64>(DType::F64, &exec, [BIG[0], BIG[1], BIG[2], 3]);
    assert_eq!(pool.stats().broadcasts, 1, "a big item lost its team");
    // As many entries as workers: batch lanes again.
    run::<f64>(DType::F64, &exec, [BIG[0], BIG[1], BIG[2], 8]);
    assert_eq!(
        pool.stats().broadcasts,
        1,
        "H >= width should not broadcast"
    );
    // 9 large entries on 8 lanes would leave one lane with two: keep the team.
    run::<f64>(DType::F64, &exec, [BIG[0], BIG[1], BIG[2], 9]);
    assert_eq!(pool.stats().broadcasts, 2, "an uneven split lost its team");
}

#[test]
fn a_tiny_total_stays_on_the_caller() {
    let tp = pool8();
    let pool = Pool::borrow(&tp);
    let exec = Exec::rayon(&pool);
    run::<f64>(DType::F64, &exec, [2, 2, 2, 3]);
    let s = pool.stats();
    assert_eq!((s.entries, s.broadcasts), (0, 0));
}

#[test]
fn steady_state_reuses_the_pool_workspace() {
    let tp = pool8();
    let pool = Pool::borrow(&tp);
    let exec = Exec::rayon(&pool);
    let dims = [SPLITTABLE[0], SPLITTABLE[1], SPLITTABLE[2], 500];
    run::<f64>(DType::F64, &exec, dims);
    let panel = pool.workspace().retained_panel_bytes();
    assert!(panel > 0, "the batch lanes never leased a panel");
    for _ in 0..3 {
        run::<f64>(DType::F64, &exec, dims);
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
        let tp = pool8();
        let pool = Pool::borrow(&tp);
        let exec = Exec::rayon(&pool);
        let dims = [SPLITTABLE[0], SPLITTABLE[1], SPLITTABLE[2], 64];
        let serial = bits(&run::<f64>(DType::F64, &Exec::serial(), dims));
        let before = pool.stats();
        let nested = exec.install(2, |_| bits(&run::<f64>(DType::F64, &exec, dims)));
        assert_eq!(nested, serial);
        let after = pool.stats();
        assert_eq!(after.broadcasts, before.broadcasts);
        // The outer `install` entered the pool; the lanes ran in place on the
        // worker, which is what makes the batch path barrier-free.
        assert_eq!(after.entries, before.entries + 1);
        assert_eq!(after.inline_runs, before.inline_runs + 1);
        let _ = tx.send(());
    });
    rx.recv_timeout(Duration::from_secs(60))
        .expect("nested batch execution deadlocked");
}
