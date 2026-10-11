//! `contract_batched`: many independent contractions with the batch as the
//! parallel axis, and the inner axis when there are fewer items than workers.
use strided_view::{StridedView, StridedViewMut};
use tcontract::{contract_batched, BatchItem, Plan, PlanConfig};
use tcontract_exec::{Exec, Pool};

mod common;
use common::plans::{matmul_problem, packed};

fn data(len: usize, seed: usize, scale: f64) -> Vec<f64> {
    (0..len).map(|j| (seed * 31 + j) as f64 * scale).collect()
}

/// A column-major view of `x`.
fn cm(x: &[f64], d: [usize; 2]) -> StridedView<'_, f64> {
    StridedView::new(x, &d, &[1, d[0] as isize], 0).unwrap()
}

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
}

/// `n` independent `ij,jk->ik` contractions with distinct data, run batched and
/// run one at a time, compared **bitwise**. That equivalence is the whole
/// contract of this module, checked at several batch widths so a chunking error
/// on the tail is not missed.
#[test]
fn batched_matches_one_at_a_time_bitwise() {
    let (m, k, n) = (12usize, 9usize, 7usize);
    let tp = pool(8);
    let pool = Pool::borrow(&tp);
    for config in [PlanConfig::default(), packed()] {
        let plan = Plan::<f64>::new(&matmul_problem(m, n, k), &config).unwrap();
        for items in [1usize, 2, 3, 7, 16] {
            for threads in [1usize, 2, 4, 8] {
                let exec = Exec::rayon(&pool).with_budget(threads).unwrap();
                // Distinct data per item, so a batch that ran the wrong item,
                // ran one twice, or skipped one cannot pass.
                let a: Vec<Vec<f64>> = (0..items).map(|i| data(m * k, i, 0.5)).collect();
                let b: Vec<Vec<f64>> = (0..items)
                    .map(|i| data(k * n, i * 17 / 31 + 1, 0.25))
                    .collect();
                let mut want = vec![vec![0.0f64; m * n]; items];
                for i in 0..items {
                    plan.execute_into(
                        &Exec::serial(),
                        1.0,
                        &cm(&a[i], [m, k]),
                        &cm(&b[i], [k, n]),
                        &mut StridedViewMut::new(&mut want[i], &[m, n], &[1, m as isize], 0)
                            .unwrap(),
                    )
                    .unwrap();
                }
                let mut got = vec![vec![0.0f64; m * n]; items];
                {
                    let mut batch: Vec<BatchItem<'_, f64>> = got
                        .iter_mut()
                        .enumerate()
                        .map(|(i, d)| BatchItem {
                            plan: &plan,
                            alpha: 1.0,
                            a: cm(&a[i], [m, k]),
                            b: cm(&b[i], [k, n]),
                            beta: 0.0,
                            source: None,
                            d: StridedViewMut::new(d, &[m, n], &[1, m as isize], 0).unwrap(),
                        })
                        .collect();
                    contract_batched(&mut batch, &exec).unwrap();
                }
                assert_eq!(got, want, "items = {items}, threads = {threads}");
            }
        }
    }
}

/// A bad item must stop the whole batch **before** anything is written. This is
/// the guarantee a loop over `Plan::execute_into` cannot give.
#[test]
fn a_bad_item_leaves_every_output_untouched() {
    let plan = Plan::<f64>::new(&matmul_problem(4, 2, 3), &PlanConfig::default()).unwrap();
    let a = vec![1.0f64; 12];
    let b = vec![1.0f64; 6];
    let mut d0 = vec![0.0f64; 8];
    let mut d1 = vec![0.0f64; 8];
    {
        fn view(x: &[f64], d: [usize; 2], s: [isize; 2]) -> StridedView<'_, f64> {
            StridedView::new(x, &d, &s, 0).unwrap()
        }
        let mut batch = vec![
            BatchItem {
                plan: &plan,
                alpha: 1.0,
                a: view(&a, [4, 3], [1, 4]),
                b: view(&b, [3, 2], [1, 3]),
                beta: 0.0,
                source: None,
                d: StridedViewMut::new(&mut d0, &[4, 2], &[1, 4], 0).unwrap(),
            },
            BatchItem {
                plan: &plan,
                alpha: 1.0,
                // Row-major A is another layout than the plan's.
                a: view(&a, [4, 3], [3, 1]),
                b: view(&b, [3, 2], [1, 3]),
                beta: 0.0,
                source: None,
                d: StridedViewMut::new(&mut d1, &[4, 2], &[1, 4], 0).unwrap(),
            },
        ];
        assert!(contract_batched(&mut batch, &Exec::serial()).is_err());
    }
    assert!(
        d0.iter().all(|&x| x == 0.0),
        "the good item ran despite a later item being invalid"
    );
    assert!(d1.iter().all(|&x| x == 0.0));
}

#[test]
fn an_empty_batch_is_ok() {
    let mut items: Vec<BatchItem<'_, f64>> = Vec::new();
    contract_batched(&mut items, &Exec::serial()).unwrap();
}

/// The scheduling rule: with at least as many items as workers the items are
/// partitioned over the width and run serially (no broadcast); with fewer, each
/// item keeps the whole `Exec` and runs its own inner SPMD.
#[test]
fn fewer_items_than_the_width_run_inner_spmd_and_more_items_partition_the_batch() {
    let (m, n, k) = (256usize, 240usize, 200usize);
    let plan = Plan::<f64>::new(&matmul_problem(m, n, k), &packed()).unwrap();
    let a = data(m * k, 1, 0.01);
    let b = data(k * n, 2, 0.02);
    let tp = pool(4);
    let pool = Pool::borrow(&tp);
    let exec = Exec::rayon(&pool);

    let run = |count: usize| -> Vec<Vec<f64>> {
        let mut outs = vec![vec![0.0f64; m * n]; count];
        let mut batch: Vec<BatchItem<'_, f64>> = outs
            .iter_mut()
            .map(|d| BatchItem {
                plan: &plan,
                alpha: 1.0,
                a: StridedView::new(&a, &[m, k], &[1, m as isize], 0).unwrap(),
                b: StridedView::new(&b, &[k, n], &[1, k as isize], 0).unwrap(),
                beta: 0.0,
                source: None,
                d: StridedViewMut::new(d, &[m, n], &[1, m as isize], 0).unwrap(),
            })
            .collect();
        contract_batched(&mut batch, &exec).unwrap();
        drop(batch);
        outs
    };
    let mut serial = vec![0.0f64; m * n];
    plan.execute_into(
        &Exec::serial(),
        1.0,
        &StridedView::new(&a, &[m, k], &[1, m as isize], 0).unwrap(),
        &StridedView::new(&b, &[k, n], &[1, k as isize], 0).unwrap(),
        &mut StridedViewMut::new(&mut serial, &[m, n], &[1, m as isize], 0).unwrap(),
    )
    .unwrap();

    // Two items on a width of four: each runs its own broadcast.
    pool.reset_stats();
    for out in run(2) {
        assert_eq!(out, serial);
    }
    assert_eq!(pool.stats().broadcasts, 2, "each item should use the pool");

    // Four items on a width of four: the batch is the parallel axis, the items
    // run serially, so nothing broadcasts.
    pool.reset_stats();
    for out in run(4) {
        assert_eq!(out, serial);
    }
    assert_eq!(
        pool.stats().broadcasts,
        0,
        "items run serially on their lanes"
    );
}
