//! The faer strategy's batched loop: many small items are spread over the pool
//! with each item serial, a single large GEMM enters the pool once with its
//! inner width, and a width-one executor never enters it. Results equal the
//! serial run exactly (each item is the same faer call on the same operands).
use strided_view::{StridedView, StridedViewMut};
use tcontract::api::{DType, DotGeneral, LayoutSpec, OperandSpec, Problem};
use tcontract::{Algorithm, Plan, PlanConfig};
use tcontract_exec::{Exec, Pool};

fn pool(n: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .unwrap()
}

/// `D[i,j,h] = sum_k A[i,k,h] B[k,j,h]` on compact column-major items.
fn batched(m: usize, n: usize, k: usize, items: usize) -> (Plan<f64>, Vec<f64>, Vec<f64>) {
    let spec = |d: &[usize]| {
        let mut s = vec![1isize];
        for &x in &d[..d.len() - 1] {
            s.push(s.last().unwrap() * x as isize);
        }
        OperandSpec::new(LayoutSpec::new(d, &s, 0).unwrap())
    };
    let problem = Problem::from_dot_general(
        DType::F64,
        spec(&[m, k, items]),
        spec(&[k, n, items]),
        spec(&[m, n, items]),
        &DotGeneral::new(&[1], &[0], &[2], &[2]),
    )
    .unwrap();
    let plan = Plan::<f64>::new(&problem, &PlanConfig::default()).unwrap();
    assert_eq!(plan.report().algorithm, Algorithm::Faer, "fuses copy-free");
    let a = (0..m * k * items).map(|x| (x % 13) as f64 - 6.0).collect();
    let b = (0..k * n * items).map(|x| (x % 7) as f64 * 0.5).collect();
    (plan, a, b)
}

fn run(plan: &Plan<f64>, dims: [usize; 4], a: &[f64], b: &[f64], exec: &Exec<'_>) -> Vec<f64> {
    let [m, n, k, h] = dims;
    let mut d = vec![0.0; m * n * h];
    plan.execute_into(
        exec,
        1.0,
        &StridedView::new(a, &[m, k, h], &[1, m as isize, (m * k) as isize], 0).unwrap(),
        &StridedView::new(b, &[k, n, h], &[1, k as isize, (k * n) as isize], 0).unwrap(),
        &mut StridedViewMut::new(&mut d, &[m, n, h], &[1, m as isize, (m * n) as isize], 0)
            .unwrap(),
    )
    .unwrap();
    d
}

#[test]
fn many_small_items_spread_over_the_pool_and_match_serial() {
    let dims = [8usize, 8, 8, 4096];
    let (plan, a, b) = batched(dims[0], dims[1], dims[2], dims[3]);
    let serial = run(&plan, dims, &a, &b, &Exec::serial());
    let tp = pool(4);
    let pool = Pool::borrow(&tp);
    let wide = run(&plan, dims, &a, &b, &Exec::rayon(&pool));
    assert_eq!(wide, serial);
    // The batch is the parallel axis: items are partitioned, nothing broadcasts.
    assert_eq!(pool.stats().broadcasts, 0);
}

#[test]
fn one_large_gemm_enters_the_pool_once_and_width_one_never() {
    let dims = [384usize, 384, 384, 1];
    let (plan, a, b) = batched(dims[0], dims[1], dims[2], dims[3]);
    let serial = run(&plan, dims, &a, &b, &Exec::serial());
    let tp = pool(4);
    let pool = Pool::borrow(&tp);
    let wide = run(&plan, dims, &a, &b, &Exec::rayon(&pool));
    // faer's parallel GEMM may reorder partial sums across threads, so the
    // agreement is numerical here; the entry count is the contract.
    let err = wide
        .iter()
        .zip(&serial)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    assert!(err < 1e-9, "{err}");
    assert_eq!(
        pool.stats().entries,
        1,
        "one pool entry for the whole batch"
    );
    pool.reset_stats();
    let one = run(
        &plan,
        dims,
        &a,
        &b,
        &Exec::rayon(&pool).with_budget(1).unwrap(),
    );
    assert_eq!(one, serial);
    assert_eq!(pool.stats().entries, 0);
}

/// A forced `Split` is honoured on this route, not silently discarded: at a
/// budget of four it resolves to two lanes of two item threads, every item runs
/// on the width it was granted, and the result is the sequential result exactly.
#[test]
fn a_forced_split_gives_every_item_its_item_threads() {
    use std::num::NonZeroUsize;
    use tcontract_exec::{BatchPolicy, BatchQuery, BatchStrategy};

    let dims = [8usize, 8, 8, 8];
    let (plan, a, b) = batched(dims[0], dims[1], dims[2], dims[3]);
    let tp = pool(4);
    let pool = Pool::borrow(&tp);
    let at = |strategy| {
        Exec::rayon(&pool)
            .with_budget(4)
            .unwrap()
            .with_batch_policy(BatchPolicy::new(strategy))
    };

    // The decision the route has to honour: two lanes of two item threads.
    let split = at(BatchStrategy::Split {
        lanes: NonZeroUsize::new(2).unwrap(),
    });
    let query = BatchQuery::new(8, 8.0 * 8.0 * 8.0)
        .with_item_parallel(true)
        .with_outer_only_when_item_serial(true);
    let decision = split.batch_decision(&query).unwrap();
    assert_eq!((decision.lanes, decision.item_threads), (2, 2));

    let want = run(&plan, dims, &a, &b, &at(BatchStrategy::Sequential));
    pool.reset_stats();
    let got = run(&plan, dims, &a, &b, &split);
    assert_eq!(got, want, "the split changed the result");
    let stats = pool.stats();
    assert_eq!(stats.entries, 1, "one pool entry for the whole batch");
    // One `Par::Threads(2)` grant per item, taken from the `Exec` inside its lane
    // (`PoolStats::inline_runs` counts the parallel requests a worker serves in
    // place, one per item here): the item width was honoured, not replaced by
    // `Par::Seq`.
    assert_eq!(
        stats.inline_runs, dims[3] as u64,
        "every item ran on its two item threads inside its lane"
    );

    // `Outer` grants no item threads, so no item takes a `Par::Threads` grant.
    pool.reset_stats();
    assert_eq!(
        run(&plan, dims, &a, &b, &at(BatchStrategy::Outer)),
        want,
        "Outer lanes changed the result"
    );
    assert_eq!(pool.stats().inline_runs, 0);
}
