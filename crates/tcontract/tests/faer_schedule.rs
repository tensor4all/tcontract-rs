//! The faer strategy's batched loop: many small items are spread over the pool
//! with each item serial; a single large GEMM runs on a team when the token
//! carries one and as `Par::Seq` on the calling thread otherwise (faer has no
//! rayon route here), and a width-one executor never enters the pool. A plain
//! token's results equal the serial run exactly (each item is the same serial
//! faer call on the same operands).
use strided_view::{StridedView, StridedViewMut};
use tcontract::api::{DType, DotGeneral, LayoutSpec, OperandSpec, Problem};
use tcontract::{Algorithm, Plan, PlanConfig};
use tcontract_exec::{Exec, Pool};

mod common;
use common::placement::{installs, max_live, pool, pool_with_reference_costs, root};

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

fn run(plan: &Plan<f64>, dims: [usize; 4], a: &[f64], b: &[f64], exec: &mut Exec<'_>) -> Vec<f64> {
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
    let serial = run(&plan, dims, &a, &b, &mut Exec::serial());
    let sp = pool_with_reference_costs(4);
    let pool = Pool::borrow(&sp);
    let mut exec = Exec::new(&pool, pool.size()).unwrap();
    let root = root(&mut exec);
    let wide = run(&plan, dims, &a, &b, &mut exec);
    assert_eq!(wide, serial);
    // The batch is the parallel axis: items are partitioned across the lanes,
    // which entered the pool once and never exceeded the budget.
    assert_eq!(installs(root), 1);
    assert!(max_live(root) <= pool.size());
}

#[test]
fn one_large_gemm_on_a_plain_token_runs_serially_and_never_enters_the_pool() {
    let dims = [384usize, 384, 384, 1];
    let (plan, a, b) = batched(dims[0], dims[1], dims[2], dims[3]);
    let serial = run(&plan, dims, &a, &b, &mut Exec::serial());
    let sp = pool(4);
    let pool = Pool::borrow(&sp);
    let mut exec = Exec::new(&pool, pool.size()).unwrap();
    let root = root(&mut exec);
    // Only a team gives faer internal parallelism: a plain token is `Par::Seq`,
    // so the result is the serial one exactly and no entry is made.
    let wide = run(&plan, dims, &a, &b, &mut exec);
    assert_eq!(wide, serial);
    assert_eq!(
        installs(root),
        0,
        "a plain token never enters the pool for faer"
    );
    let mut single = Exec::new(&pool, 1).unwrap();
    let single_root = common::placement::root(&mut single);
    let one = run(&plan, dims, &a, &b, &mut single);
    assert_eq!(one, serial);
    assert_eq!(installs(single_root), 0);
}

/// A plain token (a pool exists, the caller is outside it) runs the GEMM on the
/// calling thread as `Par::Seq` and never enters the pool, whatever the size.
/// Timing is not asserted; only the entry count and the (serial, hence exact)
/// result.
#[test]
fn a_plain_token_never_enters_the_pool() {
    for n in [32usize, 48, 64, 96] {
        let dims = [n, n, n, 1];
        let (plan, a, b) = batched(dims[0], dims[1], dims[2], dims[3]);
        let serial = run(&plan, dims, &a, &b, &mut Exec::serial());
        let sp = pool_with_reference_costs(4);
        let pool = Pool::borrow(&sp);
        let mut exec = Exec::new(&pool, pool.size()).unwrap();
        let root = root(&mut exec);
        for _ in 0..3 {
            assert_eq!(run(&plan, dims, &a, &b, &mut exec), serial, "n = {n}");
        }
        assert_eq!(installs(root), 0, "n = {n}: faer is serial, no pool entry");
        assert_eq!(max_live(root), 0, "n = {n}: no task was spawned");
    }
}

/// A token that carries a team hands faer a `Par::Team`: the GEMM runs as team
/// sections on the reserved members, so it makes no entry into the pool (neither
/// does a plain token, which runs `Par::Seq`), and the result agrees with the serial
/// one numerically.
#[test]
fn a_team_token_runs_faer_on_the_team_without_a_pool_entry() {
    use tcontract_exec::BatchPolicy;

    let dims = [384usize, 384, 384, 1];
    let (plan, a, b) = batched(dims[0], dims[1], dims[2], dims[3]);
    let serial = run(&plan, dims, &a, &b, &mut Exec::serial());
    let sp = threadteam::Pool::builder()
        .threads(4)
        .acquire_timeout(std::time::Duration::from_secs(10))
        .build()
        .expect("pool");
    let Ok(team) = sp.into_team(4) else {
        // The host could not reserve the members; the plain route is covered above.
        return;
    };
    let token = team.exec(4);
    assert!(token.carries_team());
    let root = token.ledger_root().expect("a team token is a ledger root");
    let mut exec = Exec::from_token(token, None, BatchPolicy::default());
    let wide = run(&plan, dims, &a, &b, &mut exec);
    let err = wide
        .iter()
        .zip(&serial)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    assert!(err < 1e-9, "{err}");
    assert_eq!(installs(root), 0, "team sections make no pool entry");
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
    let sp = pool(4);
    let pool = Pool::borrow(&sp);
    let at = |strategy| {
        Exec::new(&pool, 4)
            .unwrap()
            .with_batch_policy(BatchPolicy::new(strategy))
    };

    // The decision the route has to honour: two lanes of two item threads.
    let mut split = at(BatchStrategy::Split {
        lanes: NonZeroUsize::new(2).unwrap(),
    });
    let query = BatchQuery::new(8, 8.0 * 8.0 * 8.0)
        .with_item_parallel(true)
        .with_outer_only_when_item_serial(true);
    let decision = split.batch_decision(&query);
    assert_eq!((decision.lanes, decision.item_threads), (2, 2));

    let want = run(&plan, dims, &a, &b, &mut at(BatchStrategy::Sequential));
    let split_root = root(&mut split);
    let got = run(&plan, dims, &a, &b, &mut split);
    assert_eq!(got, want, "the split changed the result");
    assert_eq!(
        installs(split_root),
        1,
        "one pool entry for the whole batch"
    );
    // At most two lanes ran at once: each lane is a fork job holding its share
    // (two slots) of the four, and hands that share to its items as the width of
    // the provider call. (A lane stops counting while it is inside the provider
    // call, so the peak is a bound here, not an equality.)
    assert!(max_live(split_root) <= 2);

    // `Outer` grants no item threads: four lanes of one slot.
    let mut outer = at(BatchStrategy::Outer);
    let outer_root = root(&mut outer);
    assert_eq!(
        run(&plan, dims, &a, &b, &mut outer),
        want,
        "Outer lanes changed the result"
    );
    assert!(max_live(outer_root) <= 4);
}

/// The same split on items large enough to leave faer's serial nano-GEMM path,
/// on a team: each lane is a child team token, so the item's `Par::Team` reaches
/// the provider's parallel GEMM. One 128^3 item is 2^21 multiply-adds. Bounded
/// completion, because a lane that cannot nest would hang at a team barrier
/// instead of failing.
#[test]
fn a_split_on_large_items_nests_team_parallelism_and_matches_sequential() {
    use std::num::NonZeroUsize;
    use std::sync::mpsc;
    use std::time::Duration;
    use tcontract_exec::{BatchPolicy, BatchStrategy};

    let dims = [128usize, 128, 128, 4];
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let (plan, a, b) = batched(dims[0], dims[1], dims[2], dims[3]);
        let want = run(&plan, dims, &a, &b, &mut Exec::serial());
        let sp = threadteam::Pool::builder()
            .threads(4)
            .acquire_timeout(Duration::from_secs(10))
            .build()
            .expect("pool");
        let Ok(team) = sp.into_team(4) else {
            // The host could not reserve the members.
            tx.send(None).unwrap();
            return;
        };
        let mut split = Exec::from_token(
            team.exec(4),
            None,
            BatchPolicy::new(BatchStrategy::Split {
                lanes: NonZeroUsize::new(2).unwrap(),
            }),
        );
        let got = run(&plan, dims, &a, &b, &mut split);
        let err = got
            .iter()
            .zip(&want)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
        tx.send(Some(err)).unwrap();
    });
    let err = rx
        .recv_timeout(Duration::from_secs(60))
        .expect("a nested team GEMM deadlocked");
    if let Some(err) = err {
        assert!(err < 1e-9, "the split changed the result: {err}");
    }
}
