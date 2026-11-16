//! `Plan::execution_width`: the width a resolved plan takes on a lane narrower
//! than its context, for a caller that splits one engine between independent
//! plans (tcontract#176).
//!
//! The three tests answer the three claims the query makes: it agrees with
//! `execution_route` on the same narrowed budget for the packed driver, it
//! follows the placement a faer call takes (a team token, a plain token, a
//! single item), and it reports `None` where no width is the plan's to decide.

use std::num::NonZeroUsize;

use strided_view::{StridedView, StridedViewMut};
use tcontract::api::{DType, DotGeneral, Labels, LayoutSpec, OperandSpec, Problem};
use tcontract::{ExecutionRoute, OutputContract, Plan, PlanConfig};
use tcontract_exec::{Exec, Pool};

mod common;
use common::placement::{max_live, pool_with_reference_costs, root};
use common::plans::{matmul_problem, packed_plan};

const INIT: OutputContract = OutputContract::Initialized;

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).expect("a lane has at least one slot")
}

/// `D[i,j] = A[i,j] * B[i,j]`: every operand carries every output axis, so no
/// contracted axis is left and the plan is elementwise.
fn hadamard(m: usize, n: usize) -> Problem {
    let spec = |d: &[usize]| OperandSpec::new(LayoutSpec::new(d, &[1, m as isize], 0).unwrap());
    Problem::from_labels(
        DType::F64,
        spec(&[m, n]),
        spec(&[m, n]),
        tcontract::api::CSpec::Absent,
        spec(&[m, n]),
        &Labels::new(&[0, 1], &[0, 1], &[0, 1]),
    )
    .unwrap()
}

/// `D[h] = sum_k A[h,k] B[h,k]`: `M` and `N` are both one, so the plan is the
/// batched inner product rather than a GEMM.
fn batched_dot(h: usize, k: usize) -> Problem {
    let spec = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
    let problem = Problem::from_dot_general(
        DType::F64,
        spec(&[h, k], &[1, h as isize]),
        spec(&[h, k], &[1, h as isize]),
        spec(&[h], &[1]),
        &DotGeneral::new(&[1], &[1], &[0], &[0]),
    )
    .unwrap();
    assert_eq!(problem.roles().m().len(), 0, "no output row axis");
    problem
}

/// `D[i,j,h] = sum_k A[i,k,h] B[k,j,h]` on compact column-major items, the shape
/// `faer_schedule.rs` uses: one faer GEMM per batch item.
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
    let a = (0..m * k * items).map(|x| (x % 13) as f64 - 6.0).collect();
    let b = (0..k * n * items).map(|x| (x % 7) as f64 * 0.5).collect();
    (plan, a, b)
}

/// One execution of a [`batched`] plan.
fn run(plan: &Plan<f64>, dims: [usize; 4], a: &[f64], b: &[f64], exec: &mut Exec<'_>) {
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
}

/// The packed width is the width `execution_route` reports for the same lane
/// budget, and it never exceeds the lane: the query is the route query with the
/// context narrowed, so a scheduler can score a candidate lane without one.
#[test]
fn the_packed_width_is_the_route_width_on_the_same_lane() {
    // A small call that pays no second participant, and a large one that scales
    // with the lane: both report the route's own width at every budget.
    for (m, n, k) in [(64usize, 64, 64), (512, 512, 512)] {
        let plan = packed_plan(m, n, k);
        let sp = pool_with_reference_costs(8);
        let pool = Pool::borrow(&sp);
        let mut exec = Exec::new(&pool, 8).unwrap();
        for budget in [1usize, 4, 8] {
            let width = plan
                .execution_width(&exec, budget, 1.0, INIT)
                .unwrap()
                .unwrap_or_else(|| panic!("a packed plan reports a width ({m}x{n}x{k}, {budget})"));
            assert!(
                width.get() <= budget,
                "{m}x{n}x{k}: {width} participants on a {budget}-slot lane"
            );
            let route = plan
                .execution_route(&exec.reborrow().limit(nz(budget)), 1.0, INIT)
                .unwrap();
            let ExecutionRoute::Packed {
                width: route_width, ..
            } = route
            else {
                panic!("a packed plan takes the packed route at {budget}: {route:?}")
            };
            assert_eq!(
                width.get(),
                route_width,
                "{m}x{n}x{k} at {budget}: the two queries disagree about one lane"
            );
            if budget == 1 {
                assert_eq!(width.get(), 1, "{m}x{n}x{k}: one slot is one participant");
            }
        }
        // A candidate lane wider than the context is the context itself.
        assert_eq!(
            plan.execution_width(&exec, 96, 1.0, INIT).unwrap(),
            plan.execution_width(&exec, 8, 1.0, INIT).unwrap(),
            "{m}x{n}x{k}: the query clamps to the context's budget"
        );
    }
}

/// The faer width follows the placement the call takes: one participant on a
/// serial context and for a single item on a plain token (faer runs `Par::Seq`
/// there), the lanes the batch decision forks for many small items, and never
/// more than the lane it is asked about.
#[test]
fn the_faer_width_follows_the_placement_the_call_takes() {
    let serial = Exec::serial();
    for dims in [[512usize, 512, 512, 1], [8, 8, 8, 4096]] {
        let itemwise = dims[3] > 1;
        let (plan, a, b) = batched(dims[0], dims[1], dims[2], dims[3]);
        for budget in [1usize, 4, 8] {
            assert_eq!(
                plan.execution_width(&serial, budget, 1.0, INIT).unwrap(),
                NonZeroUsize::new(1),
                "{dims:?}: a serial context has one participant"
            );
        }

        let sp = pool_with_reference_costs(8);
        let pool = Pool::borrow(&sp);
        let mut exec = Exec::new(&pool, 8).unwrap();
        for budget in [1usize, 4, 8] {
            let width = plan
                .execution_width(&exec, budget, 1.0, INIT)
                .unwrap()
                .unwrap_or_else(|| panic!("a faer plan reports a width ({dims:?}, {budget})"));
            assert!(
                width.get() <= budget,
                "{dims:?}: {width} participants on a {budget}-slot lane"
            );
            if !itemwise {
                // One item cannot split into lanes, and a plain token carries no
                // team, so faer gets `Par::Seq` on the calling thread.
                assert_eq!(
                    width.get(),
                    1,
                    "{dims:?} at {budget}: a single item on a plain token is serial"
                );
            }
        }
        if itemwise {
            // The batch axis is the parallel axis for many small items: the
            // same shape and cost table `faer_schedule.rs` measures.
            let four = plan.execution_width(&exec, 4, 1.0, INIT).unwrap().unwrap();
            assert!(four.get() > 1, "{dims:?}: small items fork into lanes");
            // Nothing runs wider than the reported width.
            let eight = plan.execution_width(&exec, 8, 1.0, INIT).unwrap().unwrap();
            assert!(eight.get() > 1, "{dims:?}: small items fork into lanes");
            let ledger = root(&mut exec);
            run(&plan, dims, &a, &b, &mut exec);
            assert!(
                max_live(ledger) <= eight.get(),
                "{dims:?}: {} live exceeds the reported {eight}",
                max_live(ledger)
            );
        }
    }
}

/// The routes whose width is not the plan's to decide report `None`, on the same
/// context where a packed plan reports a width: `None` means "reserve what you
/// ask for", not "serial".
#[test]
fn routes_that_do_not_decide_a_width_report_none() {
    let sp = pool_with_reference_costs(4);
    let pool = Pool::borrow(&sp);
    let exec = Exec::new(&pool, 4).unwrap();

    // No output elements: nothing runs.
    let empty = Plan::<f64>::new(&matmul_problem(0, 4, 4), &PlanConfig::default()).unwrap();
    assert_eq!(empty.execution_width(&exec, 4, 1.0, INIT).unwrap(), None);

    // Alpha zero takes the output-only pass, on a plan that otherwise reports one.
    let packed = packed_plan(64, 64, 64);
    assert_eq!(packed.execution_width(&exec, 4, 0.0, INIT).unwrap(), None);
    assert!(packed
        .execution_width(&exec, 4, 1.0, INIT)
        .unwrap()
        .is_some());

    // Elementwise and the batched inner product hand the token to strided, which
    // splits by its own loop-length rule.
    let elementwise = Plan::<f64>::new(&hadamard(64, 64), &PlanConfig::default()).unwrap();
    assert_eq!(
        elementwise.execution_width(&exec, 4, 1.0, INIT).unwrap(),
        None
    );
    let inner = Plan::<f64>::new(&batched_dot(64, 4), &PlanConfig::default()).unwrap();
    assert_eq!(inner.execution_width(&exec, 4, 1.0, INIT).unwrap(), None);
}

/// The faer width is measured on the same budget `execute` leaves faer, not on
/// the whole lane: `execute` narrows the context to the plan's own stage-one
/// budget first (`Plan::batch_context`), so a plan that then runs serially must
/// not be reported at a lane width. With an item overhead in the model and 64
/// items, the wider lanes are never woken — the ledger says so — and the query
/// has to agree (tcontract#176 review).
#[test]
fn the_faer_width_applies_the_stage_one_narrowing_execution_takes() {
    use tcontract_exec::batch::{BatchPolicy, WorkModel};

    let dims = [8usize, 8, 8, 64];
    let policy =
        BatchPolicy::default().with_model(WorkModel::default().with_item_overhead_ns(2000.0));
    let sp = pool_with_reference_costs(8);
    let pool = Pool::borrow(&sp);
    for budget in [1usize, 4, 8] {
        let (plan, a, b) = batched(dims[0], dims[1], dims[2], dims[3]);
        // The lane the query is asked about, as the context the call would get.
        let mut exec = Exec::new(&pool, 8)
            .unwrap()
            .with_batch_policy(policy)
            .limit(nz(budget));
        let width = plan
            .execution_width(&exec, budget, 1.0, INIT)
            .unwrap()
            .unwrap_or_else(|| panic!("a faer plan reports a width ({dims:?}, {budget})"));
        let ledger = root(&mut exec);
        run(&plan, dims, &a, &b, &mut exec);
        let live = max_live(ledger).max(1);
        assert_eq!(
            width.get(),
            live,
            "{dims:?} at {budget}: the query reports {width}, the call occupied {live}"
        );
    }
}
