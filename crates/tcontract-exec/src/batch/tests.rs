use super::*;

fn budget(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

/// Dispatch cost of the test rule: 500 ns per participant, so wider teams cost
/// more and a batch does not always fill the budget.
fn test_cost_ns(width: usize) -> u64 {
    500 * width as u64
}

/// The test width rule: `threadteam`'s own pure rule over a flat cost.
fn test_width(serial_ns: f64, sections: u32, budget: usize) -> usize {
    threadteam::width_for_by(
        core::time::Duration::from_nanos(serial_ns.max(0.0) as u64),
        sections,
        NonZeroUsize::new(budget.max(1)).unwrap(),
        |w| core::time::Duration::from_nanos(test_cost_ns(w.get())),
    )
    .get()
}

/// `batch_decision` over the test rule, on `budget` slots.
fn bd(policy: &BatchPolicy, budget: NonZeroUsize, query: &BatchQuery) -> BatchDecision {
    let b = budget.get();
    decide(
        policy,
        b,
        &|ns, sections| test_width(ns, sections, b),
        query,
    )
}

/// Independent copy of the legacy packed computation.
///
/// `plan/mod.rs:641` narrows the `Exec` with `width_for(total_ns)`, then
/// `driver/batch.rs:38` recomputes the width against that narrowed budget
/// and applies the tiny/balanced rule. Returns `(lanes, item_threads)` where
/// `item_threads` is the narrowed team width the route uses with one lane.
fn legacy_packed(items: usize, item_muladds: f64, host_budget: usize) -> (usize, usize) {
    // `ns_per_flop()` applies to flops, and the packed driver counts two flops
    // per multiply-add.
    let item_ns = item_muladds / crate::host::muladds_per_ns();
    let total_ns = item_ns * items as f64;
    let narrowed = legacy_width_for(total_ns, host_budget);
    if items < 2 || narrowed < 2 {
        return (1, narrowed);
    }
    let width = legacy_width_for(total_ns, narrowed);
    if width < 2 {
        return (1, narrowed);
    }
    let tiny = legacy_width_for(item_ns, host_budget) < 2;
    let lanes = items.min(width);
    let balanced = 10 * items >= 9 * lanes * items.div_ceil(lanes);
    let lanes = if tiny || (items >= narrowed && balanced) {
        lanes
    } else {
        1
    };
    (lanes, if lanes > 1 { 1 } else { narrowed })
}

/// The test rule, written out: the least objective among widths that pay.
fn legacy_width_for(serial_ns: f64, budget: usize) -> usize {
    let mut best = (1usize, serial_ns);
    for k in 2..=budget {
        let d = test_cost_ns(k) as f64;
        if serial_ns >= threadteam::PAYS * d {
            let obj = d + serial_ns / k as f64;
            if obj < best.1 {
                best = (k, obj);
            }
        }
    }
    best.0
}

#[test]
fn auto_matches_legacy_packed_table() {
    let policy = BatchPolicy::default();
    let items = [0usize, 1, 2, 3, 7, 8, 9, 16, 64, 512];
    let muladds = [
        1.0,
        64.0,
        4_096.0,
        32_768.0,
        262_144.0,
        1_000_000.0,
        8_000_000.0,
    ];
    let budgets = [1usize, 2, 3, 4, 8, 16];
    for &h in &items {
        for &m in &muladds {
            for &b in &budgets {
                let query = BatchQuery::new(h, m).with_balanced_lanes(true);
                let got = bd(&policy, budget(b), &query);
                let want = legacy_packed(h, m, b);
                assert_eq!(
                    (got.lanes, got.item_threads),
                    (want.0, want.1),
                    "items={h} item_muladds={m} budget={b}"
                );
            }
        }
    }
}

#[test]
fn generic_auto_matches_tenferro_auto_lanes_table() {
    // tenferro's `CpuBatchThresholds` defaults: 50 ns per item, 16
    // multiply-adds per ns, 8 us per lane. This is the table at
    // tenferro-rs `dot_runtime/tests.rs:3025` (`auto_lane_count_needs_enough_work_per_lane`).
    let model = WorkModel::default()
        .with_item_overhead_ns(50.0)
        .with_muladds_per_ns(16.0)
        .with_min_work_per_lane_ns(8_000.0);
    let policy = BatchPolicy::new(BatchStrategy::Auto).with_model(model);
    let lanes = |item_muladds: f64, batch: usize, threads: usize| {
        bd(
            &policy,
            budget(threads),
            &BatchQuery::new(batch, item_muladds),
        )
        .lanes
    };
    // 4^3 = 64 multiply-adds: per-call overhead dominates, so many items
    // fill a lane.
    assert_eq!(lanes(64.0, 1024, 4), 4);
    assert_eq!(lanes(64.0, 64, 4), 1);
    // Larger items need fewer per lane (16^3 is 306 ns, so 27 per lane);
    // lanes never exceed the threads.
    assert_eq!(lanes(32_768.0, 8, 16), 2);
    assert_eq!(lanes(4_096.0, 128, 8), 4);
    // One lane of work is no fan-out.
    assert_eq!(lanes(4_096.0, 32, 8), 1);
    // Two items of tiny work still pass the item gate but not the work
    // gate: zero minimum work lets the item thresholds decide alone.
    let no_work_gate =
        BatchPolicy::new(BatchStrategy::Auto).with_model(model.with_min_work_per_lane_ns(0.0));
    assert_eq!(
        bd(&no_work_gate, budget(4), &BatchQuery::new(64, 64.0)).lanes,
        4
    );
    assert_eq!(
        bd(&no_work_gate, budget(4), &BatchQuery::new(1, 64.0)).lanes,
        1
    );
}

/// Independent copy of the pre-#123 faer schedule.
///
/// `strategy/faer.rs` computed `inner = width_for(item_ns)` and
/// `total = width_for(item_ns * count)` and picked outer lanes only when
/// `inner == 1 && total > 1 && count > 1`. Returns `(lanes, item_threads)`.
fn legacy_faer(count: usize, item_muladds: f64, budget: usize) -> (usize, usize) {
    let item_ns = item_muladds / crate::host::muladds_per_ns();
    let total_ns = item_ns * count as f64;
    let inner = legacy_width_for(item_ns, budget);
    let total = legacy_width_for(total_ns, budget);
    if inner == 1 && total > 1 && count > 1 {
        (total.min(count), 1)
    } else {
        (1, inner)
    }
}

#[test]
fn auto_matches_legacy_faer_table() {
    let policy = BatchPolicy::default();
    let items = [1usize, 2, 3, 8, 16, 64, 512];
    let muladds = [
        1.0,
        64.0,
        4_096.0,
        32_768.0,
        262_144.0,
        1_000_000.0,
        8_000_000.0,
    ];
    let budgets = [1usize, 2, 3, 4, 8, 16];
    for &count in &items {
        for &m in &muladds {
            for &b in &budgets {
                let query = BatchQuery::new(count, m).with_outer_only_when_item_serial(true);
                let got = bd(&policy, budget(b), &query);
                let want = legacy_faer(count, m, b);
                assert_eq!(
                    (got.lanes, got.item_threads),
                    (want.0, want.1),
                    "count={count} item_muladds={m} budget={b}"
                );
            }
        }
    }
}

#[test]
fn faer_auto_keeps_one_item_on_the_team() {
    // A large item: the inner route, with the width of one item.
    let query = BatchQuery::new(4, 1_000_000.0).with_outer_only_when_item_serial(true);
    let d = bd(&BatchPolicy::default(), budget(8), &query);
    assert!(d.item_threads > 1, "{d:?}");
    assert_eq!(d.lanes, 1);
}

#[test]
fn exec_carries_the_policy_and_budget_keeps_it() {
    let sp = threadteam::Pool::builder().threads(4).build().unwrap();
    let pool = crate::Pool::borrow(&sp);
    let exec = crate::Exec::new(&pool, 4)
        .unwrap()
        .with_batch_policy(BatchPolicy::new(BatchStrategy::Outer));
    assert_eq!(exec.batch_policy().strategy, BatchStrategy::Outer);
    let narrowed = exec.limit(2.try_into().unwrap());
    assert_eq!(narrowed.budget(), 2);
    assert_eq!(narrowed.batch_policy().strategy, BatchStrategy::Outer);
    assert_eq!(
        crate::Exec::serial().batch_policy().strategy,
        BatchStrategy::Auto
    );
}

#[test]
fn forced_strategies_are_clamped_not_fallen_back() {
    let query = BatchQuery::new(3, 1_000.0);
    // Sequential ignores the budget and the item count.
    assert_eq!(
        bd(
            &BatchPolicy::new(BatchStrategy::Sequential),
            budget(8),
            &query
        ),
        BatchDecision {
            lanes: 1,
            item_threads: 1
        }
    );
    // Outer uses min(items, budget), each item serial.
    assert_eq!(
        bd(&BatchPolicy::new(BatchStrategy::Outer), budget(8), &query),
        BatchDecision {
            lanes: 3,
            item_threads: 1
        }
    );
    // Inner is one lane with the whole budget.
    assert_eq!(
        bd(&BatchPolicy::new(BatchStrategy::Inner), budget(8), &query),
        BatchDecision {
            lanes: 1,
            item_threads: 8
        }
    );
    // Split is clamped to the items and the budget, then shares the rest.
    assert_eq!(
        bd(
            &BatchPolicy::new(BatchStrategy::Split { lanes: budget(8) }),
            budget(8),
            &query
        ),
        BatchDecision {
            lanes: 3,
            item_threads: 2
        }
    );
    // A split wider than the budget clamps to the budget.
    let many = BatchQuery::new(64, 1_000.0);
    assert_eq!(
        bd(
            &BatchPolicy::new(BatchStrategy::Split { lanes: budget(64) }),
            budget(4),
            &many
        ),
        BatchDecision {
            lanes: 4,
            item_threads: 1
        }
    );
    // Empty items clamp to one lane, never zero.
    let empty = BatchQuery::new(0, 1.0);
    assert_eq!(
        bd(&BatchPolicy::new(BatchStrategy::Outer), budget(4), &empty),
        BatchDecision {
            lanes: 1,
            item_threads: 1
        }
    );
    assert_eq!(
        bd(
            &BatchPolicy::new(BatchStrategy::Split { lanes: budget(2) }),
            budget(4),
            &empty
        ),
        BatchDecision {
            lanes: 1,
            item_threads: 4
        }
    );
}

#[test]
fn every_decision_fits_the_budget() {
    // A route that honours a decision runs `lanes` lanes of `item_threads`
    // item threads (faer does), so the split must never exceed the budget the
    // route was given: that is what keeps the call inside the host's pool.
    let budgets = [1usize, 2, 3, 4, 8, 16];
    let items = [0usize, 1, 2, 3, 7, 8, 16, 64];
    let policies = [
        BatchStrategy::Auto,
        BatchStrategy::Sequential,
        BatchStrategy::Outer,
        BatchStrategy::Inner,
        BatchStrategy::Split { lanes: budget(1) },
        BatchStrategy::Split { lanes: budget(2) },
        BatchStrategy::Split { lanes: budget(8) },
    ];
    for &b in &budgets {
        for &n in &items {
            for strategy in policies {
                let d = bd(
                    &BatchPolicy::new(strategy),
                    budget(b),
                    &BatchQuery::new(n, 512.0),
                );
                assert!(
                    d.lanes >= 1 && d.lanes * d.item_threads <= b,
                    "strategy {strategy:?}, budget {b}, items {n}: {d:?}"
                );
            }
        }
    }
}

#[test]
fn forced_strategies_use_the_original_budget() {
    // Auto narrows the whole batch to five lanes (the width its total work
    // pays for), but Outer is evaluated against the original host budget of eight.
    // The item is 40 000 multiply-adds at the 27 per ns of the `generic` row (about 1.5 us),
    // scaled to this host class's rate so that its serial time does not depend on the class.
    let query = BatchQuery::new(9, 40_000.0 * crate::host::muladds_per_ns() / 27.0)
        .with_balanced_lanes(true);
    let auto = bd(&BatchPolicy::default(), budget(8), &query);
    assert_eq!((auto.lanes, auto.item_threads), (5, 1));
    let outer = bd(&BatchPolicy::new(BatchStrategy::Outer), budget(8), &query);
    assert_eq!((outer.lanes, outer.item_threads), (8, 1));
}

#[test]
fn item_parallel_capability_keeps_item_threads_one() {
    // A moderate single item widens the narrow route to the team budget; a
    // route that forbids item parallelism keeps one thread.
    let query = BatchQuery::new(1, 1_000_000.0);
    let allowed = bd(&BatchPolicy::default(), budget(8), &query);
    assert!(allowed.item_threads > 1, "{allowed:?}");
    let forbidden = query.with_item_parallel(false);
    let d = bd(&BatchPolicy::default(), budget(8), &forbidden);
    assert_eq!((d.lanes, d.item_threads), (1, 1));
}

#[test]
fn work_gate_reduces_lanes_and_disabling_it_does_not() {
    let query = BatchQuery::new(8, 8_000_000.0).with_balanced_lanes(true);
    let disabled = WorkModel::default();
    let enabled = WorkModel::default().with_min_work_per_lane_ns(1_000_000.0);
    let d0 = bd(
        &BatchPolicy::new(BatchStrategy::Auto).with_model(disabled),
        budget(8),
        &query,
    );
    let d1 = bd(
        &BatchPolicy::new(BatchStrategy::Auto).with_model(enabled),
        budget(8),
        &query,
    );
    assert!(d0.lanes > d1.lanes, "{d0:?} vs {d1:?}");
}

#[test]
fn balanced_lanes_can_hold_a_lane_back() {
    // Nine items on eight lanes leave one lane with two: the packed rule
    // keeps the team instead of cutting an unbalanced batch.
    let query = BatchQuery::new(9, 1_000_000.0).with_item_parallel(true);
    let packed = bd(
        &BatchPolicy::new(BatchStrategy::Auto).with_model(WorkModel::default()),
        budget(8),
        &query.with_balanced_lanes(true),
    );
    let generic = bd(
        &BatchPolicy::new(BatchStrategy::Auto).with_model(WorkModel::default()),
        budget(8),
        &query.with_balanced_lanes(false),
    );
    assert_eq!(packed.lanes, 1);
    assert!(generic.lanes > 1, "{generic:?}");
}
