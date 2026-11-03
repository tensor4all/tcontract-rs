use super::*;
use crate::plan::Axis;

/// The measured case's m role: `A` is `(a s1, b s48, j s1920, c s92160)` and the
/// output is `(c s1, b s48, n s1920, a s76800)`.
fn measured() -> Vec<Role> {
    vec![
        Role {
            extent: 48,
            src: 1,
            dst: 76800,
        },
        Role {
            extent: 40,
            src: 48,
            dst: 48,
        },
        Role {
            extent: 48,
            src: 92160,
            dst: 1,
        },
    ]
}

/// The same case as the plan reports it, for the eligibility test.
fn stats() -> crate::plan::PlanStats {
    let ax = |extent: usize, sa: i64, sc: i64, sd: i64| Axis {
        extent: extent as i64,
        sa,
        sb: 0,
        sc,
        sd,
    };
    crate::plan::PlanStats {
        m: 92160,
        n: 40,
        k: 48,
        batch: 1,
        m_axes: vec![
            ax(48, 1, 1, 76800),
            ax(40, 48, 48, 48),
            ax(48, 92160, 76800, 1),
        ],
        n_axes: vec![ax(40, 0, 1920, 1920)],
        k_axes: vec![ax(48, 1920, 0, 0)],
        h_axes: vec![],
        is_pure_gemm: false,
    }
}

/// The output offsets of the same case for a block's rows, decoded from the
/// plan's mixed-radix numbering (a + 48 b + 1920 c).
fn out_offsets(rows: &[i64]) -> Vec<i64> {
    rows.iter()
        .map(|&r| (r % 48) * 76800 + (r / 48) % 40 * 48 + (r / 1920))
        .collect()
}

#[test]
fn the_measured_case_takes_a_line_of_a_and_a_whole_c() {
    let axes = measured();
    let shape = eligible(&axes, 8).expect("eligible");
    assert_eq!(shape.span, 2, "the output's fastest axis is c");
    assert_eq!(shape.line, 0, "the operand's fastest axis is a");
    assert_eq!(shape.line_len, 8, "one 64 B line of f64");
    let rows = rows(&axes, shape, &[0, 0, 0]);
    assert_eq!(rows.len(), 8 * 48);
    // Consecutive rows are consecutive a: the source run the pack reads.
    assert_eq!(rows[1] - rows[0], 1);
    // And every a comes with all 48 c's, whose output offsets are consecutive:
    // the destination run the emitter writes. This is the property the
    // flattened slice could not provide and the whole design rests on.
    let offs = out_offsets(&rows);
    for a in 0..8i64 {
        let mut run: Vec<i64> = rows
            .iter()
            .zip(&offs)
            .filter(|(r, _)| **r % 1920 == a)
            .map(|(_, o)| *o)
            .collect();
        run.sort_unstable();
        assert_eq!(run.len(), 48, "each a carries all 48 c's");
        assert_eq!(
            run.last().unwrap() - run.first().unwrap(),
            47,
            "consecutive"
        );
    }
}

#[test]
fn a_role_whose_fastest_axis_agrees_stays_on_todays_path() {
    let axes = measured();
    let same = vec![
        Role {
            extent: 48,
            src: 1,
            dst: 1,
        },
        Role {
            extent: 48,
            src: 92160,
            dst: 76800,
        },
    ];
    assert!(eligible(&same, 8).is_none(), "same fastest axis");
    assert!(eligible(&axes[..1], 8).is_none(), "one live axis");
}

#[test]
fn the_incremental_origins_match_the_materialized_ones() {
    // The driver walks the origins without allocating; the materialized list is the
    // reference the coverage gate checks, so they must agree exactly.
    let s = stats();
    let mut buf = [Role::EMPTY; MAX_AXES];
    let axes = role_into(&s, false, &mut buf);
    let shape = eligible(axes, 8).expect("the measured case is eligible");
    let want = block_origins(axes, shape);
    assert!(want.len() > 1, "more than one block");
    let mut next = Origins::new();
    let mut got = Vec::new();
    while let Some(o) = next.next(axes, shape) {
        got.push(o.to_vec());
    }
    assert_eq!(got, want);
}

#[test]
fn eligibility_asks_for_a_profitable_shape() {
    let s = stats();
    // The measured class: a thin output against a deeper contraction (`k > n`), which
    // is where the block's contiguous pack pays for its permutation and metadata.
    assert!(blocked_eligibility(&s, false, 48, 40, 256, 1536, 8, 480, 24).is_some());
    // A fat output: neutral at best over the corpus and losing on some cases.
    assert!(blocked_eligibility(&s, false, 48, 128, 256, 1536, 8, 480, 24).is_none());
    // No deeper than the output: nothing for the block to buy.
    assert!(blocked_eligibility(&s, false, 40, 40, 256, 1536, 8, 480, 24).is_none());
    assert!(blocked_eligibility(&s, false, 20, 40, 256, 1536, 8, 480, 24).is_none());
}

#[test]
fn eligibility_rejects_a_block_that_does_not_fit_after_tile_rounding() {
    let mut s = stats();
    // A span wider than the row budget: not even one line's worth fits.
    s.m_axes[2].extent = 481;
    assert!(blocked_eligibility(&s, false, 2, 1, 256, 1536, 8, 480, 24).is_none());
    // 60 x 8 = 480 rows fit exactly, but the panel, the tile grid and the row map are
    // sized from the block *rounded up to whole `MR` tiles*, so `MR = 25` does not.
    s.m_axes[2].extent = 60;
    assert!(blocked_eligibility(&s, false, 2, 1, 256, 1536, 8, 480, 24).is_some());
    assert!(blocked_eligibility(&s, false, 2, 1, 256, 1536, 8, 480, 25).is_none());
}

#[test]
fn eligibility_needs_one_k_slab_and_one_nc_panel() {
    let s = stats();
    // The measured case: k = 48 <= kc = 256 and n = 40 <= nc = 1536.
    assert!(blocked_eligibility(&s, false, 48, 40, 256, 1536, 8, 480, 24).is_some());
    assert!(
        blocked_eligibility(&s, false, 48, 40, 32, 1536, 8, 480, 24).is_none(),
        "two K slabs"
    );
    assert!(
        blocked_eligibility(&s, false, 48, 40, 256, 16, 8, 480, 24).is_none(),
        "two NC panels"
    );
    // The n role is a single axis, so a swapped plan has nothing to block.
    assert!(blocked_eligibility(&s, true, 48, 40, 256, 1536, 8, 480, 24).is_none());
}

/// The origins tile the role exactly once: every row of every operand appears
/// in exactly one block, which is the coverage the driver's gates need.
#[test]
fn the_blocks_tile_the_role_once() {
    let axes = measured();
    let shape = eligible(&axes, 8).expect("eligible");
    let mut seen: Vec<i64> = Vec::new();
    for o in block_origins(&axes, shape) {
        seen.extend(rows(&axes, shape, &o));
    }
    let total: usize = axes.iter().map(|a| a.extent).product();
    assert_eq!(seen.len(), total, "one entry per row of the role");
    seen.sort_unstable();
    assert_eq!(
        seen,
        (0..total as i64).collect::<Vec<i64>>(),
        "no gaps, no repeats"
    );
    // 40 `b` values x (48 `a` / 8 per block) = 240 blocks for this case.
    assert_eq!(block_origins(&axes, shape).len(), 240);
}

#[test]
fn gather_is_the_identity_for_todays_intervals() {
    let scatter: Vec<i64> = (0..24).map(|i| i * 7).collect();
    let rows: Vec<i64> = (0..24).collect();
    assert_eq!(gather(&scatter, &rows), scatter);
}

#[test]
fn the_output_order_is_consecutive_for_the_measured_case() {
    let axes = measured();
    let shape = eligible(&axes, 8).expect("eligible");
    let rows = rows(&axes, shape, &[0, 0, 0]);
    let offs = out_offsets(&rows);
    // The emitter is handed the block's rows in this order, so its stores run
    // consecutively in runs of 48 - one per `a`, whose `c` values are the
    // output's contiguous axis.
    let order = output_order(&offs);
    let ordered: Vec<i64> = order.iter().map(|&r| offs[r as usize]).collect();
    assert!(
        ordered.windows(2).all(|w| w[0] <= w[1]),
        "sorted by output offset"
    );
    let runs: Vec<usize> = {
        let mut runs = Vec::new();
        let mut n = 1usize;
        for w in ordered.windows(2) {
            if w[1] == w[0] + 1 {
                n += 1;
            } else {
                runs.push(n);
                n = 1;
            }
        }
        runs.push(n);
        runs
    };
    assert_eq!(runs.len(), 8, "one contiguous run per a");
    assert!(
        runs.iter().all(|&n| n >= 48),
        "each run is the block's 48 c's"
    );
}

#[test]
fn the_output_order_is_the_identity_when_the_block_is_already_ordered() {
    let offs: Vec<i64> = (0..16).map(|i| i * 3).collect();
    assert_eq!(output_order(&offs), (0..16).collect::<Vec<u32>>());
}

#[test]
fn ties_keep_the_blocks_own_order() {
    // The raw write-back contract allows repeated target addresses; the order
    // must not silently reorder them.
    let offs = vec![5i64, 1, 5, 1, 5];
    assert_eq!(output_order(&offs), vec![1u32, 3, 0, 2, 4]);
}

#[test]
fn reorder_rows_moves_whole_rows() {
    let src: Vec<i64> = (0..6).collect(); // 3 rows x 2 cols
    let mut dst = vec![0i64; 6];
    reorder_rows(&src, 2, &[2, 0, 1], &mut dst);
    assert_eq!(dst, vec![4, 5, 0, 1, 2, 3]);
}
