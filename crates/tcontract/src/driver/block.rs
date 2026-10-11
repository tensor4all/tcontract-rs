//! Axis-aware outer blocks: the membership the blocked traversal needs.
//!
//! Today the driver walks consecutive slices of one flattened row scatter
//! (`static_grid.rs`, `tile.rs`). While the panel's row order is the output's that
//! is all it needs. It is not enough when the packed operand and the output
//! disagree about their fastest axis: a consecutive slice of the source-ordered
//! flattening then holds the output's fast axis only once per
//! `product_of_the_other_axes` rows, so no reordering *inside* that slice can
//! produce a contiguous destination run. The campaign's `abjc-cbka-kj` is such a
//! contraction, and the design in `docs/design/blocked-outer-traversal.md` needs a
//! block that spans whole runs of *both* axes.
//!
//! This module is that membership, plus the conservative predicate that keeps
//! every other shape on today's path. Nothing calls it yet: it is the first,
//! behaviour-preserving step of that design, and the tests below pin both the
//! identity view of today's slices and the runs the measured case needs.

use crate::plan::PlanStats;

/// The axes a role can have: the block's scratch is sized from it.
pub(crate) const MAX_AXES: usize = 8;

/// One axis of an oriented role as the blocked path needs it: its extent and its
/// stride in the packed operand and in the output. `sc`/`sd` for the m role,
/// `sb`/`sd` for the n role, and the roles swap under orientation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)] // as below: the enumerator that consumes it lands next
pub(crate) struct Role {
    pub(crate) extent: usize,
    pub(crate) src: i64,
    pub(crate) dst: i64,
}

impl Role {
    /// A placeholder for fixed-size scratch before a role is written into it.
    pub(crate) const EMPTY: Role = Role {
        extent: 0,
        src: 0,
        dst: 0,
    };
}

/// The oriented role's axes, in the plan's own order. `swapped`: the plan
/// exchanges the row and column operands, so the row role is the user's `n`.
pub(crate) fn role_into<'a>(
    stats: &PlanStats,
    swapped: bool,
    out: &'a mut [Role; MAX_AXES],
) -> &'a [Role] {
    let axes = if swapped {
        &stats.n_axes
    } else {
        &stats.m_axes
    };
    assert!(axes.len() <= MAX_AXES, "role axes fit the scratch");
    for (o, a) in out.iter_mut().zip(axes) {
        *o = Role {
            extent: a.extent as usize,
            src: if swapped { a.sb } else { a.sa },
            dst: a.sd,
        };
    }
    &out[..axes.len()]
}

/// Whether the blocked path applies at all: the oriented row role's fastest axes
/// disagree (so today's consecutive slices cannot hold both runs), the contraction
/// fits **one K slab** and **one NC panel** (so a block's accumulator is complete
/// without crossing slabs and the block's whole output is one contiguous write),
/// and the role has at least two live axes. Everything else keeps today's path.
///
/// The two slab conditions are what keep this the narrow, safe half of the design:
/// no cross-slab persistence, no ownership or barrier change, no resized panels.
#[allow(dead_code)]
pub(crate) fn blocked_eligibility(
    stats: &PlanStats,
    swapped: bool,
    k: usize,
    n: usize,
    kc: usize,
    nc: usize,
    per_line: usize,
    mc_budget: usize,
    mr: usize,
) -> Option<BlockShape> {
    assert!(mr > 0, "the register block is non-empty");
    if k > kc || n > nc {
        return None;
    }
    // Profitability, measured rather than assumed: the block pays for a contiguous pack
    // with the grid permutation and the block metadata, so it needs more operand elements
    // than output ones. `n <= 64 and k > n` is the class that wins without losing over the
    // corpus; anything else keeps today's traversal, which is not a regression.
    if n > 64 || k <= n {
        return None;
    }
    let mut axes_buf = [Role::EMPTY; MAX_AXES];
    let axes = role_into(stats, swapped, &mut axes_buf);
    let mut shape = eligible(axes, per_line)?;
    // A block must fit the `MC` budget: the packed `A` panel is sized from `MC`, and a
    // block that overruns it would write past the panel. The block's row count is the
    // span axis whole times a line's worth along the operand's fastest axis - every
    // other axis contributes one value - so only `span` divides into the budget;
    // dividing by the other axes too made the block ten times smaller than the design's
    // geometry, and ten times as many per-block fixed costs were the whole loss.
    let span = axes[shape.span].extent.max(1);
    let per_block = (mc_budget / span).max(1);
    shape.line_len = shape.line_len.min(per_block);
    // What the panel, the tile grid and the block's row map are actually sized from is the
    // block *rounded up to whole tiles*, which can be `MR - 1` rows larger than the block;
    // and `span` alone can exceed the budget. Both are panics in `run_block`'s scratch, so
    // they are conditions here rather than assertions there.
    let block_rows = span.saturating_mul(shape.line_len);
    if block_rows.next_multiple_of(mr) > mc_budget {
        return None;
    }
    Some(shape)
}

/// How many values a block takes along an axis: the output's fastest axis whole,
/// the operand's fastest axis up to one cache line, everything else one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)] // the enumerator that consumes it lands with the design's next step
pub(crate) struct BlockShape {
    /// Index into the role's axis list of the axis a block spans whole.
    pub(crate) span: usize,
    /// Index of the axis a block takes a line's worth of.
    pub(crate) line: usize,
    /// Values taken along `line`'s axis, `1..=extent`.
    pub(crate) line_len: usize,
}

/// The role's axes ordered by an operand's stride, fastest first. `None` when the
/// role has fewer than two axes, or when an axis is singleton (nothing to gain
/// and no order to preserve).
#[allow(dead_code)] // the enumerator that consumes it lands with the design's next step
fn fastest(axes: &[Role], stride: impl Fn(&Role) -> i64) -> Option<usize> {
    let live: Vec<usize> = (0..axes.len()).filter(|&i| axes[i].extent > 1).collect();
    if live.len() < 2 {
        return None;
    }
    live.into_iter()
        .min_by_key(|&i| stride(&axes[i]).unsigned_abs())
}

/// The block a role needs, or `None` when today's slices are the right answer.
///
/// `operand`, `output`: the two stride projections of the role's axes.
/// `per_line`: elements a 64-byte line holds at this element size (8 for f64), so
/// that a block takes exactly one line's worth along the operand's fastest axis.
#[allow(dead_code)] // the enumerator that consumes it lands with the design's next step
pub(crate) fn eligible(axes: &[Role], per_line: usize) -> Option<BlockShape> {
    let (operand, output) = (|a: &Role| a.src, |a: &Role| a.dst);
    let fast_op = fastest(axes, operand)?;
    let fast_out = fastest(axes, output)?;
    if fast_op == fast_out {
        // The same axis is fastest for both: today's slices already keep the
        // operand's runs together, so there is nothing to change.
        return None;
    }
    Some(BlockShape {
        span: fast_out,
        line: fast_op,
        line_len: axes[fast_op].extent.min(per_line.max(1)),
    })
}

/// The block's global flattened logical row indices, in the operand's order: the
/// axis a block takes a line's worth of varies fastest, then the rest of the
/// block's axes in the role's own list order.
///
/// The indices are mixed-radix over the role's axes, which is the numbering the
/// plan's scatters use, so they can index every operand's scatter - the element
/// offsets differ per operand, the logical row does not.
/// [`rows`] into a caller's buffer, returning how many rows it wrote: the blocked path
/// must not allocate, and the driver reuses one stack buffer for every block.
pub(crate) fn rows_into(
    axes: &[Role],
    shape: BlockShape,
    origin: &[usize],
    out: &mut [i64],
) -> usize {
    let mut strides = [0i64; MAX_AXES];
    let mut acc = 1i64;
    for (i, a) in axes.iter().enumerate() {
        strides[i] = acc;
        acc *= a.extent as i64;
    }
    let span_len = axes[shape.span].extent;
    let mut n = 0;
    for s in 0..span_len {
        for l in 0..shape.line_len {
            // A role's extent need not be a whole number of lines: the last block on
            // the line axis is short, and `block_origins` still placed it. Stop rather
            // than clamp, so no row is listed twice - coverage is what the gates check.
            if origin[shape.line] + l >= axes[shape.line].extent {
                break;
            }
            if n == out.len() {
                return n;
            }
            let mut row = 0i64;
            for (i, &st) in strides.iter().enumerate().take(axes.len()) {
                let v = if i == shape.line {
                    origin[i] + l
                } else if i == shape.span {
                    origin[i] + s
                } else {
                    origin[i]
                };
                row += v as i64 * st;
            }
            out[n] = row;
            n += 1;
        }
    }
    n
}

/// [`rows_into`] as a `Vec`, for tests.
#[cfg(test)]
pub(crate) fn rows(axes: &[Role], shape: BlockShape, origin: &[usize]) -> Vec<i64> {
    let mut out = vec![0i64; shape.line_len * axes[shape.span].extent];
    let n = rows_into(axes, shape, origin, &mut out);
    out.truncate(n);
    out
}

/// Every block origin a role needs, one entry per axis: the span axis has a single
/// origin (a block takes it whole), the line axis steps by a line, and every other
/// axis steps by one. A block is `rows(axes, shape, &origin)`.
///
/// The origins tile the role exactly once, which is what the coverage gate checks:
/// the block rows of an eligible role, concatenated, are a permutation of
/// `0..extent_product`.
/// Every block origin a role needs, one at a time and without allocating.
///
/// The span axis has a single origin (a block takes it whole), the line axis steps by a
/// line, and every other axis steps by one. `state` is the caller's scratch; call `next`
/// until it returns `false`. The first call returns the all-zero origin. The originals
/// used to be materialized as a `Vec<Vec<usize>>`, one inner allocation per block, which
/// the blocked path cannot afford under the workspace's no-allocation contract.
pub(crate) struct Origins {
    state: [usize; MAX_AXES],
    started: bool,
}

impl Origins {
    pub(crate) fn new() -> Origins {
        Origins {
            state: [0; MAX_AXES],
            started: false,
        }
    }

    /// The next origin in `state`, or `None` when they are exhausted.
    pub(crate) fn next(&mut self, axes: &[Role], shape: BlockShape) -> Option<&[usize]> {
        if self.started {
            // An odometer over the axes that step, in the role's own order.
            // The highest-numbered axis that steps varies fastest, which is the order the
            // materialized list used; the block set is the same either way, but keeping
            // the order lets the coverage test compare them exactly.
            let mut carried = true;
            for i in (0..axes.len()).rev() {
                if i == shape.span {
                    continue;
                }
                let step = if i == shape.line { shape.line_len } else { 1 };
                self.state[i] += step;
                if self.state[i] < axes[i].extent {
                    carried = false;
                    break;
                }
                self.state[i] = 0;
            }
            if carried {
                return None;
            }
        }
        self.started = true;
        Some(&self.state[..axes.len()])
    }
}

/// [`Origins`] as a `Vec`, for tests that compare it against the coverage property.
#[cfg(test)]
pub(crate) fn block_origins(axes: &[Role], shape: BlockShape) -> Vec<Vec<usize>> {
    let mut out = vec![vec![0usize; axes.len()]];
    for i in 0..axes.len() {
        if i == shape.span {
            continue;
        }
        let step = if i == shape.line { shape.line_len } else { 1 };
        let mut next = Vec::new();
        for o in &out {
            let mut x = 0;
            while x < axes[i].extent {
                let mut v = o.clone();
                v[i] = x;
                next.push(v);
                x += step;
            }
        }
        out = next;
    }
    out
}

/// Gather one operand's element offsets for a block's rows.
#[allow(dead_code)] // the enumerator that consumes it lands with the design's next step
pub(crate) fn gather(scatter: &[i64], rows: &[i64]) -> Vec<i64> {
    let mut out = vec![0i64; rows.len()];
    gather_into(scatter, rows, &mut out);
    out
}

/// [`gather`] into a caller's buffer, so a steady-state call allocates nothing.
#[allow(dead_code)] // as above
pub(crate) fn gather_into(scatter: &[i64], rows: &[i64], out: &mut [i64]) {
    debug_assert_eq!(out.len(), rows.len(), "one offset per row");
    for (o, &r) in out.iter_mut().zip(rows) {
        *o = scatter[r as usize];
    }
}

/// The order in which a block's rows must be visited for the output stores to be
/// consecutive: `d_local`'s row indices sorted by their element offset, stably, so
/// that rows sharing an offset keep the block's own order (the raw write-back
/// contract's last-writer behaviour for repeated addresses depends on it).
///
/// This is the other half of the design: the block's membership supplies the
/// source run the pack reads, and this order turns the block's accumulated tile
/// into the sequence the emitter writes, which is why the transposition can be
/// paid once per block instead of once per micro-tile.
#[allow(dead_code)] // as above: the enumerator that consumes it lands next
pub(crate) fn output_order(d_local: &[i64]) -> Vec<u32> {
    let mut order: Vec<u32> = vec![0; d_local.len()];
    output_order_into(d_local, &mut order);
    order
}

/// [`output_order`] into a caller's buffer, so a steady-state call allocates nothing.
#[allow(dead_code)] // as above
pub(crate) fn output_order_into(d_local: &[i64], order: &mut [u32]) {
    debug_assert_eq!(order.len(), d_local.len(), "one entry per row");
    for (i, o) in order.iter_mut().enumerate() {
        *o = i as u32;
    }
    // A stable sort by the output offset: rows sharing an offset keep the block's
    // own order, which the raw write-back's last-writer behaviour depends on.
    order.sort_by_key(|&r| d_local[r as usize]);
}

/// Reorder a block's accumulated rows into `output_order`: `src` and `dst` are
/// row-major `rows x nc`, and `src`'s row `r` becomes `dst`'s row `order[r]`'s
/// source - i.e. `dst[i] = src[order[i]]` row by row. Written as a plain copy so
/// it can be a model for the driver's tiled version.
#[allow(dead_code)] // as above: the enumerator that consumes it lands next
pub(crate) fn reorder_rows<T: Copy>(src: &[T], nc: usize, order: &[u32], dst: &mut [T]) {
    assert_eq!(src.len(), order.len() * nc, "block scratch shape");
    assert_eq!(dst.len(), src.len(), "same block, reordered");
    for (i, &r) in order.iter().enumerate() {
        let (s, d) = (r as usize * nc, i * nc);
        dst[d..d + nc].copy_from_slice(&src[s..s + nc]);
    }
}

#[cfg(test)]
mod tests {
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
}
