//! Plan-time cost of describing a role: materialise-then-compress versus
//! deriving the blocks from the axes.
//!
//! Isolated probe for the worklog decision; it measures metadata construction
//! only, and makes no claim about contraction execution.

use std::time::Instant;
use tcontract_kernel::scatter::{append_block_scatter, build_scatter, IRREGULAR};

/// Local copy of the axis-driven primitive this probe measured. The probe is
/// retained as the record of a *reverted* attempt, so it must not depend on
/// code that is no longer in the tree; keeping the two functions here is what
/// makes the historical numbers reproducible against this checkout.
fn role_len(extents: &[i64]) -> usize {
    let mut total: i64 = 1;
    for &extent in extents {
        if extent == 0 {
            return 0;
        }
        total = match total.checked_mul(extent) {
            Some(product) => product,
            None => return 0,
        };
    }
    total.max(0) as usize
}

fn block_strides_from_axes(extents: &[i64], strides: &[i64], blk: usize, out: &mut Vec<i64>) {
    assert!(blk > 0);
    out.clear();
    let total = role_len(extents);
    if total == 0 {
        return;
    }
    if extents.len() <= 1 {
        let stride = strides.first().copied().unwrap_or(0);
        for b in 0..total.div_ceil(blk) {
            let n = ((b * blk + blk).min(total)) - b * blk;
            out.push(if n <= 1 { 0 } else { stride });
        }
        return;
    }
    let n = extents.len();
    let mut counter = vec![0i64; n];
    let mut offset = 0i64;
    let mut pos = 0usize;
    let mut lo = 0usize;
    while lo < total {
        let hi = (lo + blk).min(total);
        while pos < lo {
            for d in 0..n {
                counter[d] += 1;
                offset += strides[d];
                if counter[d] < extents[d] {
                    break;
                }
                offset -= strides[d] * extents[d];
                counter[d] = 0;
            }
            pos += 1;
        }
        let mut stride = 0i64;
        let mut irregular = false;
        for entry in lo + 1..hi {
            let before = offset;
            for d in 0..n {
                counter[d] += 1;
                offset += strides[d];
                if counter[d] < extents[d] {
                    break;
                }
                offset -= strides[d] * extents[d];
                counter[d] = 0;
            }
            pos += 1;
            let delta = offset - before;
            if entry == lo + 1 {
                stride = delta;
            } else if delta != stride {
                irregular = true;
            }
        }
        out.push(match hi - lo {
            0 | 1 => 0,
            _ if irregular => IRREGULAR,
            _ => stride,
        });
        lo = hi;
    }
}

/// Best of seven timings of `f`, after warming `f` itself for a fixed wall
/// time. The warm-up is per *arm* and time-based, not a call count: after the
/// idle gate the core needs a second or so at load before it reports its
/// settled clock, and an arm that ran second would otherwise inherit the first
/// arm's warmth.
fn best(f: &mut dyn FnMut()) -> f64 {
    let deadline = Instant::now() + std::time::Duration::from_millis(1000);
    while Instant::now() < deadline {
        f();
    }
    let mut b = f64::INFINITY;
    for _ in 0..7 {
        let t = Instant::now();
        f();
        b = b.min(t.elapsed().as_secs_f64());
    }
    b
}

fn main() {
    let cases: &[(&str, &[i64], &[i64], usize)] = &[
        // No commas in the labels: this prints unquoted CSV.
        ("2^24 one axis blk24", &[1 << 24], &[1], 24),
        ("2^20 one axis blk24", &[1 << 20], &[1], 24),
        ("256x256 folded blk24", &[256, 256], &[1, 256], 24),
        ("230400 folded blk24", &[230400], &[1], 24),
        ("16x8x4 mixed blk24", &[16, 8, 4], &[1, 16, 128], 24),
    ];
    println!("case,role_len,old_plan_ms,new_plan_ms,speedup,equal");
    for (name, extents, strides, blk) in cases {
        let len = role_len(extents);
        let mut old_out = Vec::new();
        let mut new_out = Vec::new();
        let mut old = || {
            let scat = build_scatter(extents, strides);
            old_out.clear();
            append_block_scatter(&mut old_out, &scat, *blk);
        };
        let mut new = || {
            new_out.clear();
            block_strides_from_axes(extents, strides, *blk, &mut new_out);
        };
        let old_s = best(&mut old);
        let new_s = best(&mut new);
        // Equality on the same inputs, before any number is reported.
        let scat = build_scatter(extents, strides);
        let mut want = Vec::new();
        append_block_scatter(&mut want, &scat, *blk);
        let mut got = Vec::new();
        block_strides_from_axes(extents, strides, *blk, &mut got);
        assert_eq!(got, want, "{name}");
        println!(
            "{name},{len},{:.4},{:.4},{:.2},{}",
            old_s * 1e3,
            new_s * 1e3,
            old_s / new_s,
            got == want
        );
    }
}
