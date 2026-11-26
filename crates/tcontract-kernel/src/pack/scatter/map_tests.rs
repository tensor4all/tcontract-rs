//! `ScatterMap` against the expanded vector, which is the reference.
//!
//! The cross-check of tcontract-rs#221: 5,000 synthetic shapes and 50,000
//! blocks (ten per shape), every verdict identical. Shapes are small enough to
//! expand in a test, and drawn to hit the awkward cases -- extent-1 axes, zero
//! and negative strides, equal strides that fold into one run, ragged tails.

use super::*;

/// splitmix64: a fixed stream, so a failure names its shape.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// A random group: rank 0..=6, extents 1..=6 with a product cap, strides that are
/// sometimes the compact (folding) ones, sometimes zero, negative or arbitrary.
fn shape(rng: &mut Rng) -> (Vec<i64>, Vec<i64>) {
    let rank = rng.below(7) as usize;
    let mut ext = Vec::new();
    let mut total = 1i64;
    for _ in 0..rank {
        let e = 1 + rng.below(6) as i64;
        if total * e > 400 {
            break;
        }
        total *= e;
        ext.push(e);
    }
    let compact = rng.below(3) == 0;
    let mut acc = 1i64;
    let strides = ext
        .iter()
        .map(|&e| {
            let s = if compact {
                acc * (1 + rng.below(2) as i64)
            } else {
                match rng.below(6) {
                    0 => 0,
                    1 => -(1 + rng.below(9) as i64),
                    2 => acc,
                    _ => 1 + rng.below(40) as i64,
                }
            };
            acc *= e;
            s
        })
        .collect();
    (ext, strides)
}

#[test]
fn verdicts_match_the_expanded_vector_on_5000_shapes_and_50000_blocks() {
    let mut rng = Rng(221);
    let (mut shapes, mut blocks) = (0usize, 0usize);
    let (mut regular, mut irregular) = (0usize, 0usize);
    for _ in 0..5_000 {
        let (ext, strides) = shape(&mut rng);
        let scat = build_scatter(&ext, &strides);
        let map = ScatterMap::new(&ext, &strides);
        assert_eq!(map.len(), scat.len(), "{ext:?} {strides:?}");

        // Entries, whole and from a random head.
        let mut all = vec![0i64; scat.len()];
        map.fill(0, &mut all);
        assert_eq!(all, scat, "fill {ext:?} {strides:?}");
        if !scat.is_empty() {
            let lo = rng.below(scat.len() as u64) as usize;
            let n = rng.below((scat.len() - lo) as u64 + 1) as usize;
            let mut part = vec![0i64; n];
            map.fill(lo, &mut part);
            assert_eq!(part, scat[lo..lo + n], "fill {lo}+{n} {ext:?} {strides:?}");
            assert_eq!(map.at(lo), scat[lo], "at {ext:?} {strides:?}");
        }

        // A cursor walking from a random head.
        if !scat.is_empty() {
            let from = rng.below(scat.len() as u64) as usize;
            let mut cur = map.cursor(from);
            for want in &scat[from..] {
                assert_eq!(cur.offset(), *want, "cursor {from} {ext:?} {strides:?}");
                cur.advance();
            }
        }

        // Run structure, from the same deltas.
        assert_eq!(
            map.run_structure(),
            run_structure(&scat),
            "run_structure {ext:?} {strides:?}"
        );
        assert_eq!(
            map.is_unit_ramp(),
            scat.len() <= 1 || scat.windows(2).all(|w| w[1] - w[0] == 1),
            "unit ramp {ext:?} {strides:?}"
        );

        // Ten random blocks.
        for _ in 0..10 {
            if scat.is_empty() {
                break;
            }
            let lo = rng.below(scat.len() as u64) as usize;
            let hi = lo + rng.below((scat.len() - lo) as u64 + 1) as usize;
            let want = run_stride(&scat[lo..hi]);
            assert_eq!(
                map.block_stride(lo, hi),
                want,
                "block [{lo},{hi}) of {ext:?} {strides:?}"
            );
            blocks += 1;
            if want == IRREGULAR {
                irregular += 1;
            } else {
                regular += 1;
            }
        }

        // Whole block-scatter vectors at several block sizes.
        for blk in [1usize, 2, 3, 4, 6, 8] {
            let want = build_block_scatter(&scat, blk);
            let mut got = vec![7];
            let (s, e) = map.append_block_scatter(&mut got, blk);
            assert_eq!(&got[s..e], &want[..], "blk {blk} of {ext:?} {strides:?}");
            assert_eq!(got[0], 7, "appends, never overwrites");
            assert_eq!(
                map.block_scatter_regular(blk),
                block_scatter_regular(&scat, blk),
                "regular blk {blk} of {ext:?} {strides:?}"
            );
        }
        shapes += 1;
    }
    // Every shape is non-empty (extents start at 1), so ten blocks each.
    assert_eq!((shapes, blocks), (5_000, 50_000));
    // The sample must exercise both verdicts, or agreement proves little.
    assert!(
        regular > 5_000 && irregular > 5_000,
        "{regular} {irregular}"
    );
}

#[test]
fn empty_and_degenerate_groups() {
    for (ext, st) in [
        (&[0i64][..], &[3i64][..]),
        (&[2, 0, 3], &[1, 2, 4]),
        (&[], &[]),
    ] {
        let scat = build_scatter(ext, st);
        let map = ScatterMap::new(ext, st);
        assert_eq!(map.len(), scat.len());
        assert_eq!(map.run_structure(), run_structure(&scat));
        for blk in [1, 2, 5] {
            let mut got = Vec::new();
            map.append_block_scatter(&mut got, blk);
            assert_eq!(got, build_block_scatter(&scat, blk));
        }
    }
    // An overflowing product has no vector either.
    assert!(ScatterMap::new(&[i64::MAX, 4], &[1, 1]).is_empty());
    // Equal maps compare equal whatever extent-1 axes sit between.
    assert_eq!(
        ScatterMap::new(&[2, 1, 3], &[1, 99, 2]),
        ScatterMap::new(&[2, 3], &[1, 2])
    );
    // A rank beyond the stack digits still works.
    let ext = vec![2i64; 40];
    let st: Vec<i64> = (0..40).map(|k| 1i64 << k).collect();
    let map = ScatterMap::new(&ext, &st);
    let mut out = [0i64; 5];
    map.fill((1 << 39) - 2, &mut out);
    assert_eq!(
        out,
        [
            (1 << 39) - 2,
            (1 << 39) - 1,
            1 << 39,
            (1 << 39) + 1,
            (1 << 39) + 2
        ]
    );
    // Wrapping, as the single-axis ramp does, for strides no layout has.
    let big = i64::MAX / 2 + 1;
    let mut got = vec![0; 6];
    ScatterMap::new(&[3, 2], &[big, 5]).fill(0, &mut got);
    let two = big.wrapping_mul(2);
    assert_eq!(got, [0, big, two, 5, big + 5, two.wrapping_add(5)][..]);
}
