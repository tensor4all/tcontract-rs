use super::*;

/// The single-axis fast path must agree with the generic odometer, for a
/// positive stride, a zero stride and a negative one.
#[test]
fn single_axis_scatter_is_a_ramp() {
    for &(extent, stride) in &[(7i64, 1i64), (5, 0), (6, -3), (1, 9)] {
        let want: Vec<i64> = (0..extent).map(|i| i * stride).collect();
        assert_eq!(
            build_scatter(&[extent], &[stride]),
            want,
            "{extent} {stride}"
        );
    }
    // The wrap is deliberate and must match the odometer's arithmetic.
    let big = i64::MAX / 2 + 1;
    assert_eq!(
        build_scatter(&[3], &[big]),
        vec![0, big, (big as i128 * 2) as i64],
        "a stride whose second entry wraps"
    );
    // A zero extent is empty, not one entry of zero.
    assert!(build_scatter(&[0], &[3]).is_empty());
}

#[test]
fn scatter_column_major() {
    // extents (2,3) strides (1,2), fastest first -> 0,1,2,3,4,5
    let s = build_scatter(&[2, 3], &[1, 2]);
    assert_eq!(s, vec![0, 1, 2, 3, 4, 5]);
}

#[test]
fn scatter_permuted() {
    // A 2x3 tensor stored row-major, viewed with the *second* index
    // fastest: extents (3,2) strides (1,3).
    let s = build_scatter(&[3, 2], &[1, 3]);
    assert_eq!(s, vec![0, 1, 2, 3, 4, 5]);
    // and with the first index fastest: extents (2,3) strides (3,1)
    let s = build_scatter(&[2, 3], &[3, 1]);
    assert_eq!(s, vec![0, 3, 1, 4, 2, 5]);
}

#[test]
fn scatter_empty_group_is_single_zero() {
    assert_eq!(build_scatter(&[], &[]), vec![0]);
}

#[test]
fn block_scatter_regularity() {
    let s = build_scatter(&[2, 3], &[3, 1]); // 0,3,1,4,2,5
    let bs = build_block_scatter(&s, 2);
    assert_eq!(bs, vec![3, 3, 3]);
    let bs = build_block_scatter(&s, 3);
    assert_eq!(bs, vec![IRREGULAR, IRREGULAR]);
    assert_eq!(regular_fraction(&bs), 0.0);
}

#[test]
fn zero_stride_is_regular_not_irregular() {
    let s = build_scatter(&[4], &[0]);
    assert_eq!(s, vec![0, 0, 0, 0]);
    let bs = build_block_scatter(&s, 2);
    assert_eq!(bs, vec![0, 0]);
    assert_eq!(regular_fraction(&bs), 1.0);
}

#[test]
fn run_structure_recognises_equal_length_runs() {
    // Four contiguous runs of 24, restarted by an outer axis.
    assert_eq!(
        run_structure(&build_scatter(&[24, 4], &[1, 200])),
        Some((24, 1))
    );
    // A single unbroken run is the whole vector.
    assert_eq!(run_structure(&build_scatter(&[24], &[1])), Some((24, 1)));
    // Constant non-unit stride is still one run: no block size breaks it.
    assert_eq!(run_structure(&build_scatter(&[12], &[4])), Some((12, 4)));
    // Unequal runs have no uniform structure.
    assert_eq!(run_structure(&[0, 1, 2, 100, 101, 200]), None);
    assert_eq!(run_structure(&[7]), None);
}

#[test]
fn unbroken_fraction_counts_straddling_blocks() {
    // 96 entries in runs of 24. Only block sizes that tile a run survive.
    assert_eq!(unbroken_fraction(96, 24, 24), 1.0);
    assert_eq!(unbroken_fraction(96, 24, 8), 1.0);
    // 16 into 24 straddles every second block of a 48-entry period.
    assert!((unbroken_fraction(96, 24, 16) - 2.0 / 3.0).abs() < 1e-12);
    assert_eq!(unbroken_fraction(96, 24, 32), 0.0);
    assert_eq!(unbroken_fraction(96, 24, 48), 0.0);
}

#[test]
fn ragged_tail_block() {
    let s: Vec<i64> = (0..5).collect();
    let bs = build_block_scatter(&s, 2);
    assert_eq!(bs, vec![1, 1, 0]); // last block has a single entry
}
