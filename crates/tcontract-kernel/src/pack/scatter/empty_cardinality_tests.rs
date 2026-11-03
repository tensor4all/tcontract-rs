use super::build_scatter;

#[test]
fn zero_extent_behind_an_overflowing_prefix_is_empty_not_a_panic() {
    // The prefix product overflows `i64`, but the zero extent makes the group
    // empty; the vector must be empty rather than a panic (or a wrapped
    // length that would allocate wrongly).
    assert!(build_scatter(&[1i64 << 62, 2, 0], &[1, 1, 1]).is_empty());
    assert!(build_scatter(&[0, 1i64 << 62], &[isize::MAX as i64, 1]).is_empty());
}
