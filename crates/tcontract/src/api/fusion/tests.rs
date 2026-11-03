use super::*;

/// An axis with `extent` and stride `a` in A (the other operands absent).
fn axis(extent: usize, a: isize) -> RoleAxis {
    RoleAxis {
        extent,
        strides: [a, 0, 0, 0],
        in_a: true,
        ..RoleAxis::default()
    }
}

#[test]
fn chained_strides_fuse_and_others_do_not() {
    let g = [axis(2, 1), axis(3, 2), axis(4, 6)];
    assert_eq!(fuse_axes(&g, &[0, 1, 2], OperandId::A), Some((24, 1)));
    assert_eq!(fuse_axes(&g, &[1, 0, 2], OperandId::A), None);
    // Signed strides chain the same way.
    let g = [axis(2, -1), axis(3, -2)];
    assert_eq!(fuse_axes(&g, &[0, 1], OperandId::A), Some((6, -1)));
}

#[test]
fn unit_and_empty_groups_fuse_to_stride_one() {
    assert_eq!(fuse_axes(&[], &[], OperandId::A), Some((1, 1)));
    let g = [axis(1, 7), axis(1, 9)];
    assert_eq!(fuse_axes(&g, &[1, 0], OperandId::A), Some((1, 1)));
    // A unit axis between two chained axes is skipped.
    let g = [axis(2, 1), axis(1, 100), axis(3, 2)];
    assert_eq!(fuse_axes(&g, &[0, 1, 2], OperandId::A), Some((6, 1)));
}

#[test]
fn bad_orders_and_overflow_are_none() {
    let g = [axis(2, 1), axis(3, 2)];
    assert_eq!(fuse_axes(&g, &[0], OperandId::A), None);
    assert_eq!(fuse_axes(&g, &[0, 2], OperandId::A), None);
    assert_eq!(fuse_axes(&[], &[0], OperandId::A), None);
    // A repeated index is not a permutation, even when it would skip a unit axis.
    let g = [axis(1, 1), axis(3, 2)];
    assert_eq!(fuse_axes(&g, &[0, 0], OperandId::A), None);
    let g = [axis(usize::MAX, 1), axis(2, isize::MAX)];
    assert_eq!(fuse_axes(&g, &[0, 1], OperandId::A), None);
}

#[test]
fn sort_by_stride_is_stable_by_magnitude() {
    let g = [axis(2, 6), axis(3, -1), axis(4, 6), axis(5, 2)];
    let mut order = [0, 1, 2, 3];
    sort_by_stride(&g, OperandId::A, &mut order);
    // |-1| < 2 < 6 == 6, the tie kept in group order.
    assert_eq!(order, [1, 3, 0, 2]);
    // An index outside the group sorts last instead of panicking.
    let mut order = [7, 0];
    sort_by_stride(&g[..1], OperandId::A, &mut order);
    assert_eq!(order, [0, 7]);
}
