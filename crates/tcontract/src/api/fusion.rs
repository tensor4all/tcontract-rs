//! Fusing one role group of axes into a single GEMM dimension.
//!
//! A GEMM backend reads each role group (M, N, K or the batch) as one
//! dimension. That works in place when the group's non-unit axes, taken in
//! some order, have strides that chain in the operand (`s_next == s_prev *
//! extent_prev`). These two functions are the shared rule; a backend decides
//! what to do with a group that does not fuse (the faer strategy declines it,
//! a BLAS backend packs the operand).

use super::{OperandId, RoleAxis};

/// Whether `order` fuses `group` in `operand`: the strides of the non-unit
/// axes, in that order, must chain (`s_next == s_prev * extent_prev`).
///
/// `order` is a permutation of `0..group.len()`. Returns the fused
/// `(extent, stride)`; a group without a non-unit axis fuses to stride one.
/// `None` when the strides do not chain, a product overflows, or `order` is
/// not a permutation of the group's axes.
///
/// # Examples
///
/// ```
/// use tcontract::api::{fuse_axes, CSpec, DType, Labels, LayoutSpec, OperandId, OperandSpec, Problem};
/// let l = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
/// // D[i,j,k] = sum_l A[i,j,l] B[l,k]: M is (i, j), contiguous in A and D.
/// let p = Problem::from_labels(
///     DType::F64,
///     l(&[2, 3, 4], &[1, 2, 6]),
///     l(&[4, 5], &[1, 4]),
///     CSpec::Absent,
///     l(&[2, 3, 5], &[1, 2, 6]),
///     &Labels::new(&[0, 1, 3], &[3, 2], &[0, 1, 2]),
/// )
/// .unwrap();
/// let m = p.roles().m();
/// assert_eq!(fuse_axes(m, &[0, 1], OperandId::A), Some((6, 1)));
/// assert_eq!(fuse_axes(m, &[1, 0], OperandId::A), None);
/// ```
pub fn fuse_axes(
    group: &[RoleAxis],
    order: &[usize],
    operand: OperandId,
) -> Option<(usize, isize)> {
    if order.len() != group.len() {
        return None;
    }
    let mut ext = 1usize;
    let mut first: Option<isize> = None;
    let mut expect = 0isize;
    for (i, &e) in order.iter().enumerate() {
        // Groups have a handful of axes: a quadratic duplicate check needs no buffer.
        if order[..i].contains(&e) {
            return None;
        }
        let axis = group.get(e)?;
        let d = axis.extent();
        if d == 1 {
            continue;
        }
        let s = axis.stride(operand);
        match first {
            None => first = Some(s),
            Some(_) if s != expect => return None,
            Some(_) => {}
        }
        expect = s.checked_mul(isize::try_from(d).ok()?)?;
        ext = ext.checked_mul(d)?;
    }
    Some((ext, first.unwrap_or(1)))
}

/// Sorts the axis indices in `order` by their stride in `operand` (by
/// magnitude, smallest first), stably: equal strides keep their order in
/// `order`. Starting from the natural order `0..group.len()`, this is the
/// order a carrier of the group would fuse it in. An index outside the group
/// sorts last.
///
/// # Examples
///
/// ```
/// use tcontract::api::{sort_by_stride, CSpec, DType, Labels, LayoutSpec, OperandId, OperandSpec, Problem};
/// let l = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
/// // M is (i, j): row-major in A, column-major in D.
/// let p = Problem::from_labels(
///     DType::F64,
///     l(&[2, 3, 4], &[12, 4, 1]),
///     l(&[4, 5], &[1, 4]),
///     CSpec::Absent,
///     l(&[2, 3, 5], &[1, 2, 6]),
///     &Labels::new(&[0, 1, 3], &[3, 2], &[0, 1, 2]),
/// )
/// .unwrap();
/// let mut order = [0, 1];
/// sort_by_stride(p.roles().m(), OperandId::A, &mut order);
/// assert_eq!(order, [1, 0]);
/// sort_by_stride(p.roles().m(), OperandId::D, &mut order);
/// assert_eq!(order, [0, 1]);
/// ```
pub fn sort_by_stride(group: &[RoleAxis], operand: OperandId, order: &mut [usize]) {
    order.sort_by_key(|&e| {
        group
            .get(e)
            .map_or(usize::MAX, |axis| axis.stride(operand).unsigned_abs())
    });
}

#[cfg(test)]
mod tests {
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
}
