use super::*;

/// `M` axes in A, `N` axes in B, all in D: an outer product with
/// `m + n` role axes.
fn outer(m: usize, n: usize) -> Problem {
    let spec = |dims: &[usize]| {
        let strides: Vec<isize> = (0..dims.len()).map(|i| 1 << i).collect();
        OperandSpec::new(LayoutSpec::new(dims, &strides, 0).unwrap())
    };
    let (la, lb): (Vec<i64>, Vec<i64>) = (
        (0..m as i64).collect(),
        (m as i64..(m + n) as i64).collect(),
    );
    let ld: Vec<i64> = la.iter().chain(&lb).copied().collect();
    Problem::from_labels(
        DType::F64,
        spec(&vec![2; m]),
        spec(&vec![2; n]),
        CSpec::Absent,
        spec(&vec![2; m + n]),
        &Labels::new(&la, &lb, &ld),
    )
    .unwrap()
}

#[test]
fn the_role_pool_is_inline_up_to_two_inline_rank_inputs() {
    let roles = &outer(INLINE_RANK, INLINE_RANK).lowered.roles;
    assert_eq!((roles.m().len(), roles.n().len()), (8, 8));
    assert!(roles.axes.is_inline());
    // Above the bound the pool spills once, to exactly the axes it holds.
    let roles = &outer(INLINE_RANK + 1, INLINE_RANK).lowered.roles;
    assert!(roles.axes.is_heap());
    assert_eq!(roles.axes.capacity(), COMBINED_RANK + 1);
}

#[test]
fn roles_stay_within_one_pool_of_role_axes() {
    assert!(size_of::<Roles>() <= (COMBINED_RANK + 1) * size_of::<RoleAxis>());
}
