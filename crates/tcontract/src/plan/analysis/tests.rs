use super::super::test_support::{build, lay, laying};

#[test]
fn empty_operand_with_unreachable_strides_still_plans() {
    // A[i,k] over an empty slice with a stride a caller may leave
    // unreachable, B[k,j] empty, D[i,j]: an empty contraction is a valid
    // problem, and planning must not overflow while building the scatter
    // vectors of the empty operands.
    let a = laying(&[2, 0], &[isize::MAX as i64, 1]);
    let b = lay(&[0, 3]);
    let d = lay(&[2, 3]);
    let p = build(&a, &[0, 1], &b, &[1, 2], &d, &[0, 2]);
    assert_eq!((p.stats.m, p.stats.n, p.stats.k), (2, 3, 0));
}

#[test]
fn multi_axis_empty_view_with_unreachable_stride_still_plans() {
    // A's empty axis is last, so folding would test `isize::MAX * 2` on the
    // first M axis before the empty-operand stride normalization runs.
    let a = laying(&[2, 2, 0], &[isize::MAX as i64, 1, 1]);
    let b = lay(&[0, 3]);
    let d = lay(&[2, 2, 3]);
    let p = build(&a, &[0, 1, 3], &b, &[3, 2], &d, &[0, 1, 2]);
    assert_eq!((p.stats.m, p.stats.n, p.stats.k), (4, 3, 0));
}

#[test]
fn empty_output_with_unreachable_strides_still_plans() {
    // The output's empty axis carries the unreachable stride, so the write
    // side has to survive the same preparation.
    let a = lay(&[2, 0]);
    let b = lay(&[0, 0]);
    let d = laying(&[2, 0], &[1, isize::MAX as i64]);
    let p = build(&a, &[0, 1], &b, &[1, 2], &d, &[0, 2]);
    assert_eq!((p.stats.m, p.stats.n, p.stats.k), (2, 0, 0));
}

#[test]
fn plain_matmul_folds_to_pure_gemm() {
    // C[i,j] = A[i,k] B[k,j], all column-major.
    let a = lay(&[4, 5]);
    let b = lay(&[5, 6]);
    let d = lay(&[4, 6]);
    let p = build(&a, &[0, 2], &b, &[2, 1], &d, &[0, 1]);
    assert_eq!(
        (p.stats.m, p.stats.n, p.stats.k, p.stats.batch),
        (4, 6, 5, 1)
    );
    assert!(p.stats.is_pure_gemm);
}

#[test]
fn adjacent_free_indices_fold() {
    // D[a,b,j] = A[a,b,k] B[k,j] with a,b adjacent in both A and D.
    let a = lay(&[4, 3, 5]);
    let b = lay(&[5, 6]);
    let d = lay(&[4, 3, 6]);
    let p = build(&a, &[0, 1, 3], &b, &[3, 2], &d, &[0, 1, 2]);
    assert_eq!(p.stats.m_axes.len(), 1, "a and b should fold into one axis");
    assert_eq!(p.stats.m, 12);
    assert!(p.stats.is_pure_gemm);
}

#[test]
fn hadamard_index_recognised() {
    // D[b,i,j] = A[b,i,k] B[b,k,j]
    let a = lay(&[2, 4, 5]);
    let b = lay(&[2, 5, 6]);
    let d = lay(&[2, 4, 6]);
    let p = build(&a, &[9, 0, 3], &b, &[9, 3, 1], &d, &[9, 0, 1]);
    assert_eq!(p.stats.batch, 2);
    assert_eq!((p.stats.m, p.stats.n, p.stats.k), (4, 6, 5));
}

#[test]
fn isolated_index_becomes_zero_stride_contraction() {
    // D[i,j] = sum_{k,r} A[i,k,r] B[k,j]: r is isolated in A.
    let a = lay(&[4, 5, 3]);
    let b = lay(&[5, 6]);
    let d = lay(&[4, 6]);
    let p = build(&a, &[0, 2, 7], &b, &[2, 1], &d, &[0, 1]);
    assert_eq!(p.stats.k, 15, "contraction extent is 5*3");
    // B must re-read the same column for each r.
    assert_eq!(p.b_k.len(), 15);
    assert_eq!(&p.b_k[..6], &[0, 1, 2, 3, 4, 0]);
}

#[test]
fn repeated_label_takes_diagonal() {
    // D[j] = sum_i A[i,i] B[i,j] -- A's trace-like diagonal.
    let a = lay(&[4, 4]);
    let b = lay(&[4, 6]);
    let d = lay(&[6]);
    let p = build(&a, &[0, 0], &b, &[0, 1], &d, &[1]);
    assert_eq!(p.stats.k, 4);
    // stride 1 + stride 4 = 5
    assert_eq!(p.a_k, vec![0, 5, 10, 15]);
}
