//! Lowering: diagonals, reductions, case 5, the C mode, validation order and
//! the equivalence of the two front ends.
use tcontract::api::{
    AliasError, CSpec, ConfigError, DType, DotGeneral, Error, Labels, LayoutSpec, Op, OperandId,
    OperandSpec, Problem, ShapeError, Unsupported,
};

fn op(d: &[usize], s: &[isize]) -> OperandSpec {
    OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap())
}

fn p(
    a: OperandSpec,
    b: OperandSpec,
    c: CSpec,
    d: OperandSpec,
    l: &Labels,
) -> Result<Problem, Error> {
    Problem::from_labels(DType::F64, a, b, c, d, l)
}

#[test]
fn a_matrix_product_has_one_axis_per_role() {
    let q = p(
        op(&[2, 3], &[1, 2]),
        op(&[3, 4], &[1, 3]),
        CSpec::Absent,
        op(&[2, 4], &[1, 2]),
        &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    )
    .unwrap();
    let r = q.roles();
    assert_eq!(
        (r.m().len(), r.n().len(), r.k().len(), r.h().len()),
        (1, 1, 1, 0)
    );
    let k = r.k()[0];
    assert_eq!(
        (
            k.extent(),
            k.stride(OperandId::A),
            k.stride(OperandId::B),
            k.stride(OperandId::D)
        ),
        (3, 2, 1, 0)
    );
    assert_eq!(q.macs(), 24);
    assert!(!q.all_batch() && !q.k_empty() && !q.out_empty());
}

#[test]
fn repeated_labels_sum_their_strides_on_every_operand() {
    // A[i,i] selects the diagonal of a 3x3 matrix (stride 1 + 3); D[j,j] writes
    // only the diagonal, so it stays injective (stride 4).
    let q = p(
        op(&[3, 3], &[1, 3]),
        op(&[3], &[1]),
        CSpec::Absent,
        op(&[3, 3], &[1, 3]),
        &Labels::new(&[0, 0], &[1], &[1, 1]),
    )
    .unwrap();
    let r = q.roles();
    // label 0 is only in A: a reduction; label 1 is in B and D: N.
    assert_eq!((r.m().len(), r.n().len(), r.k().len()), (0, 1, 1));
    assert_eq!(r.k()[0].stride(OperandId::A), 4);
    assert!(r.k()[0].in_a() && !r.k()[0].in_b());
    assert_eq!(r.n()[0].stride(OperandId::D), 4);
    // Unequal repeated extents are a shape error with the label named.
    let e = p(
        op(&[3, 2], &[1, 3]),
        op(&[3], &[1]),
        CSpec::Absent,
        op(&[3], &[1]),
        &Labels::new(&[0, 0], &[1], &[1]),
    )
    .unwrap_err();
    assert!(matches!(
        e,
        Error::Shape(ShapeError::ExtentMismatch {
            label: 0,
            expected: 3,
            found: 2
        })
    ));
}

#[test]
fn an_isolated_label_is_a_reduction_and_an_output_only_label_is_unsupported() {
    // D[i] = sum_j A[i,j] * B[] : j is isolated in A.
    let q = p(
        op(&[2, 3], &[1, 2]),
        op(&[], &[]),
        CSpec::Absent,
        op(&[2], &[1]),
        &Labels::new(&[0, 1], &[], &[0]),
    )
    .unwrap();
    assert_eq!(q.roles().k().len(), 1);
    assert_eq!(q.roles().k()[0].stride(OperandId::B), 0);
    // TAPP case 5: x only in D.
    let e = p(
        op(&[2, 2], &[1, 2]),
        op(&[2, 2], &[1, 2]),
        CSpec::Absent,
        op(&[2, 2, 2], &[1, 2, 4]),
        &Labels::new(&[0, 1], &[1, 0], &[0, 1, 9]),
    )
    .unwrap_err();
    assert!(matches!(
        e,
        Error::Unsupported(Unsupported::OutputOnlyLabel { label: 9 })
    ));
    assert!(e.is_unsupported());
}

#[test]
fn counts_c_labels_and_injectivity_are_validated_with_typed_errors() {
    let l = Labels::new(&[0, 2], &[2, 1], &[0, 1]);
    let (a, b, d) = (
        op(&[2, 3], &[1, 2]),
        op(&[3, 4], &[1, 3]),
        op(&[2, 4], &[1, 2]),
    );
    // Too few labels for the rank.
    let e = p(
        a.clone(),
        b.clone(),
        CSpec::Absent,
        d.clone(),
        &Labels::new(&[0], &[2, 1], &[0, 1]),
    )
    .unwrap_err();
    assert!(matches!(
        e,
        Error::Shape(ShapeError::LabelCount {
            operand: OperandId::A,
            modes: 2,
            labels: 1
        })
    ));
    // C labels without a separate C, and a separate C without labels.
    assert!(matches!(
        p(
            a.clone(),
            b.clone(),
            CSpec::Absent,
            d.clone(),
            &l.clone().with_c(&[0, 1])
        )
        .unwrap_err(),
        Error::Config(ConfigError::CLabels)
    ));
    assert!(matches!(
        p(
            a.clone(),
            b.clone(),
            CSpec::Separate(d.clone()),
            d.clone(),
            &l
        )
        .unwrap_err(),
        Error::Config(ConfigError::CLabels)
    ));
    // C and D with different label sets.
    let e = p(
        a.clone(),
        b.clone(),
        CSpec::Separate(op(&[2], &[1])),
        d.clone(),
        &l.clone().with_c(&[0]),
    )
    .unwrap_err();
    assert!(matches!(e, Error::Shape(ShapeError::OutputLabelMismatch)));
    // A zero stride in D addresses an element twice.
    let e = p(a, b, CSpec::Absent, op(&[2, 4], &[1, 0]), &l).unwrap_err();
    assert!(matches!(e, Error::Alias(AliasError::OutputNotInjective)));
}

#[test]
fn a_separate_c_may_differ_in_layout_and_the_problem_knows_whether_it_matches_d() {
    let l = Labels::new(&[0, 2], &[2, 1], &[0, 1]);
    let (a, b, d) = (
        op(&[2, 3], &[1, 2]),
        op(&[3, 4], &[1, 3]),
        op(&[2, 4], &[1, 2]),
    );
    let same = p(
        a.clone(),
        b.clone(),
        CSpec::Separate(op(&[2, 4], &[1, 2])),
        d.clone(),
        &l.clone().with_c(&[0, 1]),
    )
    .unwrap();
    assert!(same.c_matches_d());
    // Row-major C with column-major D: same labels, different mapping.
    let other = p(
        a.clone(),
        b.clone(),
        CSpec::Separate(op(&[2, 4], &[4, 1])),
        d.clone(),
        &l.clone().with_c(&[0, 1]),
    )
    .unwrap();
    assert!(!other.c_matches_d());
    // The labels may come in another order and still describe the same mapping.
    let swapped = p(
        a,
        b,
        CSpec::Separate(op(&[4, 2], &[2, 1])),
        d,
        &l.with_c(&[1, 0]),
    )
    .unwrap();
    assert!(swapped.c_matches_d());
    assert_eq!(other.op_c(), Op::Identity);
}

#[test]
fn dot_general_and_labels_lower_to_the_same_roles() {
    // Batched matrix product: batch axis 0, contract A[2] with B[1].
    let dot = DotGeneral::new(&[2], &[1], &[0], &[0]);
    let (a, b) = (op(&[3, 5, 4], &[1, 3, 15]), op(&[3, 4, 2], &[1, 3, 12]));
    let d = op(&[5, 2, 3], &[1, 5, 10]);
    let from_dot =
        Problem::from_dot_general(DType::F64, a.clone(), b.clone(), d.clone(), &dot).unwrap();
    let from_labels = p(
        a,
        b,
        CSpec::Output(Op::Identity),
        d,
        // lhs_free = [1], rhs_free = [2], batch = 0, contract = 4.
        &Labels::new(&[0, 1, 2], &[0, 2, 3], &[1, 3, 0]),
    )
    .unwrap();
    let strides = |q: &Problem| {
        let mut v: Vec<(usize, [isize; 4])> = q
            .roles()
            .m()
            .iter()
            .chain(q.roles().n())
            .chain(q.roles().k())
            .chain(q.roles().h())
            .map(|x| {
                (
                    x.extent(),
                    [OperandId::A, OperandId::B, OperandId::C, OperandId::D].map(|o| x.stride(o)),
                )
            })
            .collect();
        v.sort();
        v
    };
    assert_eq!(strides(&from_dot), strides(&from_labels));
    assert_eq!(from_dot.macs(), 3 * 5 * 2 * 4);
}

#[test]
fn dot_general_errors_keep_their_categories() {
    let (a, b) = (op(&[3, 4], &[1, 3]), op(&[4, 2], &[1, 4]));
    let d = op(&[3, 2], &[1, 3]);
    let go = |dot: &DotGeneral, d: &OperandSpec| {
        Problem::from_dot_general(DType::F64, a.clone(), b.clone(), d.clone(), dot).unwrap_err()
    };
    assert!(matches!(
        go(&DotGeneral::new(&[1], &[0, 1], &[], &[]), &d),
        Error::Config(ConfigError::AxisListLength { .. })
    ));
    assert!(matches!(
        go(&DotGeneral::new(&[5], &[0], &[], &[]), &d),
        Error::Config(ConfigError::AxisOutOfRange {
            operand: OperandId::A,
            axis: 5,
            rank: 2
        })
    ));
    assert!(matches!(
        go(&DotGeneral::new(&[1], &[0], &[1], &[1]), &d),
        Error::Config(ConfigError::AxisRepeated {
            operand: OperandId::A,
            axis: 1
        })
    ));
    assert!(matches!(
        go(&DotGeneral::new(&[0], &[0], &[], &[]), &d),
        Error::Shape(ShapeError::PairedExtent { .. })
    ));
    assert!(matches!(
        go(
            &DotGeneral::new(&[1], &[0], &[], &[]),
            &op(&[2, 3], &[1, 2])
        ),
        Error::Shape(ShapeError::OutputExtents { .. })
    ));
}

#[test]
fn zero_extents_and_arithmetic_overflow_are_decided_at_construction() {
    // K of extent zero: the product has no terms; the output is non-empty.
    let q = p(
        op(&[2, 0], &[1, 2]),
        op(&[0, 4], &[1, 1]),
        CSpec::Absent,
        op(&[2, 4], &[1, 2]),
        &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    )
    .unwrap();
    assert!(q.k_empty() && !q.out_empty());
    let empty = p(
        op(&[0, 3], &[1, 1]),
        op(&[3, 4], &[1, 3]),
        CSpec::Absent,
        op(&[0, 4], &[1, 1]),
        &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    )
    .unwrap();
    assert!(empty.out_empty());
    // An extent product beyond the address space.
    let big = usize::MAX / 2;
    let e = p(
        op(&[big, 3], &[1, 1]),
        op(&[3], &[1]),
        CSpec::Absent,
        op(&[big], &[1]),
        &Labels::new(&[0, 1], &[1], &[0]),
    );
    assert!(
        matches!(e, Err(Error::Shape(ShapeError::Overflow { .. }))),
        "{e:?}"
    );
    // Negative strides are fine and widen the span downwards.
    let q = p(
        op(&[3], &[-1]),
        op(&[3], &[1]),
        CSpec::Absent,
        op(&[], &[]),
        &Labels::new(&[0], &[0], &[]),
    )
    .unwrap();
    let span = q.span(OperandId::A).unwrap();
    assert_eq!((span.lo(), span.hi()), (-2, 0));
}
