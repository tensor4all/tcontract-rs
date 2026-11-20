//! Groups (equal shape) and strided runs (tensor4all/cpueinsum-rs#78).

use super::*;

fn groups(b: &GemmBatch) -> Vec<Vec<usize>> {
    b.groups().map(|g| g.ids.to_vec()).collect()
}

fn run(first: usize, len: usize, stride: [isize; 3]) -> GemmRun {
    GemmRun { first, len, stride }
}

fn single(first: usize) -> GemmRun {
    run(first, 1, [0; 3])
}

#[test]
fn groups_are_equal_shapes_in_lowest_id_order() {
    let jobs = vec![
        at(0, 2, 1, 1), // shape X
        at(2, 1, 1, 1), // shape Y
        at(3, 2, 1, 1), // X
        at(5, 3, 1, 1), // Z
        at(8, 1, 1, 1), // Y
        at(9, 2, 1, 1), // X
    ];
    let b = batch(NN, jobs).unwrap();
    assert_eq!(b.groups().len(), 3);
    assert_eq!(groups(&b), [vec![0, 2, 5], vec![1, 4], vec![3]]);
    let ms: Vec<_> = b.groups().map(|g| g.shape.m).collect();
    assert_eq!(ms, [2, 1, 3]);
}

#[test]
fn leading_dimensions_split_groups() {
    let wide = GemmJob {
        ldc: 3,
        ..at(4, 2, 2, 1)
    };
    let b = batch(NN, vec![at(0, 2, 2, 1), wide, at(10, 2, 2, 1)]).unwrap();
    assert_eq!(groups(&b), [vec![0, 2], vec![1]]);
}

#[test]
fn runs_of_empty_and_singleton_lists() {
    assert!(batch(NN, Vec::new()).unwrap().runs().is_empty());
    assert_eq!(batch(NN, vec![at(0, 2, 2, 2)]).unwrap().runs(), [single(0)]);
}

#[test]
fn one_long_run_with_independent_strides() {
    // A advances by 6, B is shared (stride 0), C advances by its footprint.
    let jobs = (0..5)
        .map(|i| GemmJob::compact(NN, [6 * i, 0, 4 * i], 2, 2, 3))
        .collect();
    assert_eq!(batch(NN, jobs).unwrap().runs(), [run(0, 5, [6, 0, 4])]);
}

#[test]
fn input_strides_may_be_negative() {
    let jobs = (0..3)
        .map(|i| GemmJob::compact(NN, [8 - 4 * i, 0, 4 * i], 2, 2, 2))
        .collect();
    assert_eq!(batch(NN, jobs).unwrap().runs(), [run(0, 3, [-4, 0, 4])]);
}

#[test]
fn stride_break_and_greedy_from_the_left() {
    // C at 0, 10, 20, 25, 30: [0, 10, 20] by 10, then [25, 30] by 5.
    let jobs = [0, 10, 20, 25, 30].map(|c| at(c, 1, 1, 1)).to_vec();
    assert_eq!(
        batch(NN, jobs).unwrap().runs(),
        [run(0, 3, [0, 0, 10]), run(3, 2, [0, 0, 5])]
    );
}

#[test]
fn input_stride_break_ends_a_run() {
    let jobs = [0, 1, 3]
        .into_iter()
        .zip([0, 4, 8])
        .map(|(a, c)| GemmJob::compact(NN, [a, 0, c], 2, 2, 1))
        .collect();
    assert_eq!(
        batch(NN, jobs).unwrap().runs(),
        [run(0, 2, [1, 0, 4]), single(2)]
    );
}

#[test]
fn shape_break_and_interleaved_shapes() {
    let jobs = vec![
        at(0, 1, 1, 1),
        at(1, 1, 1, 1),
        at(2, 2, 1, 1),
        at(4, 2, 1, 1),
    ];
    assert_eq!(
        batch(NN, jobs).unwrap().runs(),
        [run(0, 2, [0, 0, 1]), run(2, 2, [0, 0, 2])]
    );
    // Alternating shapes: runs are contiguous in original order, so none forms.
    let jobs = vec![
        at(0, 1, 1, 1),
        at(1, 2, 1, 1),
        at(3, 1, 1, 1),
        at(4, 2, 1, 1),
    ];
    assert_eq!(
        batch(NN, jobs).unwrap().runs(),
        [single(0), single(1), single(2), single(3)]
    );
}

#[test]
fn leading_dimension_break_ends_a_run() {
    let wide = GemmJob {
        lda: 5,
        ..GemmJob::compact(NN, [2, 0, 4], 2, 2, 1)
    };
    let jobs = vec![GemmJob::compact(NN, [0, 0, 0], 2, 2, 1), wide];
    assert_eq!(batch(NN, jobs).unwrap().runs(), [single(0), single(1)]);
}

#[test]
fn zero_or_negative_output_stride_forms_no_run() {
    // Empty outputs at one offset: valid, but a zero C stride proves nothing.
    let jobs = vec![at(3, 0, 2, 1); 3];
    assert_eq!(
        batch(NN, jobs).unwrap().runs(),
        [single(0), single(1), single(2)]
    );
    // Descending outputs are disjoint, but the run rule wants a positive stride.
    let jobs = [8, 4, 0].map(|c| at(c, 2, 2, 1)).to_vec();
    assert_eq!(
        batch(NN, jobs).unwrap().runs(),
        [single(0), single(1), single(2)]
    );
}

#[test]
fn output_stride_must_cover_the_ldc_footprint() {
    // C footprint (2-1)·4 + 2 = 6: stride 6 forms a run, and the outputs of
    // any smaller positive stride already overlap (rejected by `new`).
    let job = |c| GemmJob {
        ldc: 4,
        ..at(c, 2, 2, 1)
    };
    assert_eq!(
        batch(NN, vec![job(0), job(6), job(12)]).unwrap().runs(),
        [run(0, 3, [0, 0, 6])]
    );
    assert!(batch(NN, vec![job(0), job(5)]).is_err());
}

#[test]
fn runs_cover_every_job_in_order() {
    let jobs = [0, 1, 2, 7, 9, 11, 12, 20].map(|c| at(c, 1, 1, 1)).to_vec();
    let b = batch(NN, jobs).unwrap();
    let mut next = 0;
    for r in b.runs() {
        assert_eq!(r.first, next);
        assert!(r.len >= 1);
        next += r.len;
    }
    assert_eq!(next, 8);
}
