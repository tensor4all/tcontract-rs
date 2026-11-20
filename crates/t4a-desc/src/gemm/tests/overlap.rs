//! The conservative interval test on outputs, and the lowest-id report.

use super::*;

fn overlap(jobs: Vec<GemmJob>) -> Option<(usize, usize)> {
    match batch(NN, jobs) {
        Ok(_) => None,
        Err(GemmError::Overlap { job, other }) => Some((job, other)),
        Err(e) => panic!("unexpected error {e}"),
    }
}

#[test]
fn adjacent_outputs_are_disjoint() {
    assert_eq!(overlap(vec![at(4, 2, 2, 1), at(0, 2, 2, 1)]), None);
}

#[test]
fn one_shared_element_overlaps() {
    assert_eq!(overlap(vec![at(0, 2, 2, 1), at(3, 2, 2, 1)]), Some((0, 1)));
}

#[test]
fn identical_jobs_overlap() {
    assert_eq!(overlap(vec![at(0, 1, 1, 1); 2]), Some((0, 1)));
}

#[test]
fn interleaved_ldc_layouts_are_rejected_conservatively() {
    // C elements {0, 1, 4, 5} and {2, 3, 6, 7}: disjoint, but the intervals
    // [0, 6) and [2, 8) intersect. The documented interval test rejects them.
    let x = GemmJob {
        ldc: 4,
        ..at(0, 2, 2, 1)
    };
    let y = GemmJob { c_offset: 2, ..x };
    assert_eq!(overlap(vec![x, y]), Some((0, 1)));
}

#[test]
fn ldc_gap_is_part_of_the_interval() {
    // [0, (2-1)·4 + 2) = [0, 6): a job at 5 overlaps, a job at 6 does not.
    let x = GemmJob {
        ldc: 4,
        ..at(0, 2, 2, 1)
    };
    assert_eq!(overlap(vec![x, at(5, 1, 1, 1)]), Some((0, 1)));
    assert_eq!(overlap(vec![x, at(6, 1, 1, 1)]), None);
}

#[test]
fn reports_the_lowest_overlapping_id_not_the_first_in_offset_order() {
    // Jobs 3 and 4 overlap at the lowest offsets; jobs 0 and 1 overlap later.
    let jobs = vec![
        at(10, 4, 1, 1),
        at(12, 4, 1, 1),
        at(20, 1, 1, 1),
        at(0, 4, 1, 1),
        at(2, 4, 1, 1),
    ];
    assert_eq!(overlap(jobs), Some((0, 1)));
}

#[test]
fn a_long_output_overlapping_a_later_short_one_is_found() {
    // Job 1 lies inside job 0's interval but after job 2 in offset order.
    let jobs = vec![at(0, 100, 1, 1), at(50, 1, 1, 1), at(10, 1, 1, 1)];
    assert_eq!(overlap(jobs), Some((0, 2)));
    let jobs = vec![at(200, 1, 1, 1), at(50, 1, 1, 1), at(0, 100, 1, 1)];
    assert_eq!(overlap(jobs), Some((1, 2)));
}

#[test]
fn empty_outputs_never_overlap() {
    assert_eq!(
        overlap(vec![at(0, 2, 2, 1), at(1, 0, 2, 1), at(1, 2, 0, 1)]),
        None
    );
}

#[test]
fn per_job_errors_come_before_the_overlap_test() {
    let mut bad = at(9, 2, 1, 1);
    bad.ldc = 1;
    let err = batch(NN, vec![at(0, 1, 1, 1), at(0, 1, 1, 1), bad]).unwrap_err();
    assert!(matches!(err, GemmError::LeadingDim { job: 2, .. }), "{err}");
}
