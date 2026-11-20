//! The layout half of the uninitialized-output entry.

use super::*;

fn fresh(jobs: Vec<GemmJob>, c_len: usize) -> Result<(), GemmError> {
    batch(NN, jobs).unwrap().check_fresh_cover(c_len)
}

#[test]
fn exact_tiling_in_any_job_order() {
    assert_eq!(
        fresh(vec![at(6, 2, 2, 1), at(0, 3, 2, 1), at(10, 1, 3, 1)], 13),
        Ok(())
    );
}

#[test]
fn ldc_must_equal_m() {
    let wide = |c| GemmJob {
        ldc: 3,
        ..at(c, 2, 2, 1)
    };
    assert_eq!(
        fresh(vec![at(0, 1, 1, 1), wide(10), wide(1)], 20),
        Err(GemmError::FreshLayout { job: 1 })
    );
}

#[test]
fn empty_jobs_are_ignored() {
    // m = 0 gives ldc = 1 != m, and the offset lies past the output: neither counts.
    assert_eq!(fresh(vec![at(0, 2, 2, 1), at(99, 0, 2, 1)], 4), Ok(()));
}

#[test]
fn a_gap_names_the_lowest_job_after_one() {
    // Outputs [0, 2), [4, 6), [8, 10): job 2 (at 4) and job 0 (at 8) follow gaps.
    assert_eq!(
        fresh(vec![at(8, 2, 1, 1), at(0, 2, 1, 1), at(4, 2, 1, 1)], 10),
        Err(GemmError::FreshCoverage { job: Some(0) })
    );
    // Not starting at 0 is a gap before the first output.
    assert_eq!(
        fresh(vec![at(1, 2, 1, 1)], 3),
        Err(GemmError::FreshCoverage { job: Some(0) })
    );
}

#[test]
fn a_prefix_of_the_wrong_length() {
    let jobs = || vec![at(0, 2, 2, 1), at(4, 2, 2, 1)];
    assert_eq!(
        fresh(jobs(), 9),
        Err(GemmError::FreshCoverage { job: None })
    );
    assert_eq!(
        fresh(jobs(), 7),
        Err(GemmError::FreshCoverage { job: None })
    );
    assert_eq!(fresh(jobs(), 8), Ok(()));
}

#[test]
fn no_output_covers_only_an_empty_buffer() {
    assert_eq!(fresh(Vec::new(), 0), Ok(()));
    assert_eq!(fresh(vec![at(0, 0, 3, 1)], 0), Ok(()));
    assert_eq!(
        fresh(Vec::new(), 3),
        Err(GemmError::FreshCoverage { job: None })
    );
}
