//! Checked arithmetic, `isize` limits, lowest-id errors and buffer lengths.

use super::*;

const BIG: usize = isize::MAX as usize;

fn overflow_what(scalar: ScalarKind, job: GemmJob) -> &'static str {
    match GemmBatch::new(scalar, NN, vec![job]) {
        Err(GemmError::Overflow { job: 0, what }) => what,
        other => panic!("expected an overflow, got {other:?}"),
    }
}

#[test]
fn extents_and_leading_dimensions_must_fit_isize() {
    assert_eq!(overflow_what(ScalarKind::F32, at(0, BIG + 1, 0, 0)), "m");
    assert_eq!(overflow_what(ScalarKind::F32, at(0, 0, BIG + 1, 0)), "n");
    assert_eq!(overflow_what(ScalarKind::F32, at(0, 0, 0, BIG + 1)), "k");
    let job = GemmJob {
        ldb: BIG + 1,
        ..at(0, 0, 0, 0)
    };
    assert_eq!(overflow_what(ScalarKind::F32, job), "ldb");
}

#[test]
fn span_end_and_byte_end_overflow_are_reported_per_operand() {
    // (3-1)·isize::MAX + 2 = 2^64: the span itself overflows `usize`.
    let job = GemmJob {
        ldc: BIG,
        ..at(0, 2, 3, 0)
    };
    assert_eq!(overflow_what(ScalarKind::F32, job), "C span");
    // offset + 1 overflows.
    assert_eq!(
        overflow_what(ScalarKind::F32, at(usize::MAX, 1, 1, 0)),
        "C end"
    );
    // An A offset that overflows once the span is added.
    let job = GemmJob {
        a_offset: usize::MAX,
        ..at(0, 1, 1, 1)
    };
    assert_eq!(overflow_what(ScalarKind::F32, job), "A end");
}

#[test]
fn byte_end_limit_depends_on_the_scalar_kind() {
    // End 2^60 elements: 2^62 bytes as f32 fits, 2^63 bytes as f64 does not.
    let job = at((1 << 60) - 1, 1, 1, 0);
    assert!(GemmBatch::new(ScalarKind::F32, NN, vec![job]).is_ok());
    assert_eq!(overflow_what(ScalarKind::F64, job), "C byte end");
    assert_eq!(overflow_what(ScalarKind::C32, job), "C byte end");
}

#[test]
fn the_lowest_failing_job_is_reported_whatever_the_kind() {
    let mut bad_ld = at(10, 2, 1, 1);
    bad_ld.ldc = 1;
    let overflowing = at(usize::MAX, 1, 1, 0);
    let err = batch(NN, vec![at(0, 1, 1, 1), bad_ld, overflowing]).unwrap_err();
    assert!(matches!(err, GemmError::LeadingDim { job: 1, .. }), "{err}");
    let err = batch(NN, vec![at(0, 1, 1, 1), overflowing, bad_ld]).unwrap_err();
    assert!(matches!(err, GemmError::Overflow { job: 1, .. }), "{err}");
}

#[test]
fn check_lens_names_the_lowest_job_past_the_end_not_the_longest() {
    // C ends: job 0 at 8, job 1 at 4, job 2 at 12.
    let b = batch(NN, vec![at(4, 2, 2, 1), at(0, 2, 2, 1), at(8, 2, 2, 1)]).unwrap();
    assert_eq!(b.required_len(Operand::C), 12);
    assert!(b.check_lens(2, 2, 12).is_ok());
    assert_eq!(
        b.check_lens(2, 2, 6),
        Err(GemmError::Bounds {
            job: 0,
            operand: Operand::C,
            end: 8,
            available: 6
        })
    );
}

#[test]
fn check_lens_checks_a_then_b_then_c() {
    let b = batch(NN, vec![GemmJob::compact(NN, [1, 2, 3], 2, 2, 2)]).unwrap();
    assert_eq!(Operand::ALL.map(|op| b.required_len(op)), [5, 6, 7]);
    let short = |a, bb, c| match b.check_lens(a, bb, c) {
        Err(GemmError::Bounds { operand, .. }) => Some(operand),
        _ => None,
    };
    assert_eq!(short(4, 5, 6), Some(Operand::A));
    assert_eq!(short(5, 5, 6), Some(Operand::B));
    assert_eq!(short(5, 6, 6), Some(Operand::C));
    assert_eq!(short(5, 6, 7), None);
}

#[test]
fn errors_display_the_job_id() {
    let err = GemmError::Bounds {
        job: 7,
        operand: Operand::B,
        end: 9,
        available: 3,
    };
    assert_eq!(
        err.to_string(),
        "GEMM job 7: B footprint ends at 9, buffer has 3 elements"
    );
    let ld = GemmError::LeadingDim {
        job: 2,
        operand: Operand::A,
        ld: 1,
        min: 4,
    };
    assert_eq!(ld.to_string(), "GEMM job 2: lda = 1 is below the minimum 4");
}

#[test]
fn batch_is_an_owned_shareable_value() {
    fn assert_owned<T: Clone + Send + Sync + 'static>() {}
    assert_owned::<GemmBatch>();
    assert_owned::<GemmError>();
}

#[test]
fn debug_summarises_instead_of_listing_every_job() {
    let jobs = (0..100).map(|i| at(i, 1, 1, 1)).collect();
    let text = format!("{:?}", batch(NN, jobs).unwrap());
    assert!(text.contains("jobs: 100"), "{text}");
    assert_eq!(text.matches("GemmJob").count(), 4, "{text}");
}
