//! Leading-dimension rules per op, and footprints.

use super::*;

const OPS: [Trans; 3] = [Trans::N, Trans::T, Trans::C];
const M: usize = 3;
const N: usize = 5;
const K: usize = 7;

/// BLAS's minimum leading dimension of each operand for distinct m, n, k.
fn minimum(ops: GemmOps) -> [usize; 3] {
    let lda = if ops.a == Trans::N { M } else { K };
    let ldb = if ops.b == Trans::N { K } else { N };
    [lda, ldb, M]
}

fn ld_mut(job: &mut GemmJob, operand: Operand) -> &mut usize {
    match operand {
        Operand::A => &mut job.lda,
        Operand::B => &mut job.ldb,
        Operand::C => &mut job.ldc,
    }
}

#[test]
fn compact_is_the_blas_minimum_for_every_op_pair() {
    for a in OPS {
        for b in OPS {
            let ops = GemmOps { a, b };
            let job = GemmJob::compact(ops, [0; 3], M, N, K);
            assert_eq!([job.lda, job.ldb, job.ldc], minimum(ops), "{ops:?}");
            assert!(batch(ops, vec![job]).is_ok(), "{ops:?}");
        }
    }
}

#[test]
fn one_below_the_minimum_is_rejected_for_every_op_and_operand() {
    for a in OPS {
        for b in OPS {
            let ops = GemmOps { a, b };
            for operand in Operand::ALL {
                let min = minimum(ops)[operand.index()];
                let mut job = GemmJob::compact(ops, [0; 3], M, N, K);
                *ld_mut(&mut job, operand) = min - 1;
                assert_eq!(
                    batch(ops, vec![job]).unwrap_err(),
                    GemmError::LeadingDim {
                        job: 0,
                        operand,
                        ld: min - 1,
                        min
                    },
                    "{ops:?} {operand}"
                );
            }
        }
    }
}

#[test]
fn zero_leading_dimension_is_rejected_even_for_empty_extents() {
    let mut job = at(0, 0, 0, 0);
    assert_eq!((job.lda, job.ldb, job.ldc), (1, 1, 1));
    job.ldb = 0;
    assert_eq!(
        batch(NN, vec![job]).unwrap_err(),
        GemmError::LeadingDim {
            job: 0,
            operand: Operand::B,
            ld: 0,
            min: 1
        }
    );
}

#[test]
fn a_is_checked_before_b_before_c_within_a_job() {
    let mut job = at(0, 2, 2, 2);
    (job.lda, job.ldb, job.ldc) = (1, 1, 1);
    assert!(matches!(
        batch(NN, vec![job]),
        Err(GemmError::LeadingDim {
            operand: Operand::A,
            ..
        })
    ));
}

#[test]
fn footprints_follow_the_stored_shape_and_leading_dimension() {
    // op(A) = Aᵀ: stored A is k×m = 7×3 with lda = 9: (3-1)·9 + 7 = 25.
    // op(B) = Bᴴ: stored B is n×k = 5×7 with ldb = 5: (7-1)·5 + 5 = 35.
    // C is m×n = 3×5 with ldc = 4: (5-1)·4 + 3 = 19.
    let ops = GemmOps {
        a: Trans::T,
        b: Trans::C,
    };
    let job = GemmJob {
        lda: 9,
        ldc: 4,
        ..GemmJob::compact(ops, [1, 2, 3], M, N, K)
    };
    let b = batch(ops, vec![job]).unwrap();
    let lens = Operand::ALL.map(|op| b.required_len(op));
    assert_eq!(lens, [1 + 25, 2 + 35, 3 + 19]);
}

#[test]
fn k_zero_leaves_inputs_empty_but_writes_c() {
    let job = GemmJob::compact(NN, [4, 6, 2], 2, 3, 0);
    let b = batch(NN, vec![job]).unwrap();
    // Empty inputs still count their offsets, so a consumer can slice them.
    assert_eq!(Operand::ALL.map(|op| b.required_len(op)), [4, 6, 2 + 6]);
    assert_eq!(b.output_order(), [0]);
    assert_eq!(b.work(), 0);
}

#[test]
fn zero_size_outputs_are_not_outputs() {
    let jobs = vec![at(0, 2, 2, 1), at(1, 0, 2, 1), at(1, 2, 0, 1)];
    let b = batch(NN, jobs).unwrap();
    assert_eq!(b.output_order(), [0]);
    assert_eq!(b.required_len(Operand::C), 4);
}

#[test]
fn empty_batch() {
    let b = batch(NN, Vec::new()).unwrap();
    assert_eq!(Operand::ALL.map(|op| b.required_len(op)), [0; 3]);
    assert_eq!((b.groups().len(), b.runs().len()), (0, 0));
    assert!(b.output_order().is_empty());
    assert_eq!(b.work(), 0);
    assert!(b.check_lens(0, 0, 0).is_ok());
}

#[test]
fn work_sums_m_n_k() {
    let b = batch(NN, vec![at(0, 2, 3, 4), at(6, 1, 1, 5), at(7, 0, 9, 9)]).unwrap();
    assert_eq!((b.work(), b.max_job_work()), (29, 24));
}
