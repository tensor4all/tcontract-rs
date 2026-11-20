//! Unit tests of the GEMM descriptor, split by concern.

use super::{GemmBatch, GemmError, GemmJob, GemmOps, GemmRun, Operand, Trans};
use crate::ScalarKind;

mod checks;
mod fresh;
mod layout;
mod overlap;
mod partitions;

const NN: GemmOps = GemmOps {
    a: Trans::N,
    b: Trans::N,
};

/// Validate as `f64`.
fn batch(ops: GemmOps, jobs: Vec<GemmJob>) -> Result<GemmBatch, GemmError> {
    GemmBatch::new(ScalarKind::F64, ops, jobs)
}

/// A compact N/N job of `m×n×k` writing `C` at `c`, reading `A` and `B` at 0.
fn at(c: usize, m: usize, n: usize, k: usize) -> GemmJob {
    GemmJob::compact(NN, [0, 0, c], m, n, k)
}
