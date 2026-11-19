//! A grouped-GEMM job list in NVIDIA's grouped-GEMM model.
//!
//! Each job is one column-major problem with BLAS meaning: `op(A)` is `m×k`,
//! `op(B)` is `k×n`, `C` is `m×n`, and
//!
//! ```text
//! C ← alpha·op(A)·op(B) + beta·C        (in place, as in cuBLAS)
//! ```
//!
//! Operands are **element offsets into one base buffer per operand**, not
//! pointers, so the CPU API stays safe. A GPU backend turns base + offset into
//! its pointer arrays once at its own prepare time; converting to per-block
//! pointers (cuTENSOR block-sparse, CUTLASS) is the same step.
//!
//! The raw descriptor is [`GemmJob`]: plain data with public fields and no
//! invariants. [`GemmBatch::new`] validates a list of them once, for one
//! [`ScalarKind`](crate::ScalarKind) and one pair of [`GemmOps`], before any
//! consumer writes. The batch then stores what every consumer needs: the
//! required buffer lengths, the output order that proves disjointness, the
//! groups of equal geometry, and the strided runs.
//!
//! # Per-call values
//!
//! cuBLAS takes the ops, `alpha` and `beta` per group. Here they are per call,
//! which is a subset of that model and maps onto it by giving every group the
//! same values:
//!
//! - the ops and the scalar kind are prepare-time values and live in
//!   [`GemmBatch`], because the leading-dimension rules depend on the ops and
//!   the byte-span checks on the element size;
//! - `alpha` and `beta` are execute arguments of the consuming plan. This
//!   crate holds no scalar value; a consumer that offers an uninitialized-output
//!   entry rejects `beta ≠ 0` itself and asks
//!   [`GemmBatch::check_fresh_cover`] for the layout half.
//!
//! # Mapping to cuBLAS and CUTLASS
//!
//! Checked against `cublas_api.h` of cuBLAS 13.1.1 and against
//! `include/cutlass/gemm/kernel/gemm_grouped.h` of CUTLASS (`main` at
//! `dc6ae46eb5`, 2026-11-20).
//!
//! | Here | `cublasGemmGroupedBatchedEx` | CUTLASS `kernel::GemmGrouped::Arguments` |
//! | --- | --- | --- |
//! | [`GemmJob::m`], [`n`](GemmJob::n), [`k`](GemmJob::k) | `m_array`, `n_array`, `k_array` (per group; `int`, or `int64_t` in the `_64` form) | `problem_sizes[i]` (`GemmCoord`), and `host_problem_sizes` |
//! | [`GemmJob::lda`], [`ldb`](GemmJob::ldb), [`ldc`](GemmJob::ldc) | `lda_array`, `ldb_array`, `ldc_array` (per group) | `lda`, `ldb`, `ldc` per problem, and `ldd = ldc` |
//! | [`GemmJob::a_offset`], [`b_offset`](GemmJob::b_offset), [`c_offset`](GemmJob::c_offset) | `Aarray`, `Barray`, `Carray`: one pointer per problem, base + offset | `ptr_A`, `ptr_B`, `ptr_C`, and `ptr_D = ptr_C` |
//! | [`GemmOps::a`], [`GemmOps::b`] | `transa_array`, `transb_array` (per group; one value for every group) | the `LayoutA`/`LayoutB` and `kTransformA`/`kTransformB` (`ComplexTransform`) template parameters |
//! | [`GemmBatch::scalar`] | `Atype`, `Btype`, `Ctype` (one type for all three); `computeType` is the backend's choice | the `ElementA`, `ElementB`, `ElementC` template parameters |
//! | [`GemmBatch::groups`] | `group_count`, `group_size` | none: CUTLASS has no groups (`problem_count` is the job count) |
//! | `alpha`, `beta` (the consumer's execute arguments) | `alpha_array`, `beta_array` (per group) | `output_op` (`EpilogueOutputOp::Params`) |
//!
//! [`Trans::N`], [`Trans::T`] and [`Trans::C`] are `CUBLAS_OP_N`,
//! `CUBLAS_OP_T` and `CUBLAS_OP_C`. In CUTLASS, `N` is a column-major operand,
//! `T` the same buffer read as row-major with the same leading dimension, and
//! `C` that plus `ComplexTransform::kConjugate`.
//!
//! **Elementwise conjugation has no mapping.** Conjugating an operand without
//! transposing it (cpueinsum's former grouped `Op::Conjugate`) is not one of
//! N/T/C on the same storage, and cuBLAS's `CUBLAS_OP_CONJG` is a placeholder
//! its header marks "not supported in the current release". The grouped path
//! therefore offers N, T and C only.
//!
//! Not verified: whether `cublasGemmGroupedBatchedEx` accepts the complex
//! types (`CUDA_C_32F`, `CUDA_C_64F`). The typed forms in the 13.1.1 header
//! are `S` and `D` only. The descriptor does not depend on the answer.
//!
//! # Examples
//!
//! Two jobs of one geometry and one of another, sharing three buffers:
//!
//! ```
//! use t4a_desc::gemm::{GemmBatch, GemmJob, GemmOps, Operand, Trans};
//! use t4a_desc::ScalarKind;
//!
//! let ops = GemmOps { a: Trans::N, b: Trans::T };
//! let jobs = vec![
//!     GemmJob::compact(ops, [0, 0, 0], 2, 3, 4),   // A 2×4, B 3×4 (read as 4×3), C 2×3
//!     GemmJob::compact(ops, [8, 12, 6], 2, 3, 4),
//!     GemmJob::compact(ops, [16, 24, 12], 1, 1, 1),
//! ];
//! let batch = GemmBatch::new(ScalarKind::F64, ops, jobs)?;
//! assert_eq!(batch.required_len(Operand::C), 13);
//! assert_eq!(batch.groups().len(), 2);
//! assert_eq!(batch.runs().len(), 2); // jobs 0 and 1 form one strided run
//! # Ok::<(), t4a_desc::gemm::GemmError>(())
//! ```

mod batch;
mod error;
mod job;

pub use batch::{GemmBatch, GemmGroup, GemmRun};
pub use error::GemmError;
pub use job::{GemmJob, GemmOps, GemmShape, Operand, Trans};

#[cfg(test)]
mod tests;
