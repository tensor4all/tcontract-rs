# t4a-desc

Device-neutral kernel descriptors for the tensor4all stack. A descriptor says
what a batched kernel computes over flat buffers, element offsets, extents and
leading dimensions. It holds no buffer, pointer, device, executor, plan or
scalar value. CPU routes and GPU backends consume the same validated
descriptor, and the crate has no dependency at all, so libraries share it
without depending on each other (tensor4all/tcontract-rs#209).

| Item | Role |
| --- | --- |
| `ScalarKind` | The element type of the buffers: f32, f64, c32 or c64, the set cuBLAS grouped GEMM and cuTENSOR's block-sparse contraction share. |
| `gemm::GemmJob`, `gemm::Trans`, `gemm::GemmOps` | One problem of a grouped GEMM in NVIDIA's model: element offsets into one base buffer per operand, `m`, `n`, `k`, `lda`, `ldb`, `ldc`, and the call's ops N/T/C. `C ← alpha·op(A)·op(B) + beta·C`, column-major, in place. |
| `gemm::GemmBatch` | The validated, immutable job list: leading-dimension rules, checked footprints and pairwise disjoint outputs, all checked once before any consumer writes, with errors naming the lowest original job id. It also stores the cuBLAS groups (equal `m, n, k, lda, ldb, ldc`) and the strided runs (tensor4all/cpueinsum-rs#78), computed once. |

Offsets, not pointers, keep the CPU API safe: a GPU backend turns base +
offset into its pointer arrays once at its own prepare time. `alpha` and
`beta` are execute arguments of the consuming plan, not part of the
descriptor. The field-by-field mapping to `cublasGemmGroupedBatchedEx` and to
CUTLASS `GemmGrouped::Arguments` is in the rustdoc of the `gemm` module.

A block-sparse tensor descriptor in cuTENSOR's model
(tensor4all/cpueinsum-rs#81) will be a sibling of `gemm` in this crate.
