//! Device-neutral kernel descriptors shared by the tensor4all CPU and GPU
//! backends (tensor4all/tcontract-rs#209).
//!
//! A descriptor says *what* a batched kernel computes over flat buffers,
//! offsets, extents and leading dimensions, and nothing about *how* or *where*.
//! It holds no buffer, pointer, device, executor, plan or scalar value. A CPU
//! route (cpueinsum's `GroupedPlan`, tcontract's fused executor) and a GPU
//! backend (cuBLAS grouped GEMM, CUTLASS, cuTENSOR) consume the same validated
//! descriptor without translating it into another vocabulary.
//!
//! - [`gemm`]: a grouped-GEMM job list in NVIDIA's grouped-GEMM model
//!   (`cublasGemmGroupedBatchedEx`, CUTLASS `GemmGrouped`), validated once
//!   into a [`gemm::GemmBatch`] together with its groups and strided runs.
//! - A block-sparse tensor descriptor in cuTENSOR's model
//!   (tensor4all/cpueinsum-rs#81) will be a sibling module; it is not here
//!   yet.
//!
//! Like `t4a-cost`, this crate depends on nothing but `std`, so libraries can
//! share it without depending on each other.

pub mod gemm;

/// The element type of a descriptor's buffers: the set cuBLAS grouped GEMM
/// and cuTENSOR's block-sparse contraction share.
///
/// The set is closed on purpose, without `#[non_exhaustive]`: a new kind is a
/// breaking change, so every backend's `match` has to decide whether it serves
/// it instead of falling through a wildcard arm.
///
/// # Examples
///
/// ```
/// use t4a_desc::ScalarKind;
///
/// assert_eq!(ScalarKind::C64.size_of(), 16);
/// assert!(!ScalarKind::F32.is_complex());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarKind {
    /// `f32` (cuBLAS `CUDA_R_32F`).
    F32,
    /// `f64` (cuBLAS `CUDA_R_64F`).
    F64,
    /// Complex `f32`, real and imaginary parts interleaved (`CUDA_C_32F`).
    C32,
    /// Complex `f64`, real and imaginary parts interleaved (`CUDA_C_64F`).
    C64,
}

impl ScalarKind {
    /// Bytes per element.
    ///
    /// # Examples
    ///
    /// ```
    /// use t4a_desc::ScalarKind;
    ///
    /// let sizes = [ScalarKind::F32, ScalarKind::F64, ScalarKind::C32, ScalarKind::C64]
    ///     .map(ScalarKind::size_of);
    /// assert_eq!(sizes, [4, 8, 8, 16]);
    /// ```
    pub const fn size_of(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F64 | Self::C32 => 8,
            Self::C64 => 16,
        }
    }

    /// Whether the kind is complex. For a real kind, the conjugate-transpose
    /// operation [`gemm::Trans::C`] means the same as [`gemm::Trans::T`].
    ///
    /// # Examples
    ///
    /// ```
    /// use t4a_desc::ScalarKind;
    ///
    /// assert!(ScalarKind::C32.is_complex());
    /// assert!(!ScalarKind::F64.is_complex());
    /// ```
    pub const fn is_complex(self) -> bool {
        matches!(self, Self::C32 | Self::C64)
    }
}
