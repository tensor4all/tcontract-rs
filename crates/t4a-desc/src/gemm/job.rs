//! The raw descriptor: one job, the call's ops, and the grouping key.

use core::fmt;

/// The operation applied to an input operand, as BLAS's `trans` argument.
///
/// The set is BLAS's and closed on purpose: cuBLAS and every CPU BLAS accept
/// exactly these three for GEMM. Conjugation without transposition is not
/// among them (see the [module documentation](super#mapping-to-cublas-and-cutlass)).
///
/// # Examples
///
/// ```
/// use t4a_desc::gemm::{GemmJob, GemmOps, Trans};
///
/// // op(A) = Aᴴ: the stored A is k×m, so its smallest leading dimension is k.
/// let ops = GemmOps { a: Trans::C, b: Trans::N };
/// assert_eq!(GemmJob::compact(ops, [0; 3], 2, 3, 5).lda, 5);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Trans {
    /// `op(X) = X` (`CUBLAS_OP_N`).
    N,
    /// `op(X) = Xᵀ` (`CUBLAS_OP_T`).
    T,
    /// `op(X) = Xᴴ`, the conjugate transpose (`CUBLAS_OP_C`). For a real
    /// [`ScalarKind`](crate::ScalarKind) it means the same as [`Trans::T`];
    /// the batch keeps the value as given.
    C,
}

/// The ops of one call, shared by every job of a [`GemmBatch`](super::GemmBatch).
///
/// cuBLAS takes them per group; one value for all groups is the subset this
/// crate supports (see the [module documentation](super#per-call-values)).
///
/// # Examples
///
/// ```
/// use t4a_desc::gemm::{GemmOps, Trans};
///
/// let ops = GemmOps { a: Trans::T, b: Trans::N };
/// assert_eq!(ops.a, Trans::T);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GemmOps {
    /// The operation on `A`.
    pub a: Trans,
    /// The operation on `B`.
    pub b: Trans,
}

/// One problem of a grouped GEMM, column-major with BLAS meaning.
///
/// `op(A)` is `m×k`, `op(B)` is `k×n` and `C` is `m×n`;
/// `C ← alpha·op(A)·op(B) + beta·C` in place. Offsets are measured in
/// elements from the start of one base buffer per operand.
///
/// The stored matrices are `m×k` (`A`, op N) or `k×m` (`A`, op T/C), `k×n` or
/// `n×k` for `B`, and `m×n` for `C`, each with its leading dimension as the
/// column stride. BLAS's rules apply: the leading dimension is at least
/// `max(1, stored rows)`. A raw job carries no invariant; [`GemmBatch::new`]
/// checks them.
///
/// [`GemmBatch::new`]: super::GemmBatch::new
///
/// # Examples
///
/// A 2×3 block of a 4×3 output, read from a 2×2 `A` and a 2×3 `B`:
///
/// ```
/// use t4a_desc::gemm::GemmJob;
///
/// let job = GemmJob {
///     a_offset: 0, b_offset: 0, c_offset: 2,
///     m: 2, n: 3, k: 2,
///     lda: 2, ldb: 2, ldc: 4,
/// };
/// assert_eq!(job.shape().ldc, 4);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GemmJob {
    /// Element offset of the stored `A` in the `A` buffer.
    pub a_offset: usize,
    /// Element offset of the stored `B` in the `B` buffer.
    pub b_offset: usize,
    /// Element offset of `C` in the `C` buffer.
    pub c_offset: usize,
    /// Rows of `op(A)` and of `C`.
    pub m: usize,
    /// Columns of `op(B)` and of `C`.
    pub n: usize,
    /// Columns of `op(A)`, rows of `op(B)`: the contracted extent.
    pub k: usize,
    /// Leading dimension (column stride) of the stored `A`.
    pub lda: usize,
    /// Leading dimension (column stride) of the stored `B`.
    pub ldb: usize,
    /// Leading dimension (column stride) of `C`.
    pub ldc: usize,
}

impl GemmJob {
    /// The compact form: every leading dimension is the smallest BLAS allows
    /// for `ops`.
    ///
    /// `lda = max(1, m)` for op N and `max(1, k)` for T/C;
    /// `ldb = max(1, k)` for op N and `max(1, n)` for T/C; `ldc = max(1, m)`.
    /// `offsets` is `[a_offset, b_offset, c_offset]`. The dimension order is
    /// BLAS's, `m, n, k`.
    ///
    /// # Examples
    ///
    /// ```
    /// use t4a_desc::gemm::{GemmJob, GemmOps, Trans};
    ///
    /// let ops = GemmOps { a: Trans::N, b: Trans::T };
    /// let job = GemmJob::compact(ops, [0, 6, 12], 2, 3, 4);
    /// assert_eq!((job.lda, job.ldb, job.ldc), (2, 3, 2));
    /// // An empty extent still gives a leading dimension of 1.
    /// assert_eq!(GemmJob::compact(ops, [0; 3], 0, 3, 4).ldc, 1);
    /// ```
    pub const fn compact(ops: GemmOps, offsets: [usize; 3], m: usize, n: usize, k: usize) -> Self {
        let lda = at_least_one(match ops.a {
            Trans::N => m,
            Trans::T | Trans::C => k,
        });
        let ldb = at_least_one(match ops.b {
            Trans::N => k,
            Trans::T | Trans::C => n,
        });
        Self {
            a_offset: offsets[0],
            b_offset: offsets[1],
            c_offset: offsets[2],
            m,
            n,
            k,
            lda,
            ldb,
            ldc: at_least_one(m),
        }
    }

    /// The job's geometry without its offsets: its group key.
    ///
    /// # Examples
    ///
    /// ```
    /// use t4a_desc::gemm::{GemmJob, GemmOps, Trans};
    ///
    /// let ops = GemmOps { a: Trans::N, b: Trans::N };
    /// let (x, y) = (GemmJob::compact(ops, [0; 3], 2, 2, 2), GemmJob::compact(ops, [4; 3], 2, 2, 2));
    /// assert_eq!(x.shape(), y.shape());
    /// ```
    pub const fn shape(&self) -> GemmShape {
        GemmShape {
            m: self.m,
            n: self.n,
            k: self.k,
            lda: self.lda,
            ldb: self.ldb,
            ldc: self.ldc,
        }
    }

    /// The stored rows, columns and leading dimension of `operand` under `ops`.
    pub(crate) const fn stored(&self, ops: GemmOps, operand: Operand) -> (usize, usize, usize) {
        match operand {
            Operand::A => match ops.a {
                Trans::N => (self.m, self.k, self.lda),
                Trans::T | Trans::C => (self.k, self.m, self.lda),
            },
            Operand::B => match ops.b {
                Trans::N => (self.k, self.n, self.ldb),
                Trans::T | Trans::C => (self.n, self.k, self.ldb),
            },
            Operand::C => (self.m, self.n, self.ldc),
        }
    }

    /// The element offset of `operand`.
    pub(crate) const fn offset(&self, operand: Operand) -> usize {
        match operand {
            Operand::A => self.a_offset,
            Operand::B => self.b_offset,
            Operand::C => self.c_offset,
        }
    }
}

/// The geometry of a job: its `m, n, k` and leading dimensions.
///
/// Jobs with equal shapes form one group: cuBLAS's group, and the jobs that
/// share one child plan in cpueinsum (tensor4all/cpueinsum-rs#51). The order
/// is lexicographic in field order.
///
/// # Examples
///
/// ```
/// use t4a_desc::gemm::GemmShape;
///
/// let s = GemmShape { m: 2, n: 3, k: 4, lda: 2, ldb: 4, ldc: 2 };
/// assert!(s < GemmShape { m: 3, ..s });
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GemmShape {
    /// Rows of `op(A)` and of `C`.
    pub m: usize,
    /// Columns of `op(B)` and of `C`.
    pub n: usize,
    /// The contracted extent.
    pub k: usize,
    /// Leading dimension of the stored `A`.
    pub lda: usize,
    /// Leading dimension of the stored `B`.
    pub ldb: usize,
    /// Leading dimension of `C`.
    pub ldc: usize,
}

/// One of the three operand buffers of a grouped GEMM.
///
/// # Examples
///
/// ```
/// use t4a_desc::gemm::Operand;
///
/// assert_eq!(Operand::C.to_string(), "C");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operand {
    /// The `A` buffer.
    A,
    /// The `B` buffer.
    B,
    /// The `C` buffer, read and written in place.
    C,
}

impl Operand {
    pub(crate) const ALL: [Self; 3] = [Self::A, Self::B, Self::C];

    pub(crate) const fn index(self) -> usize {
        match self {
            Self::A => 0,
            Self::B => 1,
            Self::C => 2,
        }
    }
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::A => "A",
            Self::B => "B",
            Self::C => "C",
        })
    }
}

const fn at_least_one(x: usize) -> usize {
    if x == 0 {
        1
    } else {
        x
    }
}

/// Elements a stored `rows×cols` matrix with column stride `ld` spans:
/// `(cols - 1)·ld + rows`, or 0 when it has no element. `None` on overflow.
pub(crate) const fn span(rows: usize, cols: usize, ld: usize) -> Option<usize> {
    if rows == 0 || cols == 0 {
        return Some(0);
    }
    match (cols - 1).checked_mul(ld) {
        Some(v) => v.checked_add(rows),
        None => None,
    }
}
