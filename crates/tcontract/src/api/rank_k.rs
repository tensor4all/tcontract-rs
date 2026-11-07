//! The Hermitian/symmetric rank-k update (BLAS `herk`/`syrk`) as its own
//! problem kind (tensor4all/tcontract-rs#175).
//!
//! ```text
//! C ← alpha · op(A) · op(A)^H + beta · C    (hermitian)
//! C ← alpha · op(A) · op(A)^T + beta · C    (symmetric)
//! ```
//!
//! restricted to one triangle of the `n × n` matrix `C`. It is a separate kind,
//! not a flag on the general product: the general [`Problem`] has no notion of
//! a triangle, and a flag on it would still run the general product.
//! [`Plan::new_rank_k`](crate::Plan::new_rank_k) plans it and
//! [`Plan::execute_rank_k`](crate::Plan::execute_rank_k) runs it.
//!
//! The semantics follow reference BLAS `xHERK`/`xSYRK`.

use crate::api::error::{OperandId, Result, ShapeError};
use crate::api::labels::Labels;
use crate::api::problem::{CSpec, DType, LayoutSpec, Op, OperandSpec, Problem};

/// The triangle of `C` a rank-k update reads and writes, diagonal included.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Triangle {
    /// `C[i, j]` with `i >= j`.
    Lower,
    /// `C[i, j]` with `i <= j`.
    Upper,
}

impl Triangle {
    /// Whether `C[i, j]` lies in this triangle (the diagonal does in both).
    pub fn contains(self, i: usize, j: usize) -> bool {
        match self {
            Triangle::Lower => i >= j,
            Triangle::Upper => i <= j,
        }
    }
}

/// The operation applied to `A` before the product: BLAS `trans`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum RankKOp {
    /// `op(A) = A`; `A` is `n × k` and the update is `A A^H` (or `A A^T`).
    #[default]
    NoTrans,
    /// `op(A) = A^H` for the Hermitian form and `A^T` for the symmetric one
    /// (BLAS `'C'` / `'T'`); `A` is `k × n` and the update is `A^H A` (the Gram
    /// matrix) or `A^T A`. No conjugation by hand is needed.
    Trans,
}

/// The plan kind: which triangle, and which of the two products.
///
/// # Examples
///
/// ```
/// use tcontract::api::{RankK, Triangle};
/// let herk = RankK { triangle: Triangle::Lower, hermitian: true };
/// assert!(herk.triangle.contains(2, 1) && !herk.triangle.contains(1, 2));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RankK {
    /// The triangle read and written; the other one is not touched.
    pub triangle: Triangle,
    /// `true`: `op(A) op(A)^H` (`herk`), real `alpha`/`beta` and a real
    /// diagonal. `false`: `op(A) op(A)^T` (`syrk`), complex `alpha`/`beta`.
    /// For a real dtype the two coincide.
    pub hermitian: bool,
}

/// A validated rank-k update: the layouts of `A` and `C`, `op(A)` and the kind.
///
/// `C` is updated in place, as in BLAS: there is no separate output. Only the
/// requested triangle of `C` is read (when `beta != 0`) and written; see
/// [`Plan::execute_rank_k`](crate::Plan::execute_rank_k) for the full contract.
///
/// # Examples
///
/// ```
/// use tcontract::api::{DType, LayoutSpec, RankK, RankKOp, RankKProblem, Triangle};
/// // C (3 x 3, upper) <- A^H A for a 5 x 3 column-major A.
/// let p = RankKProblem::new(
///     DType::C64,
///     LayoutSpec::new(&[5, 3], &[1, 5], 0).unwrap(),
///     RankKOp::Trans,
///     LayoutSpec::new(&[3, 3], &[1, 3], 0).unwrap(),
///     RankK { triangle: Triangle::Upper, hermitian: true },
/// ).unwrap();
/// assert_eq!((p.n(), p.k()), (3, 5));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankKProblem {
    op: RankKOp,
    kind: RankK,
    n: usize,
    k: usize,
    /// The general product whose requested triangle this update computes:
    /// `D[i, j] = sum_l op_A(A)[i, l] op_B(A)[j, l]` with `B` laid out as `A`
    /// and `C` taken from `D`. It carries the spans, overflow checks and
    /// layouts; the triangle restriction is this type's alone.
    general: Problem,
}

impl RankKProblem {
    /// Validate a rank-k update.
    ///
    /// `a` is `n × k` for [`RankKOp::NoTrans`] and `k × n` for
    /// [`RankKOp::Trans`]; `c` is `n × n` and must address each of its elements
    /// once.
    ///
    /// # Errors
    ///
    /// [`ShapeError::NotMatrix`] when `a` or `c` is not a matrix (`c` is
    /// reported as [`OperandId::D`]), [`ShapeError::OutputExtents`] when `c` is
    /// not `n × n`, and the errors of [`Problem::from_labels`] for the lowered
    /// product: [`AliasError::OutputNotInjective`](crate::api::AliasError::OutputNotInjective)
    /// when `c` addresses an element twice, and the overflow errors.
    pub fn new(
        dtype: DType,
        a: LayoutSpec,
        op: RankKOp,
        c: LayoutSpec,
        kind: RankK,
    ) -> Result<Self> {
        let (ad, cd) = (a.dims(), c.dims());
        for (operand, dims) in [(OperandId::A, ad), (OperandId::D, cd)] {
            if dims.len() != 2 {
                return Err(ShapeError::NotMatrix {
                    operand,
                    rank: dims.len(),
                }
                .into());
            }
        }
        let (n, k) = match op {
            RankKOp::NoTrans => (ad[0], ad[1]),
            RankKOp::Trans => (ad[1], ad[0]),
        };
        if cd != [n, n] {
            return Err(ShapeError::OutputExtents {
                expected: vec![n, n],
                actual: cd.to_vec(),
            }
            .into());
        }
        // Labels: i = 0, j = 1, the contracted l = 2. The hermitian form
        // conjugates the factor that is transposed: `A conj(A)^T` for NoTrans,
        // `conj(A)^T A` for Trans.
        let conj = if kind.hermitian && dtype.is_complex() {
            Op::Conjugate
        } else {
            Op::Identity
        };
        let (la, lb, op_a, op_b): (&[i64], &[i64], Op, Op) = match op {
            RankKOp::NoTrans => (&[0, 2], &[1, 2], Op::Identity, conj),
            RankKOp::Trans => (&[2, 0], &[2, 1], conj, Op::Identity),
        };
        let general = Problem::from_labels(
            dtype,
            OperandSpec::new(a.clone()).with_op(op_a),
            OperandSpec::new(a).with_op(op_b),
            CSpec::Output(Op::Identity),
            OperandSpec::new(c),
            &Labels::new(la, lb, &[0, 1]),
        )?;
        Ok(Self {
            op,
            kind,
            n,
            k,
            general,
        })
    }

    /// The storage type.
    pub fn dtype(&self) -> DType {
        self.general.dtype()
    }

    /// The layout of `A`.
    pub fn a(&self) -> &LayoutSpec {
        self.general.a().layout()
    }

    /// The layout of `C`.
    pub fn c(&self) -> &LayoutSpec {
        self.general.d().layout()
    }

    /// `op(A)`.
    pub fn op(&self) -> RankKOp {
        self.op
    }

    /// The triangle and the product.
    pub fn kind(&self) -> RankK {
        self.kind
    }

    /// The order of `C`.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The contracted extent.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Multiply-accumulates of one execution over the requested triangle:
    /// `n (n + 1) / 2 · k`, half the general product's `n² k` plus the diagonal.
    pub fn macs(&self) -> u128 {
        // INVARIANT: `C` is injective and passed the lowering's span check, so
        // `n² < 2^63`; with `k < 2^64` the product stays below `2^127`.
        let n = self.n as u128;
        n * (n + 1) / 2 * self.k as u128
    }

    /// The general product whose requested triangle this update computes.
    pub(crate) fn general(&self) -> &Problem {
        &self.general
    }

    /// Whether the left (`i`) factor is read conjugated.
    pub(crate) fn conj_lhs(&self) -> bool {
        self.general.a().op().is_conj()
    }

    /// Whether the right (`j`) factor is read conjugated.
    pub(crate) fn conj_rhs(&self) -> bool {
        self.general.b().op().is_conj()
    }
}
