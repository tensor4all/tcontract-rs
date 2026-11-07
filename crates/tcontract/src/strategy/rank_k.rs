//! The reference rank-k kernel (BLAS `herk`/`syrk`, tensor4all/tcontract-rs#175).
//!
//! No packing, no blocking, no workspace and no threads: for each element of
//! the requested triangle it reads the two rows of `op(A)` it needs, sums
//! their products in ascending `l`, applies `alpha` and `beta`, and writes the
//! element. Each `A` element is therefore read once per triangle element that
//! needs it, and the work is the half product, `n (n + 1) / 2 · k`
//! multiply-adds. It is the semantic reference the register-tile families of
//! a later step are tested against, not a fast path.
//!
//! The loop follows the reference BLAS `xHERK`/`xSYRK` semantics (Dongarra,
//! Du Croz, Hammarling and Duff, *A set of level 3 basic linear algebra
//! subprograms*, ACM TOMS 16(1), 1990): `beta == 0` reads no `C`, the other
//! triangle is neither read nor written, and the Hermitian diagonal is
//! written real. It is written from that specification, not ported from an
//! implementation. Unlike reference BLAS, it does not repair a Hermitian
//! input diagonal with an imaginary part (BLAS silently drops it); the plan
//! rejects such a `C` before this kernel runs.
//!
//! Every access is bounds-checked safe indexing — `A` through its backing
//! slice, `C` through the view accessors — with indices inside the validated
//! extents by construction, so no check can fail and the kernel needs no
//! `unsafe`.

use strided_view::{StridedView, StridedViewMut};
use tcontract_kernel::{Element, Real};

use crate::api::{RankKOp, RankKProblem, Scalar, Triangle, ValueError};

/// The prepared reference rank-k update.
#[derive(Debug)]
pub(crate) struct RankKPlan {
    pub(crate) problem: RankKProblem,
}

impl RankKPlan {
    /// The values the contract excludes, checked before any write: a complex
    /// `alpha` or `beta` for the Hermitian form, and — when `beta != 0` makes
    /// the update read `C` — a diagonal element of `C` with an imaginary part.
    ///
    /// A real dtype never fails these checks: its imaginary parts are zero.
    pub(crate) fn validate<T: Scalar>(
        &self,
        alpha: T,
        beta: T,
        c: &StridedViewMut<'_, T>,
    ) -> Result<(), ValueError> {
        if !self.problem.kind().hermitian {
            return Ok(());
        }
        let zero = <T::Re as Real>::ZERO;
        // `!=` also rejects a NaN imaginary part.
        if alpha.im() != zero {
            return Err(ValueError::ComplexScalar { which: "alpha" });
        }
        if beta.im() != zero {
            return Err(ValueError::ComplexScalar { which: "beta" });
        }
        if beta != <T as Element>::zero() {
            for i in 0..self.problem.n() {
                if c.get(&[i, i]).im() != zero {
                    return Err(ValueError::HermitianDiagonal { index: i });
                }
            }
        }
        Ok(())
    }

    /// `C[tri] = alpha · op(A) op(A)^{H|T} [tri] + beta · C[tri]`, serially on
    /// the calling thread. The views must have the planned layouts and
    /// [`RankKPlan::validate`] must have accepted the scalars and `C`.
    pub(crate) fn run<T: Scalar>(
        &self,
        alpha: T,
        a: &StridedView<'_, T>,
        beta: T,
        c: &mut StridedViewMut<'_, T>,
    ) {
        let p = &self.problem;
        let (n, k) = (p.n(), p.k());
        let (triangle, hermitian) = (p.kind().triangle, p.kind().hermitian);
        let (conj_lhs, conj_rhs) = (p.conj_lhs(), p.conj_rhs());
        let zero = <T as Element>::zero();
        // `op(A)` as an `n × k` strided matrix over `a`'s storage, resolved
        // once per call: `(row stride, column stride)` of `op(A)[i, l]` before
        // the factor's conjugation.
        let (sa0, sa1) = (a.strides()[0], a.strides()[1]);
        let (si, sl) = match p.op() {
            RankKOp::NoTrans => (sa0, sa1),
            RankKOp::Trans => (sa1, sa0),
        };
        let (data, origin) = (a.data(), a.offset());
        // INVARIANT: `i < n` and `l < k` lie inside the view's extents, which
        // `StridedView::new` checked against `data`, so the index is in range;
        // the per-element offset and the bounds check are the reference
        // kernel's cost, not a fast path's.
        let at = |i: usize, l: usize| data[(origin + i as isize * si + l as isize * sl) as usize];
        let maybe_conj = |x: T, conj: bool| if conj { Element::conj(x) } else { x };
        for j in 0..n {
            let rows = match triangle {
                Triangle::Lower => j..n,
                Triangle::Upper => 0..j + 1,
            };
            for i in rows {
                let mut acc = zero;
                // `alpha == 0` reads no `A`, as BLAS does.
                if alpha != zero {
                    for l in 0..k {
                        acc = acc + maybe_conj(at(i, l), conj_lhs) * maybe_conj(at(j, l), conj_rhs);
                    }
                }
                let mut v = alpha * acc;
                // `beta == 0` reads no `C`: it may be uninitialised-looking.
                if beta != zero {
                    v = v + beta * c.get(&[i, j]);
                }
                if hermitian && i == j {
                    // Real by construction; drop the rounding residue a complex
                    // multiply could leave rather than store it.
                    v = T::from_parts(v.re(), <T::Re as Real>::ZERO);
                }
                c.set(&[i, j], v);
            }
        }
    }
}
