use strided_view::{StridedView, StridedViewMut};
use tcontract_exec::Exec;
use tcontract_kernel::Element;

use crate::api::{is_injective_layout, AliasError, ConfigError, Error, Result, Scalar, ShapeError};
use crate::strategy::elementwise::{CRead, ElementPlan, Expr, Inputs};

fn check_out<T>(c: &StridedViewMut<'_, T>) -> Result<()> {
    if is_injective_layout(c.dims(), c.strides()) {
        Ok(())
    } else {
        Err(AliasError::OutputNotInjective.into())
    }
}

/// `C[perm[0], perm[1], ...] = A`: copy with an axis permutation, so that
/// axis `k` of C is axis `perm[k]` of A.
///
/// # Errors
///
/// [`ConfigError::NotAPermutation`] for an invalid permutation,
/// [`ShapeError::OutputExtents`] for C's extents, [`AliasError::OutputNotInjective`].
///
/// # Examples
///
/// ```
/// use strided_view::{StridedView, StridedViewMut};
/// let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 x 3 column-major
/// let mut c = [0.0; 6];
/// let av = StridedView::new(&a, &[2, 3], &[1, 2], 0).unwrap();
/// let mut cv = StridedViewMut::new(&mut c, &[3, 2], &[1, 3], 0).unwrap();
/// tcontract::permute(&mut tcontract_exec::Exec::serial(), &av, &[1, 0], &mut cv).unwrap();
/// assert_eq!(c, [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
/// ```
pub fn permute<T: Scalar>(
    exec: &mut Exec<'_>,
    a: &StridedView<'_, T>,
    perm: &[usize],
    c: &mut StridedViewMut<'_, T>,
) -> Result<()> {
    let r = a.ndim();
    let mut seen = vec![false; r];
    if perm.len() != r
        || perm
            .iter()
            .any(|&p| p >= r || std::mem::replace(&mut seen[p], true))
    {
        return Err(ConfigError::NotAPermutation { rank: r }.into());
    }
    let want: Vec<usize> = perm.iter().map(|&p| a.dims()[p]).collect();
    if c.dims() != want.as_slice() {
        return Err(ShapeError::OutputExtents {
            expected: want,
            actual: c.dims().to_vec(),
        }
        .into());
    }
    check_out(c)?;
    let src = a.permute(perm).map_err(Error::backend)?;
    strided_basic::copy_into(c, &src, exec.token()).map_err(Error::backend)
}

/// `C = alpha * A + beta * C` elementwise over equal extents (any strides);
/// `beta == 0` never reads C.
///
/// # Errors
///
/// [`ShapeError::OutputExtents`], [`AliasError::OutputNotInjective`].
///
/// # Examples
///
/// ```
/// use strided_view::{StridedView, StridedViewMut};
/// let a = [1.0, 2.0];
/// let mut c = [10.0, 20.0];
/// let av = StridedView::new(&a, &[2], &[1], 0).unwrap();
/// let mut cv = StridedViewMut::new(&mut c, &[2], &[1], 0).unwrap();
/// tcontract::add(&mut tcontract_exec::Exec::serial(), 2.0, &av, 0.5, &mut cv).unwrap();
/// assert_eq!(c, [7.0, 14.0]);
/// ```
pub fn add<T: Scalar>(
    exec: &mut Exec<'_>,
    alpha: T,
    a: &StridedView<'_, T>,
    beta: T,
    c: &mut StridedViewMut<'_, T>,
) -> Result<()> {
    if a.dims() != c.dims() {
        return Err(ShapeError::OutputExtents {
            expected: a.dims().to_vec(),
            actual: c.dims().to_vec(),
        }
        .into());
    }
    check_out(c)?;
    if beta == <T as Element>::zero() {
        return strided_basic::map_into(c, a, move |x| Element::mul(alpha, x), exec.token())
            .map_err(Error::backend);
    }
    if c.is_empty() {
        return Ok(());
    }
    // SAFETY: both views are non-empty and bounds-checked with these extents;
    // C is exclusive and injective (checked above); A is a distinct borrow.
    unsafe {
        ElementPlan::scaled(c.dims(), a.strides(), c.strides()).run(
            exec,
            Expr {
                alpha,
                beta,
                conj_a: false,
                conj_b: false,
                conj_c: false,
                conj_d: false,
            },
            Inputs::Scaled(a.ptr()),
            CRead::InPlace,
            c.as_mut_ptr(),
        )
    }
}
