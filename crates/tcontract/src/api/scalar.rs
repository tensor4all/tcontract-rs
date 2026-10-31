//! The sealed storage-scalar vocabulary.

use core::mem::MaybeUninit;
use faer::linalg::matmul::{matmul_with_conj, matmul_with_conj_uninit};
use faer::{Accum, Conj, MatMut, MatRef, MatUninitMut};
use tcontract_kernel::{Element, Families};

use crate::api::DType;

pub(crate) mod sealed {
    use crate::strategy::elementwise::{CRead, ElementPlan, Expr, Inputs};
    use tcontract_exec::Exec;

    /// The sealed marker, plus the per-dtype hooks that keep the elementwise
    /// strategy instantiated once, inside this crate, instead of once per
    /// consumer of `Plan<T>` (share-generics is off in release builds, and
    /// that pass monomorphizes dozens of strided-basic kernels).
    pub trait Sealed: Sized + Copy {
        /// The type-level conjugation: `Conj` for a complex type, `Identity`
        /// for a real one (so a real type has a single operand-op instance).
        type Cj: strided_basic::ElementOp<Self>;

        /// Run an elementwise update; one non-generic instance per dtype.
        ///
        /// # Safety
        ///
        /// As [`ElementPlan::run`].
        unsafe fn elementwise(
            plan: &ElementPlan,
            exec: &mut Exec<'_>,
            e: Expr<Self>,
            inputs: Inputs<Self>,
            c: CRead<Self>,
            d: *mut Self,
        ) -> crate::api::Result<()>;
    }

    macro_rules! impl_sealed {
        ($($t:ty => $cj:ty),*) => {$(
            impl Sealed for $t {
                type Cj = $cj;
                #[inline(never)]
                unsafe fn elementwise(
                    plan: &ElementPlan,
                    exec: &mut Exec<'_>,
                    e: Expr<Self>,
                    inputs: Inputs<Self>,
                    c: CRead<Self>,
                    d: *mut Self,
                ) -> crate::api::Result<()> {
                    // SAFETY: the caller's contract.
                    unsafe { plan.run_impl::<$t>(exec, e, inputs, c, d) }
                }
            }
        )*};
    }
    impl_sealed!(
        f32 => strided_basic::Identity,
        f64 => strided_basic::Identity,
        num_complex::Complex<f32> => strided_basic::Conj,
        num_complex::Complex<f64> => strided_basic::Conj
    );
}

/// The storage and accumulation type of a contraction: `f32`, `f64`,
/// `Complex<f32>` or `Complex<f64>` (identical to `faer::c32` / `faer::c64`).
///
/// Sealed. The supertraits (the kernel registry's [`Families`] and a hidden
/// faer adapter) are implementation vocabulary of this crate; a caller or a
/// second backend only names `T: Scalar`, and the neutral backend trait does
/// not require a kernel family or faer bound of its own.
///
/// # Examples
///
/// ```
/// use tcontract::api::{DType, Scalar};
/// fn dtype_of<T: Scalar>() -> DType { T::STORAGE }
/// assert_eq!(dtype_of::<f64>(), DType::F64);
/// assert_eq!(dtype_of::<num_complex::Complex<f32>>(), DType::C32);
/// ```
pub trait Scalar:
    sealed::Sealed
    + FaerGemm
    + Element<Real = <Self as Scalar>::Re>
    + Families
    + Copy
    + PartialEq
    + strided_basic::ElementOpApply
    + core::ops::Add<Output = Self>
    + core::ops::Mul<Output = Self>
    + num_traits::One
    + Send
    + Sync
    + 'static
{
    /// The real type of the kernels' arithmetic.
    type Re: tcontract_kernel::Real;
    /// The problem-level name of the type.
    const STORAGE: DType;
}

impl Scalar for f32 {
    type Re = f32;
    const STORAGE: DType = DType::F32;
}
impl Scalar for f64 {
    type Re = f64;
    const STORAGE: DType = DType::F64;
}
impl Scalar for num_complex::Complex<f32> {
    type Re = f32;
    const STORAGE: DType = DType::C32;
}
impl Scalar for num_complex::Complex<f64> {
    type Re = f64;
    const STORAGE: DType = DType::C64;
}

/// The one faer call this strategy makes, per storage type.
///
/// A hidden supertrait of [`Scalar`](crate::api::Scalar): it keeps faer's
/// element traits out of the public bounds while letting the strategy stay
/// generic. Implemented for exactly the four storage types.
#[doc(hidden)]
pub trait FaerGemm: Sized {
    /// `D = alpha * op(A) * op(B)` (`accumulate` adds into `D`, otherwise `D`
    /// is overwritten without being read) on strided matrices.
    ///
    /// # Safety
    ///
    /// The pointers with these strides address valid elements of the stated
    /// extents; `d` is exclusive and does not alias `a` or `b`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn matmul(
        accumulate: bool,
        mnk: (usize, usize, usize),
        d: *mut Self,
        d_strides: (isize, isize),
        a: *const Self,
        a_strides: (isize, isize),
        conj_a: bool,
        b: *const Self,
        b_strides: (isize, isize),
        conj_b: bool,
        alpha: Self,
        par: faer::Par<'_>,
    );

    /// `D = alpha * op(A) * op(B)` where `D`'s elements may be uninitialized:
    /// nothing of `D` is read and every element is written before the call
    /// returns.
    ///
    /// # Safety
    ///
    /// As [`FaerGemm::matmul`] with `accumulate` false, except that `d`'s
    /// elements need not be initialized.
    #[allow(clippy::too_many_arguments)] // INVARIANT: the GEMM argument set.
    unsafe fn matmul_uninit(
        mnk: (usize, usize, usize),
        d: *mut Self,
        d_strides: (isize, isize),
        a: *const Self,
        a_strides: (isize, isize),
        conj_a: bool,
        b: *const Self,
        b_strides: (isize, isize),
        conj_b: bool,
        alpha: Self,
        par: faer::Par<'_>,
    );
}

macro_rules! faer_gemm {
    ($($t:ty),*) => {$(
        impl FaerGemm for $t {
            unsafe fn matmul(
                accumulate: bool,
                (m, n, k): (usize, usize, usize),
                d: *mut Self,
                (drs, dcs): (isize, isize),
                a: *const Self,
                (ars, acs): (isize, isize),
                conj_a: bool,
                b: *const Self,
                (brs, bcs): (isize, isize),
                conj_b: bool,
                alpha: Self,
                par: faer::Par<'_>,
            ) {
                // SAFETY: the caller's contract.
                let (am, bm, dm) = unsafe {
                    (
                        MatRef::<$t>::from_raw_parts(a, m, k, ars, acs),
                        MatRef::<$t>::from_raw_parts(b, k, n, brs, bcs),
                        MatMut::<$t>::from_raw_parts_mut(d, m, n, drs, dcs),
                    )
                };
                let conj = |c: bool| if c { Conj::Yes } else { Conj::No };
                let accum = if accumulate { Accum::Add } else { Accum::Replace };
                matmul_with_conj(dm, accum, am, conj(conj_a), bm, conj(conj_b), alpha, par);
            }

            unsafe fn matmul_uninit(
                (m, n, k): (usize, usize, usize),
                d: *mut Self,
                (drs, dcs): (isize, isize),
                a: *const Self,
                (ars, acs): (isize, isize),
                conj_a: bool,
                b: *const Self,
                (brs, bcs): (isize, isize),
                conj_b: bool,
                alpha: Self,
                par: faer::Par<'_>,
            ) {
                // SAFETY: the caller's contract; only the destination's element
                // type changes, to `MaybeUninit`.
                let (am, bm, dm) = unsafe {
                    (
                        MatRef::<$t>::from_raw_parts(a, m, k, ars, acs),
                        MatRef::<$t>::from_raw_parts(b, k, n, brs, bcs),
                        MatUninitMut::<$t>::from_raw_parts_mut(
                            d as *mut MaybeUninit<$t>,
                            m,
                            n,
                            drs,
                            dcs,
                        ),
                    )
                };
                let conj = |c: bool| if c { Conj::Yes } else { Conj::No };
                matmul_with_conj_uninit(dm, am, conj(conj_a), bm, conj(conj_b), alpha, par);
            }
        }
    )*};
}
faer_gemm!(
    f32,
    f64,
    num_complex::Complex<f32>,
    num_complex::Complex<f64>
);
