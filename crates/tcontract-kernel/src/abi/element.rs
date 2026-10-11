//! Element types.
//!
//! The engine is generic over a *real* scalar type ([`Real`]) and over a
//! *storage* element type ([`Element`]) which is either that real type or an
//! interleaved complex pair built from it.
//!
//! Complex support is deliberately expressed as "two planes of `Real`". All
//! packing, micro-kernel and write-back code below the [`Element`] boundary
//! works exclusively in terms of `Element::Real`, which is what makes the
//! planar (split-complex) strategy fall out of the design rather than being
//! bolted on.

use core::fmt;
use core::ops::{Add, AddAssign, Mul, Neg, Sub};

use num_complex::Complex;

/// A real scalar usable as the arithmetic base of a contraction.
///
/// Implemented here for `f32`/`f64`. The generic kernel builders in
/// [`portable`](crate::portable) work for any `Real`, but the family registry and
/// the contraction planner are sealed to the four storage types: a downstream
/// scalar (extended precision, dual numbers for forward-mode AD, `bf16`, ...)
/// no longer gets a contraction engine by implementing a kernel-set trait.
///
/// The supertrait list is the whole arithmetic requirement, and it is
/// deliberately short: the engine never divides and never compares for
/// ordering, so neither `Div` nor `PartialOrd` is required. That admits types
/// — fixed-point, intervals, dual numbers for forward-mode AD — which have no
/// sensible division or no total order. Note in particular that no
/// floating-point *properties* are assumed. Nothing below requires
/// associativity, so a type with unusual rounding is admissible; what is
/// required is that [`Real::ZERO`] is an additive identity, because the
/// micro-kernels initialise accumulators with it.
pub trait Real:
    Copy
    + Send
    + Sync
    + fmt::Debug
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + 'static
{
    /// The additive identity. Micro-kernels initialise their accumulator tiles
    /// with it, so it must behave as one under `+`.
    const ZERO: Self;
    /// The multiplicative identity, used to recognise `alpha == 1` and
    /// `beta == 1` and skip the scaling entirely.
    const ONE: Self;

    /// Convert from `f64`. Used for scalars supplied by the caller and by the
    /// tests; never called per element in the hot path.
    fn from_f64(v: f64) -> Self;
    /// Convert to `f64`, for diagnostics and the test oracle's error metrics.
    /// Lossy for wider types, which is acceptable because nothing computational
    /// depends on it.
    fn to_f64(self) -> f64;
}

macro_rules! impl_real {
    ($t:ty) => {
        impl Real for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;

            #[inline(always)]
            fn from_f64(v: f64) -> Self {
                v as $t
            }
            #[inline(always)]
            fn to_f64(self) -> f64 {
                self as f64
            }
        }
    };
}

impl_real!(f32);
impl_real!(f64);

/// A tensor storage element: a real scalar, or an interleaved complex pair.
///
/// The memory layout of complex elements is *interleaved* (`re`, `im`
/// adjacent), matching `TAPP_C32`/`TAPP_C64`, C99 `_Complex`, `std::complex`
/// and `num_complex::Complex`. The engine converts to planar form during
/// packing and back to interleaved at write-back.
pub trait Element: Copy + Send + Sync + fmt::Debug + PartialEq + 'static {
    /// The underlying real type.
    type Real: Real;

    /// Whether this element type carries an imaginary part.
    ///
    /// This is what decides *which* trait method supplies the kernel — real
    /// element types never consult a [`crate::ComplexMethod`] at all.
    const IS_COMPLEX: bool;
    /// Real flops per multiply-accumulate: 2 for real, 8 for complex.
    ///
    /// Reported so a harness computes throughput the same way the engine
    /// counts work. Note that this is the *textbook* count and is deliberately
    /// not reduced for 3m, whose 25% saving is a property of the method and not
    /// of the problem — charging 3m fewer flops would flatter it.
    const FLOPS_PER_MAC: u64;

    /// The additive identity.
    fn zero() -> Self;
    /// The multiplicative identity.
    fn one() -> Self;
    /// The real part.
    fn re(self) -> Self::Real;
    /// The imaginary part, or [`Real::ZERO`] for a real type.
    fn im(self) -> Self::Real;
    /// Rebuild an element from its parts. A real type discards `im`, which is
    /// why the engine can use one write-back for both domains.
    fn from_parts(re: Self::Real, im: Self::Real) -> Self;

    /// Complex conjugate; the identity for a real type.
    fn conj(self) -> Self;
    /// Element-wise product. Inherent rather than a `Mul` bound so that the
    /// trait imposes no operator impls on a foreign element type.
    fn mul(self, rhs: Self) -> Self;
    /// Element-wise sum.
    fn add(self, rhs: Self) -> Self;
    /// Element-wise difference.
    fn sub(self, rhs: Self) -> Self;

    /// Magnitude, used by the test oracle for error metrics.
    fn norm(self) -> f64 {
        let (r, i) = (self.re().to_f64(), self.im().to_f64());
        (r * r + i * i).sqrt()
    }
}

macro_rules! impl_element_real {
    ($t:ty) => {
        impl Element for $t {
            type Real = $t;
            const IS_COMPLEX: bool = false;
            const FLOPS_PER_MAC: u64 = 2;

            #[inline(always)]
            fn zero() -> Self {
                0.0
            }
            #[inline(always)]
            fn one() -> Self {
                1.0
            }
            #[inline(always)]
            fn re(self) -> Self::Real {
                self
            }
            #[inline(always)]
            fn im(self) -> Self::Real {
                0.0
            }
            #[inline(always)]
            fn from_parts(re: Self::Real, _im: Self::Real) -> Self {
                re
            }
            #[inline(always)]
            fn conj(self) -> Self {
                self
            }
            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                self * rhs
            }
            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                self + rhs
            }
            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                self - rhs
            }
        }
    };
}

impl_element_real!(f32);
impl_element_real!(f64);

macro_rules! impl_element_complex {
    ($t:ty) => {
        impl Element for Complex<$t> {
            type Real = $t;
            const IS_COMPLEX: bool = true;
            const FLOPS_PER_MAC: u64 = 8;

            #[inline(always)]
            fn zero() -> Self {
                Complex::new(0.0, 0.0)
            }
            #[inline(always)]
            fn one() -> Self {
                Complex::new(1.0, 0.0)
            }
            #[inline(always)]
            fn re(self) -> Self::Real {
                self.re
            }
            #[inline(always)]
            fn im(self) -> Self::Real {
                self.im
            }
            #[inline(always)]
            fn from_parts(re: Self::Real, im: Self::Real) -> Self {
                Complex::new(re, im)
            }
            #[inline(always)]
            fn conj(self) -> Self {
                Complex::new(self.re, -self.im)
            }
            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                self * rhs
            }
            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                self + rhs
            }
            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                self - rhs
            }
        }
    };
}

impl_element_complex!(f32);
impl_element_complex!(f64);

/// Single-precision complex, spelled as TAPP's `TAPP_C32` names it.
pub type C32 = Complex<f32>;
/// Double-precision complex, spelled as TAPP's `TAPP_C64` names it.
pub type C64 = Complex<f64>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_layout_is_interleaved() {
        // The FFI/TAPP boundary and the packing code both assume this.
        assert_eq!(core::mem::size_of::<C64>(), 16);
        assert_eq!(core::mem::align_of::<C64>(), core::mem::align_of::<f64>());
        // `offset_of!` rather than a transmute to `[f64; 2]`: it asserts the same
        // thing without unsafe, and it asserts it more precisely — the field
        // *order* is named rather than inferred from which value came out first.
        assert_eq!(core::mem::offset_of!(C64, re), 0);
        assert_eq!(core::mem::offset_of!(C64, im), core::mem::size_of::<f64>());
    }

    #[test]
    fn flops_per_mac() {
        assert_eq!(<f64 as Element>::FLOPS_PER_MAC, 2);
        assert_eq!(<C64 as Element>::FLOPS_PER_MAC, 8);
    }
}
