//! Test-only: the register-block menus of the built-in kernels, checked against
//! the reference at every entry point.
//!
//! The legacy `KernelSet` trait that used to hand these menus to the contraction
//! planner is gone: the planner resolves families from the registry. What
//! remains is the one thing the trait's test module was for -- every kernel
//! configuration the SIMD modules build must agree with the mathematical
//! definition, whatever packing and tile format it uses -- so the dispatch is
//! kept here as a private test trait over `f32` and `f64`.
//!
//! Source: lkdvos/tensorprimitives-rs, tensorcontract/src/kernel; MIT OR Apache-2.0.

use super::{reference::scalar, simd_isa};
use crate::cache;
use crate::*;

/// The per-real-type menu dispatch the tests drive: a vectorised module when the
/// target has one and the force selects it, the portable scalar kernels
/// otherwise.
trait KernelSet: Real + Sized {
    fn config_real(force: KernelForce) -> KernelConfig<Self>;
    fn config_cplx(force: KernelForce, method: ComplexMethod) -> KernelConfig<Self>;
    fn row_blocks(
        force: KernelForce,
        complex: bool,
        method: ComplexMethod,
    ) -> &'static [(usize, usize)] {
        let _ = (force, complex, method);
        &[]
    }
    fn config_at(
        force: KernelForce,
        complex: bool,
        method: ComplexMethod,
        i: usize,
    ) -> Option<KernelConfig<Self>> {
        let _ = (force, complex, method, i);
        None
    }
}

/// Wire one real type's [`KernelSet`] to the vectorised module when the target
/// has one, and to [`scalar`] otherwise.
///
/// The paths are `simd_isa::*`, so this is architecture-agnostic and the `#[cfg]`
/// asks only "is there a SIMD module at all". The dispatch functions answer
/// `None`/`&[]` when `force` selects no vectorised ISA (scalar pinned, or a
/// pin this target cannot honour), which sends the caller to the portable
/// scalar path. Off both families every arm below compiles out and the scalar
/// fallthrough is the whole body, which is why the `unreachable_code` and unused
/// allowances are needed on some targets and not others.
macro_rules! impl_kernel_set {
    ($t:ty, $real_simd:path, $cplx_simd:path, $rows_simd:path,
     $real_at_simd:path, $cplx_at_simd:path, $mr:literal, $nr:literal) => {
        impl KernelSet for $t {
            fn config_real(force: KernelForce) -> KernelConfig<Self> {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
                if let Some(c) = $real_simd(force) {
                    return c;
                }
                let _ = force;
                scalar::config_real::<$t, $mr, $nr>()
            }

            fn config_cplx(force: KernelForce, method: ComplexMethod) -> KernelConfig<Self> {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
                if let Some(c) = $cplx_simd(force, method) {
                    return c;
                }
                let _ = force;
                scalar::config_cplx::<$t, $mr, $nr>(method)
            }

            fn row_blocks(
                force: KernelForce,
                complex: bool,
                method: ComplexMethod,
            ) -> &'static [(usize, usize)] {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
                return $rows_simd(force, complex, method);
                // The portable path has one shape per method and no menu.
                #[allow(unreachable_code)]
                {
                    let _ = (force, complex, method);
                    &[]
                }
            }

            fn config_at(
                force: KernelForce,
                complex: bool,
                method: ComplexMethod,
                i: usize,
            ) -> Option<KernelConfig<Self>> {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
                return if complex {
                    $cplx_at_simd(force, method, i)
                } else {
                    $real_at_simd(force, i)
                };
                #[allow(unreachable_code)]
                {
                    let _ = (force, complex, method, i);
                    None
                }
            }
        }
    };
}

impl_kernel_set!(
    f64,
    simd_isa::config_real_f64,
    simd_isa::config_cplx_f64,
    simd_isa::row_blocks_f64,
    simd_isa::config_real_f64_at,
    simd_isa::config_cplx_f64_at,
    4,
    4
);
impl_kernel_set!(
    f32,
    simd_isa::config_real_f32,
    simd_isa::config_cplx_f32,
    simd_isa::row_blocks_f32,
    simd_isa::config_real_f32_at,
    simd_isa::config_cplx_f32_at,
    4,
    4
);

#[cfg(test)]
mod tests;
