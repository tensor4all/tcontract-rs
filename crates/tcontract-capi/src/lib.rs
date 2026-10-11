//! `libtcontract_capi`: the C ABI of the tcontract stack, built as one shared library
//! (`cdylib`), one static library and this `rlib`.
//!
//! Two surfaces live here, and nothing else:
//!
//! * the standard **TAPP** contraction API (arXiv:2601.07827,
//!   <https://github.com/TAPPorg/reference-implementation>), ABI-compatible with
//!   the pinned upstream `include/tapp.h` and `include/tapp/*.h`: tensor infos
//!   ([`tensor_info`]), products ([`product`]) and their execution
//!   ([`execute`]), executors ([`executor`]), handles and attributes
//!   ([`handle`]);
//! * the **tcontract extensions** (`include/tcontract/*.h`): DLPack 1.x types
//!   ([`dlpack`]) and borrowed operands ([`tensor`]) converted to strided views
//!   without copying, status codes and the thread-local last-error message
//!   ([`status`]), the Rayon executor extension, and the ABI version.
//!
//! The ABI is unchanged from the pinned TAPP prototypes, numeric enumerator and
//! status values, handle ownership and semantics, except for one intentional
//! break: the BLAS C symbols (`tcontract_blas_*`) and `tcontract/blas.h` are gone, and
//! [`tcontract_has_part`] answers 0 for `"blas"`. Standard TAPP consumes raw
//! pointers and tensor-info metadata, not DLPack operands; DLPack borrow helpers
//! stay available but are not a contraction entry point.
//!
//! This crate contains no contraction algorithm and no second validator: a
//! product lowers its labels once into a [`tcontract::api::Problem`], and
//! every layout, role and alias question is the contract crate's. The C side
//! validates raw ABI arguments (null pointers, handles, enumerators) and maps
//! the contract error to a status in one table ([`abi`]). Standard calls use the
//! default `PlanConfig`; there is no C tuning API.
//!
//! Every `extern "C"` entry point that runs library code catches panics through
//! [`status::ffi`]; no unwind crosses the C boundary. Handles are `Box::into_raw`
//! pointers cast to `isize`: `0` is rejected, a stale or foreign non-zero value
//! is the caller's undefined behaviour, and handles are not interchangeable
//! between processes.
//!
//! # One contraction, end to end
//!
//! ```
//! use std::ffi::c_void;
//! use tcontract_capi::*;
//!
//! // D[i,j] = sum_k A[i,k] * B[k,j] over 2x2 column-major f64 tensors.
//! let extents = [2i64, 2];
//! let strides = [1i64, 2]; // column-major, in elements
//! let (i, j, k) = (b'i' as i64, b'j' as i64, b'k' as i64);
//!
//! unsafe {
//!     let mut handle = 0isize;
//!     assert!(TAPP_check_success(TAPP_create_handle(&mut handle)));
//!     let mut exec = 0isize;
//!     assert!(TAPP_check_success(TAPP_create_executor(&mut exec)));
//!     let mut info = 0isize;
//!     assert!(TAPP_check_success(TAPP_create_tensor_info(
//!         &mut info, TAPP_F64, 2, extents.as_ptr(), strides.as_ptr(),
//!     )));
//!     let (ia, ib, id) = ([i, k], [k, j], [i, j]);
//!     let mut plan = 0isize;
//!     assert!(TAPP_check_success(TAPP_create_tensor_product(
//!         &mut plan, handle,
//!         TAPP_IDENTITY, info, ia.as_ptr(),
//!         TAPP_IDENTITY, info, ib.as_ptr(),
//!         TAPP_IDENTITY, info, id.as_ptr(), // C is required; reuse D's info
//!         TAPP_IDENTITY, info, id.as_ptr(),
//!         TAPP_F64,
//!     )));
//!     let a = [1.0f64, 2.0, 3.0, 4.0];
//!     let b = [1.0f64, 0.0, 0.0, 1.0]; // the 2x2 identity
//!     let mut d = [0.0f64; 4];
//!     let (alpha, beta) = (1.0f64, 0.0f64);
//!     let mut status = 0isize;
//!     assert!(TAPP_check_success(TAPP_execute_product(
//!         plan, exec, &mut status,
//!         &alpha as *const f64 as *const c_void,
//!         a.as_ptr() as *const c_void,
//!         b.as_ptr() as *const c_void,
//!         &beta as *const f64 as *const c_void,
//!         std::ptr::null(), // TAPP_IN_PLACE: legal because beta is zero
//!         d.as_mut_ptr() as *mut c_void,
//!     )));
//!     assert_eq!(d, a);
//!     TAPP_destroy_status(status);
//!     TAPP_destroy_tensor_product(plan);
//!     TAPP_destroy_tensor_info(info);
//!     TAPP_destroy_executor(exec);
//!     TAPP_destroy_handle(handle);
//! }
//! ```
//!
//! # Coverage
//!
//! | TAPP feature | status |
//! |---|---|
//! | `TAPP_F32` / `TAPP_F64` / `TAPP_C32` / `TAPP_C64` | supported |
//! | `TAPP_F16` / `TAPP_BF16` | rejected (`TAPP_ERROR_DATATYPE`) |
//! | Cases 1-4: contraction, Hadamard / batch, repeated indices (diagonals), isolated input indices (reductions) | supported |
//! | Case 5 isolated *output* indices (broadcast) | rejected, as TAPP permits |
//! | `TAPP_CONJUGATE` on any operand | supported (folded into packing or write-back) |
//! | computational precision | `TAPP_DEFAULT_PREC` or the storage precision; anything else is `TAPP_ERROR_UNSUPPORTED` |
//! | executors | `0` and `TAPP_create_executor` are serial; `tcontract_tapp_executor_create_rayon` owns a Rayon pool, which `TAPP_destroy_executor` joins |
//! | `D` overlapping `A`/`B`, or itself | rejected (`TAPP_ERROR_ALIASED`); `C == D` only with equal element mapping |
//! | mixed storage types across operands | rejected (`TAPP_ERROR_DATATYPE`) |
//! | batched product | supported (items validated first, then run in sequence) |
//! | `TAPP_IN_PLACE` (a null `C`) | only with `beta == 0`; a non-zero `beta` is refused, even on an empty output |
//! | `TAPP_attr_*` | exported, and refuse every key: upstream specifies none |
//!
//! `TAPP_ERROR_*` beyond `TAPP_SUCCESS` are this library's own numbering (the
//! `tcontract_status` codes); upstream `error.h` fixes only zero, so a portable
//! caller goes through [`TAPP_check_success`].
#![warn(missing_docs)]

pub mod abi;
pub mod dlpack;
pub mod execute;
pub mod executor;
pub mod handle;
pub mod product;
pub mod status;
pub mod tensor;
pub mod tensor_info;

pub use abi::*;
pub use execute::{TAPP_execute_batched_product, TAPP_execute_product};
pub use executor::{
    tcontract_rayon_opts, tcontract_tapp_executor_create_rayon,
    tcontract_tapp_executor_get_threads, tcontract_tapp_executor_set_budget, TAPP_create_executor,
    TAPP_destroy_executor,
};
pub use handle::*;
pub use product::{TAPP_create_tensor_product, TAPP_destroy_tensor_product};
pub use status::{TAPP_check_success, TAPP_explain_error};
pub use tensor_info::*;

use std::ffi::{c_char, CStr};

/// ABI version: major * 10000 + minor * 100 + patch.
pub const ABI_VERSION: u32 = 100; // 0.1.0

/// The ABI version of the loaded library (`TCONTRACT_ABI_VERSION` in the header).
#[no_mangle]
pub extern "C" fn tcontract_abi_version() -> u32 {
    ABI_VERSION
}

/// Whether the loaded library contains `part` (`"core"`, `"tapp"`): 1 or 0.
/// Both are always present; `"blas"` (removed) answers 0.
///
/// # Safety
///
/// `part` is null or a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn tcontract_has_part(part: *const c_char) -> i32 {
    if part.is_null() {
        return 0;
    }
    // SAFETY: NUL-terminated per the contract.
    let name = unsafe { CStr::from_ptr(part) }.to_bytes();
    i32::from(matches!(name, b"core" | b"tapp"))
}
