//! Library handles, the (empty) attribute functions, status release and the
//! implementation identification: the small TAPP entry points that own no
//! contraction logic.

use std::ffi::c_void;
use std::os::raw::c_int;

use crate::abi::{fail, null};
use crate::status::{ffi, TCONTRACT_ERR_UNSUPPORTED};

/// Library handle. Stateless so far, but a *non*-zero-sized allocation on
/// purpose: with an empty struct `Box::into_raw` returns `NonNull::dangling()`,
/// so every handle the library ever issued was the same value, two live handles
/// were indistinguishable, and a C program that created two and destroyed both
/// was performing a double free — harmless only for as long as the state stayed
/// empty. The field costs eight bytes once per library handle.
///
/// Note what this does *not* buy, since an earlier comment here claimed it:
/// handle validity still cannot be checked. Nothing can distinguish a pointer
/// this crate produced from an arbitrary non-zero `intptr_t`, so `0` remains the
/// only value the ABI can reject.
struct HandleState {
    /// Reserved for state a future version needs; also what keeps the type
    /// non-zero-sized. Not read.
    _reserved: u64,
}

/// Create a library handle.
///
/// # Safety
/// `handle` must be a valid, writable `*mut isize` (or null, which is rejected).
/// On success it receives a handle to be released with
/// [`TAPP_destroy_handle`].
#[no_mangle]
pub unsafe extern "C" fn TAPP_create_handle(handle: *mut isize) -> c_int {
    ffi(|| {
        if handle.is_null() {
            return Err(null("handle out-parameter"));
        }
        // SAFETY: non-null and writable per the contract.
        unsafe { *handle = Box::into_raw(Box::new(HandleState { _reserved: 0 })) as isize };
        Ok(())
    })
}

/// Release a handle from [`TAPP_create_handle`].
///
/// # Safety
/// `handle` must be live and not already destroyed. See the crate docs on the
/// handle discipline.
#[no_mangle]
pub unsafe extern "C" fn TAPP_destroy_handle(handle: isize) -> c_int {
    ffi(|| {
        if handle == 0 {
            return Err(null("handle"));
        }
        // SAFETY: a live handle from `TAPP_create_handle`.
        drop(unsafe { Box::from_raw(handle as *mut HandleState) });
        Ok(())
    })
}

// ------------------------------------------------------------- attributes
//
// `api/include/tapp/attributes.h` declares three functions over an opaque
// `TAPP_attr` (`intptr_t`) and a `TAPP_key` (`int`), and specifies no keys. They
// are implemented here as refusals rather than left out, because a symbol that
// is merely absent turns a C program that includes `<tapp.h>` and calls one into
// a **link** failure — the least diagnosable kind — whereas a refusal is a value
// the caller can check and explain. Found by the conformance suite, which
// re-declares the whole header and so notices what is missing from it.
//
// Prototypes verified against the upstream header (TAPPorg/reference-implementation,
// api/include/tapp/attributes.h), not reconstructed.

/// Set an attribute. Always fails with [`TAPP_ERROR_UNSUPPORTED`](crate::abi::TAPP_ERROR_UNSUPPORTED): this
/// implementation defines no attribute keys, and upstream specifies none.
///
/// # Safety
/// Trivially safe — no argument is dereferenced — and `unsafe` only to match the
/// declared C signature.
#[no_mangle]
pub unsafe extern "C" fn TAPP_attr_set(_attr: isize, _key: c_int, _value: *mut c_void) -> c_int {
    ffi(|| {
        Err(fail(
            TCONTRACT_ERR_UNSUPPORTED,
            "no attribute keys are defined",
        ))
    })
}

/// Get an attribute. Always fails with [`TAPP_ERROR_UNSUPPORTED`](crate::abi::TAPP_ERROR_UNSUPPORTED); `value` is
/// set to null first if it is non-null, so a caller that ignores the return code
/// reads a defined value rather than whatever was on its stack.
///
/// # Safety
/// `value` must be null or valid for one pointer-sized write.
#[no_mangle]
pub unsafe extern "C" fn TAPP_attr_get(
    _attr: isize,
    _key: c_int,
    value: *mut *mut c_void,
) -> c_int {
    if !value.is_null() {
        // SAFETY: non-null and writable per the contract.
        unsafe { *value = std::ptr::null_mut() };
    }
    ffi(|| {
        Err(fail(
            TCONTRACT_ERR_UNSUPPORTED,
            "no attribute keys are defined",
        ))
    })
}

/// Clear an attribute. Always fails with [`TAPP_ERROR_UNSUPPORTED`](crate::abi::TAPP_ERROR_UNSUPPORTED).
///
/// # Safety
/// Trivially safe; `unsafe` only to match the declared C signature.
#[no_mangle]
pub unsafe extern "C" fn TAPP_attr_clear(_attr: isize, _key: c_int) -> c_int {
    ffi(|| {
        Err(fail(
            TCONTRACT_ERR_UNSUPPORTED,
            "no attribute keys are defined",
        ))
    })
}

/// Release a status object. A no-op: execution here is synchronous, so it never
/// produces a status to release. Provided because the header declares it and a
/// conforming caller will call it.
///
/// # Safety
/// Trivially safe; `unsafe` only to match the declared C signature.
#[no_mangle]
pub unsafe extern "C" fn TAPP_destroy_status(_status: isize) -> c_int {
    ffi(|| Ok(()))
}

/// Non-standard extension: a human-readable name for this backend. Useful when
/// several TAPP implementations are linked into one benchmark driver.
#[no_mangle]
pub extern "C" fn TAPP_implementation_name() -> *const std::os::raw::c_char {
    c"tcontract-rs: tcontract (packed BSMTC, faer, elementwise)".as_ptr()
}

/// The crate version of the *loaded library*, as a static NUL-terminated
/// `"major.minor.patch"` string. Never null; do not free the result.
///
/// Non-standard. The header a caller compiled against states
/// `TCONTRACT_ABI_VERSION`; this describes the library it actually linked. When a
/// distribution ships the two separately — a JLL, a system package, an
/// `LD_PRELOAD` — they can disagree, and without this there is no way to find
/// out. The [C consumer example] in the repository compares them and fails if
/// they differ.
///
/// [C consumer example]: https://github.com/tensor4all/tcontract-rs/tree/main/examples/c-consumer
#[no_mangle]
pub extern "C" fn TAPP_implementation_version() -> *const std::os::raw::c_char {
    // `c"..."` cannot interpolate, so the NUL is appended by hand.
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const std::os::raw::c_char
}
