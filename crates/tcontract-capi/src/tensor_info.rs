//! Tensor infos: the handles, constructors, getters and setters of the TAPP
//! tensor description. A product snapshots them; nothing here validates a
//! contraction (the problem lowering owns that).

use std::os::raw::c_int;

use crate::abi::*;
use crate::status::{ffi, TCONTRACT_ERR_DTYPE, TCONTRACT_ERR_SHAPE};

/// One tensor as the caller described it: extents and strides in elements,
/// kept as `i64` because the setters can change them after creation. Everything
/// is validated, once, when a product is created.
#[derive(Clone, Debug, Default)]
pub(crate) struct TapLayout {
    pub(crate) extents: Vec<i64>,
    pub(crate) strides: Vec<i64>,
}

impl TapLayout {
    pub(crate) fn ndim(&self) -> usize {
        self.extents.len()
    }

    /// Truncate or extend; new modes get extent 1 and stride 0.
    pub(crate) fn resize(&mut self, ndim: usize) {
        self.extents.resize(ndim, 1);
        self.strides.resize(ndim, 0);
    }
}

pub(crate) struct TensorInfo {
    pub(crate) dtype: c_int,
    pub(crate) layout: TapLayout,
}

/// Bytes per element of a supported datatype.
pub(crate) fn elem_size(dtype: c_int) -> Option<usize> {
    match dtype {
        TAPP_F32 => Some(4),
        TAPP_F64 | TAPP_C32 => Some(8),
        TAPP_C64 => Some(16),
        _ => None,
    }
}

/// Describe one tensor: element type, rank, extents and strides (in elements).
///
/// Strides may be negative or zero; extents may not be negative. A rank of 0 is
/// legal and describes a scalar, in which case `extents` and `strides` are not
/// read and may be null. Overflow of the addressed range is checked when a
/// product is created, since the setters can change the description.
///
/// # Safety
/// `info` must be a valid, writable `*mut isize`. When `nmode > 0`, `extents`
/// and `strides` must each be valid for `nmode` `i64` reads. On success `*info`
/// receives a handle to release with [`TAPP_destroy_tensor_info`].
#[no_mangle]
pub unsafe extern "C" fn TAPP_create_tensor_info(
    info: *mut isize,
    dtype: c_int,
    nmode: c_int,
    extents: *const i64,
    strides: *const i64,
) -> c_int {
    ffi(move || {
        if info.is_null() {
            return Err(null("info out-parameter"));
        }
        if nmode < 0 {
            return Err(fail(TCONTRACT_ERR_SHAPE, "negative mode count"));
        }
        if elem_size(dtype).is_none() {
            return Err(fail(TCONTRACT_ERR_DTYPE, "unsupported datatype"));
        }
        let n = nmode as usize;
        if n > 0 && (extents.is_null() || strides.is_null()) {
            return Err(null("extents or strides"));
        }
        // SAFETY: `n` reads are valid per the contract.
        let (e, s) = unsafe {
            if n == 0 {
                (Vec::new(), Vec::new())
            } else {
                (
                    std::slice::from_raw_parts(extents, n).to_vec(),
                    std::slice::from_raw_parts(strides, n).to_vec(),
                )
            }
        };
        let layout = TapLayout {
            extents: e,
            strides: s,
        };
        // SAFETY: non-null and writable per the contract.
        unsafe { *info = Box::into_raw(Box::new(TensorInfo { dtype, layout })) as isize };
        Ok(())
    })
}

/// Release a tensor info from [`TAPP_create_tensor_info`].
///
/// A [`TAPP_create_tensor_product`](crate::product::TAPP_create_tensor_product) built from it does **not** borrow it — the
/// plan copies everything it needs — so an info may be destroyed while products
/// derived from it are still in use.
///
/// # Safety
/// `info` must be live and not already destroyed.
#[no_mangle]
pub unsafe extern "C" fn TAPP_destroy_tensor_info(info: isize) -> c_int {
    ffi(|| {
        if info == 0 {
            return Err(null("info"));
        }
        // SAFETY: a live handle from `TAPP_create_tensor_info`.
        drop(unsafe { Box::from_raw(info as *mut TensorInfo) });
        Ok(())
    })
}

/// The rank recorded in `info`, or `-1` if the handle is `0`.
///
/// # Safety
/// `info` must be `0` or live.
#[no_mangle]
pub unsafe extern "C" fn TAPP_get_nmodes(info: isize) -> c_int {
    // SAFETY: zero or live per the contract.
    match unsafe { (info as *const TensorInfo).as_ref() } {
        Some(t) => t.layout.ndim() as c_int,
        None => -1,
    }
}

/// Change the rank in place, truncating or extending.
///
/// New modes get extent 1 and stride 0, which is the identity for this engine:
/// extent-1 axes are dropped during planning. So growing the rank and then
/// setting extents and strides is well defined, but growing it and *not* doing
/// so leaves the tensor describing the same elements it did before.
///
/// # Safety
/// `info` must be `0` or live.
#[no_mangle]
pub unsafe extern "C" fn TAPP_set_nmodes(info: isize, nmodes: c_int) -> c_int {
    ffi(|| {
        // SAFETY: zero or live per the contract.
        let t = unsafe { (info as *mut TensorInfo).as_mut() }.ok_or_else(|| null("info"))?;
        if nmodes < 0 {
            return Err(fail(TCONTRACT_ERR_SHAPE, "negative mode count"));
        }
        t.layout.resize(nmodes as usize);
        Ok(())
    })
}

/// Copy `info`'s extents out. Returns nothing, as TAPP declares it, so query
/// the rank with [`TAPP_get_nmodes`] first.
///
/// # Safety
/// `info` must be `0` or live, and `extents` must be null or valid for
/// `TAPP_get_nmodes(info)` `i64` writes.
#[no_mangle]
pub unsafe extern "C" fn TAPP_get_extents(info: isize, extents: *mut i64) {
    // SAFETY: zero or live per the contract.
    if let Some(t) = unsafe { (info as *const TensorInfo).as_ref() } {
        if !extents.is_null() {
            // SAFETY: `ndim` writes are valid per the contract.
            unsafe {
                std::ptr::copy_nonoverlapping(t.layout.extents.as_ptr(), extents, t.layout.ndim())
            };
        }
    }
}

/// Replace `info`'s extents, keeping its rank.
///
/// # Safety
/// `info` must be `0` or live, and `extents` must be null or valid for
/// `TAPP_get_nmodes(info)` `i64` reads.
#[no_mangle]
pub unsafe extern "C" fn TAPP_set_extents(info: isize, extents: *const i64) -> c_int {
    ffi(|| {
        // SAFETY: zero or live per the contract.
        let t = unsafe { (info as *mut TensorInfo).as_mut() }.ok_or_else(|| null("info"))?;
        if extents.is_null() {
            return Err(null("extents"));
        }
        let n = t.layout.ndim();
        // SAFETY: `n` reads are valid per the contract.
        t.layout
            .extents
            .copy_from_slice(unsafe { std::slice::from_raw_parts(extents, n) });
        Ok(())
    })
}

/// Copy `info`'s strides out, in elements. As [`TAPP_get_extents`], this
/// returns nothing.
///
/// # Safety
/// `info` must be `0` or live, and `strides` must be null or valid for
/// `TAPP_get_nmodes(info)` `i64` writes.
#[no_mangle]
pub unsafe extern "C" fn TAPP_get_strides(info: isize, strides: *mut i64) {
    // SAFETY: zero or live per the contract.
    if let Some(t) = unsafe { (info as *const TensorInfo).as_ref() } {
        if !strides.is_null() {
            // SAFETY: `ndim` writes are valid per the contract.
            unsafe {
                std::ptr::copy_nonoverlapping(t.layout.strides.as_ptr(), strides, t.layout.ndim())
            };
        }
    }
}

/// Replace `info`'s strides, keeping its rank.
///
/// # Safety
/// `info` must be `0` or live, and `strides` must be null or valid for
/// `TAPP_get_nmodes(info)` `i64` reads.
#[no_mangle]
pub unsafe extern "C" fn TAPP_set_strides(info: isize, strides: *const i64) -> c_int {
    ffi(|| {
        // SAFETY: zero or live per the contract.
        let t = unsafe { (info as *mut TensorInfo).as_mut() }.ok_or_else(|| null("info"))?;
        if strides.is_null() {
            return Err(null("strides"));
        }
        let n = t.layout.ndim();
        // SAFETY: `n` reads are valid per the contract.
        t.layout
            .strides
            .copy_from_slice(unsafe { std::slice::from_raw_parts(strides, n) });
        Ok(())
    })
}
