//! Regressions from the Phase 1f review.
use std::ffi::c_void;

use tcontract_capi::dlpack::*;
use tcontract_capi::status::*;
use tcontract_capi::tensor::{
    layout, tcontract_tensor_borrow_raw, tcontract_tensor_borrow_versioned, DType,
};

#[test]
fn misaligned_data_is_rejected_before_any_view() {
    let mut buf = vec![0u8; 8 * 5 + 1];
    let mut shape = [4i64];
    let mut t = DLTensor {
        data: unsafe { buf.as_mut_ptr().add(1) } as *mut c_void, // not 8-aligned
        device: DLDevice {
            device_type: kDLCPU,
            device_id: 0,
        },
        ndim: 1,
        dtype: DLDataType {
            code: kDLFloat,
            bits: 64,
            lanes: 1,
        },
        shape: shape.as_mut_ptr(),
        strides: std::ptr::null_mut(),
        byte_offset: 0,
    };
    let e = layout(&tcontract_tensor_borrow_raw(&mut t, 0), DType::F64).unwrap_err();
    assert_eq!(e.status, TCONTRACT_ERR_INVALID_ARGUMENT, "{}", e.message);
}

#[test]
fn unsupported_dlpack_major_version_is_not_borrowed() {
    let mut shape = [1i64];
    let mut x = [0.0f64];
    let mut m = DLManagedTensorVersioned {
        version: DLPackVersion { major: 2, minor: 0 },
        manager_ctx: std::ptr::null_mut(),
        deleter: None,
        flags: 0,
        dl_tensor: DLTensor {
            data: x.as_mut_ptr() as *mut c_void,
            device: DLDevice {
                device_type: kDLCPU,
                device_id: 0,
            },
            ndim: 1,
            dtype: DLDataType {
                code: kDLFloat,
                bits: 64,
                lanes: 1,
            },
            shape: shape.as_mut_ptr(),
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        },
    };
    assert!(unsafe { tcontract_tensor_borrow_versioned(&mut m) }
        .view
        .is_null());
    m.version.major = 1;
    assert!(!unsafe { tcontract_tensor_borrow_versioned(&mut m) }
        .view
        .is_null());
}
