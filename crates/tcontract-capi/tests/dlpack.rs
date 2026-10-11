use std::ffi::c_void;

use tcontract_capi::dlpack::*;
use tcontract_capi::status::*;
use tcontract_capi::tensor::{
    dtype_of, layout, tcontract_tensor_borrow_raw, view, view_mut, DType,
};

fn tensor(
    data: &mut [f64],
    shape: &mut [i64],
    strides: Option<&mut [i64]>,
    byte_offset: u64,
) -> DLTensor {
    DLTensor {
        data: data.as_mut_ptr() as *mut c_void,
        device: DLDevice {
            device_type: kDLCPU,
            device_id: 0,
        },
        ndim: shape.len() as i32,
        dtype: DLDataType {
            code: kDLFloat,
            bits: 64,
            lanes: 1,
        },
        shape: shape.as_mut_ptr(),
        strides: strides.map_or(std::ptr::null_mut(), |s| s.as_mut_ptr()),
        byte_offset,
    }
}

#[test]
fn null_strides_mean_row_major_and_views_are_zero_copy() {
    let mut d: Vec<f64> = (0..6).map(|x| x as f64).collect();
    let mut shape = [2i64, 3];
    let mut t = tensor(&mut d, &mut shape, None, 0);
    let op = tcontract_tensor_borrow_raw(&mut t, 0);
    assert_eq!(dtype_of(&op).unwrap(), DType::F64);
    let l = layout(&op, DType::F64).unwrap();
    assert_eq!(l.strides, vec![3, 1]);
    let v = unsafe { view::<f64>(&l).unwrap() };
    assert_eq!(v.get(&[1, 2]), 5.0);
    assert_eq!(v.ptr(), d.as_ptr(), "zero copy: same address");
}

#[test]
fn column_major_negative_strides_and_byte_offset() {
    let mut d: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let mut shape = [2i64, 3];
    // Reversed rows of a column-major 2x3 stored at offset 4 elements.
    let mut strides = [-1i64, 2];
    let mut t = tensor(&mut d, &mut shape, Some(&mut strides), 5 * 8);
    let op = tcontract_tensor_borrow_raw(&mut t, 0);
    let l = layout(&op, DType::F64).unwrap();
    let v = unsafe { view::<f64>(&l).unwrap() };
    assert_eq!(v.get(&[0, 0]), 5.0);
    assert_eq!(v.get(&[1, 0]), 4.0);
    assert_eq!(v.get(&[1, 2]), 8.0);
    assert_eq!(v.ptr(), unsafe { d.as_ptr().add(5) });
}

#[test]
fn rejections_are_typed() {
    let mut d = vec![0.0f64; 4];
    let mut shape = [2i64, 2];
    let mut t = tensor(&mut d, &mut shape, None, 0);
    // read-only output
    let ro = tcontract_tensor_borrow_raw(&mut t, DLPACK_FLAG_BITMASK_READ_ONLY);
    let l = layout(&ro, DType::F64).unwrap();
    assert_eq!(
        unsafe { view_mut::<f64>(&ro, &l) }.unwrap_err().status,
        TCONTRACT_ERR_READ_ONLY
    );
    // dtype mismatch, lanes, device, misaligned offset
    let op = tcontract_tensor_borrow_raw(&mut t, 0);
    assert_eq!(
        layout(&op, DType::F32).unwrap_err().status,
        TCONTRACT_ERR_DTYPE
    );
    t.dtype.lanes = 2;
    assert_eq!(
        dtype_of(&tcontract_tensor_borrow_raw(&mut t, 0))
            .unwrap_err()
            .status,
        TCONTRACT_ERR_DTYPE
    );
    t.dtype.lanes = 1;
    t.device.device_type = 2; // CUDA
    assert_eq!(
        layout(&tcontract_tensor_borrow_raw(&mut t, 0), DType::F64)
            .unwrap_err()
            .status,
        TCONTRACT_ERR_DEVICE
    );
    t.device.device_type = kDLCPU;
    t.byte_offset = 3;
    assert_eq!(
        layout(&tcontract_tensor_borrow_raw(&mut t, 0), DType::F64)
            .unwrap_err()
            .status,
        TCONTRACT_ERR_INVALID_ARGUMENT
    );
    t.byte_offset = 0;
    // null view, null shape
    assert_eq!(
        dtype_of(&tcontract_tensor_borrow_raw(std::ptr::null_mut(), 0))
            .unwrap_err()
            .status,
        TCONTRACT_ERR_INVALID_ARGUMENT
    );
    t.shape = std::ptr::null_mut();
    assert_eq!(
        layout(&tcontract_tensor_borrow_raw(&mut t, 0), DType::F64)
            .unwrap_err()
            .status,
        TCONTRACT_ERR_INVALID_ARGUMENT
    );
}

#[test]
fn overflowing_spans_are_shape_errors() {
    let mut d = vec![0.0f64; 1];
    let mut shape = [3i64, 3];
    let mut strides = [i64::MAX / 2, 1];
    let mut t = tensor(&mut d, &mut shape, Some(&mut strides), 0);
    assert_eq!(
        layout(&tcontract_tensor_borrow_raw(&mut t, 0), DType::F64)
            .unwrap_err()
            .status,
        TCONTRACT_ERR_SHAPE
    );
    let mut shape = [-1i64];
    let mut t = tensor(&mut d, &mut shape, None, 0);
    assert_eq!(
        layout(&tcontract_tensor_borrow_raw(&mut t, 0), DType::F64)
            .unwrap_err()
            .status,
        TCONTRACT_ERR_SHAPE
    );
}

#[test]
fn empty_tensors_never_touch_data() {
    let mut shape = [0i64, 5];
    let mut strides = [1i64 << 40, 7];
    let mut t = DLTensor {
        data: std::ptr::null_mut(),
        device: DLDevice {
            device_type: kDLCPU,
            device_id: 0,
        },
        ndim: 2,
        dtype: DLDataType {
            code: kDLFloat,
            bits: 64,
            lanes: 1,
        },
        shape: shape.as_mut_ptr(),
        strides: strides.as_mut_ptr(),
        byte_offset: 0,
    };
    let op = tcontract_tensor_borrow_raw(&mut t, 0);
    let l = layout(&op, DType::F64).unwrap();
    assert!(l.empty);
    let v = unsafe { view::<f64>(&l).unwrap() };
    assert_eq!(v.dims(), &[0, 5]);
}

#[test]
fn rust_mirror_matches_the_c_layout() {
    use std::mem::{offset_of, size_of};
    assert_eq!(size_of::<DLTensor>(), 48);
    assert_eq!(offset_of!(DLManagedTensorVersioned, dl_tensor), 32);
    assert_eq!(size_of::<tcontract_capi::tensor::tcontract_tensor>(), 16);
}

#[test]
fn complex_byte_offsets_need_element_alignment_not_element_size() {
    // #16: complex elements are aligned to their real type, so an offset of
    // one real (4 bytes for complex64, 8 for complex128) is valid.
    for (bits, dt, off, ok) in [
        (128u8, DType::C64, 8u64, true),
        (64, DType::C32, 4, true),
        (128, DType::C64, 4, false),
        (64, DType::C32, 2, false),
    ] {
        let mut d = vec![0.0f64; 4];
        let mut shape = [1i64, 1];
        let mut t = tensor(&mut d, &mut shape, None, off);
        t.dtype = DLDataType {
            code: kDLComplex,
            bits,
            lanes: 1,
        };
        let r = layout(&tcontract_tensor_borrow_raw(&mut t, 0), dt);
        if ok {
            let l = r.unwrap_or_else(|e| panic!("bits={bits} off={off}: {}", e.message));
            assert_eq!(l.origin as usize, d.as_ptr() as usize + off as usize);
        } else {
            assert_eq!(r.unwrap_err().status, TCONTRACT_ERR_INVALID_ARGUMENT);
        }
    }
}
