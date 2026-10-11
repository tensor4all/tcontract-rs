//! `#[repr(C)]` mirror of DLPack 1.x (`dlpack.h`, Apache-2.0,
//! <https://github.com/dmlc/dlpack>): only the types tcontract reads.
#![allow(non_camel_case_types, non_upper_case_globals)]

use std::ffi::c_void;

/// `kDLCPU`.
pub const kDLCPU: i32 = 1;
/// `kDLCUDAHost` (pinned host memory, CPU-addressable).
pub const kDLCUDAHost: i32 = 3;
/// `kDLFloat`.
pub const kDLFloat: u8 = 2;
/// `kDLComplex`.
pub const kDLComplex: u8 = 5;
/// `DLPACK_FLAG_BITMASK_READ_ONLY`.
pub const DLPACK_FLAG_BITMASK_READ_ONLY: u64 = 1 << 0;

/// `DLDevice`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DLDevice {
    /// `DLDeviceType`.
    pub device_type: i32,
    /// Device ordinal.
    pub device_id: i32,
}

/// `DLDataType`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DLDataType {
    /// `DLDataTypeCode`.
    pub code: u8,
    /// Bits per lane.
    pub bits: u8,
    /// Lanes (must be 1 for tcontract).
    pub lanes: u16,
}

/// `DLTensor`: a borrowed view; strides and offsets in elements, except
/// `byte_offset`.
#[repr(C)]
#[derive(Debug)]
pub struct DLTensor {
    /// Base pointer.
    pub data: *mut c_void,
    /// Device.
    pub device: DLDevice,
    /// Rank.
    pub ndim: i32,
    /// Element type.
    pub dtype: DLDataType,
    /// `ndim` extents.
    pub shape: *mut i64,
    /// `ndim` element strides, or null for compact row-major.
    pub strides: *mut i64,
    /// Byte offset from `data` to the first element.
    pub byte_offset: u64,
}

/// `DLPackVersion`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DLPackVersion {
    /// Major.
    pub major: u32,
    /// Minor.
    pub minor: u32,
}

/// `DLManagedTensorVersioned`.
#[repr(C)]
pub struct DLManagedTensorVersioned {
    /// Version of the producer.
    pub version: DLPackVersion,
    /// Producer context.
    pub manager_ctx: *mut c_void,
    /// Producer deleter (never called by tcontract for borrowed tensors).
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
    /// `DLPACK_FLAG_BITMASK_*`.
    pub flags: u64,
    /// The tensor.
    pub dl_tensor: DLTensor,
}
