//! Borrowed DLPack operands and their zero-copy conversion to strided views.
#![allow(non_camel_case_types, non_upper_case_globals)]

use crate::dlpack::*;
use crate::status::*;

/// A borrowed operand: a `DLTensor` view plus the DLPack flags that a bare
/// `DLTensor` does not carry. Valid only for the duration of a call.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct tcontract_tensor {
    /// Borrowed view; never freed by tcontract.
    pub view: *mut DLTensor,
    /// `DLPACK_FLAG_BITMASK_*`.
    pub flags: u64,
}

/// Borrow a versioned tensor: copies the view pointer and flags; takes no
/// ownership and never calls the deleter.
///
/// # Safety
///
/// `m` is null or points to a valid `DLManagedTensorVersioned` that outlives
/// every use of the result.
#[no_mangle]
pub unsafe extern "C" fn tcontract_tensor_borrow_versioned(
    m: *mut DLManagedTensorVersioned,
) -> tcontract_tensor {
    if m.is_null() {
        return tcontract_tensor {
            view: std::ptr::null_mut(),
            flags: 0,
        };
    }
    // SAFETY: non-null and valid per the contract. Only DLPack major
    // version 1 has the layout mirrored here; another major version yields a
    // null view, which every call rejects with TCONTRACT_ERR_INVALID_ARGUMENT.
    unsafe {
        if (*m).version.major != 1 {
            return tcontract_tensor {
                view: std::ptr::null_mut(),
                flags: 0,
            };
        }
        tcontract_tensor {
            view: &mut (*m).dl_tensor,
            flags: (*m).flags,
        }
    }
}

/// Borrow a raw `DLTensor` with caller-supplied flags (DLPack 0.x producers).
/// Writability of an output is then the caller's precondition.
#[no_mangle]
pub extern "C" fn tcontract_tensor_borrow_raw(t: *mut DLTensor, flags: u64) -> tcontract_tensor {
    tcontract_tensor { view: t, flags }
}

/// Element types crossing the ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    /// `float32`.
    F32,
    /// `float64`.
    F64,
    /// `complex64`.
    C32,
    /// `complex128`.
    C64,
}

impl DType {
    fn size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F64 | DType::C32 => 8,
            DType::C64 => 16,
        }
    }

    /// Alignment of the Rust element type (a complex number aligns like its
    /// real part).
    fn align(self) -> usize {
        match self {
            DType::F32 | DType::C32 => 4,
            DType::F64 | DType::C64 => 8,
        }
    }
}

/// The element type of an operand.
///
/// # Errors
///
/// Null view, `lanes != 1`, or an unsupported type.
pub fn dtype_of(t: &tcontract_tensor) -> Result<DType, FfiError> {
    let v = view_ref(t)?;
    let d = v.dtype;
    if d.lanes != 1 {
        return Err(FfiError::new(
            TCONTRACT_ERR_DTYPE,
            format!("lanes = {} (must be 1)", d.lanes),
        ));
    }
    match (d.code, d.bits) {
        (kDLFloat, 32) => Ok(DType::F32),
        (kDLFloat, 64) => Ok(DType::F64),
        (kDLComplex, 64) => Ok(DType::C32),
        (kDLComplex, 128) => Ok(DType::C64),
        (c, b) => Err(FfiError::new(
            TCONTRACT_ERR_DTYPE,
            format!("unsupported dtype code {c} bits {b}"),
        )),
    }
}

fn view_ref(t: &tcontract_tensor) -> Result<&DLTensor, FfiError> {
    if t.view.is_null() {
        return Err(FfiError::new(
            TCONTRACT_ERR_INVALID_ARGUMENT,
            "null DLTensor",
        ));
    }
    // SAFETY: non-null; the caller guarantees a valid DLTensor for the call.
    Ok(unsafe { &*t.view })
}

/// Validated layout of an operand in elements of its type.
#[derive(Clone, Debug)]
pub struct Layout {
    /// Extents.
    pub dims: Vec<usize>,
    /// Element strides.
    pub strides: Vec<isize>,
    /// Pointer to the element at logical index 0 (dangling for empty).
    pub origin: *mut u8,
    /// Smallest and largest element offsets addressed (relative to origin).
    pub span: (isize, isize),
    /// Whether any extent is zero.
    pub empty: bool,
}

/// Validate an operand's device, dtype, shape, strides and offset.
///
/// # Errors
///
/// Typed statuses for every rejected field; checked arithmetic throughout.
pub fn layout(t: &tcontract_tensor, dt: DType) -> Result<Layout, FfiError> {
    let v = view_ref(t)?;
    if v.device.device_type != kDLCPU && v.device.device_type != kDLCUDAHost {
        return Err(FfiError::new(
            TCONTRACT_ERR_DEVICE,
            format!("device type {} is not CPU", v.device.device_type),
        ));
    }
    if dtype_of(t)? != dt {
        return Err(FfiError::new(TCONTRACT_ERR_DTYPE, "operand dtypes differ"));
    }
    let nd =
        usize::try_from(v.ndim).map_err(|_| FfiError::new(TCONTRACT_ERR_SHAPE, "negative ndim"))?;
    if nd > 0 && v.shape.is_null() {
        return Err(FfiError::new(TCONTRACT_ERR_INVALID_ARGUMENT, "null shape"));
    }
    let shape: &[i64] = if nd == 0 {
        &[]
    } else {
        // SAFETY: non-null, `ndim` entries per the DLPack contract.
        unsafe { std::slice::from_raw_parts(v.shape, nd) }
    };
    let shape_err = |m: &str| FfiError::new(TCONTRACT_ERR_SHAPE, m.to_string());
    let dims: Vec<usize> = shape
        .iter()
        .map(|&e| usize::try_from(e).map_err(|_| shape_err("negative extent")))
        .collect::<Result<_, _>>()?;
    let strides: Vec<isize> = if v.strides.is_null() {
        // Compact row-major (DLPack convention).
        let mut s = vec![0isize; nd];
        let mut acc = 1isize;
        for i in (0..nd).rev() {
            s[i] = acc;
            acc = acc
                .checked_mul(
                    isize::try_from(dims[i].max(1)).map_err(|_| shape_err("extent overflow"))?,
                )
                .ok_or_else(|| shape_err("stride overflow"))?;
        }
        s
    } else {
        // SAFETY: non-null, `ndim` entries per the DLPack contract.
        let st = unsafe { std::slice::from_raw_parts(v.strides, nd) };
        st.iter()
            .map(|&s| isize::try_from(s).map_err(|_| shape_err("stride overflow")))
            .collect::<Result<_, _>>()?
    };
    let size = dt.size();
    // `byte_offset` itself is unconstrained: only the effective address
    // `data + byte_offset` must be aligned to the element (the real type's
    // alignment for complex dtypes), checked below.
    let empty = dims.contains(&0);
    let (mut lo, mut hi) = (0isize, 0isize);
    if !empty {
        for (&d, &s) in dims.iter().zip(&strides) {
            let reach = isize::try_from(d - 1)
                .ok()
                .and_then(|x| x.checked_mul(s))
                .ok_or_else(|| shape_err("span overflow"))?;
            if reach < 0 {
                lo = lo
                    .checked_add(reach)
                    .ok_or_else(|| shape_err("span overflow"))?;
            } else {
                hi = hi
                    .checked_add(reach)
                    .ok_or_else(|| shape_err("span overflow"))?;
            }
        }
        if v.data.is_null() {
            return Err(FfiError::new(
                TCONTRACT_ERR_INVALID_ARGUMENT,
                "null data for a non-empty tensor",
            ));
        }
        // Byte extent of the span must be addressable.
        hi.checked_sub(lo)
            .and_then(|x| x.checked_add(1))
            .and_then(|x| x.checked_mul(size as isize))
            .ok_or_else(|| shape_err("span overflow"))?;
    }
    let origin = if empty {
        std::ptr::NonNull::<u8>::dangling().as_ptr()
    } else {
        let off = usize::try_from(v.byte_offset).map_err(|_| shape_err("byte_offset overflow"))?;
        // SAFETY: `data + byte_offset` is the first element per DLPack.
        let o = unsafe { (v.data as *mut u8).add(off) };
        if !(o as usize).is_multiple_of(dt.align()) {
            return Err(FfiError::new(
                TCONTRACT_ERR_INVALID_ARGUMENT,
                format!("data + byte_offset is not aligned to {} bytes", dt.align()),
            ));
        }
        o
    };
    Ok(Layout {
        dims,
        strides,
        origin,
        span: (lo, hi),
        empty,
    })
}

/// A read-only strided view over the addressed span (zero copy).
///
/// # Safety
///
/// The caller's contract (documented in `tcontract/core.h`): the whole span from
/// the lowest to the highest addressed element, padding included, is
/// initialized memory of `T` that no other thread writes during the call.
pub unsafe fn view<'a, T>(l: &'a Layout) -> Result<strided_view::StridedView<'a, T>, FfiError> {
    let (data, off): (&'a [T], isize) = if l.empty {
        (&[], 0)
    } else {
        let len = (l.span.1 - l.span.0 + 1) as usize;
        // SAFETY: the span [lo, hi] around the origin is the caller's valid memory.
        unsafe {
            (
                std::slice::from_raw_parts((l.origin as *const T).offset(l.span.0), len),
                -l.span.0,
            )
        }
    };
    strided_view::StridedView::new(data, &l.dims, &l.strides, off)
        .map_err(|e| FfiError::new(TCONTRACT_ERR_SHAPE, e.to_string()))
}

/// A mutable strided view over the addressed span; rejects read-only
/// operands before any write.
///
/// # Safety
///
/// As [`view`], plus no other access to the span during the call (outputs
/// whose span overlaps an input's are rejected by the callers).
pub unsafe fn view_mut<'a, T>(
    t: &tcontract_tensor,
    l: &'a Layout,
) -> Result<strided_view::StridedViewMut<'a, T>, FfiError> {
    if t.flags & DLPACK_FLAG_BITMASK_READ_ONLY != 0 {
        return Err(FfiError::new(
            TCONTRACT_ERR_READ_ONLY,
            "output is read-only",
        ));
    }
    let (data, off): (&'a mut [T], isize) = if l.empty {
        (&mut [], 0)
    } else {
        let len = (l.span.1 - l.span.0 + 1) as usize;
        // SAFETY: as `view`, exclusively.
        unsafe {
            (
                std::slice::from_raw_parts_mut((l.origin as *mut T).offset(l.span.0), len),
                -l.span.0,
            )
        }
    };
    strided_view::StridedViewMut::new(data, &l.dims, &l.strides, off)
        .map_err(|e| FfiError::new(TCONTRACT_ERR_SHAPE, e.to_string()))
}

/// Whether two operands' addressed byte ranges overlap (conservative: spans,
/// not exact element sets). Empty operands never overlap.
pub fn spans_overlap(a: &Layout, b: &Layout, elem: usize) -> bool {
    if a.empty || b.empty {
        return false;
    }
    let range = |l: &Layout| {
        let lo = l.origin as isize + l.span.0 * elem as isize;
        let hi = l.origin as isize + (l.span.1 + 1) * elem as isize;
        (lo, hi)
    };
    let (a0, a1) = range(a);
    let (b0, b1) = range(b);
    a0 < b1 && b0 < a1
}

/// Element size of a dtype.
pub fn elem_size(dt: DType) -> usize {
    dt.size()
}
