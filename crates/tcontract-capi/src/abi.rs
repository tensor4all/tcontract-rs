//! The TAPP C ABI's constants (datatypes, element operations, error codes) and
//! the one mapping from a contraction error to a C status.

use std::os::raw::c_int;

use crate::status::{
    FfiError, TCONTRACT_BUSY, TCONTRACT_ERR_ALIASED, TCONTRACT_ERR_DTYPE, TCONTRACT_ERR_INTERNAL,
    TCONTRACT_ERR_INVALID_ARGUMENT, TCONTRACT_ERR_LABELS, TCONTRACT_ERR_PANIC, TCONTRACT_ERR_SHAPE,
    TCONTRACT_ERR_UNSUPPORTED, TCONTRACT_ERR_WOULD_DEADLOCK, TCONTRACT_ERR_WOULD_MATERIALIZE,
};
use tcontract::api;

// The `datatype_t` enumerators, in the order the upstream header declares them.
// The numeric values are ABI and must not be reordered — TBLIS 1.3 and 2.0 differ
// by exactly such a reordering of their own `type_t`, which produces plausible
// wrong answers rather than an error.

/// `TAPP_F32`: single-precision real. Handled by [`tcontract`] as `f32`.
pub const TAPP_F32: c_int = 0;
/// `TAPP_F64`: double-precision real, i.e. `f64`.
pub const TAPP_F64: c_int = 1;
/// `TAPP_C32`: single-precision complex, interleaved — layout-compatible with
/// C99 `float _Complex` and `num_complex::Complex<f32>`.
pub const TAPP_C32: c_int = 2;
/// `TAPP_C64`: double-precision complex, interleaved.
pub const TAPP_C64: c_int = 3;
/// `TAPP_F16`: accepted by the enumeration, rejected by this implementation
/// with [`TAPP_ERROR_DATATYPE`]. There is no storage scalar for it.
pub const TAPP_F16: c_int = 4;
/// `TAPP_BF16`: as [`TAPP_F16`], rejected.
pub const TAPP_BF16: c_int = 5;

/// `TAPP_DEFAULT_PREC`: compute at the storage precision.
pub const TAPP_DEFAULT_PREC: c_int = -1;
/// `TAPP_F32F32_ACCUM_F32`: accepted for `f32` and `c32` storage.
pub const TAPP_F32F32_ACCUM_F32: c_int = 0;
/// `TAPP_F64F64_ACCUM_F64`: accepted for `f64` and `c64` storage.
pub const TAPP_F64F64_ACCUM_F64: c_int = 1;

/// `TAPP_IDENTITY`: read the operand as stored. The default for every operand.
pub const TAPP_IDENTITY: c_int = 0;
/// `TAPP_CONJUGATE`: complex-conjugate the operand. Free here — it is folded
/// into packing or into the write-back, never a separate pass.
pub const TAPP_CONJUGATE: c_int = 1;

// ------------------------------------------------------------------- errors

/// No error. The only value for which `TAPP_check_success` returns `true`.
pub const TAPP_SUCCESS: c_int = 0;
/// A required pointer was null, or a handle was `0` / not live. (The tcontract
/// status `TCONTRACT_ERR_INVALID_ARGUMENT`.)
pub const TAPP_ERROR_NULL: c_int = TCONTRACT_ERR_INVALID_ARGUMENT;
/// An unsupported element type, or the four operands did not agree on one.
pub const TAPP_ERROR_DATATYPE: c_int = TCONTRACT_ERR_DTYPE;
/// Extents or strides are inconsistent: a label with two different extents, a
/// negative extent, or a shape or stride that overflows.
pub const TAPP_ERROR_SHAPE: c_int = TCONTRACT_ERR_SHAPE;
/// The index labels do not describe a contraction: wrong count for the rank, or
/// `C` and `D` labelled differently.
pub const TAPP_ERROR_LABELS: c_int = TCONTRACT_ERR_LABELS;
/// A well-formed request this implementation declines: TAPP case 5 (an
/// output-only index, a broadcast), a computational precision other than the
/// storage precision, an element operation other than identity or conjugation,
/// or a null `C` with a nonzero `beta`.
pub const TAPP_ERROR_UNSUPPORTED: c_int = TCONTRACT_ERR_UNSUPPORTED;
/// A bug or an allocation failure.
pub const TAPP_ERROR_INTERNAL: c_int = TCONTRACT_ERR_INTERNAL;
/// `TAPP_destroy_executor` with calls in flight; the handle stays live.
pub const TAPP_ERROR_BUSY: c_int = TCONTRACT_BUSY;
/// `TAPP_destroy_executor` from one of the executor's own workers; the handle
/// stays live.
pub const TAPP_ERROR_WOULD_DEADLOCK: c_int = TCONTRACT_ERR_WOULD_DEADLOCK;
/// A panic was caught at the ABI boundary and reported instead of unwinding.
pub const TAPP_ERROR_PANIC: c_int = TCONTRACT_ERR_PANIC;
/// Output memory that overlaps an input or itself, or `C` and `D` that overlap
/// without being the same mapping.
pub const TAPP_ERROR_ALIASED: c_int = TCONTRACT_ERR_ALIASED;

pub(crate) fn fail(status: c_int, message: impl Into<String>) -> FfiError {
    FfiError::new(status, message)
}

pub(crate) fn null(what: &str) -> FfiError {
    fail(
        TCONTRACT_ERR_INVALID_ARGUMENT,
        format!("{what} is null or zero"),
    )
}

/// The C status of a contraction error: the one table of the ABI.
///
/// A label count or a `C`/`D` label-set mismatch is `LABELS`; extents, checked
/// sizes and layout-range incompatibilities are `SHAPE`; an output (or `C`)
/// that overlaps itself or an input is `ALIASED`; an unsupported operation
/// (including an output-only label) or an unusable forced family is
/// `UNSUPPORTED`; a wrong storage scalar is `DTYPE`.
pub(crate) fn map_err(e: api::Error) -> FfiError {
    use api::{ConfigError, Error as E, ShapeError, Unsupported};
    let status = match &e {
        E::Shape(ShapeError::LabelCount { .. } | ShapeError::OutputLabelMismatch) => {
            TCONTRACT_ERR_LABELS
        }
        E::Shape(_) | E::Layout(_) => TCONTRACT_ERR_SHAPE,
        E::Alias(_) => TCONTRACT_ERR_ALIASED,
        E::Unsupported(Unsupported::WouldMaterialize { .. }) => TCONTRACT_ERR_WOULD_MATERIALIZE,
        E::Unsupported(_) | E::Select(_) | E::Exec(_) => TCONTRACT_ERR_UNSUPPORTED,
        E::Config(ConfigError::DtypeMismatch { .. }) => TCONTRACT_ERR_DTYPE,
        E::Config(ConfigError::CLabels) => TCONTRACT_ERR_LABELS,
        E::Config(_) => TCONTRACT_ERR_INVALID_ARGUMENT,
        _ => TCONTRACT_ERR_INTERNAL,
    };
    fail(status, e.to_string())
}
