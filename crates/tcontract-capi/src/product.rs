//! Tensor products: the TAPP-to-`Labels` bridge. A product lowers its four
//! infos and label lists once into a `tcontract::Problem` and keeps one
//! `Plan` per storage type; role, layout and alias validation belong to the
//! contract crate.

use std::os::raw::c_int;

use num_complex::Complex;
use tcontract::api::{self, CSpec, DType, Labels, LayoutSpec, Op, OperandSpec, Problem};
use tcontract::{Plan, PlanConfig};

use crate::abi::*;
use crate::status::{ffi, FfiError, TCONTRACT_ERR_DTYPE, TCONTRACT_ERR_UNSUPPORTED};
use crate::tensor_info::TensorInfo;

/// The prepared contraction of one storage type.
pub(crate) enum Prepared {
    F32(Plan<f32>),
    F64(Plan<f64>),
    C32(Plan<Complex<f32>>),
    C64(Plan<Complex<f64>>),
}

pub(crate) struct Product {
    pub(crate) plan: Prepared,
}

fn op_of(op: c_int, name: &str) -> Result<Op, FfiError> {
    match op {
        TAPP_IDENTITY => Ok(Op::Identity),
        TAPP_CONJUGATE => Ok(Op::Conjugate),
        _ => Err(fail(
            TCONTRACT_ERR_UNSUPPORTED,
            format!("element operation {op} on {name} is not supported"),
        )),
    }
}

/// One operand as the problem describes it: validated extents and strides at
/// the pointer's own origin (TAPP has no logical offsets), and the element
/// operation.
fn spec(t: &TensorInfo, op: Op) -> Result<OperandSpec, FfiError> {
    let layout =
        LayoutSpec::from_signed(&t.layout.extents, &t.layout.strides, 0).map_err(map_err)?;
    Ok(OperandSpec::new(layout).with_op(op))
}

fn build<T: api::Scalar>(problem: &Problem) -> Result<Plan<T>, FfiError> {
    // The C boundary permits write-only D, so prepare that storage capability
    // once. This is not a tuning API or execution-time provider fallback.
    let config = PlanConfig {
        fresh_output: true,
        ..PlanConfig::default()
    };
    Plan::<T>::new(problem, &config).map_err(map_err)
}

/// Plan `D = op_D(alpha * op_A(A) * op_B(B) + beta * op_C(C))` from four tensor
/// infos and their index labels.
///
/// All the work that depends only on shapes, strides and labels happens here —
/// the problem is lowered and validated once, and the index classification,
/// folding and scatter vectors are built — so this is the call to hoist out of a
/// loop. The plan snapshots the metadata, owns no data pointer and no executor,
/// and can be executed against different buffers and serial or pooled executors.
///
/// `handle` must be a live library handle (zero is rejected). `prec` must be
/// [`TAPP_DEFAULT_PREC`] or the precision of the storage type
/// ([`TAPP_F32F32_ACCUM_F32`] for `f32`/`c32`, [`TAPP_F64F64_ACCUM_F64`] for
/// `f64`/`c64`); anything else is `TAPP_ERROR_UNSUPPORTED`, not ignored. Element
/// operations other than identity and conjugation are rejected too, and so are
/// four infos that do not agree on one storage type (TAPP permits mixed types;
/// this engine does not). Shapes and strides whose addressed range overflows
/// are `TAPP_ERROR_SHAPE`, and a `D` whose elements overlap one another is
/// `TAPP_ERROR_ALIASED`.
///
/// `C` is required, because the C signature has no way to omit it — pass
/// `beta == 0` at execution time to have it ignored, or pass `D`'s info for it.
///
/// # Safety
/// `plan_out` must be a valid, writable `*mut isize`. `a`, `b`, `c`, `d` must be
/// live tensor infos. Each `idx_*` must be valid for the corresponding info's
/// rank in `i64` reads, or may be null when that rank is 0. On success
/// `*plan_out` receives a handle to release with
/// [`TAPP_destroy_tensor_product`].
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn TAPP_create_tensor_product(
    plan_out: *mut isize,
    handle: isize,
    op_a: c_int,
    a: isize,
    idx_a: *const i64,
    op_b: c_int,
    b: isize,
    idx_b: *const i64,
    op_c: c_int,
    c: isize,
    idx_c: *const i64,
    op_d: c_int,
    d: isize,
    idx_d: *const i64,
    prec: c_int,
) -> c_int {
    ffi(move || {
        if plan_out.is_null() {
            return Err(null("plan out-parameter"));
        }
        if handle == 0 {
            return Err(null("library handle"));
        }
        // SAFETY: zero or live infos per the contract.
        let (ta, tb, tc, td) = unsafe {
            (
                (a as *const TensorInfo).as_ref(),
                (b as *const TensorInfo).as_ref(),
                (c as *const TensorInfo).as_ref(),
                (d as *const TensorInfo).as_ref(),
            )
        };
        let (Some(ta), Some(tb), Some(tc), Some(td)) = (ta, tb, tc, td) else {
            return Err(null("tensor info"));
        };

        // TAPP allows mixed storage types; this engine computes at a single
        // element type, so require all four to agree.
        if ta.dtype != tb.dtype || ta.dtype != tc.dtype || ta.dtype != td.dtype {
            return Err(fail(
                TCONTRACT_ERR_DTYPE,
                "operands have different storage types",
            ));
        }
        let storage_prec = match ta.dtype {
            TAPP_F32 | TAPP_C32 => TAPP_F32F32_ACCUM_F32,
            _ => TAPP_F64F64_ACCUM_F64,
        };
        if prec != TAPP_DEFAULT_PREC && prec != storage_prec {
            return Err(fail(
                TCONTRACT_ERR_UNSUPPORTED,
                format!("computational precision {prec} differs from the storage precision"),
            ));
        }
        let (oa, ob, oc, od) = (
            op_of(op_a, "A")?,
            op_of(op_b, "B")?,
            op_of(op_c, "C")?,
            op_of(op_d, "D")?,
        );

        let labels = |p: *const i64, t: &TensorInfo, name: &str| -> Result<Vec<i64>, FfiError> {
            let n = t.layout.ndim();
            if n == 0 {
                Ok(Vec::new())
            } else if p.is_null() {
                Err(null(&format!("labels of {name}")))
            } else {
                // SAFETY: `n` reads are valid per the contract.
                Ok(unsafe { std::slice::from_raw_parts(p, n) }.to_vec())
            }
        };
        let (la, lb, lc, ld) = (
            labels(idx_a, ta, "A")?,
            labels(idx_b, tb, "B")?,
            labels(idx_c, tc, "C")?,
            labels(idx_d, td, "D")?,
        );

        // The one lowering: labels, diagonals, reductions, extents, address
        // ranges and output injectivity are all checked by the problem.
        let dtype = match ta.dtype {
            TAPP_F32 => DType::F32,
            TAPP_F64 => DType::F64,
            TAPP_C32 => DType::C32,
            TAPP_C64 => DType::C64,
            _ => return Err(fail(TCONTRACT_ERR_DTYPE, "unsupported datatype")),
        };
        let problem = Problem::from_labels(
            dtype,
            spec(ta, oa)?,
            spec(tb, ob)?,
            CSpec::Separate(spec(tc, oc)?),
            spec(td, od)?,
            &Labels::new(&la, &lb, &ld).with_c(&lc),
        )
        .map_err(map_err)?;
        let plan = match dtype {
            DType::F32 => Prepared::F32(build(&problem)?),
            DType::F64 => Prepared::F64(build(&problem)?),
            DType::C32 => Prepared::C32(build(&problem)?),
            DType::C64 => Prepared::C64(build(&problem)?),
        };
        // SAFETY: non-null and writable per the contract.
        unsafe { *plan_out = Box::into_raw(Box::new(Product { plan })) as isize };
        Ok(())
    })
}

/// Release a product from [`TAPP_create_tensor_product`].
///
/// # Safety
/// `plan` must be live and not already destroyed.
#[no_mangle]
pub unsafe extern "C" fn TAPP_destroy_tensor_product(plan: isize) -> c_int {
    ffi(|| {
        if plan == 0 {
            return Err(null("plan"));
        }
        // SAFETY: a live handle from `TAPP_create_tensor_product`.
        drop(unsafe { Box::from_raw(plan as *mut Product) });
        Ok(())
    })
}
