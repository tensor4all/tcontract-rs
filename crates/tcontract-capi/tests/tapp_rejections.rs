//! TAPP C-ABI conformance: what must be refused, and how.
//!
//! Two claims are under test here and they are separate claims.
//!
//! 1. **What** is refused: TAPP case 5 (an index appearing only in the output),
//!    `TAPP_F16` / `TAPP_BF16`, and the assorted malformed inputs a C caller
//!    reaches by accident — null pointers, mismatched extents, invalid handles,
//!    values that are not enumerators.
//! 2. **How** it is refused: by returning a `TAPP_error`. A panic unwinding out
//!    of an `extern "C"` function is undefined behaviour, so "returns an error"
//!    is a soundness property of this crate and not a nicety. Every test below
//!    that expects a rejection therefore also proves the process survived it,
//!    because a panic in the callee would abort or unwind the test rather than
//!    let the assertion run.
//!
//! Where the ABI *cannot* detect an error — a non-zero handle that was never
//! created, a data pointer too short for its extents — that is recorded here in
//! a comment rather than asserted, because exercising it would be UB and a test
//! that invokes UB proves nothing.

mod common;

use std::ffi::c_void;
use std::os::raw::{c_char, c_int};

use common::*;
use tcontract_capi::*;

type C64 = num_complex::Complex<f64>;

const I: i64 = 0;
const J: i64 = 1;
const K: i64 = 2;
const X: i64 = 9;

/// The shapes of a well-formed `D[i,j] = A[i,k] B[k,j]`, for tests that perturb
/// exactly one thing about it.
fn good() -> (CTensor<'static>, CTensor<'static>, CTensor<'static>) {
    (
        CTensor::new(&[4, 3], &[1, 4], &[I, K]),
        CTensor::new(&[3, 5], &[1, 3], &[K, J]),
        CTensor::new(&[4, 5], &[1, 4], &[I, J]),
    )
}

// -------------------------------------------------------- documented rejections

/// TAPP case 5: an index that appears only in the output (broadcasting). The
/// crate documents this as rejected, "as TAPP permits".
#[test]
fn case5_isolated_output_index_is_rejected() {
    let (a, b, _) = good();
    // `X` is in `D` and nowhere else.
    let d = CTensor::new(&[4, 5, 2], &[1, 4, 20], &[I, J, X]);
    let av = seq::<C64>(a.storage(), 1);
    let bv = seq::<C64>(b.storage(), 2);
    let case = Case::plain(a, &av, b, &bv, d);
    let mut dv = case.zeroed_d();
    let e = unsafe { case.run(&mut dv) };
    assert_eq!(
        e,
        TAPP_ERROR_UNSUPPORTED,
        "case 5 must be refused with a status, got {}",
        explain_status(e)
    );
}

/// Case 5 is refused even when the broadcast index has extent 1, where it costs
/// nothing to support. The rejection is by index *class*, not by size, so it is
/// predictable from the labels alone.
#[test]
fn case5_is_rejected_even_at_extent_one() {
    let (a, b, _) = good();
    let d = CTensor::new(&[4, 5, 1], &[1, 4, 20], &[I, J, X]);
    let av = seq::<C64>(a.storage(), 3);
    let bv = seq::<C64>(b.storage(), 4);
    let case = Case::plain(a, &av, b, &bv, d);
    let mut dv = case.zeroed_d();
    assert_eq!(unsafe { case.run(&mut dv) }, TAPP_ERROR_UNSUPPORTED);
}

#[test]
fn f16_is_rejected_with_a_datatype_error() {
    let mut info = 0isize;
    let (e, s) = (&[4i64, 3], &[1i64, 4]);
    let err = unsafe { TAPP_create_tensor_info(&mut info, TAPP_F16, 2, e.as_ptr(), s.as_ptr()) };
    assert_eq!(err, TAPP_ERROR_DATATYPE, "{}", explain_status(err));
    assert_eq!(info, 0, "a failed create must not hand back a handle");
}

#[test]
fn bf16_is_rejected_with_a_datatype_error() {
    let mut info = 0isize;
    let (e, s) = (&[4i64, 3], &[1i64, 4]);
    let err = unsafe { TAPP_create_tensor_info(&mut info, TAPP_BF16, 2, e.as_ptr(), s.as_ptr()) };
    assert_eq!(err, TAPP_ERROR_DATATYPE, "{}", explain_status(err));
    assert_eq!(info, 0);
}

/// A datatype that is not a `TAPP_datatype` enumerator at all.
#[test]
fn out_of_range_datatypes_are_rejected() {
    let (e, s) = (&[4i64, 3], &[1i64, 4]);
    for dtype in [-1, 6, 7, 1234, c_int::MAX, c_int::MIN] {
        let mut info = 0isize;
        let err = unsafe { TAPP_create_tensor_info(&mut info, dtype, 2, e.as_ptr(), s.as_ptr()) };
        assert_eq!(
            err,
            TAPP_ERROR_DATATYPE,
            "dtype {dtype} was accepted: {}",
            explain_status(err)
        );
        assert_eq!(info, 0);
    }
}

/// This engine computes at one element type, so all four operands must agree.
/// TAPP permits mixed *storage* types; the crate documents that it does not.
#[test]
fn mismatched_datatypes_across_operands_are_rejected() {
    let (a, b, d) = good();
    unsafe {
        let handle = create_handle();
        let ia = a.create_info(TAPP_C64);
        let ib = b.create_info(TAPP_F64); // the odd one out
        let ic = d.create_info(TAPP_C64);
        let id = d.create_info(TAPP_C64);
        let mut plan = 0isize;
        let e = TAPP_create_tensor_product(
            &mut plan,
            handle,
            TAPP_IDENTITY,
            ia,
            a.labels_ptr(),
            TAPP_IDENTITY,
            ib,
            b.labels_ptr(),
            TAPP_IDENTITY,
            ic,
            d.labels_ptr(),
            TAPP_IDENTITY,
            id,
            d.labels_ptr(),
            TAPP_DEFAULT_PREC,
        );
        assert_eq!(e, TAPP_ERROR_DATATYPE, "{}", explain_status(e));
        assert_eq!(plan, 0);
        for i in [ia, ib, ic, id] {
            assert_eq!(TAPP_destroy_tensor_info(i), TAPP_SUCCESS);
        }
        assert_eq!(TAPP_destroy_handle(handle), TAPP_SUCCESS);
    }
}

// ------------------------------------------------------------- shape mismatches

/// The same label carrying different extents in `A` and `B`.
#[test]
fn mismatched_extents_between_operands_are_rejected() {
    let a = CTensor::new(&[4, 3], &[1, 4], &[I, K]);
    let b = CTensor::new(&[7, 5], &[1, 7], &[K, J]); // k is 3 in A, 7 here
    let d = CTensor::new(&[4, 5], &[1, 4], &[I, J]);
    let av = seq::<C64>(a.storage(), 11);
    let bv = seq::<C64>(b.storage(), 12);
    let case = Case::plain(a, &av, b, &bv, d);
    let mut dv = case.zeroed_d();
    let e = unsafe { case.run(&mut dv) };
    assert_eq!(e, TAPP_ERROR_SHAPE, "{}", explain_status(e));
}

/// A free index of `A` whose extent in `D` disagrees.
#[test]
fn mismatched_extents_between_a_and_d_are_rejected() {
    let a = CTensor::new(&[4, 3], &[1, 4], &[I, K]);
    let b = CTensor::new(&[3, 5], &[1, 3], &[K, J]);
    let d = CTensor::new(&[6, 5], &[1, 6], &[I, J]); // i is 4 in A
    let av = seq::<C64>(a.storage(), 13);
    let bv = seq::<C64>(b.storage(), 14);
    let case = Case::plain(a, &av, b, &bv, d);
    let mut dv = case.zeroed_d();
    let e = unsafe { case.run(&mut dv) };
    assert_eq!(e, TAPP_ERROR_SHAPE, "{}", explain_status(e));
}

/// A label repeated inside one tensor with inconsistent extents cannot name a
/// diagonal.
#[test]
fn repeated_label_with_inconsistent_extents_is_rejected() {
    let a = CTensor::new(&[4, 6], &[1, 4], &[I, I]);
    let b = CTensor::new(&[4, 5], &[1, 4], &[I, J]);
    let d = CTensor::new(&[5], &[1], &[J]);
    let av = seq::<C64>(a.storage(), 15);
    let bv = seq::<C64>(b.storage(), 16);
    let case = Case::plain(a, &av, b, &bv, d);
    let mut dv = case.zeroed_d();
    let e = unsafe { case.run(&mut dv) };
    assert_eq!(e, TAPP_ERROR_SHAPE, "{}", explain_status(e));
}

/// `C` and `D` must describe the same index set; a `C` missing one of `D`'s
/// labels is a label error and not a shape error.
#[test]
fn c_and_d_must_carry_the_same_label_set() {
    let (a, b, d) = good();
    let c = CTensor::new(&[4], &[1], &[I]); // no `j`
    let av = seq::<C64>(a.storage(), 17);
    let bv = seq::<C64>(b.storage(), 18);
    let cv = seq::<C64>(c.storage(), 19);
    let mut case = Case::plain(a, &av, b, &bv, d).with_c(&cv, C64::new(1.0, 0.0));
    case.c = c;
    let mut dv = case.zeroed_d();
    let e = unsafe { case.run(&mut dv) };
    assert_eq!(e, TAPP_ERROR_LABELS, "{}", explain_status(e));
}

#[test]
fn a_negative_extent_is_rejected() {
    let a = CTensor::new(&[4, -3], &[1, 4], &[I, K]);
    let b = CTensor::new(&[3, 5], &[1, 3], &[K, J]);
    let d = CTensor::new(&[4, 5], &[1, 4], &[I, J]);
    // `TAPP_create_tensor_info` accepts the layout; `Plan::new` is where a
    // negative extent is caught, so the failure surfaces from
    // `TAPP_create_tensor_product`.
    let av = seq::<C64>(16, 21);
    let bv = seq::<C64>(b.storage(), 22);
    let case = Case::plain(a, &av, b, &bv, d);
    let mut dv = case.zeroed_d();
    let e = unsafe { case.run(&mut dv) };
    assert_eq!(e, TAPP_ERROR_SHAPE, "{}", explain_status(e));
}

#[test]
fn a_negative_mode_count_is_rejected() {
    let mut info = 0isize;
    let (e, s) = (&[4i64, 3], &[1i64, 4]);
    for nmode in [-1, -7, c_int::MIN] {
        let err =
            unsafe { TAPP_create_tensor_info(&mut info, TAPP_F64, nmode, e.as_ptr(), s.as_ptr()) };
        assert_ne!(err, TAPP_SUCCESS, "nmode {nmode} was accepted");
        assert_eq!(info, 0);
    }
}

#[test]
fn set_nmodes_rejects_a_negative_count() {
    unsafe {
        let (_, _, d) = good();
        let info = d.create_info(TAPP_F64);
        assert_eq!(TAPP_set_nmodes(info, -1), TAPP_ERROR_SHAPE);
        // ...and leaves the info intact.
        assert_eq!(TAPP_get_nmodes(info), 2);
        assert_eq!(TAPP_destroy_tensor_info(info), TAPP_SUCCESS);
    }
}

// ----------------------------------------------------------------- zero extents

/// A zero extent in an output index makes the whole contraction empty. It must
/// succeed and touch nothing.
#[test]
fn a_zero_output_extent_is_a_no_op() {
    let a = CTensor::new(&[0, 3], &[1, 0], &[I, K]);
    let b = CTensor::new(&[3, 5], &[1, 3], &[K, J]);
    let d = CTensor::new(&[0, 5], &[1, 0], &[I, J]);
    let av: Vec<C64> = Vec::new();
    let bv = seq::<C64>(b.storage(), 31);
    // A one-element `D` buffer with a sentinel: the call must not write it.
    let mut dv = vec![C64::new(7.0, -7.0)];
    let case = Case::plain(a, &av, b, &bv, d);
    let e = unsafe { case.run(&mut dv) };
    assert_eq!(e, TAPP_SUCCESS, "{}", explain_status(e));
    assert_eq!(dv, vec![C64::new(7.0, -7.0)], "empty output was written to");
}

/// A zero extent on the *contracted* index makes the sum empty, so
/// `D = beta * op_C(C)` — including `D = 0` when `beta` is zero. This is the one
/// empty case that still writes `D`.
#[test]
fn a_zero_contraction_extent_scales_c() {
    let a = CTensor::new(&[4, 0], &[1, 4], &[I, K]);
    let b = CTensor::new(&[0, 5], &[1, 0], &[K, J]);
    let d = CTensor::new(&[4, 5], &[1, 4], &[I, J]);
    let av: Vec<C64> = Vec::new();
    let bv: Vec<C64> = Vec::new();
    let cv = seq::<C64>(d.storage(), 41);
    let beta = C64::new(2.0, -1.0);
    let out = Case::plain(a, &av, b, &bv, d)
        .with_c(&cv, beta)
        .check_zeroed();
    let want: Vec<C64> = cv.iter().map(|&v| v * beta).collect();
    assert_close::<C64>(&out, &want);

    // ...and with `beta = 0` it zeroes `D` rather than leaving it alone.
    let mut dv = seq::<C64>(d.storage(), 42);
    Case::plain(a, &av, b, &bv, d)
        .with_c(&cv, C64::new(0.0, 0.0))
        .check(&mut dv);
    assert!(dv.iter().all(|v| *v == C64::new(0.0, 0.0)));
}

/// A zero extent on a Hadamard index empties the batch loop, which is a third
/// distinct emptiness test from the two above: `M`/`N` and `K` are non-empty and
/// there is simply no batch to run.
#[test]
fn a_zero_hadamard_extent_is_a_no_op() {
    let a = CTensor::new(&[0, 4, 3], &[1, 0, 0], &[X, I, K]);
    let b = CTensor::new(&[0, 3, 5], &[1, 0, 0], &[X, K, J]);
    let d = CTensor::new(&[0, 4, 5], &[1, 0, 0], &[X, I, J]);
    let av: Vec<C64> = Vec::new();
    let bv: Vec<C64> = Vec::new();
    let mut dv = vec![C64::new(5.0, 5.0)];
    let case = Case::plain(a, &av, b, &bv, d);
    let e = unsafe { case.run(&mut dv) };
    assert_eq!(e, TAPP_SUCCESS, "{}", explain_status(e));
    assert_eq!(dv, vec![C64::new(5.0, 5.0)], "empty batch was written to");
}

/// An extent-1 mode carries no information and must not change the answer, even
/// though the index analysis drops such axes entirely.
#[test]
fn extent_one_modes_are_transparent() {
    let a = CTensor::new(&[4, 1, 3], &[1, 4, 4], &[I, X, K]);
    let b = CTensor::new(&[3, 5], &[1, 3], &[K, J]);
    let d = CTensor::new(&[4, 5, 1], &[1, 4, 20], &[I, J, X]);
    let av = seq::<C64>(a.storage(), 51);
    let bv = seq::<C64>(b.storage(), 52);
    // `X` is in A and D, hence a free index of A, not a broadcast.
    Case::plain(a, &av, b, &bv, d).check_zeroed();
}

// --------------------------------------------------------------- null pointers

/// Every out-parameter the ABI writes through must be checked.
#[test]
fn null_out_parameters_are_rejected() {
    let (e, s) = (&[4i64, 3], &[1i64, 4]);
    unsafe {
        assert_eq!(TAPP_create_handle(std::ptr::null_mut()), TAPP_ERROR_NULL);
        assert_eq!(TAPP_create_executor(std::ptr::null_mut()), TAPP_ERROR_NULL);
        assert_eq!(
            TAPP_create_tensor_info(std::ptr::null_mut(), TAPP_F64, 2, e.as_ptr(), s.as_ptr()),
            TAPP_ERROR_NULL
        );
        let handle = create_handle();
        let info = CTensor::new(&[4, 3], &[1, 4], &[I, K]).create_info(TAPP_F64);
        let l = [I, K];
        assert_eq!(
            TAPP_create_tensor_product(
                std::ptr::null_mut(),
                handle,
                TAPP_IDENTITY,
                info,
                l.as_ptr(),
                TAPP_IDENTITY,
                info,
                l.as_ptr(),
                TAPP_IDENTITY,
                info,
                l.as_ptr(),
                TAPP_IDENTITY,
                info,
                l.as_ptr(),
                TAPP_DEFAULT_PREC,
            ),
            TAPP_ERROR_NULL
        );
        assert_eq!(TAPP_destroy_tensor_info(info), TAPP_SUCCESS);
        assert_eq!(TAPP_destroy_handle(handle), TAPP_SUCCESS);
    }
}

/// A positive `nmode` with no extents or strides to read.
#[test]
fn null_extents_or_strides_are_rejected_when_nmode_is_positive() {
    let (e, s) = (&[4i64, 3], &[1i64, 4]);
    unsafe {
        let mut info = 0isize;
        assert_eq!(
            TAPP_create_tensor_info(&mut info, TAPP_F64, 2, std::ptr::null(), s.as_ptr()),
            TAPP_ERROR_NULL
        );
        assert_eq!(info, 0);
        assert_eq!(
            TAPP_create_tensor_info(&mut info, TAPP_F64, 2, e.as_ptr(), std::ptr::null()),
            TAPP_ERROR_NULL
        );
        assert_eq!(info, 0);
    }
}

/// A rank-zero tensor has nothing to read, so null arrays are legal there — the
/// scalar-output case a C caller writes as `TAPP_create_tensor_info(&i, t, 0,
/// NULL, NULL)`.
#[test]
fn a_rank_zero_info_needs_no_extent_or_stride_arrays() {
    unsafe {
        let mut info = 0isize;
        assert_eq!(
            TAPP_create_tensor_info(&mut info, TAPP_C64, 0, std::ptr::null(), std::ptr::null()),
            TAPP_SUCCESS
        );
        assert_ne!(info, 0);
        assert_eq!(TAPP_get_nmodes(info), 0);
        assert_eq!(TAPP_destroy_tensor_info(info), TAPP_SUCCESS);
    }
}

/// A null label array for a tensor that has modes.
#[test]
fn null_label_arrays_are_rejected() {
    let (a, b, d) = good();
    unsafe {
        let handle = create_handle();
        let ia = a.create_info(TAPP_C64);
        let ib = b.create_info(TAPP_C64);
        let id = d.create_info(TAPP_C64);
        for which in 0..4 {
            let mut plan = 0isize;
            let p = |n: usize, t: &CTensor| -> *const i64 {
                if n == which {
                    std::ptr::null()
                } else {
                    t.labels_ptr()
                }
            };
            let e = TAPP_create_tensor_product(
                &mut plan,
                handle,
                TAPP_IDENTITY,
                ia,
                p(0, &a),
                TAPP_IDENTITY,
                ib,
                p(1, &b),
                TAPP_IDENTITY,
                id,
                p(2, &d),
                TAPP_IDENTITY,
                id,
                p(3, &d),
                TAPP_DEFAULT_PREC,
            );
            assert_eq!(
                e,
                TAPP_ERROR_NULL,
                "null idx_{which} was accepted: {}",
                explain_status(e)
            );
            assert_eq!(plan, 0);
        }
        for i in [ia, ib, id] {
            assert_eq!(TAPP_destroy_tensor_info(i), TAPP_SUCCESS);
        }
        assert_eq!(TAPP_destroy_handle(handle), TAPP_SUCCESS);
    }
}

/// A zero handle is the one invalid handle value the ABI can recognise, and
/// `TAPP_create_tensor_product` must recognise it in every operand slot.
#[test]
fn zero_tensor_info_handles_are_rejected() {
    let (a, b, d) = good();
    unsafe {
        let handle = create_handle();
        let ia = a.create_info(TAPP_C64);
        let ib = b.create_info(TAPP_C64);
        let id = d.create_info(TAPP_C64);
        for which in 0..4 {
            let mut plan = 0isize;
            let h = |n: usize, good: isize| if n == which { 0 } else { good };
            let e = TAPP_create_tensor_product(
                &mut plan,
                handle,
                TAPP_IDENTITY,
                h(0, ia),
                a.labels_ptr(),
                TAPP_IDENTITY,
                h(1, ib),
                b.labels_ptr(),
                TAPP_IDENTITY,
                h(2, id),
                d.labels_ptr(),
                TAPP_IDENTITY,
                h(3, id),
                d.labels_ptr(),
                TAPP_DEFAULT_PREC,
            );
            assert_eq!(e, TAPP_ERROR_NULL, "zero handle in slot {which} accepted");
            assert_eq!(plan, 0);
        }
        for i in [ia, ib, id] {
            assert_eq!(TAPP_destroy_tensor_info(i), TAPP_SUCCESS);
        }
        assert_eq!(TAPP_destroy_handle(handle), TAPP_SUCCESS);
    }
}

/// `TAPP_execute_product` must refuse a zero plan handle and every null pointer
/// it dereferences unconditionally. `C` is the one pointer that may be null.
#[test]
fn execute_rejects_a_zero_plan_and_required_null_pointers() {
    let (a, b, d) = good();
    let av = seq::<C64>(a.storage(), 61);
    let bv = seq::<C64>(b.storage(), 62);
    let alpha = C64::new(1.0, 0.0);
    let beta = C64::new(0.0, 0.0);
    let mut dv = vec![C64::new(0.0, 0.0); d.storage()];
    unsafe {
        let handle = create_handle();
        let exec = create_executor();
        let (e, plan) = Case::plain(a, &av, b, &bv, d).create_plan(handle);
        assert_eq!(e, TAPP_SUCCESS, "{}", explain_status(e));

        let av_p = av.as_ptr() as *const c_void;
        let bv_p = bv.as_ptr() as *const c_void;
        let dv_p = dv.as_mut_ptr() as *mut c_void;
        let al_p = &alpha as *const C64 as *const c_void;
        let be_p = &beta as *const C64 as *const c_void;
        let nul: *const c_void = std::ptr::null();

        // A zero plan handle.
        assert_eq!(
            TAPP_execute_product(
                0,
                exec,
                std::ptr::null_mut(),
                al_p,
                av_p,
                bv_p,
                be_p,
                nul,
                dv_p
            ),
            TAPP_ERROR_NULL
        );
        // Each required pointer, one at a time.
        let required: [(&str, [*const c_void; 5]); 5] = [
            ("alpha", [nul, av_p, bv_p, be_p, nul]),
            ("A", [al_p, nul, bv_p, be_p, nul]),
            ("B", [al_p, av_p, nul, be_p, nul]),
            ("beta", [al_p, av_p, bv_p, nul, nul]),
            ("D", [al_p, av_p, bv_p, be_p, nul]),
        ];
        for (name, [al, a_, b_, be, c_]) in required {
            let d_ = if name == "D" {
                std::ptr::null_mut()
            } else {
                dv_p
            };
            let e = TAPP_execute_product(plan, exec, std::ptr::null_mut(), al, a_, b_, be, c_, d_);
            assert_eq!(e, TAPP_ERROR_NULL, "null {name} was accepted");
        }
        assert_eq!(TAPP_destroy_tensor_product(plan), TAPP_SUCCESS);
        assert_eq!(TAPP_destroy_executor(exec), TAPP_SUCCESS);
        assert_eq!(TAPP_destroy_handle(handle), TAPP_SUCCESS);
    }
}

/// A zero executor handle is the default serial executor. This is a tcontract
/// policy (the pinned headers do not say what a zero executor means), pinned
/// because a C caller can rely on it.
#[test]
fn a_zero_executor_handle_is_accepted() {
    let (a, b, d) = good();
    let av = seq::<C64>(a.storage(), 71);
    let bv = seq::<C64>(b.storage(), 72);
    let alpha = C64::new(1.0, 0.0);
    let beta = C64::new(0.0, 0.0);
    let mut dv = vec![C64::new(0.0, 0.0); d.storage()];
    unsafe {
        let handle = create_handle();
        let (e, plan) = Case::plain(a, &av, b, &bv, d).create_plan(handle);
        assert_eq!(e, TAPP_SUCCESS);
        let e = TAPP_execute_product(
            plan,
            0,
            std::ptr::null_mut(),
            &alpha as *const C64 as *const c_void,
            av.as_ptr() as *const c_void,
            bv.as_ptr() as *const c_void,
            &beta as *const C64 as *const c_void,
            std::ptr::null(),
            dv.as_mut_ptr() as *mut c_void,
        );
        assert_eq!(e, TAPP_SUCCESS, "{}", explain_status(e));
        assert_eq!(TAPP_destroy_tensor_product(plan), TAPP_SUCCESS);
        assert_eq!(TAPP_destroy_handle(handle), TAPP_SUCCESS);
    }
}

/// A zero library `handle` is rejected by `TAPP_create_tensor_product`: the
/// caller must supply a live handle, even though the provider keeps no state in
/// it yet.
#[test]
fn a_zero_library_handle_is_rejected_by_create_tensor_product() {
    let (a, b, d) = good();
    unsafe {
        let (e, plan) = Case::plain(
            a,
            &seq::<C64>(a.storage(), 1),
            b,
            &seq::<C64>(b.storage(), 2),
            d,
        )
        .create_plan(0);
        assert_eq!(e, TAPP_ERROR_NULL, "{}", explain_status(e));
        assert_eq!(plan, 0);
    }
}

// -------------------------------------------------------------------- batching

#[test]
fn a_negative_batch_count_is_rejected() {
    let (a, b, d) = good();
    let av = seq::<C64>(a.storage(), 81);
    let bv = seq::<C64>(b.storage(), 82);
    let alpha = C64::new(1.0, 0.0);
    let beta = C64::new(0.0, 0.0);
    let mut dv = vec![C64::new(0.0, 0.0); d.storage()];
    let ap = [av.as_ptr() as *const c_void];
    let bp = [bv.as_ptr() as *const c_void];
    let mut dp = [dv.as_mut_ptr() as *mut c_void];
    unsafe {
        let handle = create_handle();
        let (e, plan) = Case::plain(a, &av, b, &bv, d).create_plan(handle);
        assert_eq!(e, TAPP_SUCCESS);
        let e = TAPP_execute_batched_product(
            plan,
            0,
            std::ptr::null_mut(),
            -1,
            &alpha as *const C64 as *const c_void,
            ap.as_ptr(),
            bp.as_ptr(),
            &beta as *const C64 as *const c_void,
            std::ptr::null(),
            dp.as_mut_ptr(),
        );
        assert_eq!(e, TAPP_ERROR_SHAPE, "{}", explain_status(e));
        // Zero batches is a legal no-op.
        let e = TAPP_execute_batched_product(
            plan,
            0,
            std::ptr::null_mut(),
            0,
            &alpha as *const C64 as *const c_void,
            ap.as_ptr(),
            bp.as_ptr(),
            &beta as *const C64 as *const c_void,
            std::ptr::null(),
            dp.as_mut_ptr(),
        );
        assert_eq!(e, TAPP_SUCCESS, "{}", explain_status(e));
        assert_eq!(TAPP_destroy_tensor_product(plan), TAPP_SUCCESS);
        assert_eq!(TAPP_destroy_handle(handle), TAPP_SUCCESS);
    }
}

#[test]
fn batched_execute_rejects_null_pointer_arrays() {
    let (a, b, d) = good();
    let av = seq::<C64>(a.storage(), 91);
    let bv = seq::<C64>(b.storage(), 92);
    let alpha = C64::new(1.0, 0.0);
    let beta = C64::new(0.0, 0.0);
    let mut dv = vec![C64::new(0.0, 0.0); d.storage()];
    let ap = [av.as_ptr() as *const c_void];
    let bp = [bv.as_ptr() as *const c_void];
    let mut dp = [dv.as_mut_ptr() as *mut c_void];
    unsafe {
        let handle = create_handle();
        let (e, plan) = Case::plain(a, &av, b, &bv, d).create_plan(handle);
        assert_eq!(e, TAPP_SUCCESS);
        let al = &alpha as *const C64 as *const c_void;
        let be = &beta as *const C64 as *const c_void;
        let nul_in: *const *const c_void = std::ptr::null();
        let nul_out: *mut *mut c_void = std::ptr::null_mut();
        for (name, a_, b_, d_) in [
            ("A", nul_in, bp.as_ptr(), dp.as_mut_ptr()),
            ("B", ap.as_ptr(), nul_in, dp.as_mut_ptr()),
            ("D", ap.as_ptr(), bp.as_ptr(), nul_out),
        ] {
            let e = TAPP_execute_batched_product(
                plan,
                0,
                std::ptr::null_mut(),
                1,
                al,
                a_,
                b_,
                be,
                std::ptr::null(),
                d_,
            );
            assert_eq!(e, TAPP_ERROR_NULL, "null {name} array was accepted");
        }
        assert_eq!(TAPP_destroy_tensor_product(plan), TAPP_SUCCESS);
        assert_eq!(TAPP_destroy_handle(handle), TAPP_SUCCESS);
    }
}

// ------------------------------------------------------------ handle lifecycle

/// Create and destroy one of each handle type.
#[test]
fn every_handle_type_round_trips() {
    unsafe {
        let handle = create_handle();
        assert_eq!(TAPP_destroy_handle(handle), TAPP_SUCCESS);

        let exec = create_executor();
        assert_eq!(TAPP_destroy_executor(exec), TAPP_SUCCESS);

        let (a, _, _) = good();
        let info = a.create_info(TAPP_F32);
        assert_eq!(TAPP_destroy_tensor_info(info), TAPP_SUCCESS);

        let (a, b, d) = good();
        let av = seq::<f32>(a.storage(), 101);
        let bv = seq::<f32>(b.storage(), 102);
        let h = create_handle();
        let (e, plan) = Case::plain(a, &av, b, &bv, d).create_plan(h);
        assert_eq!(e, TAPP_SUCCESS);
        assert_ne!(plan, 0);
        assert_eq!(TAPP_destroy_tensor_product(plan), TAPP_SUCCESS);
        assert_eq!(TAPP_destroy_handle(h), TAPP_SUCCESS);
    }
}

/// Handles are independent allocations, so destruction order is free: infos
/// before the plan, the plan before the library handle, the executor last, or
/// any other permutation.
#[test]
fn handles_may_be_destroyed_in_any_order() {
    let (a, b, d) = good();
    let av = seq::<f64>(a.storage(), 111);
    let bv = seq::<f64>(b.storage(), 112);
    unsafe {
        let handle = create_handle();
        let exec = create_executor();
        let ia = a.create_info(TAPP_F64);
        let ib = b.create_info(TAPP_F64);
        let id = d.create_info(TAPP_F64);
        let mut plan = 0isize;
        assert_eq!(
            TAPP_create_tensor_product(
                &mut plan,
                handle,
                TAPP_IDENTITY,
                ia,
                a.labels_ptr(),
                TAPP_IDENTITY,
                ib,
                b.labels_ptr(),
                TAPP_IDENTITY,
                id,
                d.labels_ptr(),
                TAPP_IDENTITY,
                id,
                d.labels_ptr(),
                TAPP_DEFAULT_PREC,
            ),
            TAPP_SUCCESS
        );

        // Library handle first, then an info, then the plan, then the rest —
        // the reverse of a stack discipline, and the plan still executes.
        assert_eq!(TAPP_destroy_handle(handle), TAPP_SUCCESS);
        assert_eq!(TAPP_destroy_tensor_info(ib), TAPP_SUCCESS);

        let alpha = 1.0f64;
        let beta = 0.0f64;
        let mut dv = vec![0.0f64; d.storage()];
        let e = TAPP_execute_product(
            plan,
            exec,
            std::ptr::null_mut(),
            &alpha as *const f64 as *const c_void,
            av.as_ptr() as *const c_void,
            bv.as_ptr() as *const c_void,
            &beta as *const f64 as *const c_void,
            std::ptr::null(),
            dv.as_mut_ptr() as *mut c_void,
        );
        assert_eq!(e, TAPP_SUCCESS, "{}", explain_status(e));

        assert_eq!(TAPP_destroy_tensor_product(plan), TAPP_SUCCESS);
        assert_eq!(TAPP_destroy_tensor_info(id), TAPP_SUCCESS);
        assert_eq!(TAPP_destroy_executor(exec), TAPP_SUCCESS);
        assert_eq!(TAPP_destroy_tensor_info(ia), TAPP_SUCCESS);
    }
}

/// Destroying a zero handle is an error rather than a crash (except the default
/// executor, whose destruction is a no-op). That is the *only*
/// invalid handle value the ABI can detect: `TAPP_*` handles are `intptr_t`
/// casts of `Box::into_raw`, so a non-zero value that never came from a
/// `TAPP_create_*` call — including one that has already been destroyed — is
/// dereferenced and freed unconditionally. Exercising that would be undefined
/// behaviour, so this suite does not, and the crate's safety contract says as
/// much.
#[test]
fn destroying_a_zero_handle_reports_an_error() {
    unsafe {
        assert_eq!(TAPP_destroy_handle(0), TAPP_ERROR_NULL);
        // The default executor is not a resource: destroying it is a no-op.
        assert_eq!(TAPP_destroy_executor(0), TAPP_SUCCESS);
        assert_eq!(TAPP_destroy_tensor_info(0), TAPP_ERROR_NULL);
        assert_eq!(TAPP_destroy_tensor_product(0), TAPP_ERROR_NULL);
    }
}

/// `TAPP_destroy_status` is total: this implementation never creates a status
/// object, so every value — including one a C caller left uninitialised — must be
/// safe to pass.
#[test]
fn destroy_status_accepts_any_value() {
    unsafe {
        for s in [0isize, 1, -1, isize::MAX, isize::MIN, 0x1234_5678] {
            assert_eq!(TAPP_destroy_status(s), TAPP_SUCCESS);
        }
    }
}

/// `TAPP_execute_product` writes `0` through a non-null `status`.
///
/// This test used to assert the opposite, and asserting it is what got the
/// behaviour changed. Upstream's reference implementation leaves `status`
/// untouched and `status.h` declares no `TAPP_create_status`, so a caller writing
/// the idiomatic `TAPP_status s; execute(.., &s, ..); TAPP_destroy_status(s);`
/// was handing an uninitialised value to the destructor — harmless only because
/// the destructor ignores every value. Writing zero costs a branch and makes the
/// sequence defined.
#[test]
fn execute_writes_zero_through_a_non_null_status() {
    let (a, b, d) = good();
    let av = seq::<f64>(a.storage(), 121);
    let bv = seq::<f64>(b.storage(), 122);
    let alpha = 1.0f64;
    let beta = 0.0f64;
    let mut dv = vec![0.0f64; d.storage()];
    const SENTINEL: isize = 0x5EED_5EED;
    unsafe {
        let handle = create_handle();
        let (e, plan) = Case::plain(a, &av, b, &bv, d).create_plan(handle);
        assert_eq!(e, TAPP_SUCCESS);
        let mut status = SENTINEL;
        let e = TAPP_execute_product(
            plan,
            0,
            &mut status,
            &alpha as *const f64 as *const c_void,
            av.as_ptr() as *const c_void,
            bv.as_ptr() as *const c_void,
            &beta as *const f64 as *const c_void,
            std::ptr::null(),
            dv.as_mut_ptr() as *mut c_void,
        );
        assert_eq!(e, TAPP_SUCCESS);
        assert_eq!(
            status, 0,
            "status should be zeroed on success; it read back as the sentinel"
        );
        assert_eq!(TAPP_destroy_tensor_product(plan), TAPP_SUCCESS);
        assert_eq!(TAPP_destroy_handle(handle), TAPP_SUCCESS);
    }
}

// ------------------------------------------------------------- tensor info API

/// `nmode`, extents and strides survive a get/set round trip, including growing
/// and shrinking the rank.
#[test]
fn nmodes_extents_and_strides_round_trip() {
    unsafe {
        let e0 = [4i64, 3, 2];
        let s0 = [1i64, 4, 12];
        let mut info = 0isize;
        assert_eq!(
            TAPP_create_tensor_info(&mut info, TAPP_C32, 3, e0.as_ptr(), s0.as_ptr()),
            TAPP_SUCCESS
        );
        assert_eq!(TAPP_get_nmodes(info), 3);

        let mut e = [0i64; 3];
        let mut s = [0i64; 3];
        TAPP_get_extents(info, e.as_mut_ptr());
        TAPP_get_strides(info, s.as_mut_ptr());
        assert_eq!(e, e0);
        assert_eq!(s, s0);

        // Overwrite both.
        let e1 = [2i64, 5, 7];
        let s1 = [35i64, 7, 1];
        assert_eq!(TAPP_set_extents(info, e1.as_ptr()), TAPP_SUCCESS);
        assert_eq!(TAPP_set_strides(info, s1.as_ptr()), TAPP_SUCCESS);
        TAPP_get_extents(info, e.as_mut_ptr());
        TAPP_get_strides(info, s.as_mut_ptr());
        assert_eq!(e, e1);
        assert_eq!(s, s1);

        // Shrink, then grow. Growing pads with extent 1 and stride 0, so the
        // added modes are inert; shrinking drops the trailing modes.
        assert_eq!(TAPP_set_nmodes(info, 1), TAPP_SUCCESS);
        assert_eq!(TAPP_get_nmodes(info), 1);
        let mut one = [0i64; 1];
        TAPP_get_extents(info, one.as_mut_ptr());
        assert_eq!(one, [2]);
        assert_eq!(TAPP_set_nmodes(info, 3), TAPP_SUCCESS);
        TAPP_get_extents(info, e.as_mut_ptr());
        TAPP_get_strides(info, s.as_mut_ptr());
        assert_eq!(e, [2, 1, 1]);
        assert_eq!(s, [35, 0, 0]);

        assert_eq!(TAPP_set_nmodes(info, 0), TAPP_SUCCESS);
        assert_eq!(TAPP_get_nmodes(info), 0);
        assert_eq!(TAPP_destroy_tensor_info(info), TAPP_SUCCESS);
    }
}

/// The two `void`-returning getters tolerate a null destination, and the
/// setters refuse one.
#[test]
fn getters_tolerate_null_and_setters_refuse_it() {
    unsafe {
        let (a, _, _) = good();
        let info = a.create_info(TAPP_F64);
        TAPP_get_extents(info, std::ptr::null_mut());
        TAPP_get_strides(info, std::ptr::null_mut());
        assert_eq!(TAPP_set_extents(info, std::ptr::null()), TAPP_ERROR_NULL);
        assert_eq!(TAPP_set_strides(info, std::ptr::null()), TAPP_ERROR_NULL);
        assert_eq!(TAPP_destroy_tensor_info(info), TAPP_SUCCESS);
        // A zero handle is inert in the getters and an error in the setters.
        TAPP_get_extents(0, std::ptr::null_mut());
        TAPP_get_strides(0, std::ptr::null_mut());
        assert_eq!(TAPP_set_extents(0, std::ptr::null()), TAPP_ERROR_NULL);
        assert_eq!(TAPP_set_strides(0, std::ptr::null()), TAPP_ERROR_NULL);
        assert_eq!(TAPP_set_nmodes(0, 2), TAPP_ERROR_NULL);
    }
}

/// `TAPP_get_nmodes` returns `int`, so it signals an invalid handle with `-1`
/// rather than a `TAPP_error`.
#[test]
fn get_nmodes_on_a_zero_handle_returns_minus_one() {
    assert_eq!(unsafe { TAPP_get_nmodes(0) }, -1);
}

// -------------------------------------------------------------- error reporting

#[test]
fn check_success_accepts_only_zero() {
    assert!(TAPP_check_success(TAPP_SUCCESS));
    for e in [
        TAPP_ERROR_NULL,
        TAPP_ERROR_DATATYPE,
        TAPP_ERROR_SHAPE,
        TAPP_ERROR_LABELS,
        TAPP_ERROR_UNSUPPORTED,
        TAPP_ERROR_INTERNAL,
        -1,
        c_int::MAX,
    ] {
        assert!(!TAPP_check_success(e), "{e} passed as success");
    }
}

/// `TAPP_explain_error` writes at most `maxlen - 1` characters plus a NUL, and
/// returns the length it would have written — upstream `error.h`'s contract.
#[test]
fn explain_error_truncates_and_nul_terminates() {
    let codes = [
        TAPP_SUCCESS,
        TAPP_ERROR_NULL,
        TAPP_ERROR_DATATYPE,
        TAPP_ERROR_SHAPE,
        TAPP_ERROR_LABELS,
        TAPP_ERROR_UNSUPPORTED,
        TAPP_ERROR_INTERNAL,
        // An unknown code must still produce something printable.
        4242,
    ];
    for code in codes {
        let mut full: [c_char; 256] = [0x7f; 256];
        let n = unsafe { TAPP_explain_error(code, full.len(), full.as_mut_ptr()) };
        assert!(n > 0, "no message for {code}");
        let s = unsafe { std::ffi::CStr::from_ptr(full.as_ptr()) };
        assert_eq!(s.to_bytes().len(), n, "message for {code} was truncated");
        assert!(s.to_str().is_ok(), "message for {code} is not UTF-8");

        // Truncation: a buffer of 5 gets 4 characters and a NUL, and the
        // returned length is still the full one.
        let mut small: [c_char; 5] = [0x7f; 5];
        let m = unsafe { TAPP_explain_error(code, small.len(), small.as_mut_ptr()) };
        assert_eq!(m, n, "the return value must not depend on maxlen");
        let t = unsafe { std::ffi::CStr::from_ptr(small.as_ptr()) };
        assert!(t.to_bytes().len() <= 4);
        assert_eq!(t.to_bytes(), &s.to_bytes()[..t.to_bytes().len()]);
    }
}

/// `maxlen == 0` and a null buffer are both the documented "just tell me the
/// length" call, and must write nothing.
#[test]
fn explain_error_tolerates_a_null_buffer_and_zero_maxlen() {
    unsafe {
        let n = TAPP_explain_error(TAPP_ERROR_SHAPE, 0, std::ptr::null_mut());
        assert!(n > 0);
        let m = TAPP_explain_error(TAPP_ERROR_SHAPE, 64, std::ptr::null_mut());
        assert_eq!(m, n);
        // A one-byte buffer can hold only the NUL.
        let mut one: [c_char; 1] = [0x7f];
        let k = TAPP_explain_error(TAPP_ERROR_SHAPE, 1, one.as_mut_ptr());
        assert_eq!(k, n);
        assert_eq!(one[0], 0);
    }
}

#[test]
fn implementation_name_is_a_nul_terminated_string() {
    let p = TAPP_implementation_name();
    assert!(!p.is_null());
    let s = unsafe { std::ffi::CStr::from_ptr(p) };
    let s = s.to_str().expect("not UTF-8");
    assert!(!s.is_empty());
    // Names the *engine* doing the work, not just the family: a C caller
    // choosing between TAPP providers wants to know which implementation it got,
    // and the family will eventually contain more than one.
    assert!(
        s.contains("tcontract"),
        "the name should identify the engine: {s:?}"
    );
    assert!(
        s.contains("tcontract"),
        "the name should identify the project: {s:?}"
    );
    // Static storage: two calls give the same pointer, so a C caller may keep it.
    assert_eq!(TAPP_implementation_name(), p);
}

// ------------------------------------------------------------------ known gaps

/// An extent product that overflows `i64` is not guarded anywhere between
/// `TAPP_create_tensor_info` and the scatter construction inside `Plan::new`.
///
/// `build_scatter` computes `extents.iter().product()` in `i64` and then
/// `Vec::with_capacity` of that many entries. With the extents below the product
/// is `2^64`, which wraps to `0` in a release build — the plan is created
/// successfully and executes as a silent no-op — and **panics** in a debug build,
/// unwinding out of `TAPP_create_tensor_product` across the `extern "C"`
/// boundary, which is undefined behaviour. Extents whose product is large but
/// representable are worse still: they reach a multi-terabyte allocation and
/// abort.
///
/// **Fixed**, and this test is what holds it fixed. `reduce_tensor` now folds
/// each tensor's extents with `checked_mul` and reports
/// `Error::ExtentProductOverflow`, which `map_err` turns into
/// `TAPP_ERROR_SHAPE`. Checking per tensor is enough for every scatter vector,
/// because each one is as long as the product of some *subset* of one tensor's
/// axes. Note the panic was an abort rather than undefined behaviour — a panic
/// crossing `extern "C"` is turned into a process abort by the compiler — which
/// is no more shippable for being defined.
#[test]
fn an_overflowing_extent_product_is_a_shape_error() {
    const BIG: i64 = 1 << 32;
    let a = CTensor::new(&[BIG, BIG, 3], &[1, BIG, 0], &[I, X, K]);
    let b = CTensor::new(&[3, 5], &[1, 3], &[K, J]);
    let d = CTensor::new(&[BIG, BIG, 5], &[1, BIG, 0], &[I, X, J]);
    let av = seq::<f64>(8, 1);
    let bv = seq::<f64>(b.storage(), 2);
    let case = Case::plain(a, &av, b, &bv, d);
    let mut dv = vec![0.0f64; 8];
    let e = unsafe { case.run(&mut dv) };
    assert_eq!(e, TAPP_ERROR_SHAPE, "{}", explain_status(e));
}
