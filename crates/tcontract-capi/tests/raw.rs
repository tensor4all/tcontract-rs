//! A tiny f64 front end over the raw C entry points, for the tests that are
//! about executors and aliasing rather than about numerics.
#![allow(
    dead_code,
    non_snake_case,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

use std::ffi::c_void;
use std::os::raw::c_int;

use tcontract_capi::*;

/// `(extents, strides, labels)` of one operand.
pub type Op<'a> = (&'a [i64], &'a [i64], &'a [i64]);

pub const I: i64 = 'i' as i64;
pub const J: i64 = 'j' as i64;
pub const K: i64 = 'k' as i64;

/// Column-major strides for `extents`.
pub fn cm(extents: &[i64]) -> Vec<i64> {
    let mut s = 1;
    extents
        .iter()
        .map(|&e| {
            let r = s;
            s *= e.max(1);
            r
        })
        .collect()
}

pub unsafe fn info(dtype: c_int, e: &[i64], s: &[i64]) -> isize {
    let mut i = 0isize;
    let rc = TAPP_create_tensor_info(
        &mut i,
        dtype,
        e.len() as c_int,
        if e.is_empty() {
            std::ptr::null()
        } else {
            e.as_ptr()
        },
        if s.is_empty() {
            std::ptr::null()
        } else {
            s.as_ptr()
        },
    );
    assert_eq!(rc, TAPP_SUCCESS);
    i
}

/// Plan `D = op_D(alpha op_A(A) op_B(B) + beta op_C(C))`; returns the code and
/// the plan (0 on failure). The infos and the library handle are destroyed
/// at once: a plan does not borrow them.
pub unsafe fn plan_with(
    dtype: c_int,
    ops: [c_int; 4],
    a: Op,
    b: Op,
    c: Op,
    d: Op,
    prec: c_int,
) -> (c_int, isize) {
    let mut h = 0isize;
    assert_eq!(TAPP_create_handle(&mut h), TAPP_SUCCESS);
    let ids = [a, b, c, d].map(|o| info(dtype, o.0, o.1));
    let mut plan = 0isize;
    let rc = TAPP_create_tensor_product(
        &mut plan,
        h,
        ops[0],
        ids[0],
        a.2.as_ptr(),
        ops[1],
        ids[1],
        b.2.as_ptr(),
        ops[2],
        ids[2],
        c.2.as_ptr(),
        ops[3],
        ids[3],
        d.2.as_ptr(),
        prec,
    );
    for i in ids {
        assert_eq!(TAPP_destroy_tensor_info(i), TAPP_SUCCESS);
    }
    assert_eq!(TAPP_destroy_handle(h), TAPP_SUCCESS);
    (rc, plan)
}

pub unsafe fn plan_f64(a: Op, b: Op, c: Op, d: Op) -> (c_int, isize) {
    plan_with(TAPP_F64, [TAPP_IDENTITY; 4], a, b, c, d, TAPP_DEFAULT_PREC)
}

/// `D = alpha A B + beta C` on f64 data.
pub unsafe fn exec_f64(
    plan: isize,
    exec: isize,
    alpha: f64,
    a: *const f64,
    b: *const f64,
    beta: f64,
    c: *const f64,
    d: *mut f64,
) -> c_int {
    TAPP_execute_product(
        plan,
        exec,
        std::ptr::null_mut(),
        &alpha as *const f64 as *const c_void,
        a as *const c_void,
        b as *const c_void,
        &beta as *const f64 as *const c_void,
        c as *const c_void,
        d as *mut c_void,
    )
}

/// Deterministic values in `[-1, 1)`, exact in binary.
pub fn seq(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 52) as f64) / 2048.0 - 1.0
        })
        .collect()
}

/// `D[i,j] = sum_k A[i,k] B[k,j]` for column-major `m x k` and `k x n`.
pub fn naive(m: usize, n: usize, k: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut d = vec![0.0; m * n];
    for j in 0..n {
        for i in 0..m {
            d[i + m * j] = (0..k).map(|p| a[i + m * p] * b[p + k * j]).sum();
        }
    }
    d
}

pub fn assert_close(got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= tol * (1.0 + w.abs()), "[{i}] {g} vs {w}");
    }
}

pub unsafe fn rayon_exec(n: usize) -> isize {
    let mut e = 0isize;
    assert_eq!(
        tcontract_tapp_executor_create_rayon(&mut e, n, std::ptr::null()),
        TAPP_SUCCESS
    );
    e
}

pub unsafe fn serial_exec() -> isize {
    let mut e = 0isize;
    assert_eq!(TAPP_create_executor(&mut e), TAPP_SUCCESS);
    e
}

/// The pool's entry counters, or `None` for a serial executor.
pub fn pool_stats(exec: isize) -> Option<tcontract_exec::PoolStats> {
    unsafe { tcontract_capi::executor::executor_ref(exec) }.and_then(|e| e.pool_stats())
}
