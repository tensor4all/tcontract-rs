//! Validation: output aliasing, element operations, overflow, and that a batch
//! is checked by the same rules as a single call, before anything is written.
mod raw;

use std::ffi::c_void;
use std::os::raw::c_int;

use raw::*;
use tcontract_capi::*;

const N: usize = 4;

struct Sq {
    e: [i64; 2],
    s: [i64; 2],
    l: [i64; 2],
}

fn sq(labels: [i64; 2]) -> Sq {
    Sq {
        e: [N as i64, N as i64],
        s: [1, N as i64],
        l: labels,
    }
}

impl Sq {
    fn op(&self) -> Op<'_> {
        (&self.e, &self.s, &self.l)
    }
}

/// `D[i,j] = A[i,k] B[k,j]` on 4x4 column-major matrices.
unsafe fn square_plan() -> isize {
    let (a, b, d) = (sq([I, K]), sq([K, J]), sq([I, J]));
    let (rc, plan) = plan_f64(a.op(), b.op(), d.op(), d.op());
    assert_eq!(rc, TAPP_SUCCESS);
    plan
}

#[test]
fn d_overlapping_a_or_b_is_rejected_before_any_write() {
    unsafe {
        let plan = square_plan();
        let a = seq(N * N, 1);
        let b = seq(N * N, 2);
        let mut buf = seq(2 * N * N, 3);
        let keep = buf.clone();
        let p = buf.as_mut_ptr();
        // D == A, D == B, and a partial overlap with each.
        for (ap, bp, dp) in [
            (p as *const f64, b.as_ptr(), p),
            (a.as_ptr(), p as *const f64, p),
            (p as *const f64, b.as_ptr(), p.add(N)),
            (a.as_ptr(), p.add(N * N - 2) as *const f64, p),
        ] {
            let rc = exec_f64(plan, 0, 1.0, ap, bp, 0.0, std::ptr::null(), dp);
            assert_eq!(rc, TAPP_ERROR_ALIASED);
        }
        assert_eq!(buf, keep, "a rejected call must not write");
        // Disjoint halves of one allocation are fine.
        let rc = exec_f64(
            plan,
            0,
            1.0,
            a.as_ptr(),
            b.as_ptr(),
            0.0,
            std::ptr::null(),
            p.add(N * N),
        );
        assert_eq!(rc, TAPP_SUCCESS);
        assert_close(&buf[N * N..], &naive(N, N, N, &a, &b), 1e-13);
        TAPP_destroy_tensor_product(plan);
    }
}

#[test]
fn in_place_needs_the_same_mapping_and_partial_overlap_is_rejected() {
    unsafe {
        let plan = square_plan();
        let a = seq(N * N, 4);
        let b = seq(N * N, 5);
        let ab = naive(N, N, N, &a, &b);

        // C == D with equal mappings: D = A B + 2 D, in place.
        let mut d = seq(N * N, 6);
        let d0 = d.clone();
        let rc = exec_f64(
            plan,
            0,
            1.0,
            a.as_ptr(),
            b.as_ptr(),
            2.0,
            d.as_ptr(),
            d.as_mut_ptr(),
        );
        assert_eq!(rc, TAPP_SUCCESS);
        let want: Vec<f64> = ab.iter().zip(&d0).map(|(x, y)| x + 2.0 * y).collect();
        assert_close(&d, &want, 1e-13);

        // Separate C and D buffers, C of a different layout (transposed): kept.
        let (ad, bd, dd) = (sq([I, K]), sq([K, J]), sq([I, J]));
        let ct = Sq {
            e: [N as i64, N as i64],
            s: [N as i64, 1],
            l: [I, J],
        };
        let (rc, tplan) = plan_f64(ad.op(), bd.op(), ct.op(), dd.op());
        assert_eq!(rc, TAPP_SUCCESS);
        let c = seq(N * N, 7);
        let mut d = vec![f64::NAN; N * N];
        let rc = exec_f64(
            tplan,
            0,
            1.0,
            a.as_ptr(),
            b.as_ptr(),
            3.0,
            c.as_ptr(),
            d.as_mut_ptr(),
        );
        assert_eq!(rc, TAPP_SUCCESS);
        for j in 0..N {
            for i in 0..N {
                let w = ab[i + N * j] + 3.0 * c[i * N + j];
                assert!((d[i + N * j] - w).abs() < 1e-13);
            }
        }

        // The same differing mapping with C == D (equal base pointers): rejected.
        let mut buf = seq(N * N, 8);
        let keep = buf.clone();
        let bp = buf.as_mut_ptr();
        let rc = exec_f64(tplan, 0, 1.0, a.as_ptr(), b.as_ptr(), 1.0, bp, bp);
        assert_eq!(rc, TAPP_ERROR_ALIASED);
        assert_eq!(buf, keep);

        // Partial overlap of C and D (shifted views of one buffer): rejected.
        let mut buf = seq(2 * N * N, 9);
        let keep = buf.clone();
        let bp = buf.as_mut_ptr();
        let rc = exec_f64(plan, 0, 1.0, a.as_ptr(), b.as_ptr(), 1.0, bp, bp.add(3));
        assert_eq!(rc, TAPP_ERROR_ALIASED);
        assert_eq!(buf, keep);
        TAPP_destroy_tensor_product(plan);
        TAPP_destroy_tensor_product(tplan);
    }
}

#[test]
fn an_output_that_overlaps_itself_is_rejected_at_planning() {
    unsafe {
        let (a, b) = (sq([I, K]), sq([K, J]));
        // Zero stride on j: every column of D is the same memory.
        let d = Sq {
            e: [N as i64, N as i64],
            s: [1, 0],
            l: [I, J],
        };
        let (rc, plan) = plan_f64(a.op(), b.op(), d.op(), d.op());
        assert_eq!(rc, TAPP_ERROR_ALIASED);
        assert_eq!(plan, 0);
        // Strides smaller than the extent they must clear.
        let d = Sq {
            e: [N as i64, N as i64],
            s: [1, 2],
            l: [I, J],
        };
        let (rc, _) = plan_f64(a.op(), b.op(), d.op(), d.op());
        assert_eq!(rc, TAPP_ERROR_ALIASED);
        // A repeated output label addresses the diagonal: stride 1 + N.
        let ed = [N as i64, N as i64];
        let (la, lb) = ([I, K], [K, I]);
        let (rc, plan) = plan_f64(
            (&ed, &[1, N as i64], &la),
            (&ed, &[1, N as i64], &lb),
            (&[N as i64], &[(N + 1) as i64], &[I]),
            (&[N as i64], &[(N + 1) as i64], &[I]),
        );
        assert_eq!(rc, TAPP_SUCCESS, "a diagonal output is legal");
        TAPP_destroy_tensor_product(plan);
    }
}

#[test]
fn shape_and_stride_overflow_is_rejected() {
    unsafe {
        let (a, b, d) = (sq([I, K]), sq([K, J]), sq([I, J]));
        let big = [i64::MAX / 2, 3];
        let huge = Sq {
            e: [N as i64, 3],
            s: [1, i64::MAX / 2],
            l: [I, J],
        };
        let (rc, _) = plan_f64(a.op(), b.op(), huge.op(), huge.op());
        assert_eq!(
            rc, TAPP_ERROR_SHAPE,
            "stride * extent overflows the address space"
        );
        let wide = Sq {
            e: [big[0], 3],
            s: [1, 1],
            l: [I, J],
        };
        let (rc, _) = plan_f64(wide.op(), b.op(), d.op(), d.op());
        assert_ne!(
            rc, TAPP_SUCCESS,
            "an enormous extent is refused, not planned"
        );
        let neg = Sq {
            e: [N as i64, 3],
            s: [i64::MIN, 1],
            l: [I, J],
        };
        let (rc, _) = plan_f64(neg.op(), b.op(), d.op(), d.op());
        assert_ne!(rc, TAPP_SUCCESS);
    }
}

#[test]
fn unsupported_element_operations_and_dtypes_are_rejected() {
    unsafe {
        let (a, b, d) = (sq([I, K]), sq([K, J]), sq([I, J]));
        for ops in [[2, 0, 0, 0], [0, -1, 0, 0], [0, 0, 7, 0], [0, 0, 0, 99]] {
            let (rc, plan) = plan_with(
                TAPP_F64,
                ops,
                a.op(),
                b.op(),
                d.op(),
                d.op(),
                TAPP_DEFAULT_PREC,
            );
            assert_eq!(rc, TAPP_ERROR_UNSUPPORTED, "{ops:?}");
            assert_eq!(plan, 0);
        }
        // Conjugation is accepted on every operand, real storage included.
        let (rc, plan) = plan_with(
            TAPP_F64,
            [1; 4],
            a.op(),
            b.op(),
            d.op(),
            d.op(),
            TAPP_DEFAULT_PREC,
        );
        assert_eq!(rc, TAPP_SUCCESS);
        TAPP_destroy_tensor_product(plan);
        let mut i = 0isize;
        let rc = TAPP_create_tensor_info(&mut i, TAPP_F16, 2, a.e.as_ptr(), a.s.as_ptr());
        assert_eq!(rc, TAPP_ERROR_DATATYPE);
    }
}

#[test]
fn a_batch_is_validated_like_single_calls_before_anything_is_written() {
    unsafe {
        let plan = square_plan();
        let a = seq(N * N, 11);
        let b = seq(N * N, 12);
        let mut d0 = vec![f64::NAN; N * N];
        let mut d1 = vec![f64::NAN; N * N];
        let (alpha, beta) = (1.0f64, 0.0f64);
        let call = |ap: &[*const c_void], bp: &[*const c_void], dp: &mut [*mut c_void]| {
            TAPP_execute_batched_product(
                plan,
                0,
                std::ptr::null_mut(),
                ap.len() as c_int,
                &alpha as *const f64 as *const c_void,
                ap.as_ptr(),
                bp.as_ptr(),
                &beta as *const f64 as *const c_void,
                std::ptr::null(),
                dp.as_mut_ptr(),
            )
        };
        let (pa, pb) = (a.as_ptr() as *const c_void, b.as_ptr() as *const c_void);
        // Item 1 aliases its A: the whole batch is refused and item 0 is untouched.
        let mut dp = [
            d0.as_mut_ptr() as *mut c_void,
            d1.as_mut_ptr() as *mut c_void,
        ];
        let rc = call(&[pa, d1.as_ptr() as *const c_void], &[pb, pb], &mut dp);
        assert_eq!(rc, TAPP_ERROR_ALIASED);
        assert!(
            d0.iter().all(|x| x.is_nan()),
            "item 0 was written before item 1 was validated"
        );
        // The same item through a single call gives the same code.
        let rc = exec_f64(
            plan,
            0,
            1.0,
            d1.as_ptr(),
            b.as_ptr(),
            0.0,
            std::ptr::null(),
            d1.as_mut_ptr(),
        );
        assert_eq!(rc, TAPP_ERROR_ALIASED);
        // A null pointer in an item is refused up front too.
        let rc = call(&[pa, std::ptr::null()], &[pb, pb], &mut dp);
        assert_eq!(rc, TAPP_ERROR_NULL);
        assert!(d0.iter().all(|x| x.is_nan()));
        // A good batch runs every item; a null C array with beta == 0 is fine.
        let rc = call(&[pa, pa], &[pb, pb], &mut dp);
        assert_eq!(rc, TAPP_SUCCESS);
        let want = naive(N, N, N, &a, &b);
        assert_close(&d0, &want, 1e-13);
        assert_close(&d1, &want, 1e-13);
        // A null C with a nonzero beta is unsupported, for single and batch.
        let beta1 = 1.0f64;
        let rc = TAPP_execute_batched_product(
            plan,
            0,
            std::ptr::null_mut(),
            2,
            &alpha as *const f64 as *const c_void,
            [pa, pa].as_ptr(),
            [pb, pb].as_ptr(),
            &beta1 as *const f64 as *const c_void,
            std::ptr::null(),
            dp.as_mut_ptr(),
        );
        assert_eq!(rc, TAPP_ERROR_UNSUPPORTED);
        let rc = exec_f64(
            plan,
            0,
            1.0,
            a.as_ptr(),
            b.as_ptr(),
            1.0,
            std::ptr::null(),
            d0.as_mut_ptr(),
        );
        assert_eq!(rc, TAPP_ERROR_UNSUPPORTED);
        TAPP_destroy_tensor_product(plan);
    }
}

#[test]
fn output_label_order_is_free() {
    unsafe {
        // D[j,i] = sum_k A[i,k] B[k,j]: the output is the transpose.
        let (a, b) = (sq([I, K]), sq([K, J]));
        let d = sq([J, I]);
        let (rc, plan) = plan_f64(a.op(), b.op(), d.op(), d.op());
        assert_eq!(rc, TAPP_SUCCESS);
        let av = seq(N * N, 21);
        let bv = seq(N * N, 22);
        let mut dv = vec![f64::NAN; N * N];
        assert_eq!(
            exec_f64(
                plan,
                0,
                1.0,
                av.as_ptr(),
                bv.as_ptr(),
                0.0,
                std::ptr::null(),
                dv.as_mut_ptr()
            ),
            TAPP_SUCCESS
        );
        let ab = naive(N, N, N, &av, &bv);
        for i in 0..N {
            for j in 0..N {
                assert!((dv[j + N * i] - ab[i + N * j]).abs() < 1e-13);
            }
        }
        TAPP_destroy_tensor_product(plan);
    }
}

#[test]
fn op_d_conjugates_the_whole_result_including_beta_c() {
    use num_complex::Complex64 as C;
    unsafe {
        let one = [1i64];
        let s0 = [1i64];
        let l = [I];
        let ek = [1i64, 1];
        let _ = ek;
        // 1-element contraction over k: A[i,k] B[k] with i = k = 1 element each.
        let ea = [1i64, 1];
        let sa = [1i64, 1];
        let la = [I, K];
        let lb = [K];
        let mut h = 0isize;
        assert_eq!(TAPP_create_handle(&mut h), TAPP_SUCCESS);
        let ia = info(TAPP_C64, &ea, &sa);
        let ib = info(TAPP_C64, &one, &s0);
        let id = info(TAPP_C64, &one, &s0);
        let mut plan = 0isize;
        let rc = TAPP_create_tensor_product(
            &mut plan,
            h,
            TAPP_IDENTITY,
            ia,
            la.as_ptr(),
            TAPP_IDENTITY,
            ib,
            lb.as_ptr(),
            TAPP_IDENTITY,
            id,
            l.as_ptr(),
            TAPP_CONJUGATE,
            id,
            l.as_ptr(),
            TAPP_DEFAULT_PREC,
        );
        assert_eq!(rc, TAPP_SUCCESS);
        // D = conj(alpha*A*B + beta*C) with alpha = i, A = 1+i, B = 1, beta = 1+i, C = 2:
        // conj(i(1+i) + 2(1+i)) = conj(1 + 3i) = 1 - 3i
        let (alpha, beta) = (C::new(0.0, 1.0), C::new(1.0, 1.0));
        let (a, b, c) = (C::new(1.0, 1.0), C::new(1.0, 0.0), C::new(2.0, 0.0));
        let mut d = C::new(f64::NAN, f64::NAN);
        let rc = TAPP_execute_product(
            plan,
            0,
            std::ptr::null_mut(),
            &alpha as *const C as *const c_void,
            &a as *const C as *const c_void,
            &b as *const C as *const c_void,
            &beta as *const C as *const c_void,
            &c as *const C as *const c_void,
            &mut d as *mut C as *mut c_void,
        );
        assert_eq!(rc, TAPP_SUCCESS);
        assert_eq!(d, C::new(1.0, -3.0));
        TAPP_destroy_tensor_product(plan);
        for i in [ia, ib, id] {
            TAPP_destroy_tensor_info(i);
        }
        TAPP_destroy_handle(h);
    }
}

#[test]
fn errors_set_a_message_and_none_panics_across_the_abi() {
    unsafe {
        // Adversarial inputs through every creating/executing entry point: each
        // must come back as a status, never unwind or abort.
        let (a, b, d) = (sq([I, K]), sq([K, J]), sq([I, J]));
        let (rc, plan) = plan_f64(a.op(), b.op(), d.op(), d.op());
        assert_eq!(rc, TAPP_SUCCESS);
        assert_ne!(
            exec_f64(
                plan,
                0,
                1.0,
                std::ptr::null(),
                std::ptr::null(),
                0.0,
                std::ptr::null(),
                std::ptr::null_mut()
            ),
            TAPP_SUCCESS
        );
        let msg = std::ffi::CStr::from_ptr(tcontract_capi::status::tcontract_last_error());
        assert!(
            !msg.to_bytes().is_empty(),
            "a failed call records a message"
        );
        assert_ne!(
            TAPP_execute_batched_product(
                plan,
                0,
                std::ptr::null_mut(),
                i32::MAX,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut()
            ),
            TAPP_SUCCESS
        );
        let mut x = 0isize;
        assert_ne!(
            TAPP_create_tensor_info(
                &mut x,
                TAPP_F64,
                i32::MAX,
                std::ptr::null(),
                std::ptr::null()
            ),
            TAPP_SUCCESS
        );
        assert_ne!(
            TAPP_create_tensor_info(&mut x, TAPP_F64, -5, [1i64].as_ptr(), [1i64].as_ptr()),
            TAPP_SUCCESS
        );
        TAPP_destroy_tensor_product(plan);
    }
}
