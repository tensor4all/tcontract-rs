//! TAPP C-ABI conformance: the numerical surface.
//!
//! One test per claim the crate's coverage table makes, driven entirely through
//! the `extern "C"` entry points and checked against
//! `tcontract_testkit::oracle`, which shares no code with the engine. The
//! rejections, the error paths and the handle lifecycle are in
//! `rejections.rs`; the layout and symbol assumptions are in `abi_layout.rs`.
//!
//! Every case runs in all four datatypes the crate claims. Holding the shape
//! fixed across datatypes is the house rule for measurements and it is just as
//! useful here: a dispatch bug that sends `TAPP_C32` to the `f32` kernel shows
//! up as one datatype failing on a shape all four are known to handle.

mod common;

use std::ffi::c_void;

use common::*;
use tcontract_capi::*;

type C32 = num_complex::Complex<f32>;
type C64 = num_complex::Complex<f64>;

/// Run a generic case body in every datatype the crate claims to support.
macro_rules! every_dtype {
    ($f:ident) => {{
        $f::<f32>();
        $f::<f64>();
        $f::<C32>();
        $f::<C64>();
    }};
}

// Index labels are `int64_t` and TAPP puts no constraint on their values, so
// the suite uses `0` (which is also the "absent handle" sentinel elsewhere in
// the ABI, and must not be confused with it) and a negative label.
const I: i64 = 0;
const J: i64 = 1;
const K: i64 = 2;
const H: i64 = 3;
const R: i64 = -7;

// ------------------------------------------------------------------ datatypes

/// `D[i,j] = sum_k A[i,k] B[k,j]` with a column-major `A`, a row-major `B` and
/// a padded `D`, in one datatype.
///
/// The padding in `D` is load-bearing: `assert_close` compares the whole
/// buffer, so a write outside the declared layout fails the test rather than
/// landing in slack the assertion never looks at.
fn mixed_layout_gemm<T: Dtype>() {
    let a = CTensor::new(&[5, 3], &[1, 5], &[I, K]);
    let b = CTensor::new(&[3, 4], &[4, 1], &[K, J]);
    let d = CTensor::new(&[5, 4], &[1, 7], &[I, J]);
    let av = seq::<T>(a.storage(), 1);
    let bv = seq::<T>(b.storage(), 2);
    Case::plain(a, &av, b, &bv, d).check_zeroed();
}

#[test]
fn datatype_f32_contracts_correctly() {
    mixed_layout_gemm::<f32>();
}

#[test]
fn datatype_f64_contracts_correctly() {
    mixed_layout_gemm::<f64>();
}

#[test]
fn datatype_c32_contracts_correctly() {
    mixed_layout_gemm::<C32>();
}

#[test]
fn datatype_c64_contracts_correctly() {
    mixed_layout_gemm::<C64>();
}

/// The one anchor in the suite that does not go through the oracle at all.
///
/// `(1 + 2i)(3 + 4i) = -5 + 10i` pins three things a shared bug between engine
/// and oracle could hide: that `TAPP_C32`/`TAPP_C64` really are interleaved with
/// the real part first, that the sign of the cross term is right, and that the
/// datatype tag reaches the complex path rather than being reinterpreted as two
/// reals.
#[test]
fn a_single_complex_product_is_hand_checkable() {
    fn one<T: Dtype>() {
        let a = CTensor::new(&[1, 1], &[1, 1], &[I, K]);
        let b = CTensor::new(&[1, 1], &[1, 1], &[K, J]);
        let d = CTensor::new(&[1, 1], &[1, 1], &[I, J]);
        let av = vec![scalar::<T>(1.0, 2.0)];
        let bv = vec![scalar::<T>(3.0, 4.0)];
        let mut dv = vec![T::zero()];
        let e = unsafe { Case::plain(a, &av, b, &bv, d).run(&mut dv) };
        assert_eq!(e, TAPP_SUCCESS, "{}", explain_status(e));
        // 3 for the real types, where the imaginary parts were dropped.
        let want = if T::IS_COMPLEX {
            scalar::<T>(-5.0, 10.0)
        } else {
            scalar::<T>(3.0, 0.0)
        };
        assert_exact::<T>(&dv, &[want]);
    }
    every_dtype!(one);
}

/// A 2x2 real product with small integers, exact in binary floating point.
#[test]
fn a_small_real_product_is_hand_checkable() {
    fn one<T: Dtype>() {
        // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]], both column-major.
        // A B = [[19, 22], [43, 50]].
        let a = CTensor::new(&[2, 2], &[1, 2], &[I, K]);
        let b = CTensor::new(&[2, 2], &[1, 2], &[K, J]);
        let d = CTensor::new(&[2, 2], &[1, 2], &[I, J]);
        let av: Vec<T> = [1.0, 3.0, 2.0, 4.0]
            .iter()
            .map(|&v| scalar(v, 0.0))
            .collect();
        let bv: Vec<T> = [5.0, 7.0, 6.0, 8.0]
            .iter()
            .map(|&v| scalar(v, 0.0))
            .collect();
        let want: Vec<T> = [19.0, 43.0, 22.0, 50.0]
            .iter()
            .map(|&v| scalar(v, 0.0))
            .collect();
        let mut dv = vec![T::zero(); 4];
        let e = unsafe { Case::plain(a, &av, b, &bv, d).run(&mut dv) };
        assert_eq!(e, TAPP_SUCCESS, "{}", explain_status(e));
        assert_exact::<T>(&dv, &want);
    }
    every_dtype!(one);
}

// ---------------------------------------------------------------- TAPP cases

/// Case 1: free, contracted and no other index classes.
#[test]
fn case1_simple_contraction() {
    every_dtype!(mixed_layout_gemm);
}

/// Case 1, with several free indices per operand so the index analysis has to
/// fold, and with a stride permutation in every operand so nothing is
/// accidentally contiguous.
#[test]
fn case1_general_strides_in_every_operand() {
    fn one<T: Dtype>() {
        // D[i,j,r] = sum_k A[r,i,k] B[k,j]: `r` and `i` are both free indices of
        // A but are not adjacent in the same order in A and D, so they cannot
        // fold, and A's fastest mode is a free index of D's slowest.
        let a = CTensor::new(&[2, 4, 3], &[12, 1, 4], &[R, I, K]);
        let b = CTensor::new(&[3, 5], &[5, 1], &[K, J]);
        let d = CTensor::new(&[4, 5, 2], &[5, 1, 25], &[I, J, R]);
        let av = seq::<T>(a.storage(), 11);
        let bv = seq::<T>(b.storage(), 12);
        Case::plain(a, &av, b, &bv, d).check_zeroed();
    }
    every_dtype!(one);
}

/// Case 2: a Hadamard / batch index, present in A, B and D.
#[test]
fn case2_hadamard_batch_index() {
    fn one<T: Dtype>() {
        // D[h,i,j] = sum_k A[h,i,k] B[h,k,j]
        let a = CTensor::new(&[2, 4, 3], &[1, 2, 8], &[H, I, K]);
        let b = CTensor::new(&[2, 3, 5], &[1, 2, 6], &[H, K, J]);
        let d = CTensor::new(&[2, 4, 5], &[1, 2, 8], &[H, I, J]);
        let av = seq::<T>(a.storage(), 21);
        let bv = seq::<T>(b.storage(), 22);
        Case::plain(a, &av, b, &bv, d).check_zeroed();
    }
    every_dtype!(one);
}

/// Case 3: a label repeated inside one tensor selects that tensor's diagonal.
#[test]
fn case3_repeated_index_takes_a_diagonal() {
    fn one<T: Dtype>() {
        // D[j] = sum_i A[i,i] B[i,j] -- A's main diagonal, contracted with B.
        let a = CTensor::new(&[4, 4], &[1, 4], &[I, I]);
        let b = CTensor::new(&[4, 5], &[1, 4], &[I, J]);
        let d = CTensor::new(&[5], &[1], &[J]);
        let av = seq::<T>(a.storage(), 31);
        let bv = seq::<T>(b.storage(), 32);
        Case::plain(a, &av, b, &bv, d).check_zeroed();
    }
    every_dtype!(one);
}

/// Case 3 with a label repeated three times, whose strides all add.
#[test]
fn case3_a_label_repeated_three_times() {
    fn one<T: Dtype>() {
        // D[j] = sum_i A[i,i,i] B[i,j] -- the space diagonal, stride 1+4+16.
        let a = CTensor::new(&[4, 4, 4], &[1, 4, 16], &[I, I, I]);
        let b = CTensor::new(&[4, 5], &[1, 4], &[I, J]);
        let d = CTensor::new(&[5], &[1], &[J]);
        let av = seq::<T>(a.storage(), 33);
        let bv = seq::<T>(b.storage(), 34);
        Case::plain(a, &av, b, &bv, d).check_zeroed();
    }
    every_dtype!(one);
}

/// Case 3 on the *output*: `D[i,i]` writes a diagonal and leaves the rest of
/// `D` alone.
///
/// Worth separating from the input case because it is the one where a repeated
/// label changes which elements are written rather than which are read, and the
/// whole-buffer comparison is what checks the off-diagonal is untouched.
#[test]
fn case3_repeated_index_in_the_output_writes_a_diagonal() {
    fn one<T: Dtype>() {
        // D[i,i] = sum_k A[i,k] B[k]
        let a = CTensor::new(&[4, 3], &[1, 4], &[I, K]);
        let b = CTensor::new(&[3], &[1], &[K]);
        let d = CTensor::new(&[4, 4], &[1, 4], &[I, I]);
        let av = seq::<T>(a.storage(), 41);
        let bv = seq::<T>(b.storage(), 42);
        // Pre-fill `D` so an over-write of the off-diagonal is visible.
        let mut dv = seq::<T>(16, 43);
        Case::plain(a, &av, b, &bv, d).check(&mut dv);
    }
    every_dtype!(one);
}

/// Case 4: an index isolated in `A`, i.e. a reduction over it.
#[test]
fn case4_isolated_index_in_a_is_a_reduction() {
    fn one<T: Dtype>() {
        // D[i,j] = sum_{k,r} A[i,k,r] B[k,j]
        let a = CTensor::new(&[4, 3, 2], &[1, 4, 12], &[I, K, R]);
        let b = CTensor::new(&[3, 5], &[1, 3], &[K, J]);
        let d = CTensor::new(&[4, 5], &[1, 4], &[I, J]);
        let av = seq::<T>(a.storage(), 51);
        let bv = seq::<T>(b.storage(), 52);
        Case::plain(a, &av, b, &bv, d).check_zeroed();
    }
    every_dtype!(one);
}

/// Case 4 mirrored: the isolated index lives in `B`.
#[test]
fn case4_isolated_index_in_b_is_a_reduction() {
    fn one<T: Dtype>() {
        // D[i,j] = sum_{k,r} A[i,k] B[k,j,r]
        let a = CTensor::new(&[4, 3], &[1, 4], &[I, K]);
        let b = CTensor::new(&[3, 5, 2], &[1, 3, 15], &[K, J, R]);
        let d = CTensor::new(&[4, 5], &[1, 4], &[I, J]);
        let av = seq::<T>(a.storage(), 61);
        let bv = seq::<T>(b.storage(), 62);
        Case::plain(a, &av, b, &bv, d).check_zeroed();
    }
    every_dtype!(one);
}

/// Case 4 taken to the limit: every index is contracted, so `D` is a scalar and
/// is declared with `nmode = 0` and null extent / stride arrays.
#[test]
fn case4_full_reduction_to_a_rank_zero_output() {
    fn one<T: Dtype>() {
        // D = sum_{i,j} A[i,j] B[i,j]
        let a = CTensor::new(&[4, 3], &[1, 4], &[I, J]);
        let b = CTensor::new(&[4, 3], &[3, 1], &[I, J]);
        let d = CTensor::new(&[], &[], &[]);
        let av = seq::<T>(a.storage(), 71);
        let bv = seq::<T>(b.storage(), 72);
        let out = Case::plain(a, &av, b, &bv, d).check_zeroed();
        assert_eq!(out.len(), 1);
    }
    every_dtype!(one);
}

/// A label may be any `int64_t`, including zero, a negative value and something
/// far outside the range of a mode count.
#[test]
fn index_labels_may_be_any_int64() {
    fn one<T: Dtype>() {
        const P: i64 = 0;
        const Q: i64 = -1;
        const S: i64 = i64::MIN + 1;
        let a = CTensor::new(&[5, 3], &[1, 5], &[P, S]);
        let b = CTensor::new(&[3, 4], &[4, 1], &[S, Q]);
        let d = CTensor::new(&[5, 4], &[1, 5], &[P, Q]);
        let av = seq::<T>(a.storage(), 81);
        let bv = seq::<T>(b.storage(), 82);
        Case::plain(a, &av, b, &bv, d).check_zeroed();
    }
    every_dtype!(one);
}

// -------------------------------------------------------------- conjugation

/// The four operands' `TAPP_element_op` arguments, one at a time.
///
/// `op_C` and `op_D` are the subtle ones: `D = op_D(alpha * op_A(A) * op_B(B) +
/// beta * op_C(C))`, so conjugating the output conjugates the `beta * C` term
/// too, and on the second and later `KC` blocks the engine has to set `op_C`
/// equal to `op_D` to accumulate correctly (D9). The oracle defines that
/// semantics independently.
fn conjugation<T: Dtype>(mask: u8) {
    let a = CTensor::new(&[4, 3], &[1, 4], &[I, K]).with_op(op_flag(mask & 1));
    let b = CTensor::new(&[3, 5], &[1, 3], &[K, J]).with_op(op_flag(mask & 2));
    let d = CTensor::new(&[4, 5], &[1, 4], &[I, J]);
    let av = seq::<T>(a.storage(), 91);
    let bv = seq::<T>(b.storage(), 92);
    let cv = seq::<T>(d.storage(), 93);
    let mut case = Case::plain(a, &av, b, &bv, d.with_op(op_flag(mask & 8)))
        .with_alpha(scalar::<T>(0.5, -0.25))
        .with_c(&cv, scalar::<T>(-2.0, 0.75));
    case.c = case.c.with_op(op_flag(mask & 4));
    case.check_zeroed();
}

fn op_flag(bit: u8) -> std::os::raw::c_int {
    if bit != 0 {
        TAPP_CONJUGATE
    } else {
        TAPP_IDENTITY
    }
}

#[test]
fn conjugate_op_a() {
    fn one<T: Dtype>() {
        conjugation::<T>(1);
    }
    every_dtype!(one);
}

#[test]
fn conjugate_op_b() {
    fn one<T: Dtype>() {
        conjugation::<T>(2);
    }
    every_dtype!(one);
}

#[test]
fn conjugate_op_c() {
    fn one<T: Dtype>() {
        conjugation::<T>(4);
    }
    every_dtype!(one);
}

#[test]
fn conjugate_op_d() {
    fn one<T: Dtype>() {
        conjugation::<T>(8);
    }
    every_dtype!(one);
}

#[test]
fn every_conjugation_mask() {
    fn one<T: Dtype>() {
        for mask in 0..16u8 {
            conjugation::<T>(mask);
        }
    }
    every_dtype!(one);
}

/// The same 16 masks, but deep enough that the driver splits `K` into several
/// `KC` blocks, which is the only configuration where D9's "set `op_C` equal to
/// `op_D` on accumulate passes" is exercised.
///
/// `KC` is 384 real elements in single precision and 256 in double, so `k = 600`
/// forces at least two blocks in every datatype. `M` and `N` stay tiny so the
/// oracle's `O(prod of all extents)` cost is negligible.
#[test]
fn every_conjugation_mask_across_several_kc_blocks() {
    fn one<T: Dtype>() {
        const KK: i64 = 600;
        for mask in 0..16u8 {
            let a = CTensor::new(&[3, KK], &[1, 3], &[I, K]).with_op(op_flag(mask & 1));
            let b = CTensor::new(&[KK, 2], &[1, KK], &[K, J]).with_op(op_flag(mask & 2));
            let d = CTensor::new(&[3, 2], &[1, 3], &[I, J]);
            let av = seq::<T>(a.storage(), 101);
            let bv = seq::<T>(b.storage(), 102);
            let cv = seq::<T>(d.storage(), 103);
            let mut case = Case::plain(a, &av, b, &bv, d.with_op(op_flag(mask & 8)))
                .with_alpha(scalar::<T>(1.25, -0.5))
                .with_c(&cv, scalar::<T>(0.75, 0.5));
            case.c = case.c.with_op(op_flag(mask & 4));
            case.check_zeroed();
        }
    }
    every_dtype!(one);
}

/// `TAPP_CONJUGATE` on a real operand is accepted and is the identity. The
/// coverage table says "conjugation on any operand", and a C caller writing
/// generic code will pass the same op for `TAPP_F64` as for `TAPP_C64`.
#[test]
fn conjugation_of_a_real_operand_is_a_no_op() {
    fn one<T: Dtype>() {
        let a = CTensor::new(&[4, 3], &[1, 4], &[I, K]);
        let b = CTensor::new(&[3, 5], &[1, 3], &[K, J]);
        let d = CTensor::new(&[4, 5], &[1, 4], &[I, J]);
        let av = seq::<T>(a.storage(), 111);
        let bv = seq::<T>(b.storage(), 112);
        let plain = Case::plain(a, &av, b, &bv, d).check_zeroed();
        let conj = Case::plain(a.conj(), &av, b.conj(), &bv, d).check_zeroed();
        if T::IS_COMPLEX {
            // conj(A) conj(B) = conj(A B), so the two differ except when real.
            // Compared with a tolerance rather than exactly, because 3m's
            // Karatsuba recombination is not sign-symmetric: `fl(ar - ai)` is
            // unrelated to `fl(ar + ai)`, so the identity holds numerically
            // rather than bitwise under `TENSORCONTRACT_COMPLEX=3m`.
            let flipped: Vec<T> = plain.iter().map(|v| v.conj()).collect();
            assert_close::<T>(&conj, &flipped);
        } else {
            assert_exact::<T>(&conj, &plain);
        }
    }
    every_dtype!(one);
}

// ------------------------------------------------------------- alpha and beta

#[test]
fn alpha_scales_the_product_and_beta_scales_c() {
    fn one<T: Dtype>() {
        let a = CTensor::new(&[4, 3], &[1, 4], &[I, K]);
        let b = CTensor::new(&[3, 5], &[1, 3], &[K, J]);
        let d = CTensor::new(&[4, 5], &[1, 4], &[I, J]);
        let av = seq::<T>(a.storage(), 121);
        let bv = seq::<T>(b.storage(), 122);
        let cv = seq::<T>(d.storage(), 123);
        Case::plain(a, &av, b, &bv, d)
            .with_alpha(scalar::<T>(2.0, -3.0))
            .with_c(&cv, scalar::<T>(-1.0, 0.5))
            .check_zeroed();
    }
    every_dtype!(one);
}

#[test]
fn alpha_zero_gives_beta_times_c() {
    fn one<T: Dtype>() {
        let a = CTensor::new(&[4, 3], &[1, 4], &[I, K]);
        let b = CTensor::new(&[3, 5], &[1, 3], &[K, J]);
        let d = CTensor::new(&[4, 5], &[1, 4], &[I, J]);
        let av = seq::<T>(a.storage(), 131);
        let bv = seq::<T>(b.storage(), 132);
        let cv = seq::<T>(d.storage(), 133);
        let out = Case::plain(a, &av, b, &bv, d)
            .with_alpha(T::zero())
            .with_c(&cv, T::one())
            .check_zeroed();
        assert_exact::<T>(&out, &cv);
    }
    every_dtype!(one);
}

/// `beta = 0` must mean `C` is never read, whatever it contains.
///
/// The `C` buffer here is entirely NaN, which no arithmetic can absorb: a single
/// `0 * NaN` anywhere in the write-back would produce a NaN in `D`, and
/// `assert_close` rejects a non-finite output before it even compares values.
#[test]
fn beta_zero_never_reads_c() {
    fn one<T: Dtype>() {
        let a = CTensor::new(&[4, 3], &[1, 4], &[I, K]);
        let b = CTensor::new(&[3, 5], &[1, 3], &[K, J]);
        let d = CTensor::new(&[4, 5], &[1, 4], &[I, J]);
        let av = seq::<T>(a.storage(), 141);
        let bv = seq::<T>(b.storage(), 142);
        let cv = poison::<T>(d.storage());
        Case::plain(a, &av, b, &bv, d)
            .with_c(&cv, T::zero())
            .check_zeroed();
    }
    every_dtype!(one);
}

/// The same, across several `KC` blocks, where the write-back switches to its
/// accumulate path and `C` is replaced by `D`.
#[test]
fn beta_zero_never_reads_c_across_several_kc_blocks() {
    fn one<T: Dtype>() {
        const KK: i64 = 600;
        let a = CTensor::new(&[3, KK], &[1, 3], &[I, K]);
        let b = CTensor::new(&[KK, 2], &[1, KK], &[K, J]);
        let d = CTensor::new(&[3, 2], &[1, 3], &[I, J]);
        let av = seq::<T>(a.storage(), 151);
        let bv = seq::<T>(b.storage(), 152);
        let cv = poison::<T>(d.storage());
        Case::plain(a, &av, b, &bv, d)
            .with_c(&cv, T::zero())
            .check_zeroed();
    }
    every_dtype!(one);
}

/// Accumulate into a `D` that is already non-zero, by passing `D`'s own pointer
/// for `C` — the in-place update a C caller writes as
/// `TAPP_execute_product(..., beta, D, D)`.
#[test]
fn accumulation_into_a_nonzero_d() {
    fn one<T: Dtype>() {
        let a = CTensor::new(&[4, 3], &[1, 4], &[I, K]);
        let b = CTensor::new(&[3, 5], &[1, 3], &[K, J]);
        let d = CTensor::new(&[4, 5], &[1, 4], &[I, J]);
        let av = seq::<T>(a.storage(), 161);
        let bv = seq::<T>(b.storage(), 162);
        let mut dv = seq::<T>(d.storage(), 163);
        Case::plain(a, &av, b, &bv, d)
            .accumulating(T::one())
            .check(&mut dv);
    }
    every_dtype!(one);
}

/// Accumulating in place across several `KC` blocks: the first block consumes
/// `beta * C`, the rest must accumulate into `D` and must not apply `beta`
/// again.
#[test]
fn accumulation_into_a_nonzero_d_across_several_kc_blocks() {
    fn one<T: Dtype>() {
        const KK: i64 = 600;
        let a = CTensor::new(&[3, KK], &[1, 3], &[I, K]);
        let b = CTensor::new(&[KK, 2], &[1, KK], &[K, J]);
        let d = CTensor::new(&[3, 2], &[1, 3], &[I, J]);
        let av = seq::<T>(a.storage(), 171);
        let bv = seq::<T>(b.storage(), 172);
        let mut dv = seq::<T>(d.storage(), 173);
        Case::plain(a, &av, b, &bv, d)
            .accumulating(scalar::<T>(3.0, -1.0))
            .check(&mut dv);
    }
    every_dtype!(one);
}

/// A `C` with a *different* layout from `D`, which the write-back has to reach
/// through its own scatter rather than `D`'s.
#[test]
fn c_may_have_a_different_layout_from_d() {
    fn one<T: Dtype>() {
        let a = CTensor::new(&[4, 3], &[1, 4], &[I, K]);
        let b = CTensor::new(&[3, 5], &[1, 3], &[K, J]);
        // `D` column-major, `C` row-major over the same index set.
        let d = CTensor::new(&[4, 5], &[1, 4], &[I, J]);
        let c = CTensor::new(&[4, 5], &[5, 1], &[I, J]);
        let av = seq::<T>(a.storage(), 181);
        let bv = seq::<T>(b.storage(), 182);
        let cv = seq::<T>(c.storage(), 183);
        let mut case = Case::plain(a, &av, b, &bv, d).with_c(&cv, scalar::<T>(1.5, -0.5));
        case.c = c;
        case.check_zeroed();
    }
    every_dtype!(one);
}

/// `C = NULL` with a non-zero `beta` is refused, not silently reinterpreted.
///
/// Upstream `tapp/product.h` defines `TAPP_IN_PLACE` as `NULL` and leaves its
/// meaning an open `//TODO: in-place operation: set C = NULL or TAPP_IN_PLACE?`.
/// This test used to pin the old behaviour — `beta` ignored — under which a
/// caller writing `beta = 1, C = TAPP_IN_PLACE` and expecting `D += alpha*A*B`
/// got `D` silently overwritten. Since the constant's *name* suggests the
/// opposite of what the engine would do, the ambiguous combination now returns
/// `TAPP_ERROR_UNSUPPORTED`; a null `C` with `beta == 0` still overwrites `D`,
/// and in-place accumulation is expressible by passing `D`'s own pointer as `C`
/// (see `accumulation_into_a_nonzero_d`).
#[test]
fn null_c_with_nonzero_beta_is_rejected() {
    fn one<T: Dtype>() {
        let a = CTensor::new(&[4, 3], &[1, 4], &[I, K]);
        let b = CTensor::new(&[3, 5], &[1, 3], &[K, J]);
        let d = CTensor::new(&[4, 5], &[1, 4], &[I, J]);
        let av = seq::<T>(a.storage(), 191);
        let bv = seq::<T>(b.storage(), 192);
        // A null `C` with `beta == 0` is the unambiguous half and still works.
        let overwritten = Case::plain(a, &av, b, &bv, d).check_zeroed();

        // The ambiguous half: non-zero `beta` with `C = TAPP_IN_PLACE`. It must
        // report rather than compute, and must leave `D` alone while doing so.
        let before = seq::<T>(d.storage(), 193);
        let mut dv = before.clone();
        let mut case = Case::plain(a, &av, b, &bv, d);
        case.beta = scalar::<T>(1.0, 0.0);
        let e = unsafe { case.run(&mut dv) };
        assert_eq!(e, TAPP_ERROR_UNSUPPORTED, "{}", explain_status(e));
        assert_exact::<T>(&dv, &before);
        let _ = overwritten;
    }
    every_dtype!(one);
}

// ------------------------------------------------------------------- batching

/// `TAPP_execute_batched_product` over three batches must agree with three
/// `TAPP_execute_product` calls on the same plan.
#[test]
fn batched_product_matches_a_loop_of_single_products() {
    fn one<T: Dtype>() {
        const NB: usize = 3;
        let a = CTensor::new(&[4, 3], &[1, 4], &[I, K]);
        let b = CTensor::new(&[3, 5], &[1, 3], &[K, J]);
        let d = CTensor::new(&[4, 5], &[1, 4], &[I, J]);
        let avs: Vec<Vec<T>> = (0..NB)
            .map(|i| seq::<T>(a.storage(), 200 + i as u64))
            .collect();
        let bvs: Vec<Vec<T>> = (0..NB)
            .map(|i| seq::<T>(b.storage(), 210 + i as u64))
            .collect();
        let cvs: Vec<Vec<T>> = (0..NB)
            .map(|i| seq::<T>(d.storage(), 220 + i as u64))
            .collect();
        let alpha = scalar::<T>(0.75, -1.5);
        let beta = scalar::<T>(-0.5, 0.25);

        // One at a time.
        let mut want: Vec<Vec<T>> = Vec::new();
        for ((av, bv), cv) in avs.iter().zip(&bvs).zip(&cvs) {
            let case = Case::plain(a, av, b, bv, d)
                .with_alpha(alpha)
                .with_c(cv, beta);
            want.push(case.check_zeroed());
        }

        // All at once, on one plan.
        let mut dvs: Vec<Vec<T>> = (0..NB).map(|_| vec![T::zero(); d.storage()]).collect();
        let aptrs: Vec<*const c_void> = avs.iter().map(|v| v.as_ptr() as *const c_void).collect();
        let bptrs: Vec<*const c_void> = bvs.iter().map(|v| v.as_ptr() as *const c_void).collect();
        let cptrs: Vec<*const c_void> = cvs.iter().map(|v| v.as_ptr() as *const c_void).collect();
        let mut dptrs: Vec<*mut c_void> = dvs
            .iter_mut()
            .map(|v| v.as_mut_ptr() as *mut c_void)
            .collect();
        unsafe {
            let handle = create_handle();
            let exec = create_executor();
            let case = Case::plain(a, &avs[0], b, &bvs[0], d)
                .with_alpha(alpha)
                .with_c(&cvs[0], beta);
            let (e, plan) = case.create_plan(handle);
            assert_eq!(e, TAPP_SUCCESS, "{}", explain_status(e));
            let mut status = 0isize;
            let e = TAPP_execute_batched_product(
                plan,
                exec,
                &mut status,
                NB as std::os::raw::c_int,
                &alpha as *const T as *const c_void,
                aptrs.as_ptr(),
                bptrs.as_ptr(),
                &beta as *const T as *const c_void,
                cptrs.as_ptr(),
                dptrs.as_mut_ptr(),
            );
            assert_eq!(e, TAPP_SUCCESS, "{}", explain_status(e));
            assert_eq!(TAPP_destroy_status(status), TAPP_SUCCESS);
            assert_eq!(TAPP_destroy_tensor_product(plan), TAPP_SUCCESS);
            assert_eq!(TAPP_destroy_executor(exec), TAPP_SUCCESS);
            assert_eq!(TAPP_destroy_handle(handle), TAPP_SUCCESS);
        }
        for (got, w) in dvs.iter().zip(&want) {
            assert_exact::<T>(got, w);
        }
    }
    every_dtype!(one);
}

/// A null `C` *array* means "no `C` in any batch", which the shim turns into a
/// null `C` per batch.
#[test]
fn batched_product_accepts_a_null_c_array() {
    fn one<T: Dtype>() {
        const NB: usize = 2;
        let a = CTensor::new(&[4, 3], &[1, 4], &[I, K]);
        let b = CTensor::new(&[3, 5], &[1, 3], &[K, J]);
        let d = CTensor::new(&[4, 5], &[1, 4], &[I, J]);
        let avs: Vec<Vec<T>> = (0..NB)
            .map(|i| seq::<T>(a.storage(), 230 + i as u64))
            .collect();
        let bvs: Vec<Vec<T>> = (0..NB)
            .map(|i| seq::<T>(b.storage(), 240 + i as u64))
            .collect();
        let alpha = T::one();
        let beta = T::zero();
        let want: Vec<Vec<T>> = avs
            .iter()
            .zip(&bvs)
            .map(|(av, bv)| Case::plain(a, av, b, bv, d).check_zeroed())
            .collect();

        let mut dvs: Vec<Vec<T>> = (0..NB).map(|_| vec![T::zero(); d.storage()]).collect();
        let aptrs: Vec<*const c_void> = avs.iter().map(|v| v.as_ptr() as *const c_void).collect();
        let bptrs: Vec<*const c_void> = bvs.iter().map(|v| v.as_ptr() as *const c_void).collect();
        let mut dptrs: Vec<*mut c_void> = dvs
            .iter_mut()
            .map(|v| v.as_mut_ptr() as *mut c_void)
            .collect();
        unsafe {
            let handle = create_handle();
            let exec = create_executor();
            let (e, plan) = Case::plain(a, &avs[0], b, &bvs[0], d).create_plan(handle);
            assert_eq!(e, TAPP_SUCCESS, "{}", explain_status(e));
            let e = TAPP_execute_batched_product(
                plan,
                exec,
                std::ptr::null_mut(),
                NB as std::os::raw::c_int,
                &alpha as *const T as *const c_void,
                aptrs.as_ptr(),
                bptrs.as_ptr(),
                &beta as *const T as *const c_void,
                std::ptr::null(),
                dptrs.as_mut_ptr(),
            );
            assert_eq!(e, TAPP_SUCCESS, "{}", explain_status(e));
            assert_eq!(TAPP_destroy_tensor_product(plan), TAPP_SUCCESS);
            assert_eq!(TAPP_destroy_executor(exec), TAPP_SUCCESS);
            assert_eq!(TAPP_destroy_handle(handle), TAPP_SUCCESS);
        }
        for (got, w) in dvs.iter().zip(&want) {
            assert_exact::<T>(got, w);
        }
    }
    every_dtype!(one);
}

// ------------------------------------------------------------------ precision

/// The `TAPP_prectype` argument is honoured, not ignored: `TAPP_DEFAULT_PREC` and
/// the precision of the storage type give the same answer, and every other value
/// — the other precision, a low-precision enumerator, or one that is not an
/// enumerator at all — is refused when the plan is created.
#[test]
fn prec_is_checked_against_the_storage_precision() {
    fn one<T: Dtype>() {
        let a = CTensor::new(&[4, 3], &[1, 4], &[I, K]);
        let b = CTensor::new(&[3, 5], &[1, 3], &[K, J]);
        let d = CTensor::new(&[4, 5], &[1, 4], &[I, J]);
        let av = seq::<T>(a.storage(), 251);
        let bv = seq::<T>(b.storage(), 252);
        let base = Case::plain(a, &av, b, &bv, d).check_zeroed();
        let single = matches!(T::TAG, TAPP_F32 | TAPP_C32);
        let (matching, other) = if single {
            (TAPP_F32F32_ACCUM_F32, TAPP_F64F64_ACCUM_F64)
        } else {
            (TAPP_F64F64_ACCUM_F64, TAPP_F32F32_ACCUM_F32)
        };
        let out = Case::plain(a, &av, b, &bv, d)
            .with_prec(matching)
            .check_zeroed();
        assert_exact::<T>(&out, &base);
        for prec in [other, 3, 5, 6, 999, -2] {
            unsafe {
                let handle = create_handle();
                let (e, _) = Case::plain(a, &av, b, &bv, d)
                    .with_prec(prec)
                    .create_plan(handle);
                assert_eq!(e, TAPP_ERROR_UNSUPPORTED, "prec {prec} for {}", T::NAME);
                assert_eq!(TAPP_destroy_handle(handle), TAPP_SUCCESS);
            }
        }
    }
    every_dtype!(one);
}

/// One executor and one plan, executed repeatedly. A C caller builds the plan
/// once and runs it many times; nothing in the shim may accumulate state.
#[test]
fn one_plan_executes_repeatedly() {
    fn one<T: Dtype>() {
        let a = CTensor::new(&[4, 3], &[1, 4], &[I, K]);
        let b = CTensor::new(&[3, 5], &[1, 3], &[K, J]);
        let d = CTensor::new(&[4, 5], &[1, 4], &[I, J]);
        let av = seq::<T>(a.storage(), 261);
        let bv = seq::<T>(b.storage(), 262);
        let want = Case::plain(a, &av, b, &bv, d).check_zeroed();
        unsafe {
            let handle = create_handle();
            let exec = create_executor();
            let (e, plan) = Case::plain(a, &av, b, &bv, d).create_plan(handle);
            assert_eq!(e, TAPP_SUCCESS, "{}", explain_status(e));
            let alpha = T::one();
            let beta = T::zero();
            for _ in 0..3 {
                let mut dv = vec![T::zero(); d.storage()];
                let e = TAPP_execute_product(
                    plan,
                    exec,
                    std::ptr::null_mut(),
                    &alpha as *const T as *const c_void,
                    av.as_ptr() as *const c_void,
                    bv.as_ptr() as *const c_void,
                    &beta as *const T as *const c_void,
                    std::ptr::null(),
                    dv.as_mut_ptr() as *mut c_void,
                );
                assert_eq!(e, TAPP_SUCCESS, "{}", explain_status(e));
                assert_exact::<T>(&dv, &want);
            }
            assert_eq!(TAPP_destroy_tensor_product(plan), TAPP_SUCCESS);
            assert_eq!(TAPP_destroy_executor(exec), TAPP_SUCCESS);
            assert_eq!(TAPP_destroy_handle(handle), TAPP_SUCCESS);
        }
    }
    every_dtype!(one);
}

// ------------------------------------------------------------- other corners

/// An empty contraction dimension with conjugation on `C` and `D`.
///
/// `k = 0` takes a separate write-back path (`D = op_D(beta * op_C(C))`, no
/// kernel at all), so the conjugation flags have to be honoured there too and
/// not only on the accumulating path the masks above exercise.
#[test]
fn an_empty_contraction_honours_the_conjugation_flags() {
    fn one<T: Dtype>() {
        for mask in 0..4u8 {
            let a = CTensor::new(&[4, 0], &[1, 4], &[I, K]);
            let b = CTensor::new(&[0, 5], &[1, 0], &[K, J]);
            let d = CTensor::new(&[4, 5], &[1, 4], &[I, J]).with_op(op_flag(mask & 2));
            let av: Vec<T> = Vec::new();
            let bv: Vec<T> = Vec::new();
            let cv = seq::<T>(d.storage(), 271);
            let mut case = Case::plain(a, &av, b, &bv, d)
                .with_alpha(scalar::<T>(2.0, 1.0))
                .with_c(&cv, scalar::<T>(-1.5, 0.25));
            case.c = case.c.with_op(op_flag(mask & 1));
            case.check_zeroed();
        }
    }
    every_dtype!(one);
}

/// A tensor info may be built wrong and then corrected with the setters before
/// planning; the plan must read the info as it stands at
/// `TAPP_create_tensor_product` time.
///
/// That is what the `TAPP_set_*` half of `tapp/tensor.h` is *for* — a C caller
/// that keeps one info and reshapes it per call — and it is the only way the
/// mutators can be tested for effect rather than merely for round-tripping.
#[test]
fn tensor_info_setters_take_effect_before_planning() {
    fn one<T: Dtype>() {
        let a = CTensor::new(&[5, 3], &[1, 5], &[I, K]);
        let b = CTensor::new(&[3, 4], &[4, 1], &[K, J]);
        let d = CTensor::new(&[5, 4], &[1, 5], &[I, J]);
        let av = seq::<T>(a.storage(), 281);
        let bv = seq::<T>(b.storage(), 282);
        let want = Case::plain(a, &av, b, &bv, d).check_zeroed();

        unsafe {
            let handle = create_handle();
            // Every info starts with the wrong shape and the wrong strides.
            let wrong_e = [2i64, 2];
            let wrong_s = [2i64, 1];
            let mut infos = [0isize; 4];
            for i in infos.iter_mut() {
                assert_eq!(
                    TAPP_create_tensor_info(i, T::TAG, 2, wrong_e.as_ptr(), wrong_s.as_ptr()),
                    TAPP_SUCCESS
                );
            }
            for (info, t) in infos.iter().zip([&a, &b, &d, &d]) {
                assert_eq!(TAPP_set_extents(*info, t.extents.as_ptr()), TAPP_SUCCESS);
                assert_eq!(TAPP_set_strides(*info, t.strides.as_ptr()), TAPP_SUCCESS);
            }
            let mut plan = 0isize;
            let e = TAPP_create_tensor_product(
                &mut plan,
                handle,
                TAPP_IDENTITY,
                infos[0],
                a.labels_ptr(),
                TAPP_IDENTITY,
                infos[1],
                b.labels_ptr(),
                TAPP_IDENTITY,
                infos[2],
                d.labels_ptr(),
                TAPP_IDENTITY,
                infos[3],
                d.labels_ptr(),
                TAPP_DEFAULT_PREC,
            );
            assert_eq!(e, TAPP_SUCCESS, "{}", explain_status(e));
            let alpha = T::one();
            let beta = T::zero();
            let mut dv = vec![T::zero(); d.storage()];
            let e = TAPP_execute_product(
                plan,
                0,
                std::ptr::null_mut(),
                &alpha as *const T as *const c_void,
                av.as_ptr() as *const c_void,
                bv.as_ptr() as *const c_void,
                &beta as *const T as *const c_void,
                std::ptr::null(),
                dv.as_mut_ptr() as *mut c_void,
            );
            assert_eq!(e, TAPP_SUCCESS, "{}", explain_status(e));
            assert_exact::<T>(&dv, &want);
            assert_eq!(TAPP_destroy_tensor_product(plan), TAPP_SUCCESS);
            for i in infos {
                assert_eq!(TAPP_destroy_tensor_info(i), TAPP_SUCCESS);
            }
            assert_eq!(TAPP_destroy_handle(handle), TAPP_SUCCESS);
        }
    }
    every_dtype!(one);
}

/// A negative stride reverses an axis, and the ABI expresses it the only way it
/// can: the caller's data pointer sits at the far end of that axis and the
/// offsets run backwards from it.
///
/// This one cannot use `contract_reference`, which indexes a slice with the
/// computed offset and so cannot represent a base pointer in the middle of a
/// buffer. It is checked against an equivalence instead, which is independent of
/// both the engine and the oracle: reversing `A`'s rows *and* negating its row
/// stride is the identity, so the two calls must produce the same `D`.
#[test]
fn negative_strides_are_honoured() {
    fn one<T: Dtype>() {
        const M: usize = 5;
        const KK: usize = 3;
        let b = CTensor::new(&[3, 4], &[4, 1], &[K, J]);
        let d = CTensor::new(&[5, 4], &[1, 5], &[I, J]);
        let bv = seq::<T>(b.storage(), 291);

        // `A`, column-major, and the same matrix with its rows reversed.
        let fwd = seq::<T>(M * KK, 292);
        let mut rev = [T::zero(); M * KK];
        for k in 0..KK {
            for i in 0..M {
                rev[(M - 1 - i) + M * k] = fwd[i + M * k];
            }
        }

        // `D = A B` the ordinary way.
        let a_pos = CTensor::new(&[5, 3], &[1, 5], &[I, K]);
        let want = Case::plain(a_pos, &fwd, b, &bv, d).check_zeroed();

        // `D = A B` again, reading the reversed copy backwards: same extents,
        // row stride -1, and a base pointer at row `M - 1`.
        let a_neg = CTensor::new(&[5, 3], &[-1, 5], &[I, K]);
        let mut dv = vec![T::zero(); d.storage()];
        unsafe {
            let handle = create_handle();
            let ia = a_neg.create_info(T::TAG);
            let ib = b.create_info(T::TAG);
            let id = d.create_info(T::TAG);
            let mut plan = 0isize;
            let e = TAPP_create_tensor_product(
                &mut plan,
                handle,
                TAPP_IDENTITY,
                ia,
                a_neg.labels_ptr(),
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
            );
            assert_eq!(e, TAPP_SUCCESS, "{}", explain_status(e));
            let alpha = T::one();
            let beta = T::zero();
            // The base pointer a C caller would pass: `&rev[M - 1]`, so that
            // offset `-i` reaches row `i` of the original.
            let base = rev.as_ptr().add(M - 1);
            let e = TAPP_execute_product(
                plan,
                0,
                std::ptr::null_mut(),
                &alpha as *const T as *const c_void,
                base as *const c_void,
                bv.as_ptr() as *const c_void,
                &beta as *const T as *const c_void,
                std::ptr::null(),
                dv.as_mut_ptr() as *mut c_void,
            );
            assert_eq!(e, TAPP_SUCCESS, "{}", explain_status(e));
            assert_eq!(TAPP_destroy_tensor_product(plan), TAPP_SUCCESS);
            for i in [ia, ib, id] {
                assert_eq!(TAPP_destroy_tensor_info(i), TAPP_SUCCESS);
            }
            assert_eq!(TAPP_destroy_handle(handle), TAPP_SUCCESS);
        }
        assert_exact::<T>(&dv, &want);
    }
    every_dtype!(one);
}
