//! TAPP C-ABI conformance: the layout and linkage assumptions underneath it.
//!
//! The rest of the suite calls the crate's `extern "C"` functions by Rust path,
//! which checks their behaviour but not their *linkage*: a `#[no_mangle]` that
//! was renamed, dropped or given a signature the header does not describe would
//! sail through. This file therefore re-declares the whole of the upstream
//! header in an `extern "C"` block and drives a complete contraction through
//! those declarations, so the symbols have to exist, have to be `extern "C"`,
//! and have to take the argument list `api/include/tapp/*.h` publishes.
//!
//! Everything asserted here about the upstream headers was read from
//! `TAPPorg/reference-implementation` at `main`, not from the paper — the paper
//! does not print the prototypes. The numeric enumerator values in
//! `enumerator_values_match_the_upstream_headers` are quoted from
//! `api/include/tapp/datatype.h` and `api/include/tapp/product.h`.

mod common;

use std::ffi::{c_void, CStr};
use std::mem::{align_of, offset_of, size_of};
use std::os::raw::{c_char, c_int};
use std::slice;

use num_complex::Complex;

use common::explain_status;
use tcontract_capi::TAPP_DEFAULT_PREC;
use tcontract_capi::TAPP_ERROR_UNSUPPORTED;

/// The upstream header, transcribed. Types follow `tapp/*.h`: every handle is
/// `intptr_t`, `TAPP_error` / `TAPP_datatype` / `TAPP_prectype` /
/// `TAPP_element_op` are `int`, label and shape arrays are `const int64_t*`.
mod c_abi {
    use std::ffi::c_void;
    use std::os::raw::{c_char, c_int};

    /// `include/tcontract/tapp_ext.h`, `typedef struct { size_t stack_size; }`.
    #[repr(C)]
    pub struct tcontract_rayon_opts {
        pub stack_size: usize,
    }

    extern "C" {
        // tapp/error.h
        pub fn TAPP_check_success(error: c_int) -> bool;
        pub fn TAPP_explain_error(error: c_int, maxlen: usize, message: *mut c_char) -> usize;

        // tapp/handle.h
        pub fn TAPP_create_handle(handle: *mut isize) -> c_int;
        pub fn TAPP_destroy_handle(handle: isize) -> c_int;

        // tapp/executor.h
        pub fn TAPP_create_executor(exec: *mut isize) -> c_int;
        pub fn TAPP_destroy_executor(exec: isize) -> c_int;

        // tapp/status.h
        pub fn TAPP_destroy_status(status: isize) -> c_int;

        // tapp/tensor.h
        pub fn TAPP_create_tensor_info(
            info: *mut isize,
            ty: c_int,
            nmode: c_int,
            extents: *const i64,
            strides: *const i64,
        ) -> c_int;
        pub fn TAPP_destroy_tensor_info(info: isize) -> c_int;
        pub fn TAPP_get_nmodes(info: isize) -> c_int;
        pub fn TAPP_set_nmodes(info: isize, nmodes: c_int) -> c_int;
        pub fn TAPP_get_extents(info: isize, extents: *mut i64);
        pub fn TAPP_set_extents(info: isize, extents: *const i64) -> c_int;
        pub fn TAPP_get_strides(info: isize, strides: *mut i64);
        pub fn TAPP_set_strides(info: isize, strides: *const i64) -> c_int;

        // tapp/product.h
        #[allow(clippy::too_many_arguments)]
        pub fn TAPP_create_tensor_product(
            plan: *mut isize,
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
        ) -> c_int;
        pub fn TAPP_destroy_tensor_product(plan: isize) -> c_int;
        pub fn TAPP_execute_product(
            plan: isize,
            exec: isize,
            status: *mut isize,
            alpha: *const c_void,
            a: *const c_void,
            b: *const c_void,
            beta: *const c_void,
            c: *const c_void,
            d: *mut c_void,
        ) -> c_int;
        pub fn TAPP_execute_batched_product(
            plan: isize,
            exec: isize,
            status: *mut isize,
            num_batches: c_int,
            alpha: *const c_void,
            a: *const *const c_void,
            b: *const *const c_void,
            beta: *const c_void,
            c: *const *const c_void,
            d: *mut *mut c_void,
        ) -> c_int;

        // Non-standard extensions, documented in the crate.
        pub fn TAPP_implementation_name() -> *const c_char;
        pub fn TAPP_implementation_version() -> *const c_char;

        // `include/tcontract/tapp_ext.h`: the host-owned pool and the executor
        // that borrows it. Transcribed here as well so a renamed symbol or a
        // changed argument list is a link (or call) failure in this crate's own
        // suite, not only in the C program `c_abi.rs` compiles.
        pub fn tcontract_pool_create_rayon(
            out: *mut isize,
            nthreads: usize,
            opts: *const tcontract_rayon_opts,
        ) -> c_int;
        pub fn tcontract_pool_destroy(pool: isize) -> c_int;
        pub fn tcontract_tapp_executor_borrow_pool(
            out: *mut isize,
            pool: isize,
            budget: usize,
        ) -> c_int;
        pub fn tcontract_tapp_executor_get_threads(
            exec: isize,
            pool_size: *mut usize,
            budget: *mut usize,
        ) -> c_int;

        // `api/include/tapp/attributes.h`. These were missing entirely when this
        // suite was written — a C program including `<tapp.h>` and calling one
        // failed to *link*, the least diagnosable failure available — and are now
        // exported as refusals. Declared here so their absence would once again
        // be a link error in this crate's own tests rather than downstream.
        pub fn TAPP_attr_set(attr: isize, key: c_int, value: *mut c_void) -> c_int;
        pub fn TAPP_attr_get(attr: isize, key: c_int, value: *mut *mut c_void) -> c_int;
        pub fn TAPP_attr_clear(attr: isize, key: c_int) -> c_int;
    }
}

/// The pool extension resolves through the transcribed prototypes: the host
/// creates one pool, builds an executor that borrows it, reads the width back
/// and tears both down in the documented order.
#[test]
fn the_pool_extensions_resolve_and_run() {
    unsafe {
        let mut pool = 0isize;
        let opts = c_abi::tcontract_rayon_opts {
            stack_size: 1 << 20,
        };
        assert_eq!(
            c_abi::tcontract_pool_create_rayon(&mut pool, 3, &opts),
            c_int::from(0)
        );
        assert_ne!(pool, 0);
        let mut exec = 0isize;
        assert_eq!(
            c_abi::tcontract_tapp_executor_borrow_pool(&mut exec, pool, 2),
            c_int::from(0)
        );
        let (mut width, mut budget) = (0usize, 0usize);
        assert_eq!(
            c_abi::tcontract_tapp_executor_get_threads(exec, &mut width, &mut budget),
            c_int::from(0)
        );
        assert_eq!((width, budget), (3, 2));
        // The C contract: the pool outlives the executor built from it.
        assert_ne!(c_abi::tcontract_pool_destroy(pool), c_int::from(0));
        assert_eq!(c_abi::TAPP_destroy_executor(exec), c_int::from(0));
        assert_eq!(c_abi::tcontract_pool_destroy(pool), c_int::from(0));
    }
}

/// The attribute API links, refuses, and does not touch anything on the way.
///
/// Upstream specifies no attribute keys, so refusing every key is conformant;
/// what is not conformant is failing to export the symbols at all, which is what
/// this crate did until this suite transcribed the header and noticed.
/// `TAPP_attr_get` must also leave a defined value behind, so that a caller who
/// ignores the return code does not read its own uninitialised stack slot.
#[test]
fn the_attribute_api_links_and_refuses() {
    const SENTINEL: *mut c_void = usize::MAX as *mut c_void;
    unsafe {
        assert_eq!(
            c_abi::TAPP_attr_set(0, 7, std::ptr::null_mut()),
            TAPP_ERROR_UNSUPPORTED
        );
        assert_eq!(c_abi::TAPP_attr_clear(0, 7), TAPP_ERROR_UNSUPPORTED);
        let mut out = SENTINEL;
        assert_eq!(c_abi::TAPP_attr_get(0, 7, &mut out), TAPP_ERROR_UNSUPPORTED);
        assert!(out.is_null(), "attr_get left the out-parameter unwritten");
        // A null out-parameter must not be dereferenced.
        assert_eq!(
            c_abi::TAPP_attr_get(0, 7, std::ptr::null_mut()),
            TAPP_ERROR_UNSUPPORTED
        );
    }
}

// -------------------------------------------------------- linkage and symbols

/// A complete contraction driven only through the transcribed header: create
/// handle, executor and four tensor infos, build a plan, execute it, tear
/// everything down, and check the arithmetic.
///
/// If any `#[no_mangle]` in `src/lib.rs` were renamed or removed, this file
/// would not link. If any signature drifted from the header, this test would
/// pass garbage and fail on the numbers.
#[test]
fn every_documented_symbol_resolves_and_runs() {
    // D[i,j] = A[i,k] B[k,j] with A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    // column-major, so D = [[19,22],[43,50]].
    let ea = [2i64, 2];
    let sa = [1i64, 2];
    let eb = [2i64, 2];
    let sb = [1i64, 2];
    let ed = [2i64, 2];
    let sd = [1i64, 2];
    let la = [0i64, 2];
    let lb = [2i64, 1];
    let ld = [0i64, 1];
    let av = [1.0f64, 3.0, 2.0, 4.0];
    let bv = [5.0f64, 7.0, 6.0, 8.0];
    let want = [19.0f64, 43.0, 22.0, 50.0];

    unsafe {
        let mut handle = 0isize;
        assert_eq!(c_abi::TAPP_create_handle(&mut handle), 0);
        let mut exec = 0isize;
        assert_eq!(c_abi::TAPP_create_executor(&mut exec), 0);

        let mut ia = 0isize;
        let mut ib = 0isize;
        let mut ic = 0isize;
        let mut id = 0isize;
        for (h, e, s) in [
            (&mut ia, &ea, &sa),
            (&mut ib, &eb, &sb),
            (&mut ic, &ed, &sd),
            (&mut id, &ed, &sd),
        ] {
            // TAPP_F64 == 1, from datatype.h.
            assert_eq!(
                c_abi::TAPP_create_tensor_info(h, 1, 2, e.as_ptr(), s.as_ptr()),
                0
            );
        }
        assert_eq!(c_abi::TAPP_get_nmodes(ia), 2);

        // Exercise the mutators on a spare info before using it: set it to
        // something wrong, read it back, then set it right again.
        let bogus = [3i64, 3];
        let bogus_s = [1i64, 3];
        assert_eq!(c_abi::TAPP_set_extents(ic, bogus.as_ptr()), 0);
        assert_eq!(c_abi::TAPP_set_strides(ic, bogus_s.as_ptr()), 0);
        let mut got_e = [0i64; 2];
        let mut got_s = [0i64; 2];
        c_abi::TAPP_get_extents(ic, got_e.as_mut_ptr());
        c_abi::TAPP_get_strides(ic, got_s.as_mut_ptr());
        assert_eq!(got_e, bogus);
        assert_eq!(got_s, bogus_s);
        assert_eq!(c_abi::TAPP_set_nmodes(ic, 2), 0);
        assert_eq!(c_abi::TAPP_set_extents(ic, ed.as_ptr()), 0);
        assert_eq!(c_abi::TAPP_set_strides(ic, sd.as_ptr()), 0);

        let mut plan = 0isize;
        // TAPP_IDENTITY == 0, from product.h.
        let e = c_abi::TAPP_create_tensor_product(
            &mut plan,
            handle,
            0,
            ia,
            la.as_ptr(),
            0,
            ib,
            lb.as_ptr(),
            0,
            ic,
            ld.as_ptr(),
            0,
            id,
            ld.as_ptr(),
            TAPP_DEFAULT_PREC,
        );
        assert_eq!(e, 0, "{}", explain_status(e));
        assert!(c_abi::TAPP_check_success(e));

        let alpha = 1.0f64;
        let beta = 0.0f64;
        let mut dv = [0.0f64; 4];
        let mut status = 0isize;
        let e = c_abi::TAPP_execute_product(
            plan,
            exec,
            &mut status,
            &alpha as *const f64 as *const c_void,
            av.as_ptr() as *const c_void,
            bv.as_ptr() as *const c_void,
            &beta as *const f64 as *const c_void,
            std::ptr::null(),
            dv.as_mut_ptr() as *mut c_void,
        );
        assert_eq!(e, 0, "{}", explain_status(e));
        assert_eq!(dv, want);

        // The batched entry point over one batch, into a second buffer.
        let mut dv2 = [0.0f64; 4];
        let ap = [av.as_ptr() as *const c_void];
        let bp = [bv.as_ptr() as *const c_void];
        let mut dp = [dv2.as_mut_ptr() as *mut c_void];
        let e = c_abi::TAPP_execute_batched_product(
            plan,
            exec,
            &mut status,
            1,
            &alpha as *const f64 as *const c_void,
            ap.as_ptr(),
            bp.as_ptr(),
            &beta as *const f64 as *const c_void,
            std::ptr::null(),
            dp.as_mut_ptr(),
        );
        assert_eq!(e, 0, "{}", explain_status(e));
        assert_eq!(dv2, want);

        assert_eq!(c_abi::TAPP_destroy_status(status), 0);
        assert_eq!(c_abi::TAPP_destroy_tensor_product(plan), 0);
        for i in [ia, ib, ic, id] {
            assert_eq!(c_abi::TAPP_destroy_tensor_info(i), 0);
        }
        assert_eq!(c_abi::TAPP_destroy_executor(exec), 0);
        assert_eq!(c_abi::TAPP_destroy_handle(handle), 0);

        // The remaining three symbols.
        let mut buf: [c_char; 64] = [0; 64];
        assert!(c_abi::TAPP_explain_error(3, buf.len(), buf.as_mut_ptr()) > 0);
        let name = c_abi::TAPP_implementation_name();
        assert!(!name.is_null());
        assert!(CStr::from_ptr(name).to_str().is_ok());
        let version = c_abi::TAPP_implementation_version();
        assert!(!version.is_null());
        assert!(CStr::from_ptr(version).to_str().is_ok());
    }
}

// ------------------------------------------------------------------- versioning

/// The library reports its own version: `TAPP_implementation_version()` is the
/// crate version and `tcontract_abi_version()` is the same number as
/// `major * 10000 + minor * 100 + patch`.
///
/// (The pinned upstream `tapp.h` carries no `TAPP_VERSION_*` macros; the second
/// header that did, and the test that compared them with `Cargo.toml`, were
/// removed with it.)
#[test]
fn the_library_reports_its_crate_version() {
    let reported = unsafe { CStr::from_ptr(c_abi::TAPP_implementation_version()) }
        .to_str()
        .expect("the version string is not UTF-8");
    assert_eq!(reported, env!("CARGO_PKG_VERSION"));
    let parts: Vec<u32> = reported.split('.').map(|x| x.parse().unwrap()).collect();
    assert_eq!(
        tcontract_capi::tcontract_abi_version(),
        parts[0] * 10000 + parts[1] * 100 + parts[2]
    );
}

/// The C symbols and the Rust paths are the same functions, not two
/// implementations that happen to agree.
#[test]
fn the_c_symbols_are_the_crate_functions() {
    assert_eq!(
        c_abi::TAPP_implementation_name as *const () as usize,
        tcontract_capi::TAPP_implementation_name as *const () as usize
    );
    assert_eq!(
        c_abi::TAPP_check_success as *const () as usize,
        tcontract_capi::TAPP_check_success as *const () as usize
    );
    assert_eq!(
        c_abi::TAPP_execute_product as *const () as usize,
        tcontract_capi::TAPP_execute_product as *const () as usize
    );
}

// ------------------------------------------------------------ enumerator values

/// Every enumerator the crate exports, against the numbers in the upstream
/// headers.
///
/// `api/include/tapp/datatype.h`: `TAPP_F32 = 0`, `TAPP_F64 = 1`, `TAPP_C32 =
/// 2`, `TAPP_C64 = 3`, `TAPP_F16 = 4`, `TAPP_BF16 = 5`, and
/// `TAPP_DEFAULT_PREC = -1`. `api/include/tapp/product.h`: `TAPP_IDENTITY = 0`,
/// `TAPP_CONJUGATE = 1`.
///
/// This matters more than a normal constant test: the project has already been
/// bitten by two libraries that swapped two enumerators between releases (TBLIS
/// 1.3 and 2.0 exchange `TYPE_DOUBLE` and `TYPE_SCOMPLEX`), which produces
/// plausible wrong numbers and no error at all. A datatype tag is exactly the
/// kind of constant that must be pinned rather than assumed.
#[test]
fn enumerator_values_match_the_upstream_headers() {
    use tcontract_capi::*;
    assert_eq!(TAPP_F32, 0);
    assert_eq!(TAPP_F64, 1);
    assert_eq!(TAPP_C32, 2);
    assert_eq!(TAPP_C64, 3);
    assert_eq!(TAPP_F16, 4);
    assert_eq!(TAPP_BF16, 5);
    assert_eq!(TAPP_IDENTITY, 0);
    assert_eq!(TAPP_CONJUGATE, 1);
    assert_eq!(TAPP_DEFAULT_PREC, -1);
    assert_eq!(TAPP_F32F32_ACCUM_F32, 0);
    assert_eq!(TAPP_F64F64_ACCUM_F64, 1);
    // `TAPP_error` is only specified at zero: upstream `error.h` declares
    // `typedef int TAPP_error` and no enumerators, so the non-zero codes below
    // are this implementation's own and a portable caller must go through
    // `TAPP_check_success`. They are the `tcontract_status` codes (originally the
    // `tprims_status` codes of the removed `tprims-core` crate; one table for
    // every part of libtcontract_capi), pinned here so they stay stable
    // for callers that do depend on them.
    assert_eq!(TAPP_SUCCESS, 0);
    assert_eq!(TAPP_ERROR_NULL, 1);
    assert_eq!(TAPP_ERROR_SHAPE, 2);
    assert_eq!(TAPP_ERROR_DATATYPE, 3);
    assert_eq!(TAPP_ERROR_ALIASED, 6);
    assert_eq!(TAPP_ERROR_BUSY, 9);
    assert_eq!(TAPP_ERROR_WOULD_DEADLOCK, 10);
    assert_eq!(TAPP_ERROR_PANIC, 12);
    assert_eq!(TAPP_ERROR_INTERNAL, 13);
    assert_eq!(TAPP_ERROR_LABELS, 14);
    assert_eq!(TAPP_ERROR_UNSUPPORTED, 15);
}

// ------------------------------------------------------------------- data layout

/// `TAPP_C32` and `TAPP_C64` are interleaved, real part first, and
/// layout-identical to C99 `float _Complex` / `double _Complex` and to
/// `num_complex::Complex`.
///
/// The whole complex side of the ABI is a reinterpret cast — a C caller's
/// `double _Complex*` becomes a `*const Complex<f64>` with no conversion — so
/// this is the assumption everything else rests on.
#[test]
fn complex_datatypes_are_interleaved_and_c_compatible() {
    assert_eq!(size_of::<Complex<f32>>(), 2 * size_of::<f32>());
    assert_eq!(size_of::<Complex<f64>>(), 2 * size_of::<f64>());
    assert_eq!(align_of::<Complex<f32>>(), align_of::<f32>());
    assert_eq!(align_of::<Complex<f64>>(), align_of::<f64>());

    // Field offsets, which is what "interleaved, real first" actually means.
    assert_eq!(offset_of!(Complex<f32>, re), 0);
    assert_eq!(offset_of!(Complex<f32>, im), 4);
    assert_eq!(offset_of!(Complex<f64>, re), 0);
    assert_eq!(offset_of!(Complex<f64>, im), 8);

    // A C caller's array of `double _Complex` is an array of pairs of doubles;
    // reading it as `Complex<f64>` must recover the same numbers in the same
    // order, and writing it back must be byte-identical.
    let raw = [1.0f64, -2.0, 3.0, -4.0];
    let zs: &[Complex<f64>] =
        unsafe { slice::from_raw_parts(raw.as_ptr() as *const Complex<f64>, 2) };
    assert_eq!(zs, &[Complex::new(1.0, -2.0), Complex::new(3.0, -4.0)]);
    let back: &[f64] = unsafe { slice::from_raw_parts(zs.as_ptr() as *const f64, 4) };
    assert_eq!(back, raw);
}

/// The scalar types that cross the boundary are the ones the header names.
///
/// TAPP passes no aggregates by value at all: every argument is a scalar, an
/// `intptr_t` handle, or a pointer to caller-owned memory, so there is no struct
/// whose field offsets could disagree with a C compiler's. What has to match is
/// the width of the handles, of `int`, and of the `int64_t` shape and label
/// arrays.
#[test]
fn boundary_scalar_types_have_the_widths_the_header_names() {
    // `intptr_t`: handles must round-trip a pointer without losing bits.
    assert_eq!(size_of::<isize>(), size_of::<*mut c_void>());
    assert_eq!(align_of::<isize>(), align_of::<*mut c_void>());
    // `int`, the type of `TAPP_error`, `TAPP_datatype`, `TAPP_prectype`,
    // `TAPP_element_op` and `nmode`.
    assert_eq!(size_of::<c_int>(), 4);
    // `int64_t`, the type of every extent, stride and index label.
    assert_eq!(size_of::<i64>(), 8);
    // `bool`, the return type of `TAPP_check_success`, is one byte in both
    // Rust and C23 / `<stdbool.h>`.
    assert_eq!(size_of::<bool>(), 1);
    // `size_t`, `TAPP_explain_error`'s `maxlen` and return type.
    assert_eq!(size_of::<usize>(), size_of::<*const c_void>());
}

/// The handle types that carry state are distinct, non-zero, pointer-aligned
/// live allocations, so a C caller can hold several at once and tell them apart.
#[test]
fn stateful_handles_are_distinct_nonzero_and_aligned() {
    unsafe {
        // Executors carry a thread count, tensor infos a layout, so both are
        // real allocations.
        let mut xs = [0isize; 4];
        for x in xs.iter_mut() {
            assert_eq!(c_abi::TAPP_create_executor(x), 0);
            assert_ne!(*x, 0, "zero is the ABI's invalid-handle value");
            assert_eq!(
                (*x as usize) % align_of::<usize>(),
                0,
                "a handle is a cast pointer and must stay aligned"
            );
        }
        for (i, &x) in xs.iter().enumerate() {
            for &y in &xs[i + 1..] {
                assert_ne!(x, y, "two live executors must be distinguishable");
            }
        }
        for x in xs {
            assert_eq!(c_abi::TAPP_destroy_executor(x), 0);
        }

        let e = [4i64, 3];
        let s = [1i64, 4];
        let mut is = [0isize; 4];
        for i in is.iter_mut() {
            assert_eq!(
                c_abi::TAPP_create_tensor_info(i, 3, 2, e.as_ptr(), s.as_ptr()),
                0
            );
            assert_ne!(*i, 0);
            assert_eq!((*i as usize) % align_of::<usize>(), 0);
        }
        for (i, &x) in is.iter().enumerate() {
            for &y in &is[i + 1..] {
                assert_ne!(x, y, "two live tensor infos must be distinguishable");
            }
        }
        for i in is {
            assert_eq!(c_abi::TAPP_destroy_tensor_info(i), 0);
        }
    }
}

/// Library handles are distinct, non-zero, and destroyable independently.
///
/// This test used to assert the opposite, and finding that out is what changed
/// the code. `HandleState` was `struct HandleState { _private: () }` — a
/// zero-sized type, for which `Box::into_raw` returns `NonNull::dangling()`, so
/// every library handle the crate ever issued was the value `1`. Two live
/// handles were indistinguishable, and a C program that created two and
/// destroyed both was double-freeing — harmlessly, but only for as long as the
/// state stayed empty. `src/lib.rs` then gained a reserved field.
///
/// What is still *not* true, and the old comment in `src/lib.rs` claimed it, is
/// that handle validity can be checked: nothing distinguishes a pointer this
/// crate produced from an arbitrary non-zero `intptr_t`, so `0` remains the only
/// value the ABI can reject.
#[test]
fn library_handles_are_distinct_and_independently_destroyable() {
    unsafe {
        let mut h1 = 0isize;
        let mut h2 = 0isize;
        assert_eq!(c_abi::TAPP_create_handle(&mut h1), 0);
        assert_eq!(c_abi::TAPP_create_handle(&mut h2), 0);
        assert_ne!(h1, 0);
        assert_ne!(h2, 0);
        assert_ne!(
            h1, h2,
            "library handles have collapsed to one value: is HandleState zero-sized again?"
        );
        assert_eq!(
            h1 % (std::mem::align_of::<u64>() as isize),
            0,
            "unaligned handle"
        );
        // Each is a real allocation, so each can and must be destroyed once.
        assert_eq!(c_abi::TAPP_destroy_handle(h1), 0);
        assert_eq!(c_abi::TAPP_destroy_handle(h2), 0);
    }
}
