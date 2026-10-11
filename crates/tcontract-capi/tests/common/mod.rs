//! Scaffolding shared by the TAPP C-ABI conformance suite.
//!
//! Everything here drives `libtcontract_capi` the way a C caller does:
//! `intptr_t` handles, `int64_t*` extent / stride / label arrays, `void*` data
//! and scalars, an `int` status back, and one `TAPP_*` call per step. Nothing
//! reaches for the `tcontract_capi::Plan` API behind the shims, deliberately —
//! the bugs this crate can have are exactly the ones invisible from there: a
//! datatype tag dispatched to the wrong element type, a label array read at the
//! wrong length, a `beta` honoured on the wrong operand, a handle cast that
//! loses a bit.
//!
//! The numerical oracle is `tcontract_testkit::oracle::contract_reference`,
//! which shares no code with the fast path — it walks the raw layouts and label
//! lists and so independently defines what a diagonal, a reduction, a Hadamard
//! index and a conjugation mean. Where a shape is small enough to check by hand,
//! the tests do that instead, so the suite does not rest on the oracle alone.
#![allow(dead_code)]
#![allow(non_snake_case)]

use std::ffi::{c_void, CStr};
use std::os::raw::{c_char, c_int};

use tcontract_capi::*;
use tcontract_kernel::{Element, Real};
use tcontract_testkit::oracle::{contract_reference, RefOperand, RefOutput};

// ----------------------------------------------------------------- datatypes

/// A storage element type together with the `TAPP_datatype` tag that names it.
pub trait Dtype: Element {
    /// The `TAPP_datatype` enumerator.
    const TAG: c_int;
    const NAME: &'static str;
}

impl Dtype for f32 {
    const TAG: c_int = TAPP_F32;
    const NAME: &'static str = "f32";
}
impl Dtype for f64 {
    const TAG: c_int = TAPP_F64;
    const NAME: &'static str = "f64";
}
impl Dtype for num_complex::Complex<f32> {
    const TAG: c_int = TAPP_C32;
    const NAME: &'static str = "c32";
}
impl Dtype for num_complex::Complex<f64> {
    const TAG: c_int = TAPP_C64;
    const NAME: &'static str = "c64";
}

/// Relative-error budget for a whole output buffer, matching the engine's own
/// suite. The TAPP path always runs the default (planar) complex method, whose
/// error bound is the ordinary one; 3m, whose bound is relative to
/// `|Ar||Br| + |Ai||Bi|`, is not reachable through the C ABI.
pub fn tol<T: Dtype>() -> f64 {
    if core::mem::size_of::<T::Real>() == 4 {
        2e-4
    } else {
        1e-11
    }
}

/// A deterministic value sequence.
///
/// Not random: `tcontract`'s own suite owns the randomised sweep, and here
/// a failure should be reproducible from the test name alone. Values are
/// multiples of `1/2048` in `[-1, 1)`, so they are exact in `f32` as well as
/// `f64` and a single-precision failure is arithmetic rather than rounding of
/// the inputs.
pub fn seq<T: Dtype>(n: usize, seed: u64) -> Vec<T> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            let re = next_unit(&mut s);
            let im = if T::IS_COMPLEX {
                next_unit(&mut s)
            } else {
                0.0
            };
            T::from_parts(T::Real::from_f64(re), T::Real::from_f64(im))
        })
        .collect()
}

fn next_unit(s: &mut u64) -> f64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    ((*s >> 52) as f64) / 2048.0 - 1.0
}

/// `re + i*im` in whichever element type is under test; the imaginary part is
/// dropped for the real ones.
pub fn scalar<T: Dtype>(re: f64, im: f64) -> T {
    let im = if T::IS_COMPLEX { im } else { 0.0 };
    T::from_parts(T::Real::from_f64(re), T::Real::from_f64(im))
}

/// A buffer of quiet NaNs, used to prove an operand is never read.
pub fn poison<T: Dtype>(n: usize) -> Vec<T> {
    vec![scalar::<T>(f64::NAN, f64::NAN); n]
}

// ------------------------------------------------------------------- tensors

/// A tensor as a C caller describes it: the arguments of
/// `TAPP_create_tensor_info` plus the `idx_*` / `op_*` pair that
/// `TAPP_create_tensor_product` takes alongside the resulting handle.
///
/// Strides are always written out rather than derived, so a test's memory
/// layout is visible in the test.
#[derive(Clone, Copy)]
pub struct CTensor<'a> {
    pub extents: &'a [i64],
    pub strides: &'a [i64],
    pub labels: &'a [i64],
    pub op: c_int,
}

impl<'a> CTensor<'a> {
    pub fn new(extents: &'a [i64], strides: &'a [i64], labels: &'a [i64]) -> Self {
        assert_eq!(extents.len(), strides.len(), "rank mismatch in the test");
        assert_eq!(
            extents.len(),
            labels.len(),
            "label count mismatch in the test"
        );
        CTensor {
            extents,
            strides,
            labels,
            op: TAPP_IDENTITY,
        }
    }

    /// Apply `TAPP_CONJUGATE` to this operand.
    pub fn conj(mut self) -> Self {
        self.op = TAPP_CONJUGATE;
        self
    }

    pub fn with_op(mut self, op: c_int) -> Self {
        self.op = op;
        self
    }

    pub fn nmode(&self) -> c_int {
        self.extents.len() as c_int
    }

    /// Extents as `usize` (the oracle's vocabulary).
    pub fn dims(&self) -> Vec<usize> {
        self.extents.iter().map(|&e| e as usize).collect()
    }

    /// Strides as `isize`.
    pub fn strides_isize(&self) -> Vec<isize> {
        self.strides.iter().map(|&s| s as isize).collect()
    }

    /// Smallest buffer this layout can be read through, base pointer at zero.
    pub fn storage(&self) -> usize {
        let mut n = 1i64;
        for (&e, &s) in self.extents.iter().zip(self.strides) {
            if e > 0 {
                n += (e - 1) * s.abs();
            }
        }
        n.max(1) as usize
    }

    /// `NULL` for a rank-zero tensor, which is what a C caller passes and what
    /// the ABI must accept.
    pub fn extents_ptr(&self) -> *const i64 {
        if self.extents.is_empty() {
            std::ptr::null()
        } else {
            self.extents.as_ptr()
        }
    }

    pub fn strides_ptr(&self) -> *const i64 {
        if self.strides.is_empty() {
            std::ptr::null()
        } else {
            self.strides.as_ptr()
        }
    }

    pub fn labels_ptr(&self) -> *const i64 {
        if self.labels.is_empty() {
            std::ptr::null()
        } else {
            self.labels.as_ptr()
        }
    }

    /// `TAPP_create_tensor_info`, asserting success.
    pub unsafe fn create_info(&self, dtype: c_int) -> isize {
        let mut info = 0isize;
        let e = TAPP_create_tensor_info(
            &mut info,
            dtype,
            self.nmode(),
            self.extents_ptr(),
            self.strides_ptr(),
        );
        assert_eq!(
            e,
            TAPP_SUCCESS,
            "TAPP_create_tensor_info: {}",
            explain_status(e)
        );
        assert_ne!(info, 0, "a successful create must produce a live handle");
        info
    }
}

// ------------------------------------------------------------------- the `C`
//                                                                    operand

/// Where `TAPP_execute_product`'s `C` pointer comes from.
///
/// TAPP always wants four tensor *infos*, so a contraction with no separate `C`
/// still supplies `D`'s descriptor; what varies is the data pointer.
pub enum CData<'a, T> {
    /// A distinct `C` buffer.
    Buf(&'a [T]),
    /// `C` and `D` are the same array — accumulate in place.
    SameAsD,
    /// `C = NULL`, which upstream `tapp/product.h` spells `TAPP_IN_PLACE`. See
    /// `null_c_pointer_is_treated_as_beta_zero` for what this crate does with
    /// it, which is not what that name suggests.
    Null,
}

// Hand-written rather than derived, because `derive` would add a `T: Copy`
// bound this enum does not need: it only ever holds a shared slice.
impl<T> Clone for CData<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for CData<'_, T> {}

// --------------------------------------------------------------------- cases

/// One contraction, described the way the C ABI describes it.
pub struct Case<'a, T> {
    pub alpha: T,
    pub beta: T,
    pub a: CTensor<'a>,
    pub av: &'a [T],
    pub b: CTensor<'a>,
    pub bv: &'a [T],
    /// `C`'s descriptor; equal to `d` when there is no separate `C`.
    pub c: CTensor<'a>,
    pub cv: CData<'a, T>,
    pub d: CTensor<'a>,
    pub prec: c_int,
}

impl<'a, T: Dtype> Case<'a, T> {
    /// `D = A B`: no `C`, `alpha = 1`, `beta = 0`, default precision.
    pub fn plain(a: CTensor<'a>, av: &'a [T], b: CTensor<'a>, bv: &'a [T], d: CTensor<'a>) -> Self {
        Case {
            alpha: T::one(),
            beta: T::zero(),
            a,
            av,
            b,
            bv,
            c: d,
            cv: CData::Null,
            d,
            prec: TAPP_DEFAULT_PREC,
        }
    }

    /// Add a distinct `C` operand with `D`'s shape.
    pub fn with_c(mut self, cv: &'a [T], beta: T) -> Self {
        self.c = self.d;
        self.cv = CData::Buf(cv);
        self.beta = beta;
        self
    }

    /// Accumulate into `D`, i.e. `C` and `D` are the same array.
    pub fn accumulating(mut self, beta: T) -> Self {
        self.c = self.d;
        self.cv = CData::SameAsD;
        self.beta = beta;
        self
    }

    pub fn with_alpha(mut self, alpha: T) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_prec(mut self, prec: c_int) -> Self {
        self.prec = prec;
        self
    }

    /// A `D` buffer of the right size, zero-filled.
    pub fn zeroed_d(&self) -> Vec<T> {
        vec![T::zero(); self.d.storage()]
    }

    /// `TAPP_create_tensor_product` from four freshly created infos, which are
    /// then destroyed.
    ///
    /// Destroying them before the plan is ever executed is deliberate and it
    /// happens on every case in the suite: TAPP's tensor infos are
    /// *descriptors*, and a C caller is entitled to free them the moment the
    /// plan exists. A plan that borrowed them would fail here rather than in
    /// somebody else's program.
    pub unsafe fn create_plan(&self, handle: isize) -> (c_int, isize) {
        let ia = self.a.create_info(T::TAG);
        let ib = self.b.create_info(T::TAG);
        let ic = self.c.create_info(T::TAG);
        let id = self.d.create_info(T::TAG);
        let mut plan = 0isize;
        let e = TAPP_create_tensor_product(
            &mut plan,
            handle,
            self.a.op,
            ia,
            self.a.labels_ptr(),
            self.b.op,
            ib,
            self.b.labels_ptr(),
            self.c.op,
            ic,
            self.c.labels_ptr(),
            self.d.op,
            id,
            self.d.labels_ptr(),
            self.prec,
        );
        for i in [ia, ib, ic, id] {
            assert_eq!(TAPP_destroy_tensor_info(i), TAPP_SUCCESS);
        }
        (e, plan)
    }

    /// Plan and execute through the C ABI. Returns the first non-success status,
    /// or `TAPP_SUCCESS`.
    pub unsafe fn run(&self, dv: &mut [T]) -> c_int {
        let handle = create_handle();
        let exec = create_executor();
        let (e, plan) = self.create_plan(handle);
        let out = if e != TAPP_SUCCESS {
            e
        } else {
            let dptr = dv.as_mut_ptr();
            let cptr: *const c_void = match self.cv {
                CData::Buf(s) => s.as_ptr() as *const c_void,
                CData::SameAsD => dptr as *const c_void,
                CData::Null => std::ptr::null(),
            };
            // The implementation never writes `*status`, so initialising it
            // here is what keeps the `TAPP_destroy_status` below defined. See
            // `execute_never_writes_the_status_out_parameter`.
            let mut status = 0isize;
            let e = TAPP_execute_product(
                plan,
                exec,
                &mut status,
                &self.alpha as *const T as *const c_void,
                self.av.as_ptr() as *const c_void,
                self.bv.as_ptr() as *const c_void,
                &self.beta as *const T as *const c_void,
                cptr,
                dptr as *mut c_void,
            );
            assert_eq!(TAPP_destroy_status(status), TAPP_SUCCESS);
            assert_eq!(TAPP_destroy_tensor_product(plan), TAPP_SUCCESS);
            e
        };
        assert_eq!(TAPP_destroy_executor(exec), TAPP_SUCCESS);
        assert_eq!(TAPP_destroy_handle(handle), TAPP_SUCCESS);
        out
    }

    /// The oracle's answer, given `D`'s contents before the call.
    pub fn expected(&self, d_initial: &[T]) -> Vec<T> {
        let (da, db, dc, dd) = (self.a.dims(), self.b.dims(), self.c.dims(), self.d.dims());
        let (sa, sb, sc, sd) = (
            self.a.strides_isize(),
            self.b.strides_isize(),
            self.c.strides_isize(),
            self.d.strides_isize(),
        );
        let mut out = d_initial.to_vec();
        let cdata: Option<&[T]> = match self.cv {
            CData::Buf(s) => Some(s),
            CData::SameAsD => Some(d_initial),
            CData::Null => None,
        };
        let cref = cdata.map(|data| RefOperand {
            data,
            dims: &dc,
            strides: &sc,
            offset: 0,
            labels: self.c.labels,
            conj: self.c.op == TAPP_CONJUGATE,
        });
        contract_reference::<T>(
            self.alpha,
            &RefOperand {
                data: self.av,
                dims: &da,
                strides: &sa,
                offset: 0,
                labels: self.a.labels,
                conj: self.a.op == TAPP_CONJUGATE,
            },
            &RefOperand {
                data: self.bv,
                dims: &db,
                strides: &sb,
                offset: 0,
                labels: self.b.labels,
                conj: self.b.op == TAPP_CONJUGATE,
            },
            self.beta,
            cref.as_ref(),
            &mut RefOutput {
                data: &mut out,
                dims: &dd,
                strides: &sd,
                offset: 0,
                labels: self.d.labels,
                conj: self.d.op == TAPP_CONJUGATE,
            },
        )
        .expect("the oracle rejected a case the C ABI accepted");
        out
    }

    /// Run it and compare the whole `D` buffer against the oracle.
    ///
    /// The comparison covers padding as well as addressed elements, so a write
    /// outside the layout is a failure and not a silent pass.
    pub fn check(&self, dv: &mut [T]) {
        let initial = dv.to_vec();
        let want = self.expected(&initial);
        let e = unsafe { self.run(dv) };
        assert_eq!(
            e,
            TAPP_SUCCESS,
            "TAPP_execute_product: {}",
            explain_status(e)
        );
        assert_close::<T>(dv, &want);
    }

    /// Run it against a zeroed `D` and return what came out.
    pub fn check_zeroed(&self) -> Vec<T> {
        let mut dv = self.zeroed_d();
        self.check(&mut dv);
        dv
    }
}

// ------------------------------------------------------------------- helpers

pub unsafe fn create_handle() -> isize {
    let mut h = 0isize;
    assert_eq!(TAPP_create_handle(&mut h), TAPP_SUCCESS);
    assert_ne!(h, 0);
    h
}

pub unsafe fn create_executor() -> isize {
    let mut x = 0isize;
    assert_eq!(TAPP_create_executor(&mut x), TAPP_SUCCESS);
    assert_ne!(x, 0);
    x
}

/// A status code plus the message the ABI itself gives for it, so a failing
/// assertion reports what a C caller would have printed.
pub fn explain_status(error: c_int) -> String {
    let mut buf: [c_char; 128] = [0; 128];
    let n = unsafe { TAPP_explain_error(error, buf.len(), buf.as_mut_ptr()) };
    let msg = unsafe { CStr::from_ptr(buf.as_ptr()) };
    format!(
        "status {error} \"{}\" (full length {n})",
        msg.to_string_lossy()
    )
}

pub fn assert_close<T: Dtype>(got: &[T], want: &[T]) {
    assert_eq!(got.len(), want.len(), "output buffer length changed");
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.norm().is_finite(),
            "{}: element {i} is not finite: {g:?}",
            T::NAME
        );
        num += g.sub(w).norm().powi(2);
        den += w.norm().powi(2);
    }
    let rel = if den == 0.0 {
        num.sqrt()
    } else {
        (num / den).sqrt()
    };
    let budget = tol::<T>();
    assert!(
        rel <= budget,
        "{}: relative error {rel:e} exceeds {budget:e}\n got: {got:?}\nwant: {want:?}",
        T::NAME,
    );
}

/// Assert `got == want` exactly, element for element. Used where the expected
/// value is exact in binary floating point.
pub fn assert_exact<T: Dtype>(got: &[T], want: &[T]) {
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g, w, "{}: element {i}", T::NAME);
    }
}
