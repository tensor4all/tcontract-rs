//! Executing a product: pointer adaptation for a single call and for a batch.
//! Null handling is the ABI's; the semantic preflight (output overlap, `C`/`D`
//! aliasing) is the plan's own `check_raw`, and the batch validates every item
//! before running them in sequence.

use std::ffi::c_void;
use std::os::raw::c_int;

use tcontract::api;
use tcontract::{OutputContract, Plan};

use crate::abi::*;
use crate::executor::with_executor;
use crate::product::{Prepared, Product};
use crate::status::{ffi, FfiError, TCONTRACT_ERR_SHAPE, TCONTRACT_ERR_UNSUPPORTED};

/// Data pointers of one execution, as the caller passed them.
#[derive(Clone, Copy)]
struct Item {
    a: *const c_void,
    b: *const c_void,
    c: *const c_void,
    d: *mut c_void,
}

/// Everything an execution checks before it writes anything, the same for a
/// single call and for each item of a batch: the FFI-only null handling here,
/// then the plan's own semantic preflight (output overlap and `C`/`D`
/// aliasing), which is the one definition the safe API shares.
fn validate<T: api::Scalar>(plan: &Plan<T>, it: Item, beta_is_zero: bool) -> Result<(), FfiError> {
    if it.a.is_null() || it.b.is_null() || it.d.is_null() {
        return Err(null("A, B or D data pointer"));
    }
    if it.c.is_null() && !beta_is_zero {
        return Err(fail(
            TCONTRACT_ERR_UNSUPPORTED,
            "a null C (TAPP_IN_PLACE) with a nonzero beta is ambiguous in TAPP and not supported; pass D as C to accumulate in place",
        ));
    }
    plan.check_raw(
        it.a as *const T,
        it.b as *const T,
        it.c as *const T,
        it.d as *const T,
    )
    .map_err(map_err)
}

/// Run one validated item on the executor's threads. The plan chooses the actual
/// width from its work estimate and the executor's budget.
///
/// # Safety
///
/// The pointers address every offset of this plan's operands, and `validate`
/// accepted `it`.
unsafe fn run<T: api::Scalar>(
    plan: &Plan<T>,
    exec: &tcontract_exec::Exec<'_>,
    alpha: T,
    beta: T,
    it: Item,
    zero: T,
) -> Result<(), FfiError> {
    let beta = if it.c.is_null() { zero } else { beta };
    // SAFETY: forwarded; the executor lends the threads, none are spawned.
    unsafe {
        plan.execute_raw(
            exec,
            alpha,
            it.a as *const T,
            it.b as *const T,
            beta,
            it.c as *const T,
            it.d as *mut T,
        )
    }
    .map_err(map_err)
}

/// Read a scalar of the plan's element type.
///
/// # Safety
/// `p` is null or points to a `T` (possibly unaligned).
unsafe fn scalar<T: Copy>(p: *const c_void, name: &str) -> Result<T, FfiError> {
    if p.is_null() {
        return Err(null(name));
    }
    // SAFETY: non-null and a `T` per the contract.
    Ok(unsafe { std::ptr::read_unaligned(p as *const T) })
}

/// Execute `items` (validated first, as a whole, before any is written).
///
/// # Safety
/// As [`TAPP_execute_product`], for each item.
unsafe fn execute_items<T>(
    plan: &Plan<T>,
    exec: isize,
    alpha: *const c_void,
    beta: *const c_void,
    item: &dyn Fn(usize) -> Item,
    count: usize,
) -> Result<(), FfiError>
where
    T: api::Scalar + Default,
{
    let zero = T::default();
    // SAFETY: `alpha` and `beta` are `T`s per the contract.
    let (al, be) = unsafe { (scalar::<T>(alpha, "alpha")?, scalar::<T>(beta, "beta")?) };
    // Validate every item before the first write; nothing is allocated, so a
    // single product costs no more than it did before batches existed.
    for i in 0..count {
        validate(plan, item(i), be == zero)?;
    }
    // SAFETY: `exec` is zero or a live executor per the contract.
    unsafe {
        with_executor(exec, |x| {
            // INVARIANT: C plans prepare Separate C and fresh_output. This is
            // execute_raw's storage classification after the alias preflight.
            // Resolve every item on the actual caller before the first write.
            for i in 0..count {
                let it = item(i);
                let output = if be == zero || !std::ptr::eq(it.c, it.d.cast_const()) {
                    OutputContract::Fresh
                } else {
                    OutputContract::Initialized
                };
                plan.execution_route(x, al, output).map_err(map_err)?;
            }
            for i in 0..count {
                // SAFETY: validated above; the caller's pointer contract.
                run(plan, x, al, be, item(i), zero)?;
            }
            Ok(())
        })
    }
}

/// Dispatch on the plan's storage type.
///
/// # Safety
/// As [`execute_items`].
unsafe fn dispatch(
    p: &Product,
    exec: isize,
    alpha: *const c_void,
    beta: *const c_void,
    item: &dyn Fn(usize) -> Item,
    count: usize,
) -> Result<(), FfiError> {
    // SAFETY: forwarded.
    unsafe {
        match &p.plan {
            Prepared::F32(plan) => execute_items(plan, exec, alpha, beta, item, count),
            Prepared::F64(plan) => execute_items(plan, exec, alpha, beta, item, count),
            Prepared::C32(plan) => execute_items(plan, exec, alpha, beta, item, count),
            Prepared::C64(plan) => execute_items(plan, exec, alpha, beta, item, count),
        }
    }
}

/// Execute a planned product against data:
/// `D = op_D(alpha * op_A(A) * op_B(B) + beta * op_C(C))`.
///
/// Synchronous: it returns when the contraction is done. `exec` is zero (the
/// default serial executor), or an executor of this library; the call uses at
/// most its budget (read once, here), runs small work on the calling thread
/// without entering the pool, and runs wider work on the executor's pool through
/// `Exec::broadcast` — never through an environment variable or a pool of its own.
/// `alpha` and `beta` are read as the plan's element type, so they are pointers
/// to an `f32`, `f64`, `float _Complex` or `double _Complex` accordingly.
///
/// `status`, if non-null, is set to `0` before the work starts. Upstream's own
/// reference implementation leaves it untouched, and `status.h` declares no
/// `TAPP_create_status`, so a caller following the idiomatic
/// `TAPP_status s; execute(.., &s, ..); TAPP_destroy_status(s);` would otherwise
/// pass an uninitialised value to the destructor. Writing zero costs a branch
/// and makes that sequence defined.
///
/// `C` and `D` may be separate buffers, of different layouts. `C == D` is an
/// in-place update, accepted only when `C` maps every element exactly as `D`
/// does once the labels are matched (`TAPP_ERROR_ALIASED` otherwise, even with
/// equal base pointers); any other overlap between `C` and `D` is rejected as
/// partial, and so is any overlap of `D` with `A` or `B`. Overlap is judged on
/// the byte range each operand addresses, which is conservative for
/// interleaved layouts. With `beta == 0`, `C` is not read.
///
/// `c` may be null — upstream's `TAPP_IN_PLACE` — **only together with
/// `beta == 0`**, meaning `D` is overwritten. A null `c` with a non-zero `beta`
/// returns [`TAPP_ERROR_UNSUPPORTED`] rather than silently discarding `D`,
/// because `product.h` leaves the meaning of that combination an open question.
/// In-place accumulation is expressible: pass `D`'s own pointer as `c`.
///
/// Every check runs before `D` is written.
///
/// # Safety
/// `plan` must be live and `exec` zero or live, not destroyed during the call.
/// `alpha` and `beta` must each be valid for one read of the plan's element
/// type. The raw TAPP ABI carries no allocation lengths: `a` and `b` must be
/// readable, and `d` writable, at every offset the plan's extents and strides
/// generate (and `c` readable unless it is null), which this cannot check.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn TAPP_execute_product(
    plan: isize,
    exec: isize,
    status: *mut isize,
    alpha: *const c_void,
    a: *const c_void,
    b: *const c_void,
    beta: *const c_void,
    c: *const c_void,
    d: *mut c_void,
) -> c_int {
    ffi(move || {
        if !status.is_null() {
            // SAFETY: non-null and writable per the contract.
            unsafe { *status = 0 };
        }
        // SAFETY: zero or live per the contract.
        let p = unsafe { (plan as *const Product).as_ref() }.ok_or_else(|| null("plan"))?;
        let it = Item { a, b, c, d };
        // SAFETY: forwarded contract.
        unsafe { dispatch(p, exec, alpha, beta, &|_| it, 1) }
    })
}

/// Execute the same plan against `num_batches` sets of data pointers.
///
/// Distinct from a Hadamard (batch) index inside the plan, which the engine's
/// own loop nest handles and which shares packed panels; this shares only the
/// plan and the executor. The pointer arrays are validated first and every item
/// is checked with the same rules as [`TAPP_execute_product`] before any `D` is
/// written; an error found in an item's *data* during execution (none are
/// currently possible past validation) would leave earlier items written — there
/// is no whole-batch rollback. One `alpha` and one `beta` apply to every item;
/// `c` may be null, meaning every item has a null `C` (`beta` must be zero).
/// Items run one after another, each on the executor's threads.
///
/// # Safety
/// `plan` must be live and `num_batches` non-negative. `a`, `b` and `d` must
/// each be valid for `num_batches` pointer reads, and `c` likewise unless it is
/// null. Every pointer so obtained must satisfy [`TAPP_execute_product`]'s
/// obligations.
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn TAPP_execute_batched_product(
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
) -> c_int {
    ffi(move || {
        if !status.is_null() {
            // SAFETY: non-null and writable per the contract.
            unsafe { *status = 0 };
        }
        if num_batches < 0 {
            return Err(fail(TCONTRACT_ERR_SHAPE, "negative batch count"));
        }
        // SAFETY: zero or live per the contract.
        let p = unsafe { (plan as *const Product).as_ref() }.ok_or_else(|| null("plan"))?;
        if a.is_null() || b.is_null() || d.is_null() {
            return Err(null("A, B or D pointer array"));
        }
        let n = num_batches as usize;
        let item = |i: usize| {
            // SAFETY: `n` pointer reads per array per the contract.
            unsafe {
                Item {
                    a: *a.add(i),
                    b: *b.add(i),
                    c: if c.is_null() {
                        std::ptr::null()
                    } else {
                        *c.add(i)
                    },
                    d: *d.add(i),
                }
            }
        };
        // SAFETY: forwarded contract.
        unsafe { dispatch(p, exec, alpha, beta, &item, n) }
    })
}
