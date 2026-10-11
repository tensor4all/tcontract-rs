//! The `Send`/`Sync` guarantees the threaded driver is built on.
//!
//! All of these hold by auto-derivation today, and that is the reason to pin
//! them: nothing in the crate *states* them, so one `Rc`, `Cell` or raw pointer
//! added to a private field of `Plan` would withdraw one silently. The
//! breakage would then surface as the executor refusing the closure in
//! `driver` or `batch` -- or, for `Error`, at a downstream caller whose
//! `Box<dyn Error + Send + Sync>` stops accepting it, which is not this crate's
//! test suite at all.
//!
//! Nothing here runs: the assertions are discharged by type checking, and the
//! `#[test]` exists only to give them a call site.

use crate::api::{Error, Problem};
use crate::plan::{PackedPlan, PlanConfig};
use crate::Plan;
use tcontract_kernel::{C32, C64};

fn assert_send<T: Send>() {}
fn assert_send_sync<T: Send + Sync>() {}
fn assert_send_sync_static<T: Send + Sync + 'static>() {}

#[test]
fn public_types_are_send_and_sync() {
    // A plan is built once and shared by reference across every worker, so it
    // must be `Sync`; the batch axis also moves plans between threads.
    assert_send_sync::<Plan<f32>>();
    assert_send_sync::<Plan<f64>>();
    assert_send_sync::<Plan<C32>>();
    assert_send_sync::<Plan<C64>>();
    assert_send_sync::<PackedPlan>();
    assert_send_sync::<Problem>();
    assert_send_sync::<PlanConfig>();

    // The batch item is handed to exactly one thread, so `Send` is the whole
    // requirement; its output view is an exclusive borrow.
    assert_send::<crate::BatchItem<'_, f64>>();
    assert_send::<crate::BatchItem<'_, C64>>();
}

/// `Error` separately, because `'static` is the load-bearing part of its bound
/// and an added borrowed field would take it away without touching `Send`.
#[test]
fn error_can_be_boxed_as_a_std_error() {
    assert_send_sync_static::<Error>();

    // The bound above spelled out as the use it exists for: this is what an
    // application error type does with ours.
    let _boxed: Box<dyn core::error::Error + Send + Sync> = Box::new(Error::Internal("example"));
}
