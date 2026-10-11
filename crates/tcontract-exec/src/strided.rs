//! Running strided-rs kernels on an [`Exec`] (feature `strided`).
//!
//! strided-rs parallelizes through its own [`ExecContext`]; this adapter
//! bridges the two so strided fanout stays inside the borrowed pool and a
//! small operation never enters it.

use strided_basic::ExecContext;

use crate::{Exec, Par};

/// strided-basic's parallel threshold (`threading::MINTHREADLENGTH`, not
/// exported), itself Strided.jl's `MINTHREADLENGTH = 1 << 15`. Operations of
/// at most this many elements run serially.
pub const MIN_PARALLEL_LEN: usize = 1 << 15;

/// Run a strided operation of `len` logical elements on `exec`.
///
/// At or below [`MIN_PARALLEL_LEN`] `op` runs serially on the caller with
/// [`ExecContext::serial`] and the pool is never entered. Above it, the pool
/// is entered once and `op` receives a context bounded by the budget.
///
/// # Examples
///
/// ```
/// let n = tcontract_exec::strided::run_with_exec(&tcontract_exec::Exec::serial(), 10, |ctx| {
///     assert!(ctx.is_serial());
///     1
/// });
/// assert_eq!(n, 1);
/// ```
pub fn run_with_exec<R: Send>(
    exec: &Exec<'_>,
    len: usize,
    op: impl FnOnce(ExecContext) -> R + Send,
) -> R {
    let k = if len > MIN_PARALLEL_LEN {
        exec.budget()
    } else {
        1
    };
    exec.install(k, |par| {
        let ctx = match par {
            Par::Seq => ExecContext::serial(),
            Par::Threads(n) => ExecContext::max_threads(n.get()).unwrap_or_default(),
        };
        ctx.run(|| op(ctx))
    })
}
