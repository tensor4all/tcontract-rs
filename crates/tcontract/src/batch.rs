//! Many independent contractions, with the **batch** as the parallel axis.
//!
//! Parallelising over a batch of `n` contractions needs no barrier: items are
//! wholly independent, so the work is plain fork-join over the host's
//! [`Exec`], and this crate spawns no thread of its own.
//!
//! # Scheduling
//!
//! | Situation | Route |
//! |---|---|
//! | width 1 | the caller's thread, item after item |
//! | `items >= width` | the items are partitioned over the width, each item serial |
//! | fewer items than the width | items one after another, each on the whole `Exec` (its own inner parallelism) |
//!
//! so the batch axis and the inner axis never nest, and no threads idle while a
//! large item could use them.
//!
//! # What this does not do
//!
//! * **The split is static and contiguous**, by item count. Balancing it by
//!   estimated work is the obvious refinement and is deliberately not guessed at:
//!   D47 measured that block-scatter load imbalance does not cost on the dense
//!   path, and the case where it plausibly does -- **block-sparse, where items
//!   differ in size rather than in regularity** -- is not built either. That is
//!   where a dynamic claim over the batch belongs, and it is what TBLIS uses a
//!   dynamic atomic-claim scheduler for while keeping static partitioning for
//!   dense (part 15). Same division, reached independently.

use std::sync::Mutex;

use strided_view::{StridedView, StridedViewMut};
use tcontract_exec::Exec;

use crate::api::{AccumulationSource, Result, Scalar};
use crate::plan::Plan;
use crate::strategy::elementwise::CRead;

/// One contraction of a batch: a plan and the operands to run it against.
///
/// Holding these in a `&mut [BatchItem]` is what makes the parallel execution
/// sound without a single unsafe block on the caller's side: each item's `d` is a
/// `StridedViewMut` over its own exclusive borrow, so the borrow checker has
/// already proved the outputs disjoint. The plans are shared references and may
/// all be the *same* plan, which is the common case -- a batch of identically
/// shaped contractions. Heterogeneous shapes use one plan per shape.
///
/// There is deliberately no `Clone` or `Copy`: `d` is an exclusive borrow, and
/// that borrow is the whole soundness argument above.
#[derive(Debug)]
pub struct BatchItem<'a, T: Scalar> {
    /// The plan. Built once and shared across items where the shape allows.
    pub plan: &'a Plan<T>,
    /// Scales the product.
    pub alpha: T,
    /// Left operand.
    pub a: StridedView<'a, T>,
    /// Right operand.
    pub b: StridedView<'a, T>,
    /// Scales the accumulation source. Must be zero when `source` is `None`.
    pub beta: T,
    /// Where the accumulation term is read from. `None` means `d` is
    /// overwritten.
    pub source: Option<AccumulationSource<'a, T>>,
    /// Output. The exclusive borrow is load-bearing: see the type's documentation.
    pub d: StridedViewMut<'a, T>,
}

/// Run every item, parallelising over the batch on at most `exec.budget()`
/// threads of `exec` (serially on [`Exec::Serial`]).
///
/// # Ordering and results
///
/// Items are independent, so the batch imposes no order between them and the
/// result of each is **bitwise identical to running it alone at the width it
/// ran at** -- an item that runs on one thread takes exactly the serial path of
/// [`Plan::execute_into_accum`]. The batch axis adds no reduction and no
/// accumulation, so there is nothing for a width to change.
///
/// # All or nothing
///
/// Every item's views are checked against its plan **before any item runs**, so
/// a batch containing one bad item writes to no output at all. That is a
/// stronger guarantee than looping over [`Plan::execute_into_accum`] gives, and
/// it is the reason to prefer this even when serial: a partially executed batch
/// leaves the caller unable to say which outputs are valid.
///
/// The guarantee covers that layout preflight only. An item can still fail
/// while executing, after earlier items have written -- for example when the
/// batch loop was entered from inside a pool worker and an item's plan needs a
/// barrier-bearing team ([`Error::Exec`](crate::api::Error::Exec)). There is no
/// whole-batch rollback.
///
/// # Examples
///
/// ```
/// use strided_view::{StridedView, StridedViewMut};
/// use tcontract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
/// use tcontract::{contract_batched, BatchItem, Plan, PlanConfig};
/// use tcontract_exec::Exec;
///
/// let l = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
/// // One plan, shared by every item: the common case for a batch.
/// let problem = Problem::from_labels(
///     DType::F64, l(&[2, 2], &[1, 2]), l(&[2, 2], &[1, 2]), CSpec::Absent,
///     l(&[2, 2], &[1, 2]), &Labels::new(&[0, 1], &[1, 2], &[0, 2]),
/// ).unwrap();
/// let plan = Plan::<f64>::new(&problem, &PlanConfig::default()).unwrap();
///
/// let a0 = [1.0f64, 2.0, 3.0, 4.0];
/// let a1 = [5.0f64, 6.0, 7.0, 8.0];
/// let identity = [1.0f64, 0.0, 0.0, 1.0];
/// let (mut d0, mut d1) = ([0.0f64; 4], [0.0f64; 4]);
/// fn view(x: &[f64]) -> StridedView<'_, f64> {
///     StridedView::new(x, &[2, 2], &[1, 2], 0).unwrap()
/// }
///
/// // The outputs are distinct `&mut` borrows, which is what proves them
/// // disjoint: no unsafe on the caller's side.
/// let mut items = vec![
///     BatchItem { plan: &plan, alpha: 1.0, a: view(&a0), b: view(&identity), beta: 0.0,
///         source: None, d: StridedViewMut::new(&mut d0, &[2, 2], &[1, 2], 0).unwrap() },
///     BatchItem { plan: &plan, alpha: 1.0, a: view(&a1), b: view(&identity), beta: 0.0,
///         source: None, d: StridedViewMut::new(&mut d1, &[2, 2], &[1, 2], 0).unwrap() },
/// ];
/// contract_batched(&mut items, &Exec::serial()).unwrap();
/// drop(items); // release the borrows on d0 / d1
///
/// assert_eq!(d0, a0); // multiplying by the identity
/// assert_eq!(d1, a1);
/// ```
pub fn contract_batched<T: Scalar>(items: &mut [BatchItem<'_, T>], exec: &Exec<'_>) -> Result<()> {
    // Validate everything first, so a failure means no output is written.
    let mut reads = Vec::with_capacity(items.len());
    for it in items.iter() {
        let c = it.plan.validate_views(&it.a, &it.b, it.source, &it.d)?;
        it.plan.validate_c_beta(it.beta, c)?;
        reads.push(c);
    }
    if items.is_empty() {
        return Ok(());
    }
    let width = exec.budget();

    // Width one is the plain loop; fewer items than the width keep the whole
    // `Exec` for each item's own inner parallelism.
    if width == 1 || items.len() < width {
        for (it, c) in items.iter_mut().zip(reads) {
            run_item(it, c, exec)?;
        }
        return Ok(());
    }

    // `for_each_partition` takes `Fn`, so each item sits behind its own mutex,
    // locked exactly once and never contended: the `&mut` outputs already prove
    // the items disjoint. Each item runs serially on its lane, reusing the
    // outer context's workspace so a pooled batch allocates nothing per item.
    let cells: Vec<Mutex<(&mut BatchItem<'_, T>, CRead<T>)>> =
        items.iter_mut().zip(reads).map(Mutex::new).collect();
    let first_error: Mutex<Option<crate::api::Error>> = Mutex::new(None);
    let item_exec = exec
        .workspace()
        .map_or(Exec::Serial, Exec::serial_with_workspace);
    exec.for_each_partition(cells.len(), &|i| {
        let mut cell = cells[i].lock().unwrap_or_else(|e| e.into_inner());
        let (it, c) = &mut *cell;
        if let Err(e) = run_item(it, *c, &item_exec) {
            first_error
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get_or_insert(e);
        }
    });
    match first_error.into_inner().unwrap_or_else(|e| e.into_inner()) {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// One item on `exec`. The views were validated by the caller.
fn run_item<T: Scalar>(it: &mut BatchItem<'_, T>, c: CRead<T>, exec: &Exec<'_>) -> Result<()> {
    // SAFETY: the views were validated for every item before any of them ran;
    // `d` is an exclusive borrow, so it cannot alias `a`, `b` or a separate C.
    unsafe {
        it.plan
            .run_validated(exec, it.alpha, &it.a, &it.b, it.beta, c, &mut it.d)
    }
}
