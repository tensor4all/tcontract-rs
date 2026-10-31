//! The faer strategy: copy-free fusion to one strided batched GEMM.
//!
//! The index groups M (A, D), N (B, D), K (A, B) and H (A, B, D) are each put
//! in one order and fused per operand. When every group fuses in every operand
//! that carries it, the contraction is one strided batched GEMM over the
//! caller's memory and faer computes it. Anything else is declined and runs on
//! the packed driver: this strategy never copies an operand.
//!
//! The approach follows tenferro-rs's CPU `dot_general`
//! (`tenferro-cpu/src/{dot_runtime.rs, gemm/mod.rs}` at `5a4e7fd`),
//! reimplemented on the validated [`Problem`] roles. The batched loop, with
//! its outer or inner parallel schedule, moved here from the former BLAS crate.
//!
//! # Semantics
//!
//! `D = op_D(alpha * op_A(A) * op_B(B) + beta * op_C(C))` with a C mode of
//! overwrite, in-place accumulation, or a separately described C. faer adds
//! into D with coefficient one, so the update is reduced to that form without
//! copying an operand:
//!
//! * an output conjugation distributes over the sum, flipping the conjugation
//!   of A, B and C and conjugating `alpha` and `beta`;
//! * a nonzero `beta` and the previous-D term (in-place) are applied to D by
//!   the executor's output-update pass (`D = op_D(beta * op_C(D))`, parallel,
//!   skipped when it is the identity), then faer accumulates the product;
//! * a separate C that is not D itself is first written into D by the
//!   executor's output-update pass (`D = op_D(beta * op_C(C))`, one strided
//!   pass over the output through strided-basic, see
//!   [`super::elementwise`]); faer then accumulates `op_D(alpha * op_A(A) *
//!   op_B(B))`. That pass is output-sized, not an operand normalization: A, B
//!   and C are never copied or packed and `PlanReport::materialized` stays
//!   all false. With `beta == 0` C is not read. A separate C that is D itself
//!   (same mapping, same origin) is the in-place case above.
//! * a fresh (uninitialized) destination is served by faer's overwrite-only
//!   entry ([`FaerWrite::Fresh`]) instead of a zero-fill followed by the
//!   replace path, so no separate initialization pass over D is needed.
//!
//! # Allocation
//!
//! What a call allocates on the product path is faer's, not this crate's, and
//! it depends on which kernel the target gets (tcontract-rs#162):
//!
//! * **x86_64 with AVX2 or AVX-512, the four storage types**: t4a-faer's
//!   native kernels (`t4a-private-gemm-x86`) pack into a per-thread slab reused
//!   across calls, so a warm execute allocates nothing.
//!   `tests/faer_alloc_upper_bound.rs` pins that with a counting allocator.
//! * **Every other target, and every other scalar type**: faer falls back to
//!   `gemm-common`'s `gemm`, which allocates one RHS packing buffer per call
//!   once the product is large enough to pack (40-78 KB observed on aarch64
//!   macOS; 8 bytes more for `gemm`'s GEMV form). The buffer lives inside
//!   `gemm_basic_generic`, so it cannot be moved into an `Exec` workspace
//!   without forking `gemm-common`, which would mean owning a numerical path
//!   upstream does not test. That fallback residual is accepted as a known
//!   limitation of the non-SIMD targets rather than worked around here.

use std::num::NonZeroUsize;
use tcontract_exec::{with_faer_par, BatchQuery, Exec, ExecError};
use tcontract_kernel::Element;

use crate::api::{fuse_axes, sort_by_stride, CSpec, OperandId, Problem, RoleAxis, Scalar};

/// One fused group: `(extent, stride)` in each of A, B and D.
#[derive(Clone, Copy, Debug)]
struct Fused {
    extent: usize,
    a: isize,
    b: isize,
    d: isize,
}

/// What the product may assume about the destination when it starts, decided
/// by the executor from the caller's output contract and the C mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FaerWrite {
    /// The destination already holds the whole C term, `op_D(beta * op_C(C))`;
    /// the product is added into it.
    Accumulate,
    /// The destination is initialized and its prior contents may be discarded.
    Replace,
    /// The destination's elements may be uninitialized: nothing of them is read
    /// and every element is written before the call returns.
    Fresh,
}

/// The fused batched GEMM: `[M, K]`, `[K, N]`, `[M, N]` matrices over `count`
/// items.
#[derive(Clone, Debug)]
pub(crate) struct FaerPlan {
    m: Fused,
    n: Fused,
    k: Fused,
    h: Fused,
    conj_a: bool,
    conj_b: bool,
    conj_d: bool,
}

/// Run `f` on `0..n` in a stack buffer when `n` is small (planning allocates
/// nothing for the usual ranks) and in a heap buffer otherwise.
fn with_indices<R>(n: usize, f: impl FnOnce(&mut [usize]) -> R) -> R {
    const INLINE: usize = 16;
    if n <= INLINE {
        let mut buf = [0usize; INLINE];
        let order = &mut buf[..n];
        order.iter_mut().enumerate().for_each(|(i, x)| *x = i);
        f(order)
    } else {
        f(&mut (0..n).collect::<Vec<_>>())
    }
}

/// Fuse one group over the operands that carry it, trying the order each
/// carrier would choose and then the natural one. The candidates are tried
/// one at a time in one reused buffer.
fn fuse_group(group: &[RoleAxis], carriers: &[OperandId]) -> Option<Fused> {
    let fits = |order: &[usize]| {
        carriers
            .iter()
            .all(|&o| fuse_axes(group, order, o).is_some())
    };
    let fused = |order: &[usize]| {
        let get = |o: OperandId| fuse_axes(group, order, o);
        let (extent, a) = get(OperandId::A).unwrap_or((1, 1));
        // Every carrier agrees on the extent; the absent operands keep stride one.
        Fused {
            extent,
            a,
            b: get(OperandId::B).map_or(1, |x| x.1),
            d: get(OperandId::D).map_or(1, |x| x.1),
        }
    };
    with_indices(group.len(), |order| {
        let reset = |order: &mut [usize]| order.iter_mut().enumerate().for_each(|(i, x)| *x = i);
        for &o in carriers {
            // From the natural order, stable: equal strides keep that order.
            reset(order);
            sort_by_stride(group, o, order);
            if fits(order) {
                return Some(fused(order));
            }
        }
        reset(order);
        fits(order).then(|| fused(order))
    })
}

/// Output elements up to which a separate C's output pass stays in cache and
/// is cheap whatever the K.
const SEPARATE_C_MAX_OUT: usize = 1 << 16;
/// Contracted extent from which the GEMM amortizes a separate C's output pass,
/// for operands in faer's native orientation (A unit stride along M).
const SEPARATE_C_MIN_K: usize = 512;

/// Whether faer serves a separately described C at all, where D takes one
/// extra output-sized pass at `beta != 0` before the product is accumulated.
///
/// Fitted on the measured TAPP-style rows (`separate_b1` and `separate_same`,
/// 4T/8T, `tenferro-p1-gemm` and `large-batched-gemm`; results under
/// `benchmarks/benchmarks/tcontract/contract/results/2026-10-03-phase2-w2b/`);
/// no admitted case measured below 0.92 of the packed driver. The clean
/// confirmation
/// (`benchmarks/benchmarks/tcontract/contract/results/2026-10-04-phase2-w2b-clean/`)
/// shows the declined cases lose at `beta == 0` too, where there is no pass, so
/// the gate applies to every `beta` and the packed driver serves them wholly.
/// It admits:
/// * a cache-resident output (at most 2^16 elements);
/// * a K of at least 512 with A unit-stride along M (a transposed A loses to
///   packed even without the pass);
/// * a matrix-vector shape (M or N equal to 1), where packed is far slower.
fn separate_c_pays(f: &FaerPlan, p: &Problem) -> bool {
    let r = p.roles();
    let out = r
        .m()
        .iter()
        .chain(r.n())
        .chain(r.h())
        .map(|x| x.extent())
        .fold(1usize, |a, e| a.saturating_mul(e));
    out <= SEPARATE_C_MAX_OUT
        || (f.k.extent >= SEPARATE_C_MIN_K && f.m.a == 1)
        || f.m.extent == 1
        || f.n.extent == 1
}

/// The fusion of `p`, or `None` when this strategy cannot run it copy-free with
/// full semantics, or when a separate C's output pass makes faer lose to the
/// packed driver at every `beta`.
pub(crate) fn plan(p: &Problem) -> Option<FaerPlan> {
    let r = p.roles();
    // A reduction over an axis only one input carries has no matrix to hand
    // to faer without a broadcast copy.
    if r.k().iter().any(|x| !(x.in_a() && x.in_b())) {
        return None;
    }
    use OperandId::{A, B, D};
    let plan = FaerPlan {
        m: fuse_group(r.m(), &[A, D])?,
        n: fuse_group(r.n(), &[B, D])?,
        k: fuse_group(r.k(), &[A, B])?,
        h: fuse_group(r.h(), &[A, B, D])?,
        conj_a: p.a().op().is_conj(),
        conj_b: p.b().op().is_conj(),
        conj_d: p.d().op().is_conj(),
    };
    // A separate C costs an output pass whenever `beta != 0`. Where that pass
    // makes faer lose to the packed driver, the packed driver serves every
    // `beta`; declining here is what `Plan`'s strategy choice acts on.
    if matches!(p.c_spec(), CSpec::Separate(_)) && !separate_c_pays(&plan, p) {
        return None;
    }
    Some(plan)
}

impl FaerPlan {
    /// The GEMM volume `m * n * k` of one batch item, saturating.
    pub(crate) fn volume(&self) -> u64 {
        [self.m.extent, self.n.extent, self.k.extent]
            .iter()
            .fold(1u64, |v, &e| v.saturating_mul(e as u64))
    }

    /// Multiply-adds of one batch item, the faer route's `item_muladds` in the
    /// shared batch vocabulary: `m * n * k`, four times for complex (faer runs
    /// four real GEMMs).
    fn item_muladds<T: Scalar>(&self) -> f64 {
        self.m.extent as f64
            * self.n.extent as f64
            * self.k.extent as f64
            * if T::IS_COMPLEX { 4.0 } else { 1.0 }
    }

    /// Run the batched GEMM.
    ///
    /// # Safety
    ///
    /// `a`, `b` and `d` are the elements at logical index zero of non-empty
    /// layouts matching the plan's problem (K non-empty); `d` is exclusive and
    /// injective and does not alias `a` or `b`. `write` is what may be assumed
    /// about `d`: with [`FaerWrite::Accumulate`] `d` already holds the whole C
    /// term, `op_D(beta * op_C(C))` (the caller's output pass), and only
    /// `op_D(alpha * op_A(A) * op_B(B))` is added; with
    /// [`FaerWrite::Replace`] `d` is initialized and overwritten; with
    /// [`FaerWrite::Fresh`] `d`'s elements may be uninitialized and are all
    /// written without being read first.
    pub(crate) unsafe fn run<T: Scalar>(
        &self,
        exec: &mut Exec<'_>,
        alpha: T,
        a: *const T,
        b: *const T,
        write: FaerWrite,
        d: *mut T,
    ) -> Result<(), ExecError> {
        let (mut ca, mut cb) = (self.conj_a, self.conj_b);
        let mut alpha = alpha;
        if self.conj_d {
            // conj(alpha*A*B + beta*C) = conj(alpha)*conj(A)*conj(B) + conj(beta)*conj(C);
            // the C term is already in D, conjugated by the caller's pass.
            (ca, cb) = (!ca, !cb);
            alpha = Element::conj(alpha);
        }
        let count = self.h.extent;
        // The one batch decision, resolved before any item runs. The faer
        // capability keeps the inner route unless one item alone is too small
        // for a team, at which point the items take the lanes. `Auto` reproduces
        // the pre-#123 `inner`/`total` widths exactly, and a forced strategy is
        // honoured as-is.
        // Only a team gives faer internal parallelism; a token without one runs
        // every item `Par::Seq`, so the query describes the route that will run.
        let team = exec.token_ref().carries_team();
        let decision = exec.batch_decision(
            &BatchQuery::new(count, self.item_muladds::<T>())
                .with_item_parallel(team)
                .with_outer_only_when_item_serial(true)
                .with_item_sections(FAER_GEMM_SECTIONS),
        );
        let (ap, bp, dp) = (SendConst(a), SendConst(b), SendMut(d));
        let item = |i: usize, par: faer::Par<'_>| {
            let i = i as isize;
            // SAFETY: offsets `i * stride` stay inside the validated layouts
            // (i < count); output items are disjoint (D is injective).
            unsafe {
                self.item(
                    alpha,
                    ap.get().offset(i * self.h.a),
                    ca,
                    bp.get().offset(i * self.h.b),
                    cb,
                    write,
                    dp.get().offset(i * self.h.d),
                    par,
                )
            }
        };
        // INVARIANT: a decision never has zero lanes or zero item threads.
        let nz = |n: usize| NonZeroUsize::new(n).unwrap_or(NonZeroUsize::MIN);
        // Without a team every item runs `Par::Seq`, so a forced split's item
        // threads would only reserve slots nothing uses.
        let item_threads = if team { decision.item_threads } else { 1 };
        if decision.lanes > 1 {
            // Each lane runs its items on the granted `item_threads`, taken from
            // the context and never chosen here (`BatchDecision`'s route contract).
            // INVARIANT: `batch_decision` keeps `lanes * item_threads <= budget`
            // (`Split` shares what is left, and `Auto` reports item threads only
            // with one lane), so splitting a token of `lanes * item_threads`
            // slots into `lanes` jobs gives every lane exactly its share, and
            // the library receives that share as a plain width through
            // `external`.
            let total = nz(decision.lanes.saturating_mul(item_threads));
            let lanes = decision.lanes;
            exec.reborrow()
                .limit(total)
                .into_token()
                .fork(lanes, |lane, mut child| {
                    // Contiguous items per lane, as the batch partition always was.
                    let items = lane * count / lanes..(lane + 1) * count / lanes;
                    with_faer_par(child.reborrow(), |mut par| {
                        for i in items {
                            item(i, par.reborrow());
                        }
                    });
                });
        } else {
            // One pool entry (or one team section sequence) for the whole batch.
            with_faer_par(exec.token().limit(nz(item_threads)), |mut par| {
                for i in 0..count {
                    item(i, par.reborrow());
                }
            });
        }
        Ok(())
    }

    /// One GEMM of the batch.
    ///
    /// # Safety
    ///
    /// As [`FaerPlan::run`], for one item's origins.
    #[allow(clippy::too_many_arguments)] // INVARIANT: the GEMM argument set.
    unsafe fn item<T: Scalar>(
        &self,
        alpha: T,
        a: *const T,
        ca: bool,
        b: *const T,
        cb: bool,
        write: FaerWrite,
        d: *mut T,
        par: faer::Par,
    ) {
        let (m, n, k) = (self.m.extent, self.n.extent, self.k.extent);
        // SAFETY: the caller's contract; views are read-only / exclusive as faer requires.
        unsafe {
            let (d_strides, a_strides, b_strides) = (
                (self.m.d, self.n.d),
                (self.m.a, self.k.a),
                (self.k.b, self.n.b),
            );
            match write {
                // The overwrite-only entry: faer's kernel writes every element
                // of the destination in its first depth block and never loads
                // the old value, so an uninitialized `d` needs no fill here. On
                // targets and element types without that kernel it zero-fills
                // internally and multiplies as `Replace` would.
                FaerWrite::Fresh => T::matmul_uninit(
                    (m, n, k),
                    d,
                    d_strides,
                    a,
                    a_strides,
                    ca,
                    b,
                    b_strides,
                    cb,
                    alpha,
                    par,
                ),
                FaerWrite::Replace => T::matmul(
                    false,
                    (m, n, k),
                    d,
                    d_strides,
                    a,
                    a_strides,
                    ca,
                    b,
                    b_strides,
                    cb,
                    alpha,
                    par,
                ),
                FaerWrite::Accumulate => T::matmul(
                    true,
                    (m, n, k),
                    d,
                    d_strides,
                    a,
                    a_strides,
                    ca,
                    b,
                    b_strides,
                    cb,
                    alpha,
                    par,
                ),
            }
        };
    }
}

/// Dispatches one faer GEMM performs, declared to the token's width rule: one parallel
/// region per product (the team route publishes it once).
const FAER_GEMM_SECTIONS: u32 = 1;

/// Raw pointers moved into a pool closure.
#[derive(Clone, Copy)]
struct SendConst<T>(*const T);
impl<T> SendConst<T> {
    fn get(self) -> *const T {
        self.0
    }
}
// SAFETY: only dereferenced under the validated contracts documented at use.
unsafe impl<T> core::marker::Send for SendConst<T> {}
unsafe impl<T> Sync for SendConst<T> {}

#[derive(Clone, Copy)]
struct SendMut<T>(*mut T);
impl<T> SendMut<T> {
    fn get(self) -> *mut T {
        self.0
    }
}
// SAFETY: as `Send`; writes are to disjoint, injective regions.
unsafe impl<T> core::marker::Send for SendMut<T> {}
unsafe impl<T> Sync for SendMut<T> {}
