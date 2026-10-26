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

use std::num::NonZeroUsize;
use tcontract_exec::{BatchQuery, Exec, ExecError};
use tcontract_kernel::Element;

use crate::api::{CSpec, OperandId, Problem, RoleAxis, Scalar};

/// One fused group: `(extent, stride)` in each of A, B and D.
#[derive(Clone, Copy, Debug)]
struct Fused {
    extent: usize,
    a: isize,
    b: isize,
    d: isize,
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

/// Whether `order` fuses the group in operand `o`: the strides of the
/// non-unit axes must chain (`s_next == s_prev * extent_prev`). Returns the
/// fused `(extent, stride)`.
fn fuse(group: &[RoleAxis], order: &[usize], o: OperandId) -> Option<(usize, isize)> {
    let mut ext = 1usize;
    let mut first: Option<isize> = None;
    let mut expect = 0isize;
    for &e in order {
        let d = group[e].extent();
        if d == 1 {
            continue;
        }
        let s = group[e].stride(o);
        match first {
            None => first = Some(s),
            Some(_) if s != expect => return None,
            Some(_) => {}
        }
        expect = s.checked_mul(isize::try_from(d).ok()?)?;
        ext = ext.checked_mul(d)?;
    }
    Some((ext, first.unwrap_or(1)))
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
    let fits = |order: &[usize]| carriers.iter().all(|&o| fuse(group, order, o).is_some());
    let fused = |order: &[usize]| {
        let get = |o: OperandId| fuse(group, order, o);
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
            order.sort_by_key(|&e| group[e].stride(o).unsigned_abs());
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
    /// injective and does not alias `a` or `b`. With `accumulate`, `d` already holds the whole C term,
    /// `op_D(beta * op_C(C))` (the caller's output pass), and only
    /// `op_D(alpha * op_A(A) * op_B(B))` is added ; without it `d` is
    /// overwritten.
    pub(crate) unsafe fn run<T: Scalar>(
        &self,
        exec: &mut Exec<'_>,
        alpha: T,
        a: *const T,
        b: *const T,
        accumulate: bool,
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
        let decision = exec.batch_decision(
            &BatchQuery::new(count, self.item_muladds::<T>())
                .with_item_parallel(true)
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
                    accumulate,
                    dp.get().offset(i * self.h.d),
                    par,
                )
            }
        };
        // INVARIANT: a decision never has zero lanes or zero item threads.
        let nz = |n: usize| NonZeroUsize::new(n).unwrap_or(NonZeroUsize::MIN);
        if decision.lanes > 1 {
            // Each lane runs its items on the granted `item_threads`, taken from
            // the context and never chosen here (`BatchDecision`'s route contract).
            // INVARIANT: `batch_decision` keeps `lanes * item_threads <= budget`
            // (`Split` shares what is left, and `Auto` reports item threads only
            // with one lane), so splitting a token of `lanes * item_threads`
            // slots into `lanes` jobs gives every lane exactly its share, and
            // the library receives that share as a plain width through
            // `external`.
            let total = nz(decision.lanes.saturating_mul(decision.item_threads));
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
            with_faer_par(exec.token().limit(nz(decision.item_threads)), |mut par| {
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
        accumulate: bool,
        d: *mut T,
        par: faer::Par,
    ) {
        let (m, n, k) = (self.m.extent, self.n.extent, self.k.extent);
        // SAFETY: the caller's contract; views are read-only / exclusive as faer requires.
        unsafe {
            T::matmul(
                accumulate,
                (m, n, k),
                d,
                (self.m.d, self.n.d),
                a,
                (self.m.a, self.k.a),
                ca,
                b,
                (self.k.b, self.n.b),
                cb,
                alpha,
                par,
            )
        };
    }
}

/// Dispatches one faer GEMM performs, declared to the token's width rule: one parallel
/// region per product (the team route publishes it once, the rayon route is one
/// tile split).
const FAER_GEMM_SECTIONS: u32 = 1;

/// The width `Exec::external` hands over, as faer's own parallelism type.
fn to_faer(width: NonZeroUsize) -> faer::Par<'static> {
    match width.get() {
        1 => faer::Par::Seq,
        w => faer::Par::rayon(w),
    }
}

/// Runs `f` with the parallelism faer should use for `token`.
///
/// A token that carries a team (minted by a `threadteam::Team`, or a child of one)
/// hands faer its own team as `Par::Team`: faer's regions run on the reserved
/// members as team sections, at the dispatch cost the floors were derived from. A
/// plain token keeps `Exec::external`, which hands over its width as `Par::rayon`
/// (and enters the pool once, or not at all from inside it).
fn with_faer_par<R: Send>(
    mut token: threadteam::Exec<'_>,
    f: impl FnOnce(faer::Par<'_>) -> R + Send,
) -> R {
    if token.carries_team() {
        let mut f = Some(f);
        // A team token returns its carried team at once, without acquisition.
        let ran = token.team(|team| {
            let f = f.take().expect("the closure runs once");
            f(faer::Par::Team(team))
        });
        // A refusal is unreachable for a token that carries a team; fall back to the width.
        if let Ok(value) = ran {
            return value;
        }
        let f = f.take().expect("a refused team did not run the closure");
        return token.external(|w| f(to_faer(w)));
    }
    token.external(|w| f(to_faer(w)))
}

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
