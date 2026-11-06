//! The batched-inner-product strategy: a batch of dot products, not a GEMM.
//!
//! `Plan::build` selects this strategy when the contraction has no matrix
//! multiply left — the M and N groups both fuse to extent one — and the batch
//! group H and the contracted group K fuse copy-free. The step is then
//!
//! ```text
//! D[h] = sum_k op_A(A[h, k]) op_B(B[h, k])
//! ```
//!
//! which the faer route serves as one degenerate `1 x 1 x k` GEMM per batch
//! item. That per-item call is what costs: measured on an M5 Max at one
//! thread, `einsum("pi,pi->p")` (tenferro-rs#2293) spent 6.2 ns per contracted
//! element at `h = k = 4096`, 40x the host's streaming rate, while the same
//! arithmetic as one fused pass ran at 60-97% of that rate.
//!
//! One traversal, chosen from the layouts at planning and never from a size
//! threshold:
//!
//! * the batch axis is the unit-stride axis in A, B and D: one pass per
//!   contracted element over the output axis (`fma`, or `zip_update3_into`
//!   when `alpha` is not one). Every output still accumulates in ascending k,
//!   so the result is a sequential dot with fused rounding;
//! * otherwise one `zip_reduce` per batch item over the contracted axis, which
//!   is contiguous when K has unit stride and a gather when it does not. Its
//!   fold uses strided-rs' lane scheme, so the sum is reassociated into a tree.
//!
//! Both traversals are strided-basic calls: this module owns no kernel, only
//! the loop over a fused group and the update semantics of the C contract.

use core::mem::MaybeUninit;

use strided_basic::{
    fma, mul_into_uninit, zip_map2_into, zip_reduce, zip_update3_into, ElementOp, Identity,
};
use tcontract_exec::Exec;
use tcontract_kernel::Element;

use crate::api::scalar::sealed::Sealed;
use crate::api::{Error, OperandId, Problem, Result, Scalar};
use crate::strategy::elementwise::{read_view, write_view, write_view_uninit};
use crate::strategy::fuse::{fuse_group, Fused};

type Done = Result<()>;

/// What the destination holds when the product starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Write {
    /// Nothing: every element is written and no previous value is read, so an
    /// uninitialized (fresh) destination is allowed.
    Overwrite,
    /// The whole C term, `op_D(beta * op_C(C))`, established by the caller's
    /// output pass.
    Accumulate,
}

/// Batch items below which one pass per contracted element is not worth a
/// strided-basic call.
///
/// A pass reads and writes one element per batch item, so below the width of one
/// 128-byte cache line the call overhead (measured at 26-55 ns per pass on an
/// M5 Max) dominates the 8 bytes it moves. There the per-item reduction runs
/// `h` calls instead of `k`, and `h < k` is the regime this geometry is in when
/// it is small (`h = 8, k = 131072` measured 3.98 ms per pass-per-k against
/// 4.0 ms here, and 3.74 ms on the per-item GEMM route it replaces).
const PER_K_MIN_ITEMS: usize = 16;

/// A planned batched inner product: the two fused groups and the traversal.
#[derive(Clone, Debug)]
pub(crate) struct InnerProductPlan {
    /// The batch group: extent and the `[A, B, D]` element strides.
    h: Fused,
    /// The contracted group: extent and the `[A, B]` element strides.
    k: Fused,
    conj_a: bool,
    conj_b: bool,
    /// One pass per contracted element over the batch axis, rather than one
    /// reduction per batch item. Chosen when the batch axis is the contiguous
    /// one in A, B and D, which is what makes each pass vectorizable.
    per_k: bool,
}

/// Fuse the role groups of `p` into a batched inner product, or `None` when
/// the problem is not one or this strategy cannot carry its full semantics.
///
/// The checks are structural, not size based: a matrix multiply left
/// (`m * n > 1`), an unfusable group, a broadcast axis, a conjugated output
/// (whose sign fold has its own rule in the faer strategy), or an empty
/// contracted group all decline, and the plan then takes the faer or packed
/// route exactly as before.
pub(crate) fn plan(p: &Problem) -> Option<InnerProductPlan> {
    let r = p.roles();
    if r.k().is_empty() {
        return None;
    }
    // A reduction over an axis only one input carries needs a broadcast copy.
    if r.k().iter().any(|x| !(x.in_a() && x.in_b())) {
        return None;
    }
    if p.d().op().is_conj() {
        return None;
    }
    use OperandId::{A, B, D};
    let m = fuse_group(r.m(), &[A, D])?;
    let n = fuse_group(r.n(), &[B, D])?;
    let k = fuse_group(r.k(), &[A, B])?;
    let h = fuse_group(r.h(), &[A, B, D])?;
    // A step whose M and N are both one carries no matrix: it is `h` dots of
    // length `k`. A larger one is a real GEMM and stays with the faer route.
    if m.extent != 1 || n.extent != 1 {
        return None;
    }
    // A zero stride is a broadcast; the traversals here address every element
    // once, so such a problem keeps its existing route.
    let broadcast = [k.a, k.b, h.a, h.b, h.d].contains(&0);
    if k.extent == 0 || h.extent == 0 || broadcast {
        return None;
    }
    let per_k = h.extent >= PER_K_MIN_ITEMS && h.a == 1 && h.b == 1 && h.d == 1;
    Some(InnerProductPlan {
        h,
        k,
        conj_a: p.a().op().is_conj(),
        conj_b: p.b().op().is_conj(),
        per_k,
    })
}

impl InnerProductPlan {
    /// Which traversal was planned. Test support: the traversal is a layout
    /// choice, and the tests pin it directly.
    #[cfg(test)]
    pub(crate) fn per_k_for_test(&self) -> bool {
        self.per_k
    }

    /// Run `D = alpha * op_A(A) * op_B(B)` into `write`'s destination.
    ///
    /// # Safety
    ///
    /// `a` and `b` are the elements at logical index zero of non-empty layouts
    /// matching the plan's problem, `d` likewise; `d` is exclusive, injective
    /// and does not alias `a` or `b`. With [`Write::Overwrite`] the previous
    /// value of `d` is never read, so its elements may be uninitialized; with
    /// [`Write::Accumulate`] `d` already holds the whole C term.
    pub(crate) unsafe fn run<T: Scalar>(
        &self,
        exec: &mut Exec<'_>,
        alpha: T,
        a: *const T,
        b: *const T,
        d: *mut T,
        write: Write,
    ) -> Done {
        match (self.conj_a, self.conj_b) {
            // `Cj` is the identity for a real type, so its arms collapse.
            (false, false) => unsafe {
                self.run_ops::<T, Identity, Identity>(exec, alpha, a, b, d, write)
            },
            (true, false) => unsafe {
                self.run_ops::<T, <T as Sealed>::Cj, Identity>(exec, alpha, a, b, d, write)
            },
            (false, true) => unsafe {
                self.run_ops::<T, Identity, <T as Sealed>::Cj>(exec, alpha, a, b, d, write)
            },
            (true, true) => unsafe {
                self.run_ops::<T, <T as Sealed>::Cj, <T as Sealed>::Cj>(exec, alpha, a, b, d, write)
            },
        }
    }

    /// The traversal with the conjugation flags lifted to the view types.
    ///
    /// # Safety
    ///
    /// As [`run`](Self::run).
    unsafe fn run_ops<T: Scalar, OA: ElementOp<T>, OB: ElementOp<T>>(
        &self,
        exec: &mut Exec<'_>,
        alpha: T,
        a: *const T,
        b: *const T,
        d: *mut T,
        write: Write,
    ) -> Done {
        let one = <T as Element>::one();
        let zero = <T as Element>::zero();
        let (h, k) = (self.h.extent, self.k.extent);
        let dims_h = [h];
        let dims_k = [k];
        let a_h = [self.h.a];
        let b_h = [self.h.b];
        let d_h = [self.h.d];
        let a_k = [self.k.a];
        let b_k = [self.k.b];
        if self.per_k {
            for kk in 0..k {
                let aoff = kk as isize * self.k.a;
                let boff = kk as isize * self.k.b;
                // SAFETY: the caller's contract; each pass addresses the batch
                // axis at one contracted element of the validated layouts.
                let av = unsafe { read_view::<T, OA>(a.offset(aoff), &dims_h, &a_h) };
                let bv = unsafe { read_view::<T, OB>(b.offset(boff), &dims_h, &b_h) };
                if kk == 0 && write == Write::Overwrite {
                    // SAFETY: as `av`, and the destination may be uninitialized
                    // because this pass writes every element without reading it.
                    let mut dv = unsafe { write_view_uninit(d, &dims_h, &d_h) };
                    if alpha == one {
                        mul_into_uninit(&mut dv, &av, &bv, exec.token()).map_err(Error::backend)?;
                    } else {
                        zip_map2_into(
                            &mut dv,
                            &av,
                            &bv,
                            move |x, y| MaybeUninit::new(alpha * x * y),
                            exec.token(),
                        )
                        .map_err(Error::backend)?;
                    }
                } else {
                    // SAFETY: as `av`; the destination is initialized (either by
                    // the first pass or by the caller's C term).
                    let mut dv = unsafe { write_view(d, &dims_h, &d_h) };
                    if alpha == one {
                        fma(&mut dv, &av, &bv, exec.token()).map_err(Error::backend)?;
                    } else {
                        zip_update3_into::<T, T, T, Identity, OA, OB>(
                            &mut dv,
                            &av,
                            &bv,
                            move |z, x, y| alpha * x * y + z,
                            exec.token(),
                        )
                        .map_err(Error::backend)?;
                    }
                }
            }
            return Ok(());
        }
        for i in 0..h {
            let aoff = i as isize * self.h.a;
            let boff = i as isize * self.h.b;
            // SAFETY: the caller's contract; one item of the validated layouts.
            let av = unsafe { read_view::<T, OA>(a.offset(aoff), &dims_k, &a_k) };
            let bv = unsafe { read_view::<T, OB>(b.offset(boff), &dims_k, &b_k) };
            let item: T = zip_reduce(&av, &bv, |x, y| x * y, |acc, t| acc + t, zero, exec.token())
                .map_err(Error::backend)?;
            // SAFETY: `d` is exclusive and injective, so this item's element is
            // written by no other pass; the write-only form never reads `d`.
            unsafe {
                let slot = d.offset(i as isize * self.h.d);
                *slot = if i == 0 && write == Write::Overwrite {
                    alpha * item
                } else {
                    *slot + alpha * item
                };
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
