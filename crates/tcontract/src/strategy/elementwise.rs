//! The elementwise strategy: an all-batch (Hadamard) product, and the
//! output-update helper every strategy shares for `alpha == 0` and an empty
//! contraction.
//!
//! Both are one strided pass over the output's axes with the full update
//! semantics
//!
//! ```text
//! D = op_D( alpha * op_A(A) * op_B(B) + beta * op_C(C) )
//! ```
//!
//! where `C` is read from the output itself (in-place accumulation), from a
//! separately described source, or not at all (`beta == 0` reads no previous
//! value).
//!
//! The traversal belongs to strided-rs (`strided-basic`): its blocked,
//! contiguous-inner-loop kernels vectorize and split across the executor's
//! pool through [`tcontract_exec::strided::run_with_exec`]. This module only
//! specializes the update *outside* the per-element loop:
//!
//! In-place accumulation reads the previous `D` through the destination with
//! strided-basic's `map_update_into`, `zip_update2_into` and `zip_update3_into`
//! (no second view over `D`); `axpy` and `fma` cover the unit-scalar forms. The
//! write-only forms (overwrite and a separate `C`) never form a `&mut [T]` over
//! `D`: they build a `StridedViewMut<MaybeUninit<T>>` and store
//! `MaybeUninit::new(value)`, so a fresh, uninitialized destination is never
//! read.
//!
//! 1. `op_D` is folded into the other terms, since
//!    `conj(alpha*a*b + beta*c) = conj(alpha)*conj(a)*conj(b) + conj(beta)*conj(c)`;
//!    what remains is `conj_a'`, `conj_b'`, `conj_c'` and transformed scalars.
//! 2. One dispatch on those three flags (type-level `Identity`/`Conj` views;
//!    all `false` for a real element type) and on the input and C modes picks
//!    a monomorphized closure such as `|a, b| alpha * a * b`. No flag is
//!    tested per element.

use core::mem::MaybeUninit;

use strided_basic::{
    axpy, fma, map_into, map_update_into, mul_into_uninit, zip_map2_into, zip_map3_into,
    zip_update2_into, zip_update3_into, ElementOp, Identity, RawStridedMut, RawStridedRef,
    StridedError, UpdatePlan,
};
use strided_view::{StridedView, StridedViewMut};
use tcontract_exec::strided::run_with_exec;
use tcontract_exec::Exec;
use tcontract_kernel::Element;
use tinyvec::TinyVec;

use crate::api::problem::COMBINED_RANK;
use crate::api::scalar::sealed::Sealed;
use crate::api::{Error, OperandId, Problem, Result, RoleAxis, Scalar};

/// Where the `beta * op_C(C)` term is read from.
#[derive(Clone, Copy, Debug)]
pub enum CRead<T> {
    /// No term: nothing previous is read and `beta` is treated as zero.
    None,
    /// The output element itself.
    InPlace,
    /// A separate source with the problem's C strides; the pointer is C's
    /// element at logical index zero.
    Separate(*const T),
}

/// The input term of one update.
#[derive(Clone, Copy, Debug)]
pub enum Inputs<T> {
    /// No input term (an output-only pass).
    None,
    /// `alpha * op_A(A) * op_B(B)`, with the two origins.
    Product(*const T, *const T),
    /// `alpha * op_A(A)`, with the origin (the axpy-like update of `add`).
    Scaled(*const T),
}

// SAFETY: the pointers are dereferenced only under the contracts of
// `ElementPlan::run`, which every caller upholds.
unsafe impl<T> Send for Inputs<T> {}
unsafe impl<T> Sync for Inputs<T> {}
unsafe impl<T> Send for CRead<T> {}
unsafe impl<T> Sync for CRead<T> {}

/// The scalars and conjugations of one update.
#[derive(Clone, Copy, Debug)]
pub struct Expr<T> {
    pub alpha: T,
    pub beta: T,
    pub conj_a: bool,
    pub conj_b: bool,
    pub conj_c: bool,
    pub conj_d: bool,
}

#[derive(Clone, Copy)]
struct Ptr<T>(*mut T);
// SAFETY: dereferenced only through strided views over validated layouts;
// strided-rs splits writes into disjoint ranges of an injective output.
unsafe impl<T> Send for Ptr<T> {}
unsafe impl<T> Sync for Ptr<T> {}

impl<T> Ptr<T> {
    fn get(self) -> *mut T {
        self.0
    }
}

/// Where the previous-value term is read from, after `beta == 0` and a missing
/// `C` have been resolved to [`Mode::Overwrite`].
#[derive(Clone, Copy)]
enum Mode<T> {
    /// No term.
    Overwrite,
    /// The output itself (read through an alias of the destination).
    InPlace,
    /// A separate source.
    Separate(Ptr<T>),
}

/// The lowest and highest element offset a layout addresses, from its origin.
fn offset_span(dims: &[usize], strides: &[isize]) -> (isize, isize) {
    let (mut lo, mut hi) = (0isize, 0isize);
    for (&n, &s) in dims.iter().zip(strides) {
        let reach = (n as isize - 1) * s;
        if reach < 0 {
            lo += reach;
        } else {
            hi += reach;
        }
    }
    (lo, hi)
}

/// A read view over a validated layout whose logical origin is `origin`.
///
/// # Safety
///
/// Every offset the layout generates from `origin` is a live element, nothing
/// writes it during `'a` except through an identical in-place mapping.
unsafe fn read_view<'a, T, Op>(
    origin: *const T,
    dims: &'a [usize],
    strides: &'a [isize],
) -> StridedView<'a, T, Op> {
    let (lo, hi) = offset_span(dims, strides);
    // SAFETY: the caller's contract; `[lo, hi]` is the layout's reach.
    unsafe {
        let data = core::slice::from_raw_parts(origin.offset(lo), (hi - lo) as usize + 1);
        StridedView::new_unchecked(data, dims, strides, -lo)
    }
}

/// The write view of the destination.
///
/// # Safety
///
/// As [`read_view`], and the layout is injective and exclusively owned.
unsafe fn write_view<'a, T>(
    origin: *mut T,
    dims: &'a [usize],
    strides: &'a [isize],
) -> StridedViewMut<'a, T> {
    let (lo, hi) = offset_span(dims, strides);
    // SAFETY: the caller's contract.
    unsafe {
        let data = core::slice::from_raw_parts_mut(origin.offset(lo), (hi - lo) as usize + 1);
        StridedViewMut::new_unchecked(data, dims, strides, -lo)
    }
}

/// The write-only view of a destination whose previous values are never read:
/// the layout of [`write_view`], typed `MaybeUninit` so no `&mut [T]` (and so
/// no `T` validity assertion) exists before the kernel initializes it.
///
/// # Safety
///
/// As [`write_view`]; the storage need not be initialized, but the layout must
/// still be injective and exclusively owned for `'a`.
unsafe fn write_view_uninit<'a, T>(
    origin: *mut T,
    dims: &'a [usize],
    strides: &'a [isize],
) -> StridedViewMut<'a, MaybeUninit<T>> {
    let (lo, hi) = offset_span(dims, strides);
    // SAFETY: `MaybeUninit<T>` has `T`'s size and alignment, so the same live,
    // exclusively owned reach is valid for writes; `MaybeUninit` asserts no
    // value validity.
    unsafe {
        let data = core::slice::from_raw_parts_mut(
            origin.offset(lo).cast::<MaybeUninit<T>>(),
            (hi - lo) as usize + 1,
        );
        StridedViewMut::new_unchecked(data, dims, strides, -lo)
    }
}

/// The axes of a pass: extents and the `[A, B, C, D]` element strides of each
/// axis, extent-1 axes dropped.
///
/// Both vectors store inline up to the combined role count (`2 * INLINE_RANK`,
/// the most distinct labels two rank-[`INLINE_RANK`] operands can carry) and
/// spill above it, so a plan build at rank `<= INLINE_RANK` allocates no axis
/// metadata.
#[derive(Clone, Debug)]
pub struct ElementPlan {
    dims: TinyVec<[usize; COMBINED_RANK]>,
    /// The `A`, `B`, `C` and `D` strides, one `dims.len()` run each.
    strides: TinyVec<[isize; 4 * COMBINED_RANK]>,
    total: usize,
    /// The compiled plain `D = beta * op_C(C)` traversal (destination `D`, source
    /// `C`), when the destination layout allowed it. The in-place self-update
    /// `D = beta * D` cannot use it, because the typed plan forbids an aliased
    /// source and destination; that arm keeps its per-call `map_update_into`.
    beta_update: Option<UpdatePlan>,
}

impl ElementPlan {
    fn from_axes(axes: impl Iterator<Item = (usize, [isize; 4])> + Clone) -> Self {
        let axes = axes.filter(|a| a.0 != 1);
        let n = axes.clone().count();
        let mut dims: TinyVec<[usize; COMBINED_RANK]> = TinyVec::new();
        let mut strides: TinyVec<[isize; 4 * COMBINED_RANK]> = TinyVec::new();
        strides.resize(4 * n, 0);
        for (i, (e, s)) in axes.enumerate() {
            dims.push(e);
            for (j, &x) in s.iter().enumerate() {
                strides[j * n + i] = x;
            }
        }
        Self {
            total: dims.iter().product(),
            dims,
            strides,
            beta_update: None,
        }
    }

    /// The `[A, B, C, D]` stride runs.
    fn strides(&self) -> [&[isize]; 4] {
        let n = self.dims.len();
        core::array::from_fn(|j| &self.strides[j * n..(j + 1) * n])
    }

    fn from_roles<'a>(
        roles: impl Iterator<Item = &'a RoleAxis> + Clone,
        with_inputs: bool,
    ) -> Self {
        Self::from_axes(roles.map(|r| {
            let s = |o| r.stride(o);
            let (sa, sb) = if with_inputs {
                (s(OperandId::A), s(OperandId::B))
            } else {
                (0, 0)
            };
            (r.extent(), [sa, sb, s(OperandId::C), s(OperandId::D)])
        }))
    }

    /// The pass of an all-batch problem: every axis is a batch axis.
    pub(crate) fn product(p: &Problem) -> Self {
        Self::from_roles(p.roles().h().iter(), true)
    }

    /// The pass of an elementwise `D = alpha * A + beta * D` over equal extents
    /// with `A`'s and `D`'s strides.
    pub(crate) fn scaled(dims: &[usize], a: &[isize], d: &[isize]) -> Self {
        Self::from_axes(
            dims.iter()
                .zip(a.iter().zip(d))
                .map(|(&e, (&sa, &sd))| (e, [sa, 0, sd, sd])),
        )
    }

    /// The pass over the output's elements (`M`, `N` and batch axes), without
    /// inputs. Compiles the plain `D = beta * op_C(C)` replay once, here, so an
    /// output-only call does not re-plan its traversal.
    pub(crate) fn output(p: &Problem) -> Self {
        let r = p.roles();
        let plan = Self::from_roles(r.m().iter().chain(r.n()).chain(r.h()), false);
        let st = plan.strides();
        let beta_update = UpdatePlan::compile(&plan.dims, st[3], st[2]).ok();
        Self {
            beta_update,
            ..plan
        }
    }

    /// Run the update.
    ///
    /// `inputs` carries the input origins when a product term is computed; an
    /// output-only pass passes [`Inputs::None`] and the term is zero.
    ///
    /// # Errors
    ///
    /// [`Error::Backend`] with strided-rs's error; the plan's layouts were
    /// validated, so this is not expected.
    ///
    /// # Safety
    ///
    /// `d` and every origin are the elements at logical index zero of
    /// non-empty, bounds-checked layouts matching the plan's problem; `d` is
    /// exclusive and injective; inputs do not alias `d`; a separate `C` does not
    /// overlap `d` unless it is the same mapping at the same origin.
    pub(crate) unsafe fn run<T: Scalar>(
        &self,
        exec: &Exec<'_>,
        e: Expr<T>,
        inputs: Inputs<T>,
        c: CRead<T>,
        d: *mut T,
    ) -> Result<()> {
        // SAFETY: the caller's contract. The per-dtype hook is the one
        // instantiation of `run_impl`, so consumers never re-monomorphize it.
        unsafe { <T as Sealed>::elementwise(self, exec, e, inputs, c, d) }
    }

    /// The generic body of [`ElementPlan::run`]; instantiated only by the
    /// four per-dtype hooks in `api::scalar`.
    ///
    /// # Safety
    ///
    /// As [`ElementPlan::run`].
    pub(crate) unsafe fn run_impl<T: Scalar>(
        &self,
        exec: &Exec<'_>,
        e: Expr<T>,
        inputs: Inputs<T>,
        c: CRead<T>,
        d: *mut T,
    ) -> Result<()> {
        if self.total == 0 {
            return Ok(());
        }
        let zero = <T as Element>::zero();
        let mode = match c {
            CRead::None => Mode::Overwrite,
            _ if e.beta == zero => Mode::Overwrite,
            CRead::InPlace => Mode::InPlace,
            // The same mapping at the same origin is the in-place update.
            CRead::Separate(p) if core::ptr::eq(p, d) => Mode::InPlace,
            CRead::Separate(p) => Mode::Separate(Ptr(p as *mut T)),
        };
        // op_D folded into the terms: conj(alpha*a*b + beta*c) conjugates
        // every factor. A real element type has no conjugation at all.
        let flip = e.conj_d;
        let (mut alpha, mut beta) = (e.alpha, e.beta);
        let (mut ca, mut cb, mut cc) = (e.conj_a, e.conj_b, e.conj_c);
        if flip {
            alpha = Element::conj(alpha);
            beta = Element::conj(beta);
            ca = !ca;
            cb = !cb;
            cc = !cc;
        }
        let ptrs = match inputs {
            Inputs::Product(a, b) => [Ptr(a as *mut T), Ptr(b as *mut T)],
            Inputs::Scaled(a) => [Ptr(a as *mut T), Ptr(core::ptr::null_mut())],
            Inputs::None => [Ptr(core::ptr::null_mut()); 2],
        };
        let kind = match inputs {
            Inputs::Product(..) => Kind::Product,
            Inputs::Scaled(_) => Kind::Scaled,
            Inputs::None => Kind::None,
        };
        let job = Job {
            plan: self,
            alpha,
            beta,
            kind,
            ptrs,
            mode,
            d: Ptr(d),
        };
        // Dispatch on (kind, C mode) first, then only on the conjugations that
        // arm uses, so an unused operand op is never instantiated. `Cj` is the
        // identity for a real type: its two branches are one instance.
        type Id = Identity;
        macro_rules! cj {
            () => {
                <T as Sealed>::Cj
            };
        }
        let has_c = !matches!(job.mode, Mode::Overwrite);
        match (job.kind, has_c) {
            (Kind::None, false) => job.zero(exec),
            (Kind::None, true) => {
                if cc {
                    job.none_c::<cj!()>(exec)
                } else {
                    job.none_c::<Id>(exec)
                }
            }
            (Kind::Scaled, false) => {
                if ca {
                    job.scaled::<cj!()>(exec)
                } else {
                    job.scaled::<Id>(exec)
                }
            }
            (Kind::Scaled, true) => match (ca, cc) {
                (false, false) => job.scaled_c::<Id, Id>(exec),
                (false, true) => job.scaled_c::<Id, cj!()>(exec),
                (true, false) => job.scaled_c::<cj!(), Id>(exec),
                (true, true) => job.scaled_c::<cj!(), cj!()>(exec),
            },
            (Kind::Product, false) => match (ca, cb) {
                (false, false) => job.product::<Id, Id>(exec),
                (false, true) => job.product::<Id, cj!()>(exec),
                (true, false) => job.product::<cj!(), Id>(exec),
                (true, true) => job.product::<cj!(), cj!()>(exec),
            },
            (Kind::Product, true) => match (ca, cb, cc) {
                (false, false, false) => job.product_c::<Id, Id, Id>(exec),
                (false, false, true) => job.product_c::<Id, Id, cj!()>(exec),
                (false, true, false) => job.product_c::<Id, cj!(), Id>(exec),
                (false, true, true) => job.product_c::<Id, cj!(), cj!()>(exec),
                (true, false, false) => job.product_c::<cj!(), Id, Id>(exec),
                (true, false, true) => job.product_c::<cj!(), Id, cj!()>(exec),
                (true, true, false) => job.product_c::<cj!(), cj!(), Id>(exec),
                (true, true, true) => job.product_c::<cj!(), cj!(), cj!()>(exec),
            },
        }
        .map_err(Error::backend)
    }
}

#[derive(Clone, Copy)]
enum Kind {
    /// `alpha * a * b`.
    Product,
    /// `alpha * a`.
    Scaled,
    /// No input term.
    None,
}

/// One update after folding: scalars, input kind, origins and C mode.
struct Job<'p, T> {
    plan: &'p ElementPlan,
    alpha: T,
    beta: T,
    kind: Kind,
    ptrs: [Ptr<T>; 2],
    mode: Mode<T>,
    d: Ptr<T>,
}

type Done = core::result::Result<(), StridedError>;

impl<T: Scalar> Job<'_, T> {
    /// Run `body(dims, strides)` under the executor, chosen once per call.
    fn with(
        &self,
        exec: &Exec<'_>,
        body: impl FnOnce(&[usize], &[&[isize]; 4]) -> Done + Send,
    ) -> Done {
        let plan = self.plan;
        run_with_exec(exec, plan.total, |_| body(&plan.dims, &plan.strides()))
    }

    // SAFETY (all arms): the contract of `ElementPlan::run`: every origin and
    // layout is validated; the destination is exclusive and injective; a
    // separate C is disjoint from it. In-place forms never build a second view
    // of the destination; the write-only forms build a `MaybeUninit` view, so
    // they never assert that a fresh `D` already holds `T` values.

    /// `D = op_D(0)`: a stride-0 broadcast of zero.
    fn zero(&self, exec: &Exec<'_>) -> Done {
        self.with(exec, |dims, st| unsafe {
            let z = [<T as Element>::zero()];
            // Inline for every rank the combined layout keeps inline: this pass
            // serves a fresh output on the faer route, which must stay
            // allocation-free per call (tenferro's eager plan-reuse contract).
            let zeros: TinyVec<[isize; COMBINED_RANK]> =
                core::iter::repeat_n(0isize, dims.len()).collect();
            let zv = StridedView::<T, Identity>::new_unchecked(&z, dims, &zeros, 0);
            map_into(
                &mut write_view_uninit(self.d.get(), dims, st[3]),
                &zv,
                MaybeUninit::new,
            )
        })
    }

    /// `D = beta * op_C(C)`.
    fn none_c<OC: ElementOp<T>>(&self, exec: &Exec<'_>) -> Done {
        let beta = self.beta;
        self.with(exec, |dims, st| unsafe {
            match self.mode {
                // The in-place self-update aliases source and destination, which
                // the typed plan forbids; keep the per-call strided update.
                Mode::InPlace => {
                    let mut dv = write_view(self.d.get(), dims, st[3]);
                    map_update_into::<T, OC>(&mut dv, move |z| beta * z)
                }
                Mode::Separate(p) => {
                    // The plain pass replays the traversal compiled at plan time.
                    if OC::IS_IDENTITY {
                        if let Some(update) = &self.plan.beta_update {
                            let (dlo, dhi) = offset_span(dims, st[3]);
                            let dest_data = core::slice::from_raw_parts_mut(
                                self.d.get().cast::<MaybeUninit<T>>().offset(dlo),
                                (dhi - dlo) as usize + 1,
                            );
                            // SAFETY: the layout is validated, injective and
                            // exclusively owned; a fresh destination is never read.
                            let mut dest = RawStridedMut::new(dest_data, dims, st[3], -dlo)
                                .expect("validated destination layout");
                            let (slo, shi) = offset_span(dims, st[2]);
                            let src_data = core::slice::from_raw_parts(
                                p.get().offset(slo),
                                (shi - slo) as usize + 1,
                            );
                            // SAFETY: the separate C is disjoint from D and bounds
                            // validated; the layout matches the compiled plan.
                            let src = RawStridedRef::new(src_data, dims, st[2], -slo)
                                .expect("validated source layout");
                            return update.execute_uninit(&mut dest, &src, beta, |x| x);
                        }
                    }
                    let cv = read_view::<T, OC>(p.get(), dims, st[2]);
                    map_into(
                        &mut write_view_uninit(self.d.get(), dims, st[3]),
                        &cv,
                        move |x| MaybeUninit::new(beta * x),
                    )
                }
                Mode::Overwrite => unreachable!("dispatched on a C term"),
            }
        })
    }

    /// `D = alpha * op_A(A)`.
    fn scaled<OA: ElementOp<T>>(&self, exec: &Exec<'_>) -> Done {
        let alpha = self.alpha;
        self.with(exec, |dims, st| unsafe {
            let av = read_view::<T, OA>(self.ptrs[0].get(), dims, st[0]);
            map_into(
                &mut write_view_uninit(self.d.get(), dims, st[3]),
                &av,
                move |x| MaybeUninit::new(alpha * x),
            )
        })
    }

    /// `D = alpha * op_A(A) + beta * op_C(C)`.
    fn scaled_c<OA: ElementOp<T>, OC: ElementOp<T>>(&self, exec: &Exec<'_>) -> Done {
        let (alpha, beta) = (self.alpha, self.beta);
        self.with(exec, |dims, st| unsafe {
            let av = read_view::<T, OA>(self.ptrs[0].get(), dims, st[0]);
            match self.mode {
                Mode::InPlace => {
                    let mut dv = write_view(self.d.get(), dims, st[3]);
                    if beta == <T as Element>::one() && OC::IS_IDENTITY {
                        axpy(&mut dv, &av, alpha)
                    } else {
                        zip_update2_into::<T, T, OC, OA>(&mut dv, &av, move |z, x| {
                            alpha * x + beta * z
                        })
                    }
                }
                Mode::Separate(p) => {
                    let cv = read_view::<T, OC>(p.get(), dims, st[2]);
                    zip_map2_into(
                        &mut write_view_uninit(self.d.get(), dims, st[3]),
                        &av,
                        &cv,
                        move |x, z| MaybeUninit::new(alpha * x + beta * z),
                    )
                }
                Mode::Overwrite => unreachable!("dispatched on a C term"),
            }
        })
    }

    /// `D = alpha * op_A(A) * op_B(B)`.
    fn product<OA: ElementOp<T>, OB: ElementOp<T>>(&self, exec: &Exec<'_>) -> Done {
        let alpha = self.alpha;
        self.with(exec, |dims, st| unsafe {
            let av = read_view::<T, OA>(self.ptrs[0].get(), dims, st[0]);
            let bv = read_view::<T, OB>(self.ptrs[1].get(), dims, st[1]);
            let mut dv = write_view_uninit(self.d.get(), dims, st[3]);
            if alpha == <T as Element>::one() {
                mul_into_uninit(&mut dv, &av, &bv)
            } else {
                zip_map2_into(&mut dv, &av, &bv, move |x, y| {
                    MaybeUninit::new(alpha * x * y)
                })
            }
        })
    }

    /// `D = alpha * op_A(A) * op_B(B) + beta * op_C(C)`.
    fn product_c<OA: ElementOp<T>, OB: ElementOp<T>, OC: ElementOp<T>>(
        &self,
        exec: &Exec<'_>,
    ) -> Done {
        let (alpha, beta) = (self.alpha, self.beta);
        let one = <T as Element>::one();
        self.with(exec, |dims, st| unsafe {
            let av = read_view::<T, OA>(self.ptrs[0].get(), dims, st[0]);
            let bv = read_view::<T, OB>(self.ptrs[1].get(), dims, st[1]);
            match self.mode {
                Mode::InPlace => {
                    let mut dv = write_view(self.d.get(), dims, st[3]);
                    if alpha == one && beta == one && OC::IS_IDENTITY {
                        fma(&mut dv, &av, &bv)
                    } else {
                        zip_update3_into::<T, T, T, OC, OA, OB>(
                            &mut dv,
                            &av,
                            &bv,
                            move |z, x, y| alpha * x * y + beta * z,
                        )
                    }
                }
                Mode::Separate(p) => {
                    let cv = read_view::<T, OC>(p.get(), dims, st[2]);
                    zip_map3_into(
                        &mut write_view_uninit(self.d.get(), dims, st[3]),
                        &av,
                        &bv,
                        &cv,
                        move |x, y, z| MaybeUninit::new(alpha * x * y + beta * z),
                    )
                }
                Mode::Overwrite => unreachable!("dispatched on a C term"),
            }
        })
    }
}
