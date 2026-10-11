//! Test support for the tcontract stack.
//!
//! | Module | What it is |
//! |---|---|
//! | [`oracle`] | an independent label oracle: brute force over label assignments, sharing no code with the planner |
//! | [`fixtures`] | seeded data and layout helpers |
//! | [`NaiveBackend`] | a second implementation of the contraction interface: a plain loop nest over the validated problem's roles |
//! | [`custom_kernels`] | a downstream crate's own packed micro-kernels, for the selection and ownership contracts |
//!
//! [`NaiveBackend`] exists to show that the whole-backend trait can be
//! implemented from read-only [`Problem`] roles and an [`Exec`], and to serve as
//! an independent backend in consumer-selection tests. It is not a production
//! fallback and is never selected by default. It copies nothing, so it never
//! materializes; it splits the output range over the host's barrier-free
//! partitions.
//!
//! The oracle takes the original test case (layouts and labels), not the
//! production scatter or lowering output, so it can check the lowering itself.

pub mod custom_kernels;
pub mod fixtures;
pub mod oracle;

use strided_view::{StridedView, StridedViewMut};
use tcontract::api::{
    AccumulationSource, BoxedPlan, CSpec, ContractionBackend, Diagnostics, Error, LayoutError,
    OperandId, PlanningBudget, PreparedContraction, Problem, Requirements, Result, Scalar,
};
use tcontract_exec::Exec;
use tcontract_kernel::Element;

/// The naive loop-nest backend.
#[derive(Clone, Copy, Debug, Default)]
pub struct NaiveBackend;

const ID: &str = "naive-loop-nest";

/// One axis of the loop nest: extent and the `[A, B, C, D]` strides.
#[derive(Clone, Copy, Debug)]
struct Ax {
    extent: usize,
    s: [isize; 4],
}

struct NaivePlan {
    problem: Problem,
    /// Output axes (M, N, batch), first axis fastest.
    out: Vec<Ax>,
    /// Contracted axes.
    sum: Vec<Ax>,
    diag: Diagnostics,
}

fn axes<'a>(roles: impl Iterator<Item = &'a tcontract::api::RoleAxis>) -> Vec<Ax> {
    roles
        .map(|r| Ax {
            extent: r.extent(),
            s: [OperandId::A, OperandId::B, OperandId::C, OperandId::D].map(|o| r.stride(o)),
        })
        .collect()
}

impl<T: Scalar> ContractionBackend<T> for NaiveBackend {
    fn id(&self) -> &'static str {
        ID
    }

    fn prepare(
        &self,
        problem: &Problem,
        _requirements: &Requirements,
        _budget: &PlanningBudget,
    ) -> Result<BoxedPlan<T>> {
        // Never copies, so `no_materialize` is always met.
        if problem.dtype() != T::STORAGE {
            return Err(Error::Config(tcontract::api::ConfigError::DtypeMismatch {
                plan: T::STORAGE.name(),
                problem: problem.dtype().name(),
            }));
        }
        let r = problem.roles();
        Ok(Box::new(NaivePlan {
            out: axes(r.m().iter().chain(r.n()).chain(r.h())),
            sum: axes(r.k().iter()),
            problem: problem.clone(),
            diag: Diagnostics::new(ID, "loop-nest"),
        }))
    }
}

#[derive(Clone, Copy)]
struct Raw<T>(*mut T);
// SAFETY: dereferenced only at validated offsets; writes go to disjoint,
// injective output positions (one per output element, one lane per range).
unsafe impl<T> Send for Raw<T> {}
unsafe impl<T> Sync for Raw<T> {}

impl<T> Raw<T> {
    // A method, so closures capture the wrapper and not the bare pointer field.
    fn get(self) -> *mut T {
        self.0
    }
}

impl NaivePlan {
    fn check(&self, which: OperandId, dims: &[usize], strides: &[isize]) -> Result<()> {
        let l = match which {
            OperandId::A => self.problem.a().layout(),
            OperandId::B => self.problem.b().layout(),
            OperandId::D => self.problem.d().layout(),
            OperandId::C => match self.problem.c_spec() {
                CSpec::Separate(c) => c.layout(),
                _ => self.problem.d().layout(),
            },
        };
        if l.dims() != dims || l.strides() != strides {
            return Err(LayoutError::Mismatch { operand: which }.into());
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run<T: Scalar>(
        &self,
        exec: &Exec<'_>,
        alpha: T,
        a: &StridedView<'_, T>,
        b: &StridedView<'_, T>,
        beta: T,
        c: Option<*const T>,
        in_place: bool,
        d: &mut StridedViewMut<'_, T>,
    ) -> Result<()> {
        let p = &self.problem;
        self.check(OperandId::A, a.dims(), a.strides())?;
        self.check(OperandId::B, b.dims(), b.strides())?;
        self.check(OperandId::D, d.dims(), d.strides())?;
        if p.out_empty() {
            return Ok(());
        }
        let n_out: usize = self.out.iter().map(|x| x.extent).product();
        let zero = <T as Element>::zero();
        let skip_ab = p.k_empty() || alpha == zero;
        let read_c = beta != zero && (c.is_some() || in_place);
        let (ca, cb) = (p.a().op().is_conj(), p.b().op().is_conj());
        let (cc, cd) = (p.op_c().is_conj(), p.d().op().is_conj());
        let (ap, bp, dp) = (
            Raw(a.ptr() as *mut T),
            Raw(b.ptr() as *mut T),
            Raw(d.as_mut_ptr()),
        );
        let cp = Raw(c.unwrap_or(core::ptr::null()) as *mut T);
        let op = |x: T, conj: bool| if conj { Element::conj(x) } else { x };
        let lanes = exec.budget().min(n_out).max(1);
        let body = |lane: usize| {
            let (lo, hi) = (lane * n_out / lanes, (lane + 1) * n_out / lanes);
            for t in lo..hi {
                // Unravel t over the output extents (first axis fastest).
                let (mut rem, mut off) = (t, [0isize; 4]);
                for ax in &self.out {
                    let i = (rem % ax.extent) as isize;
                    rem /= ax.extent;
                    for (o, s) in off.iter_mut().zip(ax.s) {
                        *o += i * s;
                    }
                }
                let mut acc = zero;
                if !skip_ab {
                    let mut kidx = vec![0usize; self.sum.len()];
                    'k: loop {
                        let (mut pa, mut pb) = (off[0], off[1]);
                        for (q, &i) in kidx.iter().enumerate() {
                            pa += i as isize * self.sum[q].s[0];
                            pb += i as isize * self.sum[q].s[1];
                        }
                        // SAFETY: offsets are inside the bounds-checked views.
                        let (x, y) = unsafe { (*ap.get().offset(pa), *bp.get().offset(pb)) };
                        acc = Element::add(acc, Element::mul(op(x, ca), op(y, cb)));
                        let mut q = 0;
                        loop {
                            if q == kidx.len() {
                                break 'k;
                            }
                            kidx[q] += 1;
                            if kidx[q] < self.sum[q].extent {
                                break;
                            }
                            kidx[q] = 0;
                            q += 1;
                        }
                    }
                }
                // SAFETY: `off[3]` is a distinct in-bounds output position; D is
                // exclusively borrowed and injective; a separate C is in bounds.
                unsafe {
                    let q = dp.get().offset(off[3]);
                    let mut v = Element::mul(alpha, acc);
                    if read_c {
                        let old = if in_place {
                            *q
                        } else {
                            *cp.get().offset(off[2])
                        };
                        v = Element::add(v, Element::mul(beta, op(old, cc)));
                    }
                    *q = op(v, cd);
                }
            }
        };
        exec.for_each_partition(lanes, &body);
        Ok(())
    }
}

impl<T: Scalar> PreparedContraction<T> for NaivePlan {
    fn execute_into(
        &self,
        exec: &Exec<'_>,
        alpha: T,
        a: &StridedView<'_, T>,
        b: &StridedView<'_, T>,
        d: &mut StridedViewMut<'_, T>,
    ) -> Result<()> {
        self.run(exec, alpha, a, b, <T as Element>::zero(), None, false, d)
    }

    fn execute_into_accum(
        &self,
        exec: &Exec<'_>,
        alpha: T,
        a: &StridedView<'_, T>,
        b: &StridedView<'_, T>,
        beta: T,
        source: AccumulationSource<'_, T>,
        d: &mut StridedViewMut<'_, T>,
    ) -> Result<()> {
        match (self.problem.c_spec(), source) {
            (_, AccumulationSource::Absent) if beta == <T as Element>::zero() => {
                self.run(exec, alpha, a, b, beta, None, false, d)
            }
            (CSpec::Output(_), AccumulationSource::Output) => {
                self.run(exec, alpha, a, b, beta, None, true, d)
            }
            (CSpec::Separate(_), AccumulationSource::Separate(c)) => {
                self.check(OperandId::C, c.dims(), c.strides())?;
                self.run(exec, alpha, a, b, beta, Some(c.ptr()), false, d)
            }
            _ => Err(LayoutError::CMode.into()),
        }
    }

    fn diagnostics(&self) -> &Diagnostics {
        &self.diag
    }
}

// Plans are shared across threads: the plan holds only owned metadata.
const _: fn() = || {
    fn assert<X: Send + Sync>() {}
    assert::<NaivePlan>();
};
