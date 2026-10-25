//! The labels-based unary update.
//!
//! One operation covers what TensorOperations calls `tensoradd!` (a
//! permutation) and `tensortrace!` (a repeated label is a diagonal, a label
//! absent from the output is summed), which is the primitive TensorKit.jl
//! needs from a CPU backend, and what TBLIS exposes as `tblis_tensor_add`:
//!
//! ```text
//! D = alpha * op_A(A[labels_a]) + beta * D_old
//! ```
//!
//! It lowers to the same validated [`Problem`] and
//! [`Plan`] as every other contraction, with `B` a rank-zero scalar: the
//! planner, the strategies and the accumulation rules are the ones this crate
//! already has, and a caller that keeps a plan can run the same update on
//! different buffers with [`Plan::execute_into_accum`] directly.

use strided_view::{StridedView, StridedViewMut};
use tcontract_exec::Exec;
use tcontract_kernel::Element;

use crate::api::{
    AccumulationSource, CSpec, Error, Labels, LayoutSpec, Op, OperandSpec, Problem, Result, Scalar,
};
use crate::plan::{Plan, PlanConfig};

/// The labels of a unary update: those of `A` and those of `D`.
///
/// A label repeated within `A` selects a diagonal; a label of `A` absent from
/// `D` is summed; `D`'s label order is free, so a permutation is one case of
/// the same operation. A label of `D` that `A` does not carry is refused by
/// [`Problem::from_labels`].
///
/// # Examples
///
/// ```
/// use tcontract::unary::Unary;
/// let u = Unary::new(&[0, 1], &[1, 0]);
/// assert_eq!(u.labels_a(), &[0, 1]);
/// assert_eq!(u.labels_d(), &[1, 0]);
/// assert!(!u.conj_a());
/// assert!(Unary::new(&[0], &[0]).with_conj_a().conj_a());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unary<'a> {
    labels_a: &'a [i64],
    labels_d: &'a [i64],
    conj_a: bool,
}

impl<'a> Unary<'a> {
    /// The update `D[labels_d] = alpha * A[labels_a] + beta * D_old`.
    pub fn new(labels_a: &'a [i64], labels_d: &'a [i64]) -> Self {
        Self {
            labels_a,
            labels_d,
            conj_a: false,
        }
    }

    /// Use the complex conjugate of `A` (no effect on a real type).
    #[must_use]
    pub fn with_conj_a(mut self) -> Self {
        self.conj_a = true;
        self
    }

    /// Labels of `A`.
    pub fn labels_a(&self) -> &'a [i64] {
        self.labels_a
    }

    /// Labels of `D`.
    pub fn labels_d(&self) -> &'a [i64] {
        self.labels_d
    }

    /// Whether `A` is conjugated.
    pub fn conj_a(&self) -> bool {
        self.conj_a
    }
}

/// `D = alpha * op_A(A[labels_a]) + beta * D_old` over any signed strides.
///
/// `beta == 0` never reads the previous output; otherwise the update
/// accumulates in place. `alpha == 0` computes `beta * D_old` and reads no `A`.
///
/// # Errors
///
/// [`Error::Config`]/[`Error::Shape`]/[`Error::Unsupported`] from
/// [`Problem::from_labels`] for labels that do not describe an operation
/// (`labels_a` and `labels_d` must have one entry per element of `A` and `D`),
/// [`Error::Alias`] for a non-injective `D`, and [`Error::Exec`] from the
/// execution.
///
/// # Examples
///
/// ```
/// use strided_view::{StridedView, StridedViewMut};
/// use tcontract::unary::{add, Unary};
/// use tcontract_exec::Exec;
///
/// // C[j] = sum_i A[i, j], column-major 3 x 2.
/// let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let mut c = [0.0; 2];
/// let av = StridedView::new(&a, &[3, 2], &[1, 3], 0).unwrap();
/// let mut cv = StridedViewMut::new(&mut c, &[2], &[1], 0).unwrap();
/// // Label 0 is summed: it is in `A` and not in `D`.
/// add(&mut Exec::serial(), &Unary::new(&[0, 1], &[1]), 1.0, &av, 0.0, &mut cv).unwrap();
/// assert_eq!(c, [6.0, 15.0]);
/// ```
pub fn add<T: Scalar>(
    exec: &mut Exec<'_>,
    spec: &Unary<'_>,
    alpha: T,
    a: &StridedView<'_, T>,
    beta: T,
    d: &mut StridedViewMut<'_, T>,
) -> Result<()> {
    let a_spec = OperandSpec::new(LayoutSpec::new(a.dims(), a.strides(), a.offset())?).with_op(
        if spec.conj_a {
            Op::Conjugate
        } else {
            Op::Identity
        },
    );
    let d_spec = OperandSpec::new(LayoutSpec::new(d.dims(), d.strides(), d.offset())?);
    let b_spec = OperandSpec::new(LayoutSpec::new(&[], &[], 0)?);
    let c = if beta == <T as Element>::zero() {
        CSpec::Absent
    } else {
        CSpec::Output(Op::Identity)
    };
    let problem = Problem::from_labels(
        <T as Scalar>::STORAGE,
        a_spec,
        b_spec,
        c,
        d_spec,
        &Labels::new(spec.labels_a, &[], spec.labels_d),
    )?;
    let plan = Plan::<T>::new(&problem, &PlanConfig::default())?;
    // `B` is the scalar one: a rank-zero layout addresses exactly one element,
    // so this view is either valid or the rank-zero convention changed.
    let one = [<T as Element>::one()];
    let b = StridedView::new(&one[..], &[], &[], 0).map_err(Error::backend)?;
    if beta == <T as Element>::zero() {
        plan.execute_into(exec, alpha, a, &b, d)
    } else {
        plan.execute_into_accum(exec, alpha, a, &b, beta, AccumulationSource::Output, d)
    }
}
