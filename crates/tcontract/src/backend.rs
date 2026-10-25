//! The tcontract implementation of the neutral contraction interface.

use crate::api::{
    BoxedPlan, ContractionBackend, PlanningBudget, Problem, Requirements, Result, Scalar,
};
use crate::plan::{Plan, PlanConfig};

const ID: &str = "tcontract";

/// The tcontract contraction implementation behind the neutral interface.
///
/// It owns a [`PlanConfig`]; preparation applies the consumer's
/// [`Requirements`] without mutating the factory (the effective
/// `no_materialize` is the OR of the two). A prepared [`Plan`] implements
/// [`PreparedContraction`](crate::api::PreparedContraction); the box is
/// allocated once, at prepare, and no per-tile virtual call exists.
///
/// # Examples
///
/// ```
/// use strided_view::{StridedView, StridedViewMut};
/// use tcontract::api::*;
/// use tcontract::TcontractBackend;
/// use tcontract_exec::Exec;
///
/// let backend: Box<dyn ContractionBackend<f64>> = Box::new(TcontractBackend::default());
/// let l = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
/// let problem = Problem::from_dot_general(
///     DType::F64, l(&[2, 2], &[1, 2]), l(&[2, 2], &[1, 2]), l(&[2, 2], &[1, 2]),
///     &DotGeneral::new(&[1], &[0], &[], &[]),
/// ).unwrap();
/// let plan = backend.prepare(&problem, &Requirements::new(), &PlanningBudget::serial()).unwrap();
/// let (a, b, mut d) = ([1.0, 2.0, 3.0, 4.0], [1.0, 0.0, 0.0, 1.0], [0.0; 4]);
/// let av = StridedView::new(&a, &[2, 2], &[1, 2], 0).unwrap();
/// let bv = StridedView::new(&b, &[2, 2], &[1, 2], 0).unwrap();
/// let mut dv = StridedViewMut::new(&mut d, &[2, 2], &[1, 2], 0).unwrap();
/// plan.execute_into(&mut Exec::serial(), 1.0, &av, &bv, &mut dv).unwrap();
/// assert_eq!(d, a);
/// ```
#[derive(Clone, Debug, Default)]
pub struct TcontractBackend {
    /// Planning options every prepared plan starts from.
    pub config: PlanConfig,
}

impl<T: Scalar> ContractionBackend<T> for TcontractBackend {
    fn id(&self) -> &'static str {
        ID
    }

    fn prepare(
        &self,
        problem: &Problem,
        requirements: &Requirements,
        _budget: &PlanningBudget,
    ) -> Result<BoxedPlan<T>> {
        let mut config = self.config.clone();
        config.no_materialize |= requirements.no_materialize;
        Ok(Box::new(Plan::<T>::new(problem, &config)?))
    }
}
