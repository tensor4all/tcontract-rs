//! The public description of a contraction: front ends, the validated
//! [`Problem`], the error type and the whole-backend trait.

mod backend;
mod dot_general;
mod error;
mod labels;
mod problem;
pub(crate) mod scalar;
mod validate;

pub use backend::{
    AccumulationSource, BoxedPlan, ContractionBackend, Diagnostics, PlanningBudget,
    PreparedContraction, Requirements,
};
pub use dot_general::{DotGeneral, DotShape};
pub use error::{
    AliasError, ConfigError, Error, LayoutError, OperandId, Result, ShapeError, Unsupported,
};
pub use labels::Labels;
pub use problem::{CSpec, DType, LayoutSpec, Op, OperandSpec, Problem, RoleAxis, Roles, Span};
#[doc(hidden)]
pub use scalar::FaerGemm;
pub use scalar::Scalar;
pub use validate::is_injective_layout;
