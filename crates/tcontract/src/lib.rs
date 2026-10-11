//! Binary tensor contraction for the tcontract stack.
//!
//! A contraction is described once, as a validated [`api::Problem`] built from
//! one of two front ends -- [`api::Labels`] (named axes, diagonals, reductions,
//! a separately described `C`) or [`api::DotGeneral`] (paired contracting and
//! batch axes) -- and planned once into a [`Plan`] that owns everything that
//! depends only on shapes, strides and roles:
//!
//! ```text
//! D = op_D( alpha * op_A(A) * op_B(B) + beta * op_C(C) )
//! ```
//!
//! The planner picks one of three strategies (see [`Plan`] for the rules): the
//! packed TBLIS-style driver of Lukas Devos's tensorprimitives-rs (D. A.
//! Matthews, *High-Performance Tensor Contraction without Transposition*,
//! arXiv:1607.00291), which packs general strides straight into micro-kernel
//! panels; faer on a copy-free fusion to one strided batched GEMM; and an
//! elementwise pass for all-batch problems.
//!
//! [`contract_batched`] runs many independent items of one plan with the batch
//! as the parallel axis, the regime where per-contraction threading loses.
//!
//! [`unary`] is the same machinery for the one-operand update `D = alpha *
//! op_A(A[labels]) + beta * D_old`, which covers a permutation, a diagonal and
//! a reduction in one operation.
//!
//! The neutral [`api::ContractionBackend`] / [`api::PreparedContraction`]
//! traits are the extension seam for a second implementation; [`TcontractBackend`]
//! is this crate's.
//!
//! # Examples
//!
//! ```
//! use strided_view::{StridedView, StridedViewMut};
//! use tcontract::api::{DotGeneral, DType, LayoutSpec, OperandSpec, Problem};
//! use tcontract::{Plan, PlanConfig};
//! use tcontract_exec::Exec;
//!
//! // D[i, k] = sum_j A[i, j] B[j, k]
//! let l = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
//! let dot = DotGeneral::new(&[1], &[0], &[], &[]);
//! let problem = Problem::from_dot_general(
//!     DType::F64, l(&[2, 2], &[1, 2]), l(&[2, 2], &[1, 2]), l(&[2, 2], &[1, 2]), &dot,
//! ).unwrap();
//! let plan = Plan::<f64>::new(&problem, &PlanConfig::default()).unwrap();
//! let a = [1.0, 2.0, 3.0, 4.0];
//! let b = [1.0, 0.0, 0.0, 1.0];
//! let mut d = [0.0; 4];
//! let av = StridedView::new(&a, &[2, 2], &[1, 2], 0).unwrap();
//! let bv = StridedView::new(&b, &[2, 2], &[1, 2], 0).unwrap();
//! let mut dv = StridedViewMut::new(&mut d, &[2, 2], &[1, 2], 0).unwrap();
//! plan.execute_into(&Exec::serial(), 1.0, &av, &bv, &mut dv).unwrap();
//! assert_eq!(d, a);
//! ```
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]

pub mod api;
mod backend;
mod batch;
mod buffer;
mod driver;
#[cfg(feature = "phase-timing")]
mod phase;
mod plan;
mod resolve;
mod select;
mod strategy;
pub mod unary;
mod wrappers;

pub use api::{Error, Result};
pub use backend::TcontractBackend;
pub use batch::{contract_batched, BatchItem};
pub use driver::{Assignment, DynSnapshot, DynStats, DynamicReport};
pub use plan::{
    Algorithm, Axis, CacheModel, ExecutionRoute, FaerLimit, Orient, OutputContract, PackedReport,
    PackedRoute, Partition, Plan, PlanConfig, PlanReport, PlanStats, Reason, RowBlock,
    SliceAccumulationSource, Writeback, NS_PER_FLOP,
};
pub use select::{Chooser, KernelCandidate, OperandMeta, Selection, SelectionContext};
pub use wrappers::{add, permute};
