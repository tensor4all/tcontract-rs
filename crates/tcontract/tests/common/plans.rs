//! Plan builders shared by the packed-driver integration tests.
#![allow(dead_code)]

use tcontract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
use tcontract::{Plan, PlanConfig};

/// `D[i,k] = sum_j A[i,j] B[j,k]` on column-major `m x k`, `k x n`, `m x n`.
pub fn matmul_problem(m: usize, n: usize, k: usize) -> Problem {
    let spec = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
    Problem::from_labels(
        DType::F64,
        spec(&[m, k], &[1, m as isize]),
        spec(&[k, n], &[1, k as isize]),
        CSpec::Absent,
        spec(&[m, n], &[1, m as isize]),
        &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    )
    .unwrap()
}

/// An explicit grid request: forces the packed driver with the planner's own
/// grid rule.
pub fn packed() -> PlanConfig {
    PlanConfig::packed()
}

pub fn packed_plan(m: usize, n: usize, k: usize) -> Plan<f64> {
    Plan::<f64>::new(&matmul_problem(m, n, k), &packed()).unwrap()
}
