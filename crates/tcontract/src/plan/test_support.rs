//! Shared builders of the plan unit tests.

use crate::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};

use super::{PackedPlan, PlanConfig};

/// Extents and strides of one test operand.
pub(crate) struct L {
    dims: Vec<usize>,
    strides: Vec<isize>,
}

/// A column-major layout.
pub(crate) fn lay(e: &[i64]) -> L {
    let dims: Vec<usize> = e.iter().map(|&x| x as usize).collect();
    let mut strides = Vec::new();
    let mut acc = 1isize;
    for &d in &dims {
        strides.push(acc);
        acc *= d.max(1) as isize;
    }
    L { dims, strides }
}

/// A layout with explicit strides.
pub(crate) fn laying(e: &[i64], s: &[i64]) -> L {
    L {
        dims: e.iter().map(|&x| x as usize).collect(),
        strides: s.iter().map(|&x| x as isize).collect(),
    }
}

fn spec(l: &L) -> OperandSpec {
    OperandSpec::new(LayoutSpec::new(&l.dims, &l.strides, 0).unwrap())
}

/// The problem `D[ld] = A[la] B[lb]` with the given layouts.
pub(crate) fn problem(a: &L, la: &[i64], b: &L, lb: &[i64], d: &L, ld: &[i64]) -> Problem {
    Problem::from_labels(
        DType::F64,
        spec(a),
        spec(b),
        CSpec::Absent,
        spec(d),
        &Labels::new(la, lb, ld),
    )
    .unwrap()
}

/// The packed plan of [`problem`] under the default configuration.
pub(crate) fn build(a: &L, la: &[i64], b: &L, lb: &[i64], d: &L, ld: &[i64]) -> PackedPlan {
    PackedPlan::from_problem(&problem(a, la, b, lb, d, ld), &PlanConfig::default()).unwrap()
}
