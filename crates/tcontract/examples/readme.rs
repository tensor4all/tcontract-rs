//! The minimal example of the repository README (a test checks that the README
//! shows this file verbatim): `D = A * B` through `DotGeneral`, `Plan`, execute.
use strided_view::{StridedView, StridedViewMut};
use tcontract::api::{DType, DotGeneral, LayoutSpec, OperandSpec, Problem};
use tcontract::{Plan, PlanConfig, Result};
use tcontract_exec::Exec;

fn main() -> Result<()> {
    // D[i, k] = sum_j A[i, j] * B[j, k], all 2 x 2 and column-major.
    let layout = |dims: &[usize], strides: &[isize]| -> Result<OperandSpec> {
        Ok(OperandSpec::new(LayoutSpec::new(dims, strides, 0)?))
    };
    let dot = DotGeneral::new(&[1], &[0], &[], &[]);
    let problem = Problem::from_dot_general(
        DType::F64,
        layout(&[2, 2], &[1, 2])?,
        layout(&[2, 2], &[1, 2])?,
        layout(&[2, 2], &[1, 2])?,
        &dot,
    )?;

    // Planning chooses the strategy and kernel once; the plan is reusable.
    let plan = Plan::<f64>::new(&problem, &PlanConfig::default())?;
    println!("strategy: {:?}", plan.report().algorithm);

    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [0.0, 1.0, 1.0, 0.0]; // swaps the columns of A
    let mut d = [0.0; 4];
    let av = StridedView::new(&a, &[2, 2], &[1, 2], 0).map_err(tcontract::Error::backend)?;
    let bv = StridedView::new(&b, &[2, 2], &[1, 2], 0).map_err(tcontract::Error::backend)?;
    let mut dv =
        StridedViewMut::new(&mut d, &[2, 2], &[1, 2], 0).map_err(tcontract::Error::backend)?;
    plan.execute_into(&Exec::serial(), 1.0, &av, &bv, &mut dv)?;

    assert_eq!(d, [3.0, 4.0, 1.0, 2.0]);
    println!("D = {d:?}");
    Ok(())
}
