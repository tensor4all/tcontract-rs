//! Using the engine: four contractions a matrix multiply cannot express.
//!
//! ```text
//! cargo run --release -p tcontract --example contract
//! ```
//!
//! This is **documentation that compiles**, not a measurement: the tensors are
//! tiny, nothing is timed, and every result is checked against a value written
//! out by hand. It costs no CPU and is safe to run while a benchmark is in
//! flight.
//!
//! Each section is the smallest case that shows one feature of the label
//! notation: a batch index, a reduction, a diagonal, and complex conjugation
//! with plan reuse.

use num_complex::Complex64 as C64;
use strided_view::{StridedView, StridedViewMut};
use tcontract::api::{CSpec, DType, Labels, LayoutSpec, Op, OperandSpec, Problem};
use tcontract::{Plan, PlanConfig, Result};
use tcontract_exec::Exec;

/// Column-major extents and strides, as a validated operand description.
fn col_major(dims: &[usize], op: Op) -> Result<OperandSpec> {
    let mut strides = Vec::new();
    let mut acc = 1isize;
    for &d in dims {
        strides.push(acc);
        acc *= d as isize;
    }
    Ok(OperandSpec::new(LayoutSpec::new(dims, &strides, 0)?).with_op(op))
}

/// Labels from an einsum-like spelling such as `"hik,hkj->hij"`.
fn parse(spec: &str) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let (lhs, d) = spec
        .split_once("->")
        .expect("output labels are not optional");
    let (a, b) = lhs.split_once(',').expect("two input operands");
    let lab = |s: &str| s.chars().map(|c| c as i64).collect();
    (lab(a), lab(b), lab(d))
}

// Every fallible call below is `?`, not `.unwrap()`, because this file is read
// as an example of how to call the crate and the difference is visible.
fn main() -> Result<()> {
    batch_index()?;
    reduction()?;
    diagonal()?;
    complex_with_plan_reuse()?;
    println!("\nall four contractions matched their expected values");
    Ok(())
}

/// A label in `A`, `B` *and* `D` is a batch index: the contraction runs
/// independently for each of its values, with no reshaping and no loop here.
fn batch_index() -> Result<()> {
    // D[h,i,j] = sum_k A[h,i,k] * B[h,k,j], for each h.
    let (ia, ib, id) = parse("hik,hkj->hij");
    let dims = [2, 2, 2];
    let problem = Problem::from_labels(
        DType::F64,
        col_major(&dims, Op::Identity)?,
        col_major(&dims, Op::Identity)?,
        CSpec::Absent,
        col_major(&dims, Op::Identity)?,
        &Labels::new(&ia, &ib, &id),
    )?;
    let plan = Plan::<f64>::new(&problem, &PlanConfig::default())?;

    // A[h,i,k] = h + 2i + 4k + 1, since the layout is column-major over (h,i,k).
    let a: Vec<f64> = (1..=8).map(|x| x as f64).collect();
    // Per batch, B is diag(2, 1) over (k, j): it doubles k=0 and passes k=1.
    let b = vec![2.0f64, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
    let mut d = vec![0.0f64; 8];
    execute(&plan, &dims, &dims, &dims, &a, &b, &mut d)?;

    // B being diagonal means only k=j contributes, so D[h,i,j] = B[h,j,j]*A[h,i,j]:
    // the j=0 half of D (offsets 0..4) doubles, and the j=1 half passes through.
    assert_eq!(d, vec![2.0, 4.0, 6.0, 8.0, 5.0, 6.0, 7.0, 8.0]);
    report("batch index", "hik,hkj->hij", &d);
    Ok(())
}

/// A label in one input only is summed over, and needs no temporary: it becomes
/// a contraction index with stride 0 in the other operand, which the
/// block-scatter machinery treats as a perfectly regular access.
fn reduction() -> Result<()> {
    // D[i,j] = sum_{k,l} A[i,k,l] * B[k,j] -- `l` appears nowhere else.
    let (ia, ib, id) = parse("ikl,kj->ij");
    let (da, db, dd) = ([2, 2, 3], [2, 2], [2, 2]);
    let problem = Problem::from_labels(
        DType::F64,
        col_major(&da, Op::Identity)?,
        col_major(&db, Op::Identity)?,
        CSpec::Absent,
        col_major(&dd, Op::Identity)?,
        &Labels::new(&ia, &ib, &id),
    )?;
    let plan = Plan::<f64>::new(&problem, &PlanConfig::default())?;

    // Every element 1.0, so summing over l (extent 3) triples each entry.
    let a = vec![1.0f64; 12];
    let identity = vec![1.0f64, 0.0, 0.0, 1.0];
    let mut d = vec![0.0f64; 4];
    execute(&plan, &da, &db, &dd, &a, &identity, &mut d)?;

    assert_eq!(d, vec![3.0, 3.0, 3.0, 3.0]);
    report("reduction", "ikl,kj->ij", &d);
    Ok(())
}

/// A label repeated *within one operand* selects that operand's diagonal, again
/// with no copy: the two modes' strides are simply summed.
fn diagonal() -> Result<()> {
    // D[i] = sum_j A[i,j,j] * B[i] -- `j` twice in A takes its diagonal.
    let (da, dv) = ([2, 2, 2], [2]);
    let (i, j) = (b'i' as i64, b'j' as i64);
    let problem = Problem::from_labels(
        DType::F64,
        col_major(&da, Op::Identity)?,
        col_major(&dv, Op::Identity)?,
        CSpec::Absent,
        col_major(&dv, Op::Identity)?,
        &Labels::new(&[i, j, j], &[i], &[i]),
    )?;
    let plan = Plan::<f64>::new(&problem, &PlanConfig::default())?;

    // A[i,j,j] picks offsets i + 2j + 4j = i, i+6 -> elements 0,1 and 6,7.
    let a: Vec<f64> = (0..8).map(|x| x as f64).collect();
    let b = vec![1.0f64, 1.0];
    let mut d = vec![0.0f64; 2];
    execute(&plan, &da, &dv, &dv, &a, &b, &mut d)?;

    // i=0: A[0,0,0] + A[0,1,1] = 0 + 6. i=1: A[1,0,0] + A[1,1,1] = 1 + 7.
    assert_eq!(d, vec![6.0, 8.0]);
    report("diagonal", "ijj,i->i", &d);
    Ok(())
}

/// Complex contraction, conjugation, and the reason to build a [`Plan`] by hand:
/// planning is `O(M + N + K)`, so a repeated shape should pay it once.
fn complex_with_plan_reuse() -> Result<()> {
    let (ia, ib, id) = parse("ik,kj->ij");
    let dims = [2, 2];
    // Conjugation is part of the problem, fixed when the plan is built.
    let problem = Problem::from_labels(
        DType::C64,
        col_major(&dims, Op::Conjugate)?,
        col_major(&dims, Op::Identity)?,
        CSpec::Absent,
        col_major(&dims, Op::Identity)?,
        &Labels::new(&ia, &ib, &id),
    )?;
    let plan = Plan::<C64>::new(&problem, &PlanConfig::default())?;

    let identity = vec![
        C64::new(1.0, 0.0),
        C64::new(0.0, 0.0),
        C64::new(0.0, 0.0),
        C64::new(1.0, 0.0),
    ];

    // One plan, two different operands.
    for scale in [1.0f64, 10.0] {
        let a: Vec<C64> = (1..=4)
            .map(|x| C64::new(x as f64 * scale, x as f64))
            .collect();
        let mut d = vec![C64::new(0.0, 0.0); 4];
        execute(&plan, &dims, &dims, &dims, &a, &identity, &mut d)?;

        let want: Vec<C64> = a.iter().map(|z| z.conj()).collect();
        assert_eq!(d, want);
        println!(
            "complex, conjugated  ik,kj->ij  scale {scale:>4}  ->  [{}]",
            d.iter()
                .map(|z| format!("{}{:+}i", z.re, z.im))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    Ok(())
}

/// Run `plan` on column-major buffers of the given extents (`beta = 0`).
fn execute<T: tcontract::api::Scalar>(
    plan: &Plan<T>,
    da: &[usize],
    db: &[usize],
    dd: &[usize],
    a: &[T],
    b: &[T],
    d: &mut [T],
) -> Result<()> {
    let strides = |dims: &[usize]| -> Vec<isize> {
        let mut s = Vec::new();
        let mut acc = 1isize;
        for &x in dims {
            s.push(acc);
            acc *= x as isize;
        }
        s
    };
    let (sa, sb, sd) = (strides(da), strides(db), strides(dd));
    let view = |x, dims, st| StridedView::new(x, dims, st, 0).map_err(tcontract::Error::backend);
    plan.execute_into(
        &mut Exec::serial(),
        <T as tcontract_kernel::Element>::one(),
        &view(a, da, &sa)?,
        &view(b, db, &sb)?,
        &mut StridedViewMut::new(d, dd, &sd, 0).map_err(tcontract::Error::backend)?,
    )
}

fn report(what: &str, spec: &str, d: &[f64]) {
    println!("{what:<20} {spec:<14} ->  {d:?}");
}
