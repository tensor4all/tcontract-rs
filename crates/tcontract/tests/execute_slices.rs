//! `Plan::execute_slices` runs the same operation as `Plan::execute_into` on
//! plain slices and origins: bitwise equal results on the faer, packed and
//! elementwise strategies, with origins away from the slice start and negative
//! strides, and `LayoutError::Bounds` (nothing written) for a slice that
//! misses any addressed element. The plan is always built at offset zero: the
//! origin is a per-call argument, as for views.
use strided_view::{StridedView, StridedViewMut};
use tcontract::api::{
    CSpec, DType, Error, Labels, LayoutError, LayoutSpec, OperandId, OperandSpec, Problem,
};
use tcontract::{Algorithm, Plan, PlanConfig};
use tcontract_exec::Exec;

struct Operand {
    dims: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
    len: usize,
}

impl Operand {
    fn spec(&self) -> OperandSpec {
        OperandSpec::new(LayoutSpec::new(&self.dims, &self.strides, 0).unwrap())
    }
}

/// Column-major, offset zero.
fn cm(dims: &[usize]) -> Operand {
    let mut strides = Vec::new();
    let mut n = 1;
    for &d in dims {
        strides.push(n as isize);
        n *= d;
    }
    Operand {
        dims: dims.to_vec(),
        strides,
        offset: 0,
        len: n,
    }
}

/// Column-major with the first axis reversed, stored after `pad` leading
/// elements: a negative stride and a nonzero offset.
fn reversed(dims: &[usize], pad: usize) -> Operand {
    let mut o = cm(dims);
    o.strides[0] = -1;
    o.offset = pad as isize + dims[0] as isize - 1;
    o.len += pad;
    o
}

fn values(n: usize, salt: usize) -> Vec<f64> {
    (0..n)
        .map(|i| ((i * 7 + salt * 3) % 11) as f64 - 5.0)
        .collect()
}

struct Case {
    a: Operand,
    b: Operand,
    d: Operand,
    labels: [Vec<i64>; 3],
}

fn plan(c: &Case, config: &PlanConfig) -> Plan<f64> {
    let [la, lb, ld] = &c.labels;
    let p = Problem::from_labels(
        DType::F64,
        c.a.spec(),
        c.b.spec(),
        CSpec::Absent,
        c.d.spec(),
        &Labels::new(la, lb, ld),
    )
    .unwrap();
    Plan::new(&p, config).unwrap()
}

fn check(c: &Case, config: &PlanConfig, expect: Algorithm) {
    let plan = plan(c, config);
    assert_eq!(plan.report().algorithm, expect);
    let (a, b) = (values(c.a.len, 1), values(c.b.len, 2));
    let mut exec = Exec::serial();

    let mut by_view = vec![99.0; c.d.len];
    {
        let av = StridedView::new(&a, &c.a.dims, &c.a.strides, c.a.offset).unwrap();
        let bv = StridedView::new(&b, &c.b.dims, &c.b.strides, c.b.offset).unwrap();
        let mut dv =
            StridedViewMut::new(&mut by_view, &c.d.dims, &c.d.strides, c.d.offset).unwrap();
        plan.execute_into(&mut exec, 2.0, &av, &bv, &mut dv)
            .unwrap();
    }
    let mut by_slice = vec![99.0; c.d.len];
    plan.execute_slices(
        &mut exec,
        2.0,
        (&a, c.a.offset),
        (&b, c.b.offset),
        (&mut by_slice, c.d.offset),
    )
    .unwrap();
    assert_eq!(by_slice, by_view);
}

fn matmul(a: Operand, b: Operand, d: Operand) -> Case {
    Case {
        a,
        b,
        d,
        labels: [vec![0, 2], vec![2, 1], vec![0, 1]],
    }
}

#[test]
fn matches_execute_into_on_every_strategy() {
    let c = matmul(cm(&[5, 4]), cm(&[4, 3]), cm(&[5, 3]));
    check(&c, &PlanConfig::default(), Algorithm::Faer);
    check(&c, &PlanConfig::packed(), Algorithm::Packed);
    let hadamard = Case {
        a: cm(&[5, 4]),
        b: cm(&[5, 4]),
        d: cm(&[5, 4]),
        labels: [vec![0, 1], vec![0, 1], vec![0, 1]],
    };
    check(&hadamard, &PlanConfig::default(), Algorithm::Elementwise);
}

#[test]
fn honours_offsets_and_negative_strides() {
    let c = matmul(
        reversed(&[5, 4], 3),
        reversed(&[4, 3], 0),
        reversed(&[5, 3], 2),
    );
    check(&c, &PlanConfig::default(), Algorithm::Faer);
    check(&c, &PlanConfig::packed(), Algorithm::Packed);
}

#[test]
fn ignores_elements_outside_the_addressed_range() {
    // Longer slices than the layouts address: the tail is neither read nor
    // written.
    let mut c = matmul(cm(&[3, 2]), cm(&[2, 3]), cm(&[3, 3]));
    let plan = plan(&c, &PlanConfig::default());
    c.d.len += 4;
    let mut a = values(6, 1);
    a.push(f64::NAN);
    let b = values(6, 2);
    let mut d = vec![7.0; c.d.len];
    plan.execute_slices(&mut Exec::serial(), 1.0, (&a, 0), (&b, 0), (&mut d, 0))
        .unwrap();
    assert!(d[..9].iter().all(|x| x.is_finite()));
    assert_eq!(&d[9..], &[7.0; 4]);
}

#[test]
fn refuses_a_short_slice_for_each_operand_and_writes_nothing() {
    let c = matmul(reversed(&[5, 4], 3), cm(&[4, 3]), reversed(&[5, 3], 2));
    let plan = plan(&c, &PlanConfig::default());
    let (a, b) = (values(c.a.len, 1), values(c.b.len, 2));
    let mut exec = Exec::serial();
    let bounds = |r: Result<(), Error>| match r {
        Err(Error::Layout(LayoutError::Bounds { operand })) => operand,
        other => panic!("expected a bounds error, got {other:?}"),
    };

    let (ao, d_o) = (c.a.offset, c.d.offset);
    let mut d = vec![7.0; c.d.len];
    let short = |v: &[f64]| v[..v.len() - 1].to_vec();
    assert_eq!(
        bounds(plan.execute_slices(&mut exec, 1.0, (&short(&a), ao), (&b, 0), (&mut d, d_o))),
        OperandId::A
    );
    assert_eq!(
        bounds(plan.execute_slices(&mut exec, 1.0, (&a, ao), (&short(&b), 0), (&mut d, d_o))),
        OperandId::B
    );
    let mut short_d = vec![7.0; c.d.len - 1];
    assert_eq!(
        bounds(plan.execute_slices(&mut exec, 1.0, (&a, ao), (&b, 0), (&mut short_d, d_o))),
        OperandId::D
    );
    assert!(d.iter().chain(&short_d).all(|&x| x == 7.0));
}

#[test]
fn refuses_an_offset_that_reaches_before_the_slice() {
    // Reversed first axis with origin 3 < extent - 1: the lowest addressed
    // element is at -1.
    let mut a = cm(&[5, 4]);
    a.strides[0] = -1;
    let c = matmul(a, cm(&[4, 3]), cm(&[5, 3]));
    let plan = plan(&c, &PlanConfig::default());
    let mut d = vec![0.0; 15];
    let b = values(12, 2);
    let r = plan.execute_slices(
        &mut Exec::serial(),
        1.0,
        (&[0.0; 64], 3),
        (&b, 0),
        (&mut d, 0),
    );
    assert!(matches!(
        r,
        Err(Error::Layout(LayoutError::Bounds {
            operand: OperandId::A
        }))
    ));
}

#[test]
fn empty_operands_need_no_storage() {
    // Empty output: nothing is addressed, empty slices are accepted.
    let c = matmul(cm(&[0, 4]), cm(&[4, 3]), cm(&[0, 3]));
    let p = plan(&c, &PlanConfig::default());
    p.execute_slices(
        &mut Exec::serial(),
        1.0,
        (&[], 0),
        (&values(12, 2), 0),
        (&mut [], 0),
    )
    .unwrap();
    // Empty contraction: D is overwritten with zeros, A and B are not read.
    let c = matmul(cm(&[3, 0]), cm(&[0, 2]), cm(&[3, 2]));
    let p = plan(&c, &PlanConfig::default());
    let mut d = vec![7.0; 6];
    p.execute_slices(&mut Exec::serial(), 1.0, (&[], 0), (&[], 0), (&mut d, 0))
        .unwrap();
    assert_eq!(d, vec![0.0; 6]);
}
