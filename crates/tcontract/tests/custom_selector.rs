//! Non-fusable tensor contraction with the downstream crate's own kernels.
use std::cell::Cell;

use strided_view::{StridedView, StridedViewMut};
use tcontract::api::{
    AccumulationSource, ConfigError, DType, DotGeneral, Error, LayoutSpec, Op, OperandSpec, Problem,
};
use tcontract::{Algorithm, Plan, PlanConfig};
use tcontract_exec::Exec;
use tcontract_kernel::{KernelCatalog, KernelHandle, SelectError};
use tcontract_testkit::custom_kernels as own;
use tcontract_testkit::fixtures::{col_major, small_ints as ints};

fn catalog(list: &'static [&'static tcontract_kernel::KernelFamily<f64>]) -> KernelCatalog<f64> {
    // SAFETY: the testkit's families are immutable, `'static` descriptors that
    // meet the family contract (checked by their own tests).
    unsafe { KernelCatalog::<f64>::from_static_families(list) }.unwrap()
}

/// The typed selection error a contraction error carries.
fn typed(e: &Error) -> &SelectError {
    match e {
        Error::Select(s) => s,
        _ => panic!("not a typed selection error: {e:?}"),
    }
}

// C[i, n] = sum_{j<4, k<5} A[i, j, k] * B[k, n, j]: two contracted axes in
// permuted order, so faer would have to copy an operand.
const A_DIMS: [usize; 3] = [3, 4, 5];
const B_DIMS: [usize; 3] = [5, 6, 4];
const C_DIMS: [usize; 2] = [3, 6];

fn cfg() -> DotGeneral {
    DotGeneral::new(&[1, 2], &[2, 0], &[], &[])
}

fn problem() -> Problem {
    let (sa, sb, sc) = (col_major(&A_DIMS), col_major(&B_DIMS), col_major(&C_DIMS));
    let spec = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
    Problem::from_dot_general(
        DType::F64,
        spec(&A_DIMS, &sa),
        spec(&B_DIMS, &sb),
        spec(&C_DIMS, &sc),
        &cfg(),
    )
    .unwrap()
}

fn oracle(a: &[f64], b: &[f64], alpha: f64, beta: f64, c: &mut [f64]) {
    for i in 0..3 {
        for n in 0..6 {
            let mut s = 0.0;
            for j in 0..4 {
                for k in 0..5 {
                    s += a[i + 3 * j + 12 * k] * b[k + 5 * n + 30 * j];
                }
            }
            let at = i + 3 * n;
            c[at] = alpha * s + if beta == 0.0 { 0.0 } else { beta * c[at] };
        }
    }
}

fn plan(
    cat: &KernelCatalog<f64>,
    config: &PlanConfig,
    mut selector: impl FnMut(
        &tcontract::SelectionContext<'_>,
        &[tcontract::KernelCandidate<f64>],
    ) -> Result<KernelHandle<f64>, SelectError>,
) -> Result<Plan<f64>, Error> {
    Plan::<f64>::new_with_selector(&problem(), config, cat, &mut selector)
}

fn run(plan: &Plan<f64>, alpha: f64, beta: f64) -> (Vec<f64>, Vec<f64>) {
    let (a, b) = (ints::<f64>(1, 60), ints::<f64>(2, 120));
    let mut c = ints::<f64>(3, 18);
    let mut want = c.clone();
    oracle(&a, &b, alpha, beta, &mut want);
    let (sa, sb, sc) = (col_major(&A_DIMS), col_major(&B_DIMS), col_major(&C_DIMS));
    let av = StridedView::new(&a, &A_DIMS, &sa, 0).unwrap();
    let bv = StridedView::new(&b, &B_DIMS, &sb, 0).unwrap();
    let mut cv = StridedViewMut::new(&mut c, &C_DIMS, &sc, 0).unwrap();
    plan.execute_into_accum(
        &mut Exec::serial(),
        alpha,
        &av,
        &bv,
        beta,
        AccumulationSource::Output,
        &mut cv,
    )
    .unwrap();
    (c, want)
}

#[test]
fn a_non_fusable_contraction_runs_on_a_custom_kernel() {
    let cat = catalog(own::f64_families());
    let calls = Cell::new(0);
    let p = plan(&cat, &PlanConfig::default(), |ctx, cands| {
        calls.set(calls.get() + 1);
        assert_eq!((ctx.stats.m, ctx.stats.n, ctx.stats.k), (3, 6, 20));
        assert_eq!(ctx.operands[0].extents, &[3, 4, 5]);
        assert_eq!(cands.len(), 2);
        Ok(cands[1].handle)
    })
    .unwrap();
    assert_eq!(p.report().algorithm, Algorithm::Packed);
    let report = p.report().packed.as_ref().unwrap();
    assert_eq!(report.family_id, "custom.f64.3x4");
    assert_eq!(report.origin, own::ORIGIN);
    for (alpha, beta) in [(1.0, 0.0), (2.0, -1.0)] {
        let (got, want) = run(&p, alpha, beta);
        assert_eq!(got, want);
    }
    assert_eq!(calls.get(), 1, "execution never calls the selector");
}

#[test]
fn the_selected_plan_outlives_its_selector_and_catalog() {
    let p = {
        let cat = catalog(own::f64_families());
        let captured = [String::from("captured state"), String::from("dropped")];
        plan(&cat, &PlanConfig::default(), |_, cands| {
            assert_eq!(captured.len(), 2);
            Ok(cands[0].handle)
        })
        .unwrap()
        // catalog and captured state are dropped here
    };
    for _ in 0..3 {
        let (got, want) = run(&p, 1.0, 0.0);
        assert_eq!(got, want);
    }
    assert_eq!(
        p.report().packed.as_ref().unwrap().family_id,
        "custom.f64.2x2"
    );
    // The family was never registered process-wide: nothing could have looked
    // it up by id.
    assert!(matches!(
        tcontract_kernel::Registry::select::<f64>(
            "custom.f64.2x2",
            tcontract_kernel::CpuFeatures::detect()
        ),
        Err(tcontract_kernel::SelectError::UnknownId { .. })
    ));
}

#[test]
fn incompatible_choices_are_typed_errors() {
    let cat = catalog(own::f64_families());
    let other = catalog(own::f64_families());
    let g = PlanConfig::default();
    let sel_err = |r: Result<Plan<f64>, Error>| match r {
        Err(e) => typed(&e).clone(),
        Ok(_) => panic!("accepted"),
    };
    // A forced id and a selector are ambiguous.
    let forced = PlanConfig {
        kernel: tcontract_kernel::KernelChoice::Id("ref.f64.real.4x4".into()),
        ..PlanConfig::default()
    };
    assert!(matches!(
        sel_err(plan(&cat, &forced, |_, _| unreachable!())),
        SelectError::Incompatible { .. }
    ));
    let foreign = other.get("custom.f64.2x2").unwrap();
    assert!(matches!(
        sel_err(plan(&cat, &g, |_, _| Ok(foreign))),
        SelectError::ForeignHandle { .. }
    ));
    assert!(matches!(
        sel_err(plan(&cat, &g, |_, _| Err(SelectError::SelectorFailed {
            reason: "x".into()
        }))),
        SelectError::SelectorFailed { .. }
    ));
    let masked = catalog(own::f64_impossible_families());
    assert!(matches!(
        sel_err(plan(&masked, &g, |_, _| unreachable!())),
        SelectError::NoCandidates { .. }
    ));
}

/// The consolidation's behaviour change: a selector is a requirement that
/// forces the packed driver, so an all-batch problem (which the old planner
/// refused to select for, as its elementwise pass has no kernel) now selects and
/// runs on the chosen custom family.
#[test]
fn all_batch_and_zero_size_problems_still_select_or_refuse() {
    let cat = catalog(own::f64_families());
    let other = catalog(own::f64_families());
    let had = DotGeneral::new(&[], &[], &[0], &[0]);
    let spec = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
    let (dims, st) = ([4usize], [1isize]);
    let had_problem = Problem::from_dot_general(
        DType::F64,
        spec(&dims, &st),
        spec(&dims, &st),
        spec(&dims, &st),
        &had,
    )
    .unwrap();
    let called = Cell::new(false);
    let p = Plan::<f64>::new_with_selector(
        &had_problem,
        &PlanConfig::default(),
        &cat,
        &mut |_, cands| {
            called.set(true);
            Ok(cands[0].handle)
        },
    )
    .unwrap();
    assert!(called.get());
    assert_eq!(p.report().algorithm, Algorithm::Packed);
    let (x, y) = ([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]);
    let mut z = [0.0; 4];
    p.execute_into(
        &mut Exec::serial(),
        1.0,
        &StridedView::new(&x, &dims, &st, 0).unwrap(),
        &StridedView::new(&y, &dims, &st, 0).unwrap(),
        &mut StridedViewMut::new(&mut z, &dims, &st, 0).unwrap(),
    )
    .unwrap();
    assert_eq!(z, [5.0, 12.0, 21.0, 32.0]);

    // A zero extent: the plan is still selected and validated, and a bad
    // choice is rejected before anything runs.
    let (ad, bd, cd) = ([0usize, 4], [4usize, 3], [0usize, 3]);
    let (sa, sb, sc) = ([1isize, 1], [1isize, 4], [1isize, 1]);
    let zero_problem = Problem::from_dot_general(
        DType::F64,
        spec(&ad, &sa),
        spec(&bd, &sb),
        spec(&cd, &sc),
        &DotGeneral::new(&[1], &[0], &[], &[]),
    )
    .unwrap();
    let foreign = other.get("custom.f64.2x2").unwrap();
    let zero = |h: KernelHandle<f64>| {
        Plan::<f64>::new_with_selector(
            &zero_problem,
            &PlanConfig::default(),
            &cat,
            &mut move |_, _| Ok(h),
        )
    };
    assert!(matches!(
        zero(foreign),
        Err(e) if matches!(typed(&e), SelectError::ForeignHandle { .. })
    ));
    let ok = zero(cat.get("custom.f64.2x2").unwrap()).unwrap();
    let a: Vec<f64> = vec![];
    let b = ints::<f64>(1, 12);
    let mut c: Vec<f64> = vec![];
    ok.execute_into(
        &mut Exec::serial(),
        1.0,
        &StridedView::new(&a, &ad, &sa, 0).unwrap(),
        &StridedView::new(&b, &bd, &sb, 0).unwrap(),
        &mut StridedViewMut::new(&mut c, &cd, &sc, 0).unwrap(),
    )
    .unwrap();
}

/// A plan is typed by its storage scalar; a selector plan for another dtype than
/// the problem's is a configuration error, before the catalog is consulted.
#[test]
fn a_selected_plan_is_bound_to_its_storage_dtype() {
    let cat = KernelCatalog::<tcontract_kernel::C64>::builtin();
    let called = Cell::new(false);
    let e = Plan::<tcontract_kernel::C64>::new_with_selector(
        &problem(),
        &PlanConfig::default(),
        &cat,
        &mut |_, c| {
            called.set(true);
            Ok(c[0].handle)
        },
    )
    .unwrap_err();
    assert!(matches!(
        e,
        Error::Config(ConfigError::DtypeMismatch {
            plan: "c64",
            problem: "f64"
        })
    ));
    assert!(!called.get());
}

/// A direct-update, direct-B custom family: the selector is shown that B can be
/// read in place, the family updates `D` itself (with `C` accumulated in place)
/// and the plan reports it.
#[test]
fn a_direct_b_custom_kernel_updates_c_in_place_and_reports_itself() {
    let cat = catalog(own::f64_direct_families());
    let (m, n, k) = (23usize, 21usize, 13usize);
    let (a, b) = (ints::<f64>(7, m * k), ints::<f64>(8, k * n));
    let mut c = ints::<f64>(9, m * n);
    let mut want = c.clone();
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for p in 0..k {
                s += a[i + p * m] * b[p + j * k];
            }
            want[i + j * m] = s + 3.0 * want[i + j * m];
        }
    }
    let spec = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
    let problem = Problem::from_dot_general(
        DType::F64,
        spec(&[m, k], &[1, m as isize]),
        spec(&[k, n], &[1, k as isize]),
        spec(&[m, n], &[1, m as isize]),
        &DotGeneral::new(&[1], &[0], &[], &[]),
    )
    .unwrap();
    let saw_direct = Cell::new(false);
    let plan =
        Plan::<f64>::new_with_selector(&problem, &PlanConfig::default(), &cat, &mut |_, cands| {
            saw_direct.set(cands[0].direct_b);
            Ok(cands[0].handle)
        })
        .unwrap();
    plan.execute_into_accum(
        &mut Exec::serial(),
        1.0,
        &StridedView::new(&a, &[m, k], &[1, m as isize], 0).unwrap(),
        &StridedView::new(&b, &[k, n], &[1, k as isize], 0).unwrap(),
        3.0,
        AccumulationSource::Output,
        &mut StridedViewMut::new(&mut c, &[m, n], &[1, m as isize], 0).unwrap(),
    )
    .unwrap();
    assert_eq!(c, want);
    assert!(
        saw_direct.get(),
        "unit-stride B is eligible for in-place reads"
    );
    assert_eq!(
        plan.report().packed.as_ref().unwrap().family_id,
        "custom.f64.4x4.direct-b"
    );
}

/// Complex storage: a custom complex family runs a conjugated product, the
/// selector sees the complex facts, and a catalog of the wrong dtype is refused
/// at admission.
#[test]
fn custom_complex_families_need_a_dtype_correct_catalog() {
    use num_complex::Complex;
    let (m, n, k) = (7usize, 5usize, 6usize);
    let z = |len: usize, s: usize| -> Vec<Complex<f64>> {
        ints::<f64>(s, len)
            .into_iter()
            .zip(ints::<f64>(s + 50, len))
            .map(|(re, im)| Complex::new(re, im))
            .collect()
    };
    let (a, b) = (z(m * k, 1), z(k * n, 2));
    let mut c = vec![Complex::new(0.0, 0.0); m * n];
    let mut want = c.clone();
    for i in 0..m {
        for j in 0..n {
            let mut s = Complex::new(0.0, 0.0);
            for p in 0..k {
                s += a[i + p * m].conj() * b[p + j * k];
            }
            want[i + j * m] = s;
        }
    }
    let cat = unsafe { KernelCatalog::<Complex<f64>>::from_static_families(own::c64_families()) }
        .unwrap();
    let spec = |d: &[usize], s: &[isize], op: Op| {
        OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap()).with_op(op)
    };
    let problem = Problem::from_dot_general(
        DType::C64,
        spec(&[m, k], &[1, m as isize], Op::Conjugate),
        spec(&[k, n], &[1, k as isize], Op::Identity),
        spec(&[m, n], &[1, m as isize], Op::Identity),
        &DotGeneral::new(&[1], &[0], &[], &[]),
    )
    .unwrap();
    let plan = Plan::<Complex<f64>>::new_with_selector(
        &problem,
        &PlanConfig::default(),
        &cat,
        &mut |ctx, cands| {
            assert!(ctx.is_complex);
            assert_eq!(ctx.dtype, "c64");
            assert!(ctx.operands[0].conj);
            Ok(cands[0].handle)
        },
    )
    .unwrap();
    plan.execute_into(
        &mut Exec::serial(),
        Complex::new(1.0, 0.0),
        &StridedView::new(&a, &[m, k], &[1, m as isize], 0).unwrap(),
        &StridedView::new(&b, &[k, n], &[1, k as isize], 0).unwrap(),
        &mut StridedViewMut::new(&mut c, &[m, n], &[1, m as isize], 0).unwrap(),
    )
    .unwrap();
    let report = plan.report().packed.as_ref().unwrap();
    assert_eq!(report.family_id, "custom.c64.native.2x2");
    assert!(report.complex.is_some());
    assert_eq!(c, want);

    // c32 admits its own complex family; a real list is refused for complex
    // storage and vice versa.
    let admit32 =
        unsafe { KernelCatalog::<Complex<f32>>::from_static_families(own::c32_families()) };
    assert!(admit32.is_ok());
    assert!(matches!(
        unsafe { KernelCatalog::<Complex<f64>>::from_static_families(own::f64_families()) },
        Err(SelectError::DtypeMismatch { dtype: "c64", .. })
    ));
    assert!(matches!(
        unsafe { KernelCatalog::<f64>::from_static_families(own::c64_families()) },
        Err(SelectError::DtypeMismatch { dtype: "f64", .. })
    ));
}

/// A built-in family is available to a selector only when the caller's catalog
/// holds it: the fallback is a handle the selector chooses, never an implicit
/// default.
#[test]
fn a_builtin_fallback_is_explicit() {
    let own_cat = catalog(own::f64_families());
    let both = own_cat.union(&KernelCatalog::<f64>::builtin()).unwrap();
    let plan = Plan::<f64>::new_with_selector(
        &problem(),
        &PlanConfig::default(),
        &both,
        &mut |_, cands| {
            // Prefer a built-in kernel when asked to; the report says so.
            Ok(cands
                .iter()
                .find(|c| c.handle.id() == "ref.f64.real.4x4")
                .unwrap()
                .handle)
        },
    )
    .unwrap();
    let report = plan.report().packed.as_ref().unwrap();
    assert_eq!(report.family_id, "ref.f64.real.4x4");
    assert_eq!(report.origin, tcontract_kernel::Origin::Portable);
}
