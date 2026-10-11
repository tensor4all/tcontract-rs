use super::compat::{Layout, Operand, Plan};
use crate::api::{CSpec, DType, Error, Labels, LayoutSpec, OperandSpec, Problem, Scalar};
use crate::plan::PlanConfig;
use tcontract_kernel::{Blocking, KernelChoice, SelectError};

#[test]
fn builders_invalidate_cached_geometry_and_preserve_explicit_blocks() {
    let l = Layout::col_major(&[2, 2]);
    let p = Plan::new(
        Operand::new(&l, &[0, 2]),
        Operand::new(&l, &[2, 1]),
        None,
        Operand::new(&l, &[0, 1]),
    )
    .unwrap()
    .with_kernel(KernelChoice::Id("ref.f64.real.4x4".into()))
    .unwrap();
    let first = p.resolved::<f64>().unwrap();
    let changed = p
        .clone()
        .with_blocking(Blocking {
            mc: 5,
            kc: 3,
            nc: 7,
        })
        .with_threads(8);
    let r = changed.resolved::<f64>().unwrap();
    assert_eq!((r.mc, r.kc, r.nc), (8, 3, 8));
    assert_eq!(
        (
            r.with_threads(1).unwrap().mc,
            r.with_threads(1).unwrap().kc,
            r.with_threads(1).unwrap().nc
        ),
        (8, 3, 8)
    );
    assert_eq!(p.resolved::<f64>().unwrap().kc, first.kc);
}

/// The default family of every dtype and complex scheme resolves to its menu
/// head with the legacy blocking: derived from the packed footprint and
/// register-aligned, whatever the thread count.
#[test]
fn auto_head_and_blocking_follow_the_default_menu() {
    fn check<T: Scalar>() {
        let l = Layout::col_major(&[9, 9]);
        for method in tcontract_kernel::ComplexMethod::ALL {
            let p = Plan::new(
                Operand::new(&l, &[0, 2]),
                Operand::new(&l, &[2, 1]),
                None,
                Operand::new(&l, &[0, 1]),
            )
            .unwrap()
            .with_complex_method(method)
            .with_threads(4);
            let rg = p.resolved::<T>().unwrap();
            let f = rg.family();
            let raw = Blocking::derive(
                core::mem::size_of::<<T as Scalar>::Re>(),
                f.a_per_k / f.mr,
                f.b_per_k / f.nr,
            );
            assert_eq!(
                (rg.mr, rg.nr, rg.mc, rg.kc, rg.nc),
                (
                    f.mr,
                    f.nr,
                    raw.mc.next_multiple_of(f.mr),
                    raw.kc,
                    raw.nc.next_multiple_of(f.nr)
                ),
                "{} at {method:?}",
                f.id
            );
        }
    }
    check::<f32>();
    check::<f64>();
    check::<tcontract_kernel::C32>();
    check::<tcontract_kernel::C64>();
}

/// A plan for the wrong dtype is refused at construction, with the typed
/// selection error as the source, even for an empty problem.
#[test]
fn an_empty_problem_still_rejects_a_wrong_dtype_family() {
    let l = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
    let problem = Problem::from_labels(
        DType::F64,
        l(&[0, 0], &[1, 0]),
        l(&[0, 0], &[1, 0]),
        CSpec::Absent,
        l(&[0, 0], &[1, 0]),
        &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    )
    .unwrap();
    let cfg = PlanConfig {
        kernel: KernelChoice::Id("ref.c64.native.4x4".into()),
        ..PlanConfig::default()
    };
    let err = crate::Plan::<f64>::new(&problem, &cfg).unwrap_err();
    assert!(matches!(
        err,
        Error::Select(SelectError::DtypeMismatch { .. })
    ));
    let source = std::error::Error::source(&err).unwrap();
    assert!(source.downcast_ref::<SelectError>().is_some());
}
