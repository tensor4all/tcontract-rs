//! Every registered CPU-available family gets the same small, non-trivial GEMM,
//! in both orientations and with every conjugation combination, against the
//! reference oracle.
use super::common::{all_families, check_family_vs_oracle, Opts};
use super::compat::{Layout, Operand, Plan};
use crate::api::Scalar;
use tcontract_kernel::{KernelChoice, SelectError};

fn run_family<T>(id: &'static str)
where
    T: Scalar,
{
    let tol = if core::mem::size_of::<T::Real>() == 4 {
        3e-5
    } else {
        5e-12
    };
    for row_major_d in [false, true] {
        for conj_a in [false, true] {
            for conj_b in [false, true] {
                check_family_vs_oracle::<T>(
                    id,
                    Opts {
                        row_major_d,
                        conj_a,
                        conj_b,
                        ..Opts::default()
                    },
                    tol,
                );
            }
        }
    }
}

macro_rules! families_test {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            for id in all_families::<$ty>() {
                run_family::<$ty>(id);
            }
        }
    };
}

families_test!(all_f32_families, f32);
families_test!(all_f64_families, f64);
families_test!(all_c32_families, tcontract_kernel::C32);
families_test!(all_c64_families, tcontract_kernel::C64);

#[test]
fn selection_rejects_unknown_and_wrong_dtype() {
    let plan = Plan::new(
        Operand::new(&Layout::col_major(&[2, 2]), &[0, 2]),
        Operand::new(&Layout::col_major(&[2, 2]), &[2, 1]),
        None,
        Operand::new(&Layout::col_major(&[2, 2]), &[0, 1]),
    )
    .unwrap();
    assert!(matches!(
        plan.clone()
            .with_kernel(KernelChoice::Id("definitely-not-a-family".into()))
            .unwrap()
            .resolved::<f64>(),
        Err(SelectError::UnknownId { .. })
    ));
    let id = all_families::<f32>()[0];
    assert!(matches!(
        plan.with_kernel(KernelChoice::Id(id.into()))
            .unwrap()
            .resolved::<f64>(),
        Err(SelectError::DtypeMismatch { .. })
    ));
}
