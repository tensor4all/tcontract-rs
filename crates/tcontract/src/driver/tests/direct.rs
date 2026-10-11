//! Direct-C and direct-B families: the plan-level guard, the per-tile fallback
//! and the in-place column operand all have to agree with the oracle.
use super::common::{check_family_vs_oracle, Opts};
use super::compat::{contract_reference, RefOperand};
use super::compat::{ElementOp, Operand};
use super::compat::{Layout, Plan};
use tcontract_kernel::Element;
use tcontract_kernel::{CpuFeatures, KernelChoice, Registry, SelectError, C64};

const DIRECT: &str = "ref.f64.direct.4x4";
const DIRECT_B: &str = "ref.f64.direct-b.4x4";

/// The guard is `!complex && !conj_c && !conj_d && (beta == 0 || C is D)`, and
/// it has to hold in *both* directions: when it allows the direct write and
/// when it refuses it, the numbers must still match the oracle. A real operand
/// is never conjugated, so the conjugation flags can only change the verdict,
/// not the result — which is exactly what makes this a guard test.
#[test]
fn direct_c_guard_matrix() {
    for &beta in &[0.0, 1.0, 0.5] {
        for &alias in &[false, true] {
            for &(conj_c, conj_d) in &[(false, false), (true, false), (false, true)] {
                let opts = Opts {
                    conj_c,
                    conj_d,
                    alias,
                    ..Opts::real(1.5, beta)
                };
                let calls = check_family_vs_oracle::<f64>(DIRECT, opts, 1e-12);
                let expected = !conj_c && !conj_d && (beta == 0.0 || alias);
                assert_eq!(
                    calls.direct_c_allowed, expected,
                    "beta={beta} alias={alias} conj={conj_c}/{conj_d}"
                );
                assert!(calls.pack_b_needed, "a packed family always packs B");
            }
        }
    }
}

/// An output whose column scatter has two different strides inside one `NR`
/// block: `D`'s own offsets are still a valid write-back target, but no single
/// column stride expresses the tile, so every tile must take the fallback. If
/// the per-tile check were missing, `IRREGULAR` would be used as a stride.
#[test]
fn direct_family_scatter_output_falls_back_per_tile() {
    let (m, k, n1, n2) = (2usize, 7usize, 2usize, 2usize);
    let la = Layout::new(vec![m as i64, k as i64], vec![1, 2]).unwrap();
    let lb = Layout::new(vec![k as i64, n1 as i64, n2 as i64], vec![1, 7, 14]).unwrap();
    // Label 1 and label 2 both index columns, with strides 3 and 8, so their
    // mixed-radix group is `[0, 3, 8, 11]`: not one arithmetic progression, and
    // still an output the problem's injectivity test accepts.
    let ld = Layout::new(vec![m as i64, n1 as i64, n2 as i64], vec![1, 3, 8]).unwrap();
    let (ia, ib, idd) = (vec![0i64, 3], vec![3i64, 1, 2], vec![0i64, 1, 2]);
    let a: Vec<f64> = (0..la.storage_len() as usize)
        .map(|i| i as f64 * 0.25)
        .collect();
    let b: Vec<f64> = (0..lb.storage_len() as usize)
        .map(|i| (i as f64 - 3.0) * 0.125)
        .collect();
    let start: Vec<f64> = (0..ld.storage_len() as usize)
        .map(|i| i as f64 * 0.5)
        .collect();
    let mut got = start.clone();
    let plan = Plan::new(
        Operand::new(&la, &ia),
        Operand::new(&lb, &ib),
        None,
        Operand::new(&ld, &idd),
    )
    .unwrap()
    .with_kernel(KernelChoice::Id(DIRECT.into()))
    .unwrap();
    let rg = plan.resolved::<f64>().unwrap();
    // beta == 0, so the guard allows a direct write; only the column scatter
    // can send the tiles back through the scratch path.
    let calls = plan.decisions::<f64>(&rg, std::ptr::null(), got.as_mut_ptr(), 0.0);
    assert!(calls.direct_c_allowed);

    // SAFETY: full buffers per their layouts, no C read with beta = 0, and `D`
    // is borrowed exclusively.
    unsafe {
        plan.run_raw::<f64>(
            1.0,
            a.as_ptr(),
            b.as_ptr(),
            0.0,
            std::ptr::null(),
            got.as_mut_ptr(),
        )
    };

    let mut want = start;
    contract_reference::<f64>(
        1.0,
        &RefOperand {
            data: &a,
            layout: &la,
            idx: &ia,
            op: ElementOp::Identity,
        },
        &RefOperand {
            data: &b,
            layout: &lb,
            idx: &ib,
            op: ElementOp::Identity,
        },
        0.0,
        None,
        &mut want,
        &ld,
        &idd,
        ElementOp::Identity,
    )
    .unwrap();
    let diff: f64 = got.iter().zip(&want).map(|(x, y)| (x - y).powi(2)).sum();
    let scale: f64 = want.iter().map(|x| x * x).sum::<f64>().max(1.0);
    assert!((diff / scale).sqrt() < 1e-12, "{got:?} vs {want:?}");
}

/// A column operand whose k stride is one is read where it lies: no packed
/// panel, and the numbers still match.
#[test]
fn direct_b_reads_b_in_place_and_matches_oracle() {
    let calls = check_family_vs_oracle::<f64>(DIRECT_B, Opts::real(1.5, 0.5), 1e-12);
    assert!(!calls.pack_b_needed, "in-place B must not be packed");
}

/// Storing `D` row-major makes the engine exchange the operands, so the
/// kernel's column role is the user's `A` — whose k stride is `M`, not one.
/// The family must then pack, and the contraction must still be right.
#[test]
fn direct_b_disabled_when_orientation_swaps_to_strided_b() {
    let opts = Opts {
        row_major_d: true,
        ..Opts::real(1.5, 0.5)
    };
    let calls = check_family_vs_oracle::<f64>(DIRECT_B, opts, 1e-12);
    assert!(
        calls.pack_b_needed,
        "the swapped column operand has k stride M and cannot be read in place"
    );
}

/// No B workspace is asked for when B is read in place.
#[test]
fn direct_b_plan_has_zero_b_workspace() {
    let calls = check_family_vs_oracle::<f64>(DIRECT_B, Opts::real(1.0, 0.0), 1e-12);
    assert!(!calls.pack_b_needed);
}

/// The complex half of the guard is enforced before the driver ever runs: a
/// complex Direct descriptor cannot be registered, and a real one cannot be
/// selected for complex storage. This is what makes the guard's
/// `!T::IS_COMPLEX` term unreachable-by-construction rather than untested.
#[test]
fn complex_storage_cannot_reach_a_real_direct_family() {
    assert!(matches!(
        Registry::select::<C64>(DIRECT, CpuFeatures::detect()),
        Err(SelectError::DtypeMismatch { .. })
    ));
    assert!(matches!(
        Registry::select::<C64>("ref.f64.real.4x4", CpuFeatures::detect()),
        Err(SelectError::DtypeMismatch { .. })
    ));
    // And the descriptor itself refuses a complex scheme with a Direct kernel.
    let mut f = **tcontract_kernel::portable::families_f64()
        .iter()
        .find(|f| f.id == DIRECT)
        .unwrap();
    f.complex = Some(tcontract_kernel::ComplexScheme {
        method: tcontract_kernel::Method::Native,
        a: tcontract_kernel::Layout::Planar,
        b: tcontract_kernel::Layout::Planar,
        tile: tcontract_kernel::TileFormat::Planar,
    });
    f.a_per_k = 8;
    f.b_per_k = 8;
    f.tile_bound = 32;
    let err = f.validate().unwrap_err();
    assert!(err.reason.contains("complex Direct"), "{}", err.reason);
}

/// The reference oracle takes operands by struct literal, so no element
/// helper is needed here; `Element` is used only for the probe's bound.
#[allow(dead_code)]
fn _element_is_in_scope<T: Element>() {}
