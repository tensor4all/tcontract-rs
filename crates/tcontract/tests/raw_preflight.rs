//! The raw entry (`Plan::execute_raw`, the C adapter's path) runs the same
//! semantic preflight as the safe path, before any write: `D` overlapping an
//! input, `C` overlapping `D` without being the same mapping at the same
//! origin. Null pointers are the caller's (an FFI status).
use tcontract::api::{
    AliasError, CSpec, DType, Error, Labels, LayoutSpec, Op, OperandSpec, Problem,
};
use tcontract::{Plan, PlanConfig};
use tcontract_exec::Exec;

fn spec(dims: &[usize], strides: &[isize]) -> OperandSpec {
    OperandSpec::new(LayoutSpec::new(dims, strides, 0).unwrap())
}

/// `D[i,j] = sum_k A[i,k] B[k,j]` (2x3 by 3x2), with the given C mode.
fn plan(c: CSpec, labels_c: bool) -> Plan<f64> {
    let mut labels = Labels::new(&[0, 2], &[2, 1], &[0, 1]);
    if labels_c {
        labels = labels.with_c(&[0, 1]);
    }
    let p = Problem::from_labels(
        DType::F64,
        spec(&[2, 3], &[1, 2]),
        spec(&[3, 2], &[1, 3]),
        c,
        spec(&[2, 2], &[1, 2]),
        &labels,
    )
    .unwrap();
    Plan::<f64>::new(&p, &PlanConfig::default()).unwrap()
}

const A: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
const B: [f64; 6] = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0];

#[test]
fn d_overlapping_an_input_is_refused_before_any_write() {
    let p = plan(CSpec::Absent, false);
    // D lives inside A's buffer.
    let mut buf = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let before = buf;
    let (a, d) = (buf.as_ptr(), unsafe { buf.as_mut_ptr().add(3) });
    // SAFETY: every pointer is inside `buf`, which outlives the call; the
    // preflight refuses before any access.
    let e = unsafe {
        p.execute_raw(
            &mut Exec::serial(),
            1.0,
            a,
            B.as_ptr(),
            0.0,
            std::ptr::null(),
            d,
        )
    };
    assert!(
        matches!(
            e,
            Err(Error::Alias(AliasError::OutputOverlapsInput {
                operand: tcontract::api::OperandId::A
            }))
        ),
        "{e:?}"
    );
    assert_eq!(buf, before);
}

#[test]
fn a_disjoint_call_runs_and_a_zero_alpha_call_ignores_the_inputs() {
    let p = plan(CSpec::Absent, false);
    let mut d = [0.0f64; 4];
    // SAFETY: valid, disjoint buffers sized by the layouts.
    unsafe {
        p.execute_raw(
            &mut Exec::serial(),
            1.0,
            A.as_ptr(),
            B.as_ptr(),
            0.0,
            std::ptr::null(),
            d.as_mut_ptr(),
        )
        .unwrap();
    }
    assert_eq!(d, [1.0 + 5.0, 2.0 + 6.0, 3.0 + 5.0, 4.0 + 6.0]);
    // alpha == 0 reads neither A nor B, so they may be null (and aliasing them is moot).
    let mut d = [9.0f64; 4];
    // SAFETY: only `d` is accessed.
    unsafe {
        p.execute_raw(
            &mut Exec::serial(),
            0.0,
            std::ptr::null(),
            std::ptr::null(),
            0.0,
            std::ptr::null(),
            d.as_mut_ptr(),
        )
        .unwrap();
    }
    assert_eq!(d, [0.0; 4]);
}

#[test]
fn a_separate_c_is_the_same_mapping_in_place_or_disjoint_but_never_partially_overlapping() {
    // C has D's layout, so `C == D` is an in-place update.
    let p = plan(CSpec::Separate(spec(&[2, 2], &[1, 2])), true);
    let mut d = [1.0f64, 2.0, 3.0, 4.0];
    let dp = d.as_mut_ptr();
    // SAFETY: C and D are the same buffer at the same origin; A and B are disjoint.
    unsafe {
        p.execute_raw(
            &mut Exec::serial(),
            1.0,
            A.as_ptr(),
            B.as_ptr(),
            1.0,
            dp as *const f64,
            dp,
        )
        .unwrap();
    }
    assert_eq!(d, [1.0 + 6.0, 2.0 + 8.0, 3.0 + 8.0, 4.0 + 10.0]);

    // A C that overlaps D at a shifted origin is another mapping: refused.
    let mut buf = [0.0f64; 8];
    let before = buf;
    let base = buf.as_mut_ptr();
    // SAFETY: both pointers are inside `buf`; the preflight refuses first.
    let e = unsafe {
        p.execute_raw(
            &mut Exec::serial(),
            1.0,
            A.as_ptr(),
            B.as_ptr(),
            1.0,
            base.add(1) as *const f64,
            base,
        )
    };
    assert!(
        matches!(e, Err(Error::Alias(AliasError::CDOverlap))),
        "{e:?}"
    );
    assert_eq!(buf, before);

    // A disjoint C is fine.
    let c = [10.0f64, 20.0, 30.0, 40.0];
    let mut d = [0.0f64; 4];
    // SAFETY: disjoint valid buffers.
    unsafe {
        p.execute_raw(
            &mut Exec::serial(),
            1.0,
            A.as_ptr(),
            B.as_ptr(),
            1.0,
            c.as_ptr(),
            d.as_mut_ptr(),
        )
        .unwrap();
    }
    assert_eq!(d, [10.0 + 6.0, 20.0 + 8.0, 30.0 + 8.0, 40.0 + 10.0]);
}

#[test]
fn an_output_mode_plan_reads_the_previous_d_through_d() {
    let p = plan(CSpec::Output(Op::Identity), false);
    let mut d = [1.0f64, 2.0, 3.0, 4.0];
    // SAFETY: valid buffers; `c` is ignored for an in-place C.
    unsafe {
        p.execute_raw(
            &mut Exec::serial(),
            1.0,
            A.as_ptr(),
            B.as_ptr(),
            2.0,
            std::ptr::null(),
            d.as_mut_ptr(),
        )
        .unwrap();
    }
    assert_eq!(d, [2.0 + 6.0, 4.0 + 8.0, 6.0 + 8.0, 8.0 + 10.0]);
}
