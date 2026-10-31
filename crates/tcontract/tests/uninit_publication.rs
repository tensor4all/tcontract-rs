use core::mem::MaybeUninit;
use tcontract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
use tcontract::{
    execute_uninit_with, Plan, PlanConfig, SliceAccumulationSource, UninitOutputError,
};
use tcontract_exec::Exec;

#[test]
fn completed_kernels_publish_out_of_order_slices_without_copying() {
    let operand = || OperandSpec::new(LayoutSpec::new(&[1], &[1], 0).unwrap());
    let problem = Problem::from_labels(
        DType::F64,
        operand(),
        operand(),
        CSpec::Absent,
        operand(),
        &Labels::new(&[0], &[0], &[0]),
    )
    .unwrap();
    let plan = Plan::<f64>::new(&problem, &PlanConfig::default()).unwrap();
    let mut storage = [MaybeUninit::uninit(); 2];
    let ptr = storage.as_mut_ptr().cast::<f64>();
    let (out, route) = execute_uninit_with(&mut storage, |slots| {
        let mut initialized = Vec::new();
        for (i, slot) in slots.chunks_mut(1).enumerate() {
            initialized.push(plan.execute_uninit_slices(
                &mut Exec::serial(),
                (i + 1) as f64,
                (&[3.0], 0),
                (&[4.0], 0),
                0.0,
                SliceAccumulationSource::Absent,
                (slot, 0),
            )?);
        }
        initialized.reverse();
        // Empty receipts do not contribute coverage or require pointer identity.
        initialized.push(&mut []);
        Ok::<_, tcontract::Error>((initialized, "two kernels"))
    })
    .unwrap();
    assert_eq!(out.as_mut_ptr(), ptr);
    assert_eq!(out, &[12.0, 24.0]);
    assert_eq!(route, "two kernels");
}

#[test]
fn safe_write_receipts_keep_publication_provenance() {
    let mut storage = [MaybeUninit::<f64>::uninit(); 2];
    let (out, ()) = execute_uninit_with(&mut storage, |slots| {
        let receipts = slots
            .iter_mut()
            .map(|slot| core::slice::from_mut(slot.write(3.0)))
            .collect();
        Ok::<_, ()>((receipts, ()))
    })
    .unwrap();
    assert_eq!(out, &[3.0; 2]);
}

#[test]
fn incomplete_and_foreign_receipts_cannot_publish_output() {
    for which in 0..4 {
        let mut storage = [MaybeUninit::new(91.0); 2];
        let result = execute_uninit_with(&mut storage, |slots| {
            let initialized = match which {
                0 => vec![core::slice::from_mut(slots[0].write(2.0))],
                1 => vec![core::slice::from_mut(slots[1].write(2.0))],
                // Safe callbacks can return other initialized storage: it is
                // not evidence about this destination and must be rejected.
                2 => vec![Box::leak(vec![3.0].into_boxed_slice()) as &mut [f64]],
                _ => vec![Box::leak(vec![3.0; 3].into_boxed_slice()) as &mut [f64]],
            };
            Ok::<_, ()>((initialized, ()))
        });
        assert!(matches!(result, Err(UninitOutputError::Coverage)));
    }
}

#[test]
fn callback_error_preserves_error_without_publishing_partial_output() {
    let mut storage = [MaybeUninit::<f64>::uninit(); 2];
    let result = execute_uninit_with(&mut storage, |slots| {
        slots[0].write(12.0);
        Err::<(Vec<&mut [f64]>, ()), _>("later kernel failed")
    });
    assert!(matches!(
        result,
        Err(UninitOutputError::Execution("later kernel failed"))
    ));
}

#[test]
fn empty_output_needs_no_receipts_and_empty_receipts_do_not_prove_nonempty_output() {
    let (out, ()) = execute_uninit_with::<f64, _, ()>(&mut [], |_| Ok((vec![], ()))).unwrap();
    assert!(out.is_empty());
    let mut storage = [MaybeUninit::<f64>::uninit()];
    assert!(matches!(
        execute_uninit_with::<f64, _, ()>(&mut storage, |_| Ok((vec![&mut []], ()))),
        Err(UninitOutputError::Coverage)
    ));
}

#[test]
fn panic_does_not_publish_an_initialized_destination() {
    let mut storage = [MaybeUninit::<f64>::uninit(); 2];
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        execute_uninit_with::<f64, (), ()>(&mut storage, |slots| {
            slots[0].write(3.0);
            panic!("kernel panic");
        })
        .map(|_| ())
    }));
    assert!(result.is_err());
}
