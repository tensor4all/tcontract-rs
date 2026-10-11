//! The neutral interface's shape: object safety at a fixed scalar, auto traits,
//! and the error contract of the shared [`Error`].
use tcontract::api::{
    is_injective_layout, AccumulationSource, BoxedPlan, ContractionBackend, Error,
    PreparedContraction, Unsupported,
};
use tcontract_exec::Exec;

#[test]
fn interface_is_object_safe_at_fixed_scalar() {
    fn _backend(_: &dyn ContractionBackend<f64>) {}
    fn _plan(_: &dyn PreparedContraction<f32>) {}
    fn _slot(_: Vec<Box<dyn ContractionBackend<num_complex::Complex64>>>) {}
    fn _exec(_: &Exec<'_>, _: AccumulationSource<'_, f64>) {}
    fn send_sync<X: Send + Sync>() {}
    send_sync::<BoxedPlan<f64>>();
    send_sync::<Box<dyn ContractionBackend<f32>>>();
}

#[test]
fn injective_layouts() {
    assert!(is_injective_layout(&[2, 3], &[1, 2]));
    assert!(is_injective_layout(&[2, 3], &[-1, 2]));
    assert!(is_injective_layout(&[0, 3], &[0, 0]));
    assert!(is_injective_layout(&[1, 3], &[0, 1]));
    assert!(!is_injective_layout(&[2, 3], &[0, 1]));
    assert!(!is_injective_layout(&[2, 3], &[1, 1]));
}

#[test]
fn backend_errors_keep_their_source() {
    #[derive(Debug, thiserror::Error)]
    #[error("lower layer")]
    struct Lower;
    let e = Error::backend(Lower);
    let src = std::error::Error::source(&e);
    // `Backend` displays the payload; the payload stays downcastable.
    assert_eq!(e.to_string(), "backend: lower layer");
    assert!(src.is_none());
    let Error::Backend(inner) = e else { panic!() };
    assert!(inner.downcast_ref::<Lower>().is_some());
    assert!(Error::from(Unsupported::Reason("x")).is_unsupported());
    assert!(!Error::Internal("x").is_unsupported());
}
