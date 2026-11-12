use super::*;

#[test]
fn a_non_finite_output_is_a_mismatch() {
    for (x, y) in [([1.0], [f64::NAN]), ([f64::NAN], [1.0])] {
        let err = rel_err::<f64>(&x, &y);
        assert_eq!(
            verdict(Dtype::F64, err),
            "MISMATCH",
            "{x:?} vs {y:?}: {err}"
        );
    }
}
