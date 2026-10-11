//! Any partition must be bitwise identical to the serial run: every output
//! element owns one thread and accumulates over the whole of `K` in order.
use super::common::{run_with_width, Shape};

const SHAPE: Shape = Shape {
    m: 203,
    n: 157,
    k: 311,
};

#[test]
fn every_width_is_bitwise_identical_to_width_one() {
    for id in [
        "ref.f64.real-scalar.4x4",
        "ref.f64.real.4x4",
        "ref.f64.direct.4x4",
        "ref.f64.direct-b.4x4",
    ] {
        let base = run_with_width::<f64>(id, 1, SHAPE, false);
        for w in [2usize, 3, 4, 8] {
            let got = run_with_width::<f64>(id, w, SHAPE, false);
            assert!(
                base.iter()
                    .zip(&got)
                    .all(|(x, y)| x.to_bits() == y.to_bits()),
                "{id} width {w}"
            );
        }
    }
}

#[test]
fn aligned_c_lines_are_bitwise_identical_too() {
    let base = run_with_width::<f64>("ref.f64.real.4x4", 1, SHAPE, true);
    for w in [2usize, 4, 8] {
        let got = run_with_width::<f64>("ref.f64.real.4x4", w, SHAPE, true);
        assert!(
            base.iter()
                .zip(&got)
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "aligned width {w}"
        );
    }
    // Alignment must not change the answer either, only the boundaries.
    let unaligned = run_with_width::<f64>("ref.f64.real.4x4", 4, SHAPE, false);
    let aligned = run_with_width::<f64>("ref.f64.real.4x4", 4, SHAPE, true);
    assert!(unaligned
        .iter()
        .zip(&aligned)
        .all(|(x, y)| x.to_bits() == y.to_bits()));
}
