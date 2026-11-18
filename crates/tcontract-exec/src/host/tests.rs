use super::*;

#[test]
fn the_two_rates_of_a_class_agree() {
    for c in [
        HostClass::X86_64Avx512,
        HostClass::X86_64Avx2,
        HostClass::X86_64Scalar,
        HostClass::Aarch64Apple,
        HostClass::Aarch64,
        HostClass::Generic,
    ] {
        let exact = 1.0 / ns_per_flop_of(c) / 2.0;
        assert!(
            (muladds_per_ns_of(c) / exact - 1.0).abs() < 0.01,
            "{}",
            c.name()
        );
    }
}

#[test]
fn rows_per_class() {
    assert_eq!(ns_per_flop_of(HostClass::X86_64Avx512), 0.032);
    assert_eq!(muladds_per_ns_of(HostClass::X86_64Avx512), 15.6);
    for c in [
        HostClass::Aarch64Apple,
        HostClass::X86_64Avx2,
        HostClass::X86_64Scalar,
        HostClass::Aarch64,
        HostClass::Generic,
    ] {
        assert_eq!(ns_per_flop_of(c), 0.0185, "{}", c.name());
        assert_eq!(muladds_per_ns_of(c), 27.0, "{}", c.name());
    }
    // `generic` is the fastest known value of each column.
    let g = HostClass::Generic;
    assert!(ns_per_flop_of(g) <= ns_per_flop_of(HostClass::X86_64Avx512));
    assert!(muladds_per_ns_of(g) >= muladds_per_ns_of(HostClass::X86_64Avx512));
    assert_eq!(HostClass::detect(), HostClass::detect());
    assert_eq!(ns_per_flop(), ns_per_flop_of(HostClass::detect()));
    assert_eq!(muladds_per_ns(), muladds_per_ns_of(HostClass::detect()));
}
