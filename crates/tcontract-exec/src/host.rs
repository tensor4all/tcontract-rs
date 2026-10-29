//! Host class and the serial GEMM rate of each class.
//!
//! The rate the width and batch rules use is keyed by the kernel dispatch the
//! machine takes, not by the CPU model. The class is `threadteam::HostClass`,
//! re-exported here: one enum and one detection for the dispatch costs and the
//! rates below.
//!
//! | class | ns per flop | multiply-adds per ns | source |
//! | --- | --- | --- | --- |
//! | `x86_64-avx512` | 0.032 | 15.6 | AMD Ryzen AI 9 HX 470 at 2.0 GHz (boost off): GEMM f64 0.0641 ns per MAC (`faer-benchmark` `data/probes/20261026-awake-team-linux/tables-consts-exec3.md`, `gemm_per_mac`; the `tc_matmul` seq times of `tables-confirm-exec3.md` agree: 0.0678, 0.0657, 0.0644 ns per MAC at n = 48, 64, 96), halved to a flop (a MAC is two flops), then 1 / (2 * 0.032) = 15.6 |
//! | `aarch64-apple` | 0.0185 | 27 | Apple M5 Max, the serial GEMM rate (about 54 GFLOP/s, threadteam `docs/design/thresholds.md` section 3.2) |
//! | every other class | 0.0185 | 27 | the `generic` row: the fastest known value of each column (the minimum of the ns per flop, the maximum of the multiply-adds per ns over the measured classes); it keeps work serial |
//!
//! The rate of a class and the dispatch costs of the same class in `threadteam`
//! come from the same machine in the same clock state: boost on or off moves both
//! by about 1.66x and leaves their ratio stable within a few percent
//! (`raw-exec3-boost`, `tables-*-exec3-boost.md`).

pub use threadteam::HostClass;

/// Estimated serial time per real flop of a GEMM item on a host class (the
/// table in the module documentation).
pub const fn ns_per_flop_of(class: HostClass) -> f64 {
    match class {
        HostClass::X86_64Avx512 => 0.032,
        HostClass::Aarch64Apple
        | HostClass::X86_64Avx2
        | HostClass::X86_64Scalar
        | HostClass::Aarch64
        | HostClass::Generic => 0.0185,
    }
}

/// Serial multiply-adds per nanosecond of one GEMM item on a host class: the
/// same rate as [`ns_per_flop_of`] (`1 / ns_per_flop / 2`), rounded as the
/// table in the module documentation states it.
pub const fn muladds_per_ns_of(class: HostClass) -> f64 {
    match class {
        HostClass::X86_64Avx512 => 15.6,
        HostClass::Aarch64Apple
        | HostClass::X86_64Avx2
        | HostClass::X86_64Scalar
        | HostClass::Aarch64
        | HostClass::Generic => 27.0,
    }
}

/// Estimated serial time per real flop on this host (`ns_per_flop_of` of
/// [`HostClass::detect`]); the one default the width choice uses everywhere. The
/// width itself is the token's decision (`Exec::width_for` over this estimate).
pub fn ns_per_flop() -> f64 {
    ns_per_flop_of(HostClass::detect())
}

/// Serial multiply-adds per nanosecond on this host (`muladds_per_ns_of` of
/// [`HostClass::detect`]).
pub fn muladds_per_ns() -> f64 {
    muladds_per_ns_of(HostClass::detect())
}

#[cfg(test)]
mod tests {
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
}
