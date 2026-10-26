//! Host class and the serial GEMM rate of each class.
//!
//! The rate the width and batch rules use is keyed by the kernel dispatch the
//! machine takes, not by the CPU model. `HostClass` and `HostClass::detect` are
//! the same enum and the same detection as `threadteam::HostClass`, `strided`'s
//! and `faer`'s cost tables, and are kept in sync with them by hand; the
//! `tcontract` crate reuses this copy.
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

use std::sync::OnceLock;

/// The class of the machine, keyed by the kernel dispatch it will take.
///
/// Kept in sync with `threadteam::HostClass`. Detected once per process by
/// [`HostClass::detect`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HostClass {
    /// `x86_64` with AVX-512F at run time (`x86_64-avx512`).
    X86_64Avx512,
    /// `x86_64` with AVX2 and FMA, without AVX-512F (`x86_64-avx2`).
    X86_64Avx2,
    /// Any other `x86_64` (`x86_64-scalar`).
    X86_64Scalar,
    /// `aarch64` on macOS (`aarch64-apple`).
    Aarch64Apple,
    /// Any other `aarch64` (`aarch64`).
    Aarch64,
    /// Everything else (`generic`).
    Generic,
}

impl HostClass {
    /// The class of this host, detected at the first call and cached
    /// (`OnceLock`; `is_x86_feature_detected!` on x86_64).
    pub fn detect() -> Self {
        static CLASS: OnceLock<HostClass> = OnceLock::new();
        *CLASS.get_or_init(Self::detect_uncached)
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_uncached() -> Self {
        if std::arch::is_x86_feature_detected!("avx512f") {
            HostClass::X86_64Avx512
        } else if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            HostClass::X86_64Avx2
        } else {
            HostClass::X86_64Scalar
        }
    }

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    fn detect_uncached() -> Self {
        HostClass::Aarch64Apple
    }

    #[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
    fn detect_uncached() -> Self {
        HostClass::Aarch64
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn detect_uncached() -> Self {
        HostClass::Generic
    }

    /// The class name used in the docs and the tables.
    pub const fn name(self) -> &'static str {
        match self {
            HostClass::X86_64Avx512 => "x86_64-avx512",
            HostClass::X86_64Avx2 => "x86_64-avx2",
            HostClass::X86_64Scalar => "x86_64-scalar",
            HostClass::Aarch64Apple => "aarch64-apple",
            HostClass::Aarch64 => "aarch64",
            HostClass::Generic => "generic",
        }
    }
}

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
