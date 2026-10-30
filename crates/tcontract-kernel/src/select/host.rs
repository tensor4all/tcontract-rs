//! The host tuning class: the microarchitecture bucket the built-in kernel-menu
//! ordering is keyed by, and the `(class, ISA, dtype, complex method)` table that
//! decides that order.
//!
//! CPUID says whether AVX-512 is usable; it does not say how wide the datapath
//! under those instructions is. AMD's Zen 4 (family 0x19) and the Zen 5 mobile
//! parts (Strix Point and relatives, family 0x1A) execute 512-bit instructions
//! as two 256-bit halves, while Intel's AVX-512 parts and AMD's Zen 5 desktop
//! (Granite Ridge) and server (Turin) parts have full 512-bit units. A register
//! block tuned for 32 full-width vector registers is therefore not right on
//! both, and `avx512f` cannot tell them apart (tensor4all/tcontract-rs#156,
//! maintainer decision 2026-10-30, and the advisor rule `R-performance-64`).
//!
//! So the *tuning* class below is a separate fact from the selected [`Isa`] and
//! from [`crate::KernelForce`]: it never renames a family, changes an id or
//! overrides an explicit request. It only orders the built-in menu of one
//! `(dtype, complex method)`, i.e. which measured register block the planner
//! starts from and which alternatives [`crate::Registry`] offers next.
//!
//! # Where the classification comes from
//!
//! * The vendor string (CPUID leaf 0) and the family/model fields (CPUID leaf 1
//!   EAX, decoded as the Intel SDM *CPUID* chapter describes), which is the same
//!   source the kernel itself cannot afford to trust for datapath width.
//! * `avx512f` from [`CpuFeatures`], i.e. `is_x86_feature_detected!`, not the raw
//!   CPUID bit: the OS has to have enabled the AVX-512 state as well.
//! * The datapath classes themselves: issue #156's 2026-10-30 comment, which
//!   quotes AMD's Mike Clark Q&A (via TechPowerUp forum post id 5296275),
//!   <https://www.hwcooling.net/en/zen-5-tested-mobile-core-differs-considerably-from-desktop-one/>,
//!   <https://numberworld.org/blogs/2024_8_7_zen5_avx512_teardown/> and
//!   <https://phoronix.com/review/amd-zen-5-core>. Strix Point runs AVX-512 on
//!   256-bit datapaths on both its Zen 5 and its Zen 5c cores; full 512-bit
//!   datapaths are on Zen 5 desktop and server parts only.
//! * AOCL-BLIS, AMD's BLIS: it keys its configs and kernels by model
//!   (`config/zen4`, `config/zen5`, `kernels/zen4`, `kernels/zen5`), dispatches
//!   at run time and has a `BLIS_MODEL_TYPE` that picks Zen3/Zen4/Zen5 code
//!   paths (Genoa/Bergamo -> Zen4, Turin -> Zen5;
//!   <https://docs.amd.com/r/en-US/57404-AOCL-user-guide/Using-Dynamic-Dispatch>;
//!   AOCL 4.1 added the Zen 4 AVX-512 SGEMM/DGEMM/ZGEMM kernels,
//!   <https://github.com/amd/blis/releases/tag/4.1>). The issue's reading of
//!   AOCL 5.3.2 (`25cad99a`) records that Zen 5 reuses the Zen 4 GEMM kernels,
//!   that CPUID datapath detection exists there but is not used for kernel
//!   choice, and that **a Strix Point part (family 0x1A) reads as classified
//!   ZEN5/TURIN, model ranges not verified**.
//!
//! What the sources above establish is a **part-to-datapath** mapping. The CPUID
//! model ids that identify the parts come from public CPUID listings and are not
//! independently verified here, so each AMD row below says what it is relying on
//! and what is unverified. The table is deliberately conservative where a model
//! number is not backed by a checked source: an unlisted family 0x1A model gets
//! the 256-bit class, the class tuned first per the maintainer decision, so a
//! full-width part loses some headroom rather than being mis-tuned the other
//! way.

use crate::{Isa, Method};

/// Microarchitecture bucket the built-in kernel-menu ordering is keyed by.
///
/// This is not the *scheduling* class (`tcontract_exec::HostClass`, i.e.
/// `threadteam::HostClass`, keyed by the selected ISA), and it is not part of any
/// family id: it carries the datapath width, which the ISA flag cannot.
///
/// # Examples
/// ```
/// use tcontract_kernel::TuningClass;
/// assert_eq!(TuningClass::detect(), TuningClass::detect());
/// assert!(!TuningClass::Intel512.name().is_empty());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TuningClass {
    /// An Intel x86-64 part with usable AVX-512: full-width 512-bit units.
    Intel512,
    /// AMD Zen 5 desktop (Granite Ridge) or server (Turin): full-width 512-bit
    /// units.
    Amd512,
    /// 512-bit instructions on 256-bit datapaths: AMD Zen 4 (family 0x19) and
    /// the Zen 5 mobile parts (family 0x1A models such as Strix Point).
    Amd256,
    /// An x86-64 part without usable AVX-512: AVX2 (plus FMA) or scalar.
    Avx2,
    /// AArch64, where NEON is the only vector instruction set.
    Neon,
    /// Any other host, including an x86 vendor this table does not classify.
    /// The menu keeps its compiled order; no class-specific claim is made.
    Other,
}

impl TuningClass {
    /// The stable name used in reports (`PackedReport::tuning_class`,
    /// `tcbench info`).
    ///
    /// # Examples
    /// ```
    /// use tcontract_kernel::TuningClass;
    /// assert_eq!(TuningClass::Amd256.name(), "amd256");
    /// ```
    pub const fn name(self) -> &'static str {
        match self {
            TuningClass::Intel512 => "intel512",
            TuningClass::Amd512 => "amd512",
            TuningClass::Amd256 => "amd256",
            TuningClass::Avx2 => "avx2",
            TuningClass::Neon => "neon",
            TuningClass::Other => "other",
        }
    }

    /// Classify this host once per process.
    ///
    /// A process cannot see two classes, so this is a single [`std::sync::OnceLock`]
    /// over CPUID facts, not a data cache and not part of any cache key. Without
    /// the `std` feature there is no runtime CPUID or feature detection and the
    /// answer is [`TuningClass::Other`] (or [`TuningClass::Neon`] on AArch64,
    /// where NEON is architectural).
    ///
    /// # Examples
    /// ```
    /// use tcontract_kernel::TuningClass;
    /// let class = TuningClass::detect();
    /// // Whatever this host is, the name is the one reports carry.
    /// assert_eq!(class.name(), class.name());
    /// ```
    pub fn detect() -> Self {
        #[cfg(feature = "std")]
        {
            static CLASS: std::sync::OnceLock<TuningClass> = std::sync::OnceLock::new();
            *CLASS.get_or_init(detect_uncached)
        }
        #[cfg(not(feature = "std"))]
        {
            detect_uncached()
        }
    }

    /// Whether this class's **class-wide** kernel-menu order is the compiled
    /// order rather than a measurement on the class itself (issue #156).
    ///
    /// True for the classes whose class-wide row stands in for a measurement that
    /// has not landed (`Amd512`, `Amd256`) and for the two that make no ordering
    /// claim at all (`Avx2`, whose register blocks are unmeasured, and `Other`).
    /// A report that quotes a plan's kernel shape must quote this too: while it
    /// is true, the class makes no measured ordering claim for every menu. A
    /// measured per-menu row can exist above a still-provisional class-wide row;
    /// `Amd256`'s AVX-512 menus are in that state after the stage-3 run.
    ///
    /// # Examples
    /// ```
    /// use tcontract_kernel::TuningClass;
    /// assert!(TuningClass::Amd512.ordering_is_provisional());
    /// assert!(TuningClass::Amd256.ordering_is_provisional());
    /// ```
    pub fn ordering_is_provisional(self) -> bool {
        self.class_wide_order().is_some_and(|row| row.provisional)
    }

    /// The table row that states this class's order for every `(ISA, dtype,
    /// method)` menu it has no more specific row for.
    fn class_wide_order(self) -> Option<&'static OrderRow> {
        ORDER_TABLE.iter().find(|row| {
            row.class == self && row.isa.is_none() && row.dtype.is_none() && row.method.is_none()
        })
    }
}

fn detect_uncached() -> TuningClass {
    #[cfg(feature = "std")]
    {
        #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            x86::classify_host()
        }
        #[cfg(target_arch = "aarch64")]
        {
            TuningClass::Neon
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            TuningClass::Other
        }
    }
    #[cfg(not(feature = "std"))]
    {
        if cfg!(target_arch = "aarch64") {
            TuningClass::Neon
        } else {
            TuningClass::Other
        }
    }
}

// The classification is x86-only logic: `avx512f` cannot be present elsewhere,
// and AArch64 answers `Neon` before reaching a table. Keeping it in one module
// also keeps the table and its decode next to each other on the targets that
// compile them.
#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
mod x86 {
    use super::TuningClass;
    use crate::CpuFeatures;

    /// The vendor string CPUID reports.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) enum Vendor {
        Intel,
        Amd,
        Other,
    }

    impl Vendor {
        fn of(vendor_string: &[u8; 12]) -> Self {
            match vendor_string {
                b"GenuineIntel" => Vendor::Intel,
                b"AuthenticAMD" => Vendor::Amd,
                _ => Vendor::Other,
            }
        }
    }

    /// The CPUID identity the table is matched against.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) struct CpuIdentity {
        pub(super) vendor: Vendor,
        pub(super) family: u32,
        pub(super) model: u32,
    }

    /// Read the vendor string and the decoded family/model.
    ///
    /// CPUID is part of the x86-64 baseline and of every 32-bit x86 CPU a Rust
    /// `std` supports. Newer toolchains declare `__cpuid` safe; the MSRV (1.89)
    /// still declares it `unsafe`, hence the `unsafe` blocks that are unused on
    /// the newer ones.
    #[allow(unused_unsafe)]
    pub(super) fn cpuid_identity() -> CpuIdentity {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::__cpuid;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::__cpuid;

        // SAFETY: CPUID leaf 0 exists on every CPU this code can run on (above).
        let leaf0 = unsafe { __cpuid(0) };
        let mut vendor_string = [0u8; 12];
        // CPUID returns the vendor in EBX, EDX, ECX order, each little-endian.
        vendor_string[0..4].copy_from_slice(&leaf0.ebx.to_le_bytes());
        vendor_string[4..8].copy_from_slice(&leaf0.edx.to_le_bytes());
        vendor_string[8..12].copy_from_slice(&leaf0.ecx.to_le_bytes());

        // SAFETY: as for leaf 0; leaf 1 is reported by every such CPU.
        let eax = unsafe { __cpuid(1) }.eax;
        let base_family = (eax >> 8) & 0xF;
        let ext_family = (eax >> 20) & 0xFF;
        let base_model = (eax >> 4) & 0xF;
        let ext_model = (eax >> 16) & 0xF;
        // Intel SDM, CPUID leaf 01H EAX: the extended family is added to 0xF, and
        // the extended model replaces the base model for families 0x6 and 0xF.
        let family = if base_family == 0xF {
            base_family + ext_family
        } else {
            base_family
        };
        let model = if base_family == 0x6 || base_family == 0xF {
            (ext_model << 4) | base_model
        } else {
            base_model
        };
        CpuIdentity {
            vendor: Vendor::of(&vendor_string),
            family,
            model,
        }
    }

    /// One row of the `(vendor, family, model range, avx512f)` -> class table.
    ///
    /// Rows are tried in order and the first match wins, so a specific model range
    /// must be listed **above** its family-wide fallback; `tests.rs` proves no row
    /// is unreachable for its own representative identity.
    pub(super) struct ClassRow {
        pub(super) vendor: Vendor,
        /// `None` matches every family of the vendor.
        pub(super) family: Option<u32>,
        /// Inclusive model range; `None` matches every model of the family(ies).
        pub(super) models: Option<(u32, u32)>,
        /// Whether AVX-512F has to be usable for the row to match.
        pub(super) avx512f: bool,
        pub(super) class: TuningClass,
    }

    impl ClassRow {
        pub(super) fn matches(
            &self,
            vendor: Vendor,
            family: u32,
            model: u32,
            avx512f: bool,
        ) -> bool {
            self.vendor == vendor
                && self.family.is_none_or(|f| f == family)
                && self
                    .models
                    .is_none_or(|(lo, hi)| (lo..=hi).contains(&model))
                && self.avx512f == avx512f
        }
    }

    /// `(vendor, family, model range, avx512f)` -> [`TuningClass`], first match wins.
    ///
    /// Model numbers in **hexadecimal**. Where a row's range or part mapping is not
    /// backed by the sources in the module documentation it says so in the comment
    /// above it.
    // `static`, not `const`: a `const` table is re-materialized at every use, and
    // the table's own rows would then not be the ones `class_row` answers with.
    pub(super) static CLASS_TABLE: &[ClassRow] = &[
        // Intel, no model table needed: Intel's AVX-512 implementations all have
        // full-width 512-bit FMA units, and the client parts that fuse AVX-512
        // off (Alder Lake and later) report `avx512f` as absent, which sends them
        // to the row below. Unverified here: the *number* of 512-bit FMA ports
        // differs between parts (one on Rocket Lake and on the 1-FMA server
        // parts, two elsewhere); the class does not model port count, and no
        // measured row needs it yet.
        ClassRow {
            vendor: Vendor::Intel,
            family: None,
            models: None,
            avx512f: true,
            class: TuningClass::Intel512,
        },
        ClassRow {
            vendor: Vendor::Intel,
            family: None,
            models: None,
            avx512f: false,
            class: TuningClass::Avx2,
        },
        // AMD Zen 5, family 0x1A. Turin (EPYC 9005, server): full 512-bit units
        // (issue #156, 2026-10-30). **The model sub-range is unverified**: the
        // model ids are from public CPUID listings, and AOCL-BLIS classifies every
        // family 0x1A part as ZEN5/TURIN without a verified model range.
        ClassRow {
            vendor: Vendor::Amd,
            family: Some(0x1A),
            models: Some((0x00, 0x0F)),
            avx512f: true,
            class: TuningClass::Amd512,
        },
        // Strix Point (Ryzen AI 9 HX 370/470 and relatives, the `zen5-cpu`
        // measurement host): 512-bit instructions on 256-bit datapaths, on the
        // Zen 5 and the Zen 5c cores alike (issue #156, 2026-10-30). The model id
        // `0x24` is from public CPUID listings, unverified here.
        ClassRow {
            vendor: Vendor::Amd,
            family: Some(0x1A),
            models: Some((0x24, 0x24)),
            avx512f: true,
            class: TuningClass::Amd256,
        },
        // Granite Ridge (Ryzen 9000 desktop): full 512-bit units (issue #156).
        // **The model id `0x44` is unverified here**, and Fire Range, the mobile
        // part built on the same silicon, is assumed to share it — also
        // unverified.
        ClassRow {
            vendor: Vendor::Amd,
            family: Some(0x1A),
            models: Some((0x44, 0x44)),
            avx512f: true,
            class: TuningClass::Amd512,
        },
        // Any other family 0x1A model: assume the 256-bit datapath, the class the
        // maintainer decision tunes first (#156), rather than promising a
        // full-width block this table cannot establish. This is also the
        // family-0x1A answer when AVX-512 is not usable: it falls through to the
        // family-wide rows below, which is the same `Avx2` a family row would
        // repeat.
        ClassRow {
            vendor: Vendor::Amd,
            family: Some(0x1A),
            models: None,
            avx512f: true,
            class: TuningClass::Amd256,
        },
        // AMD Zen 4, family 0x19: 512-bit instructions on 256-bit datapaths
        // (Genoa, Bergamo, Raphael, Phoenix, Hawk Point; #156). Family-level, so
        // no model id is needed: every Zen 4 part has AVX-512 and none has a
        // full-width datapath. Zen 3 (family 0x19) and Zen 1/Zen 2 (family 0x17)
        // have no usable AVX-512 and need no row of their own: the family-wide
        // rows below answer `Avx2`, which is their class.
        ClassRow {
            vendor: Vendor::Amd,
            family: Some(0x19),
            models: None,
            avx512f: true,
            class: TuningClass::Amd256,
        },
        // Any other AMD family. No AMD part outside families 0x19 and 0x1A is known
        // to have AVX-512, so this is the ISA flag answering, not a model claim.
        ClassRow {
            vendor: Vendor::Amd,
            family: None,
            models: None,
            avx512f: false,
            class: TuningClass::Avx2,
        },
        ClassRow {
            vendor: Vendor::Amd,
            family: None,
            models: None,
            avx512f: true,
            class: TuningClass::Amd256,
        },
        // Any other x86 vendor (Hygon, Zhaoxin, ...): no knowledge, no claim.
        ClassRow {
            vendor: Vendor::Other,
            family: None,
            models: None,
            avx512f: false,
            class: TuningClass::Avx2,
        },
        ClassRow {
            vendor: Vendor::Other,
            family: None,
            models: None,
            avx512f: true,
            class: TuningClass::Other,
        },
    ];

    /// The table row an identity matches, or `None` when the table has no row for it
    /// (the caller then falls back to [`TuningClass::Other`]).
    pub(super) fn class_row(
        vendor: Vendor,
        family: u32,
        model: u32,
        avx512f: bool,
    ) -> Option<&'static ClassRow> {
        CLASS_TABLE
            .iter()
            .find(|row| row.matches(vendor, family, model, avx512f))
    }

    /// The pure classification, separated from CPUID so the table can be tested on
    /// any host.
    pub(super) fn classify(vendor: Vendor, family: u32, model: u32, avx512f: bool) -> TuningClass {
        class_row(vendor, family, model, avx512f).map_or(TuningClass::Other, |row| row.class)
    }

    /// The host's class, from this CPU's CPUID identity and *usable* AVX-512 (the
    /// feature detector, so the OS state counts, not the raw CPUID bit).
    pub(super) fn classify_host() -> TuningClass {
        let id = cpuid_identity();
        classify(
            id.vendor,
            id.family,
            id.model,
            CpuFeatures::detect().avx512f,
        )
    }
}

/// The storage dtype a menu-ordering row is keyed by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Dtype {
    F32,
    F64,
    C32,
    C64,
}

impl Dtype {
    /// The dtype of a storage type, from the real size and the element type's
    /// own complex flag (the split the registry already keys on).
    pub(crate) const fn of(real_size: usize, is_complex: bool) -> Self {
        match (real_size, is_complex) {
            (4, false) => Dtype::F32,
            (4, true) => Dtype::C32,
            (_, false) => Dtype::F64,
            (_, true) => Dtype::C64,
        }
    }
}

/// One row of the `(class, ISA, dtype, complex method)` -> menu-order table.
struct OrderRow {
    class: TuningClass,
    /// `None` matches every ISA menu. A row that names an ISA only orders that
    /// menu: an AVX-512 row must not permute an AVX2 menu forced on the same
    /// host (which can happen when the class is AMD but the ISA is pinned).
    isa: Option<Isa>,
    /// `None` matches every storage dtype.
    dtype: Option<Dtype>,
    /// `None` matches the real menus (no method) and every complex method.
    method: Option<Method>,
    /// Kernel-menu positions in preference order, default first; `None` keeps
    /// the compiled order.
    order: Option<&'static [usize]>,
    /// The order is not a measurement on this class: the compiled order stands
    /// in, either until a measurement lands (`Amd512`, `Amd256`) or because the
    /// class makes no ordering claim at all (`Avx2`, `Other`). A report must not
    /// call such an order measured. A measured per-menu row can sit above a
    /// still-provisional class-wide row; see [`TuningClass::ordering_is_provisional`].
    provisional: bool,
}

// Rows are keyed by class, then optionally by ISA, dtype and complex method. A
// more specific row goes **above** its class-wide row. The three x86 classes
// with an AVX-512 register block are distinct class-wide rows even though they
// agree: `Intel512` is the order measured on Cascade Lake / Ice Lake
// (`kernels/x86/avx512.rs`,
// `tensorprimitives/bench-results/phase3-kernel-shapes.txt`), while `Amd512`
// (Zen 5 desktop/server) and the class-wide `Amd256` (Zen 4 / Zen 5 mobile) stand
// in for measurements that have not landed.
const INTEL512_ORDER: OrderRow = OrderRow {
    class: TuningClass::Intel512,
    isa: None,
    dtype: None,
    method: None,
    order: None,
    provisional: false,
};
const AMD512_ORDER: OrderRow = OrderRow {
    class: TuningClass::Amd512,
    isa: None,
    dtype: None,
    method: None,
    order: None,
    provisional: true,
};
// The Amd256 stage-3 measurement (issue #156, `zen5-cpu` = Ryzen AI 9 HX 470,
// Strix Point, 256-bit datapath). The reading is
// <https://github.com/tensor4all/tcontract-rs/issues/156#issuecomment-6520150036>;
// the stage-3 report with the per-menu table is
// <https://github.com/tensor4all/tcontract-rs/issues/156#issuecomment-6519490450>
// (section B). Only the AVX-512 menus were measured, so these rows name
// `Isa::Avx512`: the AVX2, 1m and 3m menus stay on the class-wide provisional
// row below.
//
// f32 real: compiled position 1 (`avx512.f32.real.32x8`) beats the compiled
// default position 0 (`48x8`) on almost every row, including square 1024
// (35480 vs 35567 us) and `gemm_tile m=282 k=141 n=12` (25.7 vs 29.0 us).
// Position 2 (`16x10`) stays last on the menu.
const AMD256_AVX512_F32_REAL_ORDER: OrderRow = OrderRow {
    class: TuningClass::Amd256,
    isa: Some(Isa::Avx512),
    dtype: Some(Dtype::F32),
    method: None,
    order: Some(&[1, 0, 2]),
    provisional: false,
};
// f64 real, c64 native and c32 native: the compiled default (24x8, 16x6 and
// 32x6 respectively) is as good as or better than its alternates on the same
// stage-3 run, so these menus keep their compiled order. They are marked
// measured (`provisional: false`) because the measurement covers this menu;
// that does not make the class-wide row measured.
const AMD256_AVX512_F64_REAL_STANDS: OrderRow = OrderRow {
    class: TuningClass::Amd256,
    isa: Some(Isa::Avx512),
    dtype: Some(Dtype::F64),
    method: None,
    order: None,
    provisional: false,
};
const AMD256_AVX512_C64_NATIVE_STANDS: OrderRow = OrderRow {
    class: TuningClass::Amd256,
    isa: Some(Isa::Avx512),
    dtype: Some(Dtype::C64),
    method: Some(Method::Native),
    order: None,
    provisional: false,
};
const AMD256_AVX512_C32_NATIVE_STANDS: OrderRow = OrderRow {
    class: TuningClass::Amd256,
    isa: Some(Isa::Avx512),
    dtype: Some(Dtype::C32),
    method: Some(Method::Native),
    order: None,
    provisional: false,
};
// The class-wide Amd256 row: every menu the stage-3 run did not measure (AVX2,
// 1m, 3m) keeps the compiled order, and the class-wide claim stays provisional
// so a report does not call those menus measured.
const AMD256_ORDER: OrderRow = OrderRow {
    class: TuningClass::Amd256,
    isa: None,
    dtype: None,
    method: None,
    order: None,
    provisional: true,
};
/// AVX2 keeps the compiled order, which is the only one it has: its register
/// blocks are themselves provisional and unmeasured (`cfg_avx2_f64` in
/// `kernels/x86/avx2.rs`), so there is no ranking between them to key. That is
/// not a measurement, and the row says so.
const AVX2_ORDER: OrderRow = OrderRow {
    class: TuningClass::Avx2,
    isa: None,
    dtype: None,
    method: None,
    order: None,
    provisional: true,
};
/// NEON's menus are the shapes measured on an M3 Max and ordered by that
/// measurement (`kernels/aarch64/neon.rs`, "The shipped shapes are MEASURED").
const NEON_ORDER: OrderRow = OrderRow {
    class: TuningClass::Neon,
    isa: None,
    dtype: None,
    method: None,
    order: None,
    provisional: false,
};
/// A host this table does not classify makes no ordering claim at all, so its
/// compiled order is not a measurement either.
const OTHER_ORDER: OrderRow = OrderRow {
    class: TuningClass::Other,
    isa: None,
    dtype: None,
    method: None,
    order: None,
    provisional: true,
};

static ORDER_TABLE: &[OrderRow] = &[
    INTEL512_ORDER,
    AMD512_ORDER,
    AMD256_AVX512_F32_REAL_ORDER,
    AMD256_AVX512_F64_REAL_STANDS,
    AMD256_AVX512_C64_NATIVE_STANDS,
    AMD256_AVX512_C32_NATIVE_STANDS,
    AMD256_ORDER,
    AVX2_ORDER,
    NEON_ORDER,
    OTHER_ORDER,
];

/// Look up the kernel-menu ordering of one `(class, ISA, dtype, method)` menu:
/// positions in preference order, or `None` for the compiled order. The first
/// matching row wins, so a specific row must be listed above its class-wide row;
/// a class with no row at all keeps the compiled order. `isa` is the menu being
/// built, so an AVX-512 row cannot reorder an AVX2 menu forced on the same host.
pub(crate) fn menu_order(
    class: TuningClass,
    isa: Isa,
    dtype: Dtype,
    method: Option<Method>,
) -> Option<&'static [usize]> {
    ORDER_TABLE
        .iter()
        .find(|row| {
            row.class == class
                && row.isa.is_none_or(|i| i == isa)
                && row.dtype.is_none_or(|d| d == dtype)
                && row.method.is_none_or(|m| Some(m) == method)
        })
        .and_then(|row| row.order)
}

/// Apply a menu ordering in place: the listed positions first, in the order
/// given, then every other entry in its compiled order.
///
/// `order` is a table row, so a position outside `menu` or a repeated one is a
/// bug in [`ORDER_TABLE`], not a host fact the caller could recover from: it is
/// debug-asserted here, and `every_order_row_lists_distinct_positions` checks
/// every row at test time. There is no silent repair a stale row could hide in.
pub(crate) fn apply_order<T>(menu: &mut Vec<T>, order: Option<&[usize]>) {
    let Some(order) = order else { return };
    let n = menu.len();
    let mut listed = vec![false; n];
    let mut positions: Vec<usize> = Vec::with_capacity(n);
    for &position in order {
        debug_assert!(
            position < n,
            "menu order {order:?} names position {position} of a {n}-entry menu"
        );
        if let Some(taken) = listed.get_mut(position) {
            debug_assert!(!*taken, "menu order {order:?} repeats position {position}");
            *taken = true;
        }
        positions.push(position);
    }
    positions.extend((0..n).filter(|&i| !listed[i]));
    let mut slots: Vec<Option<T>> = menu.drain(..).map(Some).collect();
    let mut out: Vec<T> = Vec::with_capacity(n);
    for position in positions {
        // INVARIANT: `order` lists distinct positions below `n` (debug-asserted
        // above, and checked for every table row by
        // `every_order_row_lists_distinct_positions`), so appending the unlisted
        // indices makes `positions` a permutation of `0..n` and every slot is
        // still `Some` here.
        let item = slots[position]
            .take()
            .expect("a table menu order is a permutation of the menu");
        out.push(item);
    }
    *menu = out;
}

#[cfg(test)]
mod tests;
