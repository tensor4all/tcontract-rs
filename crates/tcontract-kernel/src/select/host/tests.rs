#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
use super::x86::{class_row, classify, cpuid_identity, Vendor, CLASS_TABLE};
use super::*;
use crate::CpuFeatures;

const CLASSES: [TuningClass; 6] = [
    TuningClass::Intel512,
    TuningClass::Amd512,
    TuningClass::Amd256,
    TuningClass::Avx2,
    TuningClass::Neon,
    TuningClass::Other,
];
#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[test]
fn strix_point_is_amd256() {
    // The `zen5-cpu` measurement host of issue #156: 512-bit instructions on
    // 256-bit datapaths.
    assert_eq!(classify(Vendor::Amd, 0x1A, 0x24, true), TuningClass::Amd256);
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[test]
fn turin_is_amd512() {
    for model in [0x00, 0x01, 0x02, 0x0F] {
        assert_eq!(
            classify(Vendor::Amd, 0x1A, model, true),
            TuningClass::Amd512,
            "model {model:#x}"
        );
    }
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[test]
fn granite_ridge_is_amd512() {
    assert_eq!(classify(Vendor::Amd, 0x1A, 0x44, true), TuningClass::Amd512);
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[test]
fn zen4_is_amd256() {
    // Genoa, Bergamo, Raphael, Phoenix, Hawk Point: family 0x19 with AVX-512.
    for model in [0x11, 0x17, 0x44, 0x74] {
        assert_eq!(
            classify(Vendor::Amd, 0x19, model, true),
            TuningClass::Amd256,
            "model {model:#x}"
        );
    }
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[test]
fn intel_ice_lake_is_intel512() {
    // Ice Lake client: family 6, model 0x7E, with AVX-512 enabled.
    assert_eq!(
        classify(Vendor::Intel, 0x6, 0x7E, true),
        TuningClass::Intel512
    );
    // Cascade Lake / Skylake-SP server.
    assert_eq!(
        classify(Vendor::Intel, 0x6, 0x55, true),
        TuningClass::Intel512
    );
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[test]
fn zen2_and_zen3_are_avx2() {
    // Zen 2 (Matisse) and Zen 3 (Vermeer): no AVX-512.
    assert_eq!(classify(Vendor::Amd, 0x17, 0x71, false), TuningClass::Avx2);
    assert_eq!(classify(Vendor::Amd, 0x19, 0x21, false), TuningClass::Avx2);
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[test]
fn intel_without_usable_avx512_is_avx2() {
    // Alder Lake fuses AVX-512 off, so CPUID reports no usable `avx512f`.
    assert_eq!(classify(Vendor::Intel, 0x6, 0x97, false), TuningClass::Avx2);
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[test]
fn an_unlisted_zen5_model_assumes_the_256_bit_datapath() {
    assert_eq!(classify(Vendor::Amd, 0x1A, 0x3F, true), TuningClass::Amd256);
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[test]
fn an_unknown_vendor_makes_no_avx512_claim() {
    assert_eq!(classify(Vendor::Other, 0x6, 0x8F, false), TuningClass::Avx2);
    assert_eq!(classify(Vendor::Other, 0x6, 0x8F, true), TuningClass::Other);
}

/// A representative `(family, model)` for the table row at `index`: the row's
/// own range where it has one, otherwise a family member no earlier row claims.
#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
fn representative(index: usize) -> (u32, u32) {
    let row = &CLASS_TABLE[index];
    let family = row.family.unwrap_or(0x6);
    let model = match row.models {
        Some((lo, _)) => lo,
        None => (0..=0xFF)
            .find(|&model| {
                !CLASS_TABLE[..index]
                    .iter()
                    .any(|earlier| earlier.matches(row.vendor, family, model, row.avx512f))
            })
            .expect("a family-wide row leaves some model unclaimed"),
    };
    (family, model)
}

/// Every row must be the one its own representative identity reaches: a row
/// placed below a broader row would be dead, and the classes it claims to
/// distinguish would silently collapse.
#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[test]
fn every_table_row_is_reachable() {
    for (i, row) in CLASS_TABLE.iter().enumerate() {
        let (family, model) = representative(i);
        let hit = class_row(row.vendor, family, model, row.avx512f);
        assert!(
            hit.is_some_and(|hit| core::ptr::eq(hit, row)),
            "row {i} ({:?}, vendor {:?}, family {family:#x}, model {model:#x}, \
             avx512f {}) is shadowed by an earlier row",
            row.class,
            row.vendor,
            row.avx512f,
        );
        assert_eq!(classify(row.vendor, family, model, row.avx512f), row.class);
    }
}

/// The class is a process constant, so the detector must agree with the table it
/// documents and must not depend on being called twice.
#[test]
fn detect_agrees_with_the_table_and_is_stable() {
    let class = TuningClass::detect();
    assert_eq!(class, TuningClass::detect());
    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        let id = cpuid_identity();
        assert_eq!(
            class,
            classify(
                id.vendor,
                id.family,
                id.model,
                CpuFeatures::detect().avx512f
            ),
            "cpuid {id:?}"
        );
    }
    #[cfg(target_arch = "aarch64")]
    assert_eq!(class, TuningClass::Neon);
}

/// The 512-bit classes require usable AVX-512, so this host cannot be one of
/// them without the feature.
#[test]
fn a_host_without_avx512_is_never_a_512_bit_class() {
    if CpuFeatures::detect().avx512f {
        return;
    }
    let class = TuningClass::detect();
    assert!(
        matches!(
            class,
            TuningClass::Avx2 | TuningClass::Neon | TuningClass::Other
        ),
        "{class:?}"
    );
}

/// Every class-wide row still keeps the compiled menu order, which is the one
/// the ISA modules measured (or, for AVX2, their provisional own order). Only a
/// measured per-menu row may reorder, and only that menu.
#[test]
fn class_wide_rows_keep_the_compiled_menu_order() {
    for class in CLASSES {
        let row = class
            .class_wide_order()
            .unwrap_or_else(|| panic!("{class:?} has no class-wide order row"));
        assert_eq!(row.order, None, "{class:?}");
        // A class-wide row has no ISA/dtype/method key, so it is the fallback
        // for every menu of its class. A measured per-menu row above it shadows
        // it for exactly that menu.
        assert_eq!(row.isa, None, "{class:?}");
        assert_eq!(row.dtype, None, "{class:?}");
        assert_eq!(row.method, None, "{class:?}");
    }
}

/// The one reordering the stage-3 Amd256 run measured: f32 real on the AVX-512
/// menu puts 32x8 (compiled position 1) first, ahead of 48x8 (position 0).
#[test]
fn amd256_avx512_f32_real_prefers_the_32x8_block() {
    assert_eq!(
        menu_order(TuningClass::Amd256, Isa::Avx512, Dtype::F32, None),
        Some(&[1, 0, 2][..])
    );
    // Keyed by ISA: an AVX2 menu forced on the same class is untouched.
    assert_eq!(
        menu_order(TuningClass::Amd256, Isa::Avx2, Dtype::F32, None),
        None
    );
    // Nor does the row leak to another class's AVX-512 menu.
    for class in [TuningClass::Intel512, TuningClass::Amd512] {
        assert_eq!(
            menu_order(class, Isa::Avx512, Dtype::F32, None),
            None,
            "{class:?}"
        );
    }
}

/// The row really permutes the f32 real AVX-512 menu the planner builds: the
/// compiled order `[48x8, 32x8, 16x10]` becomes `[32x8, 48x8, 16x10]`. (This is
/// the exact pair of menus `tcontract`'s resolve test uses to check that Auto
/// reads the first and `idx=0` the second.)
#[test]
fn amd256_f32_order_turns_the_compiled_menu_into_the_measured_one() {
    let mut menu = vec![(48, 8), (32, 8), (16, 10)];
    apply_order(
        &mut menu,
        menu_order(TuningClass::Amd256, Isa::Avx512, Dtype::F32, None),
    );
    assert_eq!(menu, [(32, 8), (48, 8), (16, 10)]);
}

/// The other Amd256 AVX-512 menus the stage-3 run measured keep their compiled
/// order, and they say so: measured on their menu, not on the whole class.
#[test]
fn amd256_avx512_measured_menus_keep_their_default() {
    for (dtype, method) in [
        (Dtype::F64, None),
        (Dtype::C64, Some(Method::Native)),
        (Dtype::C32, Some(Method::Native)),
    ] {
        assert_eq!(
            menu_order(TuningClass::Amd256, Isa::Avx512, dtype, method),
            None,
            "{dtype:?} {method:?}"
        );
        let row = ORDER_TABLE
            .iter()
            .find(|row| {
                row.class == TuningClass::Amd256
                    && row.isa == Some(Isa::Avx512)
                    && row.dtype == Some(dtype)
                    && row.method == method
            })
            .unwrap_or_else(|| panic!("no measured Amd256 row for {dtype:?} {method:?}"));
        assert!(!row.provisional, "{dtype:?} {method:?} is a measurement");
    }
    // The class-wide row stays provisional: the AVX2, 1m and 3m menus are
    // unmeasured, so their compiled order is not a measurement.
    assert!(TuningClass::Amd256.ordering_is_provisional());
}

/// Only the classes whose order is a measurement on the class itself are
/// reported as measured: `Intel512` and `Neon` have one, the other four have not.
#[test]
fn unmeasured_orders_are_marked_provisional() {
    for class in CLASSES {
        let row = class
            .class_wide_order()
            .unwrap_or_else(|| panic!("{class:?} has no class-wide order row"));
        assert_eq!(row.class, class);
        let measured = matches!(class, TuningClass::Intel512 | TuningClass::Neon);
        assert_eq!(class.ordering_is_provisional(), !measured, "{class:?}");
    }
}

/// Every ordering row lists distinct menu positions. The row's positions are
/// applied to menus of several lengths, so the range is checked where they are
/// used; a repeated position is a table bug no host can repair, and
/// `apply_order` only debug-asserts it rather than silently ignoring it.
#[test]
fn every_order_row_lists_distinct_positions() {
    for row in ORDER_TABLE {
        let Some(order) = row.order else { continue };
        for (i, &position) in order.iter().enumerate() {
            assert!(
                !order[..i].contains(&position),
                "{:?}: {order:?} repeats position {position}",
                row.class
            );
        }
    }
}

/// The reordering the table will use once a class is measured: listed positions
/// first, everything else after them in the compiled order.
#[test]
fn apply_order_permutes_without_dropping_entries() {
    let mut menu = vec!["a", "b", "c"];
    apply_order(&mut menu, None);
    assert_eq!(menu, ["a", "b", "c"]);

    let mut menu = vec!["a", "b", "c"];
    apply_order(&mut menu, Some(&[2, 0]));
    assert_eq!(menu, ["c", "a", "b"], "listed first, the rest in order");

    let mut menu = vec!["a", "b", "c", "d"];
    apply_order(&mut menu, Some(&[3, 1]));
    assert_eq!(menu, ["d", "b", "a", "c"]);

    let mut menu = vec!["a", "b", "c"];
    apply_order(&mut menu, Some(&[]));
    assert_eq!(menu, ["a", "b", "c"]);
}
