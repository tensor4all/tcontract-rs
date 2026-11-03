use super::*;

#[test]
fn built_in_ids_follow_isa_dtype_scheme_geometry() {
    let id = builtin_id;
    assert_eq!(
        id("avx2", "f64", TileFormat::Real, 8, 6),
        "avx2.f64.real.8x6"
    );
    assert_eq!(
        id("avx512", "c32", TileFormat::OneM, 8, 12),
        "avx512.c32.1m.8x12"
    );
    assert_eq!(
        id("scalar", "f64", TileFormat::Real, 4, 4),
        "ref.f64.real-scalar.4x4"
    );
    assert_eq!(
        id("scalar", "c64", TileFormat::Planar, 2, 4),
        "ref.c64.planar.2x4"
    );
    // The NEON menus cannot be listed on an x86 host; their names come from
    // the same function.
    assert_eq!(
        id("neon", "f64", TileFormat::Real, 16, 3),
        "neon.f64.real.16x3"
    );
    assert_eq!(
        id("neon", "c64", TileFormat::ThreeM, 2, 8),
        "neon.c64.3m.2x8"
    );
    assert_eq!(
        id("neon", "c32", TileFormat::Planar, 8, 6),
        "neon.c32.planar.8x6"
    );
}
