use super::*;

/// The shipped blocking, spelled out.
///
/// Every performance number in `docs/archive/tensorprimitives/notebook/` was taken against exactly
/// these, and the pending `MC`/`KC`/`NC` grid defines its arms relative to
/// them, so changing one is changing what those measurements mean. The
/// analytical model is the reason to have this test: it must stay opt-in,
/// and if it ever becomes the default that is a decision recorded in
/// `docs/archive/tensorprimitives/notebook/`, not a diff that slips through here.
#[test]
fn legacy_blocking_is_unchanged() {
    let d = |real_bytes, a, b| {
        let x = Blocking::derive(real_bytes, a, b);
        (x.mc, x.kc, x.nc)
    };
    // 8-byte reals: `kc = 256`, half a 1 MiB L2 for A, 3 MiB of L3 for B.
    assert_eq!(d(8, 1, 1), (256, 256, 1536), "f64 real");
    assert_eq!(d(8, 2, 2), (128, 256, 768), "c64 planar");
    assert_eq!(d(8, 4, 2), (64, 256, 768), "c64 1m");
    assert_eq!(d(8, 3, 3), (85, 256, 512), "c64 3m");
    // 4-byte reals: `kc = 384`, same two budgets.
    assert_eq!(d(4, 1, 1), (341, 384, 2048), "f32 real");
    assert_eq!(d(4, 2, 2), (170, 384, 1024), "c32 planar");
    assert_eq!(d(4, 4, 2), (85, 384, 1024), "c32 1m");
    assert_eq!(d(4, 3, 3), (113, 384, 682), "c32 3m");
}
