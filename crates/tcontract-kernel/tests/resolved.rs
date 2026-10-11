use tcontract_kernel::{KernelChoice, Layout, ResolvedGemm, SelectError, TileFormat, C64};

#[test]
fn explicit_portable_family_freezes_geometry_and_formats() {
    let rg = ResolvedGemm::<f64>::resolve::<f64>(&KernelChoice::Id("ref.f64.real.4x4".into()), 1)
        .unwrap();
    assert_eq!(rg.family().id, "ref.f64.real.4x4");
    assert_eq!(
        (rg.mr, rg.nr, rg.a_per_k, rg.b_per_k, rg.tile_bound),
        (4, 4, 4, 4, 16)
    );
    assert_eq!(
        (rg.a_layout, rg.b_layout, rg.tile_fmt),
        (Layout::Real, Layout::Real, TileFormat::Real)
    );
    assert_eq!(rg.mc % rg.mr, 0);
    assert_eq!(rg.nc % rg.nr, 0);
    assert!(rg.kc > 0);
}

#[test]
fn portable_native_complex_freezes_interleaved_geometry() {
    // Forced: Auto picks the widest built-in family of the machine.
    let id = KernelChoice::Id("ref.c64.native.4x4".into());
    let auto = ResolvedGemm::<f64>::resolve::<C64>(&id, 1).unwrap();
    assert_eq!(auto.family().id, "ref.c64.native.4x4");
    assert_eq!(
        (auto.a_layout, auto.b_layout, auto.tile_fmt),
        (
            Layout::Interleaved,
            Layout::Interleaved,
            TileFormat::Interleaved
        )
    );
    assert_eq!((auto.a_per_k, auto.b_per_k, auto.tile_bound), (8, 8, 32));
}

#[test]
fn resolution_returns_typed_selection_errors() {
    assert!(matches!(
        ResolvedGemm::<f64>::resolve::<f64>(&KernelChoice::Id("does.not.exist".into()), 1),
        Err(SelectError::UnknownId { .. })
    ));
    assert!(matches!(
        ResolvedGemm::<f64>::resolve::<f64>(&KernelChoice::Id("ref.c64.native.4x4".into()), 1),
        Err(SelectError::DtypeMismatch { .. })
    ));
}

#[test]
fn zero_effective_width_is_rejected() {
    assert!(matches!(
        ResolvedGemm::<f64>::resolve::<f64>(&KernelChoice::Auto, 0),
        Err(SelectError::Incompatible {
            reason: "zero effective thread width",
            ..
        })
    ));
    let rg = ResolvedGemm::<f64>::resolve::<f64>(&KernelChoice::Auto, 1).unwrap();
    assert!(matches!(
        rg.with_threads(0),
        Err(SelectError::Incompatible {
            reason: "zero effective thread width",
            ..
        })
    ));
}

#[test]
fn effective_width_retarget_preserves_family_and_register_alignment() {
    let rg = ResolvedGemm::<f32>::resolve::<f32>(&KernelChoice::Auto, 8).unwrap();
    let serial = rg.with_threads(1).unwrap();
    assert!(core::ptr::eq(rg.family(), serial.family()));
    assert_eq!(serial.mc % serial.mr, 0);
    assert_eq!(serial.nc % serial.nr, 0);
    assert_eq!(rg.kc, serial.kc);
    assert_eq!(rg.with_threads(8).unwrap().nc, rg.nc);
}
