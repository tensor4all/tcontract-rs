use tcontract_kernel::*;

const fn fam(mr: usize, nr: usize) -> KernelFamily<f64> {
    KernelFamily {
        id: "test.f64.4x4",
        origin: Origin::Portable,
        isa: Isa::Portable,
        required: CpuFeatures::NONE,
        imp: KernelImpl::Reference,
        priority: 0,
        complex: None,
        mr,
        nr,
        a_per_k: mr,
        b_per_k: nr,
        tile_bound: mr * nr,
        c_pref: CPref::Any,
        ukr: UkrFn::Tile(portable::real_tile::<f64, 4, 4>),
        edge: None,
        b_access: BAccess::Packed,
        c_update: CUpdate::ScratchTile,
        blocks: Blocksizes {
            mc: (mr * 8, mr * 16),
            kc: (256, 512),
            nc: (nr * 64, nr * 128),
        },
        caps: Caps {
            scatter_pack: true,
            conj_a: true,
            conj_b: true,
        },
        opaque: core::ptr::null(),
        inner: None,
        allow_auto: true,
    }
}

#[test]
fn valid_family_passes() {
    assert_eq!(fam(4, 4).validate(), Ok(()));
}

#[test]
fn zero_mr_rejected() {
    assert!(fam(0, 4).validate().is_err());
}

#[test]
fn mc_not_multiple_of_mr_rejected() {
    let mut f = fam(4, 4);
    f.blocks.mc = (6, 16);
    assert_eq!(
        f.validate().unwrap_err().reason,
        "mc default is not a multiple of mr"
    );
}

#[test]
fn tile_bound_too_small_rejected() {
    let mut f = fam(4, 4);
    f.tile_bound = 15;
    assert!(f.validate().is_err());
}

#[test]
fn direct_b_requires_direct_c() {
    let mut f = fam(4, 4);
    f.b_access = BAccess::Direct {
        unit_stride: Axis::Col,
    };
    assert!(f.validate().is_err());
}

#[test]
fn cpu_contains() {
    let have = CpuFeatures {
        avx2: true,
        fma: true,
        ..CpuFeatures::NONE
    };
    assert!(have.contains(CpuFeatures {
        avx2: true,
        ..CpuFeatures::NONE
    }));
    assert!(!have.contains(CpuFeatures {
        avx512f: true,
        ..CpuFeatures::NONE
    }));
    assert_eq!(
        have.missing(CpuFeatures {
            avx2: true,
            avx512f: true,
            ..CpuFeatures::NONE
        }),
        CpuFeatures {
            avx512f: true,
            ..CpuFeatures::NONE
        }
    );
}

#[test]
fn invalid_geometry_and_complex_scheme_are_rejected() {
    let mut f = fam(4, 4);
    f.blocks.kc = (0, 512);
    assert!(f.validate().is_err());
    f = fam(4, 4);
    f.mr = usize::MAX;
    f.a_per_k = usize::MAX;
    f.blocks.mc = (usize::MAX, usize::MAX);
    assert!(f.validate().is_err(), "overflow must not panic");
    f = fam(4, 4);
    f.complex = Some(ComplexScheme {
        method: Method::Native,
        a: Layout::Real,
        b: Layout::Real,
        tile: TileFormat::Real,
    });
    assert!(f.validate().is_err());
}

#[test]
fn legacy_bridge_preserves_pack_and_tile_contract() {
    let f = fam(4, 4);
    let u = f.as_ukr().unwrap();
    assert_eq!((u.mr, u.nr, u.a_per_k, u.b_per_k, u.tile), (4, 4, 4, 4, 16));
    assert_eq!(
        (u.a_pack, u.b_pack, u.tile_fmt),
        (PackFormat::Real, PackFormat::Real, TileFormat::Real)
    );
}

#[test]
fn onem_logical_tile_may_have_odd_dimensions() {
    // Real inner 6x5 gives a logical complex 3x5 tile. Only the expanded
    // real axis has an even extent; rejecting odd logical MR/NR breaks it.
    let mut f = fam(3, 5);
    f.complex = Some(ComplexScheme {
        method: Method::OneM,
        a: Layout::OneE,
        b: Layout::OneR,
        tile: TileFormat::OneM,
    });
    f.a_per_k = 12;
    f.b_per_k = 10;
    f.tile_bound = 30;
    f.validate().unwrap();
    let u = f.as_ukr().unwrap();
    assert_eq!((u.a_pack, u.b_pack), (PackFormat::OneE, PackFormat::Planar));
    f.a_per_k = 6;
    assert_eq!(
        f.validate().unwrap_err().reason,
        "packed k-step shorter than the tile"
    );
}

#[test]
fn registry_registration_cpu_mask_and_forced_errors() {
    static BASE: KernelFamily<f64> = KernelFamily {
        id: "test.base.f64",
        ..fam(4, 4)
    };
    static FAST: KernelFamily<f64> = KernelFamily {
        id: "test.avx2.f64",
        priority: 200,
        isa: Isa::Avx2,
        required: CpuFeatures {
            avx2: true,
            fma: true,
            ..CpuFeatures::NONE
        },
        ..fam(4, 4)
    };
    static COMPLEX: KernelFamily<f64> = KernelFamily {
        id: "test.base.c64",
        a_per_k: 8,
        b_per_k: 8,
        tile_bound: 32,
        complex: Some(ComplexScheme {
            method: Method::Native,
            a: Layout::Interleaved,
            b: Layout::Interleaved,
            tile: TileFormat::Interleaved,
        }),
        ukr: UkrFn::Tile(portable::cplx_tile::<f64, 4, 4>),
        ..fam(4, 4)
    };
    static LIST: [&KernelFamily<f64>; 3] = [&BASE, &FAST, &COMPLEX];
    fn list() -> &'static [&'static KernelFamily<f64>] {
        &LIST
    }
    // SAFETY: immutable 4x4 descriptors use the matching portable real and
    // interleaved kernels; FAST deliberately requires a stricter CPU subset.
    unsafe {
        register::<f64>(list);
        register::<f64>(list);
    }
    let ids: Vec<_> = Registry::families::<f64>(CpuFeatures::NONE, true)
        .iter()
        .filter(|f| f.id.starts_with("test."))
        .map(|f| f.id)
        .collect();
    assert_eq!(ids, [FAST.id, BASE.id]);
    let mut copied = <f64 as RealSlot>::providers();
    copied.clear(); // A mutable copy must not change the registered manifest.
    let available = Registry::families::<f64>(CpuFeatures::NONE, false);
    assert!(available.iter().any(|f| f.id == BASE.id));
    assert!(!available.iter().any(|f| f.id == FAST.id));
    assert!(core::ptr::eq(
        Registry::select::<f64>(BASE.id, CpuFeatures::NONE).unwrap(),
        &BASE
    ));
    assert!(matches!(
        Registry::select::<f64>(FAST.id, CpuFeatures::NONE),
        Err(SelectError::CpuUnsupported {
            missing: CpuFeatures {
                avx2: true,
                fma: true,
                ..
            },
            ..
        })
    ));
    assert!(matches!(
        Registry::select::<f32>(BASE.id, CpuFeatures::NONE),
        Err(SelectError::DtypeMismatch { dtype: "f32", .. })
    ));
    assert!(matches!(
        Registry::select::<f64>(COMPLEX.id, CpuFeatures::NONE),
        Err(SelectError::DtypeMismatch { .. })
    ));
    assert!(matches!(
        Registry::select::<f64>("no-such-id", CpuFeatures::NONE),
        Err(SelectError::UnknownId { .. })
    ));
    let info = list_kernels::<num_complex::Complex<f64>>()
        .into_iter()
        .find(|f| f.id == COMPLEX.id)
        .unwrap();
    assert_eq!(
        (info.dtype, info.license, info.crate_name),
        ("c64", "MIT OR Apache-2.0", "tcontract-kernel")
    );
    static ALIAS1: KernelFamily<f64> = KernelFamily {
        id: "alias.f64",
        ..fam(4, 4)
    };
    static ALIAS2: KernelFamily<f64> = KernelFamily {
        id: "alias.f64",
        priority: 1,
        ..fam(4, 4)
    };
    static ALIASES: [&KernelFamily<f64>; 2] = [&ALIAS1, &ALIAS2];
    fn aliases() -> &'static [&'static KernelFamily<f64>] {
        &ALIASES
    }
    // SAFETY: both immutable aliases use matching 4x4 portable kernels;
    // id ambiguity is a typed selection error, not an ABI violation.
    unsafe {
        register::<f64>(aliases);
    }
    assert_eq!(
        Registry::families::<f64>(CpuFeatures::NONE, true)
            .iter()
            .filter(|f| f.id == "alias.f64")
            .count(),
        1
    );
    assert!(matches!(
        Registry::select::<f64>("alias.f64", CpuFeatures::NONE),
        Err(SelectError::Incompatible {
            reason: "duplicate kernel id",
            ..
        })
    ));
}
