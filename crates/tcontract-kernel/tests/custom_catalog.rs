//! Caller-scoped catalogs: admission, handle identity and resolution.
use tcontract_kernel::*;

const OWN: Origin = Origin::External {
    crate_name: "downstream-kernels",
    license: "Apache-2.0",
};

macro_rules! family {
    ($name:ident, $id:literal, $mr:literal) => {
        static $name: KernelFamily<f64> = KernelFamily {
            id: $id,
            origin: OWN,
            isa: Isa::Portable,
            required: CpuFeatures::NONE,
            imp: KernelImpl::Reference,
            priority: 0,
            complex: None,
            mr: $mr,
            nr: $mr,
            a_per_k: $mr,
            b_per_k: $mr,
            tile_bound: $mr * $mr,
            c_pref: CPref::Any,
            ukr: UkrFn::Tile(portable::real_tile::<f64, $mr, $mr>),
            edge: None,
            b_access: BAccess::Packed,
            c_update: CUpdate::ScratchTile,
            blocks: Blocksizes {
                mc: (64, 64),
                kc: (256, 256),
                nc: (1024, 1024),
            },
            caps: Caps {
                scatter_pack: true,
                conj_a: true,
                conj_b: true,
            },
            opaque: core::ptr::null(),
            inner: None,
            allow_auto: false,
        };
    };
}
family!(K2, "own.f64.2x2", 2);
family!(K4, "own.f64.4x4", 4);
family!(K4_AGAIN, "own.f64.4x4", 4);

static OK: [&KernelFamily<f64>; 2] = [&K2, &K4];
static DUP: [&KernelFamily<f64>; 2] = [&K4, &K4_AGAIN];
static REPEATED: [&KernelFamily<f64>; 2] = [&K4, &K4];
static BAD: KernelFamily<f64> = KernelFamily { mr: 0, ..K4 };
static BAD_LIST: [&KernelFamily<f64>; 1] = [&BAD];
static NEEDS_BOTH: KernelFamily<f64> = KernelFamily {
    id: "own.f64.impossible",
    required: CpuFeatures {
        avx2: true,
        neon: true,
        ..CpuFeatures::NONE
    },
    ..K2
};
static MASKED: [&KernelFamily<f64>; 2] = [&K2, &NEEDS_BOTH];

fn admit(list: &'static [&'static KernelFamily<f64>]) -> Result<KernelCatalog<f64>, SelectError> {
    // SAFETY: the test kernels are the project's portable loops over the
    // declared geometry; static, immutable, thread-safe and non-panicking.
    unsafe { KernelCatalog::<f64>::from_static_families(list) }
}

#[test]
fn admission_lists_handles_with_external_provenance() {
    let catalog = admit(&OK).unwrap();
    assert_eq!(catalog.len(), 2);
    let info = catalog.list();
    assert_eq!(info[0].id, "own.f64.2x2");
    assert_eq!(info[0].origin, OWN);
    assert_eq!(info[0].crate_name, "downstream-kernels");
    assert_eq!(info[0].license, "Apache-2.0");
    let h = catalog.get("own.f64.4x4").unwrap();
    assert_eq!((h.mr(), h.nr(), h.origin()), (4, 4, OWN));
}

#[test]
fn admission_rejects_duplicates_bad_geometry_and_wrong_dtype() {
    assert!(matches!(admit(&DUP), Err(SelectError::DuplicateId { id }) if id == "own.f64.4x4"));
    assert!(matches!(
        admit(&BAD_LIST),
        Err(SelectError::Incompatible {
            reason: "mr is zero",
            ..
        })
    ));
    // The same descriptor listed twice is one family, not an alias.
    assert_eq!(admit(&REPEATED).unwrap().len(), 1);
    // A real family cannot be admitted for complex storage.
    // SAFETY: as above.
    let err = unsafe { KernelCatalog::<C64>::from_static_families(&OK) };
    assert!(matches!(
        err,
        Err(SelectError::DtypeMismatch { dtype: "c64", .. })
    ));
}

#[test]
fn handles_belong_to_their_catalog_and_the_isa_mask_is_enforced() {
    let a = admit(&OK).unwrap();
    let b = admit(&OK).unwrap();
    let from_a = a.get("own.f64.2x2").unwrap();
    assert!(a.admit(&from_a, CpuFeatures::detect()).is_ok());
    assert!(matches!(
        b.admit(&from_a, CpuFeatures::detect()),
        Err(SelectError::ForeignHandle { id }) if id == "own.f64.2x2"
    ));
    let masked = admit(&MASKED).unwrap();
    let impossible = masked.get("own.f64.impossible").unwrap();
    assert!(matches!(
        masked.admit(&impossible, CpuFeatures::detect()),
        Err(SelectError::CpuUnsupported { .. })
    ));
    assert!(matches!(
        ResolvedGemm::<f64>::resolve_handle::<f64>(
            &impossible,
            1,
            PartitionPolicy::default(),
            PartitionOpts::default()
        ),
        Err(SelectError::CpuUnsupported { .. })
    ));
}

#[test]
fn union_reexpresses_handles_and_detects_conflicts() {
    let own = admit(&OK).unwrap();
    let builtin = KernelCatalog::<f64>::builtin();
    let both = own.union(&builtin).unwrap();
    let h = both.get("own.f64.2x2").unwrap();
    assert!(both.contains(&h));
    assert!(!own.contains(&h), "union mints fresh handles");
    assert!(both.get("ref.f64.real.4x4").is_some());
    // The same descriptor in both catalogs is shared, a distinct one is an alias.
    assert_eq!(own.union(&admit(&OK[1..]).unwrap()).unwrap().len(), 2);
    assert!(matches!(
        own.union(&admit(&DUP[1..]).unwrap()),
        Err(SelectError::DuplicateId { .. })
    ));
}

#[test]
fn resolving_a_handle_matches_resolving_the_same_family_by_geometry() {
    let catalog = admit(&OK).unwrap();
    let h = catalog.get("own.f64.2x2").unwrap();
    let rg = ResolvedGemm::<f64>::resolve_handle::<f64>(
        &h,
        3,
        PartitionPolicy::default(),
        PartitionOpts::default(),
    )
    .unwrap();
    assert_eq!(
        (rg.family().id, rg.mr, rg.nr, rg.effective_threads),
        ("own.f64.2x2", 2, 2, 3)
    );
    assert_eq!(rg.mc % 2, 0);
    assert!(matches!(
        ResolvedGemm::<f64>::resolve_handle::<f64>(
            &h,
            0,
            PartitionPolicy::default(),
            PartitionOpts::default()
        ),
        Err(SelectError::Incompatible { .. })
    ));
}
