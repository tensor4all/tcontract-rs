use tcontract_kernel::*;

#[test]
fn every_tc_family_validates_and_ids_are_unique() {
    fn check<T: Families>() {
        let mut ids = std::collections::BTreeSet::new();
        let all = Registry::families::<T>(CpuFeatures::NONE, true);
        assert!(all
            .iter()
            .any(|f| f.origin == tcontract_kernel::Origin::Tensorcontract));
        for f in all {
            f.validate().unwrap_or_else(|e| panic!("{e}"));
            assert!(ids.insert(f.id), "duplicate id {}", f.id);
        }
    }
    check::<f32>();
    check::<f64>();
    check::<C32>();
    check::<C64>();
    let real: std::collections::BTreeSet<_> = list_kernels::<f64>().iter().map(|f| f.id).collect();
    assert!(list_kernels::<C64>().iter().all(|f| !real.contains(f.id)));
}

/// Unqualified Auto resolves to the head of the default menu: a built-in family
/// of the preferred instruction set, in its native complex scheme.
#[test]
fn auto_head_is_the_head_of_the_default_menu() {
    let cpu = CpuFeatures::detect();
    let isa = menu_isa(KernelForce::Auto);
    let head = Registry::families::<f64>(cpu, false)
        .into_iter()
        .find(|f| f.allow_auto)
        .unwrap();
    assert_eq!(head.origin, Origin::Tensorcontract, "{}", head.id);
    assert_eq!(head.isa, isa, "{}", head.id);
    let chead = Registry::families::<C64>(cpu, false)
        .into_iter()
        .find(|f| f.allow_auto)
        .unwrap();
    assert_eq!(
        chead.complex.unwrap().method,
        Method::Native,
        "{}",
        chead.id
    );
}

/// The **compiled** menu is still the menu the dispatch offers, entry for entry:
/// the tuning class (#156) only reorders the menu Auto chooses from, so a
/// `RowBlock` index still names the compiled position it always did.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn the_default_menu_order_is_the_dispatch_menu_order() {
    fn shapes<R: Real>(menu: &[&KernelFamily<R>]) -> Vec<(usize, usize)> {
        menu.iter().map(|f| (f.mr, f.nr)).collect()
    }

    use tcontract_kernel::kernels::x86;
    let isa = menu_isa(KernelForce::Auto);
    let xisa = match isa {
        Isa::Avx512 => x86::Isa::Avx512,
        Isa::Avx2 => x86::Isa::Avx2,
        _ => return, // a pinned scalar host has no vectorised menu to compare
    };
    let configs64 = x86::isa_configs_f64(xisa);
    assert_eq!(
        shapes(builtin_compiled_menu::<f64>(isa, None)),
        (configs64.row_blocks)(false, ComplexMethod::Planar),
        "f64 real compiled menu"
    );
    assert_eq!(
        shapes(builtin_compiled_menu::<C64>(isa, Some(Method::Native))),
        (configs64.row_blocks)(true, ComplexMethod::Planar),
        "c64 native compiled menu"
    );
    let configs32 = x86::isa_configs_f32(xisa);
    assert_eq!(
        shapes(builtin_compiled_menu::<f32>(isa, None)),
        (configs32.row_blocks)(false, ComplexMethod::Planar),
        "f32 real compiled menu"
    );
    assert_eq!(
        shapes(builtin_compiled_menu::<C32>(isa, Some(Method::Native))),
        (configs32.row_blocks)(true, ComplexMethod::Planar),
        "c32 native compiled menu"
    );
    // The tuning class is a process constant and the menus are cached per
    // `(isa, method)`, so a second lookup must answer with the same slices.
    assert!(core::ptr::eq(
        builtin_menu::<f64>(isa, None),
        builtin_menu::<f64>(isa, None)
    ));
    assert!(core::ptr::eq(
        builtin_compiled_menu::<f64>(isa, None),
        builtin_compiled_menu::<f64>(isa, None)
    ));
}

#[test]
fn every_available_family_matches_a_packed_product_oracle() {
    fn check<T: Families>(tol: f64) {
        for f in Registry::families::<T>(CpuFeatures::detect(), false) {
            f.validate().unwrap();
            // The oracle drives the packed-product arm directly, so it covers
            // the scratch-tile families; a Direct family's behaviour is pinned
            // in the contraction driver's `direct.rs` tests and an induced family's
            // packing, arithmetic and write-back in `tests/induced.rs`, with
            // the full driver path covered by the all-family sweeps.
            let Some(u) = f.as_ukr() else {
                continue;
            };
            if f.imp == KernelImpl::Induced {
                continue;
            }
            let (mr, nr, kc) = (f.mr, f.nr, 7);
            let make = |i: usize| {
                T::from_parts(
                    T::Real::from_f64((i * 37 % 19) as f64 / 7.0 - 1.0),
                    T::Real::from_f64((i * 53 % 23) as f64 / 5.0 - 2.0),
                )
            };
            let a: Vec<_> = (0..mr * kc).map(make).collect();
            let b: Vec<_> = (0..nr * kc).map(|i| make(i + 11)).collect();
            let ar: Vec<_> = (0..mr).map(|i| i as i64).collect();
            let br: Vec<_> = (0..nr).map(|i| i as i64).collect();
            let ak: Vec<_> = (0..kc).map(|p| (p * mr) as i64).collect();
            let bk: Vec<_> = (0..kc).map(|p| (p * nr) as i64).collect();
            let abs = scatter::build_block_scatter(&ar, mr);
            let bbs = scatter::build_block_scatter(&br, nr);
            let cols: Vec<_> = (0..nr).map(|j| (j * mr) as i64).collect();
            for (ca, cb) in [(false, false), (true, false), (false, true), (true, true)] {
                let mut pa = vec![T::Real::from_f64(f64::NAN); f.a_per_k * kc];
                let mut pb = vec![T::Real::from_f64(f64::NAN); f.b_per_k * kc];
                let mut tile = vec![T::Real::from_f64(f64::NAN); f.tile_bound];
                let mut out =
                    vec![T::from_parts(T::Real::from_f64(f64::NAN), T::Real::ZERO); mr * nr];
                // SAFETY: validated descriptor extents cover all panels and
                // tile; scatter offsets cover each source/output allocation.
                // Beta zero must not read the null C pointer.
                unsafe {
                    pack::pack_panel::<T>(
                        a.as_ptr(),
                        &ar,
                        &abs,
                        &ak,
                        mr,
                        ca,
                        u.a_pack,
                        pa.as_mut_ptr(),
                    );
                    pack::pack_panel::<T>(
                        b.as_ptr(),
                        &br,
                        &bbs,
                        &bk,
                        nr,
                        cb,
                        u.b_pack,
                        pb.as_mut_ptr(),
                    );
                    (u.func)(kc, pa.as_ptr(), pb.as_ptr(), tile.as_mut_ptr());
                    writeback::writeback::<T>(
                        tile.as_ptr(),
                        u.tile_fmt,
                        mr,
                        nr,
                        mr,
                        nr,
                        T::one(),
                        T::zero(),
                        core::ptr::null(),
                        &ar,
                        &cols,
                        1,
                        false,
                        out.as_mut_ptr(),
                        &ar,
                        &cols,
                        1,
                        false,
                    );
                }
                let mut err = 0.0_f64;
                for j in 0..nr {
                    for i in 0..mr {
                        let mut want = T::zero();
                        for p in 0..kc {
                            let av = a[p * mr + i];
                            let bv = b[p * nr + j];
                            want = want.add((if ca { av.conj() } else { av }).mul(if cb {
                                bv.conj()
                            } else {
                                bv
                            }));
                        }
                        let got = out[j * mr + i];
                        assert!(
                            got.re().to_f64().is_finite() && got.im().to_f64().is_finite(),
                            "{} unwritten lane",
                            f.id
                        );
                        err = err.max(
                            ((got.re().to_f64() - want.re().to_f64()).abs()
                                + (got.im().to_f64() - want.im().to_f64()).abs())
                                / (want.re().to_f64().abs() + want.im().to_f64().abs()).max(1.0),
                        );
                    }
                }
                assert!(
                    err <= tol,
                    "{} conj=({ca},{cb}), relative max error {err}",
                    f.id
                );
            }
        }
    }
    check::<f64>(1e-12);
    check::<C64>(1e-11);
    check::<f32>(3e-5);
    check::<C32>(5e-5);
}

#[test]
fn id_snapshot() {
    let ids: Vec<_> = list_kernels::<f64>().into_iter().map(|k| k.id).collect();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let want = include_str!("snapshots/f64_x86_ids.txt");
    #[cfg(target_arch = "aarch64")]
    let want = include_str!("snapshots/f64_neon_ids.txt");
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    let want = include_str!("snapshots/f64_scalar_ids.txt");
    assert_eq!(
        ids.join("\n"),
        want.trim_end(),
        "update the architecture's snapshot deliberately"
    );
}

#[test]
fn cpu_mask_and_complex_method_metadata() {
    for f in Registry::families::<C64>(CpuFeatures::NONE, true) {
        if f.complex.unwrap().method == Method::ThreeM {
            assert!(!f.allow_auto);
        }
    }
    for f in Registry::families::<f64>(CpuFeatures::NONE, false) {
        assert_eq!(f.required, CpuFeatures::NONE);
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let all = Registry::families::<f64>(CpuFeatures::NONE, true);
        assert!(all.iter().any(|f| f.isa == Isa::Avx512));
        assert!(all.iter().any(|f| f.isa == Isa::Avx2));
    }
}

/// The complete built-in manifest of an x86 target, in Auto order, for all four
/// storage dtypes (CPU-unavailable families included, so the list is the same
/// on every x86-64 machine). Ids, ordering and injectivity are pinned here;
/// any change is a deliberate one.
#[cfg(target_arch = "x86_64")]
#[test]
fn x86_64_manifest_snapshot() {
    fn lines<T: Families>(name: &str, out: &mut Vec<String>) {
        out.extend(
            list_kernels::<T>()
                .into_iter()
                .map(|k| format!("{name} {}", k.id)),
        );
    }
    let mut got = Vec::new();
    lines::<f32>("f32", &mut got);
    lines::<f64>("f64", &mut got);
    lines::<C32>("c32", &mut got);
    lines::<C64>("c64", &mut got);
    assert_eq!(
        got.join("\n"),
        include_str!("snapshots/x86_64_manifest.txt").trim_end(),
        "update the manifest deliberately, with docs/migration-2026-10.md"
    );
    // Injective per dtype.
    for dtype in ["f32", "f64", "c32", "c64"] {
        let ids: Vec<_> = got.iter().filter(|l| l.starts_with(dtype)).collect();
        let set: std::collections::BTreeSet<_> = ids.iter().collect();
        assert_eq!(ids.len(), set.len(), "{dtype}: duplicate ids");
    }
}
