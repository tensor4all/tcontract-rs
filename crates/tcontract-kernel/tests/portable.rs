use num_complex::Complex;
use tcontract_kernel::*;

#[test]
fn portable_families_validate_and_have_ids() {
    let real = Registry::families::<f64>(CpuFeatures::NONE, false);
    assert!(real.iter().any(|f| f.id == "ref.f64.real.4x4"));
    for f in real {
        f.validate().unwrap();
    }
    let complex = Registry::families::<Complex<f64>>(CpuFeatures::NONE, false);
    assert!(complex.iter().any(|f| f.id == "ref.c64.native.4x4"));
    for f in complex {
        f.validate().unwrap();
    }
    assert!(list_kernels::<f32>()
        .iter()
        .any(|f| f.id == "ref.f32.real.4x4"));
    assert!(list_kernels::<Complex<f32>>()
        .iter()
        .any(|f| f.id == "ref.c32.native.4x4"));
}

#[test]
fn portable_native_complex_tile_matches_naive() {
    let f = Registry::select::<Complex<f64>>("ref.c64.native.4x4", CpuFeatures::NONE).unwrap();
    let UkrFn::Tile(ukr) = f.ukr else { panic!() };
    for kc in [0, 1, 7] {
        let a: Vec<f64> = (0..kc * 8)
            .map(|x| ((x * 37 % 19) as f64 - 9.0) / 7.0)
            .collect();
        let b: Vec<f64> = (0..kc * 8)
            .map(|x| ((x * 53 % 23) as f64 - 11.0) / 5.0)
            .collect();
        let mut tile = vec![f64::NAN; f.tile_bound];
        // SAFETY: panels and tile cover the validated descriptor's extents.
        unsafe { ukr(kc, a.as_ptr(), b.as_ptr(), tile.as_mut_ptr()) };
        let mut err = 0.0_f64;
        for j in 0..4 {
            for i in 0..4 {
                let mut w = Complex::new(0.0, 0.0);
                for p in 0..kc {
                    w += Complex::new(a[p * 8 + 2 * i], a[p * 8 + 2 * i + 1])
                        * Complex::new(b[p * 8 + 2 * j], b[p * 8 + 2 * j + 1]);
                }
                let g = Complex::new(tile[2 * (j * 4 + i)], tile[2 * (j * 4 + i) + 1]);
                assert!(
                    g.re.is_finite() && g.im.is_finite(),
                    "tile lane was not overwritten"
                );
                err = err.max((g - w).norm() / w.norm().max(1.0));
            }
        }
        assert!(err <= 1e-12, "relative max error {err}");
    }
}

#[test]
fn interleaved_and_four_plane_writeback_match_complex_values() {
    let planes: Vec<f64> = (0..24).map(|x| x as f64 / 3.0 - 2.0).collect();
    let want: Vec<_> = (0..6)
        .map(|q| Complex::new(planes[q] - planes[6 + q], planes[12 + q] + planes[18 + q]))
        .collect();
    let interleaved: Vec<_> = want.iter().flat_map(|z| [z.re, z.im]).collect();
    for (fmt, tile) in [
        (TileFormat::FourM, &planes),
        (TileFormat::Interleaved, &interleaved),
    ] {
        for (m, n) in [(2, 3), (1, 2)] {
            let mut out = vec![Complex::new(f64::NAN, f64::NAN); 6];
            let rows = [0, 1];
            let cols = [0, 2, 4];
            let alpha = Complex::new(1.5, -0.5);
            // SAFETY: tile covers 2x3; scatter offsets address out. Beta is
            // zero, so the null C pointer must not be read.
            unsafe {
                writeback::writeback::<Complex<f64>>(
                    tile.as_ptr(),
                    fmt,
                    2,
                    3,
                    m,
                    n,
                    alpha,
                    Complex::new(0.0, 0.0),
                    core::ptr::null(),
                    &rows[..m],
                    &cols[..n],
                    1,
                    false,
                    out.as_mut_ptr(),
                    &rows[..m],
                    &cols[..n],
                    1,
                    false,
                )
            };
            for j in 0..n {
                for i in 0..m {
                    let q = j * 2 + i;
                    assert!(
                        (out[q] - alpha * want[q]).norm() < 1e-12,
                        "{fmt:?}: {}",
                        out[q]
                    );
                }
            }
        }
    }
}

#[test]
fn interleaved_pack_roundtrip() {
    let src: Vec<Complex<f64>> = (0..6)
        .map(|x| Complex::new(x as f64, -(x as f64)))
        .collect();
    let vscat = [0i64, 1, 2];
    let kscat = [0i64, 3];
    let vbs = scatter::build_block_scatter(&vscat, 4);
    for conj in [false, true] {
        let mut out = vec![f64::NAN; pack::panel_len(3, 4, 2, PackFormat::Interleaved)];
        // SAFETY: scatter addresses src and the panel length covers both k-steps.
        unsafe {
            pack::pack_panel::<Complex<f64>>(
                src.as_ptr(),
                &vscat,
                &vbs,
                &kscat,
                4,
                conj,
                PackFormat::Interleaved,
                out.as_mut_ptr(),
            )
        };
        for p in 0..2 {
            for i in 0..4 {
                let want = if i < 3 {
                    src[p * 3 + i]
                } else {
                    Complex::new(0.0, 0.0)
                };
                assert_eq!(out[p * 8 + 2 * i], want.re);
                assert_eq!(
                    out[p * 8 + 2 * i + 1],
                    if conj { -want.im } else { want.im }
                );
            }
        }
    }
}
