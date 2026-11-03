use super::*;
use crate::element::C64;
use crate::scatter::build_block_scatter;

fn pack_c(
    data: &[C64],
    vscat: &[i64],
    kscat: &[i64],
    vr: usize,
    conj: bool,
    fmt: PackFormat,
) -> Vec<f64> {
    let vbs = build_block_scatter(vscat, vr);
    let mut out = vec![f64::NAN; panel_len(vscat.len(), vr, kscat.len(), fmt)];
    unsafe {
        pack_panel::<C64>(
            data.as_ptr(),
            vscat,
            &vbs,
            kscat,
            vr,
            conj,
            fmt,
            out.as_mut_ptr(),
        )
    };
    out
}

#[test]
fn packs_real_regular_block_with_zero_fill() {
    // 3x2 block of a 4x2 column-major matrix, vr = 2 -> 2 slivers,
    // the second one half empty.
    let data: Vec<f64> = (0..8).map(|x| x as f64).collect();
    let vscat = vec![0i64, 1, 2];
    let kscat = vec![0i64, 4];
    let vbs = build_block_scatter(&vscat, 2);
    let mut out = vec![-1.0f64; panel_len(3, 2, 2, PackFormat::Real)];
    unsafe {
        pack_panel::<f64>(
            data.as_ptr(),
            &vscat,
            &vbs,
            &kscat,
            2,
            false,
            PackFormat::Real,
            out.as_mut_ptr(),
        )
    };
    // sliver 0: k=0 -> [0,1], k=1 -> [4,5]
    // sliver 1: k=0 -> [2,0], k=1 -> [6,0]
    assert_eq!(out, vec![0.0, 1.0, 4.0, 5.0, 2.0, 0.0, 6.0, 0.0]);
}

#[test]
fn packs_complex_into_planar_planes() {
    let data: Vec<C64> = (0..4)
        .map(|i| C64::new(i as f64, 10.0 + i as f64))
        .collect();
    let out = pack_c(&data, &[0, 1], &[0, 2], 2, false, PackFormat::Planar);
    // k=0: re[0,1] im[10,11]; k=1: re[2,3] im[12,13]
    assert_eq!(out, vec![0.0, 1.0, 10.0, 11.0, 2.0, 3.0, 12.0, 13.0]);
}

#[test]
fn packs_complex_into_one_e_blocks() {
    let data: Vec<C64> = vec![C64::new(1.0, 2.0), C64::new(3.0, 4.0)];
    let out = pack_c(&data, &[0, 1], &[0], 2, false, PackFormat::OneE);
    assert_eq!(
        out,
        vec![
            1.0, 2.0, 3.0, 4.0, // real step 0: [re0, im0, re1, im1]
            -2.0, 1.0, -4.0, 3.0, // real step 1: [-im0, re0, -im1, re1]
        ]
    );
    assert_eq!(out.len(), 4 * 2, "1m stores four reals per complex element");
}

#[test]
fn packs_complex_into_three_m_planes() {
    let data: Vec<C64> = vec![C64::new(1.0, 2.0), C64::new(3.0, 4.0)];
    let out = pack_c(&data, &[0, 1], &[0], 2, false, PackFormat::ThreeM);
    // [re | im | re+im]
    assert_eq!(out, vec![1.0, 3.0, 2.0, 4.0, 3.0, 7.0]);
}

#[test]
fn conjugation_negates_the_imaginary_part_in_every_format() {
    let data: Vec<C64> = vec![C64::new(1.0, 2.0), C64::new(3.0, 4.0)];

    let planar = pack_c(&data, &[0, 1], &[0], 2, true, PackFormat::Planar);
    assert_eq!(planar, vec![1.0, 3.0, -2.0, -4.0]);

    let threem = pack_c(&data, &[0, 1], &[0], 2, true, PackFormat::ThreeM);
    assert_eq!(threem, vec![1.0, 3.0, -2.0, -4.0, -1.0, -1.0]);

    let onee = pack_c(&data, &[0, 1], &[0], 2, true, PackFormat::OneE);
    assert_eq!(onee, vec![1.0, -2.0, 3.0, -4.0, 2.0, 1.0, 4.0, 3.0]);
}

#[test]
fn selected_one_r_packs_like_planar() {
    let data = [C64::new(1.0, 2.0), C64::new(3.0, 4.0)];
    let vscat = [0, 1];
    let vbs = build_block_scatter(&vscat, 2);
    let mut out = [0.0; 4];
    unsafe {
        (pack_fn::<C64>(Layout::OneR, false))(
            data.as_ptr(),
            &vscat,
            &vbs,
            &[0],
            2,
            false,
            out.as_mut_ptr(),
        );
    }
    assert_eq!(out, [1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn packed_size_ratios_match_the_method_costs() {
    let data: Vec<C64> = vec![C64::new(1.0, 2.0), C64::new(3.0, 4.0)];
    let n = |fmt| pack_c(&data, &[0, 1], &[0, 0], 2, false, fmt).len();
    assert_eq!(n(PackFormat::Planar), 8);
    assert_eq!(n(PackFormat::ThreeM), 12);
    assert_eq!(n(PackFormat::OneE), 16);
}

#[test]
fn irregular_block_uses_gather_path() {
    let data: Vec<f64> = (0..6).map(|x| x as f64).collect();
    let vscat = vec![0i64, 3, 1]; // not an arithmetic progression
    let kscat = vec![0i64];
    let vbs = build_block_scatter(&vscat, 3);
    assert_eq!(vbs, vec![IRREGULAR]);
    let mut out = vec![0.0f64; panel_len(3, 3, 1, PackFormat::Real)];
    unsafe {
        pack_panel::<f64>(
            data.as_ptr(),
            &vscat,
            &vbs,
            &kscat,
            3,
            false,
            PackFormat::Real,
            out.as_mut_ptr(),
        )
    };
    assert_eq!(out, vec![0.0, 3.0, 1.0]);
}

/// The packing definition, element by element.
fn reference(
    data: &[C64],
    vscat: &[i64],
    kscat: &[i64],
    vr: usize,
    conj: bool,
    fmt: PackFormat,
) -> Vec<f64> {
    let per = fmt.reals_per_element();
    let nsliv = vscat.len().div_ceil(vr);
    let mut out = vec![f64::NAN; panel_len(vscat.len(), vr, kscat.len(), fmt)];
    for s in 0..nsliv {
        for (p, &k) in kscat.iter().enumerate() {
            let o = s * vr * kscat.len() * per + p * vr * per;
            for t in 0..vr {
                let v = s * vr + t;
                let z = if v < vscat.len() {
                    data[(vscat[v] + k) as usize]
                } else {
                    C64::new(0., 0.)
                };
                let (re, im) = (z.re, if conj { -z.im } else { z.im });
                match fmt {
                    PackFormat::Real => out[o + t] = re,
                    PackFormat::Interleaved => {
                        out[o + 2 * t] = re;
                        out[o + 2 * t + 1] = im;
                    }
                    PackFormat::Planar => {
                        out[o + t] = re;
                        out[o + vr + t] = im;
                    }
                    PackFormat::ThreeM => {
                        out[o + t] = re;
                        out[o + vr + t] = im;
                        out[o + 2 * vr + t] = re + im;
                    }
                    PackFormat::OneE => {
                        out[o + 2 * t] = re;
                        out[o + 2 * t + 1] = im;
                        out[o + 2 * vr + 2 * t] = -im;
                        out[o + 2 * vr + 2 * t + 1] = re;
                    }
                }
            }
        }
    }
    out
}

#[test]
fn every_variant_matches_the_definition_bitwise() {
    let cpu = crate::CpuFeatures::detect();
    let simds: &[bool] = if cfg!(target_arch = "x86_64") && cpu.avx2 && cpu.fma {
        &[false, true]
    } else {
        &[false]
    };
    let data: Vec<C64> = (0..400)
        .map(|i| C64::new(i as f64 * 0.37 - 5.0, 3.0 - i as f64 * 0.11))
        .collect();
    let layouts = [
        (Layout::Interleaved, PackFormat::Interleaved),
        (Layout::Planar, PackFormat::Planar),
        (Layout::OneR, PackFormat::Planar),
        (Layout::OneE, PackFormat::OneE),
        (Layout::ThreeM, PackFormat::ThreeM),
    ];
    // Unit stride, constant stride, irregular, and partial slivers.
    let vscats: [Vec<i64>; 4] = [
        (0..11).collect(),
        (0..11).map(|i| 3 * i).collect(),
        vec![0, 5, 1, 9, 2, 7, 3, 4, 6, 8, 10],
        (0..5).collect(),
    ];
    let kscat: Vec<i64> = (0..7).map(|k| 20 * k).collect();
    for &simd in simds {
        for (layout, fmt) in layouts {
            for vscat in &vscats {
                for vr in [2usize, 4] {
                    for conj in [false, true] {
                        let vbs = build_block_scatter(vscat, vr);
                        let mut out = vec![f64::NAN; panel_len(vscat.len(), vr, kscat.len(), fmt)];
                        // SAFETY: scatters stay inside `data`, `out` has the panel size.
                        unsafe {
                            pack_fn::<C64>(layout, simd)(
                                data.as_ptr(),
                                vscat,
                                &vbs,
                                &kscat,
                                vr,
                                conj,
                                out.as_mut_ptr(),
                            )
                        };
                        let want = reference(&data, vscat, &kscat, vr, conj, fmt);
                        let b = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                        assert_eq!(
                            b(&out),
                            b(&want),
                            "{layout:?} simd={simd} vr={vr} conj={conj} {vscat:?}"
                        );
                    }
                }
            }
        }
    }
    // Real storage, both variants.
    let real: Vec<f64> = (0..200).map(|i| i as f64 * 0.5).collect();
    for &simd in simds {
        for vscat in &vscats {
            let vbs = build_block_scatter(vscat, 4);
            let mut a = vec![f64::NAN; panel_len(vscat.len(), 4, 3, PackFormat::Real)];
            let mut b = a.clone();
            // SAFETY: as above.
            unsafe {
                pack_fn::<f64>(Layout::Real, simd)(
                    real.as_ptr(),
                    vscat,
                    &vbs,
                    &[0, 17, 40],
                    4,
                    true,
                    a.as_mut_ptr(),
                );
                pack_panel::<f64>(
                    real.as_ptr(),
                    vscat,
                    &vbs,
                    &[0, 17, 40],
                    4,
                    false,
                    PackFormat::Real,
                    b.as_mut_ptr(),
                );
            }
            assert_eq!(
                a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
        }
    }
}
