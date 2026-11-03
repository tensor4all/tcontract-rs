use super::*;
use crate::C64;

fn simd_available() -> bool {
    let cpu = crate::CpuFeatures::detect();
    cfg!(target_arch = "x86_64") && cpu.avx2 && cpu.fma
}

/// The definition, element by element, with no mode dispatch.
#[allow(clippy::too_many_arguments)]
unsafe fn reference<T: Element>(
    ab: *const T::Real,
    fmt: TileFormat,
    (mr, nr, mrem, nrem): (usize, usize, usize, usize),
    (alpha, beta): (T, T),
    c: (*const T, &[i64], &[i64], bool),
    d: (*mut T, &[i64], &[i64], bool),
) {
    for j in 0..nrem {
        for i in 0..mrem {
            let (re, im) = tile_value::<T::Real>(ab, fmt, mr, nr, i, j);
            let mut v = T::from_parts(re, im).mul(alpha);
            if beta != T::zero() {
                let cv = *c.0.offset((c.1[i] + c.2[j]) as isize);
                let cv = if c.3 { cv.conj() } else { cv };
                v = v.add(cv.mul(beta));
            }
            if d.3 {
                v = v.conj();
            }
            *d.0.offset((d.1[i] + d.2[j]) as isize) = v;
        }
    }
}

/// Bitwise on every target.
fn agree<T: Element>(got: &[T], want: &[T], ctx: &str) {
    let bits = |v: &[T]| {
        v.iter()
            .map(|x| (x.re().to_f64().to_bits(), x.im().to_f64().to_bits()))
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(got), bits(want), "{ctx}");
}

fn check<T: Element>(fmts: &[TileFormat], alphas: &[T], betas: &[T], conjs: &[bool]) {
    let ab: Vec<T::Real> = (0..24)
        .map(|i| T::Real::from_f64(i as f64 / 7. + 0.3))
        .collect();
    // Sixteen elements: the largest scatter offset is 2 * 3 + 6 = 12.
    let c: Vec<T> = (0..16)
        .map(|i| T::from_parts(T::Real::from_f64(3. + i as f64), T::Real::from_f64(-2.)))
        .collect();
    let simds: &[bool] = if simd_available() {
        &[false, true]
    } else {
        &[false]
    };
    for &fmt in fmts {
        for &gather in &[false, true] {
            for &simd in simds {
                for stride in [1, 2, IRREGULAR] {
                    // A three-row live tile in a 3x2 register block, and a
                    // partial one, so edge extents are covered too.
                    let step = if stride == IRREGULAR { 3 } else { stride };
                    let rows = [0, step, 2 * step];
                    for &(mrem, nrem) in &[(3, 2), (2, 1)] {
                        for &alpha in alphas {
                            for &beta in betas {
                                for &conj_c in conjs {
                                    for &conj_d in conjs {
                                        let mut want = [T::zero(); 14];
                                        let mut got = want;
                                        let mut via_pub = want;
                                        let cp = if beta == T::zero() {
                                            core::ptr::null()
                                        } else {
                                            c.as_ptr()
                                        };
                                        let cols = [1, 6];
                                        // SAFETY: a maximum-size initialized tile
                                        // (3x2, four planes), live scatters inside
                                        // the 14-element outputs, disjoint arrays;
                                        // null C only when beta is zero.
                                        unsafe {
                                            reference::<T>(
                                                ab.as_ptr(),
                                                fmt,
                                                (3, 2, mrem, nrem),
                                                (alpha, beta),
                                                (cp, &rows, &cols, conj_c),
                                                (want.as_mut_ptr(), &rows, &cols, conj_d),
                                            );
                                            emit_fn::<T>(fmt, gather, simd)(
                                                ab.as_ptr(),
                                                3,
                                                2,
                                                mrem,
                                                nrem,
                                                alpha,
                                                beta,
                                                cp,
                                                &rows,
                                                &cols,
                                                stride,
                                                conj_c,
                                                got.as_mut_ptr(),
                                                &rows,
                                                &cols,
                                                stride,
                                                conj_d,
                                            );
                                            writeback::<T>(
                                                ab.as_ptr(),
                                                fmt,
                                                3,
                                                2,
                                                mrem,
                                                nrem,
                                                alpha,
                                                beta,
                                                cp,
                                                &rows,
                                                &cols,
                                                stride,
                                                conj_c,
                                                via_pub.as_mut_ptr(),
                                                &rows,
                                                &cols,
                                                stride,
                                                conj_d,
                                            );
                                        }
                                        let ctx = format!("{fmt:?} gather={gather} simd={simd} stride={stride} live={mrem}x{nrem} beta0={} conj_c={conj_c} conj_d={conj_d}", beta == T::zero());
                                        agree(&got, &want, &ctx);
                                        agree(&via_pub, &want, &format!("public {ctx}"));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn every_complex_variant_matches_the_scalar_definition_bitwise() {
    check::<C64>(
        &[
            TileFormat::Planar,
            TileFormat::Interleaved,
            TileFormat::OneM,
            TileFormat::ThreeM,
            TileFormat::FourM,
        ],
        &[C64::from_parts(1., 0.), C64::from_parts(1.5, 0.3)],
        &[
            C64::zero(),
            C64::from_parts(0.4, -0.2),
            C64::from_parts(1., 0.),
        ],
        &[false, true],
    );
}

#[test]
fn every_real_variant_matches_the_scalar_definition_bitwise() {
    // The conjugation flags of real storage are the identity either way.
    check::<f64>(
        &[TileFormat::Real],
        &[1., -1.5],
        &[0., 0.4, 1.],
        &[false, true],
    );
    check::<f32>(
        &[TileFormat::Real],
        &[1., -1.5],
        &[0., 0.4, 1.],
        &[false, true],
    );
}

#[test]
fn scale_only_matches_the_definition_for_every_flag_combination() {
    let c: Vec<C64> = (0..9)
        .map(|i| C64::from_parts(i as f64 + 0.5, 2.0 - i as f64))
        .collect();
    let (rows, cols) = ([0i64, 1, 2], [0i64, 3, 6]);
    for beta in [C64::zero(), C64::from_parts(0.4, -0.2)] {
        for conj_c in [false, true] {
            for conj_d in [false, true] {
                let mut got = [C64::from_parts(9., 9.); 9];
                // SAFETY: nine-element arrays cover all scatter offsets; C is
                // valid even when beta is zero.
                unsafe {
                    scale_only::<C64>(
                        beta,
                        c.as_ptr(),
                        &rows,
                        &cols,
                        conj_c,
                        got.as_mut_ptr(),
                        &rows,
                        &cols,
                        conj_d,
                    )
                };
                for (k, g) in got.iter().enumerate() {
                    let mut v = C64::zero();
                    if beta != C64::zero() {
                        let cv = if conj_c { c[k].conj() } else { c[k] };
                        v = cv.mul(beta);
                    }
                    if conj_d {
                        v = v.conj();
                    }
                    assert_eq!(
                        *g, v,
                        "beta {beta:?} conj_c {conj_c} conj_d {conj_d} at {k}"
                    );
                }
            }
        }
    }
}
