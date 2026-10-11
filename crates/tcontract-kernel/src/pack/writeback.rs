//! Accumulator write-back: `D = op_D(alpha * AB + beta * op_C(C))`.
//!
//! The micro-kernel leaves a tile in a small stack buffer, in whichever
//! [`TileFormat`] its complex method produces. Recombining that into
//! interleaved complex happens here, once per output element per `KC`-block.
//!
//! That recombination is the only per-output cost the induced methods add over
//! a hypothetical native interleaved kernel, and it differs between them:
//!
//! | tile format | work per output element |
//! |---|---|
//! | [`TileFormat::Planar`] | two loads (the interleave itself) |
//! | [`TileFormat::OneM`] | two loads, already adjacent in memory |
//! | [`TileFormat::ThreeM`] | three loads and three adds |
//!
//! Relative to the tile's own compute this is `~c / (4 * min(K, KC))` with `c`
//! the ops per element — negligible for `K >= KC`, rising like `1/(8K)` for
//! small `K`, and material only at `K` in the low single digits.
//!
//! Keeping write-back out of the kernel means one kernel serves the regular
//! fast path, the gather path, and every edge block.
//!
//! # Why this file has a fast path at all
//!
//! On compute-bound shapes the above is exact and write-back is noise. On the
//! low-arithmetic-intensity corpus cases it is not: with `k` in the tens, a
//! micro-tile's write-back costs a constant per output element against only
//! `~4k` flops of kernel work, and `perf` puts more than half of the run time
//! in this file. So the row offsets are taken from the *block* scatter when the
//! block is regular, exactly as [`crate::pack`] does, which removes the
//! scatter-table load from the innermost loop and lets the plane recombination
//! vectorise.

use crate::element::{Element, Real};
use crate::scatter::IRREGULAR;
use crate::TileFormat;

/// Read complex element `(i, j)` out of an accumulator tile.
///
/// # Safety
///
/// * `ab` must point at an accumulator tile written by a micro-kernel whose
///   [`TileFormat`] is exactly `fmt`, with the register block `(mr, nr)` that
///   kernel was selected with. The three formats disagree about both the
///   element count and the index arithmetic — `Real` reads `mr * nr` values,
///   `Planar` and `OneM` read `2 * mr * nr`, `ThreeM` reads `3 * mr * nr` — so a
///   mismatched `fmt` reads out of bounds rather than merely reading the wrong
///   value.
/// * `i < mr` and `j < nr`.
/// * The tile must be fully initialised. Kernels overwrite rather than
///   accumulate into it (see `tensorcontract::kernel`), so this holds after any kernel
///   call and does *not* hold for a freshly allocated `tensorcontract::buffer::Panel`.
#[inline(always)]
unsafe fn tile_value<R: Real>(
    ab: *const R,
    fmt: TileFormat,
    mr: usize,
    nr: usize,
    i: usize,
    j: usize,
) -> (R, R) {
    match fmt {
        TileFormat::Real => (*ab.add(j * mr + i), R::ZERO),
        TileFormat::Planar => (*ab.add(j * mr + i), *ab.add(mr * nr + j * mr + i)),
        TileFormat::Interleaved => {
            // INVARIANT: tile_value's contract covers 2*mr*nr initialized reals.
            let off = 2 * (j * mr + i);
            (*ab.add(off), *ab.add(off + 1))
        }
        TileFormat::FourM => {
            // INVARIANT: tile_value's contract covers four initialized planes.
            let plane = mr * nr;
            let off = j * mr + i;
            (
                *ab.add(off) - *ab.add(plane + off),
                *ab.add(2 * plane + off) + *ab.add(3 * plane + off),
            )
        }
        // One real `2*mr x nr` tile: row 2i is Re, row 2i+1 is Im.
        TileFormat::OneM => {
            let base = j * (2 * mr) + 2 * i;
            (*ab.add(base), *ab.add(base + 1))
        }
        // Karatsuba recombination: Cr = M1 - M2, Ci = M3 - M1 - M2.
        TileFormat::ThreeM => {
            let plane = mr * nr;
            let off = j * mr + i;
            let m1 = *ab.add(off);
            let m2 = *ab.add(plane + off);
            let m3 = *ab.add(2 * plane + off);
            (m1 - m2, m3 - m1 - m2)
        }
    }
}

/// Write one micro-tile back to `C`/`D`.
///
/// `c_r`/`c_c` and `d_r`/`d_c` are the scatter offsets of the tile's rows and
/// columns, each of length `mrem`/`nrem`. `c_rs`/`d_rs` are the corresponding
/// *block* scatter entries for the row runs — the common difference of `c_r` /
/// `d_r`, or [`IRREGULAR`]. They are redundant with `c_r`/`d_r` and only ever
/// select a faster loop; passing [`IRREGULAR`] for both is always correct.
///
/// To accumulate a later `KC` block, callers pass `beta = 1`, `c_base =
/// d_base`, `c_r = d_r`, `c_c = d_c` and `conj_c = conj_d`. Because
/// conjugation is additive and involutive, that reproduces
/// `op_D(alpha * sum_p AB_p + beta * op_C(C))` exactly.
///
/// # Safety
/// `ab` must hold a tile of the size implied by `fmt`, `mr` and `nr`; the
/// scatter offsets must be in bounds for their tensors.
#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
pub unsafe fn writeback<T: Element>(
    ab: *const T::Real,
    fmt: TileFormat,
    mr: usize,
    nr: usize,
    mrem: usize,
    nrem: usize,
    alpha: T,
    beta: T,
    c_base: *const T,
    c_r: &[i64],
    c_c: &[i64],
    c_rs: i64,
    conj_c: bool,
    d_base: *mut T,
    d_r: &[i64],
    d_c: &[i64],
    d_rs: i64,
    conj_d: bool,
) {
    // SAFETY: forwarded unchanged, with the format-specialized fast paths enabled;
    // a planner that wants the general scatter loop binds `emit_fn(fmt, true)`.
    unsafe {
        writeback_mode::<T>(
            ab, fmt, mr, nr, mrem, nrem, alpha, beta, c_base, c_r, c_c, c_rs, conj_c, d_base, d_r,
            d_c, d_rs, conj_d, false,
        )
    }
}

/// The write-back for one tile format. `gather` binds the general scatter loop
/// (a planning-time policy); `simd` binds the variant compiled with `avx2,fma`
/// enabled (x86_64 only; ignored elsewhere), decided once by the caller from
/// the CPU check the family selection already made. All variants run the same
/// source and are bitwise equal.
pub(crate) fn emit_fn<T: Element>(fmt: TileFormat, gather: bool, simd: bool) -> crate::EmitFn<T> {
    macro_rules! pick {
        ($f:literal) => {{
            #[cfg(target_arch = "x86_64")]
            if simd {
                return if gather {
                    emit_fmt_avx2::<T, $f, true>
                } else {
                    emit_fmt_avx2::<T, $f, false>
                };
            }
            let _ = simd;
            if gather {
                emit_fmt::<T, $f, true>
            } else {
                emit_fmt::<T, $f, false>
            }
        }};
    }
    match fmt {
        TileFormat::Real => pick!(0),
        TileFormat::Planar => pick!(1),
        TileFormat::OneM => pick!(2),
        TileFormat::ThreeM => pick!(3),
        TileFormat::Interleaved => pick!(4),
        TileFormat::FourM => pick!(5),
    }
}

#[inline(always)]
fn format_of(f: u8) -> TileFormat {
    // INVARIANT: emit_fn instantiates only these six private format tags.
    match f {
        0 => TileFormat::Real,
        1 => TileFormat::Planar,
        2 => TileFormat::OneM,
        3 => TileFormat::ThreeM,
        4 => TileFormat::Interleaved,
        5 => TileFormat::FourM,
        _ => unreachable!("private writeback format tag"),
    }
}

unsafe fn emit_fmt<T: Element, const F: u8, const G: bool>(
    ab: *const T::Real,
    mr: usize,
    nr: usize,
    mrem: usize,
    nrem: usize,
    alpha: T,
    beta: T,
    c_base: *const T,
    c_r: &[i64],
    c_c: &[i64],
    c_rs: i64,
    conj_c: bool,
    d_base: *mut T,
    d_r: &[i64],
    d_c: &[i64],
    d_rs: i64,
    conj_d: bool,
) {
    // SAFETY: caller satisfies the same tile/scatter contract as writeback.
    unsafe {
        writeback_mode::<T>(
            ab,
            format_of(F),
            mr,
            nr,
            mrem,
            nrem,
            alpha,
            beta,
            c_base,
            c_r,
            c_c,
            c_rs,
            conj_c,
            d_base,
            d_r,
            d_c,
            d_rs,
            conj_d,
            G,
        )
    }
}

/// [`emit_fmt`] compiled with AVX2 and FMA enabled.
///
/// # Safety
/// As [`emit_fmt`], and the CPU must support AVX2 and FMA (the family
/// selection that chose this variant checked it).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn emit_fmt_avx2<T: Element, const F: u8, const G: bool>(
    ab: *const T::Real,
    mr: usize,
    nr: usize,
    mrem: usize,
    nrem: usize,
    alpha: T,
    beta: T,
    c_base: *const T,
    c_r: &[i64],
    c_c: &[i64],
    c_rs: i64,
    conj_c: bool,
    d_base: *mut T,
    d_r: &[i64],
    d_c: &[i64],
    d_rs: i64,
    conj_d: bool,
) {
    // SAFETY: forwarded unchanged; the target feature is the caller's promise.
    unsafe {
        writeback_mode::<T>(
            ab,
            format_of(F),
            mr,
            nr,
            mrem,
            nrem,
            alpha,
            beta,
            c_base,
            c_r,
            c_c,
            c_rs,
            conj_c,
            d_base,
            d_r,
            d_c,
            d_rs,
            conj_d,
            G,
        )
    }
}

// Which scalar work the write-back loop does per element, as a const generic so
// the loop carries no flag (tensor4all-agent-rules#16). Resolved once per tile
// by `writeback_mode`, outside every element loop.
//
//   PLAIN   D = AB                    (beta = 0, alpha = 1, no D conjugation)
//   SCALE   D = alpha * AB            (beta = 0)
//   SCALE_C D = conj(alpha * AB)
//   ACC..   D = op_D(alpha * AB + beta * op_C(C)), four conjugation combinations
//   SUM     D = AB + C                (real storage, alpha = beta = 1)
//
// Real storage never conjugates, so it uses only PLAIN, SCALE and ACC.
const PLAIN: u8 = 0;
const SCALE: u8 = 1;
const SCALE_CD: u8 = 2;
const ACC: u8 = 3; // + 2 * conj_c + conj_d
                   // Real storage with alpha = beta = 1: D = AB + C, the later-KC accumulation of
                   // the usual call. Multiplying by exactly one is exact, so it is bitwise equal
                   // to the general ACC loop; complex storage keeps ACC, since (a + bi) * (1 + 0i)
                   // can change the sign of a zero.
const SUM: u8 = 7;

#[inline(always)]
unsafe fn writeback_mode<T: Element>(
    ab: *const T::Real,
    fmt: TileFormat,
    mr: usize,
    nr: usize,
    mrem: usize,
    nrem: usize,
    alpha: T,
    beta: T,
    c_base: *const T,
    c_r: &[i64],
    c_c: &[i64],
    c_rs: i64,
    conj_c: bool,
    d_base: *mut T,
    d_r: &[i64],
    d_c: &[i64],
    d_rs: i64,
    conj_d: bool,
    gather: bool,
) {
    let beta_is_zero = beta == T::zero();
    // Conjugating real storage is the identity, and `C` is not read at beta 0.
    let conj_d = conj_d && T::IS_COMPLEX;
    let conj_c = conj_c && T::IS_COMPLEX && !beta_is_zero;
    // `C` is only read when beta is nonzero, so its regularity only matters
    // then. `beta = 0` with a scattered `C` is the common case (the harness and
    // most callers overwrite `D`), and it must not be pushed onto the slow path.
    let c_ok = beta_is_zero || c_rs != IRREGULAR;
    let general = gather || d_rs == IRREGULAR || !c_ok;
    let (d0, c0) = (*d_r.get_unchecked(0), *c_r.get_unchecked(0));
    let mode = if beta_is_zero {
        if !conj_d && !gather && alpha == T::one() {
            PLAIN
        } else if conj_d {
            SCALE_CD
        } else {
            SCALE
        }
    } else if !T::IS_COMPLEX && alpha == T::one() && beta == T::one() {
        SUM
    } else {
        ACC + 2 * conj_c as u8 + conj_d as u8
    };
    // The arms differ in exactly two arguments -- how a row index becomes an
    // offset into `C` and into `D` -- and in the const `MODE`. A `macro_rules!`
    // rather than a struct of invariants: the expansion is textually the call,
    // so codegen is identical by construction. Each arm must stay a *separate*
    // instantiation, because that is what specialises the row addressing and
    // the per-element work; the duplication being removed here is in the
    // source, not in the binary.
    macro_rules! wb {
        ($mode:literal, $c_row:expr, $d_row:expr) => {
            writeback_rows::<T, _, _, $mode>(
                ab, fmt, mr, nr, mrem, nrem, alpha, beta, c_base, $c_row, c_c, d_base, $d_row, d_c,
            )
        };
    }
    macro_rules! by_mode {
        ($c_row:expr, $d_row:expr) => {
            match mode {
                PLAIN => wb!(0, $c_row, $d_row),
                SCALE => wb!(1, $c_row, $d_row),
                SCALE_CD => wb!(2, $c_row, $d_row),
                3 => wb!(3, $c_row, $d_row),
                4 => wb!(4, $c_row, $d_row),
                5 => wb!(5, $c_row, $d_row),
                6 => wb!(6, $c_row, $d_row),
                _ => wb!(7, $c_row, $d_row),
            }
        };
    }

    if general {
        by_mode!(|i| *c_r.get_unchecked(i), |i| *d_r.get_unchecked(i))
    } else if d_rs == 1 && (beta_is_zero || c_rs == 1) {
        // Unit stride: a micro-tile column is a contiguous run of `D`. Worth
        // its own instantiation rather than folding into the strided one,
        // because only a *compile-time* unit stride lets LLVM turn the plane
        // recombination into vector loads and interleaved stores. This is the
        // case `Plan::transposes_gemm` exists to create.
        by_mode!(|i| c0 + i as i64, |i| d0 + i as i64)
    } else {
        by_mode!(|i| c0 + c_rs * i as i64, |i| d0 + d_rs * i as i64)
    }
}

/// The write-back loop, with the row offsets supplied by `c_row`/`d_row` and
/// the per-element work fixed by `MODE` (see the table above).
///
/// Instantiated once per row-addressing mode and `MODE`.
///
/// # Safety
///
/// * `ab` must satisfy [`tile_value`]'s contract for `fmt` and `(mr, nr)`.
/// * `mrem <= mr` and `nrem <= nr`: they are the live extent of a possibly
///   partial edge tile, and the loops are bounded by them, not by `mr`/`nr`.
/// * `c_c` and `d_c` must each have at least `nrem` entries (`c_c` only when
///   `MODE` reads `C`).
/// * For every `i < mrem` and `j < nrem`, `c_base.offset(c_row(i) + c_c[j])`
///   must be a valid readable `T` (only for the `ACC` modes) and
///   `d_base.offset(d_row(i) + d_c[j])` a valid writable one. These offsets
///   come from the scatter vectors, so this is the obligation
///   `tensorcontract::Plan::check_bounds` discharges once per call for the
///   whole output rather than per tile.
/// * `D` must not alias `C`, `A` or `B`. The exclusive borrow in
///   `tensorcontract::TensorViewMut` is what supplies this; `Plan::run_raw` hands it to
///   the caller instead.
/// * `MODE` must agree with the scalars: `PLAIN` needs `beta == 0` and
///   `alpha == 1`; `SUM` needs real storage and `alpha == beta == 1`; `SCALE` and `SCALE_CD` need `beta == 0`; `SCALE_CD` and the
///   odd `ACC` modes conjugate `D`, the `ACC` modes with `MODE >= 5`
///   conjugate `C`. When `MODE` does not read `C`, `c_base` and `c_row` are
///   never used and may be dangling.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn writeback_rows<T: Element, CR, DR, const MODE: u8>(
    ab: *const T::Real,
    fmt: TileFormat,
    mr: usize,
    nr: usize,
    mrem: usize,
    nrem: usize,
    alpha: T,
    beta: T,
    c_base: *const T,
    c_row: CR,
    c_c: &[i64],
    d_base: *mut T,
    d_row: DR,
    d_c: &[i64],
) where
    CR: Fn(usize) -> i64,
    DR: Fn(usize) -> i64,
{
    let conj_d = MODE == SCALE_CD || (MODE >= ACC && MODE < SUM && (MODE - ACC) & 1 == 1);
    let conj_c = MODE >= ACC && MODE < SUM && (MODE - ACC) & 2 == 2;
    for j in 0..nrem {
        let dcj = *d_c.get_unchecked(j);
        if MODE == PLAIN {
            for i in 0..mrem {
                let (re, im) = tile_value::<T::Real>(ab, fmt, mr, nr, i, j);
                *d_base.offset((d_row(i) + dcj) as isize) = T::from_parts(re, im);
            }
        } else if MODE < ACC {
            for i in 0..mrem {
                let (re, im) = tile_value::<T::Real>(ab, fmt, mr, nr, i, j);
                let mut v = T::from_parts(re, im).mul(alpha);
                if conj_d {
                    v = v.conj();
                }
                *d_base.offset((d_row(i) + dcj) as isize) = v;
            }
        } else if MODE == SUM {
            let ccj = *c_c.get_unchecked(j);
            for i in 0..mrem {
                let (re, im) = tile_value::<T::Real>(ab, fmt, mr, nr, i, j);
                let cv = *c_base.offset((c_row(i) + ccj) as isize);
                *d_base.offset((d_row(i) + dcj) as isize) = T::from_parts(re, im).add(cv);
            }
        } else {
            let ccj = *c_c.get_unchecked(j);
            for i in 0..mrem {
                let (re, im) = tile_value::<T::Real>(ab, fmt, mr, nr, i, j);
                let cv = *c_base.offset((c_row(i) + ccj) as isize);
                let cv = if conj_c { cv.conj() } else { cv };
                let mut v = T::from_parts(re, im).mul(alpha).add(cv.mul(beta));
                if conj_d {
                    v = v.conj();
                }
                *d_base.offset((d_row(i) + dcj) as isize) = v;
            }
        }
    }
}

/// `D = op_D(beta * op_C(C))` over a whole (batched) matrix view.
///
/// Used when the contraction extent is zero, and as the `beta` pre-pass is not
/// needed elsewhere, this is the only place that touches `D` without a kernel.
///
/// # Safety
/// The scatter offsets must be in bounds for their tensors.
#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
pub unsafe fn scale_only<T: Element>(
    beta: T,
    c_base: *const T,
    c_r: &[i64],
    c_c: &[i64],
    conj_c: bool,
    d_base: *mut T,
    d_r: &[i64],
    d_c: &[i64],
    conj_d: bool,
) {
    // Flags resolved once, outside the element loops (tensor4all-agent-rules#16).
    let beta_is_zero = beta == T::zero();
    let conj_d = conj_d && T::IS_COMPLEX;
    let conj_c = conj_c && T::IS_COMPLEX;
    macro_rules! run {
        ($bz:literal, $cc:literal, $cd:literal) => {
            scale_loop::<T, $bz, $cc, $cd>(beta, c_base, c_r, c_c, d_base, d_r, d_c)
        };
    }
    match (beta_is_zero, conj_c, conj_d) {
        (true, _, false) => run!(true, false, false),
        (true, _, true) => run!(true, false, true),
        (false, false, false) => run!(false, false, false),
        (false, false, true) => run!(false, false, true),
        (false, true, false) => run!(false, true, false),
        (false, true, true) => run!(false, true, true),
    }
}

/// The `scale_only` loop with its flags as const generics: `BZ` is
/// `beta == 0` (C is not read), `CC` and `CD` conjugate `C` and `D`.
#[inline(always)]
unsafe fn scale_loop<T: Element, const BZ: bool, const CC: bool, const CD: bool>(
    beta: T,
    c_base: *const T,
    c_r: &[i64],
    c_c: &[i64],
    d_base: *mut T,
    d_r: &[i64],
    d_c: &[i64],
) {
    for (&dcj, &ccj) in d_c.iter().zip(c_c) {
        for (&dri, &cri) in d_r.iter().zip(c_r) {
            let mut v = T::zero();
            if !BZ {
                let cv = *c_base.offset((cri + ccj) as isize);
                let cv = if CC { cv.conj() } else { cv };
                v = cv.mul(beta);
            }
            if CD {
                v = v.conj();
            }
            *d_base.offset((dri + dcj) as isize) = v;
        }
    }
}

#[cfg(test)]
mod tests {
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
}
