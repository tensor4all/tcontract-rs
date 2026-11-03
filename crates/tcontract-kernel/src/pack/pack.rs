//! Packing: gather a block of a scatter matrix into a register-blocked panel,
//! emitting whichever complex representation the chosen method needs.
//!
//! A tensor contraction has to make a scatter/gather pass over every element of
//! `A` and `B` regardless of how complex numbers are represented. That makes
//! packing the natural place to change representation: the loads are identical
//! whatever comes out, only the store pattern differs. All three complex
//! methods therefore cost the same *reads* here and differ only in stores:
//!
//! | format | reals stored per complex element |
//! |---|---|
//! | [`PackFormat::Planar`] | 2 |
//! | [`PackFormat::ThreeM`] | 3 (the sum plane is formed on the fly) |
//! | [`PackFormat::OneE`] | 4 |
//!
//! BLIS's 1m must emit A in "1e" format — the `2x2` block
//! `[[re, -im], [im, re]]` — so its packed A panel is twice the size of the
//! planar one. That is the concrete structural difference between the methods,
//! and `Blocking::derive` accounts for it by giving 1m a proportionally
//! smaller `MC` so the L2 budget stays equal.
//!
//! Conjugation of an operand is absorbed here for free by negating the
//! imaginary part as it is written, in every format.

use crate::element::Element;
use crate::family::{Layout, PackFn};
use crate::scatter::IRREGULAR;
use crate::PackFormat;

/// Number of `T::Real` values a packed panel needs.
#[inline]
#[doc(hidden)]
pub fn panel_len(vlen: usize, vr: usize, kc: usize, fmt: PackFormat) -> usize {
    vlen.div_ceil(vr) * vr * kc * fmt.reals_per_element()
}

/// Gather `vscat.len() x kscat.len()` elements into `ceil(vscat.len()/vr)`
/// slivers of `vr` "vector-axis" values per logical k-step.
///
/// The same routine packs both operands: for `A` the vector axis is the free
/// index `M` and for `B` it is the free index `N`; in both cases `kscat` walks
/// the contraction index. Element `(v, p)` of the block is at
/// `base + vscat[v] + kscat[p]`.
///
/// Sliver `s` starts at `s * vr * kc * fmt.reals_per_element()`; within it,
/// logical k-step `p` starts at `p * vr * fmt.reals_per_element()` and is laid
/// out per [`PackFormat`].
///
/// Rows past the end of a partial sliver are zero-filled so the micro-kernel
/// can always run at full register-block size.
///
/// # Safety
/// `base` must be valid for reads at every `vscat[v] + kscat[p]`, and `out`
/// valid for `panel_len(vscat.len(), vr, kscat.len(), fmt)` writes.
#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
pub unsafe fn pack_panel<T: Element>(
    base: *const T,
    vscat: &[i64],
    vbs: &[i64],
    kscat: &[i64],
    vr: usize,
    conj: bool,
    fmt: PackFormat,
    out: *mut T::Real,
) {
    match fmt {
        PackFormat::Real => pack_fmt::<T, 0>(base, vscat, vbs, kscat, vr, conj, out),
        PackFormat::Interleaved => pack_fmt::<T, 1>(base, vscat, vbs, kscat, vr, conj, out),
        PackFormat::Planar => pack_fmt::<T, 2>(base, vscat, vbs, kscat, vr, conj, out),
        PackFormat::ThreeM => pack_fmt::<T, 3>(base, vscat, vbs, kscat, vr, conj, out),
        PackFormat::OneE => pack_fmt::<T, 4>(base, vscat, vbs, kscat, vr, conj, out),
    }
}

/// The packer for one layout. `simd` selects the variant compiled with
/// `avx2,fma` enabled (x86_64 only; ignored elsewhere); the caller decides it
/// once, from the CPU check the family selection already made, so no probing
/// happens per call. Both variants run the same source and are bitwise equal.
pub(crate) fn pack_fn<T: Element>(layout: Layout, simd: bool) -> PackFn<T> {
    macro_rules! pick {
        ($f:literal) => {{
            #[cfg(target_arch = "x86_64")]
            if simd {
                return pack_fmt_avx2::<T, $f>;
            }
            let _ = simd;
            pack_fmt::<T, $f>
        }};
    }
    match layout {
        Layout::Real => pick!(0),
        Layout::Interleaved => pick!(1),
        Layout::Planar | Layout::OneR => pick!(2),
        Layout::OneE => pick!(4),
        Layout::ThreeM => pick!(3),
    }
}

/// Resolve the conjugation flag once per panel call into a const generic, so
/// the element loops carry no flag. Real storage never conjugates.
#[inline(always)]
unsafe fn pack_dispatch<T: Element, const F: u8>(
    base: *const T,
    vscat: &[i64],
    vbs: &[i64],
    kscat: &[i64],
    vr: usize,
    conj: bool,
    out: *mut T::Real,
) {
    if F != 0 && conj {
        pack_panel_fmt::<T, F, true>(base, vscat, vbs, kscat, vr, out)
    } else {
        pack_panel_fmt::<T, F, false>(base, vscat, vbs, kscat, vr, out)
    }
}

unsafe fn pack_fmt<T: Element, const F: u8>(
    base: *const T,
    vscat: &[i64],
    vbs: &[i64],
    kscat: &[i64],
    vr: usize,
    conj: bool,
    out: *mut T::Real,
) {
    pack_dispatch::<T, F>(base, vscat, vbs, kscat, vr, conj, out)
}

/// [`pack_fmt`] compiled with AVX2 and FMA enabled, so LLVM may vectorize the
/// regular-block loops with 256-bit registers.
///
/// # Safety
/// As [`pack_fmt`], and the CPU must support AVX2 and FMA (the family
/// selection that chose this variant checked it).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn pack_fmt_avx2<T: Element, const F: u8>(
    base: *const T,
    vscat: &[i64],
    vbs: &[i64],
    kscat: &[i64],
    vr: usize,
    conj: bool,
    out: *mut T::Real,
) {
    pack_dispatch::<T, F>(base, vscat, vbs, kscat, vr, conj, out)
}

#[inline(always)]
unsafe fn pack_panel_fmt<T: Element, const F: u8, const CONJ: bool>(
    base: *const T,
    vscat: &[i64],
    vbs: &[i64],
    kscat: &[i64],
    vr: usize,
    out: *mut T::Real,
) {
    let per_k = vr * pack_reals(F);
    let kc = kscat.len();
    let vlen_total = vscat.len();
    let nsliv = vlen_total.div_ceil(vr);
    let sliver_stride = per_k * kc;

    for (s, &bs) in vbs.iter().enumerate().take(nsliv) {
        let v0 = s * vr;
        let vlen = (vlen_total - v0).min(vr);
        let dst = out.add(s * sliver_stride);

        if vlen == vr && bs != IRREGULAR {
            // Regular block: the vr values for a fixed k are an arithmetic
            // progression, so no scatter table lookups are needed.
            let b0 = vscat[v0];
            for (p, &koff) in kscat.iter().enumerate() {
                let src = base.offset((b0 + koff) as isize);
                let o = dst.add(p * per_k);
                if bs == 1 {
                    for t in 0..vr {
                        emit_fmt::<T, F, CONJ>(*src.add(t), o, vr, t);
                    }
                } else {
                    for t in 0..vr {
                        emit_fmt::<T, F, CONJ>(*src.offset((bs * t as i64) as isize), o, vr, t);
                    }
                }
            }
        } else {
            // Irregular or partial block: full gather, zero-filling the tail.
            for (p, &koff) in kscat.iter().enumerate() {
                let o = dst.add(p * per_k);
                for t in 0..vr {
                    let z = if t < vlen {
                        *base.offset((*vscat.get_unchecked(v0 + t) + koff) as isize)
                    } else {
                        T::zero()
                    };
                    emit_fmt::<T, F, CONJ>(z, o, vr, t);
                }
            }
        }
    }
}

#[inline(always)]
const fn pack_reals(fmt: u8) -> usize {
    match fmt {
        0 => 1,
        1 | 2 => 2,
        3 => 3,
        4 => 4,
        _ => unreachable!(),
    }
}

/// Write one gathered element into lane `t` of a logical k-step.
///
/// # Safety
///
/// * `o` must point at the start of one logical k-step of a packed panel laid
///   out in `fmt`, with room for `vr * fmt.reals_per_element()` reals — 1, 2, 3
///   or 4 planes of `vr` depending on the format. A `fmt` that disagrees with
///   how the buffer was sized writes past its end: `OneE` stores at
///   `3 * vr + t`, four times as far as `Real`.
/// * `t < vr`.
/// * The whole k-step is treated as write-only. `emit` never reads what is
///   there, which is what lets `tensorcontract::buffer::Panel` hand back uninitialised
///   memory — but it also means every lane must be emitted before the panel is
///   read, and that is why the caller zero-fills the lanes of an edge block
///   rather than leaving them.
#[inline(always)]
unsafe fn emit_fmt<T: Element, const F: u8, const CONJ: bool>(
    z: T,
    o: *mut T::Real,
    vr: usize,
    t: usize,
) {
    let re = z.re();
    match F {
        0 => *o.add(t) = re,
        1 => {
            // INVARIANT: emit's output covers vr*reals_per_element for this step.
            let im = if CONJ { -z.im() } else { z.im() };
            *o.add(2 * t) = re;
            *o.add(2 * t + 1) = im;
        }
        2 => {
            let im = if CONJ { -z.im() } else { z.im() };
            *o.add(t) = re;
            *o.add(vr + t) = im;
        }
        3 => {
            let im = if CONJ { -z.im() } else { z.im() };
            *o.add(t) = re;
            *o.add(vr + t) = im;
            *o.add(2 * vr + t) = re + im;
        }
        4 => {
            // Real 2x2 block [[re, -im], [im, re]], stored as two real k-steps
            // of 2*vr, with the two rows of each complex lane adjacent.
            let im = if CONJ { -z.im() } else { z.im() };
            *o.add(2 * t) = re;
            *o.add(2 * t + 1) = im;
            *o.add(2 * vr + 2 * t) = -im;
            *o.add(2 * vr + 2 * t + 1) = re;
        }
        // INVARIANT: private emit_fmt is instantiated only by format tags 0..4.
        _ => unreachable!("private packing format tag"),
    }
}

#[cfg(test)]
mod tests;
