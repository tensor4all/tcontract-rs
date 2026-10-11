//! Complex arithmetic induced from a real micro-kernel.
//!
//! BLIS's 1m and 4m methods compute a complex product with real kernels only.
//! Here they are generated from any registered real family, so a complex
//! contraction can be switched onto a real kernel's arithmetic — for reference
//! results, for a machine whose complex kernel is suspect, and for comparing
//! the three methods against each other on one panel layout.
//!
//! * **1m** packs A with [`PackFormat::OneE`] and B with `OneR` (the planar
//!   format),
//!   and calls the real kernel with *twice* the logical k, because one complex
//!   k-step is two real k-steps of the 2x2 real block
//!   `[[re, -im], [im, re]]`. The resulting `2*mr' x nr` real tile is exactly
//!   [`TileFormat::OneM`], so no recombination beyond the existing write-back is
//!   needed.
//! * **4m** packs both operands planar and makes four overwrite calls into the
//!   four planes of a [`TileFormat::FourM`] tile. The four panels are copied
//!   into contiguous real panels first, because the real kernel reads one
//!   panel per k-step while planar packing interleaves the planes; the copy is
//!   the price of reusing an unmodified real kernel and is not optimised here.
//!
//! Both take `allow_auto: false`: the compiled menus already offer their own
//! native 1m, and induced families exist to be *chosen*, not to compete with it.

use crate::*;

/// The id of the complex family induced from `real`, retained for the process
/// lifetime like every other registered id. Called once per family by the
/// registry's cache.
///
/// A built-in real family named `{isa}.{rdtype}.{real|real-scalar}.{MR}x{NR}`
/// induces `{isa}.{cdtype}.{kind}[-scalar].{mr}x{nr}` with the induced family's
/// logical tile (`kind` is `i1m` or `i4m`; the `-scalar` suffix carries the
/// base implementation, which keeps the portable and the scalar-menu bases
/// apart). Any other base (a caller's family, whose id is its own opaque
/// name) keeps the `{base}.{suffix}` spelling.
fn induced_id(real: &KernelFamily<impl Real>, kind: &str, suffix: &str, mr: usize) -> &'static str {
    let parts: Vec<&str> = real.id.split('.').collect();
    let id = match (real.origin, parts.as_slice()) {
        (Origin::External { .. }, _) => format!("{}.{suffix}", real.id),
        (_, [isa, dtype, base @ ("real" | "real-scalar"), _]) if dtype.starts_with('f') => {
            let scalar = if *base == "real-scalar" {
                "-scalar"
            } else {
                ""
            };
            format!("{isa}.c{}.{kind}{scalar}.{mr}x{}", &dtype[1..], real.nr)
        }
        _ => format!("{}.{suffix}", real.id),
    };
    Box::leak(id.into_boxed_str())
}

/// The 1m-derived descriptor for a real family, or `None` when it cannot be
/// derived: a Direct kernel has no packed-panel arm to induce from, and an odd
/// logical `mr` or `nr` has no half-tile to expand.
///
/// # Examples
/// ```
/// use tcontract_kernel::{induced, KernelImpl, portable};
/// let real = portable::families_f64()[0];
/// let one_m = induced::one_m(real).unwrap();
/// assert_eq!(one_m.id, "ref.c64.i1m.2x4");
/// assert_eq!((one_m.imp, one_m.mr), (KernelImpl::Induced, real.mr / 2));
/// assert!(induced::one_m(portable::families_f64()[2]).is_none()); // direct
/// ```
pub fn one_m<R: Real>(real: &'static KernelFamily<R>) -> Option<&'static KernelFamily<R>> {
    let UkrFn::Tile(func) = real.ukr else {
        return None;
    };
    if !real.mr.is_multiple_of(2) || !real.nr.is_multiple_of(2) {
        return None;
    }
    let mr = real.mr / 2;
    let f = KernelFamily {
        id: induced_id(real, "i1m", "1m-induced", mr),
        imp: KernelImpl::Induced,
        priority: real.priority.saturating_sub(100),
        complex: Some(ComplexScheme {
            method: Method::OneM,
            a: Layout::OneE,
            b: Layout::OneR,
            tile: TileFormat::OneM,
        }),
        mr,
        nr: real.nr,
        a_per_k: 4 * mr,
        b_per_k: 2 * real.nr,
        tile_bound: mr * 2 * real.nr,
        ukr: UkrFn::Tile(func),
        c_update: CUpdate::ScratchTile,
        inner: Some(real),
        allow_auto: false,
        ..*real
    };
    Some(Box::leak(Box::new(f)))
}

/// The 4m-derived descriptor for a real family, or `None` for a Direct kernel.
/// The tile is four planes of `mr * nr` reals, recombined by the write-back.
///
/// # Examples
/// ```
/// use tcontract_kernel::{induced, portable};
/// let f = induced::four_m(portable::families_f64()[0]).unwrap();
/// assert_eq!(f.tile_bound, 4 * f.mr * f.nr);
/// assert!(f.validate().is_ok());
/// ```
pub fn four_m<R: Real>(real: &'static KernelFamily<R>) -> Option<&'static KernelFamily<R>> {
    let UkrFn::Tile(func) = real.ukr else {
        return None;
    };
    let f = KernelFamily {
        id: induced_id(real, "i4m", "4m-induced", real.mr),
        imp: KernelImpl::Induced,
        priority: real.priority.saturating_sub(100),
        complex: Some(ComplexScheme {
            method: Method::FourM,
            a: Layout::Planar,
            b: Layout::Planar,
            tile: TileFormat::FourM,
        }),
        a_per_k: 2 * real.mr,
        b_per_k: 2 * real.nr,
        tile_bound: 4 * real.mr * real.nr,
        ukr: UkrFn::Tile(func),
        c_update: CUpdate::ScratchTile,
        inner: Some(real),
        allow_auto: false,
        ..*real
    };
    Some(Box::leak(Box::new(f)))
}

/// Reals of scratch the 4m arm needs for one tile: four contiguous real panels
/// (A's and B's real and imaginary parts) of `kc` k-steps each.
pub fn four_m_scratch(mr: usize, nr: usize, kc: usize) -> usize {
    2 * kc * (mr + nr)
}

/// Call a family's tile arm.
///
/// A plain family calls its own kernel with `kc`. An induced 1m family calls
/// the inner real kernel with `2 * kc` on the same panels, and an induced 4m
/// family copies planar panels into `scratch` and makes four calls. `scratch`
/// must hold [`four_m_scratch`] reals and may be dangling otherwise.
///
/// # Safety
/// The panel and tile extents the family declares must be satisfied, exactly as
/// for [`TileUkrFn`]; `scratch` must be writable for a 4m family.
#[inline(always)]
pub unsafe fn tile_call<R: Real>(
    fam: &DriverFamily<R>,
    kc: usize,
    a: *const R,
    b: *const R,
    tile: *mut R,
    scratch: *mut R,
) {
    let UkrFn::Tile(func) = fam.kernel else {
        // INVARIANT: `driver_family` never yields a Direct arm here, because
        // the driver dispatches on the arm itself.
        unreachable!("tile_call needs a scratch arm");
    };
    match (fam.inner, fam.method) {
        (Some(inner), Some(Method::OneM)) => {
            let UkrFn::Tile(inner_func) = inner.ukr else {
                unreachable!("an induced 1m inner family is a real tile family")
            };
            // SAFETY: the caller satisfied the induced panel extents, which are
            // the inner kernel's extents at twice the logical k.
            unsafe { inner_func(2 * kc, a, b, tile) }
        }
        (Some(inner), Some(Method::FourM)) => {
            let UkrFn::Tile(inner_func) = inner.ukr else {
                unreachable!("an induced 4m inner family is a real tile family")
            };
            // SAFETY: four_m_tile's contract is the caller's, restated below.
            unsafe { four_m_tile(inner_func, fam.mr, fam.nr, kc, a, b, tile, scratch) }
        }
        _ => {
            // SAFETY: the family's own declared extents.
            unsafe { func(kc, a, b, tile) }
        }
    }
}

/// The 4m arm: copy planar panels into contiguous real ones and make the four
/// overwrite calls, one per tile plane.
///
/// # Safety
/// `a` covers `2*kc*mr` reals, `b` covers `2*kc*nr`, `tile` covers `4*mr*nr`
/// and `scratch` covers [`four_m_scratch`]; none of `tile`, `scratch` aliases the
/// panels or each other.
pub unsafe fn four_m_tile<R: Real + 'static>(
    inner: TileUkrFn<R>,
    mr: usize,
    nr: usize,
    kc: usize,
    a: *const R,
    b: *const R,
    tile: *mut R,
    scratch: *mut R,
) {
    let (ar, ai) = (scratch, scratch.add(kc * mr));
    let (br, bi) = (ai.add(kc * mr), ai.add(kc * mr + kc * nr));
    // INVARIANT: planar packing stores `[re; im]` per k-step, so the two real
    // k-rows the inner kernel reads are the two planes, copied apart.
    // SAFETY: the panels and scratch are separate allocations of at least the
    // extents above, and the inner kernel overwrites exactly one plane.
    unsafe {
        for p in 0..kc {
            core::ptr::copy_nonoverlapping(a.add(p * 2 * mr), ar.add(p * mr), mr);
            core::ptr::copy_nonoverlapping(a.add(p * 2 * mr + mr), ai.add(p * mr), mr);
            core::ptr::copy_nonoverlapping(b.add(p * 2 * nr), br.add(p * nr), nr);
            core::ptr::copy_nonoverlapping(b.add(p * 2 * nr + nr), bi.add(p * nr), nr);
        }
        let plane = mr * nr;
        inner(kc, ar, br, tile);
        inner(kc, ai, bi, tile.add(plane));
        inner(kc, ar, bi, tile.add(2 * plane));
        inner(kc, ai, br, tile.add(3 * plane));
    }
}
