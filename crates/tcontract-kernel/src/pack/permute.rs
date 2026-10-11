//! Row permutation across a grid of accumulator tiles.
//!
//! The blocked outer traversal (`docs/design/blocked-outer-traversal.md` in
//! `tcontract`) computes a block's micro-tiles in the *packed operand's* row
//! order, because that is the order the packer can read the operand in without
//! touching one element per cache line. The emitter, however, writes `D` fastest
//! along the *output's* axis, so the block's accumulated tiles have to be
//! reordered once, as a block, before they are emitted - reordering a single
//! 24-row tile is a no-op on the shapes this exists for, which is what the earlier
//! designs got wrong.
//!
//! The permutation moves whole rows, and it moves them **as stored**: the
//! destination tile receives the source element's format slots unchanged, so every
//! plane layout (`Real`, `Planar`, `Interleaved`, `FourM`, `OneM`, `ThreeM`,
//! `OneE`) survives bit for bit and conjugation stays where it already is - at the
//! emitter, which applies it while writing `D`. That is why this function needs the
//! slot positions and not [`tile_value`](super::writeback)'s recombined value.

use crate::element::Real;
use crate::TileFormat;

/// The real slots one logical element `(i, j)` of an `mr x nr` tile occupies, in
/// the given format, written into `out` (at most four). Returns the count.
///
/// This mirrors `tile_value`'s addressing without combining planes: `Planar`'s two
/// planes, `FourM`'s four, `OneM`'s doubled rows, and the recombining methods'
/// three and four planes are all copied as they are.
pub(crate) fn slots(
    fmt: TileFormat,
    mr: usize,
    nr: usize,
    i: usize,
    j: usize,
    out: &mut [usize; 4],
) -> usize {
    let plane = mr * nr;
    let off = j * mr + i;
    match fmt {
        TileFormat::Real => {
            out[0] = off;
            1
        }
        TileFormat::Planar => {
            out[0] = off;
            out[1] = plane + off;
            2
        }
        TileFormat::Interleaved => {
            out[0] = 2 * off;
            out[1] = 2 * off + 1;
            2
        }
        TileFormat::FourM => {
            for p in 0..4 {
                out[p] = p * plane + off;
            }
            4
        }
        TileFormat::OneM => {
            // One `2*mr x nr` real tile: row `2i` is Re, `2i+1` is Im.
            let base = j * (2 * mr) + 2 * i;
            out[0] = base;
            out[1] = base + 1;
            2
        }
        // Karatsuba's three planes: `(m1 - m2, m3 - m1 - m2)`. The emitter is
        // what recombines them; this function moves them as they are.
        TileFormat::ThreeM => {
            for p in 0..3 {
                out[p] = p * plane + off;
            }
            3
        }
    }
}

/// Reorder a grid's rows, `mtiles x ntiles` tiles of `mr x nr` in `fmt`, into
/// another grid of the same shape.
///
/// `row_map[d]` is the source row of destination row `d`, where a row is the
/// global index `tile * mr + lane` into the source grid - i.e. the map is over
/// `mtiles * mr` rows and may pull a row from any source tile. Every destination
/// slot is written, including formats whose elements have several planes, so a
/// poisoned destination is fully overwritten.
///
/// # Safety
///
/// `src` and `dst` must each cover `mtiles * ntiles * mr * nr * reals` reals,
/// where `reals` is the format's reals per element, must not overlap, and
/// `row_map` must have `mtiles * mr` entries, each below that bound.
#[allow(clippy::too_many_arguments)]
pub unsafe fn permute_grid_rows<R: Real>(
    src: *const R,
    dst: *mut R,
    mr: usize,
    nr: usize,
    fmt: TileFormat,
    mtiles: usize,
    ntiles: usize,
    row_map: &[u32],
) {
    debug_assert_eq!(row_map.len(), mtiles * mr, "one entry per destination row");
    let tile = crate::tile_planes(fmt) * mr * nr;
    for t in 0..mtiles {
        for i in 0..mr {
            let r = row_map[t * mr + i] as usize;
            let (src_tile, src_lane) = (r / mr, r % mr);
            for jt in 0..ntiles {
                for jl in 0..nr {
                    let mut ss = [0usize; 4];
                    let mut sd = [0usize; 4];
                    let n = slots(fmt, mr, nr, src_lane, jl, &mut ss);
                    let m = slots(fmt, mr, nr, i, jl, &mut sd);
                    debug_assert_eq!(m, n, "same format, same slot count");
                    let base = (src_tile * ntiles + jt) * tile;
                    let dbase = (t * ntiles + jt) * tile;
                    // SAFETY: the caller's bounds cover both grids.
                    unsafe {
                        for k in 0..n {
                            *dst.add(dbase + sd[k]) = *src.add(base + ss[k]);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MR: usize = 4;
    const NR: usize = 2;

    /// Reals per element, from the format's own doc: `Real` reads `mr*nr`,
    /// `Planar`, `Interleaved` and `OneM` read two of them, `FourM` and `ThreeM`
    /// read four and three.
    fn reals(fmt: TileFormat) -> usize {
        match fmt {
            TileFormat::Real => 1,
            TileFormat::Planar | TileFormat::Interleaved | TileFormat::OneM => 2,
            TileFormat::FourM => 4,
            TileFormat::ThreeM => 3,
        }
    }

    fn formats() -> [TileFormat; 6] {
        [
            TileFormat::Real,
            TileFormat::Planar,
            TileFormat::Interleaved,
            TileFormat::FourM,
            TileFormat::OneM,
            TileFormat::ThreeM,
        ]
    }

    /// The identity map copies the grid exactly, for every format.
    #[test]
    fn identity_is_bit_identical_for_every_format() {
        for fmt in formats() {
            let r = reals(fmt);
            let n = 2 * 2 * MR * NR * r;
            let src: Vec<f64> = (0..n).map(|i| i as f64 + 0.5).collect();
            let mut dst = vec![f64::NAN; n];
            let map: Vec<u32> = (0..(2 * MR) as u32).collect();
            unsafe { permute_grid_rows(src.as_ptr(), dst.as_mut_ptr(), MR, NR, fmt, 2, 2, &map) };
            assert_eq!(dst, src, "{fmt:?}");
        }
    }

    /// A row pulled from another tile lands in the destination's row, and every
    /// destination slot is written even when the source is drawn elsewhere.
    #[test]
    fn a_row_can_come_from_another_tile() {
        let fmt = TileFormat::Planar;
        let r = reals(fmt);
        let (mt, nt) = (2usize, 1usize);
        let n = mt * nt * MR * NR * r;
        let src: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut dst = vec![-1.0f64; n];
        // Destination tile 0 row 1 takes source tile 1 row 2.
        let mut map: Vec<u32> = (0..(mt * MR) as u32).collect();
        map[1] = (MR + 2) as u32;
        unsafe { permute_grid_rows(src.as_ptr(), dst.as_mut_ptr(), MR, NR, fmt, mt, nt, &map) };
        let tile = r * MR * NR;
        let mut sd = [0usize; 4];
        let mut ss = [0usize; 4];
        let nd = slots(fmt, MR, NR, 1, 0, &mut sd);
        let ns = slots(fmt, MR, NR, 2, 0, &mut ss);
        assert_eq!(nd, ns);
        for k in 0..nd {
            assert_eq!(
                dst[sd[k]],
                src[tile + ss[k]],
                "destination row 1 comes from the other tile"
            );
        }
        assert!(dst.iter().all(|v| *v != -1.0), "every slot written");
    }

    #[test]
    fn slots_cover_distinct_positions_per_format() {
        for fmt in formats() {
            let r = reals(fmt);
            let mut seen = std::collections::BTreeSet::new();
            for i in 0..MR {
                for j in 0..NR {
                    let mut s = [0usize; 4];
                    let n = slots(fmt, MR, NR, i, j, &mut s);
                    assert_eq!(n, r, "{fmt:?}");
                    for &o in &s[..n] {
                        assert!(o < r * MR * NR, "{fmt:?} in range");
                        assert!(seen.insert(o), "{fmt:?} position {o} used twice");
                    }
                }
            }
            assert_eq!(seen.len(), r * MR * NR, "{fmt:?} covers the tile");
        }
    }
}
