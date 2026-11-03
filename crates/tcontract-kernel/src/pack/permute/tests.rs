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
