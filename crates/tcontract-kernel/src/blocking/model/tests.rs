use super::*;
use crate::blocking::probe::tests::cascade_lake;
use crate::blocking::probe::{CacheSource, BUILTIN};

/// The paper's own validation table, as a check that the reconstruction of
/// `mc` is the paper's and not something else. Intel SandyBridge (E3-1220)
/// from Table III: 32 KiB 8-way L1 with 64 sets, 256 KiB 8-way L2 with 512
/// sets, `mr = 8`, `nr = 4`, double precision. The paper derives
/// `kc = 256`, `mc = 96`.
#[test]
fn reproduces_the_papers_sandybridge_row() {
    let h = CacheHierarchy {
        l1d: CacheLevel {
            level: 1,
            size: 32 * 1024,
            line: 64,
            ways: 8,
            sets: 64,
            shared_by: 1,
        },
        l2: Some(CacheLevel {
            level: 2,
            size: 256 * 1024,
            line: 64,
            ways: 8,
            sets: 512,
            shared_by: 1,
        }),
        l3: None,
        source: CacheSource::Builtin,
    };
    let g = PanelGeom {
        real_bytes: 8,
        a_reals: 1,
        b_reals: 1,
        mr: 8,
        nr: 4,
    };
    let blk = analytical(g, 1, &h);
    assert_eq!(blk.kc, 256, "eq. (4) with C_Ar = 4");
    assert_eq!(blk.mc, 96, "Table III");
}

/// The TI C6678 row of the same table: 32 KiB 4-way L1 with 256 sets,
/// 512 KiB 4-way L2 with 2048 sets, `mr = nr = 4`, giving `kc = 256`,
/// `mc = 128`.
#[test]
fn reproduces_the_papers_c6678_row() {
    let h = CacheHierarchy {
        l1d: CacheLevel {
            level: 1,
            size: 32 * 1024,
            line: 32,
            ways: 4,
            sets: 256,
            shared_by: 1,
        },
        l2: Some(CacheLevel {
            level: 2,
            size: 512 * 1024,
            line: 64,
            ways: 4,
            sets: 2048,
            shared_by: 1,
        }),
        l3: None,
        source: CacheSource::Builtin,
    };
    let g = PanelGeom {
        real_bytes: 8,
        a_reals: 1,
        b_reals: 1,
        mr: 4,
        nr: 4,
    };
    let blk = analytical(g, 1, &h);
    assert_eq!(blk.kc, 256);
    assert_eq!(blk.mc, 128);
    assert_eq!(blk.nc, NO_L3_NC, "no L3 in this machine");
}

/// Every footprint the model reasons about must actually fit the budget it
/// was derived from. This is the invariant that would break first if a
/// formula were transcribed wrongly.
#[test]
fn footprints_fit_their_budgets() {
    let h = cascade_lake();
    let l2 = h.l2.unwrap();
    let l3 = h.l3.unwrap();
    for real_bytes in [4usize, 8] {
        for (a_reals, b_reals) in [(1, 1), (2, 2), (4, 2), (3, 3)] {
            for mr in [4usize, 8, 12, 16, 24, 32, 48] {
                for nr in [3usize, 4, 6, 8, 10, 12] {
                    let g = PanelGeom {
                        real_bytes,
                        a_reals,
                        b_reals,
                        mr,
                        nr,
                    };
                    for threads in [1usize, 8, 64] {
                        let b = analytical(g, threads, &h);
                        let a_panel = mr * b.kc * a_reals * real_bytes;
                        assert!(
                            a_panel <= h.l1d.size,
                            "A micro-panel {a_panel} over L1 for {g:?}"
                        );
                        let a_block = b.mc * b.kc * a_reals * real_bytes;
                        assert!(a_block <= l2.size, "A block {a_block} over L2 for {g:?}");
                        let b_panel = b.nc * b.kc * b_reals * real_bytes;
                        let threads_here = threads.min(h.cores_sharing(&l3));
                        assert!(
                            b_panel + a_block * threads_here <= l3.size,
                            "B panel {b_panel} + {threads_here} A blocks over L3 for {g:?}"
                        );
                    }
                }
            }
        }
    }
}

/// 1m packs four reals per complex element in `A` where planar packs two,
/// so the same L2 must buy it a smaller `mc`. Getting this wrong would rig
/// the project's headline three-way comparison, which is why it is asserted
/// on the model as well as on the shipped configs.
#[test]
fn one_m_gets_a_smaller_mc_than_planar() {
    let h = cascade_lake();
    for real_bytes in [4usize, 8] {
        let planar = analytical(
            PanelGeom {
                real_bytes,
                a_reals: 2,
                b_reals: 2,
                mr: 16,
                nr: 6,
            },
            1,
            &h,
        );
        let onem = analytical(
            PanelGeom {
                real_bytes,
                a_reals: 4,
                b_reals: 2,
                mr: 16,
                nr: 6,
            },
            1,
            &h,
        );
        // At equal micro-tile shape, doubling the reals per element halves
        // both the depth the L1 can host and the rows the L2 can host.
        assert!(
            onem.kc <= planar.kc,
            "1m kc {} vs planar {}",
            onem.kc,
            planar.kc
        );
        let a_bytes = |b: &Blocking, r: usize| b.mc * b.kc * r * real_bytes;
        assert!(
            a_bytes(&onem, 4) <= a_bytes(&planar, 2) * 3 / 2,
            "1m packed A {} vs planar {}",
            a_bytes(&onem, 4),
            a_bytes(&planar, 2)
        );
    }
}

/// Threads shrink `nc` (their private `A` blocks crowd the shared `B`
/// panel out of L3) but never `mc` or `kc`, and never below one micro-panel.
#[test]
fn threads_only_affect_nc() {
    let h = cascade_lake();
    let g = PanelGeom {
        real_bytes: 8,
        a_reals: 2,
        b_reals: 2,
        mr: 16,
        nr: 6,
    };
    let one = analytical(g, 1, &h);
    let eight = analytical(g, 8, &h);
    let over = analytical(g, 64, &h);
    assert_eq!((one.mc, one.kc), (eight.mc, eight.kc));
    assert!(eight.nc < one.nc, "eight threads must not get one's nc");
    assert_eq!(
        eight.nc, over.nc,
        "threads beyond the cores sharing this L3 are on another socket"
    );
    assert!(eight.nc >= g.nr);
}

/// Absurd inputs must produce usable numbers, never zero, never a panic.
/// A probe that reports nonsense is a performance question, not a
/// correctness one.
#[test]
fn absurd_inputs_stay_sane() {
    let tiny = CacheHierarchy {
        l1d: CacheLevel {
            level: 1,
            size: 64,
            line: 64,
            ways: 1,
            sets: 1,
            shared_by: 1,
        },
        l2: Some(CacheLevel {
            level: 2,
            size: 64,
            line: 64,
            ways: 1,
            sets: 1,
            shared_by: 1,
        }),
        l3: Some(CacheLevel {
            level: 3,
            size: 64,
            line: 64,
            ways: 1,
            sets: 1,
            shared_by: 1,
        }),
        source: CacheSource::Builtin,
    };
    let huge = PanelGeom {
        real_bytes: 16,
        a_reals: 4,
        b_reals: 3,
        mr: 4096,
        nr: 4096,
    };
    for h in [tiny, BUILTIN] {
        for g in [
            huge,
            PanelGeom {
                real_bytes: 0,
                a_reals: 0,
                b_reals: 0,
                mr: 0,
                nr: 0,
            },
        ] {
            for threads in [0usize, 1, 1024] {
                let b = analytical(g, threads, &h);
                assert!(b.mc >= 1 && b.kc >= 1 && b.nc >= 1, "{b:?} for {g:?}");
                assert!(b.mc <= MAX_BLOCK && b.kc <= MAX_BLOCK && b.nc <= MAX_BLOCK);
            }
        }
    }
}

#[test]
fn a_level_without_the_next_one_still_models() {
    let mut h = cascade_lake();
    let g = PanelGeom {
        real_bytes: 8,
        a_reals: 2,
        b_reals: 2,
        mr: 16,
        nr: 6,
    };
    let full = analytical(g, 1, &h);
    h.l3 = None;
    let no_l3 = analytical(g, 1, &h);
    assert_eq!((no_l3.mc, no_l3.kc), (full.mc, full.kc));
    assert_eq!(no_l3.nc, NO_L3_NC);
    h.l2 = None;
    let no_l2 = analytical(g, 1, &h);
    assert_eq!(no_l2.kc, full.kc);
    assert_eq!(
        no_l2.mc, g.mr,
        "one micro-panel when nothing can be resident"
    );
}

#[test]
fn switch_parses_and_defaults_to_legacy() {
    assert_eq!(BlockModel::default(), BlockModel::Legacy);
    for s in ["legacy", "LEGACY", " phase2 ", "off"] {
        assert_eq!(BlockModel::parse(s), Some(BlockModel::Legacy));
    }
    for s in ["model", "analytical", "blis", "on"] {
        assert_eq!(BlockModel::parse(s), Some(BlockModel::Analytical));
    }
    assert_eq!(BlockModel::parse("maybe"), None);
    for m in [BlockModel::Legacy, BlockModel::Analytical] {
        assert_eq!(BlockModel::parse(m.name()), Some(m));
        // `Display`/`FromStr` must agree with the inherent pair: two arms of
        // a sweep are only comparable if they name the same derivation, so
        // the spelling written into a report and the one read back from a
        // script's arm list have to be the same string.
        assert_eq!(m.to_string(), m.name());
        assert_eq!(m.to_string().parse::<BlockModel>(), Ok(m));
    }
    assert!("maybe".parse::<BlockModel>().is_err());
}
