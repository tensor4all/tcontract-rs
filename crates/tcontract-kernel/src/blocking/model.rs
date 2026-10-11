//! Hardware cache descriptors, probed at run time, and the analytical
//! blocking model they drive.
//!
//! # Why this exists
//!
//! [`Blocking::derive`](crate::Blocking::derive) hardcodes three numbers —
//! `kc = 384` for 4-byte reals and `256` otherwise, a 512 KiB budget for the
//! packed `A` block and a 3 MiB budget for the packed `B` panel. They are half
//! the L2 and a slice of the L3 of *one* machine (`ccqlin038`, Cascade Lake,
//! 1 MiB L2 per core, 25 MiB L3 per socket) and they are the only
//! machine-specific quantities in the engine's blocking. Everything else is
//! derived. This module replaces them with a model whose inputs are queried
//! from the hardware, so that a decent choice transfers to a machine nobody
//! measured on.
//!
//! It is **off by default**: [`BlockModel::Analytical`] opts in, and
//! [`BlockModel::Legacy`] (the default) keeps the historical constants. Why:
//! the `legacy` default is kept on evidence, below.
//!
//! # The model
//!
//! Low, Igual, Smith & Quintana-Ortí, *"Analytical Modeling Is Enough for
//! High-Performance BLIS"*, ACM TOMS 43(2):12, 2016 (DOI 10.1145/2925987; the
//! extended version is FLAME Working Note #74, and the equation numbers below
//! are the same in both). Its thesis is exactly what is wanted here: the cache
//! blocking of a BLIS-shaped GEMM can be *computed* from the cache geometry
//! rather than searched for empirically.
//!
//! The paper's architecture model gives each cache level `Li` a line size
//! `C_Li`, an associativity `W_Li`, a size `S_Li` and a set count `N_Li` with
//! `S_Li = N_Li * C_Li * W_Li`; `S_DATA` is the size of one scalar. What the
//! paper derives, and what is implemented in [`analytical`]:
//!
//! * `kc` — from the L1. A micro-panel `A_r` (`mr x kc`) occupies `C_Ar` lines
//!   in every L1 set, a micro-panel `B_r` (`kc x nr`) occupies `C_Br`, and one
//!   line per set is reserved for the unpacked micro-tile `C_r`, so
//!   `C_Ar + C_Br <= W_L1 - 1` (eq. 5). `C_Br = ceil(nr/mr * C_Ar)`, so
//!   `C_Ar <= floor((W_L1 - 1) / (1 + nr/mr))`, taken as large as possible, and
//!   `kc = C_Ar * N_L1 * C_L1 / (mr * S_DATA)` (eq. 4). Sizing `A_r` to a whole
//!   number of ways is what makes the *next* micro-panel of `A_c` land on the
//!   same sets as the one it replaces, which is how `B_r` survives in L1. For a
//!   2-way L1 that construction leaves no room for `B_r` at all and the paper
//!   falls back to `kc = N_L1 * C_L1 / (2 * mr * S_DATA)` (eq. 6).
//! * `mc` — from the L2, by the same argument one level out: `A_c` (`mc x kc`)
//!   is the resident block, micro-panels of `B_c` stream past it, one way is
//!   reserved for `C_c`.
//! * `nc` — from the L3, one level out again: `B_c` (`kc x nc`) is resident and
//!   `A_c` streams past it.
//!
//! **What is the paper's and what is inferred.** The `mr`/`nr` equations (1–3)
//! and the `kc` equations (4–6) are quoted above and implemented as written;
//! the register blocks themselves are *not* taken from the model here, because
//! this project measures them (see `kernel::x86`). For `mc` and `nc` the paper
//! says only that they "can be derived in a similar manner" (§4.3.1) and never
//! writes the inequalities down. The `mc` rule implemented here — reserve
//! `ceil(nr*kc*S_DATA / (N_L2*C_L2))` ways for the streaming `B_c` micro-panel
//! and one for `C_c`, give the rest to `A_c` — is therefore a reconstruction,
//! but not a guess: it reproduces the paper's own Table III `mc` exactly for
//! three of its four architectures (Intel SandyBridge 96, AMD Kaveri 1792, TI
//! C6678 128) with the paper's `kc`, `mr` and `nr` as inputs. It does *not*
//! reproduce the Intel Dunnington row (it predicts 1280 against the paper's
//! 384), and the Dunnington `kc` does not follow eq. 4 either (2 ways of `A_r`
//! where the formula asks for 3), so that row appears to carry an unstated
//! extra constraint. `nc` is a further inference by the same symmetry, and the
//! paper explicitly declines to validate `nc` because three of its four
//! machines have no L3.
//!
//! # Two things the paper does not cover
//!
//! * **Reals per element.** This engine's packed panels carry a
//!   method-dependent number of reals per *logical* element — 2 for planar, 4
//!   in `A` for 1m, 3 for 3m (see [`crate::PackFormat`]) — so every footprint
//!   below is in reals, not elements. That is what makes 1m automatically get a
//!   smaller `mc` from the same L2, and getting it wrong would rig the
//!   project's three-way method comparison.
//! * **Threading.** The driver threads over `M` with a *shared* packed `B`
//!   panel, so `nc`'s L3 budget is a per-socket resource that is **not** divided
//!   among threads — but the per-thread packed `A` blocks all live in the same
//!   L3, so they are charged against it once per thread. See [`analytical`].
//!
//! # What the model still cannot see
//!
//! `mc` is bounded from *both* sides (assumption A13 in `docs/archive/tensorprimitives/notebook/`): from
//! below by packed-`A` residency in L2, which is what this model computes, and
//! from above by the strip of `D` that one `jr` pass revisits, which it does
//! not model at all. A pending measurement (Phase 4 report part 7) is designed
//! to separate the two, so no attempt is made to guess the upper bound here.

use super::probe::{CacheHierarchy, CacheLevel};
use crate::{Blocking, ParseError};

// ---------------------------------------------------------------------------
// The switch
// ---------------------------------------------------------------------------

/// Which blocking derivation is in force.
///
/// `#[non_exhaustive]`: a third derivation is a plausible outcome of the pending
/// `MC`/`KC`/`NC` grid — a per-microarchitecture table, or the model with a
/// measured correction — and every consumer of this enum is inside the engine,
/// choosing what to compute rather than being told what to supply.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum BlockModel {
    /// The hardcoded Phase 2 constants: today's shipping behaviour, and the
    /// default **on evidence**. The pending `MC`/`KC`/`NC` grid defines its arms
    /// *relative to the derived defaults*, so changing the derivation would
    /// silently change what that measurement means; and every performance
    /// number in `docs/archive/tensorprimitives/notebook/` was taken against these constants, which the
    /// project requires be comparable through a run-time switch rather than a
    /// build-to-build diff (A15). Flip the default only after an end-to-end A/B
    /// in the configuration that ships (A20).
    #[default]
    Legacy,
    /// [`analytical`], driven by [`super::probe::hierarchy`].
    Analytical,
}

impl BlockModel {
    /// Parse a model name. Several synonyms are
    /// accepted for each arm so that a sweep script can say `on`/`off` and a
    /// report can say `legacy`/`model`.
    pub fn parse(s: &str) -> Option<BlockModel> {
        match s.trim().to_ascii_lowercase().as_str() {
            "legacy" | "phase2" | "const" | "off" => Some(BlockModel::Legacy),
            "model" | "analytic" | "analytical" | "blis" | "on" => Some(BlockModel::Analytical),
            _ => None,
        }
    }

    /// The canonical name, which [`BlockModel::parse`] round-trips. Printed
    /// alongside every measurement, because two arms of a sweep taken under
    /// different derivations are not comparable.
    pub fn name(self) -> &'static str {
        match self {
            BlockModel::Legacy => "legacy",
            BlockModel::Analytical => "model",
        }
    }
}

/// [`BlockModel::name`]'s spelling, which [`FromStr`](core::str::FromStr)
/// round-trips.
impl core::fmt::Display for BlockModel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.pad(self.name())
    }
}

/// [`BlockModel::parse`] as the standard trait. The inherent method stays: the
/// a caller that falls back to the default wants the `Option`, because an
/// unrecognised value there falls back to the default rather than failing a
/// contraction.
impl core::str::FromStr for BlockModel {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<BlockModel, ParseError> {
        BlockModel::parse(s).ok_or(ParseError::new(
            "blocking model",
            "legacy | model (also phase2, const, off; analytic, analytical, blis, on)",
        ))
    }
}

// ---------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------

/// The packed-panel geometry the model needs, all of it read off the selected
/// micro-kernel.
///
/// `a_reals`/`b_reals` are the reals a packed panel carries per *logical*
/// element ([`crate::PackFormat::reals_per_element`]), which is what makes the
/// three complex methods see the same cache budget rather than the same element
/// count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PanelGeom {
    /// `S_DATA`: bytes in one real scalar.
    pub real_bytes: usize,
    /// Reals a packed `A` panel carries per logical element: 1 real, 2 planar,
    /// 4 for 1m's "1e", 3 for 3m.
    pub a_reals: usize,
    /// Reals a packed `B` panel carries per logical element. Differs from
    /// `a_reals` only under 1m, whose `B` stays in planar "1r" form.
    pub b_reals: usize,
    /// Logical rows of the micro-tile.
    pub mr: usize,
    /// Logical columns of the micro-tile.
    pub nr: usize,
}

impl PanelGeom {
    /// Bytes one k-step of an `A` micro-panel occupies (`mr * S_DATA` in the
    /// paper, which has one real per element).
    const fn a_step(&self) -> usize {
        self.mr * self.a_reals * self.real_bytes
    }
    /// Bytes one k-step of a `B` micro-panel occupies.
    const fn b_step(&self) -> usize {
        self.nr * self.b_reals * self.real_bytes
    }
}

/// Guard rails. Blocking parameters larger than this are meaningless (the
/// driver clamps to the problem anyway) and only invite overflow on a machine
/// that reports nonsense.
const MAX_BLOCK: usize = 1 << 20;

/// The analytical blocking for one kernel geometry, thread count and machine.
///
/// `threads` is the count the *plan* will execute with. It enters in exactly one
/// place: the L3. The driver threads over `M` with a **shared** packed `B`
/// panel, so `nc`'s L3 budget is a per-socket resource that all threads use
/// cooperatively and must *not* be divided by the thread count — but each
/// thread keeps its **own** packed `A` block, and those all live in the same
/// L3, so `A`'s claim on it is charged `t` times. `mc` and `kc` come from the
/// L2 and L1, which are private to a core under the one-thread-per-core
/// assumption ([`CacheHierarchy::cores_sharing`]), so they do not depend on the
/// thread count at all.
pub fn analytical(geom: PanelGeom, threads: usize, h: &CacheHierarchy) -> Blocking {
    let geom = PanelGeom {
        real_bytes: geom.real_bytes.max(1),
        a_reals: geom.a_reals.max(1),
        b_reals: geom.b_reals.max(1),
        mr: geom.mr.max(1),
        nr: geom.nr.max(1),
    };
    let kc = model_kc(&geom, &h.l1d);
    let mc = match &h.l2 {
        Some(l2) => model_mc(&geom, kc, l2),
        // No L2: the packed block has nowhere to be resident, so make it one
        // micro-panel and let `A` stream. Inference, not the paper's.
        None => geom.mr,
    };
    let nc = match &h.l3 {
        Some(l3) => model_nc(&geom, kc, mc, threads, h, l3),
        // No L3, which is the common case the paper declines to model: `nc` is
        // "for all practical purposes redundant" (§5.2), so take a panel wide
        // enough not to re-stream `A` needlessly and leave it there.
        None => NO_L3_NC,
    };
    Blocking {
        mc: mc.clamp(1, MAX_BLOCK),
        kc: kc.clamp(1, MAX_BLOCK),
        nc: nc.clamp(1, MAX_BLOCK),
    }
}

/// `nc` when the machine has no L3 at all. BLIS uses values of this order for
/// the same reason: large enough that `A_c` is not re-packed often, small
/// enough that the panel is a sane allocation. Inferred, not the paper's.
const NO_L3_NC: usize = 4096;

/// `kc` from the L1, following eq. (4)–(6) of the paper.
fn model_kc(g: &PanelGeom, l1: &CacheLevel) -> usize {
    let way = l1.bytes_per_way().max(1);
    let (a, b) = (g.a_step().max(1), g.b_step().max(1));
    // `C_Ar <= floor((W_L1 - 1) / (1 + nr/mr))`, in bytes so that the panel
    // formats' differing reals-per-element are accounted for; taken as large as
    // possible, which is the paper's advice.
    let c_ar = l1.ways.saturating_sub(1) * a / (a + b);
    if c_ar == 0 {
        // Eq. (6): a 2-way (or effectively 2-way) L1 cannot host whole ways of
        // `A_r` and a resident `B_r` at once, so split it in half instead.
        (way / (2 * a)).max(1)
    } else {
        (c_ar * way / a).max(1)
    }
}

/// `mc` from the L2. Reconstruction of the paper's "similar manner"; see the
/// module docs for what it reproduces and what it does not.
fn model_mc(g: &PanelGeom, kc: usize, l2: &CacheLevel) -> usize {
    let way = l2.bytes_per_way().max(1);
    // Ways the streaming `B_c` micro-panel occupies while `A_c` is resident.
    let c_bc = (g.b_step() * kc).div_ceil(way).max(1);
    // ... and one more for the micro-tiles of `C_c`, exactly as eq. (5) does
    // in the L1.
    let c_ac = l2.ways.saturating_sub(1).saturating_sub(c_bc).max(1);
    // NOTE (A13): this is the *lower* bound on `mc` only — the size that keeps
    // packed `A` resident in L2. `mc` is also bounded from **above** by the
    // strip of `D` that one `jr` pass revisits, which no cache-residency model
    // can see and which is why re-deriving at `min(k, KC)` swung two corpus
    // cases +13% and -18%. Phase 4 part 7's grid is designed to separate them;
    // until it has run there is deliberately nothing here.
    (c_ac * way / (kc * g.a_reals * g.real_bytes).max(1)).max(g.mr)
}

/// `nc` from the L3, with the threading correction described on [`analytical`].
fn model_nc(
    g: &PanelGeom,
    kc: usize,
    mc: usize,
    threads: usize,
    h: &CacheHierarchy,
    l3: &CacheLevel,
) -> usize {
    let way = l3.bytes_per_way().max(1);
    // One packed `A` block per thread, all of them streaming through the L3
    // that holds the one shared `B` panel. Threads beyond the cores that share
    // this cache are on another socket with an L3 of its own.
    let sharers = threads.clamp(1, h.cores_sharing(l3));
    let a_block = mc * kc * g.a_reals * g.real_bytes;
    let c_ac = (a_block * sharers).div_ceil(way).max(1);
    let c_bc = l3.ways.saturating_sub(1).saturating_sub(c_ac).max(1);
    (c_bc * way / (kc * g.b_reals * g.real_bytes).max(1)).max(g.nr)
}

#[cfg(test)]
mod tests {
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
}
