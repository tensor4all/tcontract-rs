//! Thirty-five binary contractions from ten complex-orbital QC corpora, at
//! fixed aspect ratios.
//!
//! Each case is one literal two-operand einsum transcribed from the paper named
//! in its corpus's `CORPUS.md` (the benchmark job's `out/complex_corpus/`,
//! built on the shared `_common/binary_case.py`). The corpus unit stays "one
//! binary contraction"; permutations, denominators, conjugation and forms with
//! more than two operands are metadata there, never separate cases. One case is
//! batched: `ri-krgw/sigma.contract` (`mnQ,nQl->mnl`) carries `n` in both
//! operands and the output; it is still one binary contraction, and the harness
//! times the whole batched call (the shared `mnk()` folds the batch extent into
//! `m`, and `m*n*k` still equals the MAC count).
//!
//! Selection is the top-two-rung rule: per corpus, every binary of the two most
//! expensive formal-scaling rungs (the first two distinct `macs` monomials of
//! the corpus's own sequence order), under the plan's uniqueness rule (no two
//! selected cases may be the same contraction up to index relabeling and A/B
//! operand swap, with matching index-domain roles) and its degenerate-rung
//! clause (`complex_jk_response_corpus` has a single distinct rung, `ao^4`, and
//! contributes both of its rows).
//!
//! The plan's §5 collision resolutions are pinned by `corpus/tests.rs`:
//!
//! - `somf/somf.exchange.first` is `complex_jk_response_corpus`'s
//!   `response.exchange.matrix` verbatim, so it drops and `somf` backfills
//!   sequence-7 `cpscf.pseudo_density_virtual`;
//! - `somf/cpscf.laplace_right` is the A/B-swapped form of the selected
//!   `cpscf.laplace_left`, so it drops and `somf` backfills sequence-8
//!   `cpscf.pseudo_density_occupied`;
//! - `cd_x2camf_ccsd_corpus/t2.ring_t2` is the A/B-swapped form of the selected
//!   `cd_x2camf_eom_ccsd_corpus/sigma2.wovvo`, so it drops and `cd-x2ccsd`
//!   backfills sequence-5 `t2.oooo`.
//!
//! `pbc_r_ri_bse_corpus` keeps both of its rung-1 rows: `bse.coulomb`
//! (`Pia,Pjb->iajb`) and `bse.screened` (`Qij,Qab->ijab`) share the
//! `o^2*v^2*naux` MAC count but not their operand domains, so they are
//! different contractions under the rule; that corpus needs no backfill, and
//! `bse.screened_left` (sequence 4, rung 3) is not selected.
//!
//! Sources: Hirata et al., JCP 120, 2581 (2004) (periodic RCCSD);
//! Glasbrenner et al., JCP 150, 024104 (2019) (SO-MF/CPSCF);
//! Desmarais et al., JCTC 2021, DOI 10.1021/acs.jctc.1c00460 (complex JK/CPKS);
//! Zhou et al., JCTC, DOI 10.1021/acs.jctc.4c01245 (RI-K AO->MO setup, RI-BSE
//! kernels); Zhang et al., JCTC, DOI 10.1021/acs.jctc.3c01236 (CD/AO
//! X2CAMF-CCSD and EOM-CCSD, with Epifanovsky et al., JCP 139, 134105 (2013));
//! Armbruster et al., PCCP 10, 1748 (2008) (two-component RI-JK);
//! arXiv 2607.28554 (periodic RI-HF); Zhu-Chan G0W0.
//!
//! Index domains: occupied `o`, virtual `v`, density-fitting or Cholesky
//! `naux`, scalar atomic orbital `ao`, and the spinor-AO basis `nso = 2*ao`.
//! The two-component corpora count `o`/`v` in complex spinors, so their scalar
//! AO basis is `(o+v)/2` and `nso` is its affine double; the spatial corpora'
//! `ao` is `o+v`. `o` is even at every ratio, so `o+v` is even and the spinor
//! `(o+v)/2` is exact (`nso = o+v = 2*ao`). No selected case carries a packed
//! pair domain, a Cartesian `component` extent or a momentum `g` extent: those
//! are corpus metadata, and k-point multiplicity is call metadata, never an
//! extent.
//!
//! Extents are aspect-ratio classes, not molecules. The three v/o ratios 3, 10
//! and 20 stand for the cc-pVDZ, cc-pVTZ and cc-pVQZ rung medians of a
//! 5,239-molecule NAO statistics panel (3.90, 10.17, 20.36), and
//! `naux = round(k*nao)` uses the rung medians of naux/nao (3.747, 2.568,
//! 2.237) on the family's `nao`: `o+v` for spatial, `(o+v)/2` for spinor. The
//! extents are the largest **even** `o` with `v = f*o` under a whole-binary
//! memory budget: the largest c64 (`Complex<f64>`, 16 bytes per element) A+B+C
//! footprint over the selected cases stays within 12 GiB per cell. The binding
//! case is the ao^4 exchange kernel `somf.cpscf.exchange_response`
//! (11.87 / 8.38 / 11.87 GiB).

/// Which orbital-count convention a corpus declares (spec §3).
///
/// The values are complex in both families; the family only decides how `ao`
/// and `naux` resolve from the shared `(o, v)` pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// `ao = o + v` (molecular spatial or complex k-space spatial orbitals).
    Spatial,
    /// Two-component spinors: `o`/`v` count spinors, `ao = (o+v)/2`,
    /// `nso = 2*ao`.
    Spinor,
}

#[derive(Clone, Copy, Debug)]
pub struct Extents {
    pub o: i64,
    pub v: i64,
    /// The rung-median `naux/nao` of the NAO panel this ratio stands for.
    pub k: f64,
}

impl Extents {
    /// The scalar-AO count of the family's `nao`.
    ///
    /// The spinor family's `o`/`v` are spinor counts. `o` is even at every
    /// ratio, so `o+v` is even and the integer half is exact.
    pub fn ao(&self, family: Family) -> i64 {
        match family {
            Family::Spatial => self.o + self.v,
            Family::Spinor => (self.o + self.v) / 2,
        }
    }

    /// The spinor-AO basis, the affine double of the family's `ao`. Only the
    /// two-component corpora declare a `nso` domain.
    pub fn nso(&self, family: Family) -> i64 {
        2 * self.ao(family)
    }

    /// `round(k*nao)` at the family's `nao`.
    pub fn naux(&self, family: Family) -> i64 {
        (self.k * self.ao(family) as f64).round() as i64
    }
}

static EXTENTS: &[Extents] = &[
    // v/o = 3, the cc-pVDZ rung median (v/o 3.90, naux/nao 3.747).
    Extents {
        o: 42,
        v: 126,
        k: 3.747,
    },
    // v/o = 10, the cc-pVTZ rung median (v/o 10.17, naux/nao 2.568).
    Extents {
        o: 14,
        v: 140,
        k: 2.568,
    },
    // v/o = 20, the cc-pVQZ rung median (v/o 20.36, naux/nao 2.237).
    Extents {
        o: 8,
        v: 160,
        k: 2.237,
    },
];

/// Index domain of one einsum letter. The corpus's `index_domains` column is
/// the source of truth; a case carries exactly the domains its letters use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dom {
    O,
    V,
    Naux,
    Ao,
    /// Spinor-AO basis: `nso = 2*ao`.
    Nso,
}

macro_rules! doms {
    ($($ch:literal : $d:ident),* $(,)?) => {
        &[$(($ch, Dom::$d)),*]
    };
}

#[derive(Clone, Debug)]
pub struct Layout {
    extents: Vec<i64>,
    strides: Vec<i64>,
}

impl Layout {
    pub fn col_major(extents: &[i64]) -> Self {
        let mut strides = Vec::with_capacity(extents.len());
        let mut acc = 1i64;
        for &e in extents {
            strides.push(acc);
            acc *= e.max(1);
        }
        Self {
            extents: extents.to_vec(),
            strides,
        }
    }

    pub fn extents(&self) -> &[i64] {
        &self.extents
    }

    pub fn strides(&self) -> &[i64] {
        &self.strides
    }

    pub fn storage_len(&self) -> i64 {
        let mut n = 1i64;
        for (&e, &s) in self.extents.iter().zip(&self.strides) {
            if e > 0 {
                n += (e - 1) * s.abs();
            }
        }
        n
    }
}

#[derive(Clone, Debug)]
pub struct Case {
    pub name: &'static str,
    pub group: &'static str,
    pub c: &'static str,
    pub a: &'static str,
    pub b: &'static str,
    pub extents: &'static Extents,
    pub family: Family,
    pub doms: &'static [(char, Dom)],
}

#[derive(Clone, Debug)]
pub struct Sized {
    pub case: Case,
    pub labels: Vec<char>,
    pub extents: Vec<i64>,
    pub la: Layout,
    pub lb: Layout,
    pub lc: Layout,
    pub idx_a: Vec<i64>,
    pub idx_b: Vec<i64>,
    pub idx_c: Vec<i64>,
}

impl Sized {
    #[cfg(test)]
    pub fn extent_of(&self, ch: char) -> i64 {
        self.extents[self.labels.iter().position(|&c| c == ch).unwrap()]
    }

    pub fn macs(&self) -> u64 {
        self.extents.iter().map(|&e| e as u64).product()
    }

    pub fn elems_a(&self) -> usize {
        self.la.storage_len() as usize
    }

    pub fn elems_b(&self) -> usize {
        self.lb.storage_len() as usize
    }

    pub fn elems_c(&self) -> usize {
        self.lc.storage_len() as usize
    }

    pub fn mnk(&self) -> (u64, u64, u64) {
        let inm = |ch: char| self.case.c.contains(ch) && self.case.a.contains(ch);
        let inn = |ch: char| self.case.c.contains(ch) && self.case.b.contains(ch);
        let mut m = 1u64;
        let mut n = 1u64;
        let mut k = 1u64;
        for (i, &ch) in self.labels.iter().enumerate() {
            let e = self.extents[i] as u64;
            if inm(ch) {
                m *= e;
            } else if inn(ch) {
                n *= e;
            } else {
                k *= e;
            }
        }
        (m, n, k)
    }
}

macro_rules! zq_cases {
    ($extents:expr, $group:literal) => {
        &[
            // pbc_rccsd_corpus: Hirata et al. 2004 periodic RCCSD T1/T2 update.
            // Rung 1 o^2 v^4, rung 2 o^3 v^3 (all six rows).
            Case {
                name: concat!($group, "/krccsd.t2.vvvv"),
                group: $group,
                c: "ijab",
                a: "abcd",
                b: "ijcd",
                extents: &EXTENTS[$extents],
                family: Family::Spatial,
                doms: doms!['a': V, 'b': V, 'c': V, 'd': V, 'i': O, 'j': O],
            },
            Case {
                name: concat!($group, "/krccsd.t2.wvoov_exchange"),
                group: $group,
                c: "ijab",
                a: "akic",
                b: "kjbc",
                extents: &EXTENTS[$extents],
                family: Family::Spatial,
                doms: doms!['a': V, 'k': O, 'i': O, 'c': V, 'j': O, 'b': V],
            },
            Case {
                name: concat!($group, "/krccsd.t2.wvoov_wvovo"),
                group: $group,
                c: "ijab",
                a: "akic",
                b: "kjcb",
                extents: &EXTENTS[$extents],
                family: Family::Spatial,
                doms: doms!['a': V, 'k': O, 'i': O, 'c': V, 'j': O, 'b': V],
            },
            Case {
                name: concat!($group, "/krccsd.t2.wvovo_exchange"),
                group: $group,
                c: "ijab",
                a: "bkci",
                b: "kjac",
                extents: &EXTENTS[$extents],
                family: Family::Spatial,
                doms: doms!['b': V, 'k': O, 'c': V, 'i': O, 'j': O, 'a': V],
            },
            Case {
                name: concat!($group, "/krccsd.wvoov.exchange"),
                group: $group,
                c: "akic",
                a: "klcd",
                b: "ilad",
                extents: &EXTENTS[$extents],
                family: Family::Spatial,
                doms: doms!['k': O, 'l': O, 'c': V, 'd': V, 'i': O, 'a': V],
            },
            Case {
                name: concat!($group, "/krccsd.wvoov.tau"),
                group: $group,
                c: "akic",
                a: "klcd",
                b: "liad",
                extents: &EXTENTS[$extents],
                family: Family::Spatial,
                doms: doms!['k': O, 'l': O, 'c': V, 'd': V, 'i': O, 'a': V],
            },
            Case {
                name: concat!($group, "/krccsd.wvovo.tau"),
                group: $group,
                c: "akci",
                a: "lkcd",
                b: "liad",
                extents: &EXTENTS[$extents],
                family: Family::Spatial,
                doms: doms!['l': O, 'k': O, 'c': V, 'd': V, 'i': O, 'a': V],
            },
            // somf_cpscf_corpus: Glasbrenner et al. 2019 SO-MF/CPSCF. Rungs ao^4
            // (four rows) and ao^3 (two); the seq-7/8 pseudo-densities backfill the
            // drops of exchange.first (jk twin) and laplace_right (laplace_left twin).
            Case {
                name: concat!($group, "/somf.cpscf.exchange_response"),
                group: $group,
                c: "uv",
                a: "urvs",
                b: "sr",
                extents: &EXTENTS[$extents],
                family: Family::Spatial,
                doms: doms!['u': Ao, 'r': Ao, 'v': Ao, 's': Ao],
            },
            Case {
                name: concat!($group, "/somf.coulomb"),
                group: $group,
                c: "uv",
                a: "uvrs",
                b: "rs",
                extents: &EXTENTS[$extents],
                family: Family::Spatial,
                doms: doms!['u': Ao, 'v': Ao, 'r': Ao, 's': Ao],
            },
            Case {
                name: concat!($group, "/somf.exchange.second"),
                group: $group,
                c: "uv",
                a: "svur",
                b: "rs",
                extents: &EXTENTS[$extents],
                family: Family::Spatial,
                doms: doms!['s': Ao, 'v': Ao, 'u': Ao, 'r': Ao],
            },
            Case {
                name: concat!($group, "/somf.cpscf.laplace_left"),
                group: $group,
                c: "uk",
                a: "ul",
                b: "lk",
                extents: &EXTENTS[$extents],
                family: Family::Spatial,
                doms: doms!['u': Ao, 'l': Ao, 'k': Ao],
            },
            Case {
                name: concat!($group, "/somf.cpscf.pseudo_density_virtual"),
                group: $group,
                c: "uv",
                a: "ua",
                b: "va",
                extents: &EXTENTS[$extents],
                family: Family::Spatial,
                doms: doms!['u': Ao, 'a': V, 'v': Ao],
            },
            Case {
                name: concat!($group, "/somf.cpscf.pseudo_density_occupied"),
                group: $group,
                c: "uv",
                a: "ui",
                b: "vi",
                extents: &EXTENTS[$extents],
                family: Family::Spatial,
                doms: doms!['u': Ao, 'i': O, 'v': Ao],
            },
            // complex_jk_response_corpus: Desmarais et al. 2021 complex JK/CPKS.
            // One rung only (ao^4), so the degenerate-rung clause takes both rows.
            Case {
                name: concat!($group, "/jk.response.coulomb.matrix"),
                group: $group,
                c: "uv",
                a: "uvrs",
                b: "sr",
                extents: &EXTENTS[$extents],
                family: Family::Spatial,
                doms: doms!['u': Ao, 'v': Ao, 'r': Ao, 's': Ao],
            },
            Case {
                name: concat!($group, "/jk.response.exchange.matrix"),
                group: $group,
                c: "uv",
                a: "urvs",
                b: "rs",
                extents: &EXTENTS[$extents],
                family: Family::Spatial,
                doms: doms!['u': Ao, 'r': Ao, 'v': Ao, 's': Ao],
            },
            // cd_ri_ao2mo_corpus: Zhou et al. Eq. (43) RI-K AO->MO setup.
            // Rungs ao^2 v naux and ao v^2 naux.
            Case {
                name: concat!($group, "/ri-kao2mo.df.mo_vir_left"),
                group: $group,
                c: "Pan",
                a: "Pmn",
                b: "ma",
                extents: &EXTENTS[$extents],
                family: Family::Spatial,
                doms: doms!['P': Naux, 'm': Ao, 'n': Ao, 'a': V],
            },
            Case {
                name: concat!($group, "/ri-kao2mo.df.mo_vv_right"),
                group: $group,
                c: "Pab",
                a: "Pan",
                b: "nb",
                extents: &EXTENTS[$extents],
                family: Family::Spatial,
                doms: doms!['P': Naux, 'a': V, 'n': Ao, 'b': V],
            },
            // cd_x2camf_ccsd_corpus: Zhang et al. 2023 CD/AO X2CAMF-CCSD spinors.
            // Rung 1 ao^4 o^2; rung 2 o^3 v^3, whose seq-3 row collides with the EOM
            // sigma2 and is backfilled by seq-5 t2.oooo.
            Case {
                name: concat!($group, "/cd-x2ccsd.ao.ladder"),
                group: $group,
                c: "ijuv",
                a: "uvsr",
                b: "ijsr",
                extents: &EXTENTS[$extents],
                family: Family::Spinor,
                doms: doms!['u': Ao, 'v': Ao, 's': Ao, 'r': Ao, 'i': O, 'j': O],
            },
            Case {
                name: concat!($group, "/cd-x2ccsd.t2.ring_t1t1"),
                group: $group,
                c: "iajb",
                a: "imae",
                b: "mbje",
                extents: &EXTENTS[$extents],
                family: Family::Spinor,
                doms: doms!['i': O, 'm': O, 'a': V, 'e': V, 'b': V, 'j': O],
            },
            Case {
                name: concat!($group, "/cd-x2ccsd.wmbej.tau"),
                group: $group,
                c: "mbej",
                a: "jnfb",
                b: "mnef",
                extents: &EXTENTS[$extents],
                family: Family::Spinor,
                doms: doms!['j': O, 'n': O, 'f': V, 'b': V, 'm': O, 'e': V],
            },
            Case {
                name: concat!($group, "/cd-x2ccsd.t2.oooo"),
                group: $group,
                c: "ijab",
                a: "mnab",
                b: "mnij",
                extents: &EXTENTS[$extents],
                family: Family::Spinor,
                doms: doms!['m': O, 'n': O, 'a': V, 'b': V, 'i': O, 'j': O],
            },
            // cd_x2camf_eom_ccsd_corpus: Zhang et al. Eqs. (13)-(16) + Epifanovsky
            // et al. 2013. Rungs o^3 v^3 and o^4 v^2.
            Case {
                name: concat!($group, "/cd-x2ceom.sigma2.wovvo"),
                group: $group,
                c: "ijab",
                a: "mbej",
                b: "imae",
                extents: &EXTENTS[$extents],
                family: Family::Spinor,
                doms: doms!['m': O, 'b': V, 'e': V, 'j': O, 'i': O, 'a': V],
            },
            Case {
                name: concat!($group, "/cd-x2ceom.quartic.response_oovv"),
                group: $group,
                c: "ijmn",
                a: "ijef",
                b: "mnef",
                extents: &EXTENTS[$extents],
                family: Family::Spinor,
                doms: doms!['i': O, 'j': O, 'e': V, 'f': V, 'm': O, 'n': O],
            },
            Case {
                name: concat!($group, "/cd-x2ceom.quartic.tau"),
                group: $group,
                c: "ijab",
                a: "mnab",
                b: "ijmn",
                extents: &EXTENTS[$extents],
                family: Family::Spinor,
                doms: doms!['m': O, 'n': O, 'a': V, 'b': V, 'i': O, 'j': O],
            },
            // complex_ri_ghf_gks_corpus: Armbruster et al. 2008 two-component RI-JK.
            // Rungs o naux^2 nso and o naux nso^2 (nso is the spinor-AO basis).
            Case {
                name: concat!($group, "/ri-ghf.exchange.metric"),
                group: $group,
                c: "iuQ",
                a: "iuP",
                b: "PQ",
                extents: &EXTENTS[$extents],
                family: Family::Spinor,
                doms: doms!['i': O, 'u': Nso, 'P': Naux, 'Q': Naux],
            },
            Case {
                name: concat!($group, "/ri-ghf.exchange.matrix"),
                group: $group,
                c: "uv",
                a: "iuP",
                b: "ivP",
                extents: &EXTENTS[$extents],
                family: Family::Spinor,
                doms: doms!['i': O, 'u': Nso, 'P': Naux, 'v': Nso],
            },
            Case {
                name: concat!($group, "/ri-ghf.exchange.occupied_transform"),
                group: $group,
                c: "iuP",
                a: "uvP",
                b: "vi",
                extents: &EXTENTS[$extents],
                family: Family::Spinor,
                doms: doms!['u': Nso, 'v': Nso, 'P': Naux, 'i': O],
            },
            // pbc_r_ri_bse_corpus: Zhou et al. Eqs. (45)-(46) BSE kernels. Rung 1
            // keeps both bse.coulomb and bse.screened (same MAC count, different
            // operand domains); rung 2 is response.static, and bse.screened_left
            // (rung 3) is not selected.
            Case {
                name: concat!($group, "/ri-krbse.bse.coulomb"),
                group: $group,
                c: "iajb",
                a: "Pia",
                b: "Pjb",
                extents: &EXTENTS[$extents],
                family: Family::Spatial,
                doms: doms!['P': Naux, 'i': O, 'a': V, 'j': O, 'b': V],
            },
            Case {
                name: concat!($group, "/ri-krbse.response.static"),
                group: $group,
                c: "PQ",
                a: "Pia",
                b: "Qai",
                extents: &EXTENTS[$extents],
                family: Family::Spatial,
                doms: doms!['P': Naux, 'i': O, 'a': V, 'Q': Naux],
            },
            Case {
                name: concat!($group, "/ri-krbse.bse.screened"),
                group: $group,
                c: "ijab",
                a: "Qij",
                b: "Qab",
                extents: &EXTENTS[$extents],
                family: Family::Spatial,
                doms: doms!['Q': Naux, 'i': O, 'j': O, 'a': V, 'b': V],
            },
            // pbc_r_ri_gdf_hf_corpus: arXiv 2607.28554 periodic RI-HF. Rungs
            // ao^2 o naux and ao^2 naux.
            Case {
                name: concat!($group, "/ri-krhf.exchange.matrix"),
                group: $group,
                c: "uv",
                a: "Piu",
                b: "Piv",
                extents: &EXTENTS[$extents],
                family: Family::Spatial,
                doms: doms!['P': Naux, 'i': O, 'u': Ao, 'v': Ao],
            },
            Case {
                name: concat!($group, "/ri-krhf.coulomb.aux_density"),
                group: $group,
                c: "P",
                a: "Puv",
                b: "vu",
                extents: &EXTENTS[$extents],
                family: Family::Spatial,
                doms: doms!['P': Naux, 'u': Ao, 'v': Ao],
            },
            Case {
                name: concat!($group, "/ri-krhf.coulomb.matrix"),
                group: $group,
                c: "uv",
                a: "P",
                b: "Puv",
                extents: &EXTENTS[$extents],
                family: Family::Spatial,
                doms: doms!['P': Naux, 'u': Ao, 'v': Ao],
            },
            // pbc_r_ri_gw_corpus: Zhu-Chan G0W0. Rungs ao^2 naux^2 and ao^3 naux.
            Case {
                name: concat!($group, "/ri-krgw.screened.left"),
                group: $group,
                c: "Qnm",
                a: "Pnm",
                b: "PQ",
                extents: &EXTENTS[$extents],
                family: Family::Spatial,
                doms: doms!['P': Naux, 'n': Ao, 'm': Ao, 'Q': Naux],
            },
            Case {
                name: concat!($group, "/ri-krgw.sigma.contract"),
                group: $group,
                c: "mnl",
                a: "mnQ",
                b: "nQl",
                extents: &EXTENTS[$extents],
                family: Family::Spatial,
                doms: doms!['m': Ao, 'n': Ao, 'Q': Naux, 'l': Ao],
            },
        ]
    };
}

static OV_03: &[Case] = zq_cases!(0, "ov_03");
static OV_10: &[Case] = zq_cases!(1, "ov_10");
static OV_20: &[Case] = zq_cases!(2, "ov_20");

pub fn corpus() -> Vec<Case> {
    [OV_03, OV_10, OV_20]
        .into_iter()
        .flat_map(|cases| cases.iter().cloned())
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Stress {
    #[default]
    None,
}

impl Stress {
    pub fn parse(s: &str) -> Option<Stress> {
        (s == "none").then_some(Stress::None)
    }

    pub fn name(self) -> &'static str {
        "none"
    }
}

/// The verifier's known-value fixture: `A = B = 1` makes every output element
/// exactly the contracted extent. The shared `verify` asks `size_case` for this
/// fixture at a nonzero nominal tensor size; a fixed-extent corpus cannot scale
/// a family down to that size, so that request is answered with this dedicated
/// tiny product instead of a full family. A filtered `verify` must not execute
/// an unselected 444.6-GMAC corpus case for its self-check. Not a census cell;
/// independent of the case argument and of any `--case`/`--family` filter.
fn known_value_fixture() -> Sized {
    let case = Case {
        name: "known-value",
        group: "known-value",
        c: "ij",
        a: "ik",
        b: "kj",
        extents: &KNOWN_VALUE_EXTENTS,
        family: Family::Spatial,
        doms: doms!['i': O, 'j': V, 'k': O],
    };
    size_case_stressed(&case, 0.0, Stress::None)
}

/// The fixture's extents: `i`/`k` resolve to `o = 2`, `j` to `v = 3`, so
/// `A = 2x2`, `B = 2x3`, `C = 2x3` and the contraction has 12 MACs.
static KNOWN_VALUE_EXTENTS: Extents = Extents { o: 2, v: 3, k: 1.0 };

/// The case at a nominal tensor size. The only nonzero caller is the shared
/// verifier's known-value check (256 MiB); zqcbench's corpora are fixed-extent,
/// so that call returns the dedicated [`known_value_fixture`]. Every corpus-path
/// caller passes `0.0` and gets the case's fixed extents.
pub fn size_case(case: &Case, tensor_bytes: f64) -> Sized {
    if tensor_bytes > 0.0 {
        known_value_fixture()
    } else {
        size_case_stressed(case, tensor_bytes, Stress::None)
    }
}

pub fn size_case_stressed(case: &Case, tensor_bytes: f64, stress: Stress) -> Sized {
    let _ = tensor_bytes;
    let Stress::None = stress;
    let domain = |ch: char| {
        case.doms
            .iter()
            .find_map(|&(c, d)| (c == ch).then_some(d))
            .unwrap_or_else(|| panic!("unknown QC index {ch} in {}", case.name))
    };
    let extent = |ch: char| {
        let e = case.extents;
        match domain(ch) {
            Dom::O => e.o,
            Dom::V => e.v,
            Dom::Naux => e.naux(case.family),
            Dom::Ao => e.ao(case.family),
            Dom::Nso => e.nso(case.family),
        }
    };
    let mut labels = Vec::new();
    for indices in [case.c, case.a, case.b] {
        for ch in indices.chars() {
            if !labels.contains(&ch) {
                labels.push(ch);
            }
        }
    }
    let extents: Vec<i64> = labels.iter().map(|&ch| extent(ch)).collect();
    let shape = |indices: &str| -> Vec<i64> { indices.chars().map(extent).collect() };
    let idx = |indices: &str| -> Vec<i64> { indices.chars().map(|ch| ch as i64).collect() };
    Sized {
        case: case.clone(),
        labels,
        extents,
        la: Layout::col_major(&shape(case.a)),
        lb: Layout::col_major(&shape(case.b)),
        lc: Layout::col_major(&shape(case.c)),
        idx_a: idx(case.a),
        idx_b: idx(case.b),
        idx_c: idx(case.c),
    }
}

#[cfg(test)]
mod tests;
