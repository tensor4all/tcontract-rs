//! Twenty-four binary contractions from nine QC corpora, at fixed aspect
//! ratios.
//!
//! Each case is one literal two-operand einsum transcribed from the paper
//! named in its corpus's `CORPUS.md` (this job's `out/real_corpus/`). The
//! corpus unit stays "one binary contraction"; permutations, denominators and
//! forms with more than two operands are metadata there, never separate cases.
//! Selection is the top-two-rung rule: per corpus, every binary of the two most
//! expensive formal-scaling rungs (the first two distinct `macs` monomials of
//! the corpus's own sequence order).
//!
//! Sources: Scuseria-Janssen-Schaefer 1988 (RCCSD, JCP 89, 7382),
//! Scuseria 1991 (RCCSD(T), JCP 94, 442), Pieniazek 2008 thesis (EOM-IP,
//! Tables C.1-C.3), Nooijen-Bartlett 1995 (EOM-EA, JCP 102, 3629),
//! Salter-Trucks-Bartlett 1989 (CCSD gradient, JCP 90, 1752), Barca et al.
//! arXiv:2407.19614 (RI-HF energy and gradients), Mester-Kallay 2019
//! (TDHF/CIS response, JCTC 15, 1690), Neese-Olbrich 2002 (MO RI-J/K
//! response, CPL 362, 170).
//!
//! Index domains: occupied `o`, virtual `v`, density-fitting `naux`, atomic
//! orbital `ao = o + v`, and the packed pair domains the papers themselves
//! use for triangular arrays: `oroot = o(o-1)/2`, `vroot = v(v-1)/2`,
//! `vsym = v(v+1)/2`, `ao_root = ao(ao-1)/2`. Packed
//! cases stay packed exactly as the corpus declares them: their MACs and
//! footprints are the packed products.
//!
//! Extents are aspect-ratio classes, not molecules. The three v/o ratios
//! 3, 10 and 20 are the median v/o of the cc-pVDZ, cc-pVTZ and cc-pVQZ rungs
//! of a 5,239-molecule NAO statistics panel (3.90, 10.17, 20.36), and
//! `naux = round(k_rung*(o+v))` uses the rung medians of naux/nao (3.747,
//! 2.568, 2.237). The extents are derived from a whole-binary memory budget:
//! the largest f64 A+B+C footprint over the selected cases stays within
//! 8 GiB per cell. The binding case is the o^3 v^3 triples output of
//! `T.connected.virtual`, so at that budget (o, v, naux) = (18, 54, 270),
//! (10, 100, 282), (7, 140, 329), and `v*o` stays nearly constant
//! (972 / 1000 / 980).

#[derive(Clone, Copy, Debug)]
pub struct Extents {
    pub o: i64,
    pub v: i64,
    pub naux: i64,
}

impl Extents {
    /// `ao = o + v`: the MO basis spans the AO space.
    fn ao(&self) -> i64 {
        self.o + self.v
    }

    fn oroot(&self) -> i64 {
        self.o * (self.o - 1) / 2
    }

    fn vroot(&self) -> i64 {
        self.v * (self.v - 1) / 2
    }

    fn vsym(&self) -> i64 {
        self.v * (self.v + 1) / 2
    }

    fn ao_root(&self) -> i64 {
        let ao = self.ao();
        ao * (ao - 1) / 2
    }
}

static EXTENTS: &[Extents] = &[
    // v/o = 3, the cc-pVDZ rung median (v/o 3.90, naux/nao 3.747).
    Extents {
        o: 18,
        v: 54,
        naux: 270,
    },
    // v/o = 10, the cc-pVTZ rung median (v/o 10.17, naux/nao 2.568).
    Extents {
        o: 10,
        v: 100,
        naux: 282,
    },
    // v/o = 20, the cc-pVQZ rung median (v/o 20.36, naux/nao 2.237).
    Extents {
        o: 7,
        v: 140,
        naux: 329,
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
    ORoot,
    VRoot,
    VSym,
    AoRoot,
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

macro_rules! qc_cases {
    ($extents:expr, $group:literal) => {
        &[
            // rccsd_corpus: packed SJS-1988 b*tau heads (vroot^2*oroot,
            // vsym*vroot*oroot).
            Case {
                name: concat!($group, "/b_tau.A"),
                group: $group,
                c: "RQ",
                a: "PQ",
                b: "RP",
                extents: &EXTENTS[$extents],
                doms: doms!['P':VRoot, 'Q':VRoot, 'R':ORoot],
            },
            Case {
                name: concat!($group, "/b_tau.S"),
                group: $group,
                c: "RQ",
                a: "PQ",
                b: "RP",
                extents: &EXTENTS[$extents],
                doms: doms!['P':VSym, 'Q':VRoot, 'R':ORoot],
            },
            // rccsd_t_corpus: (T) o^3 v^4 and o^4 v^3.
            Case {
                name: concat!($group, "/T.connected.virtual"),
                group: $group,
                c: "ijkabc",
                a: "iabf",
                b: "kjcf",
                extents: &EXTENTS[$extents],
                doms: doms!['i':O, 'a':V, 'b':V, 'f':V, 'k':O, 'j':O, 'c':V],
            },
            Case {
                name: concat!($group, "/T.connected.occupied"),
                group: $group,
                c: "ijkabc",
                a: "iajm",
                b: "mkbc",
                extents: &EXTENTS[$extents],
                doms: doms!['i':O, 'a':V, 'j':O, 'm':O, 'k':O, 'b':V, 'c':V],
            },
            // ip_eomrccsd_corpus: o^3 v^2, o^4 v.
            Case {
                name: concat!($group, "/I6.t1_oovv"),
                group: $group,
                c: "klic",
                a: "id",
                b: "klcd",
                extents: &EXTENTS[$extents],
                doms: doms!['i':O, 'd':V, 'k':O, 'l':O, 'c':V],
            },
            Case {
                name: concat!($group, "/sigma2.wovvo"),
                group: $group,
                c: "ija",
                a: "jbka",
                b: "ikb",
                extents: &EXTENTS[$extents],
                doms: doms!['j':O, 'b':V, 'k':O, 'a':V, 'i':O],
            },
            Case {
                name: concat!($group, "/sigma2.woooo"),
                group: $group,
                c: "ija",
                a: "ijkl",
                b: "kla",
                extents: &EXTENTS[$extents],
                doms: doms!['i':O, 'j':O, 'k':O, 'l':O, 'a':V],
            },
            // ea_eomrccsd_corpus: o v^4, o^2 v^3.
            Case {
                name: concat!($group, "/sigma2.w_abcd"),
                group: $group,
                c: "jab",
                a: "abcd",
                b: "jcd",
                extents: &EXTENTS[$extents],
                doms: doms!['a':V, 'b':V, 'c':V, 'd':V, 'j':O],
            },
            Case {
                name: concat!($group, "/sigma2.w_alcj"),
                group: $group,
                c: "jab",
                a: "lajc",
                b: "lcb",
                extents: &EXTENTS[$extents],
                doms: doms!['l':O, 'a':V, 'j':O, 'c':V, 'b':V],
            },
            Case {
                name: concat!($group, "/sigma2.w_bljc"),
                group: $group,
                c: "jab",
                a: "lbcj",
                b: "lca",
                extents: &EXTENTS[$extents],
                doms: doms!['l':O, 'b':V, 'c':V, 'j':O, 'a':V],
            },
            Case {
                name: concat!($group, "/sigma2.w_lbdj"),
                group: $group,
                c: "jab",
                a: "lbdj",
                b: "lad",
                extents: &EXTENTS[$extents],
                doms: doms!['l':O, 'b':V, 'd':V, 'j':O, 'a':V],
            },
            // ccsd_gradient_corpus: packed ao_root*vroot^2 density head, then
            // o v^3 ao^2.
            Case {
                name: concat!($group, "/gamma.abcd"),
                group: $group,
                c: "PA",
                a: "PC",
                b: "AC",
                extents: &EXTENTS[$extents],
                doms: doms!['P':AoRoot, 'C':VRoot, 'A':VRoot],
            },
            Case {
                name: concat!($group, "/gamma.aibc"),
                group: $group,
                c: "UVai",
                a: "UVbc",
                b: "aibc",
                extents: &EXTENTS[$extents],
                doms: doms!['U':Ao, 'V':Ao, 'b':V, 'c':V, 'a':V, 'i':O],
            },
            // ri_rhf_corpus: ao^2 naux^2, ao^2 o naux.
            Case {
                name: concat!($group, "/df.factor"),
                group: $group,
                c: "uvP",
                a: "uvQ",
                b: "QP",
                extents: &EXTENTS[$extents],
                doms: doms!['u':Ao, 'v':Ao, 'Q':Naux, 'P':Naux],
            },
            Case {
                name: concat!($group, "/exchange.matrix"),
                group: $group,
                c: "uv",
                a: "iuP",
                b: "ivP",
                extents: &EXTENTS[$extents],
                doms: doms!['i':O, 'u':Ao, 'P':Naux, 'v':Ao],
            },
            Case {
                name: concat!($group, "/exchange.occupied_transform"),
                group: $group,
                c: "iuP",
                a: "uvP",
                b: "vi",
                extents: &EXTENTS[$extents],
                doms: doms!['u':Ao, 'v':Ao, 'P':Naux, 'i':O],
            },
            // ri_rhf_gradient_corpus: ao^2 o naux, then the chain-rule pair
            // contractions o^2 naux^2 (both rungs dense; the occupied pair is
            // not packed here).
            Case {
                name: concat!($group, "/gradient.Y_transform2"),
                group: $group,
                c: "uvP",
                a: "ui",
                b: "ivP",
                extents: &EXTENTS[$extents],
                doms: doms!['u':Ao, 'i':O, 'v':Ao, 'P':Naux],
            },
            Case {
                name: concat!($group, "/gradient.X"),
                group: $group,
                c: "ijP",
                a: "PQ",
                b: "ijQ",
                extents: &EXTENTS[$extents],
                doms: doms!['i':O, 'j':O, 'P':Naux, 'Q':Naux],
            },
            Case {
                name: concat!($group, "/gradient.Z_pair"),
                group: $group,
                c: "PQ",
                a: "ijP",
                b: "ijQ",
                extents: &EXTENTS[$extents],
                doms: doms!['i':O, 'j':O, 'P':Naux, 'Q':Naux],
            },
            // df_tdhf_corpus: ao^2 o naux, o^2 naux^2.
            Case {
                name: concat!($group, "/response.exchange.occupied_half"),
                group: $group,
                c: "ivP",
                a: "ui",
                b: "uvP",
                extents: &EXTENTS[$extents],
                doms: doms!['u':Ao, 'i':O, 'v':Ao, 'P':Naux],
            },
            Case {
                name: concat!($group, "/response.exchange.transform"),
                group: $group,
                c: "rjQ",
                a: "rsQ",
                b: "sj",
                extents: &EXTENTS[$extents],
                doms: doms!['r':Ao, 's':Ao, 'Q':Naux, 'j':O],
            },
            Case {
                name: concat!($group, "/response.exchange.metric"),
                group: $group,
                c: "ijP",
                a: "PQ",
                b: "ijQ",
                extents: &EXTENTS[$extents],
                doms: doms!['P':Naux, 'Q':Naux, 'i':O, 'j':O],
            },
            // ri_jk_response_corpus: o v^2 naux, o^2 v naux.
            Case {
                name: concat!($group, "/K.aux"),
                group: $group,
                c: "Pja",
                a: "Pab",
                b: "jb",
                extents: &EXTENTS[$extents],
                doms: doms!['P':Naux, 'a':V, 'b':V, 'j':O],
            },
            Case {
                name: concat!($group, "/K.matrix"),
                group: $group,
                c: "ia",
                a: "Pja",
                b: "Pij",
                extents: &EXTENTS[$extents],
                doms: doms!['P':Naux, 'j':O, 'a':V, 'i':O],
            },
        ]
    };
}

static OV_03: &[Case] = qc_cases!(0, "ov_03");
static OV_10: &[Case] = qc_cases!(1, "ov_10");
static OV_20: &[Case] = qc_cases!(2, "ov_20");

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

pub fn size_case(case: &Case, tensor_bytes: f64) -> Sized {
    size_case_stressed(case, tensor_bytes, Stress::None)
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
            Dom::Naux => e.naux,
            Dom::Ao => e.ao(),
            Dom::ORoot => e.oroot(),
            Dom::VRoot => e.vroot(),
            Dom::VSym => e.vsym(),
            Dom::AoRoot => e.ao_root(),
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
