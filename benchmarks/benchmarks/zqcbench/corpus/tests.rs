use std::collections::{BTreeMap, BTreeSet};

use super::*;

/// The plan's §5 selection: per corpus, the two most expensive formal-scaling
/// rungs, plus the three backfilled rows the collision resolutions name. The
/// tag-qualified ids are the harness family names without their ratio prefix.
const SELECTED: &[&str] = &[
    "krccsd.t2.vvvv",
    "krccsd.t2.wvoov_exchange",
    "krccsd.t2.wvoov_wvovo",
    "krccsd.t2.wvovo_exchange",
    "krccsd.wvoov.exchange",
    "krccsd.wvoov.tau",
    "krccsd.wvovo.tau",
    "somf.cpscf.exchange_response",
    "somf.coulomb",
    "somf.exchange.second",
    "somf.cpscf.laplace_left",
    "somf.cpscf.pseudo_density_virtual",
    "somf.cpscf.pseudo_density_occupied",
    "jk.response.coulomb.matrix",
    "jk.response.exchange.matrix",
    "ri-kao2mo.df.mo_vir_left",
    "ri-kao2mo.df.mo_vv_right",
    "cd-x2ccsd.ao.ladder",
    "cd-x2ccsd.t2.ring_t1t1",
    "cd-x2ccsd.wmbej.tau",
    "cd-x2ccsd.t2.oooo",
    "cd-x2ceom.sigma2.wovvo",
    "cd-x2ceom.quartic.response_oovv",
    "cd-x2ceom.quartic.tau",
    "ri-ghf.exchange.metric",
    "ri-ghf.exchange.matrix",
    "ri-ghf.exchange.occupied_transform",
    "ri-krbse.bse.coulomb",
    "ri-krbse.response.static",
    "ri-krbse.bse.screened",
    "ri-krhf.exchange.matrix",
    "ri-krhf.coulomb.aux_density",
    "ri-krhf.coulomb.matrix",
    "ri-krgw.screened.left",
    "ri-krgw.sigma.contract",
];

/// The ten corpus tags of the plan's §5 namespace clause.
const TAGS: &[&str] = &[
    "krccsd",
    "somf",
    "jk",
    "ri-kao2mo",
    "cd-x2ccsd",
    "cd-x2ceom",
    "ri-ghf",
    "ri-krbse",
    "ri-krhf",
    "ri-krgw",
];

#[test]
fn corpus_is_the_thirty_five_selected_cases_at_three_aspect_ratios() {
    let cases = corpus();
    assert_eq!(cases.len(), 3 * SELECTED.len());
    assert_eq!(
        cases.iter().map(|c| c.name).collect::<BTreeSet<_>>().len(),
        cases.len(),
        "case names must be unique"
    );
    assert_eq!(
        cases.iter().map(|c| c.group).collect::<BTreeSet<_>>().len(),
        3
    );
    for group in ["ov_03", "ov_10", "ov_20"] {
        assert_eq!(
            cases.iter().filter(|c| c.group == group).count(),
            SELECTED.len()
        );
        for case in SELECTED {
            assert!(
                cases.iter().any(|c| c.name == format!("{group}/{case}")),
                "missing {group}/{case}"
            );
        }
    }
}

#[test]
fn selected_einsums_match_the_transcription_table() {
    // (case, A, B, C) as copied into `corpus.rs`. This pins the table against a
    // second copy, so a change to one without the other fails. It is **not** a
    // check against the papers: the per-corpus sources are named in
    // `corpus.rs`'s module doc, and whether a literal matches the equation it
    // cites has to be read against the paper, which no test here can do.
    let expected = [
        ("krccsd.t2.vvvv", "abcd", "ijcd", "ijab"),
        ("krccsd.t2.wvoov_exchange", "akic", "kjbc", "ijab"),
        ("krccsd.t2.wvoov_wvovo", "akic", "kjcb", "ijab"),
        ("krccsd.t2.wvovo_exchange", "bkci", "kjac", "ijab"),
        ("krccsd.wvoov.exchange", "klcd", "ilad", "akic"),
        ("krccsd.wvoov.tau", "klcd", "liad", "akic"),
        ("krccsd.wvovo.tau", "lkcd", "liad", "akci"),
        ("somf.cpscf.exchange_response", "urvs", "sr", "uv"),
        ("somf.coulomb", "uvrs", "rs", "uv"),
        ("somf.exchange.second", "svur", "rs", "uv"),
        ("somf.cpscf.laplace_left", "ul", "lk", "uk"),
        ("somf.cpscf.pseudo_density_virtual", "ua", "va", "uv"),
        ("somf.cpscf.pseudo_density_occupied", "ui", "vi", "uv"),
        ("jk.response.coulomb.matrix", "uvrs", "sr", "uv"),
        ("jk.response.exchange.matrix", "urvs", "rs", "uv"),
        ("ri-kao2mo.df.mo_vir_left", "Pmn", "ma", "Pan"),
        ("ri-kao2mo.df.mo_vv_right", "Pan", "nb", "Pab"),
        ("cd-x2ccsd.ao.ladder", "uvsr", "ijsr", "ijuv"),
        ("cd-x2ccsd.t2.ring_t1t1", "imae", "mbje", "iajb"),
        ("cd-x2ccsd.wmbej.tau", "jnfb", "mnef", "mbej"),
        ("cd-x2ccsd.t2.oooo", "mnab", "mnij", "ijab"),
        ("cd-x2ceom.sigma2.wovvo", "mbej", "imae", "ijab"),
        ("cd-x2ceom.quartic.response_oovv", "ijef", "mnef", "ijmn"),
        ("cd-x2ceom.quartic.tau", "mnab", "ijmn", "ijab"),
        ("ri-ghf.exchange.metric", "iuP", "PQ", "iuQ"),
        ("ri-ghf.exchange.matrix", "iuP", "ivP", "uv"),
        ("ri-ghf.exchange.occupied_transform", "uvP", "vi", "iuP"),
        ("ri-krbse.bse.coulomb", "Pia", "Pjb", "iajb"),
        ("ri-krbse.response.static", "Pia", "Qai", "PQ"),
        ("ri-krbse.bse.screened", "Qij", "Qab", "ijab"),
        ("ri-krhf.exchange.matrix", "Piu", "Piv", "uv"),
        ("ri-krhf.coulomb.aux_density", "Puv", "vu", "P"),
        ("ri-krhf.coulomb.matrix", "P", "Puv", "uv"),
        ("ri-krgw.screened.left", "Pnm", "PQ", "Qnm"),
        ("ri-krgw.sigma.contract", "mnQ", "nQl", "mnl"),
    ];
    assert_eq!(expected.len(), SELECTED.len());
    for (name, a, b, c) in expected {
        let full = format!("ov_03/{name}");
        let case = OV_03
            .iter()
            .find(|case| case.name == full)
            .unwrap_or_else(|| panic!("missing {full}"));
        assert_eq!((case.a, case.b, case.c), (a, b, c), "{full}");
    }
}

fn dom_tag(dom: Dom) -> &'static str {
    match dom {
        Dom::O => "o",
        Dom::V => "v",
        Dom::Naux => "naux",
        Dom::Ao => "ao",
        Dom::Nso => "nso",
    }
}

/// One operand order's canonical form: letters relabeled
/// `<domain><first-appearance>` scanning `first`, then `second`, then the
/// output, with the two operand labels sorted so the pair is unordered within
/// this orientation. Valid for this orientation only; `contraction_key`
/// reduces over both.
fn orientation_key(
    case: &Case,
    first: &str,
    second: &str,
) -> ([String; 2], String, Vec<&'static str>) {
    let mut labels: BTreeMap<char, String> = BTreeMap::new();
    let mut next_per_domain: BTreeMap<&str, usize> = BTreeMap::new();
    let mut canon = |indices: &str| {
        indices
            .chars()
            .map(|ch| {
                if let Some(label) = labels.get(&ch) {
                    return label.clone();
                }
                let &(_, dom) = case
                    .doms
                    .iter()
                    .find(|&&(c, _)| c == ch)
                    .unwrap_or_else(|| panic!("{}: letter {ch} has no domain", case.name));
                let tag = dom_tag(dom);
                let next = next_per_domain.get(tag).copied().unwrap_or(0);
                let label = format!("{tag}{next}");
                next_per_domain.insert(tag, next + 1);
                labels.insert(ch, label.clone());
                label
            })
            .collect::<String>()
    };
    let first_labels = canon(first);
    let second_labels = canon(second);
    let output_labels = canon(case.c);
    let mut operands = [first_labels, second_labels];
    operands.sort();
    let mut summed: Vec<&str> = case
        .a
        .chars()
        .chain(case.b.chars())
        .filter(|ch| !case.c.contains(*ch))
        .map(|ch| {
            dom_tag(
                case.doms
                    .iter()
                    .find(|&&(c, _)| c == ch)
                    .unwrap_or_else(|| panic!("{}: letter {ch} has no domain", case.name))
                    .1,
            )
        })
        .collect();
    summed.sort_unstable();
    summed.dedup();
    (operands, output_labels, summed)
}

/// The operand-order-invariant canonical key: the lexicographic minimum of the
/// two orientation keys. A contraction's two orientations produce the same pair
/// of orientation keys (scanning (B, A, C) of one orientation is scanning
/// (A, B, C) of the other), so equivalent contractions share a min — and a
/// shared min implies a shared canonical orientation, i.e. a true equivalence,
/// not a coarsening. `ao` and `nso` are distinct domains, so a spinor-AO
/// contraction never collapses onto its scalar-AO namesake.
fn contraction_key(case: &Case) -> ([String; 2], String, Vec<&'static str>) {
    orientation_key(case, case.a, case.b).min(orientation_key(case, case.b, case.a))
}

/// The plan's uniqueness rule: no two selected cases may be the same
/// contraction — identical up to index relabeling and A/B operand swap, with
/// matching index-domain roles. Precedent: `cd_x2camf_eom_ccsd_corpus`'s
/// sequence-1 `sigma2.wovvo` (`mbej,imae->ijab`) is
/// `cd_x2camf_ccsd_corpus`'s sequence-3 `t2.ring_t2` with the operands
/// swapped, so the ground-state row drops and that corpus backfills sequence-5
/// `t2.oooo`; `somf`'s `exchange.first` is jk's `response.exchange.matrix`
/// verbatim and its `laplace_right` is `laplace_left` swapped. Letters are
/// canonicalized per domain, so `o`/`ao`/`nso` never collide.
#[test]
fn selected_contractions_are_pairwise_unique() {
    for group in ["ov_03", "ov_10", "ov_20"] {
        let mut seen = BTreeSet::new();
        for case in corpus().iter().filter(|case| case.group == group) {
            assert!(
                seen.insert(contraction_key(case)),
                "duplicate contraction in {group}: {}",
                case.name
            );
        }
    }
}

/// Pins the operand-order invariance `selected_contractions_are_pairwise_unique`
/// relies on, using this corpus's own precedent: the dropped
/// `cd-x2ccsd/t2.ring_t2` literal and the selected `cd-x2ceom/sigma2.wovvo`
/// written with its operands swapped. The two orientation keys of one case
/// differ (a single-order key would equate nothing), yet the invariant key
/// equates the swapped pair — so deleting the `min` in `contraction_key` fails
/// here even though no selected pair differs only by operand order.
#[test]
fn operand_swapped_duplicates_share_one_key() {
    let doms: &[(char, Dom)] = &[
        ('i', Dom::O),
        ('m', Dom::O),
        ('a', Dom::V),
        ('e', Dom::V),
        ('b', Dom::V),
        ('j', Dom::O),
    ];
    let plain = Case {
        name: "fixture/cd-x2ccsd.t2.ring_t2",
        group: "fixture",
        c: "ijab",
        a: "imae",
        b: "mbej",
        extents: &EXTENTS[0],
        family: Family::Spinor,
        doms,
    };
    let swapped = Case {
        name: "fixture/cd-x2ceom.sigma2.wovvo",
        group: "fixture",
        c: "ijab",
        a: "mbej",
        b: "imae",
        extents: &EXTENTS[0],
        family: Family::Spinor,
        doms,
    };
    assert_ne!(
        orientation_key(&plain, plain.a, plain.b),
        orientation_key(&plain, plain.b, plain.a),
        "the fixture is vacuous: one orientation's keys already coincide"
    );
    assert_eq!(
        contraction_key(&plain),
        contraction_key(&swapped),
        "a duplicate written with its operands swapped must share the key"
    );
}

#[test]
fn every_case_is_a_binary_contraction_over_its_declared_domains() {
    for case in corpus() {
        let mut count = std::collections::BTreeMap::new();
        for ch in case.a.chars().chain(case.b.chars()).chain(case.c.chars()) {
            *count.entry(ch).or_insert(0) += 1;
            assert!(
                case.doms.iter().any(|&(c, _)| c == ch),
                "{}: letter {ch} has no declared domain",
                case.name
            );
        }
        // Two appearances: a free index in C and one operand, or a summed index
        // in both operands. Three: a batch index, in A, B and C together —
        // `pbc_r_ri_gw_corpus/sigma.contract` (`mnQ,nQl->mnl`) batches over
        // `n`, so this corpus is not uniformly unit-batch.
        for (ch, n) in &count {
            assert!(
                *n == 2 || *n == 3,
                "{}: index {ch} appears {n} times",
                case.name
            );
            if *n == 3 {
                assert!(
                    case.a.contains(*ch) && case.b.contains(*ch) && case.c.contains(*ch),
                    "{}: an index that appears three times must be a batch index",
                    case.name
                );
            }
        }
        for ch in case.c.chars() {
            let n = count[&ch];
            // A free output index comes from exactly one operand; a batch
            // output index comes from both.
            assert!(
                (n == 2 && (case.a.contains(ch) ^ case.b.contains(ch)))
                    || (n == 3 && case.a.contains(ch) && case.b.contains(ch)),
                "{}: output index {ch} must be free in one operand or batch in both",
                case.name
            );
        }
        // The same letter must never name two domains.
        let letters: BTreeSet<char> = case.doms.iter().map(|&(c, _)| c).collect();
        assert_eq!(
            letters.len(),
            case.doms.len(),
            "{}: duplicate letter",
            case.name
        );
    }
}

/// The resolved layouts, `mnk` and `macs` are one consistent description:
/// `m*n*k` is the full index product even where a batch index folds into `m`
/// (`sigma.contract`), and each operand's storage length is its shape product.
#[test]
fn layouts_and_mnk_agree_with_macs() {
    for case in corpus() {
        let sized = size_case(&case, 0.0);
        let (m, n, k) = sized.mnk();
        assert_eq!(m * n * k, sized.macs(), "case {}", case.name);
        let product = |indices: &str| {
            indices
                .chars()
                .map(|ch| sized.extent_of(ch) as usize)
                .product::<usize>()
        };
        assert_eq!(
            sized.elems_a(),
            product(case.a),
            "A layout of {}",
            case.name
        );
        assert_eq!(
            sized.elems_b(),
            product(case.b),
            "B layout of {}",
            case.name
        );
        assert_eq!(
            sized.elems_c(),
            product(case.c),
            "C layout of {}",
            case.name
        );
    }
}

/// The §4 extent derivation: `v = f*o`, with `o` even so `o+v` is even;
/// spatial `ao = o+v` and `naux = round(k*ao)`; spinor `ao = (o+v)/2`
/// (exact), `nso = o+v = 2*ao` and `naux = round(k*ao)`, at the rung-median
/// `k` of the NAO panel. The `(spatial nao, naux)` and
/// `(spinor nao, nso, naux)` triples are pinned from the plan's §4 table.
#[test]
fn ratio_extents_follow_the_family_rules() {
    let expected = [
        ("ov_03", 3, 3.747, (168, 629), (84, 168, 315)),
        ("ov_10", 10, 2.568, (154, 395), (77, 154, 198)),
        ("ov_20", 20, 2.237, (168, 376), (84, 168, 188)),
    ];
    assert_eq!(EXTENTS.len(), expected.len());
    for ((group, f, k, (spatial_nao, spatial_naux), (spinor_nao, spinor_nso, spinor_naux)), e) in
        expected.into_iter().zip(EXTENTS)
    {
        assert_eq!(e.v, f * e.o, "{group}: v must be f*o");
        assert_eq!(e.k, k, "{group}: the rung-median k is pinned");
        assert_eq!(e.ao(Family::Spatial), spatial_nao, "{group}: spatial ao");
        assert_eq!(
            e.naux(Family::Spatial),
            spatial_naux,
            "{group}: spatial naux"
        );
        assert_eq!(e.ao(Family::Spinor), spinor_nao, "{group}: spinor ao");
        assert_eq!(e.nso(Family::Spinor), spinor_nso, "{group}: spinor nso");
        assert_eq!(e.naux(Family::Spinor), spinor_naux, "{group}: spinor naux");
        // The declared derivation rules, not just the table values.
        assert_eq!(spatial_nao, e.o + e.v, "{group}: spatial ao = o+v");
        assert_eq!(
            spinor_nso,
            2 * spinor_nao,
            "{group}: nso is twice the spinor ao"
        );
        assert_eq!(
            spatial_naux,
            (k * spatial_nao as f64).round() as i64,
            "{group}: spatial naux"
        );
        assert_eq!(
            spinor_naux,
            (k * spinor_nao as f64).round() as i64,
            "{group}: spinor naux"
        );
    }
    // The spinor family needs `o+v = nso = 2*ao`, so `o` is even (with
    // `v = f*o` an odd `o` would make `o+v` odd at f = 10 and 20). The extents
    // are the largest even `o` under the 12 GiB cell: at ov_10 the next even
    // value needs 14.3 GiB, at ov_03 and ov_20 the next even value needs 14.3
    // and 29.0 GiB.
    for e in EXTENTS {
        assert_eq!(e.o % 2, 0, "o must be even");
        assert_eq!(e.o + e.v, 2 * e.ao(Family::Spinor), "nso = o+v = 2*ao");
    }
}

#[test]
fn whole_binary_footprints_stay_within_the_twelve_gib_cell_budget() {
    // The extents are sized so the *whole binary* c64 footprint (A+B+C,
    // 16 bytes per element) of every selected case fits a 12 GiB per-cell
    // budget, and the binding case is the ao^4 exchange kernel
    // `somf/cpscf.exchange_response` at every ratio.
    const BYTES_PER_ELEMENT: i64 = 16;
    const CELL_BUDGET_BYTES: i64 = 12 * 1024 * 1024 * 1024;
    let mut per_group: BTreeMap<&str, (i64, &str)> = BTreeMap::new();
    for case in corpus() {
        let sized = size_case(&case, 0.0);
        let footprint =
            (sized.elems_a() + sized.elems_b() + sized.elems_c()) as i64 * BYTES_PER_ELEMENT;
        assert!(
            footprint <= CELL_BUDGET_BYTES,
            "{}: {footprint} bytes over the 12 GiB cell budget",
            case.name
        );
        let slot = per_group.entry(case.group).or_insert((0, ""));
        if footprint > slot.0 {
            *slot = (footprint, case.name);
        }
    }
    assert_eq!(per_group.len(), 3, "one binding case per ratio");
    for (group, (bytes, name)) in &per_group {
        assert!(
            name.ends_with("/somf.cpscf.exchange_response"),
            "binding case moved in {group}: {name} ({bytes} bytes)"
        );
    }
}

/// The corpus `macs` monomial for each selected case, in selection order.
/// Pinned so a domain-class or extent edit changes the computed product and
/// fails here, rather than only breaking an internal-consistency check.
const DECLARED_MACS: &[(&str, &str)] = &[
    ("krccsd.t2.vvvv", "o^2*v^4"),
    ("krccsd.t2.wvoov_exchange", "o^3*v^3"),
    ("krccsd.t2.wvoov_wvovo", "o^3*v^3"),
    ("krccsd.t2.wvovo_exchange", "o^3*v^3"),
    ("krccsd.wvoov.exchange", "o^3*v^3"),
    ("krccsd.wvoov.tau", "o^3*v^3"),
    ("krccsd.wvovo.tau", "o^3*v^3"),
    ("somf.cpscf.exchange_response", "ao^4"),
    ("somf.coulomb", "ao^4"),
    ("somf.exchange.second", "ao^4"),
    ("somf.cpscf.laplace_left", "ao^3"),
    ("somf.cpscf.pseudo_density_virtual", "ao^2*v"),
    ("somf.cpscf.pseudo_density_occupied", "ao^2*o"),
    ("jk.response.coulomb.matrix", "ao^4"),
    ("jk.response.exchange.matrix", "ao^4"),
    ("ri-kao2mo.df.mo_vir_left", "ao^2*v*naux"),
    ("ri-kao2mo.df.mo_vv_right", "ao*v^2*naux"),
    ("cd-x2ccsd.ao.ladder", "ao^4*o^2"),
    ("cd-x2ccsd.t2.ring_t1t1", "o^3*v^3"),
    ("cd-x2ccsd.wmbej.tau", "o^3*v^3"),
    ("cd-x2ccsd.t2.oooo", "o^4*v^2"),
    ("cd-x2ceom.sigma2.wovvo", "o^3*v^3"),
    ("cd-x2ceom.quartic.response_oovv", "o^4*v^2"),
    ("cd-x2ceom.quartic.tau", "o^4*v^2"),
    ("ri-ghf.exchange.metric", "o*naux^2*nso"),
    ("ri-ghf.exchange.matrix", "o*naux*nso^2"),
    ("ri-ghf.exchange.occupied_transform", "o*naux*nso^2"),
    ("ri-krbse.bse.coulomb", "o^2*v^2*naux"),
    ("ri-krbse.response.static", "o*v*naux^2"),
    ("ri-krbse.bse.screened", "o^2*v^2*naux"),
    ("ri-krhf.exchange.matrix", "ao^2*o*naux"),
    ("ri-krhf.coulomb.aux_density", "ao^2*naux"),
    ("ri-krhf.coulomb.matrix", "ao^2*naux"),
    ("ri-krgw.screened.left", "ao^2*naux^2"),
    ("ri-krgw.sigma.contract", "ao^3*naux"),
];

/// Evaluate a corpus `macs` monomial at one case's extents and family.
fn declared_macs(case: &Case, expr: &str) -> u64 {
    let e = case.extents;
    let family = case.family;
    expr.split('*')
        .map(|term| {
            let (name, exp) = match term.split_once('^') {
                Some((name, exp)) => (name, exp.parse::<u32>().expect("exponent")),
                None => (term, 1),
            };
            let base = match name {
                "o" => e.o,
                "v" => e.v,
                "naux" => e.naux(family),
                "ao" => e.ao(family),
                "nso" => e.nso(family),
                other => panic!("unknown macs domain {other} in {expr}"),
            } as u64;
            base.pow(exp)
        })
        .product()
}

#[test]
fn selection_macs_match_the_corpus_monomials() {
    assert_eq!(DECLARED_MACS.len(), SELECTED.len());
    let cases = corpus();
    for (name, expr) in DECLARED_MACS {
        assert!(SELECTED.contains(name), "{name} is not in the selection");
        for (group, e) in [
            ("ov_03", &EXTENTS[0]),
            ("ov_10", &EXTENTS[1]),
            ("ov_20", &EXTENTS[2]),
        ] {
            let full = format!("{group}/{name}");
            let case = cases
                .iter()
                .find(|case| case.name == full)
                .unwrap_or_else(|| panic!("missing {full}"));
            assert_eq!(
                size_case(case, 0.0).macs(),
                declared_macs(case, expr),
                "{full}: extent product must match the corpus monomial {expr} (extents {e:?})"
            );
        }
    }
}

/// The plan's §5 namespace clause: every family id is `<ratio>/<tag>.<case>`
/// with one of the ten corpus tags, the tag names a corpus (so case ids that
/// collide across corpora — `exchange.matrix` exists in both `ri-ghf` and
/// `ri-krhf` — stay distinct), and every corpus appears at every ratio.
#[test]
fn family_ids_carry_one_corpus_tag() {
    let mut names = BTreeSet::new();
    let mut tags = BTreeSet::new();
    for case in corpus() {
        let (group, rest) = case
            .name
            .split_once('/')
            .unwrap_or_else(|| panic!("{}: no ratio prefix", case.name));
        assert_eq!(group, case.group, "{}: group mismatch", case.name);
        assert!(
            ["ov_03", "ov_10", "ov_20"].contains(&group),
            "{}: unknown ratio {group}",
            case.name
        );
        let (tag, id) = rest
            .split_once('.')
            .unwrap_or_else(|| panic!("{}: no <tag>.<case> split", case.name));
        assert!(!id.is_empty(), "{}: empty case id", case.name);
        assert!(TAGS.contains(&tag), "{}: unknown tag {tag}", case.name);
        tags.insert(tag);
        assert!(names.insert(case.name), "duplicate family id {}", case.name);
    }
    assert_eq!(tags.len(), TAGS.len(), "every corpus must contribute a tag");
    for tag in TAGS {
        let prefix = format!("{tag}.");
        let n = corpus()
            .iter()
            .filter(|c| c.name.split_once('/').unwrap().1.starts_with(&prefix))
            .count();
        assert!(
            n > 0 && n % 3 == 0,
            "{tag}: {n} families is not 3 per group"
        );
    }
}

/// The degenerate-rung clause: `complex_jk_response_corpus` has one distinct
/// `macs` rung (`ao^4`, two rows), so the two-rung rule degenerates to it and
/// both rows are selected. The rows are not mutual twins — `uvrs,sr->uv` and
/// `urvs,rs->uv` differ in slot order — so uniqueness does not halve the
/// corpus either.
#[test]
fn the_single_rung_corpus_contributes_both_rows() {
    let jk: Vec<&Case> = OV_03
        .iter()
        .filter(|case| case.name.starts_with("ov_03/jk."))
        .collect();
    assert_eq!(jk.len(), 2, "the jk corpus must contribute both rows");
    assert_ne!(
        contraction_key(jk[0]),
        contraction_key(jk[1]),
        "the two jk rows are not mutual twins"
    );
    for case in jk {
        assert_eq!(
            size_case(case, 0.0).macs(),
            declared_macs(case, "ao^4"),
            "{}: the jk corpus has the single ao^4 rung",
            case.name
        );
    }
}
