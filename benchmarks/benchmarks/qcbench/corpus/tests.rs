use std::collections::BTreeSet;

use super::*;

/// The §4 selection: per corpus, the two most expensive formal-scaling rungs.
/// Names only; the einsum letters are pinned by the next test.
const SELECTED: &[&str] = &[
    "b_tau.A",
    "b_tau.S",
    "T.connected.virtual",
    "T.connected.occupied",
    "I6.t1_oovv",
    "sigma2.wovvo",
    "sigma2.woooo",
    "sigma2.w_abcd",
    "sigma2.w_alcj",
    "sigma2.w_bljc",
    "sigma2.w_lbdj",
    "gamma.abcd",
    "gamma.aibc",
    "df.factor",
    "exchange.matrix",
    "exchange.occupied_transform",
    "gradient.Y_transform2",
    "gradient.X",
    "gradient.Z_pair",
    "response.exchange.occupied_half",
    "response.exchange.transform",
    "response.exchange.metric",
    "K.aux",
    "K.matrix",
];

#[test]
fn corpus_is_the_twenty_four_selected_cases_at_three_aspect_ratios() {
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
fn selected_einsums_match_the_paper_transcriptions() {
    // (case, A, B, C) as transcribed in `out/real_corpus/*/cases.csv`.
    let expected = [
        ("b_tau.A", "PQ", "RP", "RQ"),
        ("b_tau.S", "PQ", "RP", "RQ"),
        ("T.connected.virtual", "iabf", "kjcf", "ijkabc"),
        ("T.connected.occupied", "iajm", "mkbc", "ijkabc"),
        ("I6.t1_oovv", "id", "klcd", "klic"),
        ("sigma2.wovvo", "jbka", "ikb", "ija"),
        ("sigma2.woooo", "ijkl", "kla", "ija"),
        ("sigma2.w_abcd", "abcd", "jcd", "jab"),
        ("sigma2.w_alcj", "lajc", "lcb", "jab"),
        ("sigma2.w_bljc", "lbcj", "lca", "jab"),
        ("sigma2.w_lbdj", "lbdj", "lad", "jab"),
        ("gamma.abcd", "PC", "AC", "PA"),
        ("gamma.aibc", "UVbc", "aibc", "UVai"),
        ("df.factor", "uvQ", "QP", "uvP"),
        ("exchange.matrix", "iuP", "ivP", "uv"),
        ("exchange.occupied_transform", "uvP", "vi", "iuP"),
        ("gradient.Y_transform2", "ui", "ivP", "uvP"),
        ("gradient.X", "PQ", "ijQ", "ijP"),
        ("gradient.Z_pair", "ijP", "ijQ", "PQ"),
        ("response.exchange.occupied_half", "ui", "uvP", "ivP"),
        ("response.exchange.transform", "rsQ", "sj", "rjQ"),
        ("response.exchange.metric", "PQ", "ijQ", "ijP"),
        ("K.aux", "Pab", "jb", "Pja"),
        ("K.matrix", "Pja", "Pij", "ia"),
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
        // In a binary contraction every index appears exactly twice: an output
        // index in C and one operand, a summed index in both operands.
        for (ch, n) in count {
            assert_eq!(n, 2, "{}: index {ch} appears {n} times", case.name);
        }
        for ch in case.c.chars() {
            assert!(
                case.a.contains(ch) ^ case.b.contains(ch),
                "{}: output index {ch} must come from exactly one operand",
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

#[test]
fn ratio_extents_follow_the_derivation_rule() {
    // v = f·o with f the ratio denominator, and naux = round(k·(o+v)) at
    // the rung-median k of the NAO statistics panel.
    let expected = [
        ("ov_03", 3, 3.747, (18, 54, 270)),
        ("ov_10", 10, 2.568, (10, 100, 282)),
        ("ov_20", 20, 2.237, (7, 140, 329)),
    ];
    for (group, f, k, (o, v, naux)) in expected {
        let e = EXTENTS
            .iter()
            .find(|e| e.o == o && e.v == v && e.naux == naux)
            .unwrap_or_else(|| panic!("{group} extents missing from EXTENTS"));
        assert_eq!(e.v, f * e.o, "{group}: v must be f·o");
        assert_eq!(
            e.naux,
            (k * (e.o + e.v) as f64).round() as i64,
            "{group}: naux must follow the rung-median rule"
        );
        assert!(
            e.naux > e.v,
            "{group}: DF auxiliary space exceeds the virtual space"
        );
    }
}

#[test]
fn packed_domains_extend_to_the_triangular_sizes() {
    let e = &EXTENTS[2];
    assert_eq!(e.ao(), e.o + e.v);
    assert_eq!(e.oroot(), e.o * (e.o - 1) / 2);
    assert_eq!(e.vroot(), e.v * (e.v - 1) / 2);
    assert_eq!(e.vsym(), e.v * (e.v + 1) / 2);
    assert_eq!(e.ao_root(), e.ao() * (e.ao() - 1) / 2);

    // The packed rank-1 heads keep their packed shape: b_tau.A is
    // vroot x vroot = vroot^2*oroot MACs, and gamma.abcd is
    // ao_root*vroot^2.
    let b_tau = size_case(
        OV_20.iter().find(|c| c.name == "ov_20/b_tau.A").unwrap(),
        0.0,
    );
    assert_eq!(b_tau.elems_a(), (e.vroot() * e.vroot()) as usize);
    assert_eq!(b_tau.macs(), (e.vroot() * e.vroot() * e.oroot()) as u64);
    let gamma = size_case(
        OV_20.iter().find(|c| c.name == "ov_20/gamma.abcd").unwrap(),
        0.0,
    );
    assert_eq!(gamma.elems_c(), (e.ao_root() * e.vroot()) as usize);
}

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

#[test]
fn whole_binary_footprints_stay_within_the_eight_gib_cell_budget() {
    // The extents are sized so the *whole binary* f64 footprint (A+B+C) of
    // every selected case fits an 8 GiB per-cell budget, and the binding case
    // is the o^3 v^3 triples output of T.connected.virtual.
    const CELL_BUDGET_ELEMS: i64 = 8 * 1024 * 1024 * 1024 / 8;
    for e in EXTENTS {
        let triples = e.o * e.o * e.o * e.v * e.v * e.v;
        assert!(
            triples <= CELL_BUDGET_ELEMS,
            "o^3 v^3 at (o={}, v={}) exceeds the 8 GiB cell budget",
            e.o,
            e.v
        );
    }
    let mut binding = (0i64, "");
    for case in corpus() {
        let sized = size_case(&case, 0.0);
        let footprint = (sized.elems_a() + sized.elems_b() + sized.elems_c()) as i64;
        assert!(footprint <= CELL_BUDGET_ELEMS, "case {}", case.name);
        if footprint > binding.0 {
            binding = (footprint, case.name);
        }
    }
    assert!(
        binding.1.ends_with("/T.connected.virtual"),
        "binding case moved: {}",
        binding.1
    );
}
