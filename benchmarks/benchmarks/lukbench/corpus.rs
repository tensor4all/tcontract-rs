//! Lukas Devos's per-shape corpus (tensor4all/tcontract-rs#61) as programs.
//!
//! The case definitions and step programs are copied from
//! `experiments/three-engine-contract/src/main.rs` in this repository
//! (`corpus()` and the per-case drivers); they are that experiment's, not new
//! shapes. Every case is a *program*: a list of input tensors and a list of
//! pairwise steps, each step contracting two slots into a new slot with a local
//! einsum equation. A single pairwise case has one step; `ij,jk,kl->il` has two
//! in the fixed order `((ij,jk),kl)`; the MPS chain has two per site.
//!
//! Layouts are column-major throughout, `alpha = 1`, `beta = 0`.

use std::collections::HashMap;

use tcontract_kernel::Element;

/// One pairwise step `D = A * B` over slots (inputs first, then one slot per
/// step).
#[derive(Clone, Debug)]
pub struct Step {
    pub lhs: usize,
    pub rhs: usize,
    pub a: Vec<char>,
    pub b: Vec<char>,
    pub d: Vec<char>,
}

/// A whole case: fixed extents, one dtype, one or more pairwise steps.
#[derive(Clone, Debug)]
pub struct Program {
    pub name: String,
    /// Corpus family label, for the CSV's `group` column. Comma-free: the
    /// family is the case-name stem, so the label is also what `--case` matches
    /// on.
    pub group: &'static str,
    /// `f64` or `c64`: this corpus fixes the dtype per case.
    pub dtype: &'static str,
    pub inputs: Vec<Vec<usize>>,
    pub steps: Vec<Step>,
    /// Shape of every slot: the inputs, then one per step.
    pub shapes: Vec<Vec<usize>>,
    /// Multiply-accumulate count of the whole program, summed over its steps.
    /// The largest program here (MPS, chi=64) is 3.4e7, so `u64` is not a
    /// constraint; the sum is what one timed call does, because one timed call
    /// is the whole program.
    pub macs: u64,
    /// Multiplier applied to the sampled inputs. The MPS chain is scaled by
    /// `1/sqrt(2*chi)` so its intermediates stay in range over 64 steps; every
    /// other case uses 1.0.
    pub scale: f64,
}

impl Program {
    pub fn n_in(&self) -> usize {
        self.inputs.len()
    }

    /// Allocation length of one slot.
    pub fn elems(&self, slot: usize) -> usize {
        numel(&self.shapes[slot])
    }

    /// Fresh, zeroed output slots: one per step, preallocated.
    pub fn new_slots<T: Element>(&self) -> Vec<Vec<T>> {
        self.shapes[self.n_in()..]
            .iter()
            .map(|d| vec![T::zero(); numel(d)])
            .collect()
    }

    /// `m`, `n`, `k` of the equivalent matrix multiplication for one step, by
    /// the same split `tcbench`'s `Sized::mnk` uses: a label that appears in `D`
    /// and in `A` is `m`, one that appears in `D` and in `B` is `n`, everything
    /// else is `k`. A label in both operands and in `D` (a batch index) counts
    /// as `m`, again as `tcbench` does.
    pub fn mnk(&self, k: usize) -> (u64, u64, u64) {
        let st = &self.steps[k];
        let mut ext: HashMap<char, usize> = HashMap::new();
        for (l, &e) in st.a.iter().zip(&self.shapes[st.lhs]) {
            ext.insert(*l, e);
        }
        for (l, &e) in st.b.iter().zip(&self.shapes[st.rhs]) {
            ext.insert(*l, e);
        }
        let mut all: Vec<char> = st.a.iter().chain(&st.b).copied().collect();
        all.sort_unstable();
        all.dedup();
        let (mut m, mut n, mut kk) = (1u64, 1u64, 1u64);
        for l in all {
            let e = ext[&l] as u64;
            if st.d.contains(&l) {
                if st.a.contains(&l) {
                    m *= e;
                } else {
                    n *= e;
                }
            } else {
                kk *= e;
            }
        }
        (m, n, kk)
    }

    /// The `m`/`n`/`k` columns of this program's CSV rows. A program with more
    /// than one step has no single equivalent GEMM shape, so the columns carry
    /// the final step's and [`Program::shape_note`] says so.
    pub fn report_mnk(&self) -> (u64, u64, u64) {
        self.mnk(self.steps.len() - 1)
    }

    /// The chain suffix of the `notes` column: empty for a single step (where
    /// `m`/`n`/`k` are the real thing), otherwise the step count, the meaning of
    /// the `m`/`n`/`k` columns and the distinct per-step shapes with their
    /// multiplicities. Kept free of commas: the CSV is written unquoted.
    pub fn shape_note(&self) -> String {
        if self.steps.len() == 1 {
            return String::new();
        }
        let mut seen: Vec<((u64, u64, u64), usize)> = Vec::new();
        for k in 0..self.steps.len() {
            let mnk = self.mnk(k);
            match seen.iter_mut().find(|(s, _)| *s == mnk) {
                Some((_, c)) => *c += 1,
                None => seen.push((mnk, 1)),
            }
        }
        let parts: Vec<String> = seen
            .iter()
            .map(|((m, n, k), c)| {
                if *c == 1 {
                    format!("{m}x{n}x{k}")
                } else {
                    format!("{m}x{n}x{k}x{c}")
                }
            })
            .collect();
        format!(
            "steps={} m/n/k=final-step per-step MxNxK={}",
            self.steps.len(),
            parts.join(";")
        )
    }
}

/// `A,B->D` of an einsum equation.
fn parse(eq: &str) -> (Vec<char>, Vec<char>, Vec<char>) {
    let (lhs, d) = eq.split_once("->").unwrap();
    let (a, b) = lhs.split_once(',').unwrap();
    (
        a.chars().collect(),
        b.chars().collect(),
        d.chars().collect(),
    )
}

/// Build a program from input shapes and `(lhs_slot, rhs_slot, equation)` steps.
/// Extents of a shared label are checked for agreement, as the experiment does.
fn program(
    name: String,
    group: &'static str,
    dtype: &'static str,
    inputs: Vec<Vec<usize>>,
    steps: Vec<(usize, usize, &str)>,
) -> Program {
    let mut shapes = inputs.clone();
    let mut out = Vec::new();
    let mut macs = 0u64;
    for (lhs, rhs, eq) in steps {
        let (a, b, d) = parse(eq);
        let mut ext: HashMap<char, usize> = HashMap::new();
        for (l, &e) in a.iter().zip(&shapes[lhs]) {
            ext.insert(*l, e);
        }
        for (l, &e) in b.iter().zip(&shapes[rhs]) {
            if let Some(&prev) = ext.get(l) {
                assert_eq!(prev, e, "{name}: extent of {l}");
            }
            ext.insert(*l, e);
        }
        let mut all: Vec<char> = a.iter().chain(&b).copied().collect();
        all.sort_unstable();
        all.dedup();
        macs += all.iter().map(|l| ext[l] as u64).product::<u64>();
        shapes.push(d.iter().map(|l| ext[l]).collect());
        out.push(Step { lhs, rhs, a, b, d });
    }
    Program {
        name,
        group,
        dtype,
        inputs,
        steps: out,
        shapes,
        macs,
        scale: 1.0,
    }
}

/// The corpus of tensor4all/tcontract-rs#61, in the order the issue lists it.
pub fn corpus() -> Vec<Program> {
    let mut v = Vec::new();
    for batch in [16usize, 64, 256] {
        for n in [2usize, 4, 8, 16] {
            v.push(program(
                format!("ikb_knb_inb_n{n}_b{batch}"),
                "ikb_knb_inb",
                "f64",
                vec![vec![n, n, batch], vec![n, n, batch]],
                vec![(0, 1, "ikb,knb->inb")],
            ));
        }
    }
    v.push(program(
        "ijk_jkl_il_8x16x8".into(),
        "ijk_jkl_il",
        "f64",
        vec![vec![8, 16, 8], vec![16, 8, 8]],
        vec![(0, 1, "ijk,jkl->il")],
    ));
    v.push(program(
        "ij_jk_ik_c64_n32".into(),
        "ij_jk_ik",
        "c64",
        vec![vec![32, 32], vec![32, 32]],
        vec![(0, 1, "ij,jk->ik")],
    ));
    v.push(program(
        "ij_jk_ik_f64_n64".into(),
        "ij_jk_ik",
        "f64",
        vec![vec![64, 64], vec![64, 64]],
        vec![(0, 1, "ij,jk->ik")],
    ));
    v.push(program(
        "ij_jk_kl_il_n64".into(),
        "ij_jk_kl_il",
        "f64",
        vec![vec![64, 64], vec![64, 64], vec![64, 64]],
        vec![(0, 1, "ij,jk->ik"), (3, 2, "ik,kl->il")],
    ));
    // MPS chain: <phi|psi> (bilinear, no conjugation) of two random open-chain
    // MPS with uniform bond dimension chi and physical dimension 2, swept left
    // to right from a random chi x chi left environment. Per site:
    //   E[a,b] A[a,s,c] -> T[b,s,c];  T[b,s,c] B[b,s,d] -> E'[c,d].
    for chi in [4usize, 8, 16, 32, 64] {
        let l = 32;
        let mut inputs = vec![vec![chi, chi]];
        for _ in 0..l {
            inputs.push(vec![chi, 2, chi]); // ket site
            inputs.push(vec![chi, 2, chi]); // bra site
        }
        let n_in = inputs.len();
        let mut steps = Vec::new();
        let mut env = 0usize;
        for site in 0..l {
            let t = n_in + steps.len();
            steps.push((env, 1 + 2 * site, "ab,asc->bsc"));
            env = n_in + steps.len();
            steps.push((t, 2 + 2 * site, "bsc,bsd->cd"));
        }
        let mut p = program(
            format!("mps_chain_L{l}_chi{chi}"),
            "mps_chain",
            "c64",
            inputs,
            steps,
        );
        p.scale = 1.0 / (2.0 * chi as f64).sqrt();
        v.push(p);
    }
    v
}

pub fn numel(dims: &[usize]) -> usize {
    dims.iter().product()
}

/// Column-major strides (the first label is stride-1).
pub fn col_major_strides(dims: &[usize]) -> Vec<isize> {
    let mut s = Vec::with_capacity(dims.len());
    let mut acc = 1isize;
    for &d in dims {
        s.push(acc);
        acc *= d as isize;
    }
    s
}

/// Slot extents and strides as `i64`, for the FFI baseline's C API:
/// `tblis_tensor` takes `i64`.
#[cfg(feature = "tblis")]
pub fn dims_strides_i64(d: &[usize]) -> (Vec<i64>, Vec<i64>) {
    (
        d.iter().map(|&x| x as i64).collect(),
        col_major_strides(d).into_iter().map(|s| s as i64).collect(),
    )
}

pub fn label_ids(v: &[char]) -> Vec<i64> {
    v.iter().map(|&c| c as i64).collect()
}

// ---------------------------------------------------------------------------
// Naive reference: a label loop over every distinct index of one step.
// Copied from `experiments/three-engine-contract/src/main.rs::naive_step`.
// ---------------------------------------------------------------------------

fn naive_step<T: Element>(st: &Step, a: &[T], da: &[usize], b: &[T], db: &[usize]) -> Vec<T> {
    let mut labels: Vec<char> = st.a.iter().chain(&st.b).copied().collect();
    labels.sort_unstable();
    labels.dedup();
    let mut ext = HashMap::new();
    for (l, &e) in st.a.iter().zip(da) {
        ext.insert(*l, e);
    }
    for (l, &e) in st.b.iter().zip(db) {
        ext.insert(*l, e);
    }
    let dd: Vec<usize> = st.d.iter().map(|l| ext[l]).collect();
    let (sa, sb, sd) = (
        col_major_strides(da),
        col_major_strides(db),
        col_major_strides(&dd),
    );
    let pos = |l: char| labels.iter().position(|&x| x == l).unwrap();
    let ia: Vec<usize> = st.a.iter().map(|&l| pos(l)).collect();
    let ib: Vec<usize> = st.b.iter().map(|&l| pos(l)).collect();
    let id: Vec<usize> = st.d.iter().map(|&l| pos(l)).collect();
    let extents: Vec<usize> = labels.iter().map(|l| ext[l]).collect();
    let mut out = vec![T::zero(); numel(&dd)];
    let mut idx = vec![0usize; labels.len()];
    loop {
        let oa: isize = ia.iter().zip(&sa).map(|(&p, &s)| idx[p] as isize * s).sum();
        let ob: isize = ib.iter().zip(&sb).map(|(&p, &s)| idx[p] as isize * s).sum();
        let od: isize = id.iter().zip(&sd).map(|(&p, &s)| idx[p] as isize * s).sum();
        out[od as usize] = Element::add(
            out[od as usize],
            Element::mul(a[oa as usize], b[ob as usize]),
        );
        let mut k = 0;
        loop {
            if k == idx.len() {
                return out;
            }
            idx[k] += 1;
            if idx[k] < extents[k] {
                break;
            }
            idx[k] = 0;
            k += 1;
        }
    }
}

/// Run every step of `p` with the naive label loop and return the final slot.
pub fn naive_program<T: Element>(p: &Program, inputs: &[Vec<T>]) -> Vec<T> {
    let mut slots: Vec<Vec<T>> = inputs.to_vec();
    for st in &p.steps {
        let r = naive_step(
            st,
            &slots[st.lhs],
            &p.shapes[st.lhs],
            &slots[st.rhs],
            &p.shapes[st.rhs],
        );
        slots.push(r);
    }
    slots.pop().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The corpus is the experiment's, not a new one: case names, dtypes, step
    /// counts and MAC counts as recorded in
    /// `experiments/three-engine-contract/results/session1.csv`. A shape or a
    /// step program that drifts from that recording fails here.
    #[test]
    fn corpus_matches_the_recorded_experiment() {
        let mut got: Vec<(String, String, usize, u64)> = corpus()
            .into_iter()
            .map(|p| (p.name, p.dtype.to_string(), p.steps.len(), p.macs))
            .collect();
        got.sort();
        let mut want: Vec<(String, String, usize, u64)> = [
            ("ij_jk_ik_c64_n32", "c64", 1, 32768),
            ("ij_jk_ik_f64_n64", "f64", 1, 262144),
            ("ij_jk_kl_il_n64", "f64", 2, 524288),
            ("ijk_jkl_il_8x16x8", "f64", 1, 8192),
            ("ikb_knb_inb_n16_b16", "f64", 1, 65536),
            ("ikb_knb_inb_n16_b256", "f64", 1, 1048576),
            ("ikb_knb_inb_n16_b64", "f64", 1, 262144),
            ("ikb_knb_inb_n2_b16", "f64", 1, 128),
            ("ikb_knb_inb_n2_b256", "f64", 1, 2048),
            ("ikb_knb_inb_n2_b64", "f64", 1, 512),
            ("ikb_knb_inb_n4_b16", "f64", 1, 1024),
            ("ikb_knb_inb_n4_b256", "f64", 1, 16384),
            ("ikb_knb_inb_n4_b64", "f64", 1, 4096),
            ("ikb_knb_inb_n8_b16", "f64", 1, 8192),
            ("ikb_knb_inb_n8_b256", "f64", 1, 131072),
            ("ikb_knb_inb_n8_b64", "f64", 1, 32768),
            ("mps_chain_L32_chi16", "c64", 64, 524288),
            ("mps_chain_L32_chi32", "c64", 64, 4194304),
            ("mps_chain_L32_chi4", "c64", 64, 8192),
            ("mps_chain_L32_chi64", "c64", 64, 33554432),
            ("mps_chain_L32_chi8", "c64", 64, 65536),
        ]
        .into_iter()
        .map(|(n, d, s, m)| (n.to_string(), d.to_string(), s, m))
        .collect();
        want.sort();
        assert_eq!(got, want);
    }

    /// A single step's equivalent GEMM shape, and the note a chain carries
    /// instead of one: `m`/`n`/`k` are the final step's and the distinct per-step
    /// shapes follow with their multiplicities.
    #[test]
    fn mnk_and_the_chain_note() {
        let cases = corpus();
        let by = |name: &str| cases.iter().find(|p| p.name == name).unwrap();
        let pair = by("ij_jk_ik_f64_n64");
        assert_eq!(pair.mnk(0), (64, 64, 64));
        assert_eq!(pair.report_mnk(), (64, 64, 64));
        assert_eq!(pair.shape_note(), "");
        let chain = by("ij_jk_kl_il_n64");
        assert_eq!(chain.mnk(0), (64, 64, 64));
        assert_eq!(chain.mnk(1), (64, 64, 64));
        assert_eq!(
            chain.shape_note(),
            "steps=2 m/n/k=final-step per-step MxNxK=64x64x64x2"
        );
        let mps = by("mps_chain_L32_chi4");
        assert_eq!(mps.mnk(0), (4, 8, 4));
        assert_eq!(mps.report_mnk(), (4, 4, 8));
        assert_eq!(
            mps.shape_note(),
            "steps=64 m/n/k=final-step per-step MxNxK=4x8x4x32;4x4x8x32"
        );
    }
}
