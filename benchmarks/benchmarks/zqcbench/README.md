# Complex-orbital binary-contraction benchmark (two-component spinor and spatial-orbital, c64)

`zqcbench` reuses `tcbench`'s execution engines, timing, TBLIS adapter, thread
enforcement and evidence format — the same shared modules `qcbench` includes via
`#[path]`, unedited. It replaces isotropic TCCG sizing with fixed
occupied/virtual/auxiliary extents at quantum-chemistry aspect ratios, and it is
the complex-double counterpart of `qcbench`: every operand is `Complex<f64>`
(`c64`, NumPy `complex128`), contiguous column-major, `alpha = 1`, `beta = 0`,
and input generation and plan construction are outside the timed call. `f64` is
not a dtype this suite accepts: the driver exits rather than skipping it, and
the real-orbital benchmark is `qcbench`.

## Corpus

Thirty-five binary contractions selected from ten complex-orbital corpus
transcriptions (source material in the benchmark job's `out/complex_corpus/`,
one `CORPUS.md` per corpus, built on the shared `_common/binary_case.py`). Three
corpora are two-component spinor routes (X2CAMF CCSD/EOM, two-component RI-JK);
seven are spatial-orbital routes — five periodic k-space (periodic CCSD,
GDF-HF, GW, BSE, the shared AO→MO staging) and two molecular (SO-MF/CPSCF and
the GHF/GKS response kernels).

Selection is uniform: per corpus, every binary of the two most expensive
formal-scaling rungs (the first two distinct `macs` monomials of the corpus's own
sequence order), under the plan's uniqueness rule: no two selected cases may be
the same contraction up to index relabeling and A/B operand swap, with matching
index-domain roles. `complex_jk_response_corpus` has a single distinct rung
(`ao^4`, two rows), so the two-rung rule degenerates to it and both rows are
selected (the degenerate-rung clause). Case ids keep the corpus vocabulary; the
harness prefixes one short tag per corpus, because case ids collide across
corpora (`exchange.matrix` exists in both `complex_ri_ghf_gks_corpus` and
`pbc_r_ri_gdf_hf_corpus`). A case id that already begins with its tag is not
double-prefixed: `ov_03/somf.coulomb`, not `ov_03/somf.somf.coulomb`.

| Corpus (tag) | Paper | Selected cases |
|---|---|---|
| `pbc_rccsd_corpus` (`krccsd`) | Hirata et al., *J. Chem. Phys.* **120**, 2581 (2004), DOI [10.1063/1.1637577](https://doi.org/10.1063/1.1637577) | `t2.vvvv`, `t2.wvoov_exchange`, `t2.wvoov_wvovo`, `t2.wvovo_exchange`, `wvoov.exchange`, `wvoov.tau`, `wvovo.tau` |
| `somf_cpscf_corpus` (`somf`) | Glasbrenner et al., *J. Chem. Phys.* **150**, 024104 (2019) | `cpscf.exchange_response`, `somf.coulomb`, `somf.exchange.second`, `cpscf.laplace_left`, `cpscf.pseudo_density_virtual`, `cpscf.pseudo_density_occupied` |
| `complex_jk_response_corpus` (`jk`) | Desmarais, Erba, Flament & Kirtman, *J. Chem. Theory Comput.* (2021), DOI [10.1021/acs.jctc.1c00460](https://doi.org/10.1021/acs.jctc.1c00460) | `response.coulomb.matrix`, `response.exchange.matrix` |
| `cd_ri_ao2mo_corpus` (`ri-kao2mo`) | Zhou et al., *J. Chem. Theory Comput.*, DOI [10.1021/acs.jctc.4c01245](https://doi.org/10.1021/acs.jctc.4c01245), Eq. (43) | `df.mo_vir_left`, `df.mo_vv_right` |
| `cd_x2camf_ccsd_corpus` (`cd-x2ccsd`) | Zhang et al., *J. Chem. Theory Comput.*, DOI [10.1021/acs.jctc.3c01236](https://doi.org/10.1021/acs.jctc.3c01236), Eqs. (5)-(12) | `ao.ladder`, `t2.ring_t1t1`, `wmbej.tau`, `t2.oooo` |
| `cd_x2camf_eom_ccsd_corpus` (`cd-x2ceom`) | same paper, Eqs. (13)-(16), with Epifanovsky et al., *J. Chem. Phys.* **139**, 134105 (2013) | `sigma2.wovvo`, `quartic.response_oovv`, `quartic.tau` |
| `complex_ri_ghf_gks_corpus` (`ri-ghf`) | Armbruster, Weigend, van Wüllen & Klopper, *Phys. Chem. Chem. Phys.* **10**, 1748 (2008), DOI [10.1039/b717719d](https://doi.org/10.1039/b717719d) | `exchange.metric`, `exchange.matrix`, `exchange.occupied_transform` |
| `pbc_r_ri_bse_corpus` (`ri-krbse`) | Zhou et al., Eqs. (45)-(46) | `bse.coulomb`, `bse.screened`, `response.static` |
| `pbc_r_ri_gdf_hf_corpus` (`ri-krhf`) | arXiv [2607.28554](https://arxiv.org/abs/2607.28554) | `exchange.matrix`, `coulomb.aux_density`, `coulomb.matrix` |
| `pbc_r_ri_gw_corpus` (`ri-krgw`) | Zhu & Chan G0W0 route, DOI [10.1021/acs.jctc.0c00704](https://doi.org/10.1021/acs.jctc.0c00704) | `screened.left`, `sigma.contract` |

The uniqueness rule's three resolutions are pinned by `corpus/tests.rs`:
`somf/somf.exchange.first` is `jk`'s `response.exchange.matrix` verbatim and is
backfilled by `somf`'s sequence-7 `cpscf.pseudo_density_virtual`;
`somf/cpscf.laplace_right` is the A/B-swapped form of `cpscf.laplace_left` and is
backfilled by sequence-8 `cpscf.pseudo_density_occupied`;
`cd_x2camf_ccsd_corpus/t2.ring_t2` is the A/B-swapped form of
`cd_x2camf_eom_ccsd_corpus/sigma2.wovvo` and is backfilled by sequence-5
`t2.oooo`. `pbc_r_ri_bse_corpus` keeps **both** of its rung-1 rows,
`bse.coulomb` (`Pia,Pjb->iajb`) and `bse.screened` (`Qij,Qab->ijab`): they share
the `o^2*v^2*naux` MAC count but not their operand domains, so they are
different contractions; there is no backfill, and `bse.screened_left` is not
selected.

**The transcription itself is not machine-checked here.** The corpus tables this
was copied from (`out/complex_corpus/` in the benchmark job) are not in this
repository, and `corpus/tests.rs` compares the literals against a second copy of
the same table rather than against a paper. Paper fidelity — that
`t2.wvoov_exchange` is the equation Hirata et al. print, with these index roles
— has to be read against the cited paper; the module doc in `corpus.rs` names
the source for each corpus so that reading is possible.

Index domains are occupied `o`, virtual `v`, density-fitting or Cholesky `naux`,
scalar atomic orbital `ao`, and the spinor-AO basis `nso = 2*ao`. The
two-component corpora count `o`/`v` in complex spinors, so their scalar-AO basis
is `(o+v)/2`; `o` is even at every ratio, so `o+v` is even and `nso = o+v =
2*ao` holds exactly. No selected case carries a packed pair domain, a Cartesian
`component` extent or a momentum `g` extent: those are corpus metadata, and
k-point multiplicity is call metadata, never an extent. One selected case is
**batched**: `ri-krgw/sigma.contract` (`mnQ,nQl->mnl`) carries the batch index
`n` in both operands and the output, so the suite is not uniformly unit-batch.
The harness times the whole batched call; the shared `mnk()` folds the batch
extent into `m` (that cell reports `m = ao²`, `n = ao`, `k = naux`, and
`m·n·k` still equals the MAC count).

The three extent sets are aspect-ratio classes, not molecules. The ratios v/o =
3, 10, 20 stand for the cc-pVDZ, cc-pVTZ and cc-pVQZ rung medians of a
5,239-molecule NAO statistics panel (3.90, 10.17, 20.36), so each ratio stands
for one basis-quality rung without pinning a molecule. `v = f·o`,
`nao = o+v` for the spatial family and `(o+v)/2` for the spinor family, and
`naux = round(k_rung·nao)` with the panel's rung medians of naux/nao (3.747,
2.568, 2.237). `o` is the largest **even** value under the budget, because
`o+v = nso = 2·ao` must be even and an odd `o` would break that at f = 10 and
f = 20. Family IDs are `<ratio>/<tag>.<case-id>`, giving 35 cases × 3 ratios =
105 families:

| Ratio | Rung it stands for | o | v | spatial nao | spatial naux | spinor nao | spinor nso | spinor naux | binding A+B+C |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ov_03` | cc-pVDZ (v/o 3.90, naux/nao 3.747) | 42 | 126 | 168 | 629 | 84 | 168 | 315 | 11.87 GiB |
| `ov_10` | cc-pVTZ (v/o 10.17, naux/nao 2.568) | 14 | 140 | 154 | 395 | 77 | 154 | 198 | 8.38 GiB |
| `ov_20` | cc-pVQZ (v/o 20.36, naux/nao 2.237) | 8 | 160 | 168 | 376 | 84 | 168 | 188 | 11.87 GiB |

The extents are derived from a *whole-binary* memory budget: the largest c64
A+B+C footprint over the suite — bound by the `ao^4` exchange kernel
`somf/cpscf.exchange_response` — stays within 12 GiB per cell, so no case is
ever skipped for memory. That is the **operand footprint, not the process
peak**: the harness holds one extra output-sized reference for its cross-arm
check, so a build **without** the `blas` feature peaks at **A+B+2C ≈ 11.9 GiB**
on the binding family — the binding output here is tiny (`ao^2`), unlike
`qcbench`'s triples output. A build **with** `blas` peaks higher, and
unconditionally so: the shared verifier runs its TTGT arm whatever engines the
caller selected (`engines/verify.rs:201`, inside `#[cfg(feature = "blas")]`),
allocating a plan, a scratch and an output-sized buffer while both operands and
the reference are live — **2A+2B+3C ≈ 23.7 GiB on the same family**, so that
build needs a host above 24 GiB. The campaign builds `BENCH_FEATURES=tblis`,
never `blas`. `corpus/tests.rs` pins the selection, the derivation rules, the
even-`o` rule and the 12 GiB footprint budget.

## Engines and correctness

- `plan`: tcontract's planner, with the same `TCBENCH_*` knobs as `tcbench`.
- `tblis`: actual C++ TBLIS through the direct FFI adapter (`--features tblis`).
- The forced packed route and TTGT remain available to the shared verification
  implementation but are not published `zqcbench` engine rows.

`verify` evaluates all selected families. At realistic extents it compares full
outputs from the planner, the forced packed route and the built baselines using
relative Frobenius error, requiring finite error at most `1e-10` for c64. It is a
cross-provider check, not a naive-loop recomputation of the expensive
contractions. Timed runs also compare each later engine with the first engine's
full output and fail on `MISMATCH`.

The driver accepts exactly `--dtype c64`; the benchmark is complex-orbital
throughout, and an `f64` request is rejected rather than skipped. The shared
verifier is not c64-only: before the corpus loop it always runs its known-value
GEMM (`A = B = 1`, every `D` element exactly `K`) for both `f64` and
`Complex<f64>` (`engines/verify.rs:66-67`, unconditional), so `zqcbench verify`
executes one `f64` gemm that no case in this corpus declares — the mirror of
`qcbench`'s stray `c64` row. `--n 1` is solely the timing-v2 identity sentinel;
`--batch 1` is the declared unit batch, not the contraction's batch (see
`ri-krgw/sigma.contract` above). Layout stress is fixed to `none`.

## Build and run

Use the same native release profile and TBLIS prefix for every arm:

```sh
export TBLIS_ROOT=/path/to/tblis-v2.0-beta2
export LD_LIBRARY_PATH="$TBLIS_ROOT/lib:$TBLIS_ROOT/lib64"
export RUSTFLAGS='-C target-cpu=native'
export CARGO_BUILD_JOBS=4
cargo build --release -p tcontract-bench --bin zqcbench --features tblis
```

Correctness precedes timing. Pin every process, including 1T:

```sh
benchmarks/scripts/pinned.sh 0 -- target/release/zqcbench verify \
  --threads 1 --dtype c64 --engines plan,tblis
benchmarks/scripts/pinned.sh 0-3 -- target/release/zqcbench verify \
  --threads 4 --dtype c64 --engines plan,tblis
```

A small timing smoke test, not a published result:

```sh
benchmarks/scripts/pinned.sh 0 -- target/release/zqcbench run \
  --threads 1 --case ov_10 --dtype c64 --engines plan,tblis \
  --prime-ms 1500 --reps 2
```

Campaign runs belong in `tcontract-benchmark`, not this source checkout; that
repository owns the revision pin, the suite declaration and the published pages.
timing-v2 requires clean committed sources.
