# QC-realistic binary-contraction benchmark (real orbitals, f64)

`qcbench` reuses `tcbench`'s execution engines, timing, TBLIS adapter, thread enforcement, and evidence format. It replaces isotropic TCCG sizing with fixed occupied/virtual/auxiliary extents at quantum-chemistry aspect ratios. Every operand is contiguous column-major, `alpha = 1`, `beta = 0`, and input generation and plan construction are outside the timed call.

## Corpus

Twenty-four binary contractions selected from nine corpus transcriptions of the QC papers (source material in the benchmark job's `out/real_corpus/`, one `CORPUS.md` per corpus). Selection is uniform: per corpus, every binary of the two most expensive formal-scaling rungs (the first two distinct `macs` monomials of the corpus's own sequence order).

| Corpus | Paper | Cases |
|---|---|---|
| RCCSD energy + T1/T2 update | Scuseria, Janssen & Schaefer, *J. Chem. Phys.* **89**, 7382 (1988), DOI [10.1063/1.455269](https://doi.org/10.1063/1.455269), Table I/II | `b_tau.A`, `b_tau.S` (packed `vroot`/`vsym`/`oroot`) |
| RCCSD(T) (T) energy | Scuseria, *J. Chem. Phys.* **94**, 442 (1991), DOI [10.1063/1.460359](https://doi.org/10.1063/1.460359), Eqs. (1)-(6) | `T.connected.virtual`, `T.connected.occupied` |
| EOM-IP-CCSD | Pieniazek, PhD thesis (USC, 2008), Tables C.1-C.3 | `I6.t1_oovv`, `sigma2.wovvo`, `sigma2.woooo` |
| EOM-EA-CCSD | Nooijen & Bartlett, *J. Chem. Phys.* **102**, 3629 (1995), DOI [10.1063/1.468592](https://doi.org/10.1063/1.468592), Eqs. (37)-(38) | `sigma2.w_abcd`, `sigma2.w_alcj`, `sigma2.w_bljc`, `sigma2.w_lbdj` |
| CCSD analytic gradient | Salter, Trucks & Bartlett, *J. Chem. Phys.* **90**, 1752 (1989), DOI [10.1063/1.456069](https://doi.org/10.1063/1.456069), Tables III/IV + Eqs. (21), (25), (26), (33), (34) | `gamma.abcd` (packed `ao_root`/`vroot`), `gamma.aibc` |
| RI-HF energy | Barca et al., [arXiv:2407.19614](https://arxiv.org/abs/2407.19614), Eqs. (4)-(18) | `df.factor`, `exchange.matrix`, `exchange.occupied_transform` |
| RI-HF gradients | Barca et al., [arXiv:2407.19614](https://arxiv.org/abs/2407.19614), Eqs. (20), (26)-(27), (31)-(35) | `gradient.Y_transform2`, `gradient.X`, `gradient.Z_pair` |
| TDHF/CIS response | Mester & Kállay, *J. Chem. Theory Comput.* **15**, 1690 (2019), DOI [10.1021/acs.jctc.8b01199](https://doi.org/10.1021/acs.jctc.8b01199), Table 1 + Eq. (12) | `response.exchange.occupied_half`, `response.exchange.transform`, `response.exchange.metric` |
| MO RI-J/K response | Neese & Olbrich, *Chem. Phys. Lett.* **362**, 170 (2002), DOI [10.1016/S0009-2614(02)01053-9](https://doi.org/10.1016/S0009-2614(02)01053-9), Eqs. (23)-(24), (30)-(31) | `K.aux`, `K.matrix` |

Index domains are occupied `o`, virtual `v`, density-fitting `naux`, atomic orbital `ao = o + v`, and the packed pair domains the papers themselves use (`oroot`, `vroot`, `vsym`, `ao_root`). Packed cases stay packed exactly as their corpus declares them, so their MACs and footprints are the packed products — `gamma.abcd`, the rank-1 head of the gradient corpus, is the largest single MAC count in the suite.

The three extent sets are aspect-ratio classes, not molecules. The ratios v/o = 3, 10, 20 are the median v/o of the cc-pVDZ, cc-pVTZ and cc-pVQZ rungs of a 5,239-molecule NAO statistics panel (3.90, 10.17, 20.36), so each ratio stands for one basis-quality rung without pinning a molecule. Family IDs are `<ratio>/<case-id>`, giving 72 families:

| Ratio | Rung median it stands for | o | v | naux |
|---|---|---:|---:|---:|
| `ov_03` | cc-pVDZ (v/o 3.90, naux/nao 3.747) | 18 | 54 | 270 |
| `ov_10` | cc-pVTZ (v/o 10.17, naux/nao 2.568) | 10 | 100 | 282 |
| `ov_20` | cc-pVQZ (v/o 20.36, naux/nao 2.237) | 7 | 140 | 329 |

Each ratio's extents follow the rung-median rule `naux = round(k_rung · (o+v))`, and `v·o` stays nearly constant (972 / 1000 / 980): a larger basis rung affords more virtuals but fewer occupied orbitals. The extents are derived from a *whole-binary* memory budget: the largest f64 A+B+C footprint over the suite — bound by the `o³v³` triples output of `T.connected.virtual` — stays within 8 GiB per cell, so no case is ever skipped for memory. That is the **operand footprint, not the process peak**: the harness holds one extra output-sized reference for its
cross-arm check, so a build **without** the `blas` feature peaks at **A+B+2C — 14.98 GiB for the binding family — and a
recording host needs above 16 GiB**. A build **with** `blas` peaks higher, and unconditionally so: the shared
verifier runs its TTGT arm whatever engines the caller selected (`engines/verify.rs:201`, inside
`#[cfg(feature = "blas")]`), allocating a plan, a scratch and an output-sized buffer while both operands and the
reference are live — **2A+2B+3C ≈ 22.5 GiB on the same family**, so that build needs a host above 24 GiB. The
bound below is the `blas`-free one. `corpus/tests.rs` pins the selection, the derivation rules, and the footprint budget.

## Engines and correctness

- `plan`: tcontract's planner, with the same `TCBENCH_*` knobs as `tcbench`.
- `tblis`: actual C++ TBLIS through the direct FFI adapter (`--features tblis`).
- The forced packed route and TTGT remain available to the shared verification implementation but are not published qcbench engine rows.

`verify` evaluates all selected families. At realistic extents it compares full outputs from the planner, forced packed route, and built baselines using relative Frobenius error, requiring finite error at most `1e-10` for f64. It is a cross-provider check, not a naive-loop recomputation of the expensive QC contractions. Timed runs also compare each later engine with the first engine's full output and fail on `MISMATCH`.

Only f64 and unit batch are accepted; the benchmark is real-orbital throughout, and `c64` is rejected rather than skipped. `--n 1` is solely the timing-v2 identity sentinel; it does not resize the tensors. Layout stress is fixed to `none`.

## Build and run

Use the same native release profile and TBLIS prefix for every arm:

```sh
export TBLIS_ROOT=/path/to/tblis-v2.0-beta2
export LD_LIBRARY_PATH="$TBLIS_ROOT/lib:$TBLIS_ROOT/lib64"
export RUSTFLAGS='-C target-cpu=native'
export CARGO_BUILD_JOBS=4
cargo build --release -p tcontract-bench --bin qcbench --features tblis
```

Correctness precedes timing. Pin every process, including 1T:

```sh
benchmarks/scripts/pinned.sh 0 -- target/release/qcbench verify \
  --threads 1 --dtype f64 --engines plan,tblis
benchmarks/scripts/pinned.sh 0-3 -- target/release/qcbench verify \
  --threads 4 --dtype f64 --engines plan,tblis
```

A small timing smoke test, not a published result:

```sh
benchmarks/scripts/pinned.sh 0 -- target/release/qcbench run \
  --threads 1 --case ov_10 --dtype f64 --engines plan,tblis \
  --prime-ms 1500 --reps 2
```

Campaign runs belong in `tcontract-benchmark`, not this source checkout; that repository owns the revision pin, the suite declaration and the published pages. timing-v2 requires clean committed sources.
