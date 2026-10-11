# OpenBLAS kernel-import experiment

User explicitly authorizes importing an OpenBLAS kernel with its license.
The optimization being tested is reuse of the actual f64 kernel selected for
Zen5, not a fictitious Zen5-exclusive source file.

## Source and correctness, before timing

OpenBLAS `31e82fa8c509e6f0d96288de3c20d8916d894e72` at
https://github.com/OpenMathLib/OpenBLAS. Zen5 maps to Cooperlake when BF16
is available, otherwise SkylakeX/Zen according to ISA. The f64 kernel is
`kernel/x86_64/dgemm_kernel_16x2_skylakex.c`. It is copied byte-for-byte into
`experiments/openblas-kernel/c/`, verified by cmp. Root BSD-3-Clause LICENSE
is copied verbatim; the source file has no separate notice. Project-owned
shim/wrapper remain MIT OR Apache-2.0 with both full license texts retained.
Detailed record is the experiment's PROVENANCE.md.

The C function is compiled with system cc `-O3 -march=native`, a project-owned
int64 BLASLONG shim and CNAME rename. KernelCatalog registers an external
MR16/NR2 scratch-tile family: existing tcontract packers, driver, scheduling,
writeback and caller pool are reused. No production default changes or
translation of assembly. Zeroing the tile is an explicit adapter cost because
OpenBLAS accumulates into C. Source's wider m16n12 path uses B packed in pairs
rather than tcontract's NR-wide k-major panels, so the minimal import does not
claim to preserve OpenBLAS's fully optimized full-GEMM blocking/tile usage.
A negative result applies to this adapter, not all possible kernel ports.

Runnable selfcheck passed K=0,1,2,3,7,16,33, exact small integers, matrix
edge tiles33x17x21, regular16x4x16 and row-major-A9x5x13 against an independent
reference. The actual-budget benchmark verifies full outputs with finite
relative Frobenius residual<=1e-10. Clippy and formatting passed.

## Fixed experiment protocol (before timings)

Same native-release executable contains both default forced-packed tcontract
and imported OpenBLAS family. All f64 cases: gemm-512-col512^3,
gemm-1024-col1024^3, strided-1024-rowa1024^3 (A row-major, B/D column-major).
No post-hoc filtering. Both arms share identical deterministic data,
alpha=1,beta=0, input allocations/layouts. Planning/pool/input setup excluded;
packing, writeback, adapter tile zeroing and entry included. Host scratch/pool
retained through existing BenchThreads, no ambient pool. Rust flags
`-C target-cpu=native`, release opt3/thinLTO/codegen-units1.

1T=CPU4,4T=4-7,8T=4-11,12T=0-11; no SMT,12T cross-L3 labelled.
Verify each budget before any timing. One warm-up per arm, five repetitions,
best wall time (arms interleaved in each process); full set A then A2 with
120-second separation. Run through pinned.sh (3-second windows,<=5% busy,
three attempts), no concurrent builds/other benchmarks. Retain all outputs.
If a whole-set validity gate fails, mark it inconclusive and at most one
complete rerun, never selectively retry a favorable case.

Adoption gate: numerical gates pass, all case/thread combinations at least
nonregressing within both A/A spreads, geometric-mean whole-call improvement
>=10% exceeding noise. Any slow case is retained and inspected. No claim that
this establishes performance on arbitrary tensor ranks, c64 or other CPUs.
Raw observations: `experiments/openblas-kernel/results/2026-10-07/`.

## Minimal adapter result and wider-path protocol

The complete16x2 adapter experiment passed all4-budget guards/checks; every
case was slower, with tcontract/OpenBLAS time ratios0.606-0.698. A/A spreads
were generally<2% (12T imported1024 GEMM8.8% maximum). This minimal import
fails its adoption gate. It cannot stand in for OpenBLAS's wider n12 path.

Before timing a separately labelled wide variant, implemented an MR16/NR12
family with B k-major-to-pair-panel repacking inside the tile callback.
Fixed stack workspace for kc<=256, no per-tile allocation or unconditional
workspace zeroing; above256 it uses the existing portable kernel rather
than violating the descriptor contract. Output tile zeroing remains because
OpenBLAS accumulates. KC seed128 (vs minimal256), MC256,NC1536; all adaptation
cost is included. Same C source, no changes to its assembly or project packers.
Selfchecks pass kc0/1/3/7/128/256/257, including the fallback, and full matrix
verification passed at8T. This is a separate adapter/blocking experiment,
not a controlled test of only register width.

Wide variant repeats **the identical3-case, f64,1/4/8/12T,5-rep,A/A protocol
and adoption criteria above**, with --wide, no exclusions, under
`results/2026-10-07-wide/`. No previous16x2 measurement is replaced.
Any kernel-family adoption beyond this experiment requires a full strided
contraction suite;3 GEMM cases alone do not authorize a production default.

## Wider-path result: reject both adapters

The first complete wide experiment failed a12T idle gate; retain the entire
attempt as `results/2026-10-07-wide-invalid/`. One complete repeat of the
predeclared experiment passed all numerical and idle checks. All3 cases and
all4 budgets are included. Ratios below are geometric means of paired A/A
best wall times, tcontract time divided by imported-family time (<1 is slower):

| T | 512 column GEMM | 1024 column GEMM | 1024 row-major A |
|---|---:|---:|---:|
|1|0.763|0.737|0.740|
|4|0.783|0.750|0.748|
|8|0.758|0.805|0.802|
|12|0.781|0.666|0.649|

The wide path improves upon the minimal adapter but still loses on every
case. At1/4/8T spreads are<=1.8%, much smaller than these losses.12T's
512 default-tcontract A/A spread is30.8% despite idle guards passing; retain it
as instability, not proof of the precise12T ratio. Other12T wide spreads
include7.5% for1024 GEMM. Adoption fails independently of these noisy rows
because the stable1/4/8T cases consistently regress. Neither adapter is
installed as a production default. This is evidence about these full-call
adapters including B repacking and different blocking, not a claim that the
OpenBLAS assembly itself is intrinsically slower on Zen5.


