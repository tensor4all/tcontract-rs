# C ABI benchmark

Per-call cost of `libtcontract_capi` from C against the same calls made directly
from Rust (`tcontract-bench` binary `capi_rust`): an empty call
(`tcontract_abi_version`) and a 2x2x2 contraction through a
prebuilt TAPP product (`TAPP_execute_product`), each on an explicit serial
executor (`nthreads == 1`: no workers) and on a 4-thread pool. Each sample
is the mean of 1000 calls; the median of 101 samples is reported.

```bash
benchmarks/c/run.sh 57 57-60 /tmp/capi   # 1T cores, 4T cores, output
```

## Observation, 2026-09-30 (single run, not a claim; of the removed API)

The 8x8 GEMM row measured the removed BLAS C entry point, which went with its C ABI crate
([#37](https://github.com/tensor4all/tcontract-rs/issues/37)); the
benchmark no longer has that case.

This table was measured with the previous C ABI (`tprims_exec*`,
`tprims_contract_plan_execute`). `bench.c` now drives TAPP and has been
smoke-run only after the redesign ([#26](https://github.com/tensor4all/tcontract-rs/issues/26));
the paired old-versus-new 1T/4T measurement with an A/A noise floor is
deferred, so no overhead change is claimed.

- tcontract-rs `509f30f` (clean tree), release profile, rustc 1.97.1, gcc -O2.
- AMD EPYC 7713P, cores 57-60 idle before the run. Raw output:
  [results/](results/).

| Case | C 1T | Rust 1T | C 4T | Rust 4T |
| --- | ---: | ---: | ---: | ---: |
| empty call | 2.9 ns | 0.3 ns | 3.0 ns | 0.3 ns |
| 8x8 GEMM | 430 ns | 145 ns | 427 ns | 120 ns |
| 2x2x2 contraction (plan) | 661 ns | 264 ns | 716 ns | 266 ns |

Closing the 4-thread pool (joining its workers) took 99 us.

### Findings

1. **The C boundary adds about 300-450 ns per small call**, 3x the Rust call
   for an 8x8 GEMM. This fails Prototype 4's gate ("C-side fixed cost within
   a small margin of the Rust call") for tiny operations. *Hypothesis, not
   profiled:* per-operand metadata allocations (the DLPack layout's `Vec`s and
   the `Arc` dims/strides inside `strided_view::StridedView`, about ten per
   GEMM call). `strided-view`'s borrowed-metadata `RawStridedRef` avoids
   them; accepting it in `tcontract` is the follow-up.
2. Small calls never enter the pool (4T rows equal 1T rows), as designed.
   A malloc-counting interposer (Phase 1f review) measured about 13
   allocations per 8x8 GEMM call from C, supporting the hypothesis in 1.

Comparison caveats: the "empty call" Rust row reads a constant through
`black_box`, not a function call, so it only bounds the C call's cost from
below; the C 1T rows use a serial executor while the Rust 1T rows use a
one-thread borrowed pool (both run serially on the caller); the Rust GEMM and
contraction rows build their input views once, outside the loop, while the C
calls rebuild them on every call (that is part of the measured boundary cost).
3. Zero copy holds: the C test checks that results land in the caller's
   buffers for row-major (NULL strides), column-major and strided inputs.
