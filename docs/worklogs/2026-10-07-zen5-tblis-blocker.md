# Zen5 TBLIS follow-up: compatibility blocker

User requested a real Zen5-kernel rerun of the fixed 1/4/8/12T native suite.
No new performance measurements have been made. The previous Zen3 results
remain unchanged; `-march=native` is not evidence of selecting a Zen5 kernel.

## Verified source evidence

- Mainline BLIS HEAD `a8037cfc4330030c726efaec1ffeb1f93d468aa3`
  (https://github.com/flame/blis) has no `config/zen5`. It has the newer BLIS
  plugin infrastructure used by TBLIS 2.0.
- AMD AOCL-BLAS HEAD `25cad99a6840855ade0a49871197f48ee0e1d317`
  (https://github.com/amd/blis, tag 5.3.2) has `config/zen5`.
  `config/zen5/bli_cntx_init_zen5.c` selects AVX512 kernels: f64
  `bli_dgemm_zen4_asm_8x24`, c64 `bli_zgemm_zen4_asm_12x4` (reuse of Zen4
  kernel names inside the actual Zen5 context), with Zen5-specific tuning.
  It does **not** contain `configure-plugin`, `bli_gks_register_ukr`, or the
  new `bli_cntx_set_ukr` API used by TBLIS 2.0's plugin. A Zen5 configuration
  is available here, but this is not a drop-in replacement for the BLIS
  used by the measured TBLIS 2.0.
- Measured TBLIS source `20cc0bcdb13ddf9fbf619138b6081a16e9e48b7f`
  requires `share/blis/configure-plugin` and calls `bli_gks_register_ukr`
  in `tblis/plugin/bli_plugin_tblis.cxx`. Current TBLIS HEAD
  `eb719e718976572e0ab53975f4e0c799faeb35f2` retains this plugin requirement.

Source checkouts are `/tmp/tcontract-blis-zen5-upstream`,
`/tmp/tcontract-blis-zen5-amd`, `/tmp/tcontract-tblis-zen5-current`.

## Attempt and outcome

Tried the existing TBLIS source with CMake
`-DFETCHCONTENT_SOURCE_DIR_BLIS=/tmp/tcontract-blis-zen5-amd`
`-DBLIS_CONFIG_FAMILY=zen5 -DBLIS_THREAD_MODEL=pthread`
`-DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS='-O3 -march=native'`
`-DCMAKE_CXX_FLAGS='-O3 -march=native'`
and separate build/install paths `/tmp/tcontract-tblis-zen5-build` /
`/tmp/tcontract-tblis-zen5-install`.

CMake stopped first because the AMD source has its own CMake project requiring
Fortran, which this host lacks. Log: `/tmp/tcontract-tblis-zen5-configure.log`.
Installing a Fortran compiler or building AMD BLIS separately would bypass
this first problem, **not** supply the missing TBLIS plugin API above.
No system installs, kernel ports, compatibility shims or changes to the
previous baseline were made.

## Unresolved scope choice

A true Zen5 TBLIS 2.0 run requires a compatible implementation/port of the
AOCL Zen5 kernels into modern BLIS (or a compatible TBLIS/AOCL adapter),
which is substantial work and cannot be disguised as a benchmark flag.
Bounded alternatives, needing user selection:

1. Add AOCL-BLAS `zen5` as a separately labelled transpose/GEMM tensor
   contraction baseline; it would not be TBLIS.
2. Measure TBLIS with mainline BLIS's `skx` AVX512 kernels on this Zen5 CPU;
   it would be AVX512, **not** a Zen5-specific BLIS configuration.

Prior advice implying a Zen5-tuned TBLIS could be obtained just by selecting
an available configuration was too optimistic. The benchmark requirement is
blocked, not completed, and there are no Zen5-TBLIS numbers to report.
