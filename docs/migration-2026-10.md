# Migration guide, October 2026 source integration

Tracking issue: [#37](https://github.com/tensor4all/tcontract-rs/issues/37). This file records what a
consumer of the old crates, family ids, environment variables and C symbols has to change. Each
integration PR extends it; this version covers PR 1 (removals), PR 2 (kernel consolidation), PR 3 (contract
consolidation) and PR 4 (C ABI, testkit and benchmarks, and the dissolution of `tensorprimitives/`).

## Crates

| Old | New |
|---|---|
| `tprims-gemm-kernel`, `tprims-kernel-tensorcontract`, `tprims-kernel-cplx` | `tcontract-kernel` (one crate; Lukas Devos's files keep their history and headers) |
| `tprims-custom-kernel-test` | `tcontract_testkit::custom_kernels` plus tests in `tcontract` |
| `tprims-linalg`, `tprims-blas-capi`, `tprims-kernel-gemm`, `tprims-kernel-pgx86` | removed in PR 1 (no retained consumer) |
| `tensorcontract` (the packed driver and its labels-based `Plan`), `tprims-contract-traits`, the old `tprims-contract` | `tcontract` (one crate: `api`, `plan`, `driver`, `strategy`; see PR 3 below) |
| `tprims-blas` | removed in PR 3: GEMM, batched GEMM and grouped GEMM are contractions; `trsm` is gone |
| `tprims-core`, `tprims-bundle`, `tensorprimitives-tapp` | `tcontract-capi` (PR 4): one crate building `libtcontract_capi`; features `blas` and `tapp` are gone, the TAPP surface is always present |
| `tprims-contract-testkit` | `tcontract-testkit` (label oracle, seeded fixtures, naive second backend, custom-kernel fixtures) |
| `tensorprimitives-bench` (`tcbench`) | `benchmarks/` (`tcontract-bench`, binary `tcbench`) |
| the `tensorprimitives/` directory | dissolved in PR 4: docs in `docs/archive/tensorprimitives/`, licences at the root, the rest deleted (see [provenance](provenance.md)) |

Module paths: `tcontract_gemm_kernel::cache` is `tcontract_kernel::blocking`; `tcontract_kernel_tensorcontract::{scalar, x86, aarch64}`
are `tcontract_kernel::kernels::{reference::scalar, x86, aarch64}`; the `KernelSet` trait is gone (PR 3); the workspace provider moved to `tcontract_exec`.
The built-in families need no registration call: `register()` of the old provider crates is gone and
`tcontract_kernel::register` remains only for explicit, unsafe external manifests.

## Source moves

`git log --follow` cannot describe a split, so the non-trivial moves are recorded here (all in PR 2; the
pure-rename commit is `Move the kernel crates into crates/tprims-kernel`, the splits are separate commits).

| Old path | New path |
|---|---|
| `crates/tprims-gemm-kernel/src/cache.rs` | `crates/tcontract-kernel/src/blocking/probe.rs` (hierarchy types, probes, `l3_domains`) and `blocking/model.rs` (`BlockModel`, `PanelGeom`, the analytical model) |
| `crates/tprims-kernel-tensorcontract/src/x86.rs` | `kernels/x86/mod.rs` (ISA enum, detection, dispatch), `kernels/x86/avx2.rs`, `kernels/x86/avx512.rs` |
| `crates/tprims-kernel-tensorcontract/src/aarch64.rs` | `kernels/aarch64/neon.rs` |
| `crates/tprims-kernel-tensorcontract/src/lib.rs` | `kernels/kernel_set.rs` in PR 2, then `kernels/menu_tests.rs` in PR 3: the public `KernelSet` trait is deleted; its dispatch survives as a private test trait carrying the kernel-vs-reference contract tests |
| `crates/tprims-kernel-tensorcontract/src/families.rs` | `kernels/mod.rs` (built-in lists, id function) |
| `crates/tprims-kernel-tensorcontract/src/{scalar,simd}.rs` | `kernels/reference/scalar.rs`, `kernels/macros.rs` |
| `crates/tprims-kernel-cplx/src/{lib,avx2}.rs` | `kernels/cplx.rs`, `kernels/x86/avx2_complex.rs` |
| `crates/tprims-gemm-kernel/src/{family,types,element,cpu}.rs` | `abi/` |
| `crates/tprims-gemm-kernel/src/{registry,resolved,custom,partition}.rs` | `select/{registry,resolve,catalog,partition}.rs` |
| `crates/tprims-gemm-kernel/src/{pack,scatter,writeback}.rs` | `pack/` |
| `crates/tprims-gemm-kernel/src/{portable,induced}.rs` | `kernels/reference/{portable,induced}.rs` |
| `crates/tprims-custom-kernel-test/src/lib.rs` | `crates/tcontract-testkit/src/custom_kernels.rs` |
| `crates/tprims-custom-kernel-test/tests/selector_blas.rs` | deleted with `tprims-blas` in PR 3 (the selector contract is covered by `tcontract/tests/custom_selector.rs`) |
| `crates/tprims-custom-kernel-test/tests/{selector_contract,concurrency,steady_state_alloc}.rs` | `crates/tcontract/tests/custom_{selector,concurrency,steady_state_alloc}.rs` |

## Family ids

Built-in ids are `{isa}.{dtype}.{scheme}.{MR}x{NR}`, from the descriptor's storage dtype and **logical**
tile geometry. Provenance is carried by `Origin`, not by the id.

| Field | Values |
|---|---|
| isa | `ref`, `avx2`, `avx512`, `neon` (the full CPU-feature mask stays in the descriptor) |
| dtype | storage dtype `f32` `f64` `c32` `c64` |
| scheme | real: `real` (SIMD and portable), `real-scalar` (the scalar menu's real family, distinct from the portable one of the same geometry); direct update: `direct`, `direct-b`; complex: `planar`, `native`, `1m`, `3m`, `4m`; induced: `i1m`, `i4m`, with a `-scalar` suffix when the base is the scalar-menu family |

Induced families take the logical geometry of the induced tile: 1m halves the base real `MR`, 4m keeps it.
Induced families of a **caller's** base family keep the old `<base>.1m-induced` / `<base>.4m-induced` suffix, and
external catalog ids stay provider-owned opaque names. Priorities, tie order and `allow_auto` are unchanged;
the order of the tables below is the Auto order. The scheme is injective per dtype (a test pins this and
the whole x86-64 manifest, `crates/tcontract-kernel/tests/snapshots/x86_64_manifest.txt`).

### x86-64 (complete; unavailable ISAs included)

**f32**

| Old | New |
|---|---|
| `tc.avx512.f32.48x8` | `avx512.f32.real.48x8` |
| `tc.avx512.f32.32x8` | `avx512.f32.real.32x8` |
| `tc.avx512.f32.16x10` | `avx512.f32.real.16x10` |
| `tc.avx2.f32.16x6` | `avx2.f32.real.16x6` |
| `tc.avx2.f32.24x4` | `avx2.f32.real.24x4` |
| `tc.avx2.f32.8x8` | `avx2.f32.real.8x8` |
| `tc.scalar.f32.4x4` | `ref.f32.real-scalar.4x4` |
| `portable.f32.4x4` | `ref.f32.real.4x4` |
| `portable.f32.4x4.direct` | `ref.f32.direct.4x4` |
| `portable.f32.4x4.direct-b` | `ref.f32.direct-b.4x4` |

**f64**

| Old | New |
|---|---|
| `tc.avx512.f64.24x8` | `avx512.f64.real.24x8` |
| `tc.avx512.f64.16x8` | `avx512.f64.real.16x8` |
| `tc.avx512.f64.8x8` | `avx512.f64.real.8x8` |
| `tc.avx2.f64.8x6` | `avx2.f64.real.8x6` |
| `tc.avx2.f64.12x4` | `avx2.f64.real.12x4` |
| `tc.avx2.f64.4x8` | `avx2.f64.real.4x8` |
| `tc.scalar.f64.4x4` | `ref.f64.real-scalar.4x4` |
| `portable.f64.4x4` | `ref.f64.real.4x4` |
| `portable.f64.4x4.direct` | `ref.f64.direct.4x4` |
| `portable.f64.4x4.direct-b` | `ref.f64.direct-b.4x4` |

**c32**

| Old | New |
|---|---|
| `tc.avx512.c32.planar.32x6` | `avx512.c32.planar.32x6` |
| `tc.avx512.c32.3m.16x10` | `avx512.c32.3m.16x10` |
| `tc.avx512.c32.planar.48x4` | `avx512.c32.planar.48x4` |
| `tc.avx512.c32.3m.32x4` | `avx512.c32.3m.32x4` |
| `tc.avx512.c32.planar.16x12` | `avx512.c32.planar.16x12` |
| `tc.avx512.c32.3m.48x3` | `avx512.c32.3m.48x3` |
| `tc.avx512.c32.planar.32x5` | `avx512.c32.planar.32x5` |
| `tc.avx512.c32.1m.32x6` | `avx512.c32.1m.32x6` |
| `tc.avx512.c32.1m.24x8` | `avx512.c32.1m.24x8` |
| `tc.avx512.c32.1m.16x8` | `avx512.c32.1m.16x8` |
| `tc.avx512.c32.1m.8x12` | `avx512.c32.1m.8x12` |
| `tc.avx2.c32.planar.8x5` | `avx2.c32.planar.8x5` |
| `tc.avx2.c32.3m.8x4` | `avx2.c32.3m.8x4` |
| `tc.avx512.f32.48x8.1m-induced` | `avx512.c32.i1m.24x8` |
| `tc.avx512.f32.48x8.4m-induced` | `avx512.c32.i4m.48x8` |
| `tc.avx2.c32.planar.16x2` | `avx2.c32.planar.16x2` |
| `tc.avx2.c32.3m.16x2` | `avx2.c32.3m.16x2` |
| `tc.avx512.f32.32x8.1m-induced` | `avx512.c32.i1m.16x8` |
| `tc.avx512.f32.32x8.4m-induced` | `avx512.c32.i4m.32x8` |
| `tc.avx512.f32.16x10.1m-induced` | `avx512.c32.i1m.8x10` |
| `tc.avx512.f32.16x10.4m-induced` | `avx512.c32.i4m.16x10` |
| `tc.avx2.c32.1m.8x6` | `avx2.c32.1m.8x6` |
| `tc.avx2.c32.1m.12x4` | `avx2.c32.1m.12x4` |
| `tc.avx2.c32.1m.4x8` | `avx2.c32.1m.4x8` |
| `cplx.avx2.c32.native.8x4` | `avx2.c32.native.8x4` |
| `tc.avx2.f32.16x6.1m-induced` | `avx2.c32.i1m.8x6` |
| `tc.avx2.f32.16x6.4m-induced` | `avx2.c32.i4m.16x6` |
| `tc.avx2.f32.24x4.1m-induced` | `avx2.c32.i1m.12x4` |
| `tc.avx2.f32.24x4.4m-induced` | `avx2.c32.i4m.24x4` |
| `tc.avx2.f32.8x8.1m-induced` | `avx2.c32.i1m.4x8` |
| `tc.avx2.f32.8x8.4m-induced` | `avx2.c32.i4m.8x8` |
| `tc.scalar.c32.planar.2x4` | `ref.c32.planar.2x4` |
| `tc.scalar.c32.3m.2x4` | `ref.c32.3m.2x4` |
| `portable.c32.native.4x4` | `ref.c32.native.4x4` |
| `tc.scalar.c32.1m.2x4` | `ref.c32.1m.2x4` |
| `portable.f32.4x4.1m-induced` | `ref.c32.i1m.2x4` |
| `portable.f32.4x4.4m-induced` | `ref.c32.i4m.4x4` |
| `tc.scalar.f32.4x4.1m-induced` | `ref.c32.i1m-scalar.2x4` |
| `tc.scalar.f32.4x4.4m-induced` | `ref.c32.i4m-scalar.4x4` |

**c64**

| Old | New |
|---|---|
| `tc.avx512.c64.planar.16x6` | `avx512.c64.planar.16x6` |
| `tc.avx512.c64.3m.8x10` | `avx512.c64.3m.8x10` |
| `tc.avx512.c64.planar.24x3` | `avx512.c64.planar.24x3` |
| `tc.avx512.c64.3m.16x4` | `avx512.c64.3m.16x4` |
| `tc.avx512.c64.planar.8x8` | `avx512.c64.planar.8x8` |
| `tc.avx512.c64.3m.24x3` | `avx512.c64.3m.24x3` |
| `tc.avx512.c64.1m.12x8` | `avx512.c64.1m.12x8` |
| `tc.avx512.c64.1m.16x6` | `avx512.c64.1m.16x6` |
| `tc.avx512.c64.1m.8x8` | `avx512.c64.1m.8x8` |
| `tc.avx2.c64.planar.4x5` | `avx2.c64.planar.4x5` |
| `tc.avx2.c64.3m.4x4` | `avx2.c64.3m.4x4` |
| `tc.avx512.f64.24x8.1m-induced` | `avx512.c64.i1m.12x8` |
| `tc.avx512.f64.24x8.4m-induced` | `avx512.c64.i4m.24x8` |
| `tc.avx2.c64.planar.8x2` | `avx2.c64.planar.8x2` |
| `tc.avx2.c64.3m.8x2` | `avx2.c64.3m.8x2` |
| `tc.avx512.f64.16x8.1m-induced` | `avx512.c64.i1m.8x8` |
| `tc.avx512.f64.16x8.4m-induced` | `avx512.c64.i4m.16x8` |
| `tc.avx512.f64.8x8.1m-induced` | `avx512.c64.i1m.4x8` |
| `tc.avx512.f64.8x8.4m-induced` | `avx512.c64.i4m.8x8` |
| `tc.avx2.c64.1m.4x6` | `avx2.c64.1m.4x6` |
| `tc.avx2.c64.1m.6x4` | `avx2.c64.1m.6x4` |
| `tc.avx2.c64.1m.2x8` | `avx2.c64.1m.2x8` |
| `cplx.avx2.c64.native.4x4` | `avx2.c64.native.4x4` |
| `tc.avx2.f64.8x6.1m-induced` | `avx2.c64.i1m.4x6` |
| `tc.avx2.f64.8x6.4m-induced` | `avx2.c64.i4m.8x6` |
| `tc.avx2.f64.12x4.1m-induced` | `avx2.c64.i1m.6x4` |
| `tc.avx2.f64.12x4.4m-induced` | `avx2.c64.i4m.12x4` |
| `tc.avx2.f64.4x8.1m-induced` | `avx2.c64.i1m.2x8` |
| `tc.avx2.f64.4x8.4m-induced` | `avx2.c64.i4m.4x8` |
| `tc.scalar.c64.planar.2x4` | `ref.c64.planar.2x4` |
| `tc.scalar.c64.3m.2x4` | `ref.c64.3m.2x4` |
| `portable.c64.native.4x4` | `ref.c64.native.4x4` |
| `tc.scalar.c64.1m.2x4` | `ref.c64.1m.2x4` |
| `portable.f64.4x4.1m-induced` | `ref.c64.i1m.2x4` |
| `portable.f64.4x4.4m-induced` | `ref.c64.i4m.4x4` |
| `tc.scalar.f64.4x4.1m-induced` | `ref.c64.i1m-scalar.2x4` |
| `tc.scalar.f64.4x4.4m-induced` | `ref.c64.i4m-scalar.4x4` |

### AArch64 (NEON)

The NEON menus (`cfg_neon_f64`, `cfg_neon_f32`) are compiled only on aarch64, so this table is derived from the
menu definitions and the same id function (unit-tested with NEON inputs); CI on macOS arm64 pins the
`f64` real list. The portable `ref.*` families of the x86-64 tables (`ref.f64.real.4x4`, `ref.f64.direct.4x4`,
`ref.c64.native.4x4`, ...) and the scalar menu are the same on every target.

**f64 / c64 (menu order; Auto order interleaves them by priority)**

| Old | New |
|---|---|
| `tc.neon.f64.16x3` | `neon.f64.real.16x3` |
| `tc.neon.f64.4x8` | `neon.f64.real.4x8` |
| `tc.neon.f64.8x6` | `neon.f64.real.8x6` |
| `tc.neon.f64.6x8` | `neon.f64.real.6x8` |
| `tc.neon.c64.planar.4x6` | `neon.c64.planar.4x6` |
| `tc.neon.c64.planar.4x5` | `neon.c64.planar.4x5` |
| `tc.neon.c64.planar.2x12` | `neon.c64.planar.2x12` |
| `tc.neon.c64.1m.2x8` | `neon.c64.1m.2x8` |
| `tc.neon.c64.1m.4x6` | `neon.c64.1m.4x6` |
| `tc.neon.c64.1m.3x8` | `neon.c64.1m.3x8` |
| `tc.neon.c64.3m.2x8` | `neon.c64.3m.2x8` |
| `tc.neon.c64.3m.4x3` | `neon.c64.3m.4x3` |
| `tc.neon.c64.3m.4x4` | `neon.c64.3m.4x4` |
| `tc.neon.f64.16x3.1m-induced` | `neon.c64.i1m.8x3` |
| `tc.neon.f64.16x3.4m-induced` | `neon.c64.i4m.16x3` |
| `tc.neon.f64.4x8.1m-induced` | `neon.c64.i1m.2x8` |
| `tc.neon.f64.4x8.4m-induced` | `neon.c64.i4m.4x8` |
| `tc.neon.f64.8x6.1m-induced` | `neon.c64.i1m.4x6` |
| `tc.neon.f64.8x6.4m-induced` | `neon.c64.i4m.8x6` |
| `tc.neon.f64.6x8.1m-induced` | `neon.c64.i1m.3x8` |
| `tc.neon.f64.6x8.4m-induced` | `neon.c64.i4m.6x8` |

**f32 / c32 (menu order; Auto order interleaves them by priority)**

| Old | New |
|---|---|
| `tc.neon.f32.8x8` | `neon.f32.real.8x8` |
| `tc.neon.f32.16x6` | `neon.f32.real.16x6` |
| `tc.neon.f32.24x4` | `neon.f32.real.24x4` |
| `tc.neon.f32.12x8` | `neon.f32.real.12x8` |
| `tc.neon.c32.planar.8x6` | `neon.c32.planar.8x6` |
| `tc.neon.c32.planar.8x5` | `neon.c32.planar.8x5` |
| `tc.neon.c32.planar.4x12` | `neon.c32.planar.4x12` |
| `tc.neon.c32.1m.8x6` | `neon.c32.1m.8x6` |
| `tc.neon.c32.1m.6x8` | `neon.c32.1m.6x8` |
| `tc.neon.c32.1m.12x4` | `neon.c32.1m.12x4` |
| `tc.neon.c32.3m.4x8` | `neon.c32.3m.4x8` |
| `tc.neon.c32.3m.8x3` | `neon.c32.3m.8x3` |
| `tc.neon.c32.3m.8x4` | `neon.c32.3m.8x4` |
| `tc.neon.f32.8x8.1m-induced` | `neon.c32.i1m.4x8` |
| `tc.neon.f32.8x8.4m-induced` | `neon.c32.i4m.8x8` |
| `tc.neon.f32.16x6.1m-induced` | `neon.c32.i1m.8x6` |
| `tc.neon.f32.16x6.4m-induced` | `neon.c32.i4m.16x6` |
| `tc.neon.f32.24x4.1m-induced` | `neon.c32.i1m.12x4` |
| `tc.neon.f32.24x4.4m-induced` | `neon.c32.i4m.24x4` |
| `tc.neon.f32.12x8.1m-induced` | `neon.c32.i1m.6x8` |
| `tc.neon.f32.12x8.4m-induced` | `neon.c32.i4m.12x8` |

## Environment variables and engine selection (PR 2)

No library reads the environment any more. Every knob is an explicit input:

| Removed | Replacement |
|---|---|
| `TENSORCONTRACT_THREADS` | none: threads come only from the `tcontract_exec::Exec` passed to `execute_*`; the plan chooses the width from its work estimate and the executor's budget |
| `TENSORCONTRACT_POOL` | none (the parked-thread pool is gone) |
| `TENSORCONTRACT_KERNEL` | `PlanConfig::isa` (`KernelForce`) |
| `TENSORCONTRACT_COMPLEX` | `PlanConfig::method` (`Option<tcontract_kernel::Method>`) |
| `TENSORCONTRACT_MC` `_KC` `_NC` `_MC_PCT` `_NC_PCT` | `PlanConfig::blocking` (`BlockingOverride`; positivity and absolute/percentage exclusivity are validated) |
| `TENSORCONTRACT_KC_COUPLE` | `PlanConfig::cache_model.kc_couple` |
| `TENSORCONTRACT_BLOCKMODEL` | `PlanConfig::cache_model.block_model` |
| `TENSORCONTRACT_WRITEBACK` | `PlanConfig::writeback` (`Writeback::{Auto, Gather}`) |
| `TENSORCONTRACT_L3_DOMAINS` | `PlanConfig::cache_model.l3_domains` |
| `TENSORCONTRACT_ORIENT` | `PlanConfig::orientation` (`Orient`) |
| `TENSORCONTRACT_ROWBLOCK` | `PlanConfig::row_block` (`RowBlock`) |
| `TENSORCONTRACT_PARTITION` | `PlanConfig::partition` (`Partition::StaticGrid { pin }` for a pinned `<pm>x<pn>` grid; `domain` is the default `None`; the `m`/`n`/`legacy` spellings no longer exist) |
| `TPRIMS_GEMM_KERNEL` | `PlanConfig::kernel` (`KernelChoice`) |
| `TPRIMS_GEMM_ENGINE` | none: the planner chooses the strategy (PR 3); an explicit kernel, partition, method, blocking, cache model or write-back request forces the packed driver |

`tcbench` accepts the knobs of the table (except `THREADS` and `POOL`) under the prefix `TCBENCH_` instead of
`TENSORCONTRACT_` (for example `TCBENCH_KERNEL`, `TCBENCH_MC`, `TCBENCH_PARTITION`) and parses them into `PlanConfig`; an
unparseable value is an error. Since PR 4 every `tcontract-bench` binary, `tcbench` included, refuses to run when any
variable with the prefix `TENSORCONTRACT_` or `TPRIMS_GEMM_` is set, so a stale sweep script cannot silently measure
the wrong configuration; the retired partition spellings (`m`, `n`, `rows`, `cols`, `legacy`) stay rejected.

## Removed C symbols (PR 1)

`tprims_blas_gemm`, `tprims_blas_gemm_batched` and `tprims/blas.h` are gone; `tcontract_has_part("blas")` returns 0 (it answers 1 only for `"core"` and `"tapp"`).

## Contraction API (PR 3)

### Old to new

| Old | New |
|---|---|
| `tcontract::ContractPlan::<T>::new(&DotGeneral, a, b, c, conj, Strategy, Flags)` | `Plan::<T>::new(&Problem, &PlanConfig)` with `Problem::from_dot_general(dtype, a, b, d, &DotGeneral)`; layouts are `LayoutSpec` (signed element strides and a logical offset) and conjugation is an `Op` of each `OperandSpec` |
| `tensorcontract::Plan::new(Operand, Operand, Option<Operand>, Operand)` over `Layout` and `i64` labels | `Problem::from_labels(dtype, a, b, CSpec, d, &Labels)`; `Layout` and `ElementOp` are replaced by `LayoutSpec` and `Op` |
| `Strategy::{Auto, PermuteGemm, Tblis}` | none: the planner chooses (rules below) and `plan.report().algorithm` says what it chose (`Packed`, `Faer`, `Elementwise`). `PermuteGemm` copied operands and is gone; its copy-free fusion survives as the faer strategy |
| `Flags { no_materialize }` | `PlanConfig::no_materialize`, OR-ed with `Requirements::no_materialize` by `TcontractBackend`. No strategy copies a whole operand, so it is always met and `Unsupported::WouldMaterialize` is only reachable from a foreign backend |
| `GemmConfig { engine, kernel, method, partition, partition_opts, .. }` | `PlanConfig { kernel, isa, partition, no_materialize, method, blocking, orientation, row_block, cache_model, writeback }`; `Engine`/`EngineChoice` and `SelectError::EngineUnsupported` are dropped (the refusals are `Error::Config`) |
| `PartitionPolicy` + `PartitionOpts` + `PartitionMode` | `Partition::{StaticGrid { pin: Option<(pm, pn)>, align_c_lines }, DynamicTiles { job_m, job_n }}`; `DynamicTiles` with `align_c_lines` is no longer expressible |
| `ContractPlan::execute(exec, alpha, a, b, beta, c)` (C is D, accumulated) | `plan.execute_into(exec, alpha, a, b, d)` (overwrite) or `plan.execute_into_accum(exec, alpha, a, b, beta, AccumulationSource, d)`; `AccumulationSource::Output` reads through the mutable D, `Separate(&c_view)` a distinct C |
| `tensorcontract::Plan::run` / `run_with` / `run_raw` / `run_raw_with` | `Plan::execute_into_accum` / `execute_raw` (the C adapter's entry, with `check_raw` as its pointer-only preflight) |
| `tensorcontract::contract_batched(&mut [BatchItem], &Exec)` | `contract_batched(&mut [BatchItem<T>], &Exec)`; items carry `StridedView`s and an `Option<AccumulationSource>` |
| `tensorcontract::batch`, `parse_einsum`, `einsum_labels`, `TensorView`, `TensorViewMut`, `ElementOp::Conjugate` per view | removed; the views are `strided_view` views and conjugation belongs to the problem (a view that disagreed with its plan can no longer be expressed) |
| `plan.stats`, `plan.selected()`, `selected_gemm()`, `dynamic_report` | `plan.report()` (`PlanReport`, `PackedReport` with family id, geometry, folded `PlanStats`, orientation, observed regularity, estimated scratch and the dynamic assignment) |
| `Plan::with_selector(operands, catalog, selector)` | `Plan::new_with_selector(&Problem, &PlanConfig, &KernelCatalog<T>, &mut selector)`; the context no longer carries the thread budget or candidate grids (a plan never reselects for another budget) and operand labels are gone |
| `tprims_contract_traits::{ContractionBackend, PreparedContraction, Problem, Layout, Conj}` | `tcontract::api::{ContractionBackend, PreparedContraction, Problem, LayoutSpec, Op}`; the traits take `&Exec` and `AccumulationSource` |
| `HostExecution`, `NativeHost`, `ExecHost`, `SerialHost`, the duplicated `Par` | removed; `tcontract_exec::{Exec, Par}` is the only host and width vocabulary |
| `tensorcontract::Error`, `tprims_contract_traits::Error`, `tprims_blas::Error` | one `tcontract::Error`: `Config`, `Shape`, `Layout`, `Alias`, `Unsupported`, `Select`, `Exec`, `Backend`, `Internal`, each with typed context |
| `tensorcontract::reference::contract_reference` (over `Layout`) | `tcontract_testkit::oracle::contract_reference` over plain `dims`/`strides`/`labels` slices; shares no code with the planner |
| `tensorcontract::kernel::{plan_config, selected_config, selected_kernel_name}` | `plan.report().packed` (family id, `mr`, `nr`, `mc`, `kc`, `nc`) |
| `TAPP_implementation_name()` | now names `tcontract`; the ABI is unchanged |

### The one lowering

`Problem::from_labels` and `Problem::from_dot_general` share one lowering: repeated labels on one operand select a
diagonal (strides add), a label on only one input is a reduction (a K axis with the other input's stride zero), output-only
labels are `Unsupported`, and the output must be injective. The result keeps the original layouts plus normalized
M/N/K/H role axes with signed A/B/C/D strides; negative strides are not rejected for their sign.
`DotGeneral` fixes `C = D` (`CSpec::Output`), output axes `[lhs_free..., rhs_free..., batch...]`.

### Strategy selection

1. An explicit kernel, selector, partition, complex method, blocking, cache model or write-back request forces the packed
   driver, including for an all-batch problem.
2. An all-batch problem runs the elementwise pass (full `op_C`, `op_D` and separate-C semantics).
3. A problem that fuses to one strided batched GEMM without copying an operand, with full semantics, runs on faer,
   unless its `m*n*k` exceeds the dtype's `PlanConfig::faer_limit` (default: c64 above 2^17, other dtypes unbounded).
   A separately described C, or a reduction over an axis one input lacks, is declined.
4. Everything else runs on the packed driver.

`alpha == 0` or an empty contraction computes `op_D(beta * op_C(C))` in one output pass, for every strategy, reading no
input. Beta zero reads neither C nor D.

### Removed capabilities (not renames)

| Old | Disposition |
|---|---|
| `gemm`, `gemm_with`, `gemm_batched`, `gemm_batched_with`, `gemm_grouped`, `GemmShape`, `MatIn`, `BatchIn` | `gemm`: a `DotGeneral` with one contracted pair (or `Labels`), keeping op, scaling and layout semantics. `gemm_batched`: batch axes of one problem (`lc=[1], rc=[0], lb=[2], rb=[2]`); a loop of independent items over one fixed-layout plan is `contract_batched`. `gemm_grouped`: one plan per shape, grouped by the host; a fixed-layout plan does not accept heterogeneous items |
| `trsm` | removed; no contraction replacement |
| `tprims-linalg` decompositions and solves | removed in PR 1; the consumer chooses another provider |
| `EngineChoice::{Faer, Packed}`, `BatchStrategy`, `Selected` | see `Strategy` above |
| `ElementOpMismatch` (a view's op differing from the plan's) | cannot occur: views carry no element operation |
| explicit kernel or partition on an all-batch problem | **behaviour change:** the old planner refused it with `EngineUnsupported` because the elementwise pass has no kernel or grid; the packed driver now honours it (tested for `StaticGrid` and `DynamicTiles`) |
| `TAPP` ABI | unchanged; the product lowers its labels once into a `Problem`, and one status table maps the contract error |

The pinned `ext/tenferro-cpu-tcontract` consumer stays on its old revision until its own migration is done; this repository
does not claim a drop-in revision bump, and `trsm`/linalg need downstream provider work.

### Source moves (PR 3)

All of these are `git mv`s followed by edits in separate commits; `git log --follow` follows a file, so the splits are
listed here.

| Old path | New path |
|---|---|
| `tensorprimitives/crates/tensorcontract/src/plan.rs` | `crates/tcontract/src/plan/analysis.rs` (index analysis: roles, folding, scatter vectors, `PackedPlan`) and `plan/orientation.rs` (orientation, row block, partition rule, with their tests) |
| `tensorprimitives/crates/tensorcontract/src/driver.rs` | `driver/mod.rs` (entry points, the per-call decisions, `Ctx`, `execute_capped`), `driver/static_grid.rs` (`BPart`, the static team's `run_strip`) and `driver/tile.rs` (`Epoch`, `Bufs`, panel packing, `compute_block`, `emit_tile`); resolution is mandatory and the foreign-scalar path is gone |
| `tensorprimitives/crates/tensorcontract/src/driver/dynamic.rs` | `crates/tcontract/src/driver/dynamic.rs` |
| `tensorprimitives/crates/tensorcontract/tests/*.rs` (`common`, `correctness`, `cplx_native`, `direct`, `dynamic`, `gemm_families`, `partition_bitwise`, `selection_boundary`, `traits`) | `crates/tcontract/src/driver/tests/*.rs` (unit tests over a test-only adapter, `compat.rs`) |
| `tensorprimitives/crates/tensorcontract/tests/{exec_pin,exec_pool,exec_seam,explicit_config,pool_workspace,workspace_alloc}.rs` | `crates/tcontract/tests/packed_*.rs` |
| `tensorprimitives/crates/tensorcontract/src/{select,resolve,batch,buffer}.rs` | `crates/tcontract/src/{select,resolve,batch,buffer}.rs` |
| `tensorprimitives/crates/tensorcontract/src/reference.rs` | `crates/tcontract-testkit/src/oracle.rs` |
| `tensorprimitives/crates/tensorcontract/src/layout.rs` | removed (`LayoutSpec` in `api/problem.rs`) |
| `crates/tcontract/src/{plan,permute_gemm,util,tblis,host,backend}.rs` (old `ContractPlan`) | `plan/mod.rs` (the new `Plan`), `strategy/{faer,elementwise}.rs`, `api/backend.rs` and `backend.rs` (`TcontractBackend`) |
| `crates/tprims-contract-traits/src/{problem,error,backend,host}.rs` | `crates/tcontract/src/api/{problem,error,backend}.rs` (`host.rs` removed) |
| `crates/tprims-blas/src/{batched,gemm,scalar}.rs` (faer batched loop) | `strategy/faer.rs` (copy-free fusion only) and `api/scalar.rs` |
| `tensorprimitives/crates/tensorprimitives-bench/src/*.rs`, `build.rs` | `benchmarks/benchmarks/tcbench/*.rs`, `benchmarks/build.rs` |
| `tensorprimitives/crates/tensorprimitives-tapp` | unchanged location; rebased onto `Plan<T>` |
| `crates/tcontract-kernel/src/kernels/kernel_set.rs` | `kernels/menu_tests.rs` (see above) |
| `tensorprimitives/crates/tensorprimitives-bench/src/engines/{sweep,shapes,orient,premise}.rs` | `engines/run.rs` (the corpus run, from `sweep`); `shapes`, `orient` and `premise` are deleted, as are `examples/{blocking_model,kernel_shapes}.rs` |

The `tcontract` examples `blocking_model` and `kernel_shapes` measured the legacy menu and are deleted with it;
`examples/contract.rs` is ported. The pure-move commits of this PR are `Move tensorcontract, contract-traits and the
strategy sources into tcontract (pure git mv)`, `Rename tcontract-testkit to tcontract-testkit (pure git mv)` and
`Move the tcbench sources into the tcontract-bench package (pure git mv)`.

### Benchmarks

`tcbench` keeps `run` (the corpus across the planner's choice, the forced packed driver, TTGT and TBLIS), `verify`,
`info` and `--stress`. `sweep`, `shapes`, `orient` and `premise` are deleted: they depended on the legacy menu and on
engine internals the contract crate no longer exposes. The three complex methods are now one knob
(`TENSORCONTRACT_COMPLEX`) rather than three engine columns. The `contract` bench rows are `plan`/`packed` (formerly
`pg`/`tblis`), and `tenferro-p1-gemm` and `large-batched-gemm` are `dot_general` corpora; `benchmarks/.../blas` is
deleted.

## C ABI (PR 4)

`libtcontract_capi` is built by `cargo build --release -p tcontract-capi` (crate types `cdylib`, `staticlib`, `rlib`; the library
name is `tcontract_capi`). The ABI keeps the semantics of #26: handles are `isize`, every entry point is panic-safe, errors are
`TCONTRACT_*` status codes, a TAPP `TAPP_error` is a `tcontract_status`, and there is no C tuning API.

| Old | New |
|---|---|
| `cargo build -p tprims-bundle` (features `tapp`, `blas`) | `cargo build -p tcontract-capi` |
| `libtensorprimitives_tapp.{so,dylib}` and `tensorprimitives-tapp.pc` | `libtcontract_capi.{so,dylib}` and `tcontract_capi.pc` (`pkg-config --cflags --libs tcontract_capi`) |
| the second `tapp.h` of `tensorprimitives-tapp` (with `TAPP_VERSION_STRING` and `TAPP_VERSION_AT_LEAST`) | deleted; `include/tapp.h` and `include/tapp/*.h` are the pinned upstream headers, verbatim. Check versions at run time: `TAPP_implementation_version()` (crate version) and `tcontract_abi_version()`, compared with `TCONTRACT_ABI_VERSION` of `<tcontract/core.h>` |
| `TAPP_implementation_version` / `TAPP_implementation_name` declared in `tapp.h` | declared in `<tcontract/core.h>` (and so in `<tcontract/tcontract.h>`) |
| `TAPP_ERROR_UNSUPPORTED` of the old header | `TCONTRACT_ERR_UNSUPPORTED` of `<tcontract/core.h>` |
| `crates/tensorprimitives-tapp/install.sh` | `crates/tcontract-capi/install.sh`: installs the whole include tree (`tapp.h`, `tapp/`, `dlpack/`, `tcontract/`), `libtcontract_capi`, `tcontract_capi.pc` and the licences |
| `examples/c-consumer` (`TAPP_CRATE_DIR`, corrosion crate `tensorprimitives-tapp`) | same example, crate `tcontract-capi`, target `tcontract_capi` |

The version test that read `TAPP_VERSION_*` out of the old header became `the_library_reports_its_crate_version`
(run-time check); no other test was removed.

### Source moves (PR 4)

The pure-move commits are `Move tprims-core, tprims-bundle and tensorprimitives-tapp into crates/tcontract-capi (pure git
mv)` and `Move tensorprimitives docs, README and CHANGELOG into docs/archive/tensorprimitives; licences to the root (pure
git mv)`; the edits and splits are separate commits.

| Old path | New path |
|---|---|
| `crates/tprims-core/src/{dlpack,status,tensor}.rs` | `crates/tcontract-capi/src/{dlpack,status,tensor}.rs` (unchanged bodies) |
| `crates/tprims-core/src/exec.rs` | `crates/tcontract-capi/src/executor.rs` |
| `crates/tprims-core/src/lib.rs`, `crates/tprims-bundle/src/lib.rs` | `crates/tcontract-capi/src/lib.rs` (crate documentation, `ABI_VERSION`, `tcontract_abi_version`, `tcontract_has_part`) |
| `tensorprimitives/crates/tensorprimitives-tapp/src/lib.rs` | `src/abi.rs` (constants and the one contract-error to status table), `src/handle.rs` (handles, attributes, status release, implementation identification), `src/tensor_info.rs` (tensor infos), `src/product.rs` (the product lowered once to a `Problem`), `src/execute.rs` (single and batched execution) |
| `crates/tprims-core/tests/*`, `tensorprimitives/crates/tensorprimitives-tapp/tests/*`, `crates/tprims-bundle/tests/*` | `crates/tcontract-capi/tests/*` (`tapp_raw.rs` is `raw.rs`) |
| `crates/tprims-core/include`, `crates/tprims-bundle/include` | `crates/tcontract-capi/include` |
| `tensorprimitives/{LICENSE-MIT,LICENSE-APACHE}` | `LICENSE-MIT`, `LICENSE-APACHE` (copyright "Lukas Devos and tensor4all contributors") |
| `tensorprimitives/docs/**`, `README.md`, `CHANGELOG.md` | `docs/archive/tensorprimitives/` (`original-README.md`, `original-CHANGELOG.md`) |
| `tensorprimitives/{CONTRIBUTING,CLAUDE,RELEASING}.md`, `.gitignore`, `scripts/`, `bench-results/` | deleted; recoverable at the parent of the deletion commit pinned in [provenance](provenance.md) |

### Documentation and rules

`README.md` is rewritten (its Rust and C examples are the compiled files `crates/tcontract/examples/readme.rs` and
`crates/tcontract-capi/tests/c/standard_consumer.c`, and `tests/readme_examples.rs` checks that the README shows them
verbatim). `docs/architecture.md`, `docs/design-principles.md`, `docs/research-map.md`, `docs/experiments.md` and the
decision log were rewritten for the new crate set; `HANDOFF.md` moved to `docs/worklogs/`. `AGENTS.md`,
`REPOSITORY_RULES.md`, the skills and CI name only existing packages; CI gained a `c-consumer` job and runs the README
example.
