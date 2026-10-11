# Migration guide: tprims → tcontract (2026-10-11, issue #105)

The library and repository `tensor4all/tprims-rs` were renamed to `tcontract`
to reflect their contraction-focused scope. This file records what a consumer
has to change. The standard TAPP interface names are preserved; the
`strided-*` crates keep their names.

## Repository

- GitHub: `tensor4all/tprims-rs` → `tensor4all/tcontract-rs` (the old URL
  redirects).
- Issue and PR links written as `tprims-rs#<n>` keep resolving through the
  redirect; update them when touched.

## Rust crates

| Old | New |
|---|---|
| `tprims-contract` | `tcontract` (the core contraction crate) |
| `tprims-exec` | `tcontract-exec` |
| `tprims-kernel` | `tcontract-kernel` |
| `tprims-capi` | `tcontract-capi` |
| `tprims-testkit` | `tcontract-testkit` |
| `tprims-bench` (benchmarks package) | `tcontract-bench` |

Against a pinned git dependency, the source changes from
`https://github.com/tensor4all/tprims-rs.git` to
`https://github.com/tensor4all/tcontract-rs.git`, and the dependency keys
change with the crate names above. Update `use` paths the same way:
`tprims_contract::…` → `tcontract::…`, `tprims_exec::…` → `tcontract_exec::…`,
and so on. Public Rust types follow: `TprimsBackend` → `TcontractBackend`.

## C ABI

| Old | New |
|---|---|
| shared library `libtprims` (`libtprims.so` / `.dylib` / `tprims.dll` + import lib) | `libtcontract_capi` (`libtcontract_capi.so` / `.dylib` / `tcontract_capi.dll`) |
| headers `tprims/{tprims,core,tapp_ext}.h` | `tcontract/{tcontract,core,tapp_ext}.h` |
| symbol prefix `tprims_` (`tprims_abi_version`, `tprims_status`, `tprims_tensor`, `tprims_tensor_borrow_raw/versioned`, `tprims_tapp_executor_create_rayon/get_threads/set_budget`, `tprims_last_error`, `tprims_rayon_opts`, `tprims_has_part`, …) | prefix `tcontract_`; prototypes and numeric status values unchanged |
| macros/constants `TPRIMS_ABI_VERSION` (100), `TPRIMS_OK`, `TPRIMS_ERR_*`, `TPRIMS_BUSY` | `TCONTRACT_ABI_VERSION`, `TCONTRACT_OK`, `TCONTRACT_ERR_*`, `TCONTRACT_BUSY`; values unchanged |
| pkg-config `tprims.pc` (`-ltprims`), CMake/Corrosion target `tprims` | `tcontract_capi.pc` (`-ltcontract_capi`), target `tcontract_capi` |

Preserved: the pinned upstream TAPP headers (`include/tapp.h`,
`include/tapp/`), the `TAPP_*` function names (including the non-standard
`TAPP_implementation_name` / `TAPP_implementation_version`), and the
`TAPP_error` numeric codes.

The Rust-side active environment knob `TPRIMS_PHASE` (per-call phase timing)
is now `TCONTRACT_PHASE`.

## What deliberately did not change

- The `strided-*` crates and their source pin.
- The `tcbench` benchmark harness.
- The benchmark harness's rejection list for **removed** environment
  variables: it still rejects `TENSORCONTRACT_*` and `TPRIMS_GEMM_*`, so a
  stale sweep script from the pre-#37 world cannot silently measure the wrong
  thing. Do not "clean up" this list.
- Names of removed artifacts in historical records (`docs/worklogs/`,
  `docs/decision-log.md`, `docs/provenance.md`,
  `docs/migration-2026-10.md`): `tprims-blas`, `tprims-core`,
  `tprims-bundle`, `tprims-linalg`, `tprims-contract-traits`,
  `tprims-gemm-kernel`, `tprims-kernel-cplx` and the removed `tprims_exec*` /
  `tprims_contract_plan_*` C ABI keep their recorded names, and recorded
  measurement data is untouched.

## Example diff (Cargo.toml)

```toml
# before
tprims-contract = { git = "https://github.com/tensor4all/tprims-rs.git", rev = "…" }
tprims-exec = { git = "https://github.com/tensor4all/tprims-rs.git", rev = "…" }

# after
tcontract = { git = "https://github.com/tensor4all/tcontract-rs.git", rev = "…" }
tcontract-exec = { git = "https://github.com/tensor4all/tcontract-rs.git", rev = "…" }
```

C consumers link `-ltcontract_capi`, include `<tcontract/tcontract.h>` (or the
TAPP-only `<tapp.h>`), and call `tcontract_*` extension symbols where they
called `tprims_*`.