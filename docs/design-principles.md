# Design principles

[Back to the overview](../README.md). This page states what the library is trying to be and the rules that follow from it. The [architecture notes](architecture.md) apply these principles to concrete crates; the [decision log](decision-log.md) records which choices are settled.

## What the library is

tcontract is a CPU library for dense binary tensor contraction: an execution context, a kernel layer, a contraction planner and drivers, and a C ABI. It is meant to sit **underneath** other libraries and applications, in Rust or through a C ABI from C, C++, Julia or Python. It is not an end-user array library and not tied to any one consumer.

## 1. Few crates, each at a real boundary

Every crate has a separate consumer or a stable interface; there is no umbrella crate that re-exports the library.

- **Why:** a consumer that needs only the execution context or only the kernels should not compile, audit or version-lock the rest.
- **Consequence:** each crate has its own documentation and tests, and its public API makes sense without a sibling above it. Convenience layers (array types, `ndarray`/`mdarray` adapters, einsum strings) belong to consumers or to thin adapter crates above the library.
- **Consequence:** a crate is added only for a separate consumer and a stable interface, not to mirror a taxonomy. Capabilities that lost their consumer were removed rather than kept as parts (see the [migration guide](migration-2026-10.md)).

## 2. One direction of dependency

The dependency graph is acyclic: `tcontract-exec` and `tcontract-kernel` at the bottom, `tcontract` on top of both, `tcontract-capi` on top of `tcontract`; `tcontract-testkit` and the benchmarks consume them.

- A lower crate never learns about a higher one.
- A crate depends on the narrowest lower contract it needs. `tcontract` depends on the packed-kernel contract of `tcontract-kernel`, because that is what direct contraction requires.

## 3. One representation, one validator, one error

A contraction is one lowered, role-grouped `Problem` with `Labels` and `DotGeneral` as front ends. Validation lives in `tcontract`; the C ABI lowers its labels into the same `Problem` and maps the one `tcontract::Error` to one status table. There is no second validator, no duplicated executor and no per-layer error type.

## 4. Short names under one prefix

Crates are named `tcontract-<part>` with a part name that says what it does: `tcontract-exec`, `tcontract-kernel`, `tcontract`, `tcontract-capi`.

- The prefix identifies the family and avoids claiming generic crates.io names.
- The prefix is neutral: no consumer or project name, because the library is meant for users beyond its original project.

## 5. One C library, the standard ABI

`tcontract-capi` is the only crate with C symbols and builds `libtcontract_capi` (`cdylib`, `staticlib`, `rlib`). It contains no algorithm.

- Everything in the library shares one execution runtime and one set of handle types, so an executor is valid in every call. The executor is the standard `TAPP_executor`, created and destroyed through the C API; the C host owns it and it owns its Rayon pool. Handles of another TAPP provider cannot be mixed in.
- Where a standard exists, the ABI is the standard: contraction is [TAPP](https://arxiv.org/abs/2601.07827) with the prototypes of the pinned upstream headers, and tcontract adds only named extensions (`tcontract_tapp_*`, DLPack operands, status codes). Nonzero error codes are provider-defined, so callers test success with `TAPP_check_success`.
- Separate libraries per part are not supported: they would duplicate handle types, thread pools and, for static libraries, Rust runtime symbols.
- There is no C tuning API; kernel and blocking choices are Rust `PlanConfig` inputs.
- The ABI is versioned from the start, panics are caught at every entry point, errors are returned as status codes, and the version and contents are queryable at run time.

## 6. Zero copy at the boundary

Data crosses the C ABI as [DLPack](https://dmlc.github.io/dlpack/latest/) descriptors so that arrays from NumPy, PyTorch, JAX and Julia pass through without copying.

- Operands are borrowed; ownership stays with the caller.
- Strides, offsets and column-major layouts are respected rather than normalized by copying.
- Anything DLPack cannot express, such as lazy conjugation, is an explicit argument, not a hidden conversion.

## 7. No hidden costs

A caller must be able to see and control what an operation costs.

- **No hidden copies.** If a layout forces materialization, the operation reports the chosen strategy, and a caller can forbid it.
- **No hidden threads.** Every expensive operation receives an explicit execution context. No crate uses an ambient global pool, and no library crate reads an environment variable, spawns threads on its own (including fallback teams when a pool is narrower than a plan), or keeps threads alive after the context is closed.
- **No hidden allocation in hot paths.** Scratch sizes are queryable and scratch is reusable across calls.
- **Explicit plans.** Validation and planning can be separated from execution so repeated calls pay only for the work.

## 8. The host owns execution

Whoever embeds the stack decides how many threads run and on which runtime.

- A Rust caller passes an execution context. A C, Julia or Python host can create, use and close a Rayon pool through the ABI, or run serially.
- Closing a pool the library created joins its threads, so a host can shut down cleanly. Closing a context that borrows the host's pool never stops the host's threads.
- One thread budget governs both batch-level and inner parallelism so nested parallelism does not oversubscribe.

## 9. Correct first, then measured

- Numerical correctness gates every performance claim: comparison with an independent oracle, extreme scales, complex values, noncontiguous and negative-stride views.
- Optimizations and provider choices require recorded, reproducible measurements under the [experiment protocol](experiments.md). A design hypothesis stays labelled as one until evidence exists.
- Unfavorable and inconclusive results are kept.

## 10. Credit and provenance

Published algorithms are implemented independently and cited. Translated code and imported tests keep their upstream notices and licenses, and are recorded under the [provenance policy](provenance.md). Scientific credit is due even when no license obligation exists.

## 11. A clear scope

Kept out of the stack, on purpose:

- N-ary einsum and contraction-order planning: frontends call `tcontract` per binary step.
- Automatic differentiation, tracing, device transfer and GPU backends.
- Array container types and ecosystem adapters.
- Dense linear algebra, BLAS entry points and iterative solvers: GEMM is a contraction, and the other capabilities were removed ([#37](https://github.com/tensor4all/tcontract-rs/issues/37), [migration guide](migration-2026-10.md)).

A narrow scope is what lets each crate stay small enough to be reused on its own.
