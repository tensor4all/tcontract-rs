<!-- unified contribution policy: ../../docs/contributing/rust-repositories.md in tensor4all/tensor4all-meta -->
# Contributing to tcontract-rs

The rules that apply to every Rust repository in `tensor4all` are in one place:
**[tensor4all-meta · docs/contributing/rust-repositories.md](https://github.com/tensor4all/tensor4all-meta/blob/main/docs/contributing/rust-repositories.md)**.
Read that first — it covers the issue-first workflow, what a pull request body has to contain, red-first tests, the
gates every repository runs, the review requirement, and the version policy.

This file only records what is specific to `tcontract-rs`:

## Local facts

- **MSRV**: `rust-version = "1.89"` in the workspace `Cargo.toml`, inherited by every crate
  (`rust-version.workspace = true`). The `msrv` job builds `cargo build --workspace --all-targets` on `1.89.0`; every
  other Rust job uses `stable`, and `nightly` is installed only in the `placement` lane for `cargo-public-api`.
- **Feature combinations CI builds**: default features. `tcontract-kernel` has `default = ["std"]`, but no crate is
  `#![no_std]` and no lane turns that feature off, so the lanes differ by profile and target rather than by feature
  set: `cargo test --profile ci --workspace` on both operating systems, `--profile release-test` on Linux, the C ABI
  test and its exported-symbol list, and an installed and a prebuilt `cmake` consumer.
- **Platforms claimed**: `ubuntu-latest` and `macos-latest` for `clippy`, `test` and `capi`; `fmt`, `placement`,
  `scripts`, `test-release`, `c-consumer`, `msrv` and `doc` run on `ubuntu-latest` only.
- **Local gate**: `cargo test --workspace`.

## Anything here that departs from the shared rules

Nothing departs from the rules in the shared page. This repository adds a `deny`/`cargo-public-api` placement check
(no `rayon` type in `tcontract-exec`'s public API), the `release-test` profile lane, and the C consumers that link the
built library.
