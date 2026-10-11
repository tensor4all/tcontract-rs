# tcontract-capi (libtcontract_capi)

The **TAPP** (Tensor Algebra Processing Primitives, arXiv:2601.07827) C ABI over
`tcontract` (the packed block-scatter driver, with faer and elementwise strategies) — a native-Rust,
transpose-free dense tensor contraction implementation.

Built as `lib`, `cdylib` and `staticlib`, so the same crate serves a Rust
dependent, a `dlopen`-ing host and a statically linked C program.

```
D[idx_D] = alpha * op_A(A[idx_A]) * op_B(B[idx_B]) + beta * op_C(C[idx_C])
```

TAPP is a vendor-neutral interface: an application written against it can be
pointed at TBLIS, cuTENSOR or this engine without changing a line.

## What is covered

* Datatypes `TAPP_F32`, `TAPP_F64`, `TAPP_C32`, `TAPP_C64`.
* `TAPP_CONJUGATE` on any operand.
* TAPP cases 1–4: contraction, Hadamard/batch indices, repeated indices
  (diagonals), and isolated input indices (reductions).
* Generally-strided operands, including negative and zero strides.

**Case 5, output broadcasting, is rejected** with a status code rather than
silently mishandled, which the specification permits.

`TAPP_IN_PLACE` and the non-zero `TAPP_error` codes are two deliberate
departures from upstream, both documented in the shipped header — upstream marks
the first `//TODO` and specifies no codes for the second.

## From C

The header is **shipped rather than fetched**: `include/` holds the upstream TAPP headers verbatim at a pinned commit
(`include/tapp/README.md` lists it with checksums) plus tcontract' own
`include/tcontract/*.h`, because upstream TAPP has no releases and no tags, so
tracking its `main` would be silent drift.

```bash
cargo build --release -p tcontract-capi
./install.sh --prefix=/opt/tapp        # headers, libraries, pkg-config

cc myprog.c $(PKG_CONFIG_PATH=/opt/tapp/lib/pkgconfig \
              pkg-config --cflags --libs tcontract_capi) -lm
```

`examples/c-consumer` in the repository is a CMake project that consumes this
three ways (corrosion, a prebuilt library, an installed prefix); CI runs the
last two, because they fail differently.

## Verification

The ABI is checked against the real upstream interface three ways: a test that
re-declares the entire upstream signature set in an `extern "C"` block, so a
renamed symbol or changed prototype is a link failure; a test that pins the
enumerator values quoted from upstream's `datatype.h` and `product.h`; and the C
consumer above, which compiles *this* header against the built library and
checks numerics.

## Status

Prerelease, and **not yet published**. The API may change.

Full documentation, the measurement record and the benchmark harness are in the
[repository](https://github.com/tensor4all/tcontract-rs).

## License

MIT OR Apache-2.0.
