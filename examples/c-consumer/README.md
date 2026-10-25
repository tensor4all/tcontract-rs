# Consuming this engine from C or C++

A worked example of linking the TAPP surface into a C or C++ project, and the
reference for how to do it in your own build.

`main.c` is not a toy: it runs a real `f64` contraction and a real `c64` one
with `TAPP_CONJUGATE` on an operand, and checks both against reference loops
written in C. It is run by CI, so the shipped header is verified against the
library it describes rather than assumed to match.

## What you need

* A Rust toolchain **at build time only** — nothing Rust is required to *run*
  the result. Rust 1.89 or newer (released 2025-08-04); `rustup` installs a
  pinned toolchain into `~/.cargo` without root and without touching the system.
* CMake ≥ 3.22, or any build system that can link a library.

There is no C++ ABI involvement: the boundary is C, so no name mangling, no
exceptions crossing it, and no libstdc++ version coupling. That makes this
*more* ABI-stable against a C++ project than a C++ library would be.

## Build it

```bash
# Corrosion drives cargo from CMake and imports the crate as a CMake target.
cmake -S examples/c-consumer -B build/c-consumer -DCMAKE_BUILD_TYPE=Release
cmake --build build/c-consumer
ctest --test-dir build/c-consumer --output-on-failure
```

To link a library you have already built instead — a separate Rust build step,
an offline site, or simply not wanting CMake to invoke cargo:

```bash
cargo build --release -p tcontract-capi
cmake -S examples/c-consumer -B build/c-consumer \
      -DCMAKE_BUILD_TYPE=Release -DTAPP_USE_CORROSION=OFF \
      -DTAPP_LIBRARY_DIR=$PWD/target/release
cmake --build build/c-consumer
```

## In your own CMake project

```cmake
include(FetchContent)
FetchContent_Declare(Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.5.1)
FetchContent_MakeAvailable(Corrosion)

corrosion_import_crate(
    MANIFEST_PATH  ${TCONTRACT_DIR}/Cargo.toml
    CRATES         tcontract-capi)

target_link_libraries(your_target PRIVATE tcontract_capi)
target_include_directories(your_target PRIVATE
    ${TCONTRACT_DIR}/crates/tcontract-capi/include)
```

`CRATES` matters: it keeps the unpublished benchmark harness out of your
build.

Corrosion works out the platform link line for you, which is the main reason to
prefer it. On Linux it resolves a static link to `gcc_s;util;rt;pthread;m;dl;c`;
getting that wrong by hand is the usual first failure.

## Without CMake

```bash
cargo build --release -p tcontract-capi

cc myprog.c -I crates/tcontract-capi/include \
   -L target/release -ltcontract_capi \
   -Wl,-rpath,$PWD/target/release -lm
```

Prefer the **cdylib** (`libtcontract_capi.so` / `.dylib`) over the
`staticlib`. The static archive bundles the Rust standard library: it is ~26 MB,
needs `-lpthread -ldl -lm` explicitly, and exports internal symbols that collide
if two Rust static libraries end up in one binary. The cdylib is ~5 MB and
exports essentially only the TAPP symbols.

## Installing into a prefix

If you want the library somewhere other than `target/release` — a module tree, a
container image, `/usr/local` — build it and then install it:

```bash
cargo build --release -p tcontract-capi
crates/tcontract-capi/install.sh --prefix=/opt/tapp
```

That lays out `lib/libtcontract_capi.so`, `include/`,
`lib/pkgconfig/tcontract_capi.pc` and the two licences, after which the
library is consumable without knowing anything about cargo:

```bash
export PKG_CONFIG_PATH=/opt/tapp/lib/pkgconfig
cc myprog.c $(pkg-config --cflags --libs tcontract_capi) \
   -Wl,-rpath,/opt/tapp/lib -lm
```

From CMake, `-DTAPP_PREFIX=/opt/tapp` in this example project does the same
through `pkg_check_modules`. This is the same script the BinaryBuilder recipe
the Yggdrasil recipe calls, so the layout a JLL presents and the layout a
manual install produces are the same layout by construction.

### Set the SONAME, because cargo will not

**rustc emits `-soname` only for the `dylib` crate type, never for a `cdylib`.**
A library built with a plain `cargo build` therefore has no `DT_SONAME`, and
everything that links against it records a bare filename instead. On macOS it is
worse: `LC_ID_DYLIB` defaults to the absolute path the linker wrote to, so the
`.dylib` is not relocatable at all.

Set them at link time. `install.sh` warns if you did not:

```bash
# Linux, FreeBSD
RUSTFLAGS='-C link-arg=-Wl,-soname,libtcontract_capi.so' \
  cargo build --release -p tcontract-capi

# macOS
RUSTFLAGS='-C link-arg=-Wl,-install_name,@rpath/libtcontract_capi.dylib' \
  cargo build --release -p tcontract-capi

# musl, where the cdylib is otherwise *dropped* -- cargo prints "dropping
# unsupported crate type `cdylib`" and exits 0, leaving you a header and no
# library
RUSTFLAGS='-C target-feature=-crt-static -C link-arg=-Wl,-soname,libtcontract_capi.so' \
  cargo build --release -p tcontract-capi
```

Do not reach for `-C rpath` to get the macOS install name: it sets it, and also
bakes build-tree `LC_RPATH` entries into the library.

## Offline and air-gapped sites

Cargo fetches from crates.io on first build, which a cmake+C++ build does not.
Vendor once, then build with the network unused:

```bash
mkdir -p .cargo
cargo vendor vendor >> .cargo/config.toml
cargo build --offline --release -p tcontract-capi
```

Ship `vendor/` alongside the source and the build needs no network: the crates
are `publish = false`, and the tree depends only on crates.io packages
(`num-complex`, `threadteam` and the other strided-rs crates, `t4a-faer`,
`thiserror`, ...), all of which `cargo vendor` covers. There is no dedicated CI
offline lane.

## What the API looks like

See `crates/tcontract-capi/include/`, which documents the handle
discipline, the supported TAPP cases, and the two places this implementation
departs from an underspecified upstream (`TAPP_IN_PLACE`, and the non-zero error
codes). The short version, as `main.c` uses it:

```c
TAPP_create_handle(&handle);
TAPP_create_executor(&exec);
TAPP_create_tensor_info(&info, TAPP_C64, nmode, extents, strides);  /* per tensor */
TAPP_create_tensor_product(&plan, handle, op_A, A, idx_A, ..., TAPP_DEFAULT_PREC);
TAPP_execute_product(plan, exec, NULL, &alpha, a, b, &beta, c, d);
```

Strides are in **elements**, not bytes. Complex buffers are interleaved with the
real part first, so a `double _Complex*` (or `std::complex<double>*`) is passed
straight through with no conversion.

## Status

This is an early engine in the middle of its performance phase — the top-level
`README.md` carries the current release — with one microarchitecture measured end
to end. The C surface is verified — the
conformance suite drives the C symbols, and this example checks numerical
results through the shipped header — but do not read that as a claim that the
performance is competitive everywhere. See the top-level `README.md` for what is
measured and what is not, and `docs/archive/tensorprimitives/notebook/` for why.
