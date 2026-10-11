# Benchmarks

- [tcontract](tcontract/README.md): exec entry, BLAS, linear algebra, contraction
  and the Rust side of the C ABI comparison, each at 1T and 4T.
- [C ABI](../c/README.md): the same calls from C through `libtcontract_capi`.

strided-rs kernel benchmarks live in
[strided-rs-benchmark-suite](https://github.com/tensor4all/strided-rs-benchmark-suite).

## Result Policy

- Keep benchmark setup and allocation outside timed regions unless the page
  says otherwise.
- Record the tcontract-rs commit, CPU, core set and profile beside measured
  results.
- Run thread-count variants sequentially.
