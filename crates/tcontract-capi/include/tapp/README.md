# Vendored TAPP headers

`../tapp.h` and the `*.h` files in this directory are copied verbatim from
`api/include/` of <https://github.com/TAPPorg/reference-implementation> at
commit `77c32d744ee6d339f504620cc80b8679601669bc` (BSD 3-Clause, see
`LICENSE.md` and `AUTHORS.md`, copied from the same commit). They are the ABI
baseline of the tcontract contraction C API; do not edit them. To move the
baseline, replace all of them from one new commit, update the commit here and in
`docs/provenance.md`, and rerun `cargo test -p tcontract-capi`.

tcontract-specific declarations (the worker-pool executor, extra status codes) live in
`../tcontract/tapp_ext.h`, never here.

SHA-256 of the files as vendored (checked by `cargo test -p tcontract-capi`):

```
a47a8bd0388b0d6fbd56c76be9d4896143818af669ffdc9350ebda5a647a5b41  tapp.h
ee938f260cbfc4fe61b9e76625d94d4712943324978a88ab42e63bceeac6e67c  tapp/attributes.h
76dd5e06d404613ae1b7d0e325d9416023cfc91e32788963a0e809c3d5e2db43  tapp/datatype.h
121b285870e38fcd53cc2d49ee021aeffff7729c45ba941edad09d9b22325ed2  tapp/error.h
9d96100f433a36989b20897a0681cab1a477179a381c73aaa178b85a2f822542  tapp/executor.h
fc5ca8c068d67f2601356d5538b50e566bbe00a1e61fff134dc8af52c64bbaf6  tapp/handle.h
4af72687dd76c90edff6b4ed64c6ff27e8f932f8c05308d65303d8d28d53f448  tapp/product.h
e44f3095b43fb72280f61e15ad37693641744c7dc26863291312c024fe73d0ef  tapp/status.h
f5c11dd51ee5b05b274519a21a66463eb844c2ab62e5f9e5a645dcd6cd11fb38  tapp/tensor.h
b54f995b42dbe568d5a8423a471b034710cad2e3f5b7a893bd6f1bb4e1ab5ddd  tapp/util.h
24c0e15936973d0eb42ec344f1a1c9c51c900e1da8edec77de10ae31a3e36e79  tapp/LICENSE.md
72b6ec6b0a0bbbd3698bcc4517a506ac7a170edf2c768f6bfb272e4ba5e7f0fd  tapp/AUTHORS.md
```
