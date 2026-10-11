//! The ported driver tests: end-to-end numerical cases of the packed driver
//! against the independent oracle, one family at a time, plus the partition,
//! blocking and selection-boundary cases.

mod c_call;
mod common;
mod compat;
mod correctness;
mod cplx_native;
mod direct;
mod dynamic;
mod gemm_families;
mod partition_bitwise;
mod selection_boundary;
mod traits;
