//! Packing, scatter vectors and write-back.

#[allow(clippy::module_inception)]
pub mod pack;
pub mod permute;
pub mod scatter;
pub mod writeback;

pub use pack::*;
