//! GEMM kernels: the family contract and its registry, packing and write-back,
//! cache blocking, and every project-owned microkernel family (Lukas Devos's
//! tensorcontract kernels, the portable reference kernels and the native
//! complex kernels). Arithmetic traits, packing, scatter, write-back and the
//! tensorcontract kernels were moved from Lukas Devos's tensorcontract:
//! lkdvos/tensorprimitives-rs; MIT OR Apache-2.0.
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]

pub mod abi;
pub mod blocking;
pub mod kernels;
#[doc(hidden)]
pub mod pack;
pub mod select;

// The ABI and selection types live at the crate root, as before.
use abi::{element, family, types};
use blocking as cache;

// Module paths the typed API and its doctests use.
pub use kernels::menu_isa;
pub use kernels::reference::{induced, portable};
pub use pack::{scatter, writeback};
pub use select::partition;

pub use abi::cpu::{CpuFeatures, Isa, KernelForce};
pub use abi::element::{Element, Real, C32, C64};
pub use abi::family::*;
pub use abi::types::*;
pub use pack::scatter::IRREGULAR;
pub use select::catalog::{KernelCatalog, KernelHandle};
pub use select::partition::{PartitionOpts, PartitionPolicy};
pub use select::registry::{
    list_kernels, register, Families, KernelInfo, RealSlot, Registry, SelectError,
};
pub use select::resolve::{resolve_legacy_auto, KernelChoice, ResolvedGemm};
