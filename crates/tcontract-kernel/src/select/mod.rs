//! Family selection: the registry, resolution with frozen blocking, caller-scoped
//! catalogs and the partition policy.

pub mod catalog;
mod host;
pub use host::TuningClass;
pub mod partition;
pub mod registry;
pub mod resolve;
