//! Cache blocking: hardware facts (`probe`) and the analytical model (`model`).

pub mod model;
pub mod probe;

pub use model::*;
pub use probe::*;
