//! The non-packed strategies: faer on a copy-free fusion, and the elementwise
//! pass (which also serves the shared output update).

pub(crate) mod elementwise;
pub(crate) mod faer;
pub(crate) mod fuse;
pub(crate) mod inner_product;
pub(crate) mod rank_k;
