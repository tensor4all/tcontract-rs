//! The cost-model types shared by the tensor4all providers
//! (tensor4all/tcontract-rs#205).
//!
//! Three things a provider of a batched kernel needs to price its work,
//! without any of the stacks it serves: the host's CPU class
//! ([`CpuClass`], the key fitted rows are recorded under), the time of one
//! item at a width ([`ItemTime`]), and the fitted row a provider's offline fit
//! produces ([`ItemFit`]). Nothing here starts a task, owns a pool or
//! allocates scratch: placement is `threadteam`'s, the batch decision that
//! minimises an [`ItemTime`] is `tcontract-exec`'s, and this crate depends on
//! nothing but `std`, so a cost model can be shared (tensor4all/tlinalg-rs#119)
//! without either library depending on the other.
//!
//! The fitted rows are data, not code: a provider fits them offline per host
//! class and dtype (the generic fit is `experiments/faer-item-width/item_fit.py`
//! in tcontract-rs; this crate carries its Rust contract, [`ItemFit`], not the
//! fit itself) and turns one into an [`ItemTime`] per item.

mod fit;
pub mod host;
mod item_time;

pub use fit::ItemFit;
pub use host::CpuClass;
pub use item_time::ItemTime;
