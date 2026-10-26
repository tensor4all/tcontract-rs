//! CPU resources for the tcontract stack: scratch memory and contraction policy.
//!
//! Placement (where tasks run, and how many at once) belongs to
//! [`threadteam`]: a host owns a [`threadteam::Pool`], lends it through
//! [`Pool`], and each operation takes an [`Exec`], a bundle of one
//! [`threadteam::Exec`] token, an optional workspace and a [`BatchPolicy`].
//! This crate starts no task and exposes no scheduler type. See
//! `PERFORMANCE_TIPS.md`, CPU Threading Contract.
//!
//! # Examples
//!
//! ```
//! use tcontract_exec::{Exec, Pool};
//! let sp = threadteam::Pool::builder().threads(2).build().unwrap();
//! let pool = Pool::borrow(&sp);
//! let mut exec = Exec::new(&pool, 2).unwrap();
//! assert_eq!(exec.budget(), pool.size().min(2));
//! let total = std::sync::atomic::AtomicUsize::new(0);
//! exec.token().for_ranges(100, 1.try_into().unwrap(), |r| {
//!     total.fetch_add(r.len(), std::sync::atomic::Ordering::Relaxed);
//! });
//! assert_eq!(total.into_inner(), 100);
//! ```
pub mod batch;
mod error;
mod exec;
pub mod host;
mod pool;
mod workspace;

pub use batch::{BatchDecision, BatchPolicy, BatchQuery, BatchStrategy, WorkModel};
pub use error::ExecError;
pub use exec::Exec;
pub use host::HostClass;
pub use pool::Pool;
pub use workspace::{
    ArenaProvider, TeamLease, Workspace, WorkspaceProvider, WorkspaceReq, WorkspaceStats,
};
