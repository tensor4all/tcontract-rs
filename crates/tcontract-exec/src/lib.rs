//! Explicit execution contexts for the tcontract stack.
//!
//! A host lends its Rayon pool through [`Pool::borrow`]; operations take an
//! [`Exec`]. Work of width one runs on the calling thread and never touches
//! the pool; wider work enters the pool once, and not at all when the caller
//! is already one of its workers. See `PERFORMANCE_TIPS.md`, CPU Threading
//! Contract.
//!
//! # Examples
//!
//! ```
//! use tcontract_exec::{Exec, Pool};
//! let tp = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
//! let pool = Pool::borrow(&tp);
//! let exec = Exec::rayon(&pool);
//! let width = exec.install(2, |par| par.threads());
//! assert_eq!(width, 2);
//! assert_eq!(pool.stats().entries, 1);
//! ```
mod error;
mod exec;
mod pool;
mod width;
mod workspace;

#[cfg(feature = "strided")]
pub mod strided;

pub use error::ExecError;
pub use exec::{Exec, Par};
pub use pool::{Pool, PoolStats};
pub use width::WidthPolicy;
pub use workspace::{ArenaProvider, TeamLease, WorkspaceProvider, WorkspaceReq, WorkspaceStats};
