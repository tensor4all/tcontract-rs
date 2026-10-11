//! Shared harness code for tcontract benchmarks.
//!
//! [`threads::BenchThreads`] enforces the thread-count contract of
//! `PERFORMANCE_TIPS.md`: the requested count builds the `Exec`, conflicting
//! thread environment variables abort the run, and the effective width is
//! printed and asserted at startup.
pub mod corpus;
pub mod partition;
pub mod threads;
pub mod timing;
