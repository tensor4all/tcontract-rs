//! Minimal median timing shared by the tcontract benchmark binaries.

use std::time::{Duration, Instant};

/// Median of `runs` timed calls of `f` after `warmup` untimed calls, in ns.
pub fn median_ns(warmup: usize, runs: usize, f: impl FnMut()) -> f64 {
    let samples = samples_ns(Duration::ZERO, warmup, runs, f);
    samples[samples.len() / 2]
}

/// `runs` timed calls of `f`, in ns and sorted ascending, after untimed calls
/// for `prime` of wall time and then `warmup` more.
pub fn samples_ns(prime: Duration, warmup: usize, runs: usize, mut f: impl FnMut()) -> Vec<f64> {
    let start = Instant::now();
    while start.elapsed() < prime {
        f();
    }
    for _ in 0..warmup {
        f();
    }
    let mut samples: Vec<f64> = (0..runs.max(1))
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_nanos() as f64
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    samples
}

/// `name` from the environment as `usize`, or `default`.
pub fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
