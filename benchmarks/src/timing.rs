//! Minimal median timing shared by the tcontract benchmark binaries.

use std::time::Instant;

/// Median of `runs` timed calls of `f` after `warmup` untimed calls, in ns.
pub fn median_ns(warmup: usize, runs: usize, mut f: impl FnMut()) -> f64 {
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
    samples[samples.len() / 2]
}

/// `name` from the environment as `usize`, or `default`.
pub fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
