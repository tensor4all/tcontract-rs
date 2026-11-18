//! The time of one faer GEMM at a width, per host class (tensor4all/tcontract-rs#194).
//!
//! The shared batch decision divides an item's serial estimate ideally over a
//! width unless the provider says how its kernel scales. faer's parallel GEMM
//! does not scale that way for small and mid-size products: on a 96-core Xeon the
//! ideal rule chose widths at which 21% to 41% of the rank-2 products with
//! `m, n, k` in 16..=512 ran more than 5% slower than at width one, up to nine
//! times slower (`experiments/faer-item-width`, acceptance against the base).
//! This module is faer's [`ItemTime`], as data fitted offline per host class,
//! like `tcontract_exec::host::muladds_per_ns`; nothing is measured at run time.
//!
//! For a product of `W` multiply-adds (four per complex one) on an `m x n` output,
//! a row gives
//!
//! ```text
//! at(1) = fixed_ns + W / muladds_per_ns
//! at(w) = fixed_ns + parallel_ns + W / muladds_per_ns / min(w, P) + per_width_ns * (w - 1)
//!         + ((slow + slow_per_width * (w - 1)) * W / muladds_per_ns  if n > slow_cols * w)
//! P     = ceil(m / tile_m) * ceil(n / tile_n)
//! ```
//!
//! `P` bounds the participants the output tiles can feed. The last term is the
//! regime in which faer's x86 kernel packs the left operand once a participant
//! gets more than `3 * nr` columns (`t4a-private-gemm-x86` 0.1.28,
//! `lib.rs`, `pack_lhs = .. || n > 3 * nr * n_threads`); measured, that regime
//! costs a share of the serial time instead of saving one, and the share grows
//! with the width (c32 `16 x 320 x 160`: 1.3x serial time at width 4, 3.3x at 8,
//! 3.9x at 16; 37 us serial). The fitted
//! `muladds_per_ns` and `fixed_ns` are faer's own serial rate and per-call cost on
//! that class; they replace the host table's rate for this decision only.
//!
//! The class is `tcontract_exec::host::CpuClass`: the CPU vendor, family, model and logical CPU count read from
//! `/proc/cpuinfo` (Linux), so two SKUs of one microarchitecture with different
//! core counts are different classes. A host or dtype without a row — any other
//! CPU, another OS, and f64 on the fitted class for now — supplies no time, and the decision keeps its ideal-division rule
//! there, exactly as before.
//!
//! The rows, the fitting script and the raw measurements are in
//! `experiments/faer-item-width/` (`fit.py`, `results/`).

use tcontract_exec::host::CpuClass;
use tcontract_exec::ItemTime;

use crate::api::DType;

/// One fitted row (see the module documentation).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Row {
    fixed_ns: f64,
    muladds_per_ns: f64,
    parallel_ns: f64,
    per_width_ns: f64,
    tile_m: usize,
    tile_n: usize,
    slow_cols: usize,
    slow: f64,
    slow_per_width: f64,
}

/// `GenuineIntel` family 6 model 85 with 96 logical CPUs: 2 x Xeon Platinum 9242
/// (Cascade Lake-AP, 4 NUMA nodes of 24 cores, no SMT), `lg` cluster nodes,
/// threadteam 0.5.1, t4a-faer 0.33.0, t4a-private-gemm-x86 0.1.28, a team token of
/// 96. Fitted by `experiments/faer-item-width/fit.py` on the 1331 shapes `m, n, k`
/// in {16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512} at 13 widths, two rounds
/// (`results/fit-*`). `muladds_per_ns` counts four per complex multiply-add.
const XEON_9242_2S: [(DType, Row); 4] = [
    (
        DType::F32,
        Row {
            fixed_ns: 340.0,
            muladds_per_ns: 83.804,
            parallel_ns: 2000.0,
            per_width_ns: 50.0,
            tile_m: usize::MAX,
            tile_n: 48,
            slow_cols: 12,
            slow: 0.5,
            slow_per_width: 0.1,
        },
    ),
    (
        DType::F64,
        Row {
            fixed_ns: 216.0,
            muladds_per_ns: 41.672,
            parallel_ns: 15000.0,
            per_width_ns: 10.0,
            tile_m: 64,
            tile_n: 48,
            slow_cols: 12,
            slow: 0.5,
            slow_per_width: 0.0,
        },
    ),
    (
        DType::C32,
        Row {
            fixed_ns: 310.0,
            muladds_per_ns: 84.699,
            parallel_ns: 5000.0,
            per_width_ns: 25.0,
            tile_m: 64,
            tile_n: 32,
            slow_cols: 12,
            slow: 0.5,
            slow_per_width: 0.05,
        },
    ),
    (
        DType::C64,
        Row {
            fixed_ns: 353.0,
            muladds_per_ns: 41.282,
            parallel_ns: 6000.0,
            per_width_ns: 0.0,
            tile_m: 64,
            tile_n: 24,
            slow_cols: 12,
            slow: 0.5,
            slow_per_width: 0.05,
        },
    ),
];

/// The dtypes whose fitted row the decision uses on that class. The f64 row is
/// fitted but withheld: on the 512-shape acceptance sweep it left four products
/// above 1.25x width 1 (a single slow width of faer's partition, `80 x 512 x 40`
/// at width 16, and narrow-`n` products such as `80 x 40 x 512`), so f64 keeps
/// the ideal-division rule until those are priced. Adding `DType::F64` here
/// re-enables it.
const XEON_9242_2S_SHIPPED: [DType; 3] = [DType::F32, DType::C32, DType::C64];

/// The row of `dtype` on the host class `key`, if the decision uses one.
pub(crate) fn row_of(key: &CpuClass, dtype: DType) -> Option<Row> {
    let xeon_9242_2s =
        key.vendor == "GenuineIntel" && key.family == 6 && key.model == 85 && key.cpus == 96;
    if !xeon_9242_2s || !XEON_9242_2S_SHIPPED.contains(&dtype) {
        return None;
    }
    XEON_9242_2S
        .iter()
        .find(|(d, _)| *d == dtype)
        .map(|&(_, row)| row)
}

/// The row of `dtype` on this host, if its class has one.
pub(crate) fn host_row(dtype: DType) -> Option<Row> {
    CpuClass::detect().and_then(|key| row_of(key, dtype))
}

/// faer's time under `row` for one product of `muladds` multiply-adds on an
/// `m x n` output.
pub(crate) fn item_time(row: Row, m: usize, n: usize, muladds: f64) -> ItemTime {
    let work = muladds / row.muladds_per_ns;
    let split = m
        .div_ceil(row.tile_m)
        .saturating_mul(n.div_ceil(row.tile_n));
    ItemTime::new(row.fixed_ns, work)
        .with_parallel(row.parallel_ns, split, row.per_width_ns)
        .with_slow_below(
            n.div_ceil(row.slow_cols),
            row.slow * work,
            row.slow_per_width * work,
        )
}

#[cfg(test)]
mod tests;
