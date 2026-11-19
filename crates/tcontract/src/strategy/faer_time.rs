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
//! For a product of `W` multiply-adds (four per complex one) on an `m x n x k`
//! GEMM, a row gives
//!
//! ```text
//! at(1) = fixed_ns + work,  work = W / muladds_per_ns
//! at(w) = fixed_ns + parallel_ns + work * min(1, r * L) + r * task_elem_ns * (rows_t + cols_t) * k
//!         + per_width_ns * (w - 1)
//!         + (slow * work + slow_per_width_ns * (w - 1)  if n > 3 * nr * w)
//! r     = ceil(P / w),  P = ceil(m / (2 mr)) * ceil(n / 32)
//! L     = rows_t * cols_t / (m n),  rows_t = min(m, 2 mr),  cols_t = min(n, 32)
//! ```
//!
//! `P`, `L` and the band are faer's own partition, not fitted
//! (`t4a-private-gemm-x86` 0.1.28, `lib.rs`; tensor4all/tcontract-rs#203). The
//! kernel tile `mr x nr` is `gemm_impl`'s AVX-512 table ([`kernel_tile`]).
//! `MilliPar::call` cuts the output into tasks of `mf * mr` rows by
//! `nf * nr = 32` columns that the `w` participants take from one counter, so
//! width `w` runs `r` rounds of at most the largest task; `mf` is 2 for a wide
//! output (`n >= 2m`) and for every `w >= ceil(m / mr) / 4`, and the
//! narrower widths where faer groups more rows give the same `r * L`. Each round
//! also moves a task's operands (`rows_t x k` and `k x cols_t`) to its
//! participant, the `task_elem_ns` term: a narrow-`n` product of two tasks
//! (f64 `80 x 40 x 512`: 32 and 8 columns, `L = 0.8`) then stays serial
//! while a tall one (`512 x 32 x 512`: six tasks) does not. The last term is
//! the regime in which the kernel packs the left operand
//! (`gemm_impl`: `pack_lhs = .. || n > 3 * nr * n_threads`), where every task
//! of a row block waits on the participant packing it (`claim_pack`); measured,
//! it costs about the same 2-5 us per participant whatever the work (f64
//! `80 x 512 x 40`, 16 tasks: 0.82-0.90x width 1 at widths 5 to 11, 2.3x at 16,
//! 0.44x at 43, out of the band). The fitted `muladds_per_ns` and `fixed_ns` are
//! faer's own serial rate and per-call cost on that class; they replace the
//! host table's rate for this decision only. `gain_margin` is the row's median
//! relative error at the widths it chooses on the fit grid, derived after the
//! fit (`item_fit.py`, `margin`): a width above one has to beat width one by
//! more than it.
//!
//! The class is `t4a_cost::CpuClass`: the CPU vendor, family, model and logical CPU count read from
//! `/proc/cpuinfo` (Linux), so two SKUs of one microarchitecture with different
//! core counts are different classes. A host or dtype without a row — any other
//! CPU, another OS, and f64 on the fitted class for now — supplies no time, and the decision keeps its ideal-division rule
//! there, exactly as before.
//!
//! The rows, the fitting script and the raw measurements are in
//! `experiments/faer-item-width/` (`fit.py`, `results/`).

use t4a_cost::{CpuClass, ItemFit, ItemTime};

use crate::api::DType;

/// One fitted row (see the module documentation): the fit's parameters in an
/// `ItemFit`, and faer's one fitted feature beside them, the cost per element of
/// moving a task's operands. The task partition and the slow band are faer's own
/// rule ([`item_time`]), not data.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Row {
    fit: ItemFit,
    task_elem_ns: f64,
}

/// `GenuineIntel` family 6 model 85 with 96 logical CPUs: 2 x Xeon Platinum 9242
/// (Cascade Lake-AP, 4 NUMA nodes of 24 cores, no SMT), `lg` cluster nodes,
/// threadteam 0.6.0, colmat 0.34.0, t4a-private-gemm-x86 0.1.28, a team token of
/// 96. Fitted by `experiments/faer-item-width/fit.py` on the 1331 shapes `m, n, k`
/// in {16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512} at 13 widths, two rounds
/// (`results/fit2-*`). `muladds_per_ns` counts four per complex multiply-add.
const XEON_9242_2S: [(DType, Row); 4] = [
    (
        DType::F32,
        Row {
            fit: ItemFit {
                fixed_ns: 416.0,
                muladds_per_ns: 83.449,
                parallel_ns: 18000.0,
                per_width_ns: 100.0,
                slow: 0.5,
                slow_per_width_ns: 8000.0,
                gain_margin: 0.202,
            },
            task_elem_ns: 0.0,
        },
    ),
    (
        DType::F64,
        Row {
            fit: ItemFit {
                fixed_ns: 254.0,
                muladds_per_ns: 41.994,
                parallel_ns: 13000.0,
                per_width_ns: 200.0,
                slow: 0.5,
                slow_per_width_ns: 6000.0,
                gain_margin: 0.264,
            },
            task_elem_ns: 0.1,
        },
    ),
    (
        DType::C32,
        Row {
            fit: ItemFit {
                fixed_ns: 336.0,
                muladds_per_ns: 84.89,
                parallel_ns: 15000.0,
                per_width_ns: 50.0,
                slow: 0.5,
                slow_per_width_ns: 8000.0,
                gain_margin: 0.254,
            },
            task_elem_ns: 0.2,
        },
    ),
    (
        DType::C64,
        Row {
            fit: ItemFit {
                fixed_ns: 375.0,
                muladds_per_ns: 41.553,
                parallel_ns: 11000.0,
                per_width_ns: 100.0,
                slow: 0.5,
                slow_per_width_ns: 6000.0,
                gain_margin: 0.287,
            },
            task_elem_ns: 0.6,
        },
    ),
];

/// The dtypes whose fitted row the decision uses on that class. The f64 row is
/// fitted but withheld: on threadteam 0.6 / colmat 0.34 even the best measured
/// width per shape leaves the 512-shape acceptance sweep's f64 geo-mean
/// t(best)/t(1) at 0.65 for budget 16, above the 0.592 the row must reach
/// (tensor4all/tcontract-rs#203), so f64 keeps the ideal-division rule until
/// that gate is restated for this stack. Adding `DType::F64` here enables it.
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

/// The microkernel tile `(mr, nr)` of faer's x86 kernel for a product of `m`
/// rows: `gemm_impl`'s `(InstrSet::Avx512, _)` arms (`t4a-private-gemm-x86`
/// 0.1.28). Every fitted class is AVX-512.
fn kernel_tile(dtype: DType, m: usize) -> (usize, usize) {
    match dtype {
        DType::F32 => (96, 4),
        DType::F64 if m > 48 => (48, 4),
        DType::F64 => (24, 8),
        DType::C32 => (48, 4),
        DType::C64 => (24, 4),
    }
}

/// faer's time under `row` for one `m x n x k` product of `dtype` of
/// `muladds` multiply-adds.
pub(crate) fn item_time(
    row: Row,
    dtype: DType,
    (m, n, k): (usize, usize, usize),
    muladds: f64,
) -> ItemTime {
    // faer's features: the task partition of `MilliPar::call` and the slow
    // band of the kernel's left-operand packing.
    let (mr, nr) = kernel_tile(dtype, m);
    let (m, n) = (m.max(1), n.max(1));
    let (task_rows, task_cols) = (m.min(2 * mr), n.min(32));
    let tasks = m.div_ceil(2 * mr).saturating_mul(n.div_ceil(32));
    let share = (task_rows * task_cols) as f64 / (m as f64 * n as f64);
    let task_ns = row.task_elem_ns * ((task_rows + task_cols) * k) as f64;
    row.fit
        .item_time(muladds, tasks, share, task_ns, n.div_ceil(3 * nr))
}

#[cfg(test)]
mod tests;
