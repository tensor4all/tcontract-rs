//! How the output is cut into worker cells.
//!
//! The driver's grid — `pm` row strips of whole `MR` panels by `pn` column
//! groups of whole `NR` slivers — is a *policy*, not a property of the
//! contraction, so it is named here and chosen at planning time. Every element
//! still has exactly one owning thread and accumulates over the whole of `K` in
//! the original order, which is what keeps any policy bitwise identical to the
//! serial run.

/// Which grid the driver builds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PartitionPolicy {
    /// `pm` row strips by `pn` column groups. `pm == pn == 0` keeps the
    /// driver's own cost model, which is what every unconfigured plan uses.
    StaticGrid {
        /// Row strips, or zero for the driver's cost model.
        pm: usize,
        /// Column groups, or zero for the driver's cost model.
        pn: usize,
    },
    /// Opt-in dynamic assignment of output work to the active team: workers
    /// atomically claim row bands (or, for short-wide problems, row-band by
    /// column-subtile jobs) of whole register blocks. It assigns work; it does
    /// not split K, accumulate atomically or change any arithmetic, so the result
    /// is bitwise identical to the static and serial runs for a fixed blocking.
    DynamicTiles {
        /// Job rows: a positive multiple of the family's logical `MR`.
        job_m: usize,
        /// Job columns: a positive multiple of the family's logical `NR`.
        job_n: usize,
    },
}

impl Default for PartitionPolicy {
    fn default() -> Self {
        Self::StaticGrid { pm: 0, pn: 0 }
    }
}

/// Non-grid execution options that still change which cells exist.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PartitionOpts {
    /// Round strip boundaries down so each strip starts on a 64-byte line of
    /// `C`. Costs a little balance, buys aligned output runs.
    pub align_c_lines: bool,
}

/// Row range of strip `r` of `pm` over `m` rows, in whole `MR` panels.
///
/// `align` is the element alignment strips are rounded down to, or zero for
/// none; it must be a multiple of `mr` when set. The last strip always ends at
/// `m`, so a rounded boundary can never leave a tail unclaimed.
///
/// # Examples
/// ```
/// use tcontract_kernel::partition::strip;
/// // 37 rows are five 8-row panels, so three strips take 1, 2 and 2 panels.
/// assert_eq!(strip(0, 3, 37, 8, 0), (0, 8));
/// assert_eq!(strip(1, 3, 37, 8, 0), (8, 24));
/// assert_eq!(strip(2, 3, 37, 8, 0), (24, 37));
/// ```
pub fn strip(r: usize, pm: usize, m: usize, mr: usize, align: usize) -> (usize, usize) {
    debug_assert!(pm > 0 && mr > 0);
    let npanels = m.div_ceil(mr);
    let raw = |k: usize| (k * npanels / pm) * mr;
    let round = |x: usize| {
        if align > mr {
            x - x % align
        } else {
            x
        }
    };
    let last = r + 1 == pm;
    let lo = round(raw(r)).min(m);
    // The tail belongs to whoever is last, whatever rounding would do to it.
    let hi = if last { m } else { round(raw(r + 1)).min(m) };
    (lo.min(hi), hi)
}
