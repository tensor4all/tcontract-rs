//! [`PlanConfig`]: every explicit planning input, as one owned value.
//!
//! The configuration is dtype-independent and holds no callback. An unset
//! field keeps the baseline behaviour; a set one is a requirement that is
//! honoured or refused, never silently weakened. Nothing here reads the
//! environment: harnesses parse their own flags into these values before a
//! plan is built.
//!
//! GEMM-engine selection (`Engine`, `EngineChoice`) does not exist any more;
//! the strategies are chosen by the planner (see [`Plan`](super::Plan)) and a
//! request only the packed driver can honour forces it.

use tcontract_kernel::blocking::BlockModel;
use tcontract_kernel::{
    BlockingOverride, KernelChoice, KernelForce, Method, PartitionOpts, PartitionPolicy, Tuning,
};

use super::orientation::{Orient, RowBlock};
use crate::api::{ConfigError, DType, Error, Result};

/// Per-dtype upper bound on the GEMM volume `m * n * k` (per batch item) of a
/// fused problem that the planner sends to faer (rule 3 of
/// [`Plan`](super::Plan)). Above it the packed driver runs instead. `None`
/// means no bound.
///
/// The default bounds only `c64`, at `2^17`: on the #61 corpus (Apple M5 Max,
/// 1 thread) faer wins the MPS steps up to chi = 16 (`2 chi^3 = 8192`), ties
/// at chi = 32 (`65536`) and loses by 1.17x at chi = 64 (`524288`). The value
/// is provisional until the crossover sweep of
/// [#63](https://github.com/tensor4all/tcontract-rs/issues/63) fixes the
/// predicate and the per-dtype values; the other dtypes are unbounded until
/// measured. The 2026-10-09 sweep of
/// [#69](https://github.com/tensor4all/tcontract-rs/issues/69) measures both C modes,
/// and with a [`NONE`](Self::NONE) arm both sides of the c64 bound: on the host it
/// ran on, faer is ahead of the packed driver at every fusable volume it measured
/// at one thread — including above the bound, by 1.05x to 1.58x — while at four
/// threads the packed driver is ahead there for 18 of 79 groups. The crossover is
/// therefore host- and width-dependent, no single per-dtype volume is justified by
/// that evidence, and these values stay provisional. See
/// `docs/decision-log.md`.
///
/// # Examples
///
/// ```
/// use tcontract::api::DType;
/// use tcontract::FaerLimit;
///
/// let l = FaerLimit::default();
/// assert_eq!(l.get(DType::C64), Some(1 << 17));
/// assert_eq!(l.get(DType::F64), None);
/// assert_eq!(FaerLimit::NONE.get(DType::C64), None);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaerLimit {
    /// Bound for `f32`.
    pub f32: Option<u64>,
    /// Bound for `f64`.
    pub f64: Option<u64>,
    /// Bound for `c32`.
    pub c32: Option<u64>,
    /// Bound for `c64`.
    pub c64: Option<u64>,
}

impl FaerLimit {
    /// No bound for any dtype: every fusable problem goes to faer (the rule
    /// before #63).
    pub const NONE: Self = Self {
        f32: None,
        f64: None,
        c32: None,
        c64: None,
    };

    /// The bound for `dtype`.
    pub fn get(&self, dtype: DType) -> Option<u64> {
        match dtype {
            DType::F32 => self.f32,
            DType::F64 => self.f64,
            DType::C32 => self.c32,
            DType::C64 => self.c64,
        }
    }
}

impl Default for FaerLimit {
    fn default() -> Self {
        Self {
            c64: Some(1 << 17),
            ..Self::NONE
        }
    }
}

/// How the packed driver cuts the output into worker cells. Every partition is
/// bitwise identical to the serial run for a fixed blocking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Partition {
    /// `pm` row strips by `pn` column groups of whole register blocks.
    StaticGrid {
        /// Pin the grid to `(pm, pn)`, clamped to the panels, blocks and the
        /// width a call may use; `None` keeps the planner's own cost model.
        pin: Option<(usize, usize)>,
        /// Round strip boundaries down to a 64-byte line of `C`.
        align_c_lines: bool,
    },
    /// Workers claim jobs of `job_m x job_n` output elements (positive
    /// multiples of the family's logical `MR` and `NR`).
    DynamicTiles {
        /// Job rows.
        job_m: usize,
        /// Job columns.
        job_n: usize,
    },
}

impl Partition {
    pub(crate) fn policy(self) -> (PartitionPolicy, PartitionOpts) {
        match self {
            Partition::StaticGrid { pin, align_c_lines } => {
                let (pm, pn) = pin.unwrap_or((0, 0));
                (
                    PartitionPolicy::StaticGrid { pm, pn },
                    PartitionOpts { align_c_lines },
                )
            }
            Partition::DynamicTiles { job_m, job_n } => (
                PartitionPolicy::DynamicTiles { job_m, job_n },
                PartitionOpts::default(),
            ),
        }
    }
}

/// Explicit inputs of the cache-blocking model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheModel {
    /// Which blocking derivation is in force. The default is the legacy
    /// constants every committed measurement was taken against.
    pub block_model: BlockModel,
    /// Override the number of L3 domains the threads span; `None` probes the
    /// hardware. Zero is ignored.
    pub l3_domains: Option<usize>,
    /// Couple `kc` to this depth and re-derive `mc`/`nc` against the cache
    /// budgets at it (legacy derivation only).
    pub kc_couple: Option<usize>,
}

/// How the write-back of a micro-tile reaches `D`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Writeback {
    /// The format-specialised write-back where the family has one, with the
    /// scratch-tile fallback. The default.
    #[default]
    Auto,
    /// Always the general scatter write-back.
    Gather,
}

/// Planning options.
///
/// # Examples
///
/// ```
/// use tcontract::{Partition, PlanConfig};
///
/// let mut cfg = PlanConfig::default();
/// assert!(!cfg.requires_packed());
/// cfg.partition = Some(Partition::DynamicTiles { job_m: 48, job_n: 8 });
/// assert!(cfg.requires_packed());
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PlanConfig {
    /// A kernel family: `Auto`, or an exact stable id. An explicit id forces
    /// the packed driver.
    pub kernel: KernelChoice,
    /// Instruction-set preference for the default family menu.
    pub isa: KernelForce,
    /// Grid policy. `Some` forces the packed driver; `None` means the packed
    /// default rule (a static grid from the planner's cost model) when the
    /// packed driver is chosen.
    pub partition: Option<Partition>,
    /// Refuse a plan that would copy a whole operand ([`Unsupported::WouldMaterialize`](crate::api::Unsupported)).
    /// Bounded packing inside the packed driver is not a materialization.
    pub no_materialize: bool,
    /// Accepted for compatibility; no longer consulted. Every plan serves a
    /// fresh, uninitialized output, so no packed alternative is prepared: the
    /// Faer route hands an uninitialized `D` to faer's overwrite-only entry
    /// (`strategy::faer`, [`crate::Strategy`]), which writes every element
    /// without reading it, and the packed driver fills only the tiles it
    /// serves.
    pub fresh_output: bool,
    /// Complex scheme. `None` keeps the baseline default; a forced family must
    /// agree. Forces the packed driver when set.
    pub method: Option<Method>,
    /// Absolute and percentage cache-blocking overrides. Forces the packed
    /// driver when any is set.
    pub blocking: BlockingOverride,
    /// Which operand plays the GEMM row role. Affects the role orientation
    /// once, never tensor storage.
    pub orientation: Orient,
    /// Micro-tile row block request, resolved against the selected menu.
    pub row_block: RowBlock,
    /// Explicit inputs of the blocking model.
    pub cache_model: CacheModel,
    /// Write-back mode.
    pub writeback: Writeback,
    /// Largest fused GEMM the planner sends to faer, per dtype. Does not
    /// force the packed driver by itself.
    pub faer_limit: FaerLimit,
}

impl PlanConfig {
    /// The baseline configuration with the packed driver forced: an explicit
    /// grid request with the planner's own grid rule.
    pub fn packed() -> Self {
        PlanConfig {
            partition: Some(Partition::StaticGrid {
                pin: None,
                align_c_lines: false,
            }),
            ..PlanConfig::default()
        }
    }

    /// Whether this configuration can only be honoured by the packed driver:
    /// an explicit kernel, partition, complex method, blocking, cache model or
    /// write-back request. (Orientation and row block shape the packed plan
    /// but do not force it by themselves.)
    pub fn requires_packed(&self) -> bool {
        self.kernel != KernelChoice::Auto
            || self.partition.is_some()
            || self.method.is_some()
            || self.blocking.is_set()
            || self.cache_model != CacheModel::default()
            || self.writeback != Writeback::Auto
    }

    /// The kernel layer's tuning inputs.
    pub(crate) fn tuning(&self) -> Tuning {
        Tuning {
            kernel_force: self.isa,
            block_model: self.cache_model.block_model,
            blocking: self.blocking,
            kc_couple: self.cache_model.kc_couple,
            writeback_gather: self.writeback == Writeback::Gather,
        }
    }

    /// Check the request on its own, before any problem is seen.
    ///
    /// # Errors
    ///
    /// [`ConfigError::NotPositive`] for a zero blocking size, percentage,
    /// `kc` coupling, L3 domain count or job extent;
    /// [`ConfigError::BlockingExclusive`] for an absolute size and a
    /// percentage on one dimension; [`ConfigError::Option`] for a pinned grid
    /// with a zero side.
    pub fn validate(&self) -> Result<()> {
        let positive = |v: Option<usize>, what: &'static str| match v {
            Some(0) => Err(Error::from(ConfigError::NotPositive { what })),
            _ => Ok(()),
        };
        let b = &self.blocking;
        positive(b.mc, "blocking mc")?;
        positive(b.kc, "blocking kc")?;
        positive(b.nc, "blocking nc")?;
        positive(b.mc_pct, "blocking mc percentage")?;
        positive(b.nc_pct, "blocking nc percentage")?;
        if b.mc.is_some() && b.mc_pct.is_some() {
            return Err(ConfigError::BlockingExclusive { dim: "mc" }.into());
        }
        if b.nc.is_some() && b.nc_pct.is_some() {
            return Err(ConfigError::BlockingExclusive { dim: "nc" }.into());
        }
        positive(self.cache_model.kc_couple, "kc coupling")?;
        positive(self.cache_model.l3_domains, "L3 domain count")?;
        match self.partition {
            Some(Partition::StaticGrid {
                pin: Some((pm, pn)),
                ..
            }) if pm == 0 || pn == 0 => {
                Err(ConfigError::Option("a pinned static grid needs positive pm and pn").into())
            }
            Some(Partition::DynamicTiles { job_m, job_n }) if job_m == 0 || job_n == 0 => {
                Err(ConfigError::NotPositive {
                    what: "dynamic tile job extent",
                }
                .into())
            }
            _ => Ok(()),
        }
    }
}
