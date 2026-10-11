//! Micro-kernels, complex methods, and cache blocking parameters.
//!
//! # The three complex methods
//!
//! Complex contraction can be induced from real arithmetic in several ways.
//! This crate implements three and lets the caller choose, because which one
//! wins depends on the shape, the element type and the machine — and because
//! the whole point of the project is to be able to measure them against each
//! other on equal footing. Select with [`ComplexMethod`], per plan via
//! `PlanConfig::method` of `tcontract` (`planar` | `1m` | `3m`); the
//! default is [`ComplexMethod::Planar`].
//!
//! All three share the *same* index analysis, scatter machinery, five-loop
//! driver and write-back path. They differ only in what packing emits, what
//! the micro-kernel computes, and how the accumulator tile is read back.
//!
//! ## Packed panel formats
//!
//! With `VR` the register block along the packed axis (`MR` for A, `NR` for
//! B), one *logical* k-step of a sliver holds:
//!
//! ```text
//! real            [ x_0 .. x_{VR-1} ]                                   VR
//!
//! planar          [ re_0..re_{VR-1} | im_0..im_{VR-1} ]                2*VR
//!
//! 1m, operand A   [ re_0, im_0, re_1, im_1, ...                        4*MR
//!   ("1e")          | -im_0, re_0, -im_1, re_1, ... ]
//! 1m, operand B   [ re_0..re_{NR-1} | im_0..im_{NR-1} ]                2*NR
//!   ("1r")          (bit-identical to planar)
//!
//! 3m              [ re_0..re_{VR-1} | im_0..im_{VR-1}                  3*VR
//!                   | (re+im)_0..(re+im)_{VR-1} ]
//! ```
//!
//! ## What each method costs
//!
//! | | reals packed per complex elt (A / B) | real FMAs per k per tile | accumulator planes |
//! |---|---|---|---|
//! | planar | 2 / 2 | `4*MR*NR` | 2 |
//! | 1m | **4** / 2 | `4*MR*NR` | 2 (as `2*MR x NR` real) |
//! | 3m | 3 / 3 | **`3*MR*NR`** | 3 |
//!
//! * **planar** — real and imaginary parts separated during the scatter pass
//!   that packing has to make anyway, then a fused complex micro-kernel over
//!   the two planes. No in-register shuffles, and the smallest packed A panel.
//! * **1m** — Van Zee's induced method, what BLIS and TBLIS 2.x use. Each
//!   complex element of A becomes the real `2x2` block `[[re, -im], [im, re]]`
//!   and each of B a `2x1` column, so a *single real* micro-kernel of shape
//!   `2*MR x NR` over `2*KC` steps produces the complex product. Costs 2x the
//!   packed-A footprint and 2x the packing store traffic; needs no complex
//!   kernel at all.
//! * **3m** — Karatsuba. `M1 = Ar*Br`, `M2 = Ai*Bi`, `M3 = (Ar+Ai)*(Br+Bi)`,
//!   then `Cr = M1 - M2` and `Ci = M3 - M1 - M2`. **25% fewer flops**, and the
//!   sum planes are formed for free during packing. The error bound is weaker
//!   than the other two (it is relative to `|Ar||Br| + |Ai||Bi|` rather than to
//!   the complex magnitudes, so it can lose relative accuracy under
//!   cancellation), which is why it is opt-in rather than the default.
//!
//! # Kernel contract
//!
//! A micro-kernel *overwrites* an accumulator tile with the panel product;
//! `alpha`, `beta`, the scattered write-back to `C`/`D` and any recombination
//! of planes are applied afterwards by the write-back. Keeping them
//! separate is what lets one kernel serve the regular fast path, the gather
//! path and every edge block without duplication.
//!
//! Kernels take the *logical* `kc` (in complex elements for complex methods)
//! and know their own panel layout, so the driver does not have to.

use crate::cache;

/// Why a string was not the name of one of this module's settings.
///
/// The [`FromStr`](core::str::FromStr) error for [`ComplexMethod`] and
/// [`cache::BlockModel`] — the two settings a caller ever spells out, in a
/// sweep script's arm list, a benchmark flag or a CSV column. One
/// type for both, because the failure is the same one ("that is not a spelling
/// I know") and only the accepted list differs; and deliberately *not*
/// `tcontract::Error`, which is `#[non_exhaustive]` and enumerates the ways a
/// *contraction* is ill-formed. A misspelled method name is not one of those.
///
/// Keeps no copy of the offending string, so it stays `Copy` and
/// allocation-free. The caller still has its input; what it does not have, and
/// what the message supplies, is the list of spellings that would have worked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParseError {
    setting: &'static str,
    accepted: &'static str,
}

impl ParseError {
    #[doc(hidden)]
    pub const fn new(setting: &'static str, accepted: &'static str) -> ParseError {
        ParseError { setting, accepted }
    }
}

impl core::fmt::Display for ParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "unrecognised {}: expected {}",
            self.setting, self.accepted
        )
    }
}

// `core::error::Error`, not `std::error::Error`: the same trait since Rust 1.81,
// and this way the impl does not reach for `std` on a path that has no other
// need of it — the parsers themselves work with the `std` feature off.
impl core::error::Error for ParseError {}

/// How to induce complex arithmetic from real micro-kernels.
///
/// Deliberately **not** `#[non_exhaustive]`, unlike [`cache::BlockModel`] and
/// [`cache::CacheSource`]. A downstream kernel provider must match on
/// this to hand back a kernel per method, and a variant it cannot name is one it
/// cannot service: the catch-all arm that `#[non_exhaustive]` would force could
/// only panic or return a kernel in the wrong packed format, which the driver
/// then trusts. So a fourth method is a breaking change on purpose. [`PackFormat`]
/// and [`TileFormat`] are open for the same reason — the same implementors read
/// them to describe what their kernel produces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ComplexMethod {
    /// Planar / split-complex packing with a fused complex micro-kernel.
    #[default]
    Planar,
    /// Van Zee's 1m induced method: "1e" packing for A, "1r" for B, and a
    /// single real micro-kernel. This is what BLIS and TBLIS 2.x do.
    OneM,
    /// Karatsuba 3m: three real products instead of four. 25% fewer flops,
    /// weaker error bound.
    ThreeM,
}

impl ComplexMethod {
    /// All three methods, so a harness can iterate them without spelling the
    /// list out and silently miss one when a fourth is added.
    pub const ALL: [ComplexMethod; 3] = [
        ComplexMethod::Planar,
        ComplexMethod::OneM,
        ComplexMethod::ThreeM,
    ];

    /// Parse a method name, case- and whitespace-insensitively. Accepts the
    /// short spellings (`planar` | `1m` | `3m`) as well as `split`, `onem`, `threem` and `karatsuba`.
    pub fn parse(s: &str) -> Option<ComplexMethod> {
        match s.trim().to_ascii_lowercase().as_str() {
            "planar" | "split" => Some(ComplexMethod::Planar),
            "1m" | "onem" => Some(ComplexMethod::OneM),
            "3m" | "threem" | "karatsuba" => Some(ComplexMethod::ThreeM),
            _ => None,
        }
    }

    /// The canonical short name, which [`ComplexMethod::parse`] round-trips.
    /// Used in kernel names and every CSV column heading.
    pub fn name(self) -> &'static str {
        match self {
            ComplexMethod::Planar => "planar",
            ComplexMethod::OneM => "1m",
            ComplexMethod::ThreeM => "3m",
        }
    }
}

/// [`ComplexMethod::name`]'s spelling, which [`FromStr`](core::str::FromStr)
/// round-trips.
impl core::fmt::Display for ComplexMethod {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // `pad` rather than `write_str`, so a width in the format string is
        // honoured: these names are column headings as often as they are prose.
        f.pad(self.name())
    }
}

/// [`ComplexMethod::parse`] as the standard trait. The inherent method stays
/// because a
/// caller that falls back to the default rather than reporting wants the `Option`, and because it can be
/// called where a trait method's error type would only be discarded.
impl core::str::FromStr for ComplexMethod {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<ComplexMethod, ParseError> {
        ComplexMethod::parse(s).ok_or(ParseError::new(
            "complex method",
            "planar | 1m | 3m (also split, onem, threem, karatsuba)",
        ))
    }
}

/// What packing should emit for one operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PackFormat {
    /// One real value per element.
    Real,
    /// Two planes per k-step: all real parts, then all imaginary parts. Used
    /// for both operands under [`ComplexMethod::Planar`], and for operand B
    /// under [`ComplexMethod::OneM`] (BLIS's "1r").
    Planar,
    /// BLIS's "1e": operand A under [`ComplexMethod::OneM`]. Four reals per
    /// complex element, laid out as two real k-steps of `2*VR`.
    OneE,
    /// Three planes per k-step: real, imaginary, and their sum.
    ThreeM,
    /// Adjacent real and imaginary values for each logical lane.
    Interleaved,
}

impl PackFormat {
    /// Reals emitted per element per logical k-step.
    #[inline]
    pub const fn reals_per_element(self) -> usize {
        match self {
            PackFormat::Real => 1,
            PackFormat::Planar | PackFormat::Interleaved => 2,
            PackFormat::ThreeM => 3,
            PackFormat::OneE => 4,
        }
    }
}

/// How the write-back should read a micro-kernel's accumulator tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TileFormat {
    /// `ab[j*MR + i]`, imaginary part zero.
    Real,
    /// Real plane then imaginary plane, each `MR x NR` column-major.
    Planar,
    /// One real `2*MR x NR` column-major tile; row `2i` is the real part of
    /// complex row `i` and row `2i+1` the imaginary part.
    OneM,
    /// Three `MR x NR` planes `M1`, `M2`, `M3`; the complex value is
    /// `(M1 - M2) + i*(M3 - M1 - M2)`.
    ThreeM,
    /// Adjacent real/imaginary values, complex column-major.
    Interleaved,
    /// Four column-major planes: Ar·Br, Ai·Bi, Ar·Bi, Ai·Br.
    /// Write-back reconstructs `(p0 - p1) + i*(p2 + p3)`.
    FourM,
}

/// Real values per logical tile element for a scratch format.
///
/// # Examples
/// ```
/// use tcontract_kernel::{tile_planes, TileFormat};
/// assert_eq!(tile_planes(TileFormat::FourM), 4);
/// assert_eq!(tile_planes(TileFormat::Interleaved), 2);
/// ```
pub const fn tile_planes(tile: TileFormat) -> usize {
    match tile {
        TileFormat::Real => 1,
        TileFormat::Planar | TileFormat::OneM | TileFormat::Interleaved => 2,
        TileFormat::ThreeM => 3,
        TileFormat::FourM => 4,
    }
}

/// A micro-kernel: register-blocked panel-panel product.
#[derive(Clone, Copy)]
pub struct Ukr<T> {
    /// Rows of the micro-tile, in *logical* (complex for complex methods)
    /// elements.
    pub mr: usize,
    /// Columns of the micro-tile.
    pub nr: usize,
    /// Reals in an A sliver per logical k-step (`mr * a_pack.reals_per_element()`).
    pub a_per_k: usize,
    /// Reals in a B sliver per logical k-step.
    pub b_per_k: usize,
    /// Reals in the accumulator tile: `planes * mr * nr`.
    pub tile: usize,
    /// Layout `func` requires of the packed A sliver.
    pub a_pack: PackFormat,
    /// Layout `func` requires of the packed B sliver.
    pub b_pack: PackFormat,
    /// Layout `func` leaves the accumulator tile in.
    pub tile_fmt: TileFormat,
    /// The kernel itself, as a plain function pointer.
    ///
    /// It *overwrites* `ab` with the panel product and does nothing else — no
    /// `alpha`, no `beta`, no store to `C`/`D`, no recombination of planes. See
    /// the module's "Kernel contract" section for why that division is what
    /// makes one kernel serve the regular path, the gather path and every edge
    /// block.
    ///
    /// # Safety
    /// `a` must address `a_per_k * kc` values, `b` must address `b_per_k * kc`
    /// values, and `ab` must address `tile` values. Vectorised kernels reach
    /// this pointer through a plain-`fn` trampoline, so the CPU-feature check
    /// belongs to whoever built the [`Ukr`], not to the caller.
    pub func: unsafe fn(kc: usize, a: *const T, b: *const T, ab: *mut T),
    /// Kernel name, e.g. `"avx512-planar"`. Reported by
    /// the plan report's family id and used to label measurements, so a number can
    /// always be traced to the code that produced it.
    pub name: &'static str,
}

impl<T> core::fmt::Debug for Ukr<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Ukr({} {}x{})", self.name, self.mr, self.nr)
    }
}

/// Cache blocking parameters, in logical elements.
///
/// * `mc x kc` is the packed A block, sized for L2.
/// * `kc x nc` is the packed B block, sized for L3.
///
/// There are two derivations, chosen by
/// [`Tuning::block_model`]:
/// [`Blocking::derive`], the hardcoded Phase 2 heuristic, which is the
/// **default**; and [`Blocking::model`], the BLIS analytical model driven by
/// cache descriptors probed from the hardware, which is what transfers to a
/// machine nobody measured on. The switch is applied on every path that
/// produces a configuration, so every kernel — vectorised, scalar, default
/// shape or menu alternate — goes through the same one.
///
/// `mc` and `nc` are always whole multiples of the micro-tile's `mr` and `nr`
/// respectively; the driver's loop arithmetic relies on it, and
/// [`KernelConfig::with_blocking`] re-imposes it after any change.
///
/// **No `Default`**, deliberately. There is no blocking that is right without
/// knowing the element size and the packed footprint — that is what
/// [`Blocking::derive`] is — and the zero value a derived `Default` would give
/// is not merely useless, it is silently *valid*: every consumer clamps rather
/// than rejects, so `Plan::with_blocking(Blocking::default())` would run the
/// whole contraction correctly at `mc = mr`, `kc = 1`, `nc = nr`, which is one
/// k-step per pass over one micro-tile. Use [`Blocking::derive`],
/// [`Blocking::model`], or write the three numbers out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Blocking {
    /// Rows of the packed `A` block, sized so `mc x kc` reals fit the L2.
    /// Also bounds the strip of `D` that one pass over the `jr` loop revisits,
    /// which is a *second*, opposing constraint the derivation does not model —
    /// see A13 in `docs/archive/tensorprimitives/notebook/`.
    pub mc: usize,
    /// Contraction depth of one pass. First-order for the complex method
    /// ranking, because it decides whether the `A` sliver is an L1 resident or
    /// an L2 stream.
    pub kc: usize,
    /// Columns of the packed `B` block, sized so `kc x nc` reals fit the L3.
    pub nc: usize,
}

impl Blocking {
    /// Derive blocking from the packed footprint the kernel actually produces.
    ///
    /// Doing it this way rather than from `size_of::<Element>()` is what makes
    /// 1m automatically get a smaller `MC`: its packed A carries four reals per
    /// complex element instead of two, so the same L2 budget buys half as many
    /// rows. Getting that wrong would hand 1m an unfair L2 overflow and make
    /// the comparison meaningless.
    pub fn derive(real_bytes: usize, a_reals: usize, b_reals: usize) -> Blocking {
        let kc = if real_bytes <= 4 { 384 } else { 256 };
        Blocking::derive_at_depth(real_bytes, a_reals, b_reals, kc)
    }

    /// [`Blocking::derive`] with `kc` supplied rather than chosen.
    ///
    /// Exists for the `MC`/`KC`/`NC` sweep, which needs to vary `kc` and get
    /// the cache-budget-consistent `mc`/`nc` that go with it rather than a
    /// grid of independently chosen numbers.
    ///
    /// Tempting use it takes some care with: re-deriving against `min(k, KC)`
    /// once the contraction depth is known. A contraction shallower than `KC`
    /// — `k = 24` against a `KC` of 256 describes the entire `abcijk` third of
    /// the corpus — otherwise packs into an `A` block sized for a panel ten
    /// times deeper, using a tenth of its L2 budget. Doing that measured
    /// **+13% on one case and −18% on another**, because `MC` also bounds the
    /// strip of `D` that a `jr` pass revisits, which this budget does not
    /// model. See the Phase 4 report before reaching for it.
    #[doc(hidden)]
    pub fn derive_at_depth(
        real_bytes: usize,
        a_reals: usize,
        b_reals: usize,
        kc: usize,
    ) -> Blocking {
        // Roughly half of a 1 MiB L2 for the packed A block, 3 MiB of L3 for B.
        const A_BUDGET: usize = 512 * 1024;
        const B_BUDGET: usize = 3 * 1024 * 1024;
        let kc = kc.max(1);
        let mc = (A_BUDGET / (kc * a_reals * real_bytes)).max(1);
        let nc = (B_BUDGET / (kc * b_reals * real_bytes)).max(1);
        Blocking { mc, kc, nc }
    }

    /// The analytical alternative to [`Blocking::derive`]: BLIS's model over
    /// cache descriptors probed at run time, with no machine-specific constant
    /// anywhere in it.
    ///
    /// Everything it needs about the kernel is already on the [`Ukr`] — the
    /// micro-tile shape and the packing formats, from which the reals per
    /// element follow — so a kernel gets the right blocking without declaring
    /// anything new. `threads` is the count the plan will run with, and it
    /// matters only for `nc`; see [`cache::analytical`].
    ///
    /// Off by default. See [`Tuning::block_model`] for the switch and
    /// [`cache::BlockModel`] for why.
    pub fn model<T>(ukr: &Ukr<T>, threads: usize) -> Blocking {
        cache::analytical(
            cache::PanelGeom {
                real_bytes: core::mem::size_of::<T>(),
                a_reals: ukr.a_pack.reals_per_element(),
                b_reals: ukr.b_pack.reals_per_element(),
                mr: ukr.mr,
                nr: ukr.nr,
            },
            threads,
            &cache::hierarchy(),
        )
    }
}

/// Everything the driver needs for one element type and complex method.
///
/// This is what the built-in kernel menus return, and the two halves are
/// not independent: the blocking is derived from the packed footprint *this*
/// micro-kernel produces, and `mc`/`nc` must stay aligned to its register
/// block. Build one with [`scalar::config_real`](crate::kernels::reference::scalar::config_real) or `config_cplx`
/// rather than by hand.
#[derive(Clone, Copy, Debug)]
pub struct KernelConfig<T> {
    /// The micro-kernel and its shape.
    pub ukr: Ukr<T>,
    /// Cache blocking that suits it.
    pub blk: Blocking,
}

impl<T> KernelConfig<T> {
    /// Apply the blocking derivation and overrides of `tuning` at a known
    /// thread count, then round `mc`/`nc` to whole multiples of the register
    /// block (the driver's loop arithmetic relies on that).
    ///
    /// The thread count reaches the blocking *here* rather than through
    /// [`Blocking::derive`]'s arguments, which is the smallest place it can
    /// enter: the [`Ukr`] in hand carries every other input the model needs.
    /// Under the legacy derivation the count is ignored. Apply this once to a
    /// raw configuration: the percentage overrides scale whatever the
    /// derivation produced, so a second application would square them.
    #[doc(hidden)]
    pub fn normalise_for(self, tuning: &Tuning, threads: usize) -> Self {
        let mut blk = match tuning.block_model {
            cache::BlockModel::Legacy => match tuning.kc_couple {
                // A *coupled* `kc` re-derives `mc`/`nc` against the same cache
                // budgets at the new depth; the plain `kc` override changes `kc`
                // alone and leaves the `D` strip a `jr` pass revisits exactly as
                // wide as it was. Both arms matter because `MC` is bounded from
                // two sides and only the pair separates them (A13).
                Some(kc) => Blocking::derive_at_depth(
                    core::mem::size_of::<T>(),
                    self.ukr.a_pack.reals_per_element(),
                    self.ukr.b_pack.reals_per_element(),
                    kc,
                ),
                None => self.blk,
            },
            cache::BlockModel::Analytical => Blocking::model(&self.ukr, threads),
        };
        // Preserve legacy arithmetic and API; canonical resolution selects
        // checked multiplication and reports invalid overrides as errors.
        if tuning.blocking.is_set() {
            // INVARIANT: this legacy multiplication callback always returns
            // Some, so only the pre-existing arithmetic can fail here.
            blk = tuning.blocking.apply(blk, |a, b| Some(a * b)).unwrap();
        }
        self.with_blocking(blk)
    }

    /// Re-derive the blocking for a plan's thread count, if the derivation in
    /// force actually depends on it.
    ///
    /// A deliberate no-op under the legacy constants, and not merely as an
    /// optimisation: the percentage overrides scale *whatever the derivation
    /// produced*, so applying [`KernelConfig::normalise_for`] a second time to
    /// an already-scaled `blk` would square the scaling. The analytical path
    /// recomputes from the model each time and so is safe to re-run.
    #[doc(hidden)]
    pub fn retarget_threads(self, tuning: &Tuning, threads: usize) -> Self {
        match tuning.block_model {
            cache::BlockModel::Legacy => self,
            cache::BlockModel::Analytical => self.normalise_for(tuning, threads),
        }
    }

    /// Replace the blocking parameters, re-imposing the register-block
    /// alignment invariant.
    #[must_use = "with_blocking returns a new configuration; the receiver is unchanged"]
    pub fn with_blocking(mut self, blk: Blocking) -> Self {
        self.blk.mc = blk.mc.next_multiple_of(self.ukr.mr).max(self.ukr.mr);
        self.blk.nc = blk.nc.next_multiple_of(self.ukr.nr).max(self.ukr.nr);
        self.blk.kc = blk.kc.max(1);
        self
    }
}

/// Absolute and percentage overrides of the cache blocking. An absolute size
/// replaces the derived value; a percentage scales it. An absolute size and a
/// percentage for the same dimension are not meant to be combined.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlockingOverride {
    /// Rows of the packed `A` block.
    pub mc: Option<usize>,
    /// Contraction depth of one pass.
    pub kc: Option<usize>,
    /// Columns of the packed `B` block.
    pub nc: Option<usize>,
    /// Percentage of the derived `mc`.
    pub mc_pct: Option<usize>,
    /// Percentage of the derived `nc`.
    pub nc_pct: Option<usize>,
}

impl BlockingOverride {
    /// Whether any override is requested.
    pub fn is_set(&self) -> bool {
        *self != Self::default()
    }

    /// Apply the overrides to `blk` with checked multiplication; `None` when a
    /// percentage scaling overflows. No register rounding happens here.
    pub fn apply_checked(self, blk: Blocking) -> Option<Blocking> {
        self.apply(blk, usize::checked_mul)
    }

    pub(crate) fn apply(
        self,
        mut blk: Blocking,
        multiply: impl Fn(usize, usize) -> Option<usize>,
    ) -> Option<Blocking> {
        blk.mc = self.mc.unwrap_or(blk.mc);
        blk.kc = self.kc.unwrap_or(blk.kc);
        blk.nc = self.nc.unwrap_or(blk.nc);
        if let Some(p) = self.mc_pct {
            blk.mc = (multiply(blk.mc, p)? / 100).max(1);
        }
        if let Some(p) = self.nc_pct {
            blk.nc = (multiply(blk.nc, p)? / 100).max(1);
        }
        Some(blk)
    }
}

/// Every knob that used to be a `TENSORCONTRACT_*` environment variable inside
/// the kernel layer, as an explicit input. `Tuning::default()` is the baseline
/// with all of them unset.
///
/// Used by the typed resolver ([`ResolvedGemm::with_tuning`](crate::ResolvedGemm::with_tuning))
/// and by the default-menu normalisation; a planner owns one and passes it down.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tuning {
    /// Instruction-set preference for the tensorcontract menu and the legacy
    /// Auto selection.
    pub kernel_force: crate::KernelForce,
    /// Which blocking derivation is in force.
    pub block_model: cache::BlockModel,
    /// Absolute and percentage blocking overrides.
    pub blocking: BlockingOverride,
    /// A coupled `kc`: sets `kc` and re-derives `mc`/`nc` at that depth
    /// (legacy derivation only).
    pub kc_couple: Option<usize>,
    /// Force the general scatter write-back instead of the format-specialized
    /// one.
    pub writeback_gather: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipped blocking, spelled out.
    ///
    /// Every performance number in `docs/archive/tensorprimitives/notebook/` was taken against exactly
    /// these, and the pending `MC`/`KC`/`NC` grid defines its arms relative to
    /// them, so changing one is changing what those measurements mean. The
    /// analytical model is the reason to have this test: it must stay opt-in,
    /// and if it ever becomes the default that is a decision recorded in
    /// `docs/archive/tensorprimitives/notebook/`, not a diff that slips through here.
    #[test]
    fn legacy_blocking_is_unchanged() {
        let d = |real_bytes, a, b| {
            let x = Blocking::derive(real_bytes, a, b);
            (x.mc, x.kc, x.nc)
        };
        // 8-byte reals: `kc = 256`, half a 1 MiB L2 for A, 3 MiB of L3 for B.
        assert_eq!(d(8, 1, 1), (256, 256, 1536), "f64 real");
        assert_eq!(d(8, 2, 2), (128, 256, 768), "c64 planar");
        assert_eq!(d(8, 4, 2), (64, 256, 768), "c64 1m");
        assert_eq!(d(8, 3, 3), (85, 256, 512), "c64 3m");
        // 4-byte reals: `kc = 384`, same two budgets.
        assert_eq!(d(4, 1, 1), (341, 384, 2048), "f32 real");
        assert_eq!(d(4, 2, 2), (170, 384, 1024), "c32 planar");
        assert_eq!(d(4, 4, 2), (85, 384, 1024), "c32 1m");
        assert_eq!(d(4, 3, 3), (113, 384, 682), "c32 3m");
    }
}
