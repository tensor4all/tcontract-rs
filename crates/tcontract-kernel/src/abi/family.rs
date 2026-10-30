//! Typed kernel-family contracts. Follows the context/control separation in
//! BLIS (Van Zee & van de Geijn, TOMS 2015) and TBLIS (Matthews, SISC 2018).
//! New descriptor code; existing pack/tile semantics come from tensorcontract.

use crate::{tile_planes, CpuFeatures, Element, Isa, PackFormat, Real, TileFormat, Ukr};

/// Kernel provenance.
///
/// # Examples
/// ```
/// use tcontract_kernel::Origin;
/// assert_eq!(Origin::Tensorcontract.crate_name(), "tcontract-kernel");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Origin {
    /// Lukas Devos's tensorcontract kernels.
    Tensorcontract,
    /// Project-owned portable kernels.
    Portable,
    /// Project-owned native interleaved complex SIMD kernels
    /// (`tcontract-kernel`).
    Cplx,
    /// A downstream crate's own kernels, admitted through a caller-scoped
    /// [`KernelCatalog`](crate::KernelCatalog); the provider reports itself.
    External {
        /// Providing crate.
        crate_name: &'static str,
        /// License of the underlying kernel implementation.
        license: &'static str,
    },
}
impl Origin {
    /// Crate providing this implementation.
    ///
    /// # Examples
    /// ```
    /// use tcontract_kernel::Origin;
    /// assert_eq!(Origin::Portable.crate_name(), "tcontract-kernel");
    /// let own = Origin::External { crate_name: "my-kernels", license: "Apache-2.0" };
    /// assert_eq!((own.crate_name(), own.license()), ("my-kernels", "Apache-2.0"));
    /// ```
    pub const fn crate_name(self) -> &'static str {
        match self {
            Self::Tensorcontract => "tcontract-kernel",
            Self::Portable => "tcontract-kernel",
            Self::Cplx => "tcontract-kernel",
            Self::External { crate_name, .. } => crate_name,
        }
    }
    /// License of the underlying kernel implementation (not the adapter).
    ///
    /// # Examples
    /// ```
    /// use tcontract_kernel::Origin;
    /// assert_eq!(Origin::Portable.license(), "MIT OR Apache-2.0");
    /// ```
    pub const fn license(self) -> &'static str {
        match self {
            Self::Tensorcontract | Self::Portable | Self::Cplx => "MIT OR Apache-2.0",
            Self::External { license, .. } => license,
        }
    }
}

/// Implementation category reported by a family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum KernelImpl {
    /// Portable/reference implementation.
    Reference,
    /// ISA-optimized implementation.
    Optimized,
    /// Complex arithmetic induced from a real kernel.
    Induced,
}

/// Complex arithmetic method, separate from the packed layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Method {
    /// Native complex arithmetic.
    Native,
    /// One real product on expanded/reordered panels.
    OneM,
    /// Three products, with an opt-in cancellation/accuracy trade-off.
    ThreeM,
    /// Four real products recombined by write-back.
    FourM,
}

/// Packed operand layout; external complex storage stays interleaved.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Layout {
    /// Real values.
    Real,
    /// Adjacent real/imaginary values.
    Interleaved,
    /// Separate real/imaginary planes per k-step.
    Planar,
    /// 1m expanded operand.
    OneE,
    /// 1m reordered operand.
    OneR,
    /// Real, imaginary and their sum.
    ThreeM,
}

/// The operand and scratch-tile formats of one complex family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ComplexScheme {
    /// Arithmetic method.
    pub method: Method,
    /// A packing layout.
    pub a: Layout,
    /// B packing layout.
    pub b: Layout,
    /// Scratch tile layout.
    pub tile: TileFormat,
}

/// Preferred output-contiguous axis (a performance hint, not a restriction).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CPref {
    /// Contiguous columns within a row.
    Row,
    /// Contiguous rows within a column.
    Col,
    /// No preference.
    Any,
}
/// Axis that must have unit stride for a direct-B path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    /// Row axis.
    Row,
    /// Column axis.
    Col,
}
/// How a family reads B.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BAccess {
    /// Driver-packed panels.
    Packed,
    /// B may be read in place when the indicated axis has unit stride.
    Direct {
        /// Required contiguous axis.
        unit_stride: Axis,
    },
}
/// Who applies alpha/beta and stores output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CUpdate {
    /// Kernel overwrites a scratch tile, driver writes back.
    ScratchTile,
    /// Kernel applies scaling and writes a strided output tile.
    Direct,
}
/// Default and maximum cache blocksizes; max is reserved for tail merging.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Blocksizes {
    /// Row block (default, max), multiples of MR.
    pub mc: (usize, usize),
    /// Depth block (default, max).
    pub kc: (usize, usize),
    /// Column block (default, max), multiples of NR.
    pub nc: (usize, usize),
}
/// Operand capabilities checked at plan construction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Caps {
    /// Scatter operands can be packed.
    pub scatter_pack: bool,
    /// A conjugation is supported.
    pub conj_a: bool,
    /// B conjugation is supported.
    pub conj_b: bool,
}

/// Scratch-tile kernel: overwrite the entire declared tile with A·B.
/// Pointers must cover the family's packed panel and tile sizes. The tile may
/// be uninitialized: do not read it or form initialized references to it before
/// writing, and initialize every element required by the tile format and
/// logical geometry before returning (not unused extra `tile_bound` capacity).
pub type TileUkrFn<R> = unsafe fn(kc: usize, a: *const R, b: *const R, tile: *mut R);
/// Live-extent scratch-tile kernel: overwrite the entire declared tile with A·B
/// restricted to `m_live x n_live`, from zero.
///
/// The panels are the family's **full** packed panels — the strides are the
/// declared tile's, so the slivers are read exactly as [`TileUkrFn`] reads them —
/// and the tile is the family's full `mr x nr` accumulator tile. Only the
/// `m_live x n_live` live block needs K-loop work; every other slot of the
/// declared tile must still be initialized (zeroed is the natural choice) before
/// returning, because the write-back and the collect-grid permutation read the
/// declared tile, not the live extent. `m_live <= mr` and `n_live <= nr`.
pub type TileEdgeFn<R> =
    unsafe fn(kc: usize, a: *const R, b: *const R, tile: *mut R, m_live: usize, n_live: usize);
/// Direct kernel: `d = alpha_d*d + beta_ab*(A*B)` on an m×n tile.
/// A is packed with unit row stride; B and D use the supplied strides.
/// With alpha_d zero D may be uninitialized: the kernel must not read it or
/// form initialized references to it before storing. Otherwise D must be live.
/// Input values must be live, all addressed storage must be in bounds, and
/// output must not alias input panels. Initialize every addressed D element
/// before returning.
pub type DirectUkrFn<R> = unsafe fn(
    m: usize,
    n: usize,
    k: usize,
    d: *mut R,
    rs_d: isize,
    cs_d: isize,
    a: *const R,
    a_cs: isize,
    b: *const R,
    b_rs: isize,
    b_cs: isize,
    alpha_d: R,
    beta_ab: R,
    aux: &UkrAux<R>,
);
/// Typed kernel pointer; the variant must agree with `CUpdate`.
#[derive(Clone, Copy, Debug)]
pub enum UkrFn<R: 'static> {
    /// Overwrites scratch (no alpha/beta or output access).
    Tile(TileUkrFn<R>),
    /// Writes a strided output tile with alpha/beta.
    Direct(DirectUkrFn<R>),
}
/// Auxiliary state passed to direct and induced kernels.
#[derive(Clone, Copy, Debug)]
pub struct UkrAux<R: 'static> {
    /// Next A sliver for optional prefetching.
    pub a_next: *const R,
    /// Next B sliver for optional prefetching.
    pub b_next: *const R,
    /// Real kernel underlying an induced family.
    pub inner: Option<&'static KernelFamily<R>>,
    /// The family's [`KernelFamily::opaque`], for adapters whose kernel cannot
    /// be a plain `fn` pointer of this crate's signature.
    pub opaque: *const (),
}
/// Driver-facing view of a validated family: register geometry, packing
/// metadata and the kernel arm to dispatch on.
///
/// The driver needs this rather than [`Ukr`] alone because a
/// [`UkrFn::Direct`] family has no [`Ukr::func`] scratch kernel; both arms
/// share every other field, so one struct serves both.
#[derive(Clone, Copy, Debug)]
pub struct DriverFamily<R: 'static> {
    /// Micro-tile rows, in logical elements.
    pub mr: usize,
    /// Micro-tile columns.
    pub nr: usize,
    /// Reals in an A sliver per logical k-step.
    pub a_per_k: usize,
    /// Reals in a B sliver per logical k-step.
    pub b_per_k: usize,
    /// Reals in the accumulator tile.
    pub tile: usize,
    /// Layout the packed A sliver must have.
    pub a_pack: PackFormat,
    /// Layout the packed B sliver must have.
    pub b_pack: PackFormat,
    /// Layout a scratch accumulator tile is in.
    pub tile_fmt: TileFormat,
    /// How the driver supplies B.
    pub b_access: BAccess,
    /// The kernel arm.
    pub kernel: UkrFn<R>,
    /// Live-extent entry for the tile arm, if this family has one.
    ///
    /// `None` for every family whose kernel is not a plain real or native
    /// planar micro-kernel: the induced 1m/4m wrappers, the native 1m and 3m
    /// methods, the native interleaved complex kernel, the NEON and the scalar
    /// menus. A Direct family never has one; its own kernel already takes the
    /// live extent.
    pub edge: Option<TileEdgeFn<R>>,
    /// Value copied into [`UkrAux::opaque`] for each call.
    pub opaque: *const (),
    /// Induced families' inner real kernel, or none.
    pub inner: Option<&'static KernelFamily<R>>,
    /// Complex method, or none for real storage.
    pub method: Option<Method>,
}

// SAFETY: as for `KernelFamily`: `opaque` is immutable, process-constant
// adapter state and every other field is a plain value.
unsafe impl<R: 'static> Sync for DriverFamily<R> {}

impl<R: 'static> DriverFamily<R> {
    /// Reals of scratch the tile arm needs beyond its accumulator tile: the
    /// contiguous panels the 4m method copies planar packing into, and nothing
    /// at all for the other arms.
    pub fn induced_scratch(&self, kc: usize) -> usize {
        if self.method == Some(Method::FourM) {
            crate::induced::four_m_scratch(self.mr, self.nr, kc)
        } else {
            0
        }
    }
}

/// Monomorphized packer for one element type and layout.
/// Caller supplies valid scatter offsets and a sufficiently sized output;
/// every packed lane, including edge padding, is overwritten.
pub type PackFn<T> = unsafe fn(
    base: *const T,
    vscat: &[i64],
    vbs: &[i64],
    kscat: &[i64],
    vr: usize,
    conj: bool,
    out: *mut <T as Element>::Real,
);

/// Monomorphized write-back for one storage type and frozen tile format.
/// Caller supplies a fully overwritten tile, valid live scatters and disjoint
/// output. C is not read when beta is zero. All other raw writeback obligations
/// apply, including positive live extents no larger than MR/NR.
pub type EmitFn<T> = unsafe fn(
    *const <T as Element>::Real,
    usize,
    usize,
    usize,
    usize,
    T,
    T,
    *const T,
    &[i64],
    &[i64],
    i64,
    bool,
    *mut T,
    &[i64],
    &[i64],
    i64,
    bool,
);

/// Immutable descriptor keyed by the arithmetic real type.
/// Complex families are distinguished by `complex`, not by `R`.
/// The pointer's implementation must satisfy the described unsafe contract.
#[derive(Clone, Copy, Debug)]
pub struct KernelFamily<R: 'static> {
    /// Stable kernel identifier.
    pub id: &'static str,
    /// Implementation provenance.
    pub origin: Origin,
    /// Diagnostic ISA label.
    pub isa: Isa,
    /// CPU capabilities necessary to call the kernel.
    pub required: CpuFeatures,
    /// Reference, optimized or induced implementation.
    pub imp: KernelImpl,
    /// Auto priority; higher is preferred.
    pub priority: u16,
    /// Complex formats, or none for a real family.
    pub complex: Option<ComplexScheme>,
    /// Logical tile rows.
    pub mr: usize,
    /// Logical tile columns.
    pub nr: usize,
    /// Packed reals per A k-step.
    pub a_per_k: usize,
    /// Packed reals per B k-step.
    pub b_per_k: usize,
    /// Scratch tile capacity in reals.
    pub tile_bound: usize,
    /// Preferred output layout.
    pub c_pref: CPref,
    /// Kernel pointer.
    pub ukr: UkrFn<R>,
    /// Live-extent entry for the [`CUpdate::ScratchTile`] arm, or none.
    ///
    /// The same panels and the same declared tile as `ukr`'s tile arm, with the
    /// live row/column extents as extra arguments, so a partial micro-tile need
    /// not compute a whole register block (see [`TileEdgeFn`]).
    ///
    /// `None` unless the family's arithmetic is a plain real or native planar
    /// micro-kernel of this crate's x86 menus: an induced family's `ukr` is the
    /// *inner* real kernel and calling it directly here would bypass the 1m/4m
    /// transformation, so induced descriptors, the native 1m and 3m methods, the
    /// native interleaved complex kernel and the scalar/NEON menus all keep the
    /// full-tile entry.
    pub edge: Option<TileEdgeFn<R>>,
    /// B access policy.
    pub b_access: BAccess,
    /// Output update contract.
    pub c_update: CUpdate,
    /// Default/max cache block dimensions.
    pub blocks: Blocksizes,
    /// Supported operand operations.
    pub caps: Caps,
    /// Adapter state a shim reads back out of [`UkrAux::opaque`], or null.
    ///
    /// A family generated from a function that is not a plain `extern "Rust"
    /// fn` of this crate's own signature — a foreign microkernel reached
    /// through a context-free shim — passes the real pointer here and the shim
    /// reads it back, so no global lookup or per-call registration is needed.
    /// It must stay valid and immutable for the process lifetime.
    pub opaque: *const (),
    /// The real family this one's arithmetic is induced from, or none. Set
    /// only for [`KernelImpl::Induced`] families; the driver scales the inner
    /// kernel's k loop rather than calling `ukr` itself.
    pub inner: Option<&'static KernelFamily<R>>,
    /// Whether Auto may choose this family.
    pub allow_auto: bool,
}

/// Invalid descriptor geometry or format combination.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FamilyError {
    /// Invalid family's identifier.
    pub id: &'static str,
    /// Specific violated invariant.
    pub reason: &'static str,
}
impl core::fmt::Display for FamilyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "kernel {}: {}", self.id, self.reason)
    }
}
impl core::error::Error for FamilyError {}

impl<R: Real> KernelFamily<R> {
    /// Validate descriptor geometry and supported format combinations.
    ///
    /// # Errors
    /// Returns `FamilyError` naming the invalid size, packing/tile contract,
    /// complex scheme, or direct-update combination. Does not invoke kernels.
    pub fn validate(&self) -> Result<(), FamilyError> {
        let fail = |reason| {
            Err(FamilyError {
                id: self.id,
                reason,
            })
        };
        if self.mr == 0 {
            return fail("mr is zero");
        }
        if self.nr == 0 {
            return fail("nr is zero");
        }
        if !self.blocks.mc.0.is_multiple_of(self.mr) {
            return fail("mc default is not a multiple of mr");
        }
        if !self.blocks.nc.0.is_multiple_of(self.nr) {
            return fail("nc default is not a multiple of nr");
        }
        for (default, max) in [self.blocks.mc, self.blocks.kc, self.blocks.nc] {
            if default == 0 {
                return fail("default blocksize is zero");
            }
            if default > max {
                return fail("default blocksize exceeds max");
            }
        }
        if !self.blocks.mc.1.is_multiple_of(self.mr) || !self.blocks.nc.1.is_multiple_of(self.nr) {
            return fail("max blocksize is not a tile multiple");
        }
        let (a, b, tile) = match self.complex {
            None => (Layout::Real, Layout::Real, TileFormat::Real),
            Some(s) if supported(s) => (s.a, s.b, s.tile),
            Some(_) => return fail("complex scheme not in the supported table"),
        };
        let Some(size) = self
            .mr
            .checked_mul(self.nr)
            .and_then(|n| n.checked_mul(tile_planes(tile)))
        else {
            return fail("tile size overflow");
        };
        if self.tile_bound < size {
            return fail("tile_bound below tile size");
        }
        let a_len = self.mr.checked_mul(pack_format(a).reals_per_element());
        let b_len = self.nr.checked_mul(pack_format(b).reals_per_element());
        if a_len.is_none_or(|n| self.a_per_k < n) || b_len.is_none_or(|n| self.b_per_k < n) {
            return fail("packed k-step shorter than the tile");
        }
        if matches!(self.ukr, UkrFn::Direct(_)) != (self.c_update == CUpdate::Direct) {
            return fail("ukr kind disagrees with c_update");
        }
        if self.complex.is_some() && self.c_update == CUpdate::Direct {
            return fail("complex Direct kernels are not supported");
        }
        if matches!(self.b_access, BAccess::Direct { .. }) && self.c_update != CUpdate::Direct {
            return fail("direct B needs a Direct kernel");
        }
        if self.imp == KernelImpl::Induced {
            let Some(inner) = self.inner else {
                return fail("induced family has no inner real kernel");
            };
            if inner.complex.is_some() || inner.imp == KernelImpl::Induced {
                return fail("induced family's inner kernel is not a real family");
            }
        } else if self.inner.is_some() {
            return fail("only an induced family may carry an inner kernel");
        }
        Ok(())
    }

    /// Re-express a validated scratch family in the legacy driver contract.
    /// Returns none for direct kernels or unsupported complex schemes.
    pub fn as_ukr(&self) -> Option<Ukr<R>> {
        let UkrFn::Tile(func) = self.ukr else {
            return None;
        };
        let g = self.driver_family()?;
        Some(Ukr {
            mr: g.mr,
            nr: g.nr,
            a_per_k: g.a_per_k,
            b_per_k: g.b_per_k,
            tile: g.tile,
            a_pack: g.a_pack,
            b_pack: g.b_pack,
            tile_fmt: g.tile_fmt,
            func,
            name: self.id,
        })
    }

    /// Everything the raw driver needs: geometry, packing metadata and which
    /// kernel arm to dispatch on. Returns none for unsupported complex
    /// schemes, which validation rejects before selection can reach one.
    ///
    /// # Examples
    /// ```
    /// let f = tcontract_kernel::portable::families_f64()[0];
    /// assert_eq!(f.driver_family().unwrap().mr, f.mr);
    /// ```
    pub fn driver_family(&self) -> Option<DriverFamily<R>> {
        let (a_pack, b_pack, tile_fmt) = match self.complex {
            None => (PackFormat::Real, PackFormat::Real, TileFormat::Real),
            Some(s) if supported(s) => (pack_format(s.a), pack_format(s.b), s.tile),
            Some(_) => return None,
        };
        Some(DriverFamily {
            mr: self.mr,
            nr: self.nr,
            a_per_k: self.a_per_k,
            b_per_k: self.b_per_k,
            tile: self.tile_bound,
            a_pack,
            b_pack,
            tile_fmt,
            b_access: self.b_access,
            kernel: self.ukr,
            edge: self.edge,
            opaque: self.opaque,
            inner: self.inner,
            method: self.complex.map(|s| s.method),
        })
    }
}

// SAFETY: `opaque` points at immutable, process-constant adapter state (in
// practice a `fn` pointer), and every other field is a plain value, so sharing
// a family between threads is sharing read-only data.
unsafe impl<R: 'static> Sync for KernelFamily<R> {}

fn supported(s: ComplexScheme) -> bool {
    matches!(
        (s.method, s.a, s.b, s.tile),
        (
            Method::Native,
            Layout::Planar,
            Layout::Planar,
            TileFormat::Planar
        ) | (
            Method::Native,
            Layout::Interleaved,
            Layout::Interleaved,
            TileFormat::Interleaved
        ) | (
            Method::FourM,
            Layout::Planar,
            Layout::Planar,
            TileFormat::FourM
        ) | (Method::OneM, Layout::OneE, Layout::OneR, TileFormat::OneM)
            | (Method::OneM, Layout::OneR, Layout::OneE, TileFormat::OneM)
            | (
                Method::ThreeM,
                Layout::ThreeM,
                Layout::ThreeM,
                TileFormat::ThreeM
            )
    )
}
pub(crate) fn pack_format(layout: Layout) -> PackFormat {
    match layout {
        Layout::Real => PackFormat::Real,
        Layout::Planar | Layout::OneR => PackFormat::Planar,
        Layout::OneE => PackFormat::OneE,
        Layout::ThreeM => PackFormat::ThreeM,
        Layout::Interleaved => PackFormat::Interleaved,
    }
}
