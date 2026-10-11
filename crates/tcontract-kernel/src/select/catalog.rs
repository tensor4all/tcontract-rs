//! Caller-scoped kernel catalogs and opaque trusted handles.
//!
//! A downstream crate that implements its own packed microkernels admits their
//! static descriptors once, through the single `unsafe` constructor
//! [`KernelCatalog::from_static_families`], and later *selects* among them with
//! safe code. A selector never sees or returns a raw descriptor, pointer or
//! numeric slot: it returns a [`KernelHandle`], which can only be minted by a
//! catalog and carries that catalog's identity, so a handle from another
//! catalog is rejected rather than trusted.
//!
//! The catalog is an immutable list owned by the caller. It is not a second
//! process registry, owns no workspace and is not consulted during execution:
//! a plan keeps the selected handle, whose descriptor and code are `'static`.
//! New design for issue #28; arithmetic and packing contracts are unchanged
//! from [`KernelFamily`].
//!
//! # What safe downstream code cannot do
//!
//! A handle is minted only by an admitted catalog. It cannot be built from its
//! parts or mutated (see [`KernelHandle`]), converted from a number or pointer,
//! ```compile_fail,E0277
//! let _: tcontract_kernel::KernelHandle<f64> = 3usize.into();
//! ```
//! made from a raw descriptor, as a selector's return value would have to be,
//! ```compile_fail,E0308
//! fn pick(f: &'static tcontract_kernel::KernelFamily<f64>)
//!     -> Result<tcontract_kernel::KernelHandle<f64>, tcontract_kernel::SelectError> {
//!     Ok(f)
//! }
//! ```
//! or asked for the descriptor behind it:
//! ```compile_fail,E0624
//! fn peek(h: tcontract_kernel::KernelHandle<f64>) { let _ = h.family(); }
//! ```
//! A handle of one storage type is not a handle of another:
//! ```compile_fail,E0308
//! fn widen(h: tcontract_kernel::KernelHandle<f64>) -> tcontract_kernel::KernelHandle<tcontract_kernel::C64> { h }
//! ```
//! And the admission constructor is `unsafe`:
//! ```compile_fail,E0133
//! let _ = tcontract_kernel::KernelCatalog::<f64>::from_static_families(&[]);
//! ```

use crate::{
    CpuFeatures, Families, Isa, KernelFamily, KernelImpl, Origin, Registry, SelectError,
    {BAccess, Blocksizes, CPref, CUpdate, Caps, ComplexScheme},
};
use core::marker::PhantomData;
use core::sync::atomic::{AtomicU64, Ordering};

/// Identity source for catalogs: a handle remembers the catalog that minted it.
static NEXT_CATALOG: AtomicU64 = AtomicU64::new(1);

/// An opaque, trusted reference to one admitted kernel family for storage type
/// `T` (so `c64` and `f64` handles are different types).
///
/// Handles are `Copy` and read-only. There is no safe constructor, no mutable
/// access to the descriptor and no getter for the kernel function or its
/// opaque state: the only ways to obtain one are
/// [`KernelCatalog::from_static_families`] (unsafe, provider-checked),
/// [`KernelCatalog::builtin`] and the candidate list a selector receives.
///
/// A handle cannot be built from its parts in safe code:
/// ```compile_fail,E0451
/// use tcontract_kernel::KernelHandle;
/// let h: KernelHandle<f64> = KernelHandle { family: todo!(), catalog: 0, _t: todo!() };
/// ```
/// and it exposes no mutable geometry:
/// ```compile_fail,E0616
/// fn poke(mut h: tcontract_kernel::KernelHandle<f64>) { h.family = todo!(); }
/// ```
pub struct KernelHandle<T: Families> {
    family: &'static KernelFamily<T::Real>,
    catalog: u64,
    _t: PhantomData<fn() -> T>,
}
impl<T: Families> Clone for KernelHandle<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: Families> Copy for KernelHandle<T> {}
impl<T: Families> PartialEq for KernelHandle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.catalog == other.catalog && core::ptr::eq(self.family, other.family)
    }
}
impl<T: Families> Eq for KernelHandle<T> {}
impl<T: Families> core::fmt::Debug for KernelHandle<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("KernelHandle")
            .field("id", &self.family.id)
            .field("dtype", &T::DTYPE)
            .finish()
    }
}

impl<T: Families> KernelHandle<T> {
    /// Stable identifier of the admitted family.
    pub fn id(&self) -> &'static str {
        self.family.id
    }
    /// Provenance reported in diagnostics.
    pub fn origin(&self) -> Origin {
        self.family.origin
    }
    /// Diagnostic ISA label.
    pub fn isa(&self) -> Isa {
        self.family.isa
    }
    /// CPU capabilities the kernel needs.
    pub fn required(&self) -> CpuFeatures {
        self.family.required
    }
    /// Reference, optimized or induced implementation.
    pub fn implementation(&self) -> KernelImpl {
        self.family.imp
    }
    /// Complex method and packed/tile layouts; none for a real family.
    pub fn complex(&self) -> Option<ComplexScheme> {
        self.family.complex
    }
    /// Logical tile rows.
    pub fn mr(&self) -> usize {
        self.family.mr
    }
    /// Logical tile columns.
    pub fn nr(&self) -> usize {
        self.family.nr
    }
    /// Packed reals per A k-step.
    pub fn a_per_k(&self) -> usize {
        self.family.a_per_k
    }
    /// Packed reals per B k-step.
    pub fn b_per_k(&self) -> usize {
        self.family.b_per_k
    }
    /// Preferred output layout (a hint).
    pub fn c_pref(&self) -> CPref {
        self.family.c_pref
    }
    /// How the family reads B.
    pub fn b_access(&self) -> BAccess {
        self.family.b_access
    }
    /// Who applies alpha/beta and stores the output.
    pub fn c_update(&self) -> CUpdate {
        self.family.c_update
    }
    /// Default/maximum cache blocksizes.
    pub fn blocks(&self) -> Blocksizes {
        self.family.blocks
    }
    /// Operand capabilities.
    pub fn caps(&self) -> Caps {
        self.family.caps
    }
    /// Auto priority; higher is preferred.
    pub fn priority(&self) -> u16 {
        self.family.priority
    }

    /// The trusted descriptor, for binding by the planner. Crate-private: a
    /// safe caller never holds the descriptor of a handle.
    pub(crate) fn family(&self) -> &'static KernelFamily<T::Real> {
        self.family
    }
}

/// Immutable, caller-scoped list of kernel families with minted handles.
///
/// # Examples
/// ```
/// use tcontract_kernel::KernelCatalog;
/// let catalog = KernelCatalog::<f64>::builtin();
/// assert!(catalog.handles().any(|h| h.id() == "ref.f64.real.4x4"));
/// // A handle belongs to the catalog that minted it.
/// let other = KernelCatalog::<f64>::builtin();
/// let h = catalog.get("ref.f64.real.4x4").unwrap();
/// assert!(catalog.contains(&h) && !other.contains(&h));
/// ```
pub struct KernelCatalog<T: Families> {
    id: u64,
    families: Vec<&'static KernelFamily<T::Real>>,
    _t: PhantomData<fn() -> T>,
}
impl<T: Families> core::fmt::Debug for KernelCatalog<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("KernelCatalog")
            .field("dtype", &T::DTYPE)
            .field(
                "ids",
                &self.families.iter().map(|k| k.id).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl<T: Families> KernelCatalog<T> {
    fn mint(families: Vec<&'static KernelFamily<T::Real>>) -> Self {
        Self {
            id: NEXT_CATALOG.fetch_add(1, Ordering::Relaxed),
            families,
            _t: PhantomData,
        }
    }

    /// Check the descriptor facts every admitted family must satisfy.
    fn check(family: &'static KernelFamily<T::Real>) -> Result<(), SelectError> {
        if family.complex.is_some() != T::IS_COMPLEX {
            return Err(SelectError::DtypeMismatch {
                id: family.id.into(),
                dtype: T::DTYPE,
            });
        }
        family.validate().map_err(|e| SelectError::Incompatible {
            id: family.id.into(),
            reason: e.reason,
        })?;
        if family.driver_family().is_none() {
            return Err(SelectError::Incompatible {
                id: family.id.into(),
                reason: "family has no driver-expressible kernel",
            });
        }
        Ok(())
    }

    /// Admit a provider's static descriptors as one catalog.
    ///
    /// This is the single point where the provider promises that its code is
    /// correct; selecting an admitted [`KernelHandle`] is safe.
    ///
    /// # Errors
    /// `DtypeMismatch` when a descriptor's real/complex kind is not `T`'s,
    /// `Incompatible` for malformed geometry, formats or sizes (the checks of
    /// [`KernelFamily::validate`]), and `DuplicateId` when two distinct
    /// descriptors share an id.
    ///
    /// # Safety
    /// Every descriptor must be immutable and process-constant, and its kernel
    /// function (and any `opaque` state) must:
    /// * implement the declared arithmetic for the declared packed layouts,
    ///   panel strides and footprints (`a_per_k`/`b_per_k` per k-step,
    ///   `tile_bound` reals of tile), reading and writing nothing outside them;
    /// * overwrite the complete tile (scratch kernels) or honour the documented
    ///   direct-update ABI, including not reading `D` when `alpha_d` is zero;
    /// * require no CPU capability beyond `required` (a truthful ISA mask);
    /// * be safe to call concurrently from several threads, with immutable
    ///   auxiliary state, and never panic or unwind: a worker that disappears
    ///   inside a barrier-bearing region can deadlock the rest of the team.
    ///
    /// Geometry validation cannot prove any of these properties.
    ///
    /// # Examples
    /// ```
    /// use tcontract_kernel::{portable, KernelCatalog};
    /// // SAFETY: the project's own portable kernels meet the obligations.
    /// let catalog = unsafe { KernelCatalog::<f64>::from_static_families(portable::families_f64()) };
    /// // Real and complex storage share `f64` descriptors, but not catalogs.
    /// assert!(catalog.is_err(), "the list also holds a complex family");
    /// ```
    pub unsafe fn from_static_families(
        families: &'static [&'static KernelFamily<T::Real>],
    ) -> Result<Self, SelectError> {
        let mut seen: Vec<&'static KernelFamily<T::Real>> = Vec::with_capacity(families.len());
        for &family in families {
            Self::check(family)?;
            if let Some(&prior) = seen.iter().find(|prior| prior.id == family.id) {
                if !core::ptr::eq(prior, family) {
                    return Err(SelectError::DuplicateId {
                        id: family.id.into(),
                    });
                }
                continue;
            }
            seen.push(family);
        }
        Ok(Self::mint(seen))
    }

    /// A safe snapshot of every built-in and registered family for `T` that
    /// this CPU can run, in descending priority. Providers must already be
    /// registered (`tcontract_blas::builtin_catalog` registers the workspace's).
    pub fn builtin() -> Self {
        Self::mint(Registry::families::<T>(CpuFeatures::detect(), false))
    }

    /// A new catalog holding this catalog's families and `other`'s.
    ///
    /// Handles of the sources are *foreign* to the result; fetch fresh ones
    /// with [`get`](Self::get) or [`handles`](Self::handles). This is how a
    /// custom set opts into a built-in fallback.
    ///
    /// # Errors
    /// `DuplicateId` when distinct descriptors share an id.
    pub fn union(&self, other: &Self) -> Result<Self, SelectError> {
        let mut families = self.families.clone();
        for &family in &other.families {
            match families.iter().find(|f| f.id == family.id) {
                Some(prior) if !core::ptr::eq(*prior, family) => {
                    return Err(SelectError::DuplicateId {
                        id: family.id.into(),
                    })
                }
                Some(_) => {}
                None => families.push(family),
            }
        }
        Ok(Self::mint(families))
    }

    /// Number of admitted families.
    pub fn len(&self) -> usize {
        self.families.len()
    }
    /// Whether the catalog is empty.
    pub fn is_empty(&self) -> bool {
        self.families.is_empty()
    }
    /// Handles of every admitted family, in admission order.
    pub fn handles(&self) -> impl Iterator<Item = KernelHandle<T>> + '_ {
        self.families.iter().map(|&family| KernelHandle {
            family,
            catalog: self.id,
            _t: PhantomData,
        })
    }
    /// The handle with the given id, if admitted here.
    pub fn get(&self, id: &str) -> Option<KernelHandle<T>> {
        self.handles().find(|h| h.id() == id)
    }
    /// Whether `handle` was minted by this catalog.
    pub fn contains(&self, handle: &KernelHandle<T>) -> bool {
        handle.catalog == self.id
            && self
                .families
                .iter()
                .any(|f| core::ptr::eq(*f, handle.family))
    }
    /// Metadata of every admitted family, including provenance.
    pub fn list(&self) -> Vec<crate::KernelInfo> {
        let cpu = CpuFeatures::detect();
        self.families
            .iter()
            .map(|f| crate::KernelInfo {
                id: f.id,
                origin: f.origin,
                crate_name: f.origin.crate_name(),
                license: f.origin.license(),
                isa: f.isa,
                dtype: T::DTYPE,
                complex: f.complex,
                mr: f.mr,
                nr: f.nr,
                available_on_this_cpu: cpu.contains(f.required),
            })
            .collect()
    }

    /// Whether this catalog's `handle` can run on `cpu`, as a typed error.
    ///
    /// # Errors
    /// `ForeignHandle` for a handle of another catalog, `CpuUnsupported` when
    /// the ISA mask is not met.
    pub fn admit(&self, handle: &KernelHandle<T>, cpu: CpuFeatures) -> Result<(), SelectError> {
        if !self.contains(handle) {
            return Err(SelectError::ForeignHandle {
                id: handle.id().into(),
            });
        }
        if !cpu.contains(handle.family.required) {
            return Err(SelectError::CpuUnsupported {
                id: handle.id().into(),
                missing: cpu.missing(handle.family.required),
            });
        }
        Ok(())
    }
}
