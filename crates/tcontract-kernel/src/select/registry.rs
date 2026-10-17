//! Process-wide immutable-family registration; workspace storage is not global.
//! Kernel providers depend on this contract, never the other way around.

use crate::{
    ComplexScheme, CpuFeatures, Element, FamilyError, Isa, KernelFamily, Method, Origin, Real, C32,
    C64,
};
use std::sync::Mutex;

type List<R> = fn() -> &'static [&'static KernelFamily<R>];

/// The default-menu cache for one real type: `(ISA, complex method)` to the
/// immutable built-in menu. One entry per combination, process-lifetime.
#[doc(hidden)]
pub type MenuCache<R> =
    std::sync::Mutex<Vec<((Isa, Option<Method>), &'static [&'static KernelFamily<R>])>>;
mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for num_complex::Complex<f32> {}
    impl Sealed for num_complex::Complex<f64> {}
}

/// Real types supported by the built-in registration slots.
/// Sealed: the four storage types are the only scalars the registry serves.
/// Mutable registration storage is not part of the public API.
///
/// ```compile_fail
/// use tcontract_kernel::RealSlot;
/// let _ = <f64 as RealSlot>::slot();
/// ```
/// The typed dispatch method cannot bypass the unsafe registration boundary.
/// ```compile_fail
/// use tcontract_kernel::{RealSlot, KernelFamily};
/// fn none() -> &'static [&'static KernelFamily<f64>] { &[] }
/// <f64 as RealSlot>::add_provider(none);
/// ```
pub trait RealSlot: Real + sealed::Sealed {
    /// Copy of provider callbacks; modifying it does not change the registry.
    #[doc(hidden)]
    fn providers() -> Vec<List<Self>>;
    /// Internal typed registration dispatch.
    ///
    /// # Safety
    /// The same ABI/ISA and immutable-manifest obligations as [`register`].
    #[doc(hidden)]
    unsafe fn add_provider(list: List<Self>);
}
macro_rules! real_slot {
    ($r:ty, $slot:ident) => {
        static $slot: Mutex<Vec<List<$r>>> = Mutex::new(Vec::new());
        impl RealSlot for $r {
            fn providers() -> Vec<List<Self>> {
                $slot.lock().unwrap_or_else(|e| e.into_inner()).clone()
            }
            unsafe fn add_provider(list: List<Self>) {
                let mut lists = $slot.lock().unwrap_or_else(|e| e.into_inner());
                if !lists.iter().any(|&f| core::ptr::fn_addr_eq(f, list)) {
                    lists.push(list);
                }
            }
        }
    };
}
real_slot!(f32, F32_PROVIDERS);
real_slot!(f64, F64_PROVIDERS);

/// Storage element types supported by the family registry.
/// Sealed to f32/f64/c32/c64; each uses its real type's registration slot.
pub trait Families: Element + sealed::Sealed {
    /// Scalar name used in diagnostic manifests.
    #[doc(hidden)]
    const DTYPE: &'static str;
    /// Unfiltered family list for this element's arithmetic real type.
    #[doc(hidden)]
    fn all_families() -> Vec<&'static KernelFamily<Self::Real>>;
    /// The immutable built-in families only, without registered providers.
    ///
    /// The default menu is cached from this list, so registration must not be
    /// able to change it; a registered provider is reached by explicit id.
    #[doc(hidden)]
    fn builtins() -> &'static [&'static KernelFamily<Self::Real>];
    /// The default-menu cache, one per element type, keyed by `(ISA, method)`.
    ///
    /// The menu is immutable (built-ins only) and bounded by the number of ISA
    /// and complex-method combinations, so it lives for the process.
    #[doc(hidden)]
    fn menu_cache() -> &'static MenuCache<Self::Real>;
    /// Complex families induced from every registered real family, built once.
    #[doc(hidden)]
    fn induced() -> &'static [&'static KernelFamily<Self::Real>];
}
macro_rules! families {
    ($t:ty, $r:ty, $name:literal, $builtin:path) => {
        impl Families for $t {
            const DTYPE: &'static str = $name;
            fn all_families() -> Vec<&'static KernelFamily<$r>> {
                registered::<$r>($builtin())
            }
            fn builtins() -> &'static [&'static KernelFamily<$r>] {
                $builtin()
            }
            fn menu_cache() -> &'static MenuCache<$r> {
                static CACHE: std::sync::OnceLock<MenuCache<$r>> = std::sync::OnceLock::new();
                CACHE.get_or_init(|| std::sync::Mutex::new(Vec::new()))
            }
            fn induced() -> &'static [&'static KernelFamily<$r>] {
                static INDUCED: std::sync::OnceLock<Vec<&'static KernelFamily<$r>>> =
                    std::sync::OnceLock::new();
                INDUCED.get_or_init(|| {
                    let mut out = Vec::new();
                    for real in Self::all_families().iter().filter(|f| f.complex.is_none()) {
                        out.extend(
                            [crate::induced::one_m(real), crate::induced::four_m(real)]
                                .into_iter()
                                .flatten(),
                        );
                    }
                    out
                })
            }
        }
    };
}
families!(f32, f32, "f32", crate::kernels::builtin_f32);
families!(f64, f64, "f64", crate::kernels::builtin_f64);
families!(C32, f32, "c32", crate::kernels::builtin_f32);
families!(C64, f64, "c64", crate::kernels::builtin_f64);

fn registered<R: RealSlot>(
    builtin: &'static [&'static KernelFamily<R>],
) -> Vec<&'static KernelFamily<R>> {
    let lists = R::providers();
    // Registration callbacks may initialize other providers; do not hold the
    // registry mutex while invoking them.
    builtin
        .iter()
        .copied()
        .chain(lists.into_iter().flat_map(|list| list().iter().copied()))
        .collect()
}

/// Register a provider's static descriptor list, idempotently by callback.
/// Lists are inspected at plan creation, not inside kernel execution.
///
/// Calling this boundary requires an explicit ABI/ISA safety promise.
/// ```compile_fail
/// use tcontract_kernel::{register, KernelFamily};
/// fn none() -> &'static [&'static KernelFamily<f64>] { &[] }
/// register::<f64>(none);
/// ```
///
/// # Safety
/// The callback must return an immutable, process-constant manifest. Each
/// accepted descriptor's function must implement its declared arithmetic,
/// panel strides/footprints, complete tile overwrite or direct-update ABI,
/// and CPU requirements. Geometry validation cannot prove these promises.
///
/// # Examples
/// ```
/// use tcontract_kernel::{register, KernelFamily};
/// fn none() -> &'static [&'static KernelFamily<f64>] { &[] }
/// // SAFETY: the immutable empty manifest has no kernel ABI obligations.
/// unsafe { register::<f64>(none); register::<f64>(none); }
/// ```
pub unsafe fn register<R: RealSlot>(list: List<R>) {
    // SAFETY: forwarded unchanged from the registration contract.
    unsafe { R::add_provider(list) };
}

/// Family lookup and CPU filtering. Lookups happen during planning.
#[derive(Debug)]
pub struct Registry;
impl Registry {
    /// Families for `T`, in descending stable priority order.
    /// With `include_unavailable`, also return families requiring other CPUs.
    ///
    /// # Examples
    /// ```
    /// use tcontract_kernel::{Registry, CpuFeatures};
    /// let families = Registry::families::<f64>(CpuFeatures::detect(), false);
    /// assert!(families.iter().all(|f| CpuFeatures::detect().contains(f.required)));
    /// ```
    pub fn families<T: Families>(
        cpu: CpuFeatures,
        include_unavailable: bool,
    ) -> Vec<&'static KernelFamily<T::Real>> {
        let mut families: Vec<_> = T::all_families()
            .into_iter()
            .chain(T::induced().iter().copied())
            .filter(|f| f.complex.is_some() == T::IS_COMPLEX)
            .filter(|f| include_unavailable || cpu.contains(f.required))
            .collect();
        families.sort_by_key(|f| core::cmp::Reverse(f.priority));
        // Enumerate each id once. Resolution rejects distinct descriptors
        // aliasing that id, rather than panicking on provider input.
        let mut seen = std::collections::HashSet::new();
        families.retain(|f| seen.insert(f.id));
        families
    }

    /// Resolve an explicit id without substituting another family.
    ///
    /// # Errors
    /// `UnknownId` for an unknown family, `DtypeMismatch` for another scalar, `CpuUnsupported` for
    /// missing instructions, or `Incompatible` for an invalid descriptor.
    ///
    /// # Examples
    /// ```
    /// use tcontract_kernel::{Registry, CpuFeatures, SelectError};
    /// assert!(matches!(Registry::select::<f64>("nonexistent.f64", CpuFeatures::NONE),
    ///     Err(SelectError::UnknownId { .. })));
    /// ```
    pub fn select<T: Families>(
        id: &str,
        cpu: CpuFeatures,
    ) -> Result<&'static KernelFamily<T::Real>, SelectError> {
        let all: Vec<_> = T::all_families()
            .into_iter()
            .chain(T::induced().iter().copied())
            .collect();
        if let Some(f) = all.iter().find(|f| f.id == id) {
            if all
                .iter()
                .any(|other| other.id == id && !core::ptr::eq(*other, *f))
            {
                return Err(SelectError::Incompatible {
                    id: id.into(),
                    reason: "duplicate kernel id",
                });
            }
            if f.complex.is_some() != T::IS_COMPLEX {
                return Err(SelectError::DtypeMismatch {
                    id: id.into(),
                    dtype: T::DTYPE,
                });
            }
            if !cpu.contains(f.required) {
                return Err(SelectError::CpuUnsupported {
                    id: id.into(),
                    missing: cpu.missing(f.required),
                });
            }
            f.validate()
                .map_err(|FamilyError { reason, .. }| SelectError::Incompatible {
                    id: id.into(),
                    reason,
                })?;
            return Ok(f);
        }
        let built: Vec<_> = list_kernels::<f32>()
            .into_iter()
            .chain(list_kernels::<f64>())
            .chain(list_kernels::<C32>())
            .chain(list_kernels::<C64>())
            .collect();
        if built.iter().any(|f| f.id == id) {
            return Err(SelectError::DtypeMismatch {
                id: id.into(),
                dtype: T::DTYPE,
            });
        }
        Err(SelectError::UnknownId { id: id.into() })
    }
}

/// Built-family metadata, independent of execution or workspace ownership.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelInfo {
    /// Stable identifier.
    pub id: &'static str,
    /// Provenance category.
    pub origin: Origin,
    /// Providing crate.
    pub crate_name: &'static str,
    /// Underlying implementation's license.
    pub license: &'static str,
    /// ISA label.
    pub isa: Isa,
    /// Storage scalar name.
    pub dtype: &'static str,
    /// Complex method/layouts, when applicable.
    pub complex: Option<ComplexScheme>,
    /// Logical tile rows.
    pub mr: usize,
    /// Logical tile columns.
    pub nr: usize,
    /// CPU availability (not part of the stable identifier).
    pub available_on_this_cpu: bool,
}

/// List all built families for `T`, including CPU-unavailable families.
///
/// # Examples
/// ```
/// use tcontract_kernel::list_kernels;
/// assert!(list_kernels::<f64>().iter().all(|info| info.dtype == "f64"));
/// ```
pub fn list_kernels<T: Families>() -> Vec<KernelInfo> {
    let cpu = CpuFeatures::detect();
    Registry::families::<T>(cpu, true)
        .into_iter()
        .map(|f| KernelInfo {
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

/// Selection failed at planning; forced choices never silently substitute.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SelectError {
    /// Unknown identifier.
    UnknownId {
        /// Requested identifier.
        id: String,
    },
    /// CPU lacks required instructions.
    CpuUnsupported {
        /// Requested identifier.
        id: String,
        /// Missing instruction-set capabilities.
        missing: CpuFeatures,
    },
    /// Identifier belongs to another scalar type.
    DtypeMismatch {
        /// Requested identifier.
        id: String,
        /// Scalar required by the plan.
        dtype: &'static str,
    },
    /// Family cannot represent this operation or descriptor is invalid.
    Incompatible {
        /// Requested identifier.
        id: String,
        /// Unsupported operand or invalid descriptor condition.
        reason: &'static str,
    },
    /// Policy is designed but not implemented.
    NotImplemented {
        /// Unimplemented policy's name.
        what: &'static str,
    },
    /// Two distinct descriptors in one caller-scoped catalog share an id.
    DuplicateId {
        /// The shared identifier.
        id: String,
    },
    /// A selector returned a handle that its catalog did not mint.
    ForeignHandle {
        /// Identifier of the rejected handle.
        id: String,
    },
    /// A selector returned a catalog member that was not an admissible
    /// candidate for this operation (for example a family that cannot convey
    /// the requested conjugation or complex method).
    NotACandidate {
        /// Identifier of the rejected handle.
        id: String,
        /// Why the family was not admissible.
        reason: &'static str,
    },
    /// The catalog holds no family admissible for this operation.
    NoCandidates {
        /// Storage scalar required by the plan.
        dtype: &'static str,
    },
    /// The caller's selector declined to choose; planning fails without a
    /// fallback.
    SelectorFailed {
        /// The selector's explanation.
        reason: String,
    },
}
impl core::fmt::Display for SelectError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnknownId { id } => write!(f, "unknown kernel id {id}; choose an id from list_kernels"),
            Self::CpuUnsupported { id, missing } => write!(f, "kernel {id} requires unavailable CPU features {missing:?}; choose an available kernel"),
            Self::DtypeMismatch { id, dtype } => write!(f, "kernel {id} does not serve {dtype}; choose a matching scalar family"),
            Self::Incompatible { id, reason } => write!(f, "kernel {id}: {reason}"),
            Self::NotImplemented { what } => write!(f, "{what} is not implemented; choose an implemented policy"),
            Self::DuplicateId { id } => write!(f, "kernel catalog holds two distinct descriptors with id {id}"),
            Self::ForeignHandle { id } => write!(f, "kernel handle {id} was not minted by the supplied catalog; select from the offered candidates"),
            Self::NotACandidate { id, reason } => write!(f, "kernel {id} is not an admissible candidate: {reason}"),
            Self::NoCandidates { dtype } => write!(f, "the catalog has no kernel admissible for this {dtype} operation"),
            Self::SelectorFailed { reason } => write!(f, "kernel selector failed: {reason}"),
        }
    }
}
impl core::error::Error for SelectError {}
