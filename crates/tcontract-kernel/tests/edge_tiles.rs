//! Which families carry a live-extent ("edge") micro-tile entry, and which
//! must not.
//!
//! The entry is what lets the driver stop paying for a whole register block on a
//! partial micro-tile (`TileEdgeFn`). It is only sound for a family whose `ukr`
//! tile arm *is* the micro-kernel: an induced 1m/4m descriptor's `ukr` is its
//! inner real kernel, which the wrapper calls with a doubled logical k (1m) or on
//! four copied real panels (4m), so an inherited edge entry would be called on
//! this family's own geometry and skip that transformation. The native 1m and 3m
//! methods and the native interleaved complex kernel are the same kind of
//! wrapper-free but different-geometry case, and NEON and the scalar menus simply
//! have no measured variant.
use tcontract_kernel::*;

/// Is this the plain real or native-planar micro-kernel of an x86 menu — the
/// one shape family the live-extent bodies are generated for? False for every
/// family of another target, which is what keeps the check below meaningful on
/// AArch64: there it asserts that NEON and the scalar menus carry no entry.
fn is_x86_plain_scratch(f: &KernelFamily<impl Real>) -> bool {
    let plain_method = match f.complex {
        None => true,
        Some(s) => s.method == Method::Native && s.a == Layout::Planar && s.b == Layout::Planar,
    };
    plain_method
        && matches!(f.isa, Isa::Avx2 | Isa::Avx512)
        && f.origin == Origin::Tensorcontract
        && f.imp == KernelImpl::Optimized
        && f.c_update == CUpdate::ScratchTile
        && f.b_access == BAccess::Packed
}

fn check<T: Families>() {
    let all = Registry::families::<T>(CpuFeatures::NONE, true);
    assert!(all.len() > 4, "the built-in manifest is not that small");
    for f in all {
        f.validate().unwrap_or_else(|e| panic!("{e}"));
        if is_x86_plain_scratch(f) {
            assert!(
                f.edge.is_some(),
                "{} is a plain x86 real/planar scratch family without a live-extent entry",
                f.id
            );
        } else {
            assert!(
                f.edge.is_none(),
                "{} must keep the full-tile entry (isa {:?}, imp {:?}, complex {:?})",
                f.id,
                f.isa,
                f.imp,
                f.complex
            );
        }
        // The driver only ever reads the entry off the driver-facing view, so
        // it has to travel there too - a descriptor field that never reaches
        // the driver is a silent no-op.
        if let Some(g) = f.driver_family() {
            assert_eq!(
                g.edge.is_some(),
                f.edge.is_some(),
                "{}: the driver-facing conversion dropped (or invented) the entry",
                f.id
            );
        }
    }
}

#[test]
fn only_the_plain_scratch_families_carry_a_live_extent_entry() {
    check::<f32>();
    check::<f64>();
    check::<C32>();
    check::<C64>();
}

/// The three families a reader is most likely to assume inherit the real entry,
/// by category, so that a regression names the kind it broke rather than one
/// family id.
#[test]
fn the_wrapped_methods_keep_the_full_tile_entry() {
    let cpu = CpuFeatures::NONE;
    for f in Registry::families::<C64>(cpu, true) {
        let method = f.complex.map(|s| s.method);
        if method == Some(Method::OneM) || method == Some(Method::ThreeM) {
            assert!(f.edge.is_none(), "{} is a wrapped method", f.id);
        }
        if f.complex.is_some_and(|s| s.a == Layout::Interleaved) {
            assert!(f.edge.is_none(), "{} is interleaved", f.id);
        }
        if f.imp == KernelImpl::Induced {
            assert!(f.edge.is_none(), "{} is induced", f.id);
            assert!(f.inner.is_some());
        }
        if f.isa == Isa::Portable {
            assert!(f.edge.is_none(), "{} is a scalar/portable family", f.id);
        }
    }
}

/// `one_m`/`four_m` build their descriptors with `..*real`. Every field the
/// base family has that is *not* overridden is inherited that way, so the entry
/// has to be overridden explicitly - for a base family that does carry one,
/// which is the only case where inheritance would show.
///
/// Only the x86 real/planar menus carry an entry (`is_x86_plain_scratch`), so
/// this is an x86-only test: on AArch64 the NEON and scalar menus have none, and
/// the assertion below has nothing to induce from rather than passing vacuously.
/// There, `only_the_plain_scratch_families_carry_a_live_extent_entry` still
/// asserts that every compiled family - induced ones included, since they are
/// built from a family without an entry - keeps `edge: None`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn induced_families_blank_the_inherited_entry() {
    let mut with_edge = 0usize;
    for f in Registry::families::<f64>(CpuFeatures::NONE, true) {
        if f.complex.is_some() || f.edge.is_none() {
            continue;
        }
        with_edge += 1;
        for induced in [induced::one_m(f), induced::four_m(f)]
            .into_iter()
            .flatten()
        {
            assert!(
                induced.edge.is_none(),
                "{} inherited the real family's live-extent entry",
                induced.id
            );
            induced.validate().unwrap_or_else(|e| panic!("{e}"));
        }
    }
    assert!(
        with_edge > 0,
        "no real family with an entry was available to induce from"
    );
}
