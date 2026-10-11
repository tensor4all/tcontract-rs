//! Test-only: the register-block menus of the built-in kernels, checked against
//! the reference at every entry point.
//!
//! The legacy `KernelSet` trait that used to hand these menus to the contraction
//! planner is gone: the planner resolves families from the registry. What
//! remains is the one thing the trait's test module was for -- every kernel
//! configuration the SIMD modules build must agree with the mathematical
//! definition, whatever packing and tile format it uses -- so the dispatch is
//! kept here as a private test trait over `f32` and `f64`.
//!
//! Source: lkdvos/tensorprimitives-rs, tensorcontract/src/kernel; MIT OR Apache-2.0.

use super::{reference::scalar, simd_isa};
use crate::cache;
use crate::*;

/// The per-real-type menu dispatch the tests drive: a vectorised module when the
/// target has one and the force selects it, the portable scalar kernels
/// otherwise.
trait KernelSet: Real + Sized {
    fn config_real(force: KernelForce) -> KernelConfig<Self>;
    fn config_cplx(force: KernelForce, method: ComplexMethod) -> KernelConfig<Self>;
    fn row_blocks(
        force: KernelForce,
        complex: bool,
        method: ComplexMethod,
    ) -> &'static [(usize, usize)] {
        let _ = (force, complex, method);
        &[]
    }
    fn config_at(
        force: KernelForce,
        complex: bool,
        method: ComplexMethod,
        i: usize,
    ) -> Option<KernelConfig<Self>> {
        let _ = (force, complex, method, i);
        None
    }
}

/// Wire one real type's [`KernelSet`] to the vectorised module when the target
/// has one, and to [`scalar`] otherwise.
///
/// The paths are `simd_isa::*`, so this is architecture-agnostic and the `#[cfg]`
/// asks only "is there a SIMD module at all". The dispatch functions answer
/// `None`/`&[]` when `force` selects no vectorised ISA (scalar pinned, or a
/// pin this target cannot honour), which sends the caller to the portable
/// scalar path. Off both families every arm below compiles out and the scalar
/// fallthrough is the whole body, which is why the `unreachable_code` and unused
/// allowances are needed on some targets and not others.
macro_rules! impl_kernel_set {
    ($t:ty, $real_simd:path, $cplx_simd:path, $rows_simd:path,
     $real_at_simd:path, $cplx_at_simd:path, $mr:literal, $nr:literal) => {
        impl KernelSet for $t {
            fn config_real(force: KernelForce) -> KernelConfig<Self> {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
                if let Some(c) = $real_simd(force) {
                    return c;
                }
                let _ = force;
                scalar::config_real::<$t, $mr, $nr>()
            }

            fn config_cplx(force: KernelForce, method: ComplexMethod) -> KernelConfig<Self> {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
                if let Some(c) = $cplx_simd(force, method) {
                    return c;
                }
                let _ = force;
                scalar::config_cplx::<$t, $mr, $nr>(method)
            }

            fn row_blocks(
                force: KernelForce,
                complex: bool,
                method: ComplexMethod,
            ) -> &'static [(usize, usize)] {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
                return $rows_simd(force, complex, method);
                // The portable path has one shape per method and no menu.
                #[allow(unreachable_code)]
                {
                    let _ = (force, complex, method);
                    &[]
                }
            }

            fn config_at(
                force: KernelForce,
                complex: bool,
                method: ComplexMethod,
                i: usize,
            ) -> Option<KernelConfig<Self>> {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
                return if complex {
                    $cplx_at_simd(force, method, i)
                } else {
                    $real_at_simd(force, i)
                };
                #[allow(unreachable_code)]
                {
                    let _ = (force, complex, method, i);
                    None
                }
            }
        }
    };
}

impl_kernel_set!(
    f64,
    simd_isa::config_real_f64,
    simd_isa::config_cplx_f64,
    simd_isa::row_blocks_f64,
    simd_isa::config_real_f64_at,
    simd_isa::config_cplx_f64_at,
    4,
    4
);
impl_kernel_set!(
    f32,
    simd_isa::config_real_f32,
    simd_isa::config_cplx_f32,
    simd_isa::row_blocks_f32,
    simd_isa::config_real_f32_at,
    simd_isa::config_cplx_f32_at,
    4,
    4
);

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    use crate::kernels::x86;

    /// The baseline tuning every test configuration is normalised with.
    fn tuning() -> Tuning {
        Tuning::default()
    }

    fn real<T: KernelSet>() -> KernelConfig<T> {
        T::config_real(KernelForce::Auto).normalise_for(&tuning(), 1)
    }

    fn cplx<T: KernelSet>(m: ComplexMethod) -> KernelConfig<T> {
        <T as KernelSet>::config_cplx(KernelForce::Auto, m).normalise_for(&tuning(), 1)
    }

    fn menu<T: KernelSet>(complex: bool, m: ComplexMethod) -> &'static [(usize, usize)] {
        <T as KernelSet>::row_blocks(KernelForce::Auto, complex, m)
    }

    fn at<T: KernelSet>(complex: bool, m: ComplexMethod, i: usize) -> Option<KernelConfig<T>> {
        <T as KernelSet>::config_at(KernelForce::Auto, complex, m, i)
            .map(|c| c.normalise_for(&tuning(), 1))
    }

    /// Every kernel must agree with the mathematical definition, whatever
    /// packing and tile format it uses. This is the contract the driver relies
    /// on, so it is checked directly rather than only end to end.
    fn check_real<T: KernelSet>(tol: f64) {
        check_real_cfg::<T>(real::<T>(), tol);
        for (i, &(mr, nr)) in menu::<T>(false, ComplexMethod::default())
            .iter()
            .enumerate()
        {
            let cfg = at::<T>(false, ComplexMethod::default(), i)
                .unwrap_or_else(|| panic!("real menu offers {mr}x{nr} with no kernel"));
            assert_eq!(
                (cfg.ukr.mr, cfg.ukr.nr),
                (mr, nr),
                "menu entry {i} misdescribes itself"
            );
            check_real_cfg::<T>(cfg, tol);
        }
    }

    fn check_real_cfg<T: KernelSet>(cfg: KernelConfig<T>, tol: f64) {
        let (mr, nr) = (cfg.ukr.mr, cfg.ukr.nr);
        let kc = 37usize;
        let a: Vec<T> = (0..cfg.ukr.a_per_k * kc)
            .map(|i| T::from_f64(((i * 37 % 19) as f64 - 9.0) / 7.0))
            .collect();
        let b: Vec<T> = (0..cfg.ukr.b_per_k * kc)
            .map(|i| T::from_f64(((i * 53 % 23) as f64 - 11.0) / 5.0))
            .collect();
        let mut got = vec![T::ZERO; cfg.ukr.tile];
        unsafe { (cfg.ukr.func)(kc, a.as_ptr(), b.as_ptr(), got.as_mut_ptr()) };

        for j in 0..nr {
            for i in 0..mr {
                let mut want = 0.0f64;
                for p in 0..kc {
                    want += a[p * mr + i].to_f64() * b[p * nr + j].to_f64();
                }
                let g = got[j * mr + i].to_f64();
                assert!(
                    (g - want).abs() <= tol * want.abs().max(1.0),
                    "{} real mismatch at ({i},{j}): {g} vs {want}",
                    cfg.ukr.name
                );
            }
        }
    }

    /// Read a complex value out of an accumulator tile.
    fn tile_value<T: Real>(
        ab: &[T],
        fmt: TileFormat,
        mr: usize,
        nr: usize,
        i: usize,
        j: usize,
    ) -> (f64, f64) {
        match fmt {
            TileFormat::Real => (ab[j * mr + i].to_f64(), 0.0),
            TileFormat::Planar => (ab[j * mr + i].to_f64(), ab[mr * nr + j * mr + i].to_f64()),
            TileFormat::Interleaved => (
                ab[2 * (j * mr + i)].to_f64(),
                ab[2 * (j * mr + i) + 1].to_f64(),
            ),
            TileFormat::FourM => {
                let plane = mr * nr;
                let off = j * mr + i;
                (
                    ab[off].to_f64() - ab[plane + off].to_f64(),
                    ab[2 * plane + off].to_f64() + ab[3 * plane + off].to_f64(),
                )
            }
            TileFormat::OneM => (
                ab[j * (2 * mr) + 2 * i].to_f64(),
                ab[j * (2 * mr) + 2 * i + 1].to_f64(),
            ),
            TileFormat::ThreeM => {
                let m1 = ab[j * mr + i].to_f64();
                let m2 = ab[mr * nr + j * mr + i].to_f64();
                let m3 = ab[2 * mr * nr + j * mr + i].to_f64();
                (m1 - m2, m3 - m1 - m2)
            }
        }
    }

    /// Build a packed sliver in the format the kernel expects, from complex
    /// values supplied as `(re, im)` pairs indexed `[k][lane]`.
    fn pack_for<T: Real>(fmt: PackFormat, vr: usize, vals: &[Vec<(f64, f64)>]) -> Vec<T> {
        let per_k = vr * fmt.reals_per_element();
        let mut out = vec![T::ZERO; per_k * vals.len()];
        for (p, row) in vals.iter().enumerate() {
            let base = p * per_k;
            for (t, &(re, im)) in row.iter().enumerate() {
                match fmt {
                    PackFormat::Real => out[base + t] = T::from_f64(re),
                    PackFormat::Interleaved => {
                        out[base + 2 * t] = T::from_f64(re);
                        out[base + 2 * t + 1] = T::from_f64(im);
                    }
                    PackFormat::Planar => {
                        out[base + t] = T::from_f64(re);
                        out[base + vr + t] = T::from_f64(im);
                    }
                    PackFormat::ThreeM => {
                        out[base + t] = T::from_f64(re);
                        out[base + vr + t] = T::from_f64(im);
                        out[base + 2 * vr + t] = T::from_f64(re + im);
                    }
                    PackFormat::OneE => {
                        out[base + 2 * t] = T::from_f64(re);
                        out[base + 2 * t + 1] = T::from_f64(im);
                        out[base + 2 * vr + 2 * t] = T::from_f64(-im);
                        out[base + 2 * vr + 2 * t + 1] = T::from_f64(re);
                    }
                }
            }
        }
        out
    }

    fn check_cplx<T: KernelSet>(method: ComplexMethod, tol: f64) {
        check_cplx_cfg::<T>(cplx::<T>(method), method, tol);
        for (i, &(mr, nr)) in menu::<T>(true, method).iter().enumerate() {
            let cfg = at::<T>(true, method, i).unwrap_or_else(|| {
                panic!("{} menu offers {mr}x{nr} with no kernel", method.name())
            });
            assert_eq!(
                (cfg.ukr.mr, cfg.ukr.nr),
                (mr, nr),
                "menu entry {i} misdescribes itself"
            );
            check_cplx_cfg::<T>(cfg, method, tol);
        }
    }

    fn check_cplx_cfg<T: KernelSet>(cfg: KernelConfig<T>, method: ComplexMethod, tol: f64) {
        let (mr, nr) = (cfg.ukr.mr, cfg.ukr.nr);
        let kc = 29usize;

        let av: Vec<Vec<(f64, f64)>> = (0..kc)
            .map(|p| {
                (0..mr)
                    .map(|i| {
                        (
                            (((p * mr + i) * 31 % 17) as f64 - 8.0) / 6.0,
                            (((p * mr + i) * 13 % 11) as f64 - 5.0) / 3.0,
                        )
                    })
                    .collect()
            })
            .collect();
        let bv: Vec<Vec<(f64, f64)>> = (0..kc)
            .map(|p| {
                (0..nr)
                    .map(|j| {
                        (
                            (((p * nr + j) * 41 % 13) as f64 - 6.0) / 4.0,
                            (((p * nr + j) * 7 % 19) as f64 - 9.0) / 5.0,
                        )
                    })
                    .collect()
            })
            .collect();

        let a = pack_for::<T>(cfg.ukr.a_pack, mr, &av);
        let b = pack_for::<T>(cfg.ukr.b_pack, nr, &bv);
        assert_eq!(a.len(), cfg.ukr.a_per_k * kc);
        assert_eq!(b.len(), cfg.ukr.b_per_k * kc);

        let mut got = vec![T::ZERO; cfg.ukr.tile];
        unsafe { (cfg.ukr.func)(kc, a.as_ptr(), b.as_ptr(), got.as_mut_ptr()) };

        #[allow(clippy::needless_range_loop)]
        for j in 0..nr {
            for i in 0..mr {
                let (mut wr, mut wi) = (0.0f64, 0.0f64);
                for p in 0..kc {
                    let (ar, ai) = av[p][i];
                    let (br, bi) = bv[p][j];
                    wr += ar * br - ai * bi;
                    wi += ar * bi + ai * br;
                }
                let (gr, gi) = tile_value(&got, cfg.ukr.tile_fmt, mr, nr, i, j);
                assert!(
                    (gr - wr).abs() <= tol * wr.abs().max(1.0),
                    "{} [{}] re mismatch at ({i},{j}): {gr} vs {wr}",
                    cfg.ukr.name,
                    method.name()
                );
                assert!(
                    (gi - wi).abs() <= tol * wi.abs().max(1.0),
                    "{} [{}] im mismatch at ({i},{j}): {gi} vs {wi}",
                    cfg.ukr.name,
                    method.name()
                );
            }
        }
    }

    #[test]
    fn kernels_match_reference_f64() {
        check_real::<f64>(1e-12);
        for m in ComplexMethod::ALL {
            check_cplx::<f64>(m, 1e-12);
        }
    }

    #[test]
    fn kernels_match_reference_f32() {
        check_real::<f32>(1e-4);
        for m in ComplexMethod::ALL {
            check_cplx::<f32>(m, 1e-4);
        }
    }

    /// Every kernel family the *CPU* can run, not just the one dispatch would
    /// choose, against the same contract.
    ///
    /// This is what exercises the AVX2 kernels on an AVX-512 machine under a
    /// plain `cargo test`. Without it they would be compiled and never
    /// executed here, and the only coverage would be a pinned-ISA run someone
    /// has to remember to do.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn check_isa<T: KernelSet>(s: x86::IsaConfigs<T>, tol: f64) {
        let default = {
            let c = (s.real)();
            (c.ukr.mr, c.ukr.nr)
        };
        let menu = (s.row_blocks)(false, ComplexMethod::default());
        assert_eq!(menu[0], default, "{} real menu head", s.isa.name());
        check_real_cfg::<T>((s.real)(), tol);
        for (i, &shape) in menu.iter().enumerate() {
            // Duplicate `MR` is legal since A35 — an `NR`-only alternate is the
            // whole point of positional keying — but a duplicate *shape* is not:
            // it is a menu slot that can never be chosen over the earlier one.
            assert!(
                !menu[..i].contains(&shape),
                "{} real {shape:?} twice",
                s.isa.name()
            );
            let cfg = (s.config_at)(false, ComplexMethod::default(), i)
                .unwrap_or_else(|| panic!("{} real menu has no kernel at {i}", s.isa.name()));
            assert_eq!((cfg.ukr.mr, cfg.ukr.nr), shape);
            check_real_cfg::<T>(cfg, tol);
        }
        for m in ComplexMethod::ALL {
            let default = {
                let c = (s.cplx)(m);
                (c.ukr.mr, c.ukr.nr)
            };
            let menu = (s.row_blocks)(true, m);
            assert_eq!(menu[0], default, "{} {} menu head", s.isa.name(), m.name());
            check_cplx_cfg::<T>((s.cplx)(m), m, tol);
            for (i, &shape) in menu.iter().enumerate() {
                assert!(
                    !menu[..i].contains(&shape),
                    "{} {} {shape:?} twice",
                    s.isa.name(),
                    m.name()
                );
                let cfg = (s.config_at)(true, m, i).unwrap_or_else(|| {
                    panic!("{} {} menu has no kernel at {i}", s.isa.name(), m.name())
                });
                assert_eq!((cfg.ukr.mr, cfg.ukr.nr), shape);
                check_cplx_cfg::<T>(cfg, m, tol);
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn every_available_x86_isa_matches_reference() {
        for &isa in x86::available_isas() {
            check_isa::<f64>(x86::isa_configs_f64(isa), 1e-12);
            check_isa::<f32>(x86::isa_configs_f32(isa), 1e-4);
        }
    }

    /// The `Ukr` contract is what makes the kernels interchangeable, so a new
    /// instruction set must not quietly bring a different packing convention
    /// with it: the driver, the packing traversal and the write-back are shared
    /// and know nothing about the ISA. Shapes may differ; formats may not.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn x86_isas_agree_on_the_pack_contract() {
        fn check<T: KernelSet>(sets: &[x86::IsaConfigs<T>]) {
            let Some((first, rest)) = sets.split_first() else {
                return;
            };
            let fmts = |c: &KernelConfig<T>| (c.ukr.a_pack, c.ukr.b_pack, c.ukr.tile_fmt);
            for s in rest {
                assert_eq!(
                    fmts(&(s.real)()),
                    fmts(&(first.real)()),
                    "{} vs {} real formats",
                    s.isa.name(),
                    first.isa.name()
                );
                for m in ComplexMethod::ALL {
                    assert_eq!(
                        fmts(&(s.cplx)(m)),
                        fmts(&(first.cplx)(m)),
                        "{} vs {} {} formats",
                        s.isa.name(),
                        first.isa.name(),
                        m.name()
                    );
                }
            }
            // And the widths must stay derivable from the shape alone, for
            // every entry on every menu — that is what `pack` assumes.
            for s in sets {
                for (complex, m) in ComplexMethod::ALL
                    .iter()
                    .map(|&m| (true, m))
                    .chain(core::iter::once((false, ComplexMethod::default())))
                {
                    for i in 0..(s.row_blocks)(complex, m).len() {
                        // `normalise` is what the `KernelSet` impls apply, and
                        // it is where the register-block alignment of `mc`/`nc`
                        // that the driver's loop arithmetic relies on comes from.
                        let c = (s.config_at)(complex, m, i)
                            .unwrap()
                            .normalise_for(&tuning(), 1);
                        let planes = c.ukr.tile / (c.ukr.mr * c.ukr.nr);
                        assert_eq!(c.ukr.a_per_k, c.ukr.mr * c.ukr.a_pack.reals_per_element());
                        assert_eq!(c.ukr.b_per_k, c.ukr.nr * c.ukr.b_pack.reals_per_element());
                        assert_eq!(c.ukr.tile, planes * c.ukr.mr * c.ukr.nr);
                        assert_eq!(c.blk.mc % c.ukr.mr, 0);
                        assert_eq!(c.blk.nc % c.ukr.nr, 0);
                    }
                }
            }
        }
        let isas = x86::available_isas();
        check::<f64>(
            &isas
                .iter()
                .map(|&i| x86::isa_configs_f64(i))
                .collect::<Vec<_>>(),
        );
        check::<f32>(
            &isas
                .iter()
                .map(|&i| x86::isa_configs_f32(i))
                .collect::<Vec<_>>(),
        );
    }

    /// A repeated *shape* is a menu slot nothing can ever select, since the rule
    /// and every override resolve to the first match; and the head of the menu
    /// must be exactly what the default builders return, or `base` and `idx=0`
    /// would mean different things.
    ///
    /// A repeated `MR` is deliberately **allowed** — that is what positional
    /// keying bought (A35), and it is the only way an `NR`-only alternate can be
    /// on the menu at all. What it costs is that `RowBlock::Pin(mr)`
    /// can no longer name such an entry; `RowBlock::Index(i)` is the way to reach it, which
    /// is what that spelling always implied.
    #[test]
    fn row_block_menus_are_well_formed() {
        fn check<T: KernelSet>(complex: bool, method: ComplexMethod, default: (usize, usize)) {
            let menu = menu::<T>(complex, method);
            if menu.is_empty() {
                return; // no vectorised kernels on this CPU
            }
            assert_eq!(menu[0], default, "menu head is not the default shape");
            for (i, &shape) in menu.iter().enumerate() {
                assert!(shape.0 > 0 && shape.1 > 0);
                assert!(
                    !menu[..i].contains(&shape),
                    "{shape:?} appears twice on a menu, so the later one is unreachable"
                );
            }
        }
        fn shape<T: KernelSet>(c: KernelConfig<T>) -> (usize, usize) {
            (c.ukr.mr, c.ukr.nr)
        }
        check::<f64>(false, ComplexMethod::default(), shape(real::<f64>()));
        check::<f32>(false, ComplexMethod::default(), shape(real::<f32>()));
        for m in ComplexMethod::ALL {
            check::<f64>(true, m, shape(cplx::<f64>(m)));
            check::<f32>(true, m, shape(cplx::<f32>(m)));
        }
    }

    /// The contraction planner builds its default menu from the registry: the
    /// built-in families of the preferred ISA and the requested complex scheme,
    /// in registry order. That must be the menu the dispatch above offers, entry
    /// for entry, or a row-block index would name a different shape than it
    /// always did.
    #[test]
    fn the_registry_menu_is_the_dispatch_menu() {
        use crate::{Families, Method, Origin, Registry};
        fn check<T: Families>(method: ComplexMethod, want_method: Method)
        where
            T::Real: KernelSet,
        {
            for force in [KernelForce::Auto, KernelForce::Scalar] {
                let want = <T::Real as KernelSet>::row_blocks(force, T::IS_COMPLEX, method);
                let isa = super::super::menu_isa(force);
                let got: Vec<(usize, usize)> =
                    Registry::families::<T>(CpuFeatures::detect(), false)
                        .into_iter()
                        .filter(|f| {
                            f.imp != KernelImpl::Induced
                                && f.origin == Origin::Tensorcontract
                                && f.isa == isa
                                && (!T::IS_COMPLEX
                                    || f.complex.is_some_and(|s| s.method == want_method))
                        })
                        .map(|f| (f.mr, f.nr))
                        .collect();
                // The dispatch menu is empty exactly when no vectorised module
                // answers; the registry then offers only the portable family.
                if !want.is_empty() {
                    assert_eq!(got, want, "{} {method:?} {force:?}", T::DTYPE);
                } else {
                    assert!(got.len() <= 1, "{} {method:?} {force:?}: {got:?}", T::DTYPE);
                }
            }
        }
        check::<f32>(ComplexMethod::Planar, Method::Native);
        check::<f64>(ComplexMethod::Planar, Method::Native);
        for (m, kind) in [
            (ComplexMethod::Planar, Method::Native),
            (ComplexMethod::OneM, Method::OneM),
            (ComplexMethod::ThreeM, Method::ThreeM),
        ] {
            check::<C32>(m, kind);
            check::<C64>(m, kind);
        }
    }

    #[test]
    fn panel_sizes_are_self_consistent() {
        for m in ComplexMethod::ALL {
            let c = cplx::<f64>(m);
            assert_eq!(c.ukr.a_per_k, c.ukr.mr * c.ukr.a_pack.reals_per_element());
            assert_eq!(c.ukr.b_per_k, c.ukr.nr * c.ukr.b_pack.reals_per_element());
            assert_eq!(c.blk.mc % c.ukr.mr, 0);
            assert_eq!(c.blk.nc % c.ukr.nr, 0);
        }
    }

    #[test]
    fn one_m_packs_a_twice_as_large_as_planar() {
        // The structural cost of 1m, and the reason its MC must be smaller.
        let planar = cplx::<f64>(ComplexMethod::Planar);
        let onem = cplx::<f64>(ComplexMethod::OneM);
        assert_eq!(planar.ukr.a_pack.reals_per_element(), 2);
        assert_eq!(onem.ukr.a_pack.reals_per_element(), 4);
        assert_eq!(onem.ukr.b_pack, planar.ukr.b_pack, "B is 1r either way");
        // Same L2 budget, so twice the reals per element buys half the rows.
        let bytes =
            |c: &KernelConfig<f64>| c.blk.mc * c.blk.kc * c.ukr.a_pack.reals_per_element() * 8;
        assert!(
            bytes(&onem) <= bytes(&planar) * 3 / 2,
            "1m packed A block {} vs planar {}",
            bytes(&onem),
            bytes(&planar)
        );
    }

    #[test]
    fn three_m_does_fewer_flops() {
        let planar = cplx::<f64>(ComplexMethod::Planar);
        let threem = cplx::<f64>(ComplexMethod::ThreeM);
        // 3 planes of accumulator instead of 2, but 3 products instead of 4.
        assert_eq!(threem.ukr.tile, 3 * threem.ukr.mr * threem.ukr.nr);
        assert_eq!(planar.ukr.tile, 2 * planar.ukr.mr * planar.ukr.nr);
    }

    /// Re-deriving a config at a plan's thread count must be safe to do on top
    /// of the derivation that already happened. Under the legacy constants that
    /// means doing nothing at all — the percentage overrides scale the derived
    /// value, so a second pass would square them.
    #[test]
    fn retargeting_threads_is_stable() {
        for m in ComplexMethod::ALL {
            for model in [cache::BlockModel::Legacy, cache::BlockModel::Analytical] {
                let tuning = Tuning {
                    block_model: model,
                    ..Tuning::default()
                };
                let cfg = f64::config_cplx(KernelForce::Auto, m).normalise_for(&tuning, 1);
                let again = cfg.retarget_threads(&tuning, 8);
                match model {
                    cache::BlockModel::Legacy => assert_eq!(cfg.blk, again.blk),
                    // The model may legitimately give eight threads a narrower
                    // `nc` — they crowd each other's packed `A` out of the
                    // shared L3 — but never a different `mc` or `kc`, and a
                    // second application must be a fixed point.
                    cache::BlockModel::Analytical => {
                        assert_eq!((cfg.blk.mc, cfg.blk.kc), (again.blk.mc, again.blk.kc));
                        assert_eq!(again.blk, again.retarget_threads(&tuning, 8).blk);
                    }
                }
            }
        }
    }

    #[test]
    fn method_parsing_roundtrips() {
        for m in ComplexMethod::ALL {
            assert_eq!(ComplexMethod::parse(m.name()), Some(m));
            // The trait impls must be the inherent pair, not a second spelling
            // of it: a CSV column written by one and read by the other is the
            // normal case here.
            assert_eq!(m.to_string(), m.name());
            assert_eq!(m.to_string().parse::<ComplexMethod>(), Ok(m));
        }
        assert_eq!(ComplexMethod::parse("nope"), None);
        assert!("nope".parse::<ComplexMethod>().is_err());
    }
}
