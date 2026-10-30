//! Project-owned portable reference kernels (MIT OR Apache-2.0), not a port.
//! Ordinary dot-product loops, without intrinsics. Family/context separation
//! follows BLIS (Van Zee & van de Geijn, TOMS 2015).

use crate::*;

/// Overwrite a column-major real MR×NR tile with packed A·B.
///
/// # Examples
/// ```
/// let a = [1.0_f64; 8]; let b = [1.0_f64; 8]; let mut tile = [f64::NAN; 16];
/// // SAFETY: two full 4-lane slivers and a distinct 4x4 output tile.
/// unsafe { tcontract_kernel::portable::real_tile::<f64, 4, 4>(
///     2, a.as_ptr(), b.as_ptr(), tile.as_mut_ptr()) };
/// assert_eq!(tile, [2.0; 16]);
/// ```
///
/// # Safety
/// MR/NR are positive; A/B cover `kc*MR`/`kc*NR` readable reals and tile
/// covers `MR*NR` writable reals. Output does not alias either input.
pub unsafe fn real_tile<R: Real, const MR: usize, const NR: usize>(
    kc: usize,
    a: *const R,
    b: *const R,
    tile: *mut R,
) {
    let mut acc = [[R::ZERO; MR]; NR];
    // INVARIANT: the caller supplies complete packed slivers and the full
    // writable tile, including padded lanes; accumulation needs zero starts.
    // SAFETY: the function's panel/tile extent and non-aliasing contract.
    unsafe {
        for p in 0..kc {
            let ap = a.add(p * MR);
            let bp = b.add(p * NR);
            for (j, col) in acc.iter_mut().enumerate() {
                let bj = *bp.add(j);
                for (i, sum) in col.iter_mut().enumerate() {
                    *sum += *ap.add(i) * bj;
                }
            }
        }
        for (j, col) in acc.iter().enumerate() {
            for (i, &value) in col.iter().enumerate() {
                *tile.add(j * MR + i) = value;
            }
        }
    }
}

/// Native complex product of interleaved panels into an interleaved tile.
///
/// # Examples
/// ```
/// let a = [1.0_f64; 8]; // four values 1+i
/// let b = [0.0_f64, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]; // four values i
/// let mut tile = [f64::NAN; 32];
/// // SAFETY: full 4-lane complex panels and a distinct 4x4 complex tile.
/// unsafe { tcontract_kernel::portable::cplx_tile::<f64, 4, 4>(
///     1, a.as_ptr(), b.as_ptr(), tile.as_mut_ptr()) };
/// assert!(tile.chunks_exact(2).all(|z| z == [-1.0, 1.0]));
/// ```
///
/// # Safety
/// MR/NR are positive. A/B cover `2*kc*MR`/`2*kc*NR` readable reals and
/// tile covers `2*MR*NR` writable reals, without aliasing either input.
pub unsafe fn cplx_tile<R: Real, const MR: usize, const NR: usize>(
    kc: usize,
    a: *const R,
    b: *const R,
    tile: *mut R,
) {
    let mut re = [[R::ZERO; MR]; NR];
    let mut im = [[R::ZERO; MR]; NR];
    // INVARIANT: full packed slivers and tile are covered by the caller;
    // only the accumulators are zero-initialized, not the output buffer.
    // SAFETY: the function's extent and non-aliasing contract.
    unsafe {
        for p in 0..kc {
            let ap = a.add(p * 2 * MR);
            let bp = b.add(p * 2 * NR);
            for (j, (rcol, icol)) in re.iter_mut().zip(im.iter_mut()).enumerate() {
                let (br, bi) = (*bp.add(2 * j), *bp.add(2 * j + 1));
                for (i, (r, z)) in rcol.iter_mut().zip(icol.iter_mut()).enumerate() {
                    let (ar, ai) = (*ap.add(2 * i), *ap.add(2 * i + 1));
                    *r += ar * br - ai * bi;
                    *z += ar * bi + ai * br;
                }
            }
        }
        for (j, (rcol, icol)) in re.iter().zip(im.iter()).enumerate() {
            for (i, (&r, &z)) in rcol.iter().zip(icol.iter()).enumerate() {
                *tile.add(2 * (j * MR + i)) = r;
                *tile.add(2 * (j * MR + i) + 1) = z;
            }
        }
    }
}

/// Direct-update real micro-kernel: `D = alpha_d·D + beta_ab·(A·B)`.
///
/// One of the two arms the driver dispatches on ([`UkrFn::Direct`]). `A` is a
/// packed column-major panel with unit row stride, `B` and `D` are addressed by
/// the strides the driver derives, and with `alpha_d == 0` the old `D` is never
/// read — which is how `beta == 0` and the scratch-tile fallback are expressed.
///
/// Packing B and letting the kernel read the panel (`b_rs = NR`, `b_cs = 1`) and
/// reading `B` in place (`b_rs` the k stride, `b_cs` the column stride) are the
/// same call here; the family's `b_access` decides which the driver supplies.
///
/// # Examples
/// ```
/// // D += A·B with A a 4x2 column-major panel and B a 2x1 column:
/// // AB = [2, 3] and D starts at [5, 7].
/// let a = [1.0_f64, 0., 0., 0., 0., 1., 0., 0.];
/// let b = [2.0_f64, 3.0];
/// let mut d = [5.0_f64, 7., 0., 0.];
/// let aux = tcontract_kernel::UkrAux {
///     a_next: core::ptr::null(),
///     b_next: core::ptr::null(),
///     inner: None,
///     opaque: core::ptr::null(),
/// };
/// // SAFETY: full 4x2 A panel, a 2x1 B and four writable D reals.
/// unsafe { tcontract_kernel::portable::real_direct::<f64, 4, 4>(
///     4, 1, 2, d.as_mut_ptr(), 1, 4, a.as_ptr(), 4,
///     b.as_ptr(), 1, 2, 1.0, 1.0, &aux) };
/// assert_eq!(d, [7.0, 10., 0., 0.]);
/// ```
///
/// # Safety
/// `MR`/`NR` are positive and `m <= MR`, `n <= NR`. A covers `k*MR` readable
/// reals; B covers `k*|b_rs| + (n-1)*b_cs` and D the same in `rs_d`/`cs_d`, all
/// addresses valid. `D` does not alias the A panel, and is read only when
/// `alpha_d` is nonzero.
pub unsafe fn real_direct<R: Real, const MR: usize, const NR: usize>(
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
    _aux: &UkrAux<R>,
) {
    let mut acc = [[R::ZERO; MR]; NR];
    // INVARIANT: only the accumulators are zero-initialized; the caller covers
    // A, B and D, and `m`/`n` are the live extent of a possibly edge tile.
    // SAFETY: the function's index-extent and non-aliasing contract.
    unsafe {
        for p in 0..k {
            let ap = a.offset(p as isize * a_cs);
            for (j, col) in acc.iter_mut().enumerate().take(n) {
                let bj = *b.offset(p as isize * b_rs + j as isize * b_cs);
                for (i, sum) in col.iter_mut().enumerate().take(m) {
                    *sum += *ap.add(i) * bj;
                }
            }
        }
        for (j, col) in acc.iter().enumerate().take(n) {
            for (i, &value) in col.iter().enumerate().take(m) {
                let dij = d.offset(i as isize * rs_d + j as isize * cs_d);
                *dij = if alpha_d == R::ZERO {
                    beta_ab * value
                } else {
                    alpha_d * *dij + beta_ab * value
                };
            }
        }
    }
}

macro_rules! families {
    ($r:ty, $real:ident, $cplx:ident, $direct:ident, $direct_b:ident, $list:ident,
     $real_id:literal, $cplx_id:literal, $direct_id:literal, $direct_b_id:literal) => {
        static $real: KernelFamily<$r> = KernelFamily {
            id: $real_id,
            origin: Origin::Portable,
            isa: Isa::Portable,
            required: CpuFeatures::NONE,
            imp: KernelImpl::Reference,
            priority: 0,
            complex: None,
            mr: 4,
            nr: 4,
            a_per_k: 4,
            b_per_k: 4,
            tile_bound: 16,
            c_pref: CPref::Any,
            ukr: UkrFn::Tile(real_tile::<$r, 4, 4>),
            // The portable reference loops already take the live extent in
            // their direct arm and have no vectorised tile to shorten.
            edge: None,
            b_access: BAccess::Packed,
            c_update: CUpdate::ScratchTile,
            blocks: Blocksizes {
                mc: (64, 64),
                kc: (256, 256),
                nc: (1024, 1024),
            },
            caps: Caps {
                scatter_pack: true,
                conj_a: true,
                conj_b: true,
            },
            opaque: core::ptr::null(),
            inner: None,
            allow_auto: true,
        };
        static $cplx: KernelFamily<$r> = KernelFamily {
            id: $cplx_id,
            complex: Some(ComplexScheme {
                method: Method::Native,
                a: Layout::Interleaved,
                b: Layout::Interleaved,
                tile: TileFormat::Interleaved,
            }),
            a_per_k: 8,
            b_per_k: 8,
            tile_bound: 32,
            ukr: UkrFn::Tile(cplx_tile::<$r, 4, 4>),
            ..$real
        };
        static $direct: KernelFamily<$r> = KernelFamily {
            id: $direct_id,
            ukr: UkrFn::Direct(real_direct::<$r, 4, 4>),
            c_update: CUpdate::Direct,
            allow_auto: false,
            ..$real
        };
        static $direct_b: KernelFamily<$r> = KernelFamily {
            id: $direct_b_id,
            b_access: BAccess::Direct {
                unit_stride: Axis::Col,
            },
            ..$direct
        };
        /// Static real and native-interleaved complex fallback descriptors.
        #[doc = concat!("\n# Examples\n```\nlet families = tcontract_kernel::portable::",
                                    stringify!($list), "();\nassert_eq!(families.len(), 4);\n",
                                    "assert!(families.iter().all(|f| f.validate().is_ok()));\n```")]
        pub fn $list() -> &'static [&'static KernelFamily<$r>] {
            static LIST: [&KernelFamily<$r>; 4] = [&$real, &$cplx, &$direct, &$direct_b];
            &LIST
        }
    };
}
families!(
    f32,
    REAL32,
    CPLX32,
    DIRECT32,
    DIRECT_B32,
    families_f32,
    "ref.f32.real.4x4",
    "ref.c32.native.4x4",
    "ref.f32.direct.4x4",
    "ref.f32.direct-b.4x4"
);
families!(
    f64,
    REAL64,
    CPLX64,
    DIRECT64,
    DIRECT_B64,
    families_f64,
    "ref.f64.real.4x4",
    "ref.c64.native.4x4",
    "ref.f64.direct.4x4",
    "ref.f64.direct-b.4x4"
);
