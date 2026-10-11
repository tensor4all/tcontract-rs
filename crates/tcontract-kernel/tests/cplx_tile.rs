//! The leaf tile ABI of the native interleaved AVX2+FMA kernels: arithmetic,
//! full overwrite (NaN-poisoned scratch, `kc == 0` included) and exact
//! footprints (panels end flush against a `PROT_NONE` page, so a read past
//! `2*MR*kc` / `2*NR*kc` reals faults), and ISA gating.
#![cfg(all(target_arch = "x86_64", unix))]

use tcontract_kernel::*;

/// `len` elements of `R` that end exactly at an inaccessible page.
struct Guarded<R> {
    base: *mut u8,
    map_len: usize,
    ptr: *mut R,
    len: usize,
}

impl<R: Copy> Guarded<R> {
    fn new(len: usize, fill: impl Fn(usize) -> R) -> Self {
        let page = 4096usize;
        let bytes = (len * core::mem::size_of::<R>()).max(1);
        let pages = bytes.div_ceil(page);
        let map_len = (pages + 1) * page;
        // SAFETY: anonymous private mapping, protected below; freed in Drop.
        unsafe {
            let base = libc::mmap(
                core::ptr::null_mut(),
                map_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );
            assert_ne!(base, libc::MAP_FAILED);
            let base = base as *mut u8;
            assert_eq!(
                libc::mprotect(base.add(pages * page) as *mut _, page, libc::PROT_NONE),
                0
            );
            // Flush against the guard page; keep natural alignment of R.
            let ptr = base.add(pages * page - len * core::mem::size_of::<R>()) as *mut R;
            for i in 0..len {
                ptr.add(i).write(fill(i));
            }
            Self {
                base,
                map_len,
                ptr,
                len,
            }
        }
    }
    fn as_ptr(&self) -> *const R {
        self.ptr
    }
    fn as_mut_ptr(&mut self) -> *mut R {
        self.ptr
    }
    fn to_vec(&self) -> Vec<R> {
        // SAFETY: `len` initialized elements.
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }.to_vec()
    }
}

impl<R> Drop for Guarded<R> {
    fn drop(&mut self) {
        // SAFETY: the mapping created in `new`.
        unsafe { libc::munmap(self.base as *mut _, self.map_len) };
    }
}

fn avx2_fma() -> bool {
    CpuFeatures::detect().contains(CpuFeatures {
        avx2: true,
        fma: true,
        ..CpuFeatures::NONE
    })
}

trait Rl: Real + Into<f64> + Copy {
    fn from(x: f64) -> Self;
    const EPS: f64;
}
impl Rl for f32 {
    fn from(x: f64) -> Self {
        x as f32
    }
    const EPS: f64 = f32::EPSILON as f64;
}
impl Rl for f64 {
    fn from(x: f64) -> Self {
        x
    }
    const EPS: f64 = f64::EPSILON;
}

fn val(i: usize, salt: usize) -> f64 {
    let x = (i * 37 + salt * 11) % 23;
    x as f64 / 4.0 - 2.5
}

fn check<R: Rl>(f: &'static KernelFamily<R>, id: &str) {
    assert_eq!(f.id, id);
    f.validate().unwrap();
    let UkrFn::Tile(ukr) = f.ukr else {
        panic!("tile kernel")
    };
    let (mr, nr) = (f.mr, f.nr);
    for kc in [0usize, 1, 2, 3, 7, 64, 129] {
        let mut a = Guarded::<R>::new(2 * mr * kc, |i| R::from(val(i, 1)));
        let mut b = Guarded::<R>::new(2 * nr * kc, |i| R::from(val(i, 2)));
        // The tile is poisoned: the kernel must write every element.
        let mut tile = Guarded::<R>::new(f.tile_bound, |_| R::from(f64::NAN));
        // SAFETY: AVX2+FMA checked by the caller; exact panel and tile
        // footprints, distinct allocations.
        unsafe { ukr(kc, a.as_ptr(), b.as_ptr(), tile.as_mut_ptr()) };
        let got = tile.to_vec();
        let (av, bv) = (a.to_vec(), b.to_vec());
        let _ = (&mut a, &mut b);
        for j in 0..nr {
            for i in 0..mr {
                let (mut re, mut im, mut mag) = (0.0f64, 0.0f64, 0.0f64);
                for p in 0..kc {
                    let (ar, ai): (f64, f64) = (
                        av[p * 2 * mr + 2 * i].into(),
                        av[p * 2 * mr + 2 * i + 1].into(),
                    );
                    let (br, bi): (f64, f64) = (
                        bv[p * 2 * nr + 2 * j].into(),
                        bv[p * 2 * nr + 2 * j + 1].into(),
                    );
                    re += ar * br - ai * bi;
                    im += ai * br + ar * bi;
                    mag += (ar.hypot(ai)) * (br.hypot(bi));
                }
                let (gr, gi): (f64, f64) = (
                    got[2 * (j * mr + i)].into(),
                    got[2 * (j * mr + i) + 1].into(),
                );
                // Bound scaled to K and |A||B| (a loose relative denominator
                // would hide cancellation errors): 4 K eps |A||B|.
                let bound = 4.0 * (kc as f64 + 1.0) * R::EPS * mag;
                assert!(
                    (gr - re).abs() <= bound && (gi - im).abs() <= bound,
                    "{id} kc={kc} ({i},{j}): got ({gr},{gi}) want ({re},{im}) bound {bound:e}"
                );
            }
        }
        // Everything in the tile beyond 2*MR*NR reals is untouched (NaN).
        for x in &got[2 * mr * nr..] {
            assert!(Into::<f64>::into(*x).is_nan(), "{id}: wrote past the tile");
        }
    }
}

#[test]
fn c64_tile_is_correct_overwrites_and_reads_exactly_its_footprint() {
    if !avx2_fma() {
        eprintln!("skipping: no AVX2+FMA on this host (ISA body not executed)");
        return;
    }
    check(
        tcontract_kernel::kernels::cplx::families_f64()[0],
        "avx2.c64.native.4x4",
    );
}

#[test]
fn c32_tile_is_correct_overwrites_and_reads_exactly_its_footprint() {
    if !avx2_fma() {
        eprintln!("skipping: no AVX2+FMA on this host (ISA body not executed)");
        return;
    }
    check(
        tcontract_kernel::kernels::cplx::families_f32()[0],
        "avx2.c32.native.8x4",
    );
}

#[test]
fn tile_handles_pure_real_pure_imaginary_and_cancellation() {
    if !avx2_fma() {
        return;
    }
    let f = tcontract_kernel::kernels::cplx::families_f64()[0];
    let UkrFn::Tile(ukr) = f.ukr else { panic!() };
    let (mr, nr) = (f.mr, f.nr);
    // kc = 2: a = i, b = i then a = 1, b = 1: i*i + 1*1 = 0 exactly.
    let mut a = vec![0.0f64; 2 * mr * 2];
    let mut b = vec![0.0f64; 2 * nr * 2];
    for i in 0..mr {
        a[2 * i + 1] = 1.0; // step 0: i
        a[2 * mr + 2 * i] = 1.0; // step 1: 1
    }
    for j in 0..nr {
        b[2 * j + 1] = 1.0;
        b[2 * nr + 2 * j] = 1.0;
    }
    let mut tile = vec![f64::NAN; f.tile_bound];
    // SAFETY: AVX2+FMA present; exact footprints.
    unsafe { ukr(2, a.as_ptr(), b.as_ptr(), tile.as_mut_ptr()) };
    assert!(tile[..2 * mr * nr].iter().all(|&x| x == 0.0), "{tile:?}");
    // (1+2i)(3-4i) = 11 + 2i
    let mut a = vec![0.0f64; 2 * mr];
    let mut b = vec![0.0f64; 2 * nr];
    for i in 0..mr {
        a[2 * i] = 1.0;
        a[2 * i + 1] = 2.0;
    }
    for j in 0..nr {
        b[2 * j] = 3.0;
        b[2 * j + 1] = -4.0;
    }
    unsafe { ukr(1, a.as_ptr(), b.as_ptr(), tile.as_mut_ptr()) };
    assert!(tile[..2 * mr * nr].chunks(2).all(|z| z == [11.0, 2.0]));
}

#[test]
fn family_metadata_and_isa_gating() {
    for f in tcontract_kernel::kernels::cplx::families_f64() {
        check_meta(f, "c64", 4, 4);
        assert_eq!(f.b_per_k, 8);
        assert_eq!(f.tile_bound, 32);
        assert!(f.a_per_k == 8);
    }
    for f in tcontract_kernel::kernels::cplx::families_f32() {
        check_meta(f, "c32", 8, 4);
        assert_eq!(f.a_per_k, 16);
        assert_eq!(f.tile_bound, 64);
    }
    // A CPU without AVX2/FMA is refused at selection, never called.
    {
        let id = "avx2.c64.native.4x4";
        let masked = CpuFeatures {
            avx2: false,
            ..CpuFeatures::detect()
        };
        assert!(matches!(
            Registry::select::<C64>(id, masked),
            Err(SelectError::CpuUnsupported { .. })
        ));
        let no_fma = CpuFeatures {
            fma: false,
            ..CpuFeatures::detect()
        };
        assert!(matches!(
            Registry::select::<C64>(id, no_fma),
            Err(SelectError::CpuUnsupported { .. })
        ));
        // Listed even when unavailable, with the availability flag honest.
        let listed = list_kernels::<C64>();
        let info = listed.iter().find(|k| k.id == id).expect("listed");
        assert_eq!(info.available_on_this_cpu, avx2_fma());
    }
    assert!(matches!(
        Registry::select::<C32>("avx2.c32.native.8x4", CpuFeatures::NONE),
        Err(SelectError::CpuUnsupported { .. })
    ));
    // A c64 id is not a c32 family.
    assert!(matches!(
        Registry::select::<C32>("avx2.c64.native.4x4", CpuFeatures::detect()),
        Err(SelectError::DtypeMismatch { .. })
    ));
}

fn check_meta<R: Real>(f: &KernelFamily<R>, dtype: &str, mr: usize, nr: usize) {
    assert!(f.id.starts_with("avx2.") && f.id.contains(".native.") && f.id.contains(dtype));
    assert_eq!((f.mr, f.nr), (mr, nr));
    assert!(!f.allow_auto, "opt-in only");
    assert_eq!(f.isa, Isa::Avx2);
    assert!(f.required.avx2 && f.required.fma);
    assert_eq!(f.imp, KernelImpl::Optimized);
    assert_eq!(f.c_update, CUpdate::ScratchTile);
    assert!(matches!(f.b_access, BAccess::Packed));
    assert!(matches!(f.ukr, UkrFn::Tile(_)));
    let s = f.complex.unwrap();
    assert_eq!(
        (s.method, s.a, s.b, s.tile),
        (
            Method::Native,
            Layout::Interleaved,
            Layout::Interleaved,
            TileFormat::Interleaved
        )
    );
    assert!(f.caps.scatter_pack && f.caps.conj_a && f.caps.conj_b);
    assert_eq!(f.origin, Origin::Cplx);
    assert_eq!(f.origin.crate_name(), "tcontract-kernel");
    assert_eq!(f.origin.license(), "MIT OR Apache-2.0");
    f.validate().unwrap();
}

#[test]
fn auto_never_selects_these_families() {
    let cpu = CpuFeatures::detect();
    let head64 = Registry::families::<C64>(cpu, false)
        .into_iter()
        .find(|f| f.allow_auto)
        .unwrap();
    assert!(!(head64.id.starts_with("avx2.") && head64.id.contains(".native.")));
    let head32 = Registry::families::<C32>(cpu, false)
        .into_iter()
        .find(|f| f.allow_auto)
        .unwrap();
    assert!(!(head32.id.starts_with("avx2.") && head32.id.contains(".native.")));
}
