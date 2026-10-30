//! Live-extent ("edge") micro-tiles through the packed driver.
//!
//! A partial micro-tile — the last `MR` block of a row panel or the last `NR`
//! block of a column panel — used to cost a whole register block. The families
//! that carry a `TileEdgeFn` now compute only their live extent and zero the rest
//! of the declared tile; everything here is about *proving that path ran* and
//! that it agrees with the reference:
//!
//! * every `(m_live, n_live)` in `1..=MR x 1..=NR`, with the family forced by id
//!   and the family id re-read from the plan report (so a plan that quietly took
//!   another family fails rather than passes);
//! * `beta = 0` into a poisoned (NaN) output, so an unwritten output element
//!   fails rather than passes;
//! * more than one `KC` block, nontrivial `alpha`/`beta`, a separate `C`, and
//!   operand conjugation;
//! * the collect-grid path, with a partial final column and a nontrivial row
//!   permutation;
//! * the paths that must *not* change: Direct-B/Direct-C, and the wrapped
//!   methods (native 1m/3m, the interleaved complex kernel, the induced 1m/4m
//!   descriptors), which keep the full-tile entry.
//!
//! # Which families, on which target
//!
//! The live-extent entry exists only on the x86 real and native-planar menus
//! (`tcontract-kernel/tests/edge_tiles.rs`), and on AArch64 those families are
//! not compiled at all rather than present-but-unsupported. So the x86 ids are
//! collected only on x86, and every sweep also runs the portable `ref.*.4x4`
//! families, which exist on every target: there the sweep proves the *fallback*
//! (a partial tile keeps the full-tile entry) and keeps the collect-grid
//! assertion exercised off x86. A family that is compiled but cannot run on this
//! CPU (an `avx512.*` id on an AVX2-only host) is skipped; an id the manifest does
//! not have is a broken test and fails (`available`).
//!
//! What each layer proves, so that no assertion here is read as more than it is:
//! this file forces the *family* (by id, re-checked against the report) and
//! checks the *values* of a partial tile end to end, but a family that ran its
//! full-tile entry instead would produce the same values. That the live-extent
//! body itself is reached, is correct at every extent, and leaves the whole
//! declared tile initialised is pinned one layer down, where the entry is called
//! directly at every `(m_live, n_live)` from a poisoned tile
//! (`tcontract-kernel/src/kernels/menu_tests.rs`,
//! `edge_entries_match_reference_at_every_live_extent`), that the entry exists
//! for exactly these families is pinned in `tcontract-kernel/tests/edge_tiles.rs`,
//! and that the driver *dispatches* to it is pinned by the counting wrapper in
//! `tests/edge_dispatch.rs`.
use num_complex::Complex64;
use tcontract::api::{
    AccumulationSource, CSpec, DType, DotGeneral, Labels, LayoutSpec, Op, OperandSpec, Problem,
    Scalar,
};
use tcontract::{Algorithm, Orient, PackedReport, Plan, PlanConfig};
use tcontract_exec::{ArenaProvider, Exec};
use tcontract_kernel::{
    tile_planes, BlockingOverride, CpuFeatures, Element, Families, KernelChoice, Real, Registry,
    SelectError,
};

mod common;
use common::{for_each_index, out_dims, reference, rel_err, T};

// ---------------------------------------------------------------------------
// Forcing a family
// ---------------------------------------------------------------------------

/// Whether this CPU can run `id` for storage type `T`.
///
/// Two outcomes that must not be confused: a family this target *compiles* but
/// this CPU cannot execute is skipped (only an AVX-512 host can run the
/// `avx512.*` entries; this one is an AVX2-only EPYC), while an id the manifest
/// does not have at all is a broken test rather than a skip. That is why the
/// callers only list ids of families their target compiles.
fn available<T: Families>(id: &str) -> bool {
    match Registry::select::<T>(id, CpuFeatures::detect()) {
        Ok(_) => true,
        Err(SelectError::CpuUnsupported { .. }) => false,
        Err(e) => panic!("{id} is not a usable family id: {e}"),
    }
}

/// The skip/fail split above is the point of `available`: a misspelled id fails
/// instead of quietly shrinking the sweep. (The skip branch needs an `avx512.*`
/// family on a host without usable AVX-512, which a test cannot arrange on an
/// AVX-512 host; `every_live_extent_matches_the_reference` covers it where it
/// applies.)
#[test]
#[should_panic(expected = "is not a usable family id")]
fn an_unknown_family_id_fails_rather_than_being_skipped() {
    assert!(!available::<f64>("ref.f64.real.9999x9999"));
}

/// The families `Auto` selects on x86, real and native-planar, both precisions.
///
/// Gated to x86: on AArch64 these families are not compiled into the manifest at
/// all (`tcontract-kernel/src/kernels/mod.rs`), so listing one here would be an
/// unknown id, not an unsupported CPU.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn x86_families() -> Vec<(&'static str, DType)> {
    let mut out: Vec<(&'static str, DType)> = Vec::new();
    let mut push = |id: &'static str, dtype: DType, ok: bool| {
        if ok {
            out.push((id, dtype));
        }
    };
    push(
        "avx2.f64.real.8x6",
        DType::F64,
        available::<f64>("avx2.f64.real.8x6"),
    );
    push(
        "avx2.c64.planar.4x5",
        DType::C64,
        available::<tcontract_kernel::C64>("avx2.c64.planar.4x5"),
    );
    push(
        "avx2.f32.real.16x6",
        DType::F32,
        available::<f32>("avx2.f32.real.16x6"),
    );
    push(
        "avx2.c32.planar.8x5",
        DType::C32,
        available::<tcontract_kernel::C32>("avx2.c32.planar.8x5"),
    );
    push(
        "avx512.f64.real.24x8",
        DType::F64,
        available::<f64>("avx512.f64.real.24x8"),
    );
    push(
        "avx512.c64.planar.16x6",
        DType::C64,
        available::<tcontract_kernel::C64>("avx512.c64.planar.16x6"),
    );
    push(
        "avx512.f32.real.48x8",
        DType::F32,
        available::<f32>("avx512.f32.real.48x8"),
    );
    push(
        "avx512.c32.planar.32x6",
        DType::C32,
        available::<tcontract_kernel::C32>("avx512.c32.planar.32x6"),
    );
    out
}

/// The portable reference families, which every target compiles.
///
/// None of them carries a live-extent entry (the scalar loops have no vectorised
/// tile to shorten), so on this side the sweeps prove the *fallback*: a partial
/// tile keeps the full-tile entry and stays correct.
fn portable_families() -> Vec<(&'static str, DType)> {
    let mut out: Vec<(&'static str, DType)> = Vec::new();
    for (id, dtype) in [
        ("ref.f64.real.4x4", DType::F64),
        ("ref.c64.native.4x4", DType::C64),
        ("ref.f32.real.4x4", DType::F32),
        ("ref.c32.native.4x4", DType::C32),
    ] {
        let runnable = match dtype {
            DType::F64 => available::<f64>(id),
            DType::F32 => available::<f32>(id),
            DType::C64 => available::<tcontract_kernel::C64>(id),
            DType::C32 => available::<tcontract_kernel::C32>(id),
        };
        assert!(runnable, "{id} is portable and must run on every target");
        out.push((id, dtype));
    }
    out
}

/// Every family the sweeps force: the portable reference menus on all targets,
/// and the x86 real/planar menus where they are compiled.
fn forced_families() -> Vec<(&'static str, DType)> {
    #[allow(unused_mut)] // INVARIANT: x86 appends its vectorised menus below.
    let mut out = portable_families();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    out.extend(x86_families());
    assert!(!out.is_empty(), "no family is available on this host");
    out
}

fn tol<S: Scalar>() -> f64 {
    if core::mem::size_of::<S::Re>() == 4 {
        1e-4
    } else {
        1e-12
    }
}

/// Run `body::<S>(id)` once per available forced family, at that family's
/// storage type.
macro_rules! for_each_family {
    ($body:ident) => {
        for (id, dtype) in forced_families() {
            match dtype {
                DType::F64 => $body::<f64>(id),
                DType::F32 => $body::<f32>(id),
                DType::C64 => $body::<Complex64>(id),
                DType::C32 => $body::<num_complex::Complex32>(id),
            }
        }
    };
}

// ---------------------------------------------------------------------------
// One case
// ---------------------------------------------------------------------------

/// How the accumulation term `C` is described.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CMode {
    /// No `C` term at all: the plan is only ever called through
    /// `execute_into`, which is `beta = 0` and reads no output.
    Absent,
    /// `C` is `D` itself: a later `KC` slab reads back what the previous one
    /// emitted.
    Output,
    /// A separately described `C` with its own storage.
    Separate,
}

/// A `T` laid out column-major, or with `order`'s axes physically permuted (axis
/// `order[0]` fastest).
fn tensor<S: Scalar>(dims: &[usize], seed: u64, order: Option<&[usize]>) -> T<S> {
    let t = T::<S>::new(dims, seed);
    match order {
        None => t,
        Some(o) => t.restride(o),
    }
}

/// How one case is set up.
struct Spec {
    /// Free-axis extents of the row role, in the role's own order, and `n`, `k`.
    /// A single entry is a plain matmul; two entries make the row role
    /// axis-aware, which is what the blocked traversal needs.
    row: Vec<usize>,
    n: usize,
    k: usize,
    /// Physical axis order of `A`, fastest first; `None` is column-major.
    a_order: Option<Vec<usize>>,
    ca: bool,
    cb: bool,
    /// `alpha.re`; the imaginary part is half of it.
    alpha: f64,
    /// `beta.re`, with `-beta.re` as its imaginary part.
    beta: f64,
    c: CMode,
    /// `KC` block override, to force more than one K slab.
    kc: Option<usize>,
    /// Pin the orientation. The blocked traversal needs the row role to be the
    /// two-axis side, so the case that reaches it pins `Force(false)`.
    orient: Option<bool>,
}

impl Spec {
    fn matmul(m: usize, n: usize, k: usize) -> Self {
        Self {
            row: vec![m],
            n,
            k,
            a_order: None,
            ca: false,
            cb: false,
            alpha: 1.5,
            beta: 0.0,
            c: CMode::Absent,
            kc: Some(16),
            orient: None,
        }
    }
}

struct Case<S: Scalar> {
    label: String,
    plan: Plan<S>,
    a: T<S>,
    b: T<S>,
    c: Option<T<S>>,
    d: T<S>,
    want: T<S>,
    alpha: S,
    beta: S,
    c_mode: CMode,
}

impl<S: Scalar> Case<S> {
    fn new(id: &str, spec: &Spec) -> Self {
        let rank = spec.row.len() + 1;
        let a_dims: Vec<usize> = spec.row.iter().copied().chain([spec.k]).collect();
        let a = tensor::<S>(&a_dims, 1, spec.a_order.as_deref());
        let b = tensor::<S>(&[spec.k, spec.n], 2, None);
        // The row role's free axes, then the contracted one: `A[row..., k]` and
        // `B[k, n]`.
        let dot = DotGeneral::new(&[rank - 1], &[0], &[], &[]);
        let dims = out_dims(&dot, &a.dims, &b.dims);
        let c = (spec.c == CMode::Separate).then(|| tensor::<S>(&dims, 3, None));
        let re = |x: f64| <S as Scalar>::Re::from_f64(x);
        let alpha = <S as Element>::from_parts(re(spec.alpha), re(spec.alpha * 0.5));
        let beta = match spec.beta {
            0.0 => <S as Element>::zero(),
            x => <S as Element>::from_parts(re(x), re(-x)),
        };
        // With `beta = 0` D starts poisoned: the overwrite path must not read
        // it, and every output element must be written. Accumulating from the
        // output (`C` is `D`) needs live values, and they are the oracle's `C`.
        let mut d = tensor::<S>(&dims, 4, None);
        if spec.c == CMode::Absent {
            for v in d.data.iter_mut() {
                *v = <S as Element>::from_parts(re(f64::NAN), re(f64::NAN));
            }
        }
        // The oracle: the same logical values, whatever their physical layout.
        // With no `C` term the value is irrelevant (`beta` is zero).
        let zero_c = match spec.c {
            CMode::Output => d.clone(),
            _ => T::<S>::new(&dims, 3),
        };
        let want = reference(
            &dot,
            alpha,
            &a,
            spec.ca,
            &b,
            spec.cb,
            beta,
            c.as_ref().unwrap_or(&zero_c),
        );
        let op = |conj: bool| {
            if conj {
                Op::Conjugate
            } else {
                Op::Identity
            }
        };
        let spec_of = |t: &T<S>, o: Op| {
            OperandSpec::new(LayoutSpec::new(&t.dims, &t.strides, t.offset).unwrap()).with_op(o)
        };
        // Labels: the row role's free axes are `0..f`, the contracted axis `f`
        // and `n` is `f + 1`. `A` is `[0, .., f-1, f]`, `B` is `[f, f + 1]`, and
        // the output is `A`'s free labels followed by `n`.
        let f = rank as i64 - 1;
        let mut la: Vec<i64> = (0..f).collect();
        la.push(f);
        let mut ld: Vec<i64> = (0..f).collect();
        ld.push(f + 1);
        let mut labels = Labels::new(&la, &[f, f + 1], &ld);
        if spec.c == CMode::Separate {
            labels = labels.with_c(&ld);
        }
        let cspec = match (&c, spec.c) {
            (Some(c), _) => CSpec::Separate(spec_of(c, Op::Identity)),
            (None, CMode::Output) => CSpec::Output(Op::Identity),
            (None, _) => CSpec::Absent,
        };
        let problem = Problem::from_labels(
            S::STORAGE,
            spec_of(&a, op(spec.ca)),
            spec_of(&b, op(spec.cb)),
            cspec,
            spec_of(&d, Op::Identity),
            &labels,
        )
        .unwrap();
        let config = PlanConfig {
            kernel: KernelChoice::Id(id.into()),
            blocking: BlockingOverride {
                kc: spec.kc,
                ..Default::default()
            },
            orientation: match spec.orient {
                Some(swap) => Orient::Force(swap),
                None => Orient::Rule,
            },
            ..PlanConfig::default()
        };
        let plan = Plan::<S>::new(&problem, &config).unwrap();
        // A forced id both pins the family and forces the packed driver, so the
        // report must name it: a plan that fell back to another family would
        // make every assertion below vacuous.
        let report = plan.report();
        assert_eq!(report.algorithm, Algorithm::Packed);
        assert_eq!(report.packed.as_ref().unwrap().family_id, id);
        Self {
            label: format!(
                "{id} row={:?} n={} k={} ca={} cb={} alpha={} beta={} c={:?}",
                spec.row, spec.n, spec.k, spec.ca, spec.cb, spec.alpha, spec.beta, spec.c as u8
            ),
            plan,
            a,
            b,
            c,
            d,
            want,
            alpha,
            beta,
            c_mode: spec.c,
        }
    }

    fn report(&self) -> &PackedReport {
        self.plan.report().packed.as_ref().unwrap()
    }

    fn run_with(&mut self, exec: &mut Exec<'_>) {
        let a = self.a.view();
        let b = self.b.view();
        let cv = self.c.as_ref().map(|c| c.view());
        let mut dv = self.d.view_mut();
        match (&cv, self.c_mode) {
            (Some(cv), _) => self
                .plan
                .execute_into_accum(
                    exec,
                    self.alpha,
                    &a,
                    &b,
                    self.beta,
                    AccumulationSource::Separate(cv),
                    &mut dv,
                )
                .unwrap(),
            (None, CMode::Output) => self
                .plan
                .execute_into_accum(
                    exec,
                    self.alpha,
                    &a,
                    &b,
                    self.beta,
                    AccumulationSource::Output,
                    &mut dv,
                )
                .unwrap(),
            (None, _) => self
                .plan
                .execute_into(exec, self.alpha, &a, &b, &mut dv)
                .unwrap(),
        }
        // Every output element was written: the poison is NaN, and no emitter
        // writes one.
        for_each_index(&self.d.dims, |i| {
            let v = self.d.get(i);
            let (re, im) = (
                <S as Scalar>::Re::to_f64(Element::re(v)),
                <S as Scalar>::Re::to_f64(Element::im(v)),
            );
            assert!(
                !re.is_nan() && !im.is_nan(),
                "{}: output at {i:?} was never written",
                self.label
            );
        });
        let err = rel_err(&self.d, &self.want);
        assert!(err < tol::<S>(), "{}: rel err {err}", self.label);
    }

    fn run(&mut self) {
        let mut exec = Exec::serial();
        self.run_with(&mut exec);
    }
}

fn case<S: Scalar>(id: &str, m: usize, n: usize, k: usize) -> Case<S> {
    Case::new(id, &Spec::matmul(m, n, k))
}

/// The register tile and the scratch tile's plane count of a forced family, read
/// from the plan report: the live extents the tests sweep and the plane count the
/// grid arithmetic needs are exactly the shape that ran.
fn geometry<S: Scalar>(id: &str) -> (usize, usize, usize) {
    let r = case::<S>(id, 1, 1, 2);
    let report = r.report();
    let planes = report.complex.map_or(1, |s| tile_planes(s.tile));
    (report.mr, report.nr, planes)
}

/// The register tile of a forced family.
fn tile<S: Scalar>(id: &str) -> (usize, usize) {
    let (mr, nr, _) = geometry::<S>(id);
    (mr, nr)
}

// ---------------------------------------------------------------------------
// 1. Every live extent
// ---------------------------------------------------------------------------

/// `m` in `1..=MR` and `n` in `1..=NR`, i.e. every partial row and column block
/// the driver can produce, at `KC = 16` and `k = 40` so the K loop is three
/// slabs, and `beta = 0` into a poisoned output.
fn every_live_extent<S: Scalar>(id: &str) {
    let (mr, nr) = tile::<S>(id);
    for m in 1..=mr {
        for n in 1..=nr {
            case::<S>(id, m, n, 40).run();
        }
    }
}

#[test]
fn every_live_extent_matches_the_reference() {
    for_each_family!(every_live_extent);
}

// ---------------------------------------------------------------------------
// 2. Several K slabs, nontrivial alpha/beta, a separate C, conjugation
// ---------------------------------------------------------------------------

/// The partial *last* row and column tiles, accumulated over three K slabs, with
/// `alpha` and `beta` carrying an imaginary part and both operands conjugated.
fn accumulate_partial<S: Scalar>(id: &str) {
    let (mr, nr) = tile::<S>(id);
    // `mr + 3` and `nr + 1`: the last row tile and the last column tile are both
    // partial, and `k = 53` at `KC = 16` is four K slabs.
    let mut spec = Spec {
        row: vec![mr + 3],
        n: nr + 1,
        k: 3 * 16 + 5,
        a_order: None,
        ca: true,
        cb: true,
        alpha: 0.75,
        beta: 1.25,
        c: CMode::Output,
        kc: Some(16),
        orient: None,
    };
    Case::<S>::new(id, &spec).run();
    // The same, accumulating from a separately described `C`.
    spec.c = CMode::Separate;
    Case::<S>::new(id, &spec).run();
}

#[test]
fn partial_tiles_accumulate_over_several_kc_blocks() {
    for_each_family!(accumulate_partial);
}

/// A whole number of tiles does *not* reach the live-extent path; it is the
/// baseline the tests above are compared against.
fn whole_tiles<S: Scalar>(id: &str) {
    let (mr, nr) = tile::<S>(id);
    case::<S>(id, 3 * mr, 2 * nr, 3 * 16 + 1).run();
}

#[test]
fn whole_tiles_still_match_the_reference() {
    for_each_family!(whole_tiles);
}

// ---------------------------------------------------------------------------
// 3. The collect-grid path
// ---------------------------------------------------------------------------

/// The blocked outer traversal collects a block's tiles in a grid, permutes whole
/// rows into the output's order and emits from there
/// (`docs/design/blocked-outer-traversal.md`). It is selected when the row
/// role's operand-fastest axis differs from its output-fastest axis, `n <= 64`,
/// `k > n`, `k <= kc`, `n <= nc`, serially, and for a non-Direct family.
///
/// The row role here is two-dimensional: `A[row0, row1, k]` with `row1`
/// contiguous and `D[row0, row1, n]` with `row0` contiguous, so the two fastest
/// axes disagree. `row0 = 5` and `row1 = 3` give a block of 15 rows, which no
/// register tile this test forces divides, so the block's last *row* tile is
/// partial (`n = 5` is a whole column tile for the `4x5` family and a partial one
/// for every family with `NR > 5`), and the block's rows are in the operand's
/// `row1`-major order while the output's are `row0`-major — that is the
/// nontrivial permutation.
///
/// The observable is the workspace: the block's two tile grids are added to the
/// worker's tile region, and `ArenaProvider::traced` reports the page capacities
/// the arena allocated. That region is `tile_planes(tile_fmt) * MR * NR` reals
/// for the family's own scratch tile plus `2 * grid_elems` for the grids
/// (`crates/tcontract/src/driver/mod.rs`, `grid_elems`), and `PageBuf::ensure`
/// rounds the whole request up to a power of two (`crates/tcontract-exec/src/
/// workspace.rs`), which `page` reverses here. The twin with `A` column-major is
/// the same shape and family, with both fastest axes on `row0`, so it is not
/// eligible — which is what makes the comparison say something about the shape
/// rather than about the tile region in general.
///
/// The tile region is the *second* traced page: `with_worker` grows the `A` panel
/// before the tile region, and a fresh provider traces each buffer once. Both
/// expectations below are exact rather than lower bounds, so a shape whose block
/// is not the assumed `row0 * row1` rows fails instead of being attributed to the
/// wrong buffer: `row0 = 5` is the span axis whole and `row1 = 3` fits one
/// 64-byte line at every dtype this test forces (one line holds 8 f64, 16 f32
/// and 4 c64 / 8 c32 elements).
fn collect_grid<S: Scalar>(id: &str) {
    let (row0, row1, n, k) = (5usize, 3usize, 5usize, 64usize);
    // `row1` contiguous in `A`, `row0` contiguous in `D`, and the orientation
    // pinned so the row role stays the two-axis side for every register block.
    let spec = |a_order: Option<Vec<usize>>| Spec {
        row: vec![row0, row1],
        n,
        k,
        a_order,
        ca: false,
        cb: false,
        alpha: 1.5,
        beta: 0.0,
        c: CMode::Absent,
        kc: None,
        orient: Some(false),
    };
    let (mr, nr, planes) = geometry::<S>(id);
    let elem = core::mem::size_of::<S::Re>();
    let m = row0 * row1;
    let tiles = m.div_ceil(mr) * n.div_ceil(nr);
    // One tile is the family's own scratch tile, `planes` planes of `MR x NR`.
    let tile_bytes = planes * mr * nr * elem;
    let grid_lower = 2 * tiles * tile_bytes;
    let grid = tile_page::<S>(id, &spec(Some(vec![1, 2, 0])));
    let twin = tile_page::<S>(id, &spec(None));
    assert_eq!(
        grid,
        page(tile_bytes + grid_lower),
        "{id}: the collect grid's tile region is not the tile plus two grids of {tiles} tiles \
         ({tile_bytes} B + {grid_lower} B), so the collect-grid path did not run"
    );
    assert_eq!(
        twin,
        page(tile_bytes),
        "{id}: the ineligible twin's tile region is not exactly its own {planes}-plane tile"
    );
    assert!(
        twin < grid_lower,
        "{id}: the twin's tile page ({twin} B) is not smaller than the grids, so the comparison \
         above proves nothing"
    );
}

/// The page capacity `PageBuf::ensure` records for a request of `bytes`
/// (`crates/tcontract-exec/src/workspace.rs`): it grows by doubling, so the traced
/// capacity is the request rounded up to a power of two.
fn page(bytes: usize) -> usize {
    bytes.next_power_of_two()
}

/// Run one case and return the page capacity of its tile region.
///
/// The arena traces the worker's two buffers in the order `with_worker` grows
/// them, the `A` panel first and the tile region second, so the second record is
/// the tile region; both are asserted to be there, and the `A` panel's page has to
/// be large enough to be a packed panel of the block's rows. There is no fallback
/// that reads the minimum and hopes: a fixture where the two pages collide is a
/// broken test.
fn tile_page<S: Scalar>(id: &str, spec: &Spec) -> usize {
    let mut c = Case::<S>::new(id, spec);
    let swapped = c.report().swapped;
    assert!(!swapped, "{id}: the row role is `row0 x row1` here");
    let rows = c.report().stats.m;
    let k = c.report().stats.k;
    let arena = ArenaProvider::traced();
    {
        let mut exec = Exec::serial_with_workspace(&arena);
        c.run_with(&mut exec);
    }
    let pages: Vec<usize> = arena.trace_take().iter().map(|&(_, b)| b).collect();
    assert_eq!(
        pages.len(),
        2,
        "{id}: expected one growth of the `A` panel and one of the tile region, got {pages:?}"
    );
    let elem = core::mem::size_of::<S::Re>();
    assert!(
        pages[0] >= rows * k * elem,
        "{id}: the first traced page ({} B) cannot be the `A` panel of {rows} x {k} reals",
        pages[0]
    );
    pages[1]
}

#[test]
fn the_collect_grid_path_permutes_partial_tiles() {
    for_each_family!(collect_grid);
}

// ---------------------------------------------------------------------------
// 4. Direct-B / Direct-C retention
// ---------------------------------------------------------------------------

/// A Direct family writes `D` (or reads `B` in place) inside its own kernel and
/// already takes the live extent there. It has no live-extent entry, so a partial
/// tile must keep the old path — and stay correct.
#[test]
fn direct_families_keep_their_own_paths() {
    for id in ["ref.f64.direct.4x4", "ref.f64.direct-b.4x4"] {
        if !available::<f64>(id) {
            continue;
        }
        let (mr, nr) = tile::<f64>(id);
        for (m, n) in [(1, 1), (mr - 1, nr - 1), (2 * mr + 1, 3), (2, 3 * nr + 1)] {
            case::<f64>(id, m, n, 40).run();
            let mut acc = Case::<f64>::new(
                id,
                &Spec {
                    beta: 1.25,
                    c: CMode::Output,
                    k: 53,
                    kc: Some(16),
                    ..Spec::matmul(m, n, 0)
                },
            );
            acc.run();
        }
    }
}

// ---------------------------------------------------------------------------
// 5. The wrapped methods keep the full-tile entry
// ---------------------------------------------------------------------------

/// Native 1m and 3m, the interleaved complex kernel and the induced 1m/4m
/// descriptors all have `edge: None` (asserted at the ABI level in
/// `tcontract-kernel/tests/edge_tiles.rs`). Here they run partial tiles through
/// the driver anyway, so the *dispatch* is proven to fall back rather than the
/// ABI assertion alone carrying the claim.
///
/// These wrappers live in the x86 menus only (the portable complex family is one
/// interleaved native kernel, which the sweeps above already run with partial
/// tiles), so this is an x86 test; the ABI-level assertion stays portable.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn wrapped_methods_keep_the_full_tile_path() {
    let mut ids: Vec<String> = Vec::new();
    for id in [
        "avx2.c64.1m.4x6",
        "avx2.c64.3m.4x4",
        "avx2.c64.native.4x4",
        "avx512.c64.1m.12x8",
        "avx512.c64.3m.8x10",
    ] {
        if available::<tcontract_kernel::C64>(id) {
            ids.push(id.to_string());
        }
    }
    // The induced descriptors are built from a real family, so their ids are
    // derived rather than written down.
    if let Ok(real) = Registry::select::<f64>("avx2.f64.real.8x6", CpuFeatures::detect()) {
        for f in [
            tcontract_kernel::induced::one_m(real),
            tcontract_kernel::induced::four_m(real),
        ]
        .into_iter()
        .flatten()
        {
            ids.push(f.id.to_string());
        }
    }
    assert!(!ids.is_empty(), "no wrapped family is available");
    for id in ids {
        let (mr, nr) = tile::<Complex64>(&id);
        let mut spec = Spec::matmul(mr + 1, nr, 40);
        spec.ca = true;
        spec.alpha = 0.75;
        spec.beta = 1.25;
        spec.c = CMode::Output;
        Case::<Complex64>::new(&id, &spec).run();
    }
}
