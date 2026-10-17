//! The five-loop driver.
//!
//! tcontract: threads come only from a [`tcontract_exec::Exec`] (not upstream); this
//! crate spawns none, and without an `Exec` the driver runs serially.
//!
//! Structurally identical to BLIS's GEMM: two levels of cache blocking with a
//! packing step at each, wrapped around a register-blocked micro-kernel. The
//! only tensor-specific part is that every matrix access goes through a
//! scatter vector.
//!
//! ```text
//! for each Hadamard (batch) index h          -- offsets all four operands
//!   loop 5: for jc in 0..N step NC
//!     loop 4: for pc in 0..K step KC
//!               pack B[pc:pc+KC, jc:jc+NC] -> Bp    (NR slivers)
//!       loop 3: for ic in 0..M step MC
//!                 pack A[ic:ic+MC, pc:pc+KC] -> Ap  (MR slivers)
//!         loop 2: for jr in 0..NC step NR
//!           loop 1: for ir in 0..MC step MR
//!                     micro-kernel -> MR x NR tile
//!                     write back to C/D through the scatter vectors
//! ```
//!
//! The loop nest is identical for real and for all three complex methods. What
//! the complex method changes is captured entirely by the selected
//! [`Ukr`](crate::kernel::Ukr): its `a_pack`/`b_pack` formats, its per-k sliver
//! widths, its tile size and its `tile_fmt`. Nothing here branches on the
//! method, which is what makes the three genuinely comparable — they share
//! every line of index analysis, packing traversal, loop arithmetic and
//! write-back scatter.
//!
//! `beta` and the `C` operand are consumed on the first `pc` iteration only;
//! later iterations accumulate into `D`.
//!
//! # Threading
//!
//! BLIS-style, and two-dimensional: the output is cut into a `pm x pn` grid of
//! contiguous row strips of whole `MR` panels by column groups of whole `NR`
//! blocks, one thread each. [`Plan::partition`] decides `(pm, pn)` and is the
//! single definition of it; `pn > 1` only when `ceil(M / MR) < p`, so on all but
//! four of the 49 corpus cases this is still a 1-D partition of `M` and behaves
//! exactly as the first version did.
//!
//! The two axes are cut in different places, and deliberately:
//!
//! * The **row strips** are cut once, outside everything, and each thread runs
//!   loops 3, 2 and 1 over its own strip with its own packed-`A` block.
//! * The **column groups** are cut *inside* loop 5, per `NC` block: every thread
//!   iterates the same `(h, jc, pc)` sequence over the whole of `N` and `K`, and
//!   within each `jc` block takes its group's contiguous range of `NR` slivers.
//!
//! Cutting `N` inside loop 5 rather than over the whole range is what keeps the
//! packed-`B` panel single and shared. It stays exactly the L3-sized panel the
//! `NC` budget is derived for — a top-level split of `N` would need `pn` panels
//! and `pn` times the L3 — and it keeps loops 5 and 4 identical across all `p`
//! threads, which is what makes the barrier counts agree without anyone tracking
//! them (see `run_strip`).
//!
//! `B` is packed cooperatively: within a column group, the `pm` threads that
//! share it split its slivers, so each thread packs slivers it will itself read
//! and no thread packs anything it will not. Two barriers per `(jc, pc)` bracket
//! the packing — one so nobody is still reading the previous panel, one so the
//! new one is complete — and they are **per column group**, of `pm` threads,
//! because a group's slice of the panel is written and read only by its own
//! threads. A one-thread column group therefore needs no barrier at all, which
//! is the case a pure `N` split (`pm == 1`) degenerates to: no synchronisation
//! anywhere in the loop nest.
//!
//! The packed `A` block is per thread and so is **duplicated `pn` times**: the
//! `pn` threads of one row strip each pack that strip for themselves. That is
//! deliberate rather than merely convenient. It costs `ceil(panels/pm) * MR * K`
//! element moves per thread against `ceil(panels/pm) * ceil(blocks/pn) * MR * NR
//! * K` lane-FMAs, i.e. one packed element per `NR * ceil(blocks/pn)` FMA slots,
//! and it buys back the alternative's cost: a single packer per row strip needs a
//! barrier *inside* loop 3, at `M/MC` times the frequency of the ones above it,
//! and leaves the block in one thread's L2 for the others to pull across L3
//! instead of each having it in its own. `Plan::partition` prices the duplication
//! explicitly and will not split `N` when a thread would be left with too few
//! `NR` blocks to amortise it over.
//!
//! Four properties this buys, all of them deliberate:
//!
//! * **No reduction.** Loop 4 (`pc`) accumulates into `D` in place, so
//!   parallelising it would need either a temporary per thread or atomics.
//!   Partitioning the *output* instead gives every element a single owning
//!   thread, which accumulates over the full `K` in the original order. Both
//!   axes have this property; `K` is the one that does not, and it is left
//!   serial.
//! * **Bitwise identical to serial**, therefore, for any thread count and any
//!   `(pm, pn)` — the floating-point operations per output element are the same
//!   operations in the same order. That is a strong enough invariant to test
//!   directly, and the `threaded_matches_serial_*` tests do.
//! * **Blocks stay aligned with the block scatter.** Strips are whole `MR`
//!   panels and groups whole `NR` slivers, so every thread's micro-tiles are the
//!   ones the serial driver would have used. The write-back fast path, the
//!   row-block rule and the orientation rule are untouched by threading.
//! * **The serial path is unchanged.** With `pm == pn == 1` the only difference
//!   from the pre-threading driver is two `Option` checks and a handful of
//!   integer divisions per `(jc, pc)` iteration, nowhere near the hot loops.
//!   Every measurement committed in `docs/archive/tensorprimitives/notebook/` was taken single-threaded and
//!   stays comparable.
//!
//! # Batch-axis claiming
//!
//! Everything above cuts *one* entry of the in-plan batch (the Hadamard modes)
//! across the team, and the team runs the entries one after another with its
//! barriers per entry. For tiny entries, or at least as many entries as workers,
//! that is the wrong cut: on the default partition the driver instead gives each
//! of a few lanes a contiguous share of the entries, each run serially with the
//! serial blocking and the lane's own buffers, with no barrier and no shared `B`
//! panel. The result is bitwise the serial one. The width rule is
//! [`Exec::batch_decision`](tcontract_exec::Exec::batch_decision) with a
//! balanced-lanes query; see `driver/route.rs`.
//!
//! Known limits, in the order they will bite (see the Phase 4 report):
//! parallelism is capped at `ceil(M / MR) * ceil(N / NR)`, and a column group can
//! only be as wide as the `jc` block it is cut from, so a tail `NC` block with
//! fewer slivers than groups leaves some threads idle for that block; and `NC`'s
//! L3 budget is still charged as if one core owned the cache.
//!
//! The per-call spawn cost that used to head that list — `std::thread::scope`
//! rather than a pool, ~20–36 µs per thread and the whole story below a megabyte
//! (A43, D46) — is gone: the threads of one call are the workers of the host's
//! [`tcontract_exec::Pool`], co-scheduled by [`tcontract_exec::Exec::broadcast`]. For
//! many small contractions, [`crate::batch`] parallelises over a *batch*
//! instead of inside each contraction.

use std::sync::Barrier;

use tcontract_exec::{Exec, ExecError, WorkspaceProvider, WorkspaceReq};

use crate::buffer::Panel;
mod block;
mod route;
pub(crate) use route::call_workspace_req;
pub(crate) use route::execution_geometry;
mod dynamic;
mod static_grid;
#[cfg(test)]
mod tests;
mod tile;
use crate::plan::PackedPlan;
pub(crate) use dynamic::dynamic_report;
pub use dynamic::{Assignment, DynSnapshot, DynStats, DynamicReport};
use static_grid::{run_strip, BPart};
use tcontract_kernel::pack::panel_len;
use tcontract_kernel::scatter::append_block_scatter;
use tcontract_kernel::writeback::scale_only;
use tcontract_kernel::UkrFn;
use tcontract_kernel::{Axis, BAccess, Blocking, DriverFamily, Element};
use tile::{compute_block, pack_a_rows, pack_b_slivers, Bufs, Epoch};

/// Operand-dependent decisions the driver makes once per execute, kept
/// separate from the plan-level resolution so tests can pin the real rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ResolvedCall {
    /// The shared B panel must be packed this call.
    pub pack_b_needed: bool,
    /// Eligible tiles may be updated directly, without a scratch tile.
    pub direct_c_allowed: bool,
}

/// The same two decisions the driver takes, for a given plan, resolution and
/// operand pair.
#[cfg(test)]
pub(crate) fn driver_decisions<T>(
    plan: &PackedPlan,
    rg: &tcontract_kernel::ResolvedGemm<T::Real>,
    c: *const T,
    d: *mut T,
    beta: T,
) -> ResolvedCall
where
    T: Element,
{
    // Which user operand plays the kernel's column role, exactly as
    // `execute_capped` decides it.
    let swap = plan.transposes_gemm(rg.mr);
    let (bk, bn) = if swap {
        (&plan.a_k, &plan.a_m)
    } else {
        (&plan.b_k, &plan.b_n)
    };
    ResolvedCall {
        pack_b_needed: pack_b_needed(rg.family().b_access, bk, bn, rg.nr),
        direct_c_allowed: direct_c_allowed(plan, c, d, beta),
    }
}

/// Whether the family must pack this call's column operand: it must accept an
/// in-place B, B's k stride must be one, and every `NR` block's column offsets
/// must be an arithmetic progression, so one stride expresses the tile.
///
/// This reads the operand's own scatter rather than its block scatter, so it
/// can decide before any scratch buffer exists.
pub(crate) fn pack_b_needed(b_access: BAccess, bk: &[i64], bn: &[i64], nr: usize) -> bool {
    if !matches!(
        b_access,
        BAccess::Direct {
            unit_stride: Axis::Col
        }
    ) {
        return true;
    }
    if bk.len() > 1 && !bk.windows(2).all(|w| w[1] - w[0] == 1) {
        return true;
    }
    !tcontract_kernel::scatter::block_scatter_regular(bn, nr)
}

/// Whether the Direct kernel may write D in place. Real storage only, no C or D
/// conjugation, and either C is not read at all or C *is* D — because the
/// kernel has a single output pointer it scales as `alpha_d*D + beta_ab*A*B`.
fn direct_c_allowed<T: Element>(plan: &PackedPlan, c: *const T, d: *mut T, beta: T) -> bool {
    !T::IS_COMPLEX
        && !plan.conj_c
        && !plan.conj_d
        && (beta == T::zero()
            || (core::ptr::eq(c, d as *const T)
                && plan.c_m == plan.d_m
                && plan.c_n == plan.d_n
                && plan.h_c == plan.h_d))
}

/// Where the five block-scatter vectors sit inside one reused scatter buffer.
///
/// They are built together into the team's buffer so a steady-state execute
/// allocates nothing; these offsets are what keeps them separate slices.
#[derive(Clone, Copy, Default)]
struct ScatterRuns {
    a: (usize, usize),
    b: (usize, usize),
    dm: (usize, usize),
    dn: (usize, usize),
    cm: (usize, usize),
}

impl ScatterRuns {
    fn slice<'a>(&self, buf: &'a [i64], run: (usize, usize)) -> &'a [i64] {
        &buf[run.0..run.1]
    }
}

/// A raw pointer shared across the threads of one [`execute`] call.
///
/// Rust will not send a bare pointer between threads, and rightly, so the
/// promise is made explicitly here rather than silently at each use: the
/// operands are read-only for the duration, and every thread writes only the
/// output elements of its own `(row strip, column group)` cell. The cells
/// partition the output's `(i, j)` index space by construction, so no two
/// threads write the same element — and no two write the same *byte*, because
/// `D`'s scatter is injective, which the serial write-back's read-modify-write
/// on every `pc` block past the first already requires.
#[derive(Clone, Copy)]
struct Shared<T>(*mut T);

// SAFETY: see the type's documentation. The disjointness is a property of the
// strip partition in `execute`, which is the only place `Shared` is created.
//
// `T: Send` on both, and it is `Send` rather than `Sync` that is wanted even for
// the `Sync` impl: `Shared` is written through from several threads at once, so
// it is morally a split `&mut T` rather than a shared `&T`, and the obligation
// it discharges is that values of `T` may be produced and dropped on a thread
// other than the one that created them. Every instantiation is `T: Element`,
// which is already `Copy + Send + Sync + 'static`, so the bound costs nothing
// here -- it is there so the impls cannot silently start covering a `T` that
// does not deserve them. `buffer::Panel` bounds its `Send` the same way.
unsafe impl<T: Send> Send for Shared<T> {}
unsafe impl<T: Send> Sync for Shared<T> {}

/// Everything one thread of the loop nest needs that does not vary with its
/// row strip. Exists so that the nest can be written once and run either
/// serially or on `p` threads, rather than duplicated.
/// Where a block's tile grid starts inside a worker's tile region: after the tile and
/// the induced scratch, and the second grid after the first. Null when the blocked path
/// does not apply, which is what `Bufs::grid` being null means downstream.
fn grid_base<R>(tile: *mut R, grid_off: usize, grid_elems: usize, second: bool) -> *mut R {
    if grid_elems == 0 {
        core::ptr::null_mut()
    } else {
        // SAFETY: the workspace was sized for both grids when it is not null.
        unsafe { tile.add(grid_off + if second { grid_elems } else { 0 }) }
    }
}

/// Rows a blocked outer traversal will hold in one panel. Two thirds of the legacy
/// `MC` for f64 (256) is not enough for a whole line of the operand's fastest axis
/// times the output's fastest axis, so the blocked path buys its own budget and the
/// eligibility shrinks the block to fit it. See
/// `docs/design/blocked-outer-traversal.md`.
const BLOCK_MC_BUDGET: usize = 480;

struct Ctx<'a, T: Element> {
    plan: &'a PackedPlan,
    fam: DriverFamily<T::Real>,
    packers: (tcontract_kernel::PackFn<T>, tcontract_kernel::PackFn<T>),
    emitter: tcontract_kernel::EmitFn<T>,
    /// This call's operand-dependent decisions.
    call: ResolvedCall,
    /// Whether the kernel's column role takes B in place this call.
    direct_b: bool,
    mr: usize,
    nr: usize,
    mc: usize,
    kc: usize,
    nc: usize,
    m: usize,
    n: usize,
    k: usize,
    /// Reals between one column group's slice of the packed `B` panel and the
    /// next. See `execute` for why the panel is cut per group and not per sliver.
    b_group: usize,
    am: &'a [i64],
    ak: &'a [i64],
    bk: &'a [i64],
    bn: &'a [i64],
    cm: &'a [i64],
    cn: &'a [i64],
    dm: &'a [i64],
    dn: &'a [i64],
    ha: &'a [i64],
    hb: &'a [i64],
    hc: &'a [i64],
    /// Every block-scatter vector this call needs, laid out by `runs`.
    scatter: &'a [i64],
    runs: ScatterRuns,
    /// Whether the blocked outer traversal applies to this call, and with which block
    /// shape. `None` keeps today's consecutive-slice traversal; see
    /// `docs/design/blocked-outer-traversal.md`.
    blocked: Option<block::BlockShape>,
    /// Reals from the tile base to the collect grid, and one grid's length.
    grid_off: usize,
    grid_elems: usize,
    conj_a: bool,
    conj_b: bool,
    alpha: T,
    beta: T,
    a: Shared<T>,
    b: Shared<T>,
    c: Shared<T>,
    d: Shared<T>,
    /// The shared packed-`B` panel: written cooperatively, read by everyone.
    bp: Shared<T::Real>,
}

// A beta-zero call may have no C storage. Even an unused `offset` must stay
// within its allocation: use D's live origin and batch offsets, never C's gaps.
fn c_for_call<T: Element>(
    plan: &PackedPlan,
    beta: T,
    c: *const T,
    d: *mut T,
) -> (*const T, &[i64]) {
    if beta == T::zero() {
        (d.cast_const(), &plan.h_d)
    } else {
        (c, &plan.h_c)
    }
}

fn lcm(a: usize, b: usize) -> usize {
    fn gcd(mut a: usize, mut b: usize) -> usize {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    }
    a / gcd(a, b) * b
}

/// Execute a packed plan with a resolved family on `exec`, with `workspace`
/// (or, when `None`, the one `exec` lends). The width is `exec.budget()`;
/// effective blocking uses the active grid width, not the plan's requested
/// width.
///
/// The route is fixed before anything is allocated or written. A caller that is
/// already a worker of the target pool cannot be co-scheduled, so a plan whose
/// partition needs a barrier-bearing team returns
/// [`ExecError::Unavailable`] with nothing written; every other route runs
/// barrier-free on that pool.
///
/// # Safety
/// * `a`, `b`, `d` must be valid for all offsets generated by the plan's
///   scatter vectors (reads for `a`/`b`, reads and writes for `d`).
/// * `c` must likewise be valid for reads unless `beta` is zero, in which case
///   it is never dereferenced and may be dangling.
/// * `d` must not alias `a` or `b`.
/// * `rg` must be validated for `T`, the plan's conjugations, scratch ABI and
///   every active width; the owning `Plan` does this when it is built.
#[allow(clippy::too_many_arguments)] // INVARIANT: the contraction argument set plus the seam.
pub(crate) unsafe fn execute_packed<T: Element>(
    plan: &PackedPlan,
    rg: &tcontract_kernel::ResolvedGemm<T::Real>,
    exec: &Exec<'_>,
    workspace: Option<&dyn WorkspaceProvider>,
    alpha: T,
    a: *const T,
    b: *const T,
    beta: T,
    c: *const T,
    d: *mut T,
) -> Result<(), ExecError> {
    // SAFETY: raw pointer and descriptor obligations forwarded unchanged.
    unsafe { execute_capped(plan, alpha, a, b, beta, c, d, exec, workspace, rg, None) }
}

/// [`execute_packed`] with opt-in [`DynStats`] counters, for tests and
/// benchmarks of `Partition::DynamicTiles`. Ordinary execution carries no
/// counters.
///
/// # Safety
/// As [`execute_packed`].
#[cfg(test)]
#[allow(clippy::too_many_arguments)] // INVARIANT: as `execute_packed`.
pub(crate) unsafe fn execute_packed_instrumented<T: Element>(
    plan: &PackedPlan,
    rg: &tcontract_kernel::ResolvedGemm<T::Real>,
    exec: &Exec<'_>,
    workspace: Option<&dyn WorkspaceProvider>,
    stats: &DynStats,
    alpha: T,
    a: *const T,
    b: *const T,
    beta: T,
    c: *const T,
    d: *mut T,
) -> Result<(), ExecError> {
    // SAFETY: raw pointer and descriptor obligations forwarded unchanged.
    unsafe {
        execute_capped(
            plan,
            alpha,
            a,
            b,
            beta,
            c,
            d,
            exec,
            workspace,
            rg,
            Some(stats),
        )
    }
}

/// The loop nest on `exec`: its budget is the width this call may use, and its
/// workspace (or `workspace`, when given) is where the call's buffers live.
///
/// # Safety
///
/// As [`execute_packed`].
// Eight arguments, against clippy's seven: seven of them are the contraction
// itself -- `alpha`, four operands, `beta` and the plan -- and bundling them into a
// struct to satisfy a count would put a layer between the ABI-facing entry points
// and the loop nest for no reader's benefit.
#[allow(clippy::too_many_arguments)]
unsafe fn execute_capped<T>(
    plan: &PackedPlan,
    alpha: T,
    a: *const T,
    b: *const T,
    beta: T,
    c: *const T,
    d: *mut T,
    exec: &Exec<'_>,
    workspace: Option<&dyn WorkspaceProvider>,
    resolution: &tcontract_kernel::ResolvedGemm<T::Real>,
    stats: Option<&dynamic::DynStats>,
) -> Result<(), ExecError>
where
    T: Element,
{
    let rg = *resolution;
    if plan.is_empty() {
        return Ok(());
    }
    let fam = rg
        .family()
        .driver_family()
        .expect("validated family ABI for this scheme");
    let (mr, nr) = (fam.mr, fam.nr);

    // Row/column orientation. Exchanging `(A, M)` with `(B, N)` computes
    // `D^T = B^T A^T`, which is the same contraction seen through the
    // transposed matrix view of `C` and `D`. Nothing below branches on it
    // again: from here on `am`/`ak`/`ptr_a` *are* the row operand, whichever
    // tensor that is. See `Plan::transposes_gemm` for why it is worth doing.
    //
    // Packing formats remain fixed to kernel roles: row-A uses 1e and
    // column-B uses 1r under 1m, whichever user tensor fills that role.
    let swap = plan.transposes_gemm(mr);
    let (ptr_a, ptr_b) = if swap { (b, a) } else { (a, b) };
    let (am, ak, conj_a) = if swap {
        (&plan.b_n, &plan.b_k, plan.conj_b)
    } else {
        (&plan.a_m, &plan.a_k, plan.conj_a)
    };
    let (bk, bn, conj_b) = if swap {
        (&plan.a_k, &plan.a_m, plan.conj_a)
    } else {
        (&plan.b_k, &plan.b_n, plan.conj_b)
    };
    let (cm, cn) = if swap {
        (&plan.c_n, &plan.c_m)
    } else {
        (&plan.c_m, &plan.c_n)
    };
    let (dm, dn) = if swap {
        (&plan.d_n, &plan.d_m)
    } else {
        (&plan.d_m, &plan.d_n)
    };
    let (ha, hb) = if swap {
        (&plan.h_b, &plan.h_a)
    } else {
        (&plan.h_a, &plan.h_b)
    };

    let (c, hc) = c_for_call(plan, beta, c, d);
    let workspace = workspace.or_else(|| exec.workspace());
    let m = am.len();
    let n = bn.len();
    let k = ak.len();

    // Empty contraction dimension: D = op_D(beta * op_C(C)).
    if plan.has_empty_contraction() {
        for h in 0..plan.stats.batch {
            scale_only::<T>(
                beta,
                c.offset(hc[h] as isize),
                cm,
                cn,
                plan.conj_c,
                d.offset(plan.h_d[h] as isize),
                dm,
                dn,
                plan.conj_d,
            );
        }
        return Ok(());
    }

    // Operand-dependent decisions, made once for the whole call: the family's
    // B access, this B's strides, and whether D can be updated in place. Both
    // are allocation-free, because the partition below depends on them and the
    // team's buffer is only borrowed once the shape is known.
    let geometry = execution_geometry::<T>(plan, &rg, exec)?;
    let call = ResolvedCall {
        pack_b_needed: !geometry.direct_b,
        direct_c_allowed: direct_c_allowed(plan, c, d, beta),
    };
    let route::Geometry {
        lanes,
        pm,
        pn,
        p,
        dyn_jobs,
        direct_b,
    } = geometry;

    // NOTE (Phase 4): `cfg.blk` is still the untouched Phase 2 heuristic, and
    // `MC`/`NC` in it are sized for a `KC`-deep panel. On a third of the corpus
    // the contraction is far shallower than `KC` (`k = 24` against 256), so the
    // packed `A` block uses a tenth of its budget and `B` is re-streamed
    // `M/MC` times for nothing. Re-deriving against `min(k, KC)` was tried and
    // is *not* a win: it swings individual cases by +13% and −18% with no rule
    // visible, because `MC` has a second constraint this model omits — the
    // strip of `D` that one `jr` pass revisits. See the Phase 4 report; this is
    // what the `MC`/`KC`/`NC` sweep has to settle.
    // Row strips are whole `MR` panels and column groups whole `NR` slivers, so
    // every thread's micro-tiles line up with the block scatter and with the
    // write-back's fast path. `Plan::partition` owns the choice of how many of
    // each; it caps them at the panel and block counts, so a contraction with
    // three row panels and two column blocks uses six threads at most however
    // many were asked for and however much work it contains.
    // Reporting and execution share geometry resolution; unavailable worker
    // routes were rejected before taking a workspace or writing anything.
    let spmd = pm > 1;
    // INVARIANT: the plan validated serial's maximal NC, and p >= 1.
    let at_width = rg
        .with_threads(p)
        .expect("validated effective-width blocking");
    let blocking = Blocking {
        mc: at_width.mc,
        kc: at_width.kc,
        nc: at_width.nc,
    };
    let Blocking { mc, kc, nc } = blocking;
    let mc = mc.min(m.next_multiple_of(mr));
    let nc = nc.min(n.next_multiple_of(nr));
    // Whether the blocked outer traversal applies, computed here because this is
    // where the plan's oriented role axes and the resolved blocking meet. It needs
    // the role's fastest axes to disagree *and* the contraction to fit one K slab and
    // one NC panel - which is what keeps a block's accumulator complete without
    // cross-slab work, ownership or barrier changes.
    // The block buys a contiguous pack and pays for it with a per-block permutation and
    // metadata (measured at about 9 ms a call on the campaign's shapes). How much the pack
    // is worth is the operand's element count against the output's - `m*k` against `m*n`,
    // i.e. roughly `k` against `n`. Over the 98 tcbench rows at 1T, with the path on and
    // off measured alternately in one session, `n <= 64 and k > n` is the only class that
    // wins without losing: 4 wins, 0 losses, 2 neutral, median x1.14, the individual cases
    // up to x1.74 (`abjc-cbka-kj` f64 42.0 -> 24.1 ms). A thin output against a deeper
    // contraction is also exactly the domain TBLIS's own advantage lives in
    // (`docs/design/blocked-outer-traversal.md`, section 9).
    let blocked = block::blocked_eligibility(
        &plan.stats,
        swap,
        k,
        n,
        kc,
        nc,
        (64 / core::mem::size_of::<T>()).max(1),
        BLOCK_MC_BUDGET,
        mr,
    )
    // **Serial only**, and not with a family that writes `D` itself:
    //
    // * The blocked path enumerates every block of the role, which is only a worker's own
    //   work when its strip is the whole `m` range - so `pm == 1`. It also fills only its
    //   own column tiles, while the grid permutation reads every column tile of the grid,
    //   so `pn == 1` as well; with a column-group worker the permutation would read grid
    //   slots nobody initialized. Partitioning the block itself is a separate change.
    // * A Direct family writes `D` inside the tile call instead of the tile, so a
    //   collected tile would never be written and the emission below would read scratch
    //   that no kernel touched. `direct_tile` also tests `collect` now, which is the
    //   invariant itself; this filter keeps such a call off the path entirely.
    .filter(|_| pm == 1 && pn == 1)
    .filter(|_| !matches!(fam.kernel, UkrFn::Direct(_)));
    // Two whole grids when it applies: the block's own row order and the output's.
    // The block is the span axis whole times a line's worth along the operand's fastest
    // axis; every other axis contributes one value, exactly as `block::rows` builds it. It
    // sizes both the packed panel and the tile grid.
    let block_rows = match blocked {
        None => 0,
        Some(shape) => {
            let mut axes_buf = [block::Role::EMPTY; block::MAX_AXES];
            let axes = block::role_into(&plan.stats, swap, &mut axes_buf);
            axes[shape.span].extent * shape.line_len
        }
    };
    let grid_elems = match blocked {
        None => 0,
        // One slot per (row tile, column tile) of a block.
        Some(_) => {
            tcontract_kernel::tile_planes(fam.tile_fmt)
                * mr
                * nr
                * block_rows.div_ceil(mr)
                * n.div_ceil(nr)
        }
    };
    // The blocked path sizes `MC` by the block rather than the other way round - the
    // eligibility already made the block fit `BLOCK_MC_BUDGET`.
    let mc = match blocked {
        None => mc,
        Some(_) => block_rows.next_multiple_of(mr),
    };
    // The blocked path hands the packer a *raw* panel pointer and asks it to write
    // `block_rows` rows, and hands the emitter a raw tile pointer into a grid it also
    // sized. `pack_a_rows`'s contract - "the caller supplies a sufficiently sized
    // output" - is discharged here or nowhere, and a mistake in this geometry would
    // corrupt the heap rather than fail. Check it once per call, where all four
    // numbers are in hand; the hot loops keep their `debug_assert`s.
    if blocked.is_some() {
        assert!(
            block_rows <= mc,
            "blocked path: {block_rows} block rows exceed the {mc}-row A panel"
        );
        assert!(
            k <= kc && n <= nc,
            "blocked path: one K slab and one NC panel are the condition for a single \
             emission; k={k} kc={kc} n={n} nc={nc}"
        );
        let slots = block_rows.div_ceil(mr).max(1) * n.div_ceil(nr).max(1);
        assert!(
            grid_elems >= slots * tcontract_kernel::tile_planes(fam.tile_fmt) * mr * nr,
            "blocked path: {grid_elems} grid reals cannot hold {slots} tiles"
        );
    }
    // The grids live after the tile and the induced scratch in the worker's region.
    let grid_off = fam.tile + fam.induced_scratch(kc);
    #[cfg(feature = "phase-timing")]
    if std::env::var_os("TCONTRACT_PHASE").is_some() {
        eprintln!("BLOCKED {blocked:?} grid_elems={grid_elems}");
        if blocked.is_some() {
            // Which strides the role actually has, not just how long its axes are: two
            // shapes with the same `BlockShape` and different strides behave very
            // differently under this traversal, and that is what this prints.
            let mut axes_buf = [block::Role::EMPTY; block::MAX_AXES];
            let axes = block::role_into(&plan.stats, swap, &mut axes_buf);
            eprintln!("BLOCKED-AXES {axes:?}");
        }
    }
    // C-line aligned strips, when asked for: one line of `C` is 64 bytes, so
    // the boundary is a multiple of both the panel and the line.
    let align = match rg.opts.align_c_lines {
        true => lcm(mr, (64 / core::mem::size_of::<T>()).max(1)),
        false => 0,
    };

    // Panel sizes come from the kernel's declared per-k sliver widths, so a
    // method that packs more reals per element (1m's "1e", 3m's sum plane)
    // automatically gets a correspondingly larger buffer. `A` is per thread and
    // allocated inside it, so it is first-touched on the node that will use it;
    // `B` is shared and allocated here.
    //
    // The `B` panel is cut into one slice per *column group*, not indexed by
    // absolute sliver, and the two differ: a group's sliver range moves between
    // `jc` blocks, because a tail block has fewer slivers to divide, so with
    // absolute indexing one group's next block would land on top of another
    // group's current one. There is nothing to order them — the barriers are per
    // group by design, and at `pm == 1` there are none at all — so it has to be
    // structural. It costs at most one sliver of padding per group, and it is
    // also a small win: a group's slice is contiguous, so groups do not share
    // cache lines at their boundaries.
    //
    // The stride between slices is the *worst-case* sliver size, `kc` deep and
    // not `pc_len` deep, precisely so that two groups sitting on different `pc`
    // blocks at the same moment still cannot overlap.
    let ap_len = panel_len(mc, mr, kc, fam.a_pack);
    let group_cap = nc.div_ceil(nr).div_ceil(pn);
    let b_group = panel_len(group_cap * nr, nr, kc, fam.b_pack);
    // Under batch-axis claiming every lane packs its own `B` into one slice of
    // the buffer, so the slices are cache-line aligned to keep lanes apart.
    let (panels, b_group) = match lanes > 1 {
        true => (
            lanes,
            b_group.next_multiple_of(64 / core::mem::size_of::<T::Real>()),
        ),
        false => (pn, b_group),
    };
    // What this call needs from the owner, if it has one. Everything is in
    // bytes except the element counts of the scratch vectors.
    let element = core::mem::size_of::<T::Real>();
    let req = WorkspaceReq {
        a_bytes: ap_len * element,
        tile_bytes: (fam.tile + fam.induced_scratch(kc)) * element + 2 * grid_elems * element,
        worker_scatter: 0,
        // A direct-B call never touches the panel, so it asks for none.
        b_bytes: if direct_b {
            0
        } else {
            panels * b_group * element
        },
        team_scatter: am.len().div_ceil(mr)
            + bn.len().div_ceil(nr)
            + dm.len().div_ceil(mr)
            + dn.len().div_ceil(nr)
            + if beta == T::zero() {
                0
            } else {
                cm.len().div_ceil(mr)
            },
        barriers: if pm > 1 { pn } else { 0 },
    };
    if let Some(st) = stats {
        st.note_call(p, req.b_bytes);
    }
    let mut lease = workspace.map(|ws| ws.take_team(&req, pm, pn));
    // The panel lives in the lease when there is one, and in a call-local
    // allocation otherwise; a direct-B call sizes it to zero either way. The
    // pointer is taken before the scatter borrow, which is held to the end.
    let has_lease = lease.is_some();
    let mut local_bp = match has_lease || direct_b {
        true => None,
        false => Some(Panel::<T::Real>::new(panels * b_group)),
    };
    let bp_ptr = match lease.as_mut() {
        Some(l) => l.panel(req.b_bytes) as *mut T::Real,
        None => match local_bp.as_mut() {
            Some(p) => p.as_mut_ptr(),
            // A direct-B call reads B in place, so it has no panel at all.
            None => std::ptr::NonNull::<T::Real>::dangling().as_ptr(),
        },
    };
    // A leased team set also carries the scatter vectors and the barriers, so a
    // steady-state call reuses their capacity instead of allocating five
    // vectors and a barrier list. Both are taken as shared slices once the
    // filling is done, so they can be borrowed together.
    let mut local_scatter = Vec::new();
    let local_bars: Vec<Barrier>;
    let runs = {
        let buf = match lease.as_mut() {
            Some(l) => l.scatter_mut(),
            None => &mut local_scatter,
        };
        buf.clear();
        buf.reserve(req.team_scatter);
        ScatterRuns {
            a: append_block_scatter(buf, am, mr),
            b: append_block_scatter(buf, bn, nr),
            dm: append_block_scatter(buf, dm, mr),
            dn: append_block_scatter(buf, dn, nr),
            cm: match beta == T::zero() {
                true => (buf.len(), buf.len()),
                false => append_block_scatter(buf, cm, mr),
            },
        }
    };
    let (scatter_buf, bars) = match lease.as_ref() {
        Some(l) => (l.scatter(), l.barriers()),
        None => {
            local_bars = (0..pn).map(|_| Barrier::new(pm)).collect();
            (local_scatter.as_slice(), local_bars.as_slice())
        }
    };

    let cx = Ctx::<T> {
        plan,
        fam,
        packers: rg.packers::<T>(),
        emitter: rg.emitter::<T>(),
        call,
        direct_b,
        mr,
        nr,
        mc,
        kc,
        nc,
        m,
        n,
        k,
        b_group,
        am,
        ak,
        bk,
        bn,
        cm,
        cn,
        dm,
        dn,
        ha,
        hb,
        hc,
        scatter: scatter_buf,
        runs,
        blocked,
        grid_off,
        grid_elems,
        conj_a,
        conj_b,
        alpha,
        beta,
        a: Shared(ptr_a as *mut T),
        b: Shared(ptr_b as *mut T),
        c: Shared(c as *mut T),
        d: Shared(d),
        bp: Shared(bp_ptr),
    };

    // Worker buffers come from the owner when there is one, and from a fresh
    // per-call panel otherwise; either way they are the caller-of-`f`'s for the
    // duration of `f`, and the 4m scratch is carved out of the tile.
    let with_buffers = |f: &mut dyn FnMut(*mut T::Real, *mut T::Real)| match workspace {
        Some(ws) => {
            ws.with_worker(&req, &mut |a, tile, _| {
                f(a as *mut T::Real, tile as *mut T::Real)
            });
        }
        None => {
            let mut ap = Panel::<T::Real>::new(ap_len);
            // The same region the workspace request asks for: tile, induced scratch and
            // the blocked path's two tile grids.
            let mut tile =
                Panel::<T::Real>::new(fam.tile + fam.induced_scratch(kc) + 2 * grid_elems);
            f(ap.as_mut_ptr(), tile.as_mut_ptr());
        }
    };
    let scratch_off = fam.tile;

    if lanes > 1 {
        // Barrier-free: lane `i` runs a contiguous share of the batch entries,
        // each with the serial blocking and its own `A`, `B` and tile, so every
        // entry's arithmetic is the serial path's.
        let items = plan.stats.batch;
        let lane = |i: usize| {
            let bpart = BPart {
                panel: i,
                ..BPart::SERIAL
            };
            with_buffers(&mut |ap, tile| {
                // SAFETY: `execute`'s contract covers the accesses; the lanes'
                // item ranges are disjoint, so they write disjoint entries of
                // `D`, and each lane's buffers and `B` slice are its own.
                unsafe {
                    run_strip::<T>(
                        &cx,
                        i * items / lanes..(i + 1) * items / lanes,
                        0,
                        m,
                        ap,
                        tile,
                        tile.add(scratch_off),
                        bpart,
                    )
                }
            });
        };
        // Barrier-free lanes on the context's pool: from outside it the call
        // enters the pool once, and from one of its workers it runs in place and
        // spawns the lanes into the pool it already belongs to.
        exec.for_each_partition(lanes, &lane);
        return Ok(());
    }

    if p == 1 {
        with_buffers(&mut |ap, tile| {
            // SAFETY: `execute`'s contract, and these buffers are exclusive to
            // this call for its duration.
            unsafe {
                run_strip::<T>(
                    &cx,
                    0..plan.stats.batch,
                    0,
                    m,
                    ap,
                    tile,
                    tile.add(scratch_off),
                    BPart::SERIAL,
                )
            }
        });
        return Ok(());
    }

    // One barrier per column group, each shared by exactly the `pm` threads
    // that write and read that group's slice of the packed `B` panel. Groups
    // never need to synchronise with each other, so they do not: the barrier is
    // `pm`-way, not `p`-way. At `pm == 1` there is nothing to synchronise and
    // the threads take no barrier at all — which `bars` being empty expresses.

    // One thread's whole job, as a function of its index in the `pm x pn` grid.
    // Written once; the broadcast below runs it on the `Exec`'s workers. The
    // partition, the strips and therefore the arithmetic are the same at every
    // width, which is why the result stays bitwise identical to serial.
    let claim = std::sync::atomic::AtomicUsize::new(0);
    let cell = |t: usize| {
        let cx = &cx;
        if let Some((job_m, job_n, _)) = dyn_jobs {
            with_buffers(&mut |ap, tile| {
                let bufs = Bufs {
                    ap,
                    tile,
                    scratch: tile.add(scratch_off),
                    grid: grid_base(tile, grid_off, grid_elems, false),
                    grid2: grid_base(tile, grid_off, grid_elems, true),
                    ntiles: (n.div_ceil(nr)).max(1),
                };
                // SAFETY: `execute`'s contract covers the accesses; the claim
                // counter hands each job to one worker, and jobs partition the
                // output within an epoch. The barrier has `p` participants.
                unsafe {
                    dynamic::run_dynamic::<T>(cx, t, p, job_m, job_n, &claim, &bars[0], bufs, stats)
                };
            });
            return;
        }
        let (r, g) = (t / pn, t % pn);
        let bpart = BPart {
            g,
            panel: g,
            pn,
            r,
            pm,
            bar: (pm > 1).then(|| &bars[g]),
        };
        let (lo, hi) = tcontract_kernel::partition::strip(r, pm, cx.m, mr, align);
        with_buffers(&mut |ap, tile| {
            // SAFETY: `execute`'s contract covers the accesses; the strips and
            // column groups partition the output, so this thread's writes are
            // disjoint from every other thread's.
            unsafe {
                run_strip::<T>(
                    cx,
                    0..plan.stats.batch,
                    lo,
                    hi,
                    ap,
                    tile,
                    tile.add(scratch_off),
                    bpart,
                )
            };
        });
    };

    // `broadcast` runs nothing when it declines, so a refusal is a real
    // either/or and never a partial execution. The worker case was rejected
    // above, so what is left is the co-scheduled team.
    if !spmd {
        // `pm == 1`: every column group has exactly one thread and writes and
        // reads only its own slice of the panel, so no barrier is taken and the
        // cells are an ordinary barrier-free partition. The arithmetic and the
        // active-width blocking are the same as the broadcast path's.
        exec.for_each_partition(p, &cell);
        return Ok(());
    }
    exec.broadcast(p, &cell)?;
    Ok(())
}
