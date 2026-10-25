//! `PartitionPolicy::DynamicTiles`: coverage, determinism against the static
//! and serial runs, the epoch protocol's counters, and the rejection cases.
//!
//! Blocking is frozen with `Plan::with_blocking` wherever bitwise equality is
//! claimed, so only the scheduling differs between the runs compared.
#![allow(clippy::too_many_arguments)]
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering, Ordering::SeqCst};

use super::compat::{contract_reference, RefOperand};
use super::compat::{
    dynamic_report, execute_resolved, execute_resolved_instrumented, Layout, Operand, Plan,
};
use crate::api::Scalar;
use crate::driver::{Assignment, DynStats};
use tcontract_exec::{ArenaProvider, Exec, Pool, WorkspaceProvider};
use tcontract_kernel::{
    Blocking, KernelChoice, PartitionOpts, PartitionPolicy, SelectError, C32, C64,
};
use tcontract_kernel::{Element, Real};

/// Workers of a [`Team`]'s pool are named `NAME0`, `NAME1`, ...
const WORKER_PREFIX: &str = "dyn-team-";

/// A team of `width` workers on a pool of its own, and optionally a lent
/// workspace. A refusing team runs the whole execution on one of its own
/// workers, where a barrier-bearing team cannot be co-scheduled, so the driver
/// plans the call again without a team.
struct Team {
    width: usize,
    refuse: bool,
    ws: Option<ArenaProvider>,
    pool: Pool<'static>,
    /// Ledger root of the token the last execution ran on (0 before any).
    root: AtomicU64,
}
impl Team {
    fn new(width: usize) -> Self {
        let tp = strided_exec::Pool::builder()
            .threads(width.max(2))
            .thread_name(|i| format!("{WORKER_PREFIX}{i}"))
            .build()
            .unwrap();
        Self {
            width,
            refuse: false,
            ws: None,
            pool: Pool::owned(tp),
            root: AtomicU64::new(0),
        }
    }
    fn exec(&self) -> Exec<'_> {
        Exec::new(&self.pool, self.width).unwrap()
    }
    /// Times the last execution's work entered the pool from outside it.
    fn installs(&self) -> usize {
        strided_exec::ledger::installs(self.root.load(Ordering::Relaxed))
    }
}

fn value<T: Element>(i: usize) -> T {
    let r = T::Real::from_f64((i % 97) as f64 * 0.13 - 3.0);
    let im = T::Real::from_f64((i % 31) as f64 * 0.07 - 1.0);
    T::from_parts(r, if T::IS_COMPLEX { im } else { T::Real::ZERO })
}

#[derive(Clone, Copy, Debug)]
struct Spec {
    m: usize,
    n: usize,
    k: usize,
    batch: usize,
    /// Store D row-major (the driver exchanges the operands).
    row_major_d: bool,
    /// Pad every leading stride so the layouts are not compact.
    padded: bool,
    /// C is D (aliased) rather than a distinct buffer.
    alias: bool,
    alpha: f64,
    beta: f64,
    conj_a: bool,
    conj_d: bool,
    blocking: Blocking,
}
impl Spec {
    fn new(m: usize, n: usize, k: usize) -> Self {
        Self {
            m,
            n,
            k,
            batch: 1,
            row_major_d: false,
            padded: false,
            alias: false,
            alpha: 1.3,
            beta: -0.4,
            conj_a: false,
            conj_d: false,
            blocking: Blocking {
                mc: 8,
                kc: 8,
                nc: 16,
            },
        }
    }
}

struct Built {
    la: Layout,
    lb: Layout,
    ld: Layout,
    idx: [Vec<i64>; 3],
}
fn layouts(s: &Spec) -> Built {
    let pad = if s.padded { 3 } else { 0 };
    let (m, n, k, b) = (s.m as i64, s.n as i64, s.k as i64, s.batch as i64);
    let batched = s.batch > 1;
    let mk = |rows: i64, cols: i64, row_major: bool| {
        let (ld, other) = if row_major {
            (cols + pad, rows)
        } else {
            (rows + pad, cols)
        };
        let (rs, cs) = if row_major { (ld, 1) } else { (1, ld) };
        let mut ext = vec![rows, cols];
        let mut st = vec![rs, cs];
        if batched {
            ext.push(b);
            st.push(ld * other);
        }
        Layout::new(ext, st).unwrap()
    };
    let h = |v: &[i64]| {
        let mut v = v.to_vec();
        if batched {
            v.push(3);
        }
        v
    };
    Built {
        la: mk(m, k, s.row_major_d),
        lb: mk(k, n, false),
        ld: mk(m, n, s.row_major_d),
        idx: [h(&[0, 2]), h(&[2, 1]), h(&[0, 1])],
    }
}

/// One execution. `policy == None` is the default static selection.
fn run<T>(
    id: &str,
    s: &Spec,
    policy: Option<PartitionPolicy>,
    team: &Team,
    stats: Option<&DynStats>,
) -> Vec<T>
where
    T: Scalar,
{
    try_run::<T>(id, s, policy, team, stats).expect("this team must serve the plan")
}

/// [`run`] keeping the driver's route error.
fn try_run<T>(
    id: &str,
    s: &Spec,
    policy: Option<PartitionPolicy>,
    team: &Team,
    stats: Option<&DynStats>,
) -> crate::api::Result<Vec<T>>
where
    T: Scalar,
{
    if team.refuse {
        team.exec()
            .into_token()
            .external(|_| run_on_team::<T>(id, s, policy, team, stats))
    } else {
        run_on_team::<T>(id, s, policy, team, stats)
    }
}

fn run_on_team<T>(
    id: &str,
    s: &Spec,
    policy: Option<PartitionPolicy>,
    team: &Team,
    stats: Option<&DynStats>,
) -> crate::api::Result<Vec<T>>
where
    T: Scalar,
{
    let mut exec = team.exec();
    team.root.store(
        exec.token()
            .ledger_root()
            .expect("a pooled token is a root"),
        Ordering::Relaxed,
    );
    let workspace = team.ws.as_ref().map(|w| w as &dyn WorkspaceProvider);
    let b = layouts(s);
    let a: Vec<T> = (0..b.la.storage_len() as usize).map(value).collect();
    let bv: Vec<T> = (0..b.lb.storage_len() as usize)
        .map(|i| value::<T>(i + 7))
        .collect();
    let mut d: Vec<T> = (0..b.ld.storage_len() as usize)
        .map(|i| value::<T>(i + 3))
        .collect();
    let c0 = d.clone();
    let oa = Operand::new(&b.la, &b.idx[0]);
    let oa = if s.conj_a { oa.conj() } else { oa };
    let od = Operand::new(&b.ld, &b.idx[2]);
    let od = if s.conj_d { od.conj() } else { od };
    let oc = Operand::new(&b.ld, &b.idx[2]);
    let mut plan = Plan::new(oa, Operand::new(&b.lb, &b.idx[1]), Some(oc), od)
        .unwrap()
        .with_kernel(KernelChoice::Id(id.into()))
        .unwrap()
        .with_blocking(s.blocking)
        .with_threads(team.width);
    if let Some(p) = policy {
        plan = plan.with_partition(p, PartitionOpts::default());
    }
    let rg = plan.resolved::<T>().unwrap();
    let alpha = T::from_parts(T::Real::from_f64(s.alpha), T::Real::ZERO);
    let beta = T::from_parts(T::Real::from_f64(s.beta), T::Real::ZERO);
    let cptr = if s.alias { d.as_ptr() } else { c0.as_ptr() };
    // SAFETY: buffers are sized by their layouts; D is exclusive and distinct
    // from A and B; C is either D itself or a separate copy.
    unsafe {
        match stats {
            Some(st) => execute_resolved_instrumented(
                &plan,
                &rg,
                &mut exec,
                workspace,
                st,
                alpha,
                a.as_ptr(),
                bv.as_ptr(),
                beta,
                cptr,
                d.as_mut_ptr(),
            ),
            None => execute_resolved(
                &plan,
                &rg,
                &mut exec,
                workspace,
                alpha,
                a.as_ptr(),
                bv.as_ptr(),
                beta,
                cptr,
                d.as_mut_ptr(),
            ),
        }
    }?;
    Ok(d)
}

fn dynamic(job_m: usize, job_n: usize) -> Option<PartitionPolicy> {
    Some(PartitionPolicy::DynamicTiles { job_m, job_n })
}

const F64: &str = "ref.f64.real.4x4";

/// Dynamic equals static equals serial, bitwise, at every width.
fn assert_bitwise<T>(id: &str, s: Spec, job: (usize, usize))
where
    T: Scalar,
{
    let serial = run::<T>(id, &s, None, &Team::new(1), None);
    let grid = Some(PartitionPolicy::StaticGrid { pm: 2, pn: 2 });
    let stat = run::<T>(id, &s, grid, &Team::new(4), None);
    assert!(serial == stat, "{id}: static grid differs from serial");
    for w in [1, 2, 3, 4, 8] {
        let got = run::<T>(id, &s, dynamic(job.0, job.1), &Team::new(w), None);
        assert!(got == serial, "{id} {s:?} job {job:?} width {w}");
    }
}

#[test]
fn bitwise_equal_to_static_and_serial_across_widths_and_modes() {
    // Few bands (2-D tile mode at width 4+), many bands (row-band mode),
    // ragged M/N/NC tails, several K slabs (kc = 8, k = 45).
    for (spec, job) in [
        (Spec::new(21, 70, 45), (8, 8)),
        (Spec::new(21, 70, 45), (4, 4)),
        (Spec::new(70, 21, 45), (16, 8)),
        (Spec::new(5, 130, 17), (4, 32)),
        (Spec::new(64, 3, 9), (64, 4)),
    ] {
        assert_bitwise::<f64>(F64, spec, job);
    }
}

#[test]
fn every_dtype_family_and_orientation_matches_bitwise() {
    let mut s = Spec::new(26, 38, 21);
    for row_major in [false, true] {
        s.row_major_d = row_major;
        assert_bitwise::<f32>("ref.f32.real.4x4", s, (8, 8));
        assert_bitwise::<f64>(F64, s, (8, 12));
        assert_bitwise::<C32>("ref.c32.native.4x4", s, (8, 8));
        assert_bitwise::<C64>("ref.c64.native.4x4", s, (4, 16));
        assert_bitwise::<C64>("ref.c64.i1m.2x4", s, (8, 8));
        assert_bitwise::<C64>("ref.c64.i4m.4x4", s, (8, 8));
    }
}

#[test]
fn operations_strides_batches_and_aliasing_are_preserved() {
    let base = Spec::new(19, 27, 14);
    let cases = [
        Spec {
            alias: true,
            ..base
        },
        Spec { beta: 0.0, ..base },
        Spec {
            padded: true,
            ..base
        },
        Spec { batch: 3, ..base },
        Spec {
            conj_a: true,
            conj_d: true,
            ..base
        },
        Spec {
            row_major_d: true,
            batch: 2,
            padded: true,
            alias: true,
            ..base
        },
    ];
    for s in cases {
        assert_bitwise::<f64>(F64, s, (8, 8));
        assert_bitwise::<C64>("ref.c64.native.4x4", s, (8, 8));
    }
}

#[test]
fn results_match_the_reference_oracle() {
    let s = Spec {
        conj_a: true,
        ..Spec::new(23, 31, 40)
    };
    let b = layouts(&s);
    let a: Vec<C64> = (0..b.la.storage_len() as usize).map(value).collect();
    let bv: Vec<C64> = (0..b.lb.storage_len() as usize)
        .map(|i| value::<C64>(i + 7))
        .collect();
    let c0: Vec<C64> = (0..b.ld.storage_len() as usize)
        .map(|i| value::<C64>(i + 3))
        .collect();
    let mut want = c0.clone();
    contract_reference::<C64>(
        C64::from_parts(1.3, 0.0),
        &RefOperand {
            data: &a,
            layout: &b.la,
            idx: &b.idx[0],
            op: super::compat::ElementOp::Conjugate,
        },
        &RefOperand {
            data: &bv,
            layout: &b.lb,
            idx: &b.idx[1],
            op: super::compat::ElementOp::Identity,
        },
        C64::from_parts(-0.4, 0.0),
        Some(&RefOperand {
            data: &c0,
            layout: &b.ld,
            idx: &b.idx[2],
            op: super::compat::ElementOp::Identity,
        }),
        &mut want,
        &b.ld,
        &b.idx[2],
        super::compat::ElementOp::Identity,
    )
    .unwrap();
    let got = run::<C64>("ref.c64.native.4x4", &s, dynamic(8, 8), &Team::new(4), None);
    let err: f64 = got
        .iter()
        .zip(&want)
        .map(|(x, y)| x.sub(*y).norm())
        .fold(0.0, f64::max);
    assert!(err < 1e-10, "max abs error {err:e}");
}

/// Expected epochs for the spec at active width `p`, from the same retargeted
/// blocking the driver uses.
struct Expect {
    epochs: usize,
    b_slivers: usize,
    a_elems_bands: usize,
    a_elems_tiles: usize,
    kslabs: usize,
    tiles_m: usize,
    tiles_n: usize,
}
fn expect(s: &Spec, mr: usize, nr: usize, job_n: usize) -> Expect {
    let (mut m, mut n) = (s.m, s.n);
    if s.row_major_d {
        // Oriented rows are the user's N when the driver exchanges roles; the
        // counts below use whichever the report says, so callers pass swapped
        // specs for that case.
        std::mem::swap(&mut m, &mut n);
    }
    let nc = s
        .blocking
        .nc
        .next_multiple_of(nr)
        .min(n.next_multiple_of(nr));
    let kc = s.blocking.kc;
    let kslabs = s.k.div_ceil(kc);
    let (mut epochs, mut b_slivers, mut ab, mut at) = (0, 0, 0, 0);
    let mut jc = 0;
    while jc < n {
        let jc_len = nc.min(n - jc);
        for slab in 0..kslabs {
            let pc_len = kc.min(s.k - slab * kc);
            epochs += 1;
            b_slivers += jc_len.div_ceil(nr);
            ab += m * pc_len;
            at += m * pc_len * jc_len.div_ceil(job_n);
        }
        jc += nc;
    }
    Expect {
        epochs: epochs * s.batch,
        b_slivers: b_slivers * s.batch,
        a_elems_bands: ab * s.batch,
        a_elems_tiles: at * s.batch,
        kslabs,
        tiles_m: m.div_ceil(mr),
        tiles_n: n.div_ceil(nr),
    }
}

#[test]
fn jobs_cover_every_tile_exactly_once_and_counters_follow_the_protocol() {
    // 4x4 family. Width 2 with 4 bands: row-band mode. Width 8 with 4 bands:
    // 2-D mode. Repeated executions with a reset in between.
    for (s, job, width, mode) in [
        (Spec::new(32, 70, 45), (8, 8), 2, Assignment::RowBands),
        (Spec::new(32, 70, 45), (8, 8), 8, Assignment::Tiles2D),
        (
            Spec {
                batch: 2,
                ..Spec::new(13, 21, 19)
            },
            (4, 4),
            3,
            Assignment::RowBands,
        ),
    ] {
        let e = expect(&s, 4, 4, job.1);
        let stats = DynStats::with_coverage(8, e.tiles_m, e.tiles_n);
        for round in 0..2 {
            stats.reset();
            let team = Team::new(width);
            let serial = run::<f64>(F64, &s, None, &Team::new(1), None);
            let got = run::<f64>(F64, &s, dynamic(job.0, job.1), &team, Some(&stats));
            assert!(got == serial, "{s:?} round {round}");
            let snap = stats.snapshot();
            assert_eq!(snap.epochs, e.epochs, "{s:?}");
            assert_eq!(
                snap.b_slivers_packed, e.b_slivers,
                "B packed once per sliver per epoch"
            );
            match mode {
                Assignment::RowBands => {
                    assert_eq!((snap.band_epochs, snap.tile_epochs), (e.epochs, 0));
                    assert_eq!(
                        snap.a_elems_packed, e.a_elems_bands,
                        "A once per band chunk"
                    );
                }
                Assignment::Tiles2D => {
                    assert_eq!((snap.band_epochs, snap.tile_epochs), (0, e.epochs));
                    assert_eq!(
                        snap.a_elems_packed, e.a_elems_tiles,
                        "A packed once per column job: the documented 2-D ceiling"
                    );
                    assert!(snap.a_elems_packed > e.a_elems_bands);
                }
            }
            assert_eq!(snap.jobs_per_worker.iter().sum::<usize>(), snap.claims);
            let visits = stats.tile_visits();
            let want = (s.batch * e.kslabs) as u32;
            assert!(
                visits.iter().all(|&v| v == want),
                "every tile exactly once per slab: {visits:?}"
            );
        }
    }
}

#[test]
fn the_report_names_the_policy_width_and_mode() {
    let s = Spec::new(32, 70, 45);
    let b = layouts(&s);
    let plan = Plan::new(
        Operand::new(&b.la, &b.idx[0]),
        Operand::new(&b.lb, &b.idx[1]),
        None,
        Operand::new(&b.ld, &b.idx[2]),
    )
    .unwrap()
    .with_kernel(KernelChoice::Id(F64.into()))
    .unwrap()
    .with_partition(
        PartitionPolicy::DynamicTiles { job_m: 8, job_n: 8 },
        PartitionOpts::default(),
    );
    let rg = plan.resolved::<f64>().unwrap();
    let r = dynamic_report(&plan, &rg, 2).unwrap();
    assert_eq!(
        (r.job_m, r.job_n, r.row_bands, r.active_width),
        (8, 8, 4, 2)
    );
    assert_eq!(r.assignment, Assignment::RowBands);
    assert_eq!(
        dynamic_report(&plan, &rg, 64).unwrap().assignment,
        Assignment::Tiles2D
    );
    // Fewer jobs than the budget caps the team; a static resolution has no report.
    let tiny = Spec::new(4, 4, 4);
    let tb = layouts(&tiny);
    let tp = Plan::new(
        Operand::new(&tb.la, &tb.idx[0]),
        Operand::new(&tb.lb, &tb.idx[1]),
        None,
        Operand::new(&tb.ld, &tb.idx[2]),
    )
    .unwrap()
    .with_kernel(KernelChoice::Id(F64.into()))
    .unwrap()
    .with_partition(
        PartitionPolicy::DynamicTiles { job_m: 4, job_n: 4 },
        PartitionOpts::default(),
    );
    let trg = tp.resolved::<f64>().unwrap();
    assert_eq!(dynamic_report(&tp, &trg, 8).unwrap().active_width, 1);
    let st = Plan::new(
        Operand::new(&b.la, &b.idx[0]),
        Operand::new(&b.lb, &b.idx[1]),
        None,
        Operand::new(&b.ld, &b.idx[2]),
    )
    .unwrap();
    assert!(dynamic_report(&st, &st.resolved::<f64>().unwrap(), 4).is_none());
}

#[test]
fn width_one_runs_on_the_caller_without_claims_and_a_refusal_falls_back() {
    let s = Spec::new(32, 70, 45);
    let serial = run::<f64>(F64, &s, None, &Team::new(1), None);
    let stats = DynStats::new(4);
    let one = Team::new(1);
    let got = run::<f64>(F64, &s, dynamic(8, 8), &one, Some(&stats));
    assert!(got == serial);
    assert_eq!(one.installs(), 0, "width 1 never enters the pool");
    assert_eq!(stats.snapshot().claims, 0, "no atomic claims at width 1");

    // A same-pool worker cannot be co-scheduled at a barrier, so the dynamic
    // team is refused before anything runs and the call is planned again as
    // barrier-free cells of the same blocking: the numbers are the serial ones,
    // no epoch or claim of the dynamic protocol ran, and the cells entered no
    // pool (they ran in place on the worker).
    let stats = DynStats::new(4);
    let mut refuse = Team::new(4);
    refuse.refuse = true;
    let got = run::<f64>(F64, &s, dynamic(8, 8), &refuse, Some(&stats));
    assert!(got == serial, "the fallback changed the result");
    let snap = stats.snapshot();
    assert_eq!((snap.epochs, snap.claims), (0, 0), "the dynamic team ran");
    assert_eq!(
        snap.calls, 1,
        "one counted call: the refused plan is not one"
    );
    assert_eq!(refuse.installs(), 0, "the cells ran in place on the worker");
}

#[test]
fn direct_c_and_direct_b_families_run_in_both_assignment_modes() {
    for id in ["ref.f64.direct.4x4", "ref.f64.direct-b.4x4"] {
        for (s, job, width) in [
            (Spec::new(32, 70, 45), (8, 8), 2),
            (Spec::new(32, 70, 45), (8, 8), 8),
            // C is D: the direct kernel may update D in place.
            (
                Spec {
                    alias: true,
                    ..Spec::new(32, 70, 45)
                },
                (8, 8),
                3,
            ),
            // A distinct C cannot be updated in place and takes the scratch
            // fallback of the same family.
            (
                Spec {
                    alias: false,
                    ..Spec::new(24, 40, 20)
                },
                (8, 8),
                4,
            ),
            (
                Spec {
                    beta: 0.0,
                    ..Spec::new(24, 40, 20)
                },
                (4, 4),
                4,
            ),
            // Output conjugation disables the direct tile path entirely.
            (
                Spec {
                    conj_d: true,
                    alias: true,
                    ..Spec::new(24, 40, 20)
                },
                (8, 8),
                4,
            ),
        ] {
            let serial = run::<f64>(id, &s, None, &Team::new(1), None);
            let got = run::<f64>(id, &s, dynamic(job.0, job.1), &Team::new(width), None);
            assert!(got == serial, "{id} {s:?} width {width}");
        }
    }
}

#[test]
fn direct_b_allocates_no_b_buffer_and_packs_no_slivers() {
    let s = Spec {
        alias: true,
        ..Spec::new(32, 64, 45)
    };
    let stats = DynStats::new(4);
    let mut team = Team::new(4);
    team.ws = Some(ArenaProvider::new());
    let serial = run::<f64>("ref.f64.direct-b.4x4", &s, None, &Team::new(1), None);
    let got = run::<f64>(
        "ref.f64.direct-b.4x4",
        &s,
        dynamic(8, 8),
        &team,
        Some(&stats),
    );
    assert!(got == serial);
    let snap = stats.snapshot();
    assert_eq!(snap.b_bytes_requested, 0);
    assert_eq!(snap.b_slivers_packed, 0);
    assert!(
        snap.epochs > 0,
        "the epoch protocol (counter reset, ordering) still runs"
    );
    assert_eq!(team.ws.as_ref().unwrap().retained_panel_bytes(), 0);

    // The packed-B family of the same shape does ask for a panel.
    let packed = DynStats::new(4);
    run::<f64>(F64, &s, dynamic(8, 8), &Team::new(4), Some(&packed));
    assert!(packed.snapshot().b_bytes_requested > 0);
}

#[test]
fn invalid_policies_are_rejected_at_resolution_even_for_empty_problems() {
    for (m, n, k) in [(16, 16, 8), (0, 8, 8), (8, 8, 0)] {
        let s = Spec::new(m, n, k);
        let b = layouts(&s);
        let base = || {
            Plan::new(
                Operand::new(&b.la, &b.idx[0]),
                Operand::new(&b.lb, &b.idx[1]),
                None,
                Operand::new(&b.ld, &b.idx[2]),
            )
            .unwrap()
            .with_kernel(KernelChoice::Id(F64.into()))
            .unwrap()
        };
        let bad = |policy, opts| {
            base()
                .with_partition(policy, opts)
                .resolved::<f64>()
                .unwrap_err()
        };
        let d = PartitionOpts::default();
        for (jm, jn) in [(0, 8), (8, 0), (6, 8), (8, 6)] {
            assert!(matches!(
                bad(
                    PartitionPolicy::DynamicTiles {
                        job_m: jm,
                        job_n: jn
                    },
                    d
                ),
                SelectError::Incompatible { .. }
            ));
        }
        // DynamicTiles with `align_c_lines` is unrepresentable now: the typed
        // `Partition::DynamicTiles` carries no alignment option.
        assert!(base()
            .with_partition(PartitionPolicy::DynamicTiles { job_m: 8, job_n: 8 }, d)
            .resolved::<f64>()
            .is_ok());
    }
}

// ---- skewed worker progress, with deterministic coordination ---------------

mod skew {
    use super::*;
    use std::cell::Cell;
    use tcontract_kernel::{KernelFamily, UkrFn};

    /// Worker 0 of the team's pool is the slow one.
    fn slow() -> bool {
        std::thread::current().name() == Some(&format!("{WORKER_PREFIX}0")[..])
    }
    static ENTERED: AtomicBool = AtomicBool::new(false);
    static OTHER_CALLS: AtomicUsize = AtomicUsize::new(0);
    static TIMED_OUT: AtomicBool = AtomicBool::new(false);
    /// Of the 224 tile calls the other workers make for the seven jobs worker 0
    /// does not hold: by 200, every one of those jobs has been claimed.
    const THRESHOLD: usize = 200;

    fn wait_for(cond: impl Fn() -> bool) {
        let start = std::time::Instant::now();
        while !cond() {
            if start.elapsed() > std::time::Duration::from_secs(20) {
                // Record rather than panic: unwinding out of a kernel would
                // strand the other workers at a barrier.
                TIMED_OUT.store(true, SeqCst);
                return;
            }
            std::thread::yield_now();
        }
    }

    /// Worker 0's first tile waits until the others have finished many tiles;
    /// the others' first tile waits until worker 0 has claimed a job (entered
    /// its first tile). So worker 0 provably holds a job while stalled, and the
    /// rest of the epoch is provably finished by the others.
    unsafe fn skewed(k: usize, a: *const f64, b: *const f64, out: *mut f64) {
        thread_local! { static FIRST: Cell<bool> = const { Cell::new(true) }; }
        let first = FIRST.replace(false);
        if slow() {
            if first {
                ENTERED.store(true, SeqCst);
                wait_for(|| OTHER_CALLS.load(SeqCst) >= THRESHOLD);
            }
        } else {
            if first {
                wait_for(|| ENTERED.load(SeqCst));
            }
            OTHER_CALLS.fetch_add(1, SeqCst);
        }
        // SAFETY: the same panel/tile ABI as the portable 4x4 kernel.
        unsafe { tcontract_kernel::portable::real_tile::<f64, 4, 4>(k, a, b, out) };
    }

    fn manifest() -> &'static [&'static KernelFamily<f64>] {
        static LIST: std::sync::OnceLock<[&'static KernelFamily<f64>; 1]> =
            std::sync::OnceLock::new();
        LIST.get_or_init(|| {
            let mut f = **tcontract_kernel::portable::families_f64()
                .iter()
                .find(|f| f.id == "ref.f64.real.4x4")
                .unwrap();
            f.id = "test.skew.f64.4x4";
            f.allow_auto = false;
            f.ukr = UkrFn::Tile(skewed);
            [Box::leak(Box::new(f))]
        })
    }

    #[test]
    fn a_stalled_worker_does_not_hold_up_the_rest_and_the_result_is_exact() {
        // SAFETY: the manifest copies the validated portable 4x4 footprint,
        // ISA and overwrite contract; `skewed` only waits and then calls it.
        unsafe { tcontract_kernel::register::<f64>(manifest) };
        // One epoch (k <= kc), 8 bands of 64 columns: worker 0 holds one band
        // while the other three finish the remaining seven.
        let s = Spec {
            blocking: Blocking {
                mc: 8,
                kc: 64,
                nc: 64,
            },
            ..Spec::new(64, 64, 32)
        };
        let serial = run::<f64>("ref.f64.real.4x4", &s, None, &Team::new(1), None);
        let stats = DynStats::new(4);
        let team = Team::new(4);
        let got = run::<f64>("test.skew.f64.4x4", &s, dynamic(8, 8), &team, Some(&stats));
        assert!(!TIMED_OUT.load(SeqCst), "coordination timed out");
        assert!(ENTERED.load(SeqCst), "worker 0 really did hold a job");
        assert!(got == serial);
        let jobs = stats.snapshot().jobs_per_worker;
        assert_eq!(jobs.iter().sum::<usize>(), 8, "{jobs:?}");
        assert_eq!(
            jobs[0], 1,
            "the stalled worker finished only its one job: {jobs:?}"
        );
    }
}

#[test]
fn a_worker_that_claims_no_work_still_takes_both_barriers() {
    // Eight workers requested, but only two jobs exist: the width is capped by
    // the jobs, and a narrow tail epoch (n = 70, nc = 16 leaves a 6-column
    // block) must not strand anyone. Several K slabs make many epochs.
    let s = Spec::new(8, 70, 45);
    let serial = run::<f64>(F64, &s, None, &Team::new(1), None);
    for w in [2, 3, 8] {
        let got = run::<f64>(F64, &s, dynamic(8, 32), &Team::new(w), None);
        assert!(got == serial, "width {w}");
    }
}
