//! Where the planner's faer route and the packed driver cross over, per dtype
//! and per thread count, on copy-free fusable contractions.
//!
//! One `Problem` per case is planned three times: with [`PlanConfig::default`]
//! (what the planner itself chooses, `faer` below the dtype's `FaerLimit` and
//! packed above it), with [`PlanConfig::packed`] (the packed driver forced) and
//! with `PlanConfig { faer_limit: FaerLimit::NONE, ..PlanConfig::default() }`
//! (`faer` forced for every fusable problem, the rule before #63). Only the
//! execution is timed, on prebuilt plans and a preallocated output, so the
//! number is the execution of the plan and neither its construction nor any
//! allocation of `A`/`B`/`D`.
//!
//! The three arms share one `packed` statistic row (`batch,m,n,k,mnk`), and
//! every arm is checked against the same naive label oracle before any timing;
//! the `faer_forced` arm is the only one that reaches `faer` above the dtype's
//! `FaerLimit`, so its `ns` is the c64 faer time the `default` route cannot
//! show.
//!
//! Two C modes are recorded per case:
//!
//! * `absent` — `CSpec::Absent`, `Plan::execute_into`, i.e. `D = alpha * A * B`
//!   overwriting a zero `D` (the overwrite form).
//! * `output` — `CSpec::Output(Op::Identity)`, `Plan::execute_into_accum` with
//!   `beta = 1` and [`AccumulationSource::Output`], i.e. `D += alpha * A * B`
//!   in place over a non-zero `D`, the form an MPS step pays.
//!
//! In both modes the correctness oracle is computed from the labels: the
//! `absent` oracle is the bare product; the `output` oracle is `alpha * sum +
//! beta * D_old`.
//!
//! This answers the size question of [tcontract-rs#63] / [#69]: the crossover of
//! the two engines on the classes that fuse to one strided batched GEMM is a
//! measurement, at 1T and 4T, for `f32`, `f64`, `c32` and `c64`. Nothing here
//! decides a `FaerLimit` value; it records the two arms.
//!
//! [tcontract-rs#63]: https://github.com/tensor4all/tcontract-rs/issues/63
//! [#69]: https://github.com/tensor4all/tcontract-rs/issues/69
//!
//! Shape classes, column-major contiguous, labels as in
//! `experiments/three-engine-contract`:
//!
//! * `gemm`         `D[i,k] = sum_j A[i,j] B[j,k]`, square `n`.
//! * `gemm_batched` `D[i,k,b] = sum_j A[i,j,b] B[j,k,b]`.
//! * `mps_env`      `ab,asc->bsc`: `A = chi x chi`, `B = chi x 2 x chi`,
//!   `D = chi x 2 x chi` (the label order `bsc`, as the equation writes it).
//! * `mps_site`     `bsc,bsd->cd`: `chi` in the sizes above.
//!
//! Every case is checked against a naive label oracle and the two arms against
//! each other before any timing; a mismatch aborts the run instead of
//! publishing a number.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use num_complex::{Complex32 as C32, Complex64 as C64};
use strided_view::{StridedView, StridedViewMut};
use tcontract::api::{
    AccumulationSource, CSpec, DType, Labels, LayoutSpec, Op, OperandSpec, Problem, Scalar,
};
use tcontract::{FaerLimit, Plan, PlanConfig, Reason};
use tcontract_exec::{Exec, Pool};
use tcontract_kernel::{Element, Real};

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

/// One contraction, described by extents and integer labels. Strides are
/// column-major over each operand's own extents.
#[derive(Clone, Debug)]
struct Case {
    class: &'static str,
    params: String,
    a: Vec<usize>,
    la: Vec<i64>,
    b: Vec<usize>,
    lb: Vec<i64>,
    d: Vec<usize>,
    ld: Vec<i64>,
}

/// ASCII label codes, so the integer labels in the CSV match the equations.
fn ids(s: &str) -> Vec<i64> {
    s.chars().map(|c| c as i64).collect()
}

/// The C mode: overwrite (`absent`) or in-place accumulation (`output`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CMode {
    Absent,
    Output,
}

impl CMode {
    fn name(self) -> &'static str {
        match self {
            CMode::Absent => "absent",
            CMode::Output => "output",
        }
    }

    /// The `CSpec` a problem is built with in this mode. `output` maps C to the
    /// old D itself with no element-wise op.
    fn c_spec(self) -> CSpec {
        match self {
            CMode::Absent => CSpec::Absent,
            CMode::Output => CSpec::Output(Op::Identity),
        }
    }

    const ALL: [CMode; 2] = [CMode::Absent, CMode::Output];
}

/// Which planner arms a run records. `default` and `packed` are always planned
/// (the packed `PlanStats` supplies the `batch,m,n,k,mnk` every row shares);
/// `faer_forced` — `PlanConfig { faer_limit: FaerLimit::NONE, ..default }`, the
/// rule before #63 — is the arm this sweep adds to reach `faer` above the
/// dtype's `FaerLimit`.
const ARMS: [&str; 3] = ["default", "packed", "faer_forced"];

fn cases() -> Vec<Case> {
    let mut v = Vec::new();
    for &n in &[32usize, 64, 128, 256] {
        v.push(Case {
            class: "gemm",
            params: format!("n={n}"),
            a: vec![n, n],
            la: ids("ij"),
            b: vec![n, n],
            lb: ids("jk"),
            d: vec![n, n],
            ld: ids("ik"),
        });
    }
    for &(n, batch) in &[(16usize, 16usize), (32, 16), (64, 16), (32, 64), (64, 64)] {
        v.push(Case {
            class: "gemm_batched",
            params: format!("n={n} batch={batch}"),
            a: vec![n, n, batch],
            la: ids("ijb"),
            b: vec![n, n, batch],
            lb: ids("jkb"),
            d: vec![n, n, batch],
            ld: ids("ikb"),
        });
    }
    for &chi in &[16usize, 24, 32, 48, 64, 96, 128] {
        // E[a,b] A[a,s,c] -> T[b,s,c]; D's extents follow `bsc`, so chi x 2 x chi.
        v.push(Case {
            class: "mps_env",
            params: format!("chi={chi}"),
            a: vec![chi, chi],
            la: ids("ab"),
            b: vec![chi, 2, chi],
            lb: ids("asc"),
            d: vec![chi, 2, chi],
            ld: ids("bsc"),
        });
    }
    for &chi in &[16usize, 24, 32, 48, 64, 96, 128] {
        // T[b,s,c] B[b,s,d] -> E'[c,d].
        v.push(Case {
            class: "mps_site",
            params: format!("chi={chi}"),
            a: vec![chi, 2, chi],
            la: ids("bsc"),
            b: vec![chi, 2, chi],
            lb: ids("bsd"),
            d: vec![chi, chi],
            ld: ids("cd"),
        });
    }
    v
}

// ---------------------------------------------------------------------------
// Shapes, data, oracle
// ---------------------------------------------------------------------------

fn col_major(dims: &[usize]) -> Vec<isize> {
    let mut s = Vec::with_capacity(dims.len());
    let mut acc = 1isize;
    for &d in dims {
        s.push(acc);
        acc *= d.max(1) as isize;
    }
    s
}

fn product(dims: &[usize]) -> usize {
    dims.iter().product()
}

/// A deterministic, dependency-free fill in `[-0.5, 0.5)` (plus a small
/// imaginary part for complex storage).
fn fill<S: Scalar>(len: usize) -> Vec<S> {
    (0..len)
        .map(|i| {
            let re = ((i * 37) % 101) as f64 / 101.0 - 0.5;
            let im = ((i * 53) % 97) as f64 / 97.0 - 0.5;
            <S as Element>::from_parts(Real::from_f64(re), Real::from_f64(im))
        })
        .collect()
}

/// `D = sum over the shared labels of A[l] B[l]`, enumerated directly from the
/// labels, as the CHECK oracle of `experiments/three-engine-contract`.
#[allow(clippy::too_many_arguments)]
fn oracle<S: Scalar>(
    a: &[S],
    da: &[usize],
    la: &[i64],
    b: &[S],
    db: &[usize],
    lb: &[i64],
    dd: &[usize],
    ld: &[i64],
) -> Vec<S> {
    let mut labels: Vec<i64> = la.iter().chain(lb).chain(ld).copied().collect();
    labels.sort_unstable();
    labels.dedup();
    let mut ext: HashMap<i64, usize> = HashMap::new();
    for (l, &e) in la.iter().zip(da) {
        ext.insert(*l, e);
    }
    for (l, &e) in lb.iter().zip(db) {
        ext.insert(*l, e);
    }
    for (l, &e) in ld.iter().zip(dd) {
        ext.insert(*l, e);
    }
    let pos = |l: i64| labels.iter().position(|&x| x == l).unwrap();
    let (ia, ib, id): (Vec<usize>, Vec<usize>, Vec<usize>) = (
        la.iter().map(|&l| pos(l)).collect(),
        lb.iter().map(|&l| pos(l)).collect(),
        ld.iter().map(|&l| pos(l)).collect(),
    );
    let extents: Vec<usize> = labels.iter().map(|l| ext[l]).collect();
    let (sa, sb, sd) = (col_major(da), col_major(db), col_major(dd));
    let mut out = vec![<S as Element>::zero(); product(dd)];
    let mut idx = vec![0usize; labels.len()];
    loop {
        let oa: usize = ia.iter().zip(&sa).map(|(&p, &s)| idx[p] * s as usize).sum();
        let ob: usize = ib.iter().zip(&sb).map(|(&p, &s)| idx[p] * s as usize).sum();
        let od: usize = id.iter().zip(&sd).map(|(&p, &s)| idx[p] * s as usize).sum();
        out[od] = Element::add(out[od], Element::mul(a[oa], b[ob]));
        let mut k = 0;
        loop {
            if k == idx.len() {
                return out;
            }
            idx[k] += 1;
            if idx[k] < extents[k] {
                break;
            }
            idx[k] = 0;
            k += 1;
        }
    }
}

fn rel_err<S: Scalar>(got: &[S], want: &[S]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want
        .iter()
        .map(|&z| Element::norm(z))
        .fold(f64::MIN_POSITIVE, f64::max);
    got.iter()
        .zip(want)
        .map(|(&g, &w)| Element::sub(g, w).norm())
        .fold(0.0f64, f64::max)
        / scale
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

/// Best of `reps` calls in nanoseconds, after `prime_ms` of untimed, time-based
/// priming. The same wall-clock warm-up is given to every arm, so an arm that
/// is much shorter is not biased by a fixed call count.
fn timed(reps: usize, prime_ms: u64, mut f: impl FnMut()) -> f64 {
    let until = Instant::now() + Duration::from_millis(prime_ms);
    while Instant::now() < until {
        f();
    }
    let mut best = f64::INFINITY;
    for _ in 0..reps.max(1) {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64() * 1e9);
    }
    best
}

/// Run one plan once, in `mode`. `absent` overwrites D; `output` accumulates
/// `alpha * A * B + beta * D` into D in place, which is what the corpus pays.
#[allow(clippy::too_many_arguments)]
fn apply<S: Scalar>(
    plan: &Plan<S>,
    mode: CMode,
    exec: &Exec<'_>,
    alpha: S,
    av: &StridedView<'_, S>,
    bv: &StridedView<'_, S>,
    beta: S,
    dv: &mut StridedViewMut<'_, S>,
) {
    match mode {
        CMode::Absent => plan.execute_into(exec, alpha, av, bv, dv).unwrap(),
        CMode::Output => plan
            .execute_into_accum(exec, alpha, av, bv, beta, AccumulationSource::Output, dv)
            .unwrap(),
    }
}

#[allow(clippy::too_many_arguments)]
fn time_plan<S: Scalar>(
    plan: &Plan<S>,
    mode: CMode,
    d: &mut [S],
    dims: &[usize],
    strides: &[isize],
    av: &StridedView<'_, S>,
    bv: &StridedView<'_, S>,
    exec: &Exec<'_>,
    reps: usize,
    prime_ms: u64,
) -> f64 {
    let mut dv = StridedViewMut::new(d, dims, strides, 0).unwrap();
    let alpha = <S as Element>::one();
    let beta = <S as Element>::one();
    timed(reps, prime_ms, || {
        apply(plan, mode, exec, alpha, av, bv, beta, &mut dv);
    })
}

fn reason_str(r: &Reason) -> String {
    match r {
        Reason::Forced => "forced".to_string(),
        Reason::AllBatch => "all_batch".to_string(),
        Reason::Fused => "fused".to_string(),
        Reason::AboveFaerLimit { volume, limit } => {
            format!("above_faer_limit(volume={volume} limit={limit})")
        }
        _ => "other".to_string(),
    }
}

// ---------------------------------------------------------------------------
// One case
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn one_case<S: Scalar>(
    case: &Case,
    mode: CMode,
    threads: usize,
    exec: &Exec<'_>,
    arms: &[&str],
    reps: usize,
    prime_ms: u64,
    timing: bool,
    rows: &mut Vec<String>,
    checks: &mut Vec<String>,
) {
    let dtype = <S as Scalar>::STORAGE;
    let (sa, sb, sd) = (col_major(&case.a), col_major(&case.b), col_major(&case.d));
    let a = fill::<S>(product(&case.a));
    let b = fill::<S>(product(&case.b));

    let l = |dims: &[usize], st: &[isize]| OperandSpec::new(LayoutSpec::new(dims, st, 0).unwrap());
    let problem = Problem::from_labels(
        dtype,
        l(&case.a, &sa),
        l(&case.b, &sb),
        mode.c_spec(),
        l(&case.d, &sd),
        &Labels::new(&case.la, &case.lb, &case.ld),
    )
    .unwrap();
    let plan_default = Plan::<S>::new(&problem, &PlanConfig::default()).unwrap();
    let plan_packed =
        Plan::<S>::new(&problem, &PlanConfig::packed()).expect("the forced-packed plan builds");
    let plan_faer = arms.contains(&"faer_forced").then(|| {
        Plan::<S>::new(
            &problem,
            &PlanConfig {
                faer_limit: FaerLimit::NONE,
                ..PlanConfig::default()
            },
        )
        .unwrap_or_else(|e| {
            // The faer route must answer for every problem `default` sent to
            // packed; a refusal is reported here, not routed around.
            panic!(
                "faer_forced refuses {} {} {}: {e}",
                dtype.name(),
                case.class,
                case.params
            )
        })
    });

    let stats = plan_packed
        .report()
        .packed
        .as_ref()
        .expect("the forced-packed plan carries a packed report")
        .stats
        .clone();
    let mnk = (stats.m as u64) * (stats.n as u64) * (stats.k as u64);
    let alg_default = plan_default.report().algorithm.name();
    let reason_default = reason_str(&plan_default.report().reason);
    let alg_packed = plan_packed.report().algorithm.name();
    let alg_faer = plan_faer
        .as_ref()
        .map(|p| (p.report().algorithm.name(), reason_str(&p.report().reason)));

    let av = StridedView::new(&a, &case.a, &sa, 0).unwrap();
    let bv = StridedView::new(&b, &case.b, &sb, 0).unwrap();
    let alpha = <S as Element>::one();
    let beta = <S as Element>::one();

    // The old D: zero for the overwrite form (exactly the recorded absent
    // rows), a non-zero fill for the accumulating form.
    let start: Vec<S> = match mode {
        CMode::Absent => vec![<S as Element>::zero(); product(&case.d)],
        CMode::Output => fill::<S>(product(&case.d)),
    };

    // Correctness before timing: naive label oracle, then each arm, then the
    // arms against each other. `absent` wants the bare product; `output` wants
    // `alpha * sum + beta * D_old`.
    let prod = oracle::<S>(
        &a, &case.a, &case.la, &b, &case.b, &case.lb, &case.d, &case.ld,
    );
    let want: Vec<S> = match mode {
        CMode::Absent => prod,
        CMode::Output => prod
            .iter()
            .zip(&start)
            .map(|(&p, &s)| Element::add(Element::mul(alpha, p), Element::mul(beta, s)))
            .collect(),
    };
    let mut d = start.clone();
    {
        let mut dv = StridedViewMut::new(&mut d, &case.d, &sd, 0).unwrap();
        apply(&plan_default, mode, exec, alpha, &av, &bv, beta, &mut dv);
    }
    let got_default = d.clone();
    let rel_default = rel_err(&got_default, &want);
    d.copy_from_slice(&start);
    {
        let mut dv = StridedViewMut::new(&mut d, &case.d, &sd, 0).unwrap();
        apply(&plan_packed, mode, exec, alpha, &av, &bv, beta, &mut dv);
    }
    let got_packed = d.clone();
    let rel_packed = rel_err(&got_packed, &want);
    let rel_arms = rel_err(&got_packed, &got_default);
    let rel_faer = plan_faer.as_ref().map(|plan| {
        d.copy_from_slice(&start);
        {
            let mut dv = StridedViewMut::new(&mut d, &case.d, &sd, 0).unwrap();
            apply(plan, mode, exec, alpha, &av, &bv, beta, &mut dv);
        }
        rel_err(&d, &want)
    });

    let tol = if dtype == DType::F64 || dtype == DType::C64 {
        1e-10
    } else {
        1e-5
    };
    let faer_check = match (&alg_faer, rel_faer) {
        (Some((alg, reason)), Some(rel)) => {
            format!(" faer_forced={alg} route_faer={reason} rel_faer={rel:.3e}")
        }
        _ => String::new(),
    };
    checks.push(format!(
        "CHECK {} {} {} c_mode={} threads={threads} default={alg_default} packed={alg_packed}\
         {faer_check} rel_default={rel_default:.3e} rel_packed={rel_packed:.3e} \
         rel_arms={rel_arms:.3e} tol={tol:.0e}",
        dtype.name(),
        case.class,
        case.params,
        mode.name(),
    ));
    assert!(
        rel_default < tol && rel_packed < tol && rel_arms < tol && rel_faer.is_none_or(|r| r < tol),
        "{} {} {} c_mode={} threads={threads} failed the oracle: default={rel_default:.3e} \
         packed={rel_packed:.3e} arms={rel_arms:.3e} faer_forced={rel_faer:?} (tol {tol:.0e})",
        dtype.name(),
        case.class,
        case.params,
        mode.name(),
    );

    if !timing {
        return;
    }
    let ns_default = time_plan(
        &plan_default,
        mode,
        &mut d,
        &case.d,
        &sd,
        &av,
        &bv,
        exec,
        reps,
        prime_ms,
    );
    let ns_packed = time_plan(
        &plan_packed,
        mode,
        &mut d,
        &case.d,
        &sd,
        &av,
        &bv,
        exec,
        reps,
        prime_ms,
    );
    let ns_faer = plan_faer.as_ref().map(|plan| {
        time_plan(
            plan, mode, &mut d, &case.d, &sd, &av, &bv, exec, reps, prime_ms,
        )
    });

    let base = format!(
        "{},{},{},{},{},{},{},{},{},{threads}",
        case.class,
        dtype.name(),
        case.params,
        mode.name(),
        stats.batch,
        stats.m,
        stats.n,
        stats.k,
        mnk,
    );
    rows.push(format!(
        "{base},default,{alg_default},{reason_default},{ns_default:.1}"
    ));
    rows.push(format!("{base},packed,{alg_packed},forced,{ns_packed:.1}"));
    if let (Some((alg, reason)), Some(ns)) = (alg_faer, ns_faer) {
        rows.push(format!("{base},faer_forced,{alg},{reason},{ns:.1}"));
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch<S: Scalar>(
    case: &Case,
    mode: CMode,
    threads: usize,
    exec: &Exec<'_>,
    arms: &[&str],
    reps: usize,
    prime_ms: u64,
    timing: bool,
    rows: &mut Vec<String>,
    checks: &mut Vec<String>,
) {
    one_case::<S>(
        case, mode, threads, exec, arms, reps, prime_ms, timing, rows, checks,
    );
}

// ---------------------------------------------------------------------------
// Host state
// ---------------------------------------------------------------------------

fn cpu_list_len(list: &str) -> usize {
    list.split(',')
        .filter(|p| !p.is_empty())
        .map(|p| match p.split_once('-') {
            Some((lo, hi)) => hi.parse::<usize>().unwrap() - lo.parse::<usize>().unwrap() + 1,
            None => 1,
        })
        .sum()
}

/// The CPUs this process may run on, from `/proc/self/status`; `None` off
/// Linux or when the line is absent.
fn allowed_cpus() -> Option<usize> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status
        .lines()
        .find(|l| l.starts_with("Cpus_allowed_list:"))?;
    Some(cpu_list_len(line.split_whitespace().nth(1)?))
}

/// Refuse a requested width the host environment would override.
fn check_env(threads: usize) {
    for var in [
        "RAYON_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ] {
        if let Ok(v) = std::env::var(var) {
            let ok = v.parse::<usize>().ok() == Some(threads);
            assert!(ok, "{var}={v} conflicts with --threads {threads}");
        }
    }
}

const HEADER: &str = "class,dtype,params,c_mode,batch,m,n,k,mnk,threads,arm,algorithm,reason,ns";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut threads = 1usize;
    let mut reps = 5usize;
    let mut prime_ms = 500u64;
    let mut csv: Option<String> = None;
    let mut check_only = false;
    // `default` and `packed` are always recorded; `faer_forced` is opt-in so the
    // earlier two-arm run stays reproducible.
    let mut arms: Vec<&str> = ARMS.to_vec();
    let mut i = 1;
    while i < args.len() {
        let next = || args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--arms" => {
                let spec = next();
                arms.clear();
                for a in spec.split(',').filter(|a| !a.is_empty()) {
                    assert!(
                        ARMS.contains(&a),
                        "--arms: unknown arm {a} (known: {})",
                        ARMS.join(",")
                    );
                    arms.push(ARMS.iter().find(|&&x| x == a).unwrap());
                }
                assert!(arms.contains(&"packed"), "--arms must include packed");
                i += 1;
            }
            "--threads" => {
                threads = next().parse().unwrap_or(1);
                i += 1;
            }
            "--reps" => {
                reps = next().parse().unwrap_or(5);
                i += 1;
            }
            "--prime-ms" => {
                prime_ms = next().parse().unwrap_or(500);
                i += 1;
            }
            "--csv" => {
                csv = Some(next());
                i += 1;
            }
            "--check-only" => check_only = true,
            other => eprintln!("warning: ignoring {other}"),
        }
        i += 1;
    }
    assert!(threads >= 1, "--threads must be positive");
    check_env(threads);
    match allowed_cpus() {
        Some(n) => assert_eq!(
            n, threads,
            "the process is allowed on {n} CPUs but --threads {threads}"
        ),
        None => eprintln!("warning: cannot read Cpus_allowed_list; pinning not verified"),
    }

    let mut checks = vec![format!(
        "# faer-limit-sweep {} threads={threads} reps={reps} prime_ms={prime_ms} \
         arms={} (tcontract execution only, prebuilt plans, preallocated output)",
        env!("CARGO_PKG_VERSION"),
        arms.join(","),
    )];
    let mut rows = vec![HEADER.to_string()];
    let cases = cases();

    let timing = !check_only;
    if threads == 1 {
        let exec = Exec::serial();
        run_all(
            &cases,
            threads,
            &exec,
            &arms,
            reps,
            prime_ms,
            timing,
            &mut rows,
            &mut checks,
        );
    } else {
        let tp = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("the host pool builds");
        assert_eq!(tp.current_num_threads(), threads, "pool width");
        let pool = Pool::borrow(&tp);
        let exec = Exec::rayon(&pool);
        assert_eq!(exec.budget(), threads, "exec budget");
        run_all(
            &cases,
            threads,
            &exec,
            &arms,
            reps,
            prime_ms,
            timing,
            &mut rows,
            &mut checks,
        );
    }

    for line in &checks {
        println!("{line}");
    }
    for row in &rows {
        println!("{row}");
    }
    if let Some(path) = csv {
        use std::io::Write;
        let fresh = std::fs::metadata(&path)
            .map(|m| m.len() == 0)
            .unwrap_or(true);
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .expect("the CSV path is writable");
        if fresh {
            writeln!(f, "{HEADER}").unwrap();
        }
        for row in rows.iter().skip(1) {
            writeln!(f, "{row}").unwrap();
        }
        eprintln!("appended {} rows to {path}", rows.len() - 1);
    }
}

#[allow(clippy::too_many_arguments)]
fn run_all(
    cases: &[Case],
    threads: usize,
    exec: &Exec<'_>,
    arms: &[&str],
    reps: usize,
    prime_ms: u64,
    timing: bool,
    rows: &mut Vec<String>,
    checks: &mut Vec<String>,
) {
    for case in cases {
        for mode in CMode::ALL {
            dispatch::<f32>(
                case, mode, threads, exec, arms, reps, prime_ms, timing, rows, checks,
            );
            dispatch::<f64>(
                case, mode, threads, exec, arms, reps, prime_ms, timing, rows, checks,
            );
            dispatch::<C32>(
                case, mode, threads, exec, arms, reps, prime_ms, timing, rows, checks,
            );
            dispatch::<C64>(
                case, mode, threads, exec, arms, reps, prime_ms, timing, rows, checks,
            );
        }
    }
}
