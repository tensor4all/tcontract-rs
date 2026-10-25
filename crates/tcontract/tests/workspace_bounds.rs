//! The bounded `Workspace` contract: component-wise capacity, a single
//! output-route decision for sizing and dispatch, and a requirement that
//! upper-bounds what the driver allocates (#121 review).
//!
//! Its own binary with one test so nothing else pollutes a retained-byte
//! observation.
use tcontract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
use tcontract::{Error, OutputContract, Plan, PlanConfig};
use tcontract_exec::{Exec, Workspace, WorkspaceReq};
use tcontract_kernel::KernelChoice;

fn spec(d: &[usize], s: &[isize]) -> OperandSpec {
    OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap())
}

fn col_major(dims: &[usize]) -> Vec<isize> {
    let mut strides = Vec::with_capacity(dims.len());
    let mut stride = 1isize;
    for &dim in dims {
        strides.push(stride);
        stride *= dim as isize;
    }
    strides
}

/// `D[i,j] = sum_k A[i,k] B[k,j]`, column-major, no C.
fn matmul(m: usize, n: usize, k: usize) -> Problem {
    Problem::from_labels(
        DType::F64,
        spec(&[m, k], &col_major(&[m, k])),
        spec(&[k, n], &col_major(&[k, n])),
        CSpec::Absent,
        spec(&[m, n], &col_major(&[m, n])),
        &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    )
    .unwrap()
}

/// The same with an in-place output C (so `beta != 0` is legal).
fn matmul_output_c(m: usize, n: usize, k: usize) -> Problem {
    Problem::from_labels(
        DType::F64,
        spec(&[m, k], &col_major(&[m, k])),
        spec(&[k, n], &col_major(&[k, n])),
        CSpec::Output(tcontract::api::Op::Identity),
        spec(&[m, n], &col_major(&[m, n])),
        &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    )
    .unwrap()
}

/// A plan, its operands and a workspace grown to the plan's own bound.
struct Fixture {
    plan: Plan<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    d: Vec<f64>,
    ws: Workspace,
}

impl Fixture {
    fn new(problem: &Problem, dims: [usize; 3], config: &PlanConfig) -> Self {
        let [m, n, k] = dims;
        let plan = Plan::<f64>::new(problem, config).unwrap();
        let exec = Exec::serial();
        let ws = Workspace::with_req(&plan.workspace_req(&exec, OutputContract::Initialized));
        let data = |len: usize| (0..len).map(|i| (i as f64 * 0.13) % 3.0 - 1.0).collect();
        Self {
            plan,
            a: data(m * k),
            b: data(k * n),
            c: data(m * n),
            d: vec![0.0; m * n],
            ws,
        }
    }

    /// One raw execute with a nonzero beta, returning the error if any.
    fn run(&mut self, beta: f64) -> tcontract::Result<()> {
        let mut exec = Exec::serial();
        // SAFETY: the pointers cover the plan's validated ranges; D is exclusive.
        unsafe {
            self.plan.execute_raw_in(
                &mut self.ws,
                &mut exec,
                1.0,
                self.a.as_ptr(),
                self.b.as_ptr(),
                beta,
                self.c.as_ptr(),
                self.d.as_mut_ptr(),
                OutputContract::Initialized,
            )
        }
    }
}

/// A total-only bound cannot hide a missing component: a workspace whose A
/// covers the whole requirement but whose B is zero is refused.
#[test]
fn a_total_only_bound_cannot_hide_a_missing_panel() {
    let problem = matmul(8, 8, 8);
    let plan = Plan::<f64>::new(&problem, &PlanConfig::packed()).unwrap();
    let mut exec = Exec::serial();
    let req = plan.workspace_req(&exec, OutputContract::Initialized);
    assert!(req.b_bytes > 0, "the packed route needs a B panel");
    let mut ws = Workspace::with_req(&WorkspaceReq {
        a_bytes: req.bytes(),
        ..Default::default()
    });
    let a = vec![0.0f64; 64];
    let b = vec![0.0f64; 64];
    let mut d = vec![0.0f64; 64];
    // SAFETY: the pointers cover the plan's ranges; the call must refuse first.
    let err = unsafe {
        plan.execute_raw_in(
            &mut ws,
            &mut exec,
            1.0,
            a.as_ptr(),
            b.as_ptr(),
            0.0,
            core::ptr::null(),
            d.as_mut_ptr(),
            OutputContract::Initialized,
        )
    }
    .unwrap_err();
    assert!(matches!(err, Error::Workspace { .. }), "{err}");
}

/// An initialized Faer plan with `fresh_output` sizes zero and dispatches the
/// initialized route, so the workspace never grows for it (#121 review).
#[test]
fn initialized_faer_with_fresh_prepared_does_not_grow() {
    let config = PlanConfig {
        fresh_output: true,
        ..PlanConfig::default()
    };
    let problem = matmul_output_c(8, 8, 8);
    let plan = Plan::<f64>::new(&problem, &config).unwrap();
    assert_eq!(plan.report().algorithm, tcontract::Algorithm::Faer);
    let mut exec = Exec::serial();
    assert_eq!(
        plan.workspace_req(&exec, OutputContract::Initialized)
            .bytes(),
        0,
        "an initialized Faer route needs no workspace"
    );

    let [m, n, k] = [8usize, 8, 8];
    let a: Vec<f64> = (0..m * k).map(|i| i as f64 * 0.1).collect();
    let b: Vec<f64> = (0..k * n).map(|i| i as f64 * 0.1).collect();
    let mut d = vec![0.0f64; m * n];
    let mut ws = Workspace::with_req(&WorkspaceReq::default());
    let before = ws.retained_bytes();
    // SAFETY: the pointers cover the plan's ranges.
    unsafe {
        plan.execute_raw_in(
            &mut ws,
            &mut exec,
            1.0,
            a.as_ptr(),
            b.as_ptr(),
            0.0,
            core::ptr::null(),
            d.as_mut_ptr(),
            OutputContract::Initialized,
        )
        .unwrap();
    }
    // The first execute must already stay on the initialized route; sizing said
    // zero, so the workspace must not grow.
    assert_eq!(
        ws.retained_bytes(),
        before,
        "the initialized route grew the workspace"
    );
    // SAFETY: as above.
    unsafe {
        plan.execute_raw_in(
            &mut ws,
            &mut exec,
            1.0,
            a.as_ptr(),
            b.as_ptr(),
            0.0,
            core::ptr::null(),
            d.as_mut_ptr(),
            OutputContract::Initialized,
        )
        .unwrap();
    }
}

/// `workspace_req` upper-bounds the driver across the reviewed shapes: a
/// workspace sized to it does not grow during execute.
#[test]
fn the_bound_covers_the_reviewed_shapes() {
    // A transposed-looking 64x1 output with a 4x4 family and beta != 0. The
    // orientation is forced so the driver swaps the operands, which is the case
    // the scatter sizing must follow.
    let squared_4x4 = PlanConfig {
        kernel: KernelChoice::Id("ref.f64.real-scalar.4x4".into()),
        orientation: tcontract::Orient::Force(true),
        ..PlanConfig::default()
    };
    let mut swapped = Fixture::new(&matmul_output_c(64, 1, 4), [64, 1, 4], &squared_4x4);
    swapped.run(0.5).unwrap();
    let before = swapped.ws.retained_bytes();
    swapped.run(0.5).unwrap();
    assert_eq!(
        swapped.ws.retained_bytes(),
        before,
        "the 64x1 transposed shape grew the workspace"
    );

    // A small explicit MC blocking: the blocked path can replace MC with rounded
    // block rows, so the A bound must cover the larger of the two.
    let small_mc = PlanConfig {
        blocking: tcontract_kernel::BlockingOverride {
            mc: Some(8),
            ..Default::default()
        },
        ..PlanConfig::default()
    };
    let mut blocked = Fixture::new(&matmul(64, 64, 8), [64, 64, 8], &small_mc);
    blocked.run(0.0).unwrap();
    let before = blocked.ws.retained_bytes();
    blocked.run(0.0).unwrap();
    assert_eq!(
        blocked.ws.retained_bytes(),
        before,
        "the small-MC shape grew the workspace"
    );

    // A batched packed shape with beta != 0.
    let spec = |d: &[usize], s: &[isize]| OperandSpec::new(LayoutSpec::new(d, s, 0).unwrap());
    let (d, s) = ([8usize, 8, 8], col_major(&[8, 8, 8]));
    let batched = Problem::from_labels(
        DType::F64,
        spec(&d, &s),
        spec(&d, &s),
        CSpec::Output(tcontract::api::Op::Identity),
        spec(&d, &s),
        &Labels::new(&[0, 2, 3], &[2, 1, 3], &[0, 1, 3]),
    )
    .unwrap();
    let plan = Plan::<f64>::new(&batched, &PlanConfig::packed()).unwrap();
    let mut exec = Exec::serial();
    let mut ws = Workspace::with_req(&plan.workspace_req(&exec, OutputContract::Initialized));
    let (len, dims) = (512usize, [8usize, 8, 8]);
    let a: Vec<f64> = (0..len).map(|i| i as f64 * 0.1).collect();
    let b: Vec<f64> = (0..len).map(|i| i as f64 * 0.1).collect();
    let c: Vec<f64> = (0..len).map(|i| i as f64 * 0.1).collect();
    let mut d = vec![0.0f64; len];
    let mut run = |ws: &mut Workspace, d: &mut [f64]| {
        // SAFETY: the pointers cover the plan's validated ranges; D is exclusive.
        unsafe {
            plan.execute_raw_in(
                ws,
                &mut exec,
                1.0,
                a.as_ptr(),
                b.as_ptr(),
                0.5,
                c.as_ptr(),
                d.as_mut_ptr(),
                OutputContract::Initialized,
            )
            .unwrap();
        }
    };
    run(&mut ws, &mut d);
    let before = ws.retained_bytes();
    run(&mut ws, &mut d);
    assert_eq!(
        ws.retained_bytes(),
        before,
        "the batched shape grew the workspace"
    );
    let _ = dims;
}

/// A workspace not warmed for every worker of the executor is refused before any
/// write, rather than letting a worker allocate its own slot mid-execute.
#[test]
fn a_workspace_not_prepared_for_the_executor_is_refused() {
    use tcontract_exec::Pool;

    // A 128^3 item narrows to a two-worker team at budget two, so two slots are
    // needed and `with_req` warms only one.
    let m = 128;
    let plan = Plan::<f64>::new(&matmul(m, m, m), &PlanConfig::packed()).unwrap();
    let tp = threadteam::Pool::builder().threads(2).build().unwrap();
    let pool = Pool::borrow(&tp);
    let mut exec = Exec::new(&pool, 2).unwrap();
    let mut ws = Workspace::with_req(&plan.workspace_req(&exec, OutputContract::Initialized));
    assert_eq!(ws.slots(), 1);

    let a = vec![1.0f64; m * m];
    let b = vec![1.0f64; m * m];
    let mut d = vec![0.0f64; m * m];
    // SAFETY: the pointers cover the plan's ranges; the call must refuse first.
    let err = unsafe {
        plan.execute_raw_in(
            &mut ws,
            &mut exec,
            1.0,
            a.as_ptr(),
            b.as_ptr(),
            0.0,
            core::ptr::null(),
            d.as_mut_ptr(),
            OutputContract::Initialized,
        )
    }
    .unwrap_err();
    assert!(matches!(err, Error::Workspace { .. }), "{err}");

    // After warming both slots it runs.
    ws.prepare_for(&mut exec);
    assert_eq!(ws.slots(), 2);
    // SAFETY: as above.
    unsafe {
        plan.execute_raw_in(
            &mut ws,
            &mut exec,
            1.0,
            a.as_ptr(),
            b.as_ptr(),
            0.0,
            core::ptr::null(),
            d.as_mut_ptr(),
            OutputContract::Initialized,
        )
        .unwrap();
    }
}
