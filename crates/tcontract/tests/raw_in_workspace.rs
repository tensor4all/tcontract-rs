//! `Plan::execute_raw_in`: a `Workspace` sized to `Plan::workspace_req` makes a
//! serial packed execute allocation-free, and a short one is refused before any
//! write (#121).
//!
//! Its own binary because the counter is global.
use std::alloc::{GlobalAlloc, Layout as AllocLayout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use tcontract::{Error, OutputContract, Plan, PlanConfig};
use tcontract_exec::{Exec, Workspace, WorkspaceReq};

mod common;
use common::plans::matmul_problem;

static COUNT: AtomicUsize = AtomicUsize::new(0);

struct Counting;

// SAFETY: every method forwards to the system allocator unchanged; the counter
// is an atomic with no aliasing.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: AllocLayout) -> *mut u8 {
        COUNT.fetch_add(1, Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: AllocLayout) {
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

const SHAPE: (usize, usize, usize) = (64, 48, 40);

#[test]
fn a_sized_workspace_is_allocation_free_and_a_short_one_is_refused() {
    let (m, n, k) = SHAPE;
    // Forced packed so the plan owns a workspace requirement at all.
    let plan = Plan::<f64>::new(&matmul_problem(m, n, k), &PlanConfig::packed()).unwrap();
    let exec = Exec::serial();
    let req = plan.workspace_req(&exec, OutputContract::Initialized);
    assert!(req.bytes() > 0, "the packed route needs a workspace");

    let a: Vec<f64> = (0..m * k).map(|i| (i as f64 * 0.37) % 3.0 - 1.0).collect();
    let b: Vec<f64> = (0..k * n).map(|i| (i as f64 * 0.11) % 2.0 - 0.5).collect();
    let mut d = vec![0.0f64; m * n];

    // A workspace sized to the bound, warmed by one execute.
    let mut ws = Workspace::with_req(&req);
    // SAFETY: the pointers cover the plan's validated ranges and `d` is
    // exclusive; no separate C (CSpec::Absent), so `c` may be null.
    unsafe {
        plan.execute_raw_in(
            &mut ws,
            &exec,
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

    let before = COUNT.load(Relaxed);
    // SAFETY: as above.
    unsafe {
        plan.execute_raw_in(
            &mut ws,
            &exec,
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
    assert_eq!(
        COUNT.load(Relaxed) - before,
        0,
        "a steady execute within the bound allocated"
    );

    // A workspace smaller than the bound is refused with `Error::Workspace`
    // before any write; the output sentinel is untouched.
    let mut sentinel = vec![7.0f64; m * n];
    let mut short = Workspace::with_req(&WorkspaceReq {
        a_bytes: 8,
        ..Default::default()
    });
    // SAFETY: as above; the write must not happen.
    let err = unsafe {
        plan.execute_raw_in(
            &mut short,
            &exec,
            1.0,
            a.as_ptr(),
            b.as_ptr(),
            0.0,
            core::ptr::null(),
            sentinel.as_mut_ptr(),
            OutputContract::Initialized,
        )
    }
    .unwrap_err();
    assert!(
        matches!(err, Error::Workspace { needed, have } if needed > have),
        "{err}"
    );
    assert!(
        sentinel.iter().all(|&x| x == 7.0),
        "a refused workspace wrote to D"
    );
}
