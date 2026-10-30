//! The driver's live-extent dispatch, pinned by a counting edge entry.
//!
//! `tests/packed_edge_tiles.rs` proves that partial micro-tiles are *correct*,
//! not that the family's live-extent entry ran: a family that kept its full-tile
//! entry would produce the same values (that file's header says so). This test
//! closes the gap through the custom-catalog seam: `tcontract-testkit`'s
//! `custom.f64.edge.4x4` is an ordinary portable 4x4 scalar kernel whose
//! live-extent entry counts its calls, so deleting the driver's `fam.edge` branch
//! (`crates/tcontract/src/driver/tile.rs`) or narrowing its
//! `mrem < mr || nrem < nr` guard fails here instead of passing everywhere.
//!
//! The case is a plain f64 matmul, `D[m, n] = sum_k A[m, k] * B[k, n]`, at one K
//! slab and one block, so the count is exactly the number of partial micro-tiles:
//!
//! * `m = n = 5` against a 4x4 tile gives three partial tiles - `(1, 4)`,
//!   `(4, 1)` and `(1, 1)` - and one whole one, so the entry is called three
//!   times and last for the `(1, 1)` tile;
//! * `m = n = 8` gives four whole tiles and no call at all.
//!
//! Both runs also compare against a local dot-product reference, so a dispatch
//! that reached the entry with the wrong live extent (or left the declared tile
//! uninitialized) fails on the values rather than only on the count.
use std::sync::atomic::Ordering;

use strided_view::{StridedView, StridedViewMut};
use tcontract::api::{DType, DotGeneral, LayoutSpec, OperandSpec, Problem};
use tcontract::{Algorithm, Plan, PlanConfig};
use tcontract_exec::Exec;
use tcontract_kernel::KernelCatalog;
use tcontract_testkit::custom_kernels as own;
use tcontract_testkit::fixtures::col_major;

const ID: &str = "custom.f64.edge.4x4";

fn catalog() -> KernelCatalog<f64> {
    // SAFETY: the testkit's families are immutable, `'static` descriptors that
    // meet the family contract (checked by their own tests).
    unsafe { KernelCatalog::<f64>::from_static_families(own::f64_edge_families()) }.unwrap()
}

fn spec(dims: &[usize]) -> OperandSpec {
    OperandSpec::new(LayoutSpec::new(dims, &col_major(dims), 0).unwrap())
}

/// `D[m, n] = sum_k A[m, k] * B[k, n]`, column-major, no `C` term.
fn problem(m: usize, n: usize, k: usize) -> Problem {
    Problem::from_dot_general(
        DType::F64,
        spec(&[m, k]),
        spec(&[k, n]),
        spec(&[m, n]),
        &DotGeneral::new(&[1], &[0], &[], &[]),
    )
    .unwrap()
}

/// A plan that selects the counting family from its private catalog.
fn plan(cat: &KernelCatalog<f64>, m: usize, n: usize, k: usize) -> Plan<f64> {
    let p = Plan::<f64>::new_with_selector(
        &problem(m, n, k),
        &PlanConfig::default(),
        cat,
        &mut |_, _| Ok(cat.get(ID).unwrap()),
    )
    .unwrap();
    assert_eq!(p.report().algorithm, Algorithm::Packed);
    assert_eq!(
        p.report().packed.as_ref().unwrap().family_id,
        ID,
        "the plan fell back to another family, which would make the count vacuous"
    );
    p
}

fn data(len: usize, seed: usize) -> Vec<f64> {
    (0..len)
        .map(|i| ((i * 7 + seed * 3) % 11) as f64 - 5.0)
        .collect()
}

/// Run one contraction and check every output element against the reference.
fn run(plan: &Plan<f64>, m: usize, n: usize, k: usize) {
    let (a, b) = (data(m * k, 1), data(k * n, 2));
    // Poisoned: `beta = 0` overwrites, and an unwritten element stays NaN.
    let mut d = vec![f64::NAN; m * n];
    let av = StridedView::new(&a, &[m, k], &col_major(&[m, k]), 0).unwrap();
    let bv = StridedView::new(&b, &[k, n], &col_major(&[k, n]), 0).unwrap();
    let mut dv = StridedViewMut::new(&mut d, &[m, n], &col_major(&[m, n]), 0).unwrap();
    plan.execute_into(&mut Exec::serial(), 1.0, &av, &bv, &mut dv)
        .unwrap();
    let mut worst = 0.0f64;
    for j in 0..n {
        for i in 0..m {
            let mut want = 0.0f64;
            for p in 0..k {
                want += a[i + m * p] * b[p + k * j];
            }
            let got = d[i + m * j];
            assert!(
                got.is_finite(),
                "{m}x{n}x{k}: output at ({i}, {j}) was never written"
            );
            worst = worst.max((got - want).abs());
        }
    }
    assert_eq!(worst, 0.0, "{m}x{n}x{k}: max |D - reference| = {worst}");
}

#[test]
fn a_partial_tile_calls_the_edge_entry_and_a_whole_one_does_not() {
    let cat = catalog();
    let k = 8;

    // 5x5 against a 4x4 tile: `ceil(5/4) * ceil(5/4) = 4` micro-tiles, three of
    // them partial (`mrem < 4 || nrem < 4`), in the driver's own order
    // (`(4, 4)`, `(1, 4)`, `(4, 1)`, `(1, 1)` over the column then row tile).
    let (m, n) = (5, 5);
    own::EDGE_CALLS.swap(0, Ordering::Relaxed);
    own::EDGE_LAST.swap(0, Ordering::Relaxed);
    let p = plan(&cat, m, n, k);
    run(&p, m, n, k);
    assert_eq!(
        own::EDGE_CALLS.load(Ordering::Relaxed),
        3,
        "the three partial micro-tiles of a 5x5 block did not all take the \
         live-extent entry"
    );
    assert_eq!(
        own::EDGE_LAST.load(Ordering::Relaxed),
        (1 << 8) | 1,
        "the last partial tile is the (1, 1) corner"
    );

    // 8x8 against a 4x4 tile: four whole micro-tiles, so the entry is never
    // called.
    let (m, n) = (8, 8);
    own::EDGE_CALLS.swap(0, Ordering::Relaxed);
    let p = plan(&cat, m, n, k);
    run(&p, m, n, k);
    assert_eq!(
        own::EDGE_CALLS.load(Ordering::Relaxed),
        0,
        "a whole micro-tile took the live-extent entry instead of the full-tile one"
    );
}
