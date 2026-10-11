//! Correctness self-check for the OpenBLAS kernel import.
//!
//! Three layers, each failing the process with a nonzero exit on mismatch:
//!
//! 1. Tile product: packed `16xkc` A and `2xkc` B against a naive reference
//!    for `kc` in `{0, 1, 2, 3, 7, 16, 33}` (K0/K1/tails/regular).
//! 2. Exact known values: small-integer panels where the result is computed
//!    in exact `f64` arithmetic.
//! 3. End-to-end through the tcontract packed driver: default built-in tcontract
//!    family vs the OpenBLAS family (selected by id), both compared to a
//!    naive contraction, for a column-major matrix and a strided-A case.

use openblas_kernel::{catalog, tile_product, MR, NR};
use tcontract::api::{DType, DotGeneral, LayoutSpec, OperandSpec, Problem};
use tcontract::{Chooser, Plan, PlanConfig};
use tcontract_exec::Exec;
use tcontract_kernel::SelectError;

fn check(what: &str, ok: bool) {
    if ok {
        println!("ok    {what}");
    } else {
        println!("FAIL  {what}");
        std::process::exit(1);
    }
}

fn approx(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(&x, &y)| (x - y).abs() <= tol * (1.0 + x.abs() + y.abs()))
}

/// Deterministic `(-1, 1)` values, no external RNG dependency.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x as f64 / u64::MAX as f64) * 2.0 - 1.0
    }
}

fn ref_tile(kc: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0f64; MR * NR];
    for p in 0..kc {
        for i in 0..MR {
            let av = a[i + MR * p];
            for j in 0..NR {
                c[j * MR + i] += av * b[j + NR * p];
            }
        }
    }
    c
}

fn tile_case(kc: usize, rng: &mut Rng) {
    let mut a = vec![0.0f64; MR * kc];
    let mut b = vec![0.0f64; NR * kc];
    for x in &mut a {
        *x = rng.next();
    }
    for x in &mut b {
        *x = rng.next();
    }
    let want = ref_tile(kc, &a, &b);
    let mut got = vec![f64::NAN; MR * NR];
    // SAFETY: panels and tile have the exact extents the kernel declares.
    unsafe { tile_product(kc, a.as_ptr(), b.as_ptr(), got.as_mut_ptr()) };
    check(&format!("tile kc={kc}"), approx(&got, &want, 1e-12));
}

fn wide_case(kc: usize, rng: &mut Rng) {
    let a: Vec<f64> = (0..MR * kc).map(|_| rng.next()).collect();
    let b: Vec<f64> = (0..12 * kc).map(|_| rng.next()).collect();
    let mut want = vec![0.0; MR * 12];
    for p in 0..kc {
        for j in 0..12 {
            for i in 0..MR {
                want[j * MR + i] += a[p * MR + i] * b[p * 12 + j];
            }
        }
    }
    let mut got = vec![f64::NAN; MR * 12];
    // SAFETY: exact sized panels and exclusive output tile, including kc=0.
    unsafe { openblas_kernel::tile_product_wide(kc, a.as_ptr(), b.as_ptr(), got.as_mut_ptr()) };
    check(&format!("wide tile kc={kc}"), approx(&got, &want, 1e-12));
}

fn tile_exact_known() {
    // Small-integer panels: A[i,p] = i + 1 + 2p, B[j,p] = j + 3 - p, kc = 2.
    let kc = 2usize;
    let mut a = vec![0.0f64; MR * kc];
    let mut b = vec![0.0f64; NR * kc];
    for p in 0..kc {
        for i in 0..MR {
            a[i + MR * p] = (i + 1 + 2 * p) as f64;
        }
        for j in 0..NR {
            b[j + NR * p] = (j + 3 - p) as f64;
        }
    }
    let want = ref_tile(kc, &a, &b);
    let mut got = vec![f64::NAN; MR * NR];
    unsafe { tile_product(kc, a.as_ptr(), b.as_ptr(), got.as_mut_ptr()) };
    // Exact: all products and sums are small integers.
    check("tile exact known values", got == want);
}

fn layout(dims: &[usize], strides: &[isize]) -> OperandSpec {
    OperandSpec::new(LayoutSpec::new(dims, strides, 0).unwrap())
}

/// `D[i,j] = sum_k A[i,k] * B[k,j]` with the given A/B strides, run through
/// the default packed driver and the OpenBLAS family, against a naive result.
fn end_to_end(m: usize, n: usize, k: usize, a_s: [isize; 2], b_s: [isize; 2], rng: &mut Rng) {
    let dot = DotGeneral::new(&[1], &[0], &[], &[]);
    let problem = Problem::from_dot_general(
        DType::F64,
        layout(&[m, k], &a_s),
        layout(&[k, n], &b_s),
        layout(&[m, n], &[1, m as isize]),
        &dot,
    )
    .unwrap();

    let mut a = vec![0.0f64; m * k];
    let mut b = vec![0.0f64; k * n];
    for x in &mut a {
        *x = rng.next();
    }
    for x in &mut b {
        *x = rng.next();
    }

    // Naive reference through the exact strides the problem describes.
    let mut d_ref = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for kk in 0..k {
                acc += a[(i as isize * a_s[0] + kk as isize * a_s[1]) as usize]
                    * b[(kk as isize * b_s[0] + j as isize * b_s[1]) as usize];
            }
            d_ref[(i as isize + j as isize * m as isize) as usize] = acc;
        }
    }

    let exec = Exec::serial();
    let cfg = PlanConfig::packed();

    let dflt = Plan::<f64>::new(&problem, &cfg).unwrap();
    let mut d_dflt = vec![0.0f64; m * n];
    dflt.execute_slices(
        &exec,
        1.0,
        (a.as_slice(), 0),
        (b.as_slice(), 0),
        (d_dflt.as_mut_slice(), 0),
    )
    .unwrap();

    let cat = catalog();
    let mut selector = |_: &tcontract::SelectionContext<'_>,
                        cands: &[tcontract::KernelCandidate<f64>]|
     -> Result<tcontract_kernel::KernelHandle<f64>, SelectError> {
        cands
            .iter()
            .map(|c| c.handle)
            .find(|h| h.id() == openblas_kernel::OPENBLAS_FAMILY.id)
            .ok_or(SelectError::NoCandidates { dtype: "f64" })
    };
    let openblas = Plan::<f64>::new_with_selector(
        &problem,
        &cfg,
        &cat,
        &mut selector as &mut Chooser<'_, f64>,
    )
    .unwrap();
    let mut d_ob = vec![0.0f64; m * n];
    openblas
        .execute_slices(
            &exec,
            1.0,
            (a.as_slice(), 0),
            (b.as_slice(), 0),
            (d_ob.as_mut_slice(), 0),
        )
        .unwrap();

    let tag = format!("e2e m={m} n={n} k={k} a={a_s:?} b={b_s:?}");
    check(
        &format!("{tag}: default vs reference"),
        approx(&d_dflt, &d_ref, 1e-9),
    );
    check(
        &format!("{tag}: openblas vs reference"),
        approx(&d_ob, &d_ref, 1e-9),
    );
}

fn main() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for kc in [0usize, 1, 2, 3, 7, 16, 33] {
        tile_case(kc, &mut rng);
    }
    tile_exact_known();
    for kc in [0, 1, 3, 7, 128, 256, 257] {
        wide_case(kc, &mut rng);
    }

    // Column-major matrix (unit contraction stride) and a strided-A case.
    end_to_end(33, 17, 21, [1, 33], [1, 21], &mut rng);
    end_to_end(16, 4, 16, [1, 16], [1, 16], &mut rng);
    end_to_end(9, 5, 13, [13, 1], [1, 13], &mut rng); // A row-major: strided

    println!("all self-checks passed");
}
