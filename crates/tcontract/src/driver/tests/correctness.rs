//! End-to-end correctness against the brute-force oracle.
//!
//! Two complementary strategies:
//!
//! 1. **Randomised small problems with tiny cache blocking.** Setting
//!    `MC`/`KC`/`NC` to a handful of elements forces every level of the
//!    five-loop nest, every partial block and every packing edge case to fire
//!    on tensors small enough for the `O(prod of all extents)` oracle.
//! 2. **Large pure-GEMM problems with default blocking.** These cross the real
//!    `MC`/`KC`/`NC` boundaries and are checked against a straightforward
//!    triple loop, which is fast enough at these sizes.
//!
//! Together they cover the loop arithmetic both structurally and at scale.

use num_complex::Complex;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::compat::{contract_reference, RefOperand};
use super::compat::{ElementOp, Operand};
use super::compat::{Layout, Plan};
use crate::api::Scalar;
use tcontract_exec::{Exec, Pool};
use tcontract_kernel::{Blocking, ComplexMethod};
use tcontract_kernel::{Element, Real};

/// One pool for every thread-count sweep, wide enough for the widest of them,
/// so each run takes a budget of it instead of building threads.
fn sweep_pool() -> &'static Pool<'static> {
    static TP: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    static POOL: std::sync::OnceLock<Pool<'static>> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        let tp = TP.get_or_init(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(64)
                .build()
                .unwrap()
        });
        Pool::borrow(tp)
    })
}

// ---------------------------------------------------------------- utilities

fn sample<T: Element>(rng: &mut ChaCha8Rng) -> T {
    let re = T::Real::from_f64(rng.gen_range(-1.0..1.0));
    let im = if T::IS_COMPLEX {
        T::Real::from_f64(rng.gen_range(-1.0..1.0))
    } else {
        T::Real::ZERO
    };
    T::from_parts(re, im)
}

fn fill<T: Element>(n: usize, rng: &mut ChaCha8Rng) -> Vec<T> {
    (0..n).map(|_| sample::<T>(rng)).collect()
}

fn rel_error<T: Element>(got: &[T], want: &[T]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for (&g, &w) in got.iter().zip(want) {
        num += g.sub(w).norm().powi(2);
        den += w.norm().powi(2);
    }
    if den == 0.0 {
        num.sqrt()
    } else {
        (num / den).sqrt()
    }
}

/// 3m's error bound is relative to `|Ar||Br| + |Ai||Bi|` rather than to the
/// complex magnitudes, so it can lose relative accuracy under cancellation.
/// That is the documented price of the 25% flop saving, and the tolerance
/// reflects it rather than hiding it.
fn tol<T: Element>(method: ComplexMethod) -> f64 {
    let base = if core::mem::size_of::<T::Real>() == 4 {
        2e-4
    } else {
        1e-11
    };
    if T::IS_COMPLEX && method == ComplexMethod::ThreeM {
        base * 100.0
    } else {
        base
    }
}

/// A dense layout whose modes are contiguous in a random order, producing an
/// arbitrary stride permutation.
fn random_layout(extents: &[i64], rng: &mut ChaCha8Rng) -> Layout {
    let n = extents.len();
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        order.swap(i, rng.gen_range(0..=i));
    }
    let mut strides = vec![0i64; n];
    let mut acc = 1i64;
    for &d in &order {
        strides[d] = acc;
        acc *= extents[d];
    }
    Layout::new(extents.to_vec(), strides).expect("built with one stride per extent")
}

fn shuffled<T: Clone>(v: &[T], rng: &mut ChaCha8Rng) -> Vec<T> {
    let mut out = v.to_vec();
    for i in (1..out.len()).rev() {
        out.swap(i, rng.gen_range(0..=i));
    }
    out
}

// ------------------------------------------------------- randomised problems

struct Problem {
    idx_a: Vec<i64>,
    idx_b: Vec<i64>,
    idx_c: Vec<i64>,
    idx_d: Vec<i64>,
    la: Layout,
    lb: Layout,
    lc: Layout,
    ld: Layout,
    conj_a: bool,
    conj_b: bool,
    conj_c: bool,
    conj_d: bool,
    use_c: bool,
}

/// Generate a contraction covering all TAPP index cases the engine supports.
fn random_problem(rng: &mut ChaCha8Rng, complex: bool) -> Problem {
    let mut next_label = 0i64;
    let new_labels = |n: usize, next: &mut i64| -> Vec<i64> {
        (0..n)
            .map(|_| {
                *next += 1;
                *next
            })
            .collect()
    };

    let nm = rng.gen_range(0..=2);
    let nn = rng.gen_range(0..=2);
    let nk = rng.gen_range(1..=2);
    let nh = rng.gen_range(0..=1);
    let nia = rng.gen_range(0..=1); // isolated in A -> reduction
    let nib = rng.gen_range(0..=1); // isolated in B -> reduction

    let m = new_labels(nm, &mut next_label);
    let n = new_labels(nn, &mut next_label);
    let k = new_labels(nk, &mut next_label);
    let h = new_labels(nh, &mut next_label);
    let ia = new_labels(nia, &mut next_label);
    let ib = new_labels(nib, &mut next_label);

    let extent_of = |l: i64, rng: &mut ChaCha8Rng| -> i64 {
        let _ = l;
        rng.gen_range(1..=4)
    };
    let mut ext: Vec<(i64, i64)> = Vec::new();
    for &l in m.iter().chain(&n).chain(&k).chain(&h).chain(&ia).chain(&ib) {
        let e = extent_of(l, rng);
        ext.push((l, e));
    }
    let e_of = |l: i64| ext.iter().find(|x| x.0 == l).unwrap().1;

    let mut idx_a: Vec<i64> = m.iter().chain(&k).chain(&h).chain(&ia).copied().collect();
    let mut idx_b: Vec<i64> = k.iter().chain(&n).chain(&h).chain(&ib).copied().collect();
    let idx_d_base: Vec<i64> = m.iter().chain(&n).chain(&h).copied().collect();

    // With some probability, repeat a label inside A: selects A's diagonal.
    if !idx_a.is_empty() && rng.gen_bool(0.25) {
        let l = idx_a[rng.gen_range(0..idx_a.len())];
        idx_a.push(l);
    }
    if !idx_b.is_empty() && rng.gen_bool(0.15) {
        let l = idx_b[rng.gen_range(0..idx_b.len())];
        idx_b.push(l);
    }

    idx_a = shuffled(&idx_a, rng);
    idx_b = shuffled(&idx_b, rng);
    let idx_d = shuffled(&idx_d_base, rng);
    // C carries the same labels as D but may be laid out differently.
    let idx_c = shuffled(&idx_d_base, rng);

    let shape = |idx: &[i64]| -> Vec<i64> { idx.iter().map(|&l| e_of(l)).collect() };
    Problem {
        la: random_layout(&shape(&idx_a), rng),
        lb: random_layout(&shape(&idx_b), rng),
        lc: random_layout(&shape(&idx_c), rng),
        ld: random_layout(&shape(&idx_d), rng),
        conj_a: complex && rng.gen_bool(0.3),
        conj_b: complex && rng.gen_bool(0.3),
        conj_c: complex && rng.gen_bool(0.3),
        conj_d: complex && rng.gen_bool(0.2),
        use_c: rng.gen_bool(0.6),
        idx_a,
        idx_b,
        idx_c,
        idx_d,
    }
}

fn op(b: bool) -> ElementOp {
    if b {
        ElementOp::Conjugate
    } else {
        ElementOp::Identity
    }
}

fn check_problem<T>(
    p: &Problem,
    rng: &mut ChaCha8Rng,
    blocking: Option<Blocking>,
    method: ComplexMethod,
) -> f64
where
    T: Scalar,
{
    let a: Vec<T> = fill(p.la.storage_len() as usize, rng);
    let b: Vec<T> = fill(p.lb.storage_len() as usize, rng);
    let c: Vec<T> = fill(p.lc.storage_len() as usize, rng);
    let alpha = sample::<T>(rng);
    let beta = if p.use_c { sample::<T>(rng) } else { T::zero() };

    let mut got: Vec<T> = fill(p.ld.storage_len() as usize, rng);
    let mut want = got.clone();

    let mut plan = Plan::new(
        Operand {
            layout: &p.la,
            idx: &p.idx_a,
            op: op(p.conj_a),
        },
        Operand {
            layout: &p.lb,
            idx: &p.idx_b,
            op: op(p.conj_b),
        },
        p.use_c.then(|| Operand {
            layout: &p.lc,
            idx: &p.idx_c,
            op: op(p.conj_c),
        }),
        Operand {
            layout: &p.ld,
            idx: &p.idx_d,
            op: op(p.conj_d),
        },
    )
    .expect("plan")
    .with_complex_method(method);
    if let Some(blk) = blocking {
        plan = plan.with_blocking(blk);
    }

    unsafe {
        plan.run_raw::<T>(
            alpha,
            a.as_ptr(),
            b.as_ptr(),
            beta,
            c.as_ptr(),
            got.as_mut_ptr(),
        );
    }

    contract_reference::<T>(
        alpha,
        &RefOperand {
            data: &a,
            layout: &p.la,
            idx: &p.idx_a,
            op: op(p.conj_a),
        },
        &RefOperand {
            data: &b,
            layout: &p.lb,
            idx: &p.idx_b,
            op: op(p.conj_b),
        },
        beta,
        p.use_c.then_some(&RefOperand {
            data: &c,
            layout: &p.lc,
            idx: &p.idx_c,
            op: op(p.conj_c),
        }),
        &mut want,
        &p.ld,
        &p.idx_d,
        op(p.conj_d),
    )
    .expect("reference");

    rel_error(&got, &want)
}

/// Blocking small enough that a 4x4x4 problem still spans several blocks.
const TINY: Blocking = Blocking {
    mc: 1,
    kc: 2,
    nc: 1,
};

fn randomised_sweep<T>(seed: u64, iters: usize, complex: bool)
where
    T: Scalar,
{
    // Every complex method must produce the same answer as the oracle on the
    // same problem. Running all three over the same generated problems is the
    // property that keeps them genuinely interchangeable.
    let methods: &[ComplexMethod] = if complex {
        &ComplexMethod::ALL
    } else {
        &[ComplexMethod::Planar]
    };
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for i in 0..iters {
        let p = random_problem(&mut rng, complex);
        for &method in methods {
            for blk in [Some(TINY), None] {
                let err = check_problem::<T>(&p, &mut rng, blk, method);
                assert!(
                    err <= tol::<T>(method),
                    "iteration {i} method {} blocking {blk:?}: relative error {err:e}\n{p}",
                    method.name(),
                );
            }
        }
    }
}

impl std::fmt::Display for Problem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "  idx_a={:?} idx_b={:?} idx_c={:?} idx_d={:?}\n  \
             la={:?}\n  lb={:?}\n  lc={:?}\n  ld={:?}\n  \
             conj a/b/c/d = {}/{}/{}/{} use_c={}",
            self.idx_a,
            self.idx_b,
            self.idx_c,
            self.idx_d,
            self.la,
            self.lb,
            self.lc,
            self.ld,
            self.conj_a,
            self.conj_b,
            self.conj_c,
            self.conj_d,
            self.use_c,
        )
    }
}

/// The blocking the analytical model derives for this machine, for one element
/// type and complex method, at a given thread count.
fn model_blocking<T>(method: ComplexMethod, threads: usize) -> Blocking
where
    T: Scalar,
{
    use tcontract_kernel::blocking::{analytical, hierarchy, PanelGeom};
    let t = Layout::col_major(&[2, 2]);
    let probe = Plan::new(
        Operand::new(&t, &[0, 2]),
        Operand::new(&t, &[2, 1]),
        None,
        Operand::new(&t, &[0, 1]),
    )
    .unwrap()
    .with_complex_method(method);
    let rg = probe.resolved::<T>().unwrap();
    let f = rg.family();
    analytical(
        PanelGeom {
            real_bytes: core::mem::size_of::<T::Real>(),
            a_reals: f.a_per_k / f.mr,
            b_reals: f.b_per_k / f.nr,
            mr: f.mr,
            nr: f.nr,
        },
        threads,
        &hierarchy(),
    )
}

/// Run the oracle against the analytical model's blocking.
///
/// The model ships **off** (`Tuning::block_model` opts in), so
/// nothing else in this suite exercises the numbers it derives — and they are a
/// different regime from the constants, not a nudge: a `kc` two to eight times
/// shallower, an `mc` four to six times wider, an `nc` in the tens of thousands
/// where the constant is under a thousand. All of that lands in the driver's
/// loop arithmetic and its buffer sizing, so it is checked directly rather than
/// only when someone sets the variable. Both thread counts are covered because
/// the model derives a different `nc` for each.
fn model_blocking_sweep<T>(seed: u64, iters: usize, complex: bool)
where
    T: Scalar,
{
    let methods: &[ComplexMethod] = if complex {
        &ComplexMethod::ALL
    } else {
        &[ComplexMethod::Planar]
    };
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for i in 0..iters {
        let p = random_problem(&mut rng, complex);
        for &method in methods {
            for threads in [1, 8] {
                let blk = model_blocking::<T>(method, threads);
                let err = check_problem::<T>(&p, &mut rng, Some(blk), method);
                assert!(
                    err <= tol::<T>(method),
                    "iteration {i} method {} model blocking {blk:?} at {threads} thread(s): \
                     relative error {err:e}\n{p}",
                    method.name(),
                );
            }
        }
    }
}

#[test]
fn analytical_blocking_matches_the_oracle() {
    model_blocking_sweep::<f64>(0xA11A, 40, false);
    model_blocking_sweep::<f32>(0xA11B, 40, false);
    model_blocking_sweep::<Complex<f64>>(0xA11C, 40, true);
    model_blocking_sweep::<Complex<f32>>(0xA11D, 40, true);
}

#[test]
fn randomised_f64() {
    randomised_sweep::<f64>(0xC0FFEE, 300, false);
}

#[test]
fn randomised_f32() {
    randomised_sweep::<f32>(0xBEEF, 200, false);
}

#[test]
fn randomised_c64() {
    randomised_sweep::<Complex<f64>>(0xD00D, 300, true);
}

#[test]
fn randomised_c32() {
    randomised_sweep::<Complex<f32>>(0xFEED, 200, true);
}

// -------------------------------------------------- large blocking boundaries

fn naive_gemm<T: Element>(m: usize, n: usize, k: usize, a: &[T], b: &[T]) -> Vec<T> {
    let mut c = vec![T::zero(); m * n];
    for j in 0..n {
        for p in 0..k {
            let bv = b[p + j * k];
            for i in 0..m {
                c[i + j * m] = c[i + j * m].add(a[i + p * m].mul(bv));
            }
        }
    }
    c
}

/// Sizes chosen to straddle the real `MC`/`KC`/`NC` boundaries with an
/// awkward remainder in every dimension. Run for every complex method, since
/// each has its own `MC` (1m's packed A is twice the size, so its blocks fall
/// in different places).
fn large_gemm_case<T>(m: usize, n: usize, k: usize, seed: u64)
where
    T: Scalar,
{
    large_gemm_case_oriented::<T>(m, n, k, seed, false);
    // Same product with a row-major `D`, which is what makes the driver
    // compute `D^T = B^T A^T` instead. The whole engine runs mirrored — 1m's
    // asymmetric "1e"/"1r" pack formats included — so it is worth checking at a
    // size that crosses every cache block, not only in the randomised sweep.
    large_gemm_case_oriented::<T>(m, n, k, seed ^ 0x9E37, true);
}

fn large_gemm_case_oriented<T>(m: usize, n: usize, k: usize, seed: u64, row_major_d: bool)
where
    T: Scalar,
{
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let a: Vec<T> = fill(m * k, &mut rng);
    let b: Vec<T> = fill(k * n, &mut rng);
    let cm = naive_gemm(m, n, k, &a, &b);
    // The reference is column-major; the row-major variant stores the same
    // matrix transposed, so compare against the transposed reference.
    let want: Vec<T> = if row_major_d {
        (0..m * n).map(|t| cm[(t % n) * m + t / n]).collect()
    } else {
        cm
    };

    let la = Layout::col_major(&[m as i64, k as i64]);
    let lb = Layout::col_major(&[k as i64, n as i64]);
    let ld = if row_major_d {
        Layout::new(vec![m as i64, n as i64], vec![n as i64, 1]).unwrap()
    } else {
        Layout::col_major(&[m as i64, n as i64])
    };

    let methods: &[ComplexMethod] = if T::IS_COMPLEX {
        &ComplexMethod::ALL
    } else {
        &[ComplexMethod::Planar]
    };
    for &method in methods {
        let mut d = vec![T::zero(); m * n];
        let plan = Plan::new(
            Operand::new(&la, &[0, 2]),
            Operand::new(&lb, &[2, 1]),
            None,
            Operand::new(&ld, &[0, 1]),
        )
        .unwrap()
        .with_complex_method(method);
        assert!(plan.stats.is_pure_gemm);
        // The point of the row-major variant is that it takes the swapped
        // path; if the heuristic stops firing here the test still passes but
        // has quietly stopped testing anything.
        let (mr, ..) = plan.selected_config::<T>();
        assert_eq!(
            plan.transposes_gemm(mr),
            row_major_d,
            "row_major_d={row_major_d} should decide the orientation at mr={mr}"
        );

        unsafe {
            plan.run_raw::<T>(
                T::one(),
                a.as_ptr(),
                b.as_ptr(),
                T::zero(),
                d.as_ptr(),
                d.as_mut_ptr(),
            )
        };

        let err = rel_error(&d, &want);
        assert!(
            err <= tol::<T>(method),
            "{m}x{n}x{k} [{}]: relative error {err:e}",
            method.name()
        );
    }
}

#[test]
fn large_gemm_crosses_cache_blocks_f64() {
    // default f64 blocking is mc=256 kc=256 nc>=1536
    large_gemm_case::<f64>(301, 197, 523, 1);
}

#[test]
fn large_gemm_crosses_cache_blocks_c64() {
    large_gemm_case::<Complex<f64>>(211, 143, 401, 2);
}

#[test]
fn large_gemm_crosses_cache_blocks_f32() {
    large_gemm_case::<f32>(401, 233, 797, 3);
}

#[test]
fn large_gemm_crosses_cache_blocks_c32() {
    large_gemm_case::<Complex<f32>>(277, 181, 613, 4);
}

// ------------------------------------------------------------- degenerate cases

#[test]
fn empty_contraction_dimension_scales_c() {
    // D[i,j] = beta * C[i,j] when the contracted extent is zero.
    let la = Layout::col_major(&[3, 0]);
    let lb = Layout::col_major(&[0, 2]);
    let lc = Layout::col_major(&[3, 2]);
    let a: Vec<f64> = vec![];
    let b: Vec<f64> = vec![];
    let c: Vec<f64> = (0..6).map(|x| x as f64).collect();
    let mut d = vec![-1.0f64; 6];
    let plan = Plan::new(
        Operand::new(&la, &[0, 2]),
        Operand::new(&lb, &[2, 1]),
        Some(Operand::new(&lc, &[0, 1])),
        Operand::new(&lc, &[0, 1]),
    )
    .unwrap();
    assert!(plan.has_empty_contraction());
    unsafe {
        plan.run_raw::<f64>(1.0, a.as_ptr(), b.as_ptr(), 2.0, c.as_ptr(), d.as_mut_ptr());
    }
    assert_eq!(d, vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
}

#[test]
fn zero_sized_output_is_a_no_op() {
    let la = Layout::col_major(&[0, 3]);
    let lb = Layout::col_major(&[3, 2]);
    let ld = Layout::col_major(&[0, 2]);
    let b: Vec<f64> = vec![0.0; 6];
    let mut d: Vec<f64> = vec![];
    let plan = Plan::new(
        Operand::new(&la, &[0, 2]),
        Operand::new(&lb, &[2, 1]),
        None,
        Operand::new(&ld, &[0, 1]),
    )
    .unwrap();
    assert!(plan.is_empty());
    unsafe {
        plan.run_raw::<f64>(
            1.0,
            core::ptr::NonNull::dangling().as_ptr(),
            b.as_ptr(),
            0.0,
            d.as_ptr(),
            d.as_mut_ptr(),
        );
    }
}

#[test]
fn scalar_output_full_reduction() {
    // d = sum_{i,j} A[i,j] * B[i,j]  (a full double contraction to a scalar)
    let la = Layout::col_major(&[4, 5]);
    let lb = Layout::col_major(&[4, 5]);
    let ld = Layout::col_major(&[]);
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let a: Vec<f64> = fill(20, &mut rng);
    let b: Vec<f64> = fill(20, &mut rng);
    let mut d = vec![0.0f64];
    let plan = Plan::new(
        Operand::new(&la, &[0, 1]),
        Operand::new(&lb, &[0, 1]),
        None,
        Operand::new(&ld, &[]),
    )
    .unwrap();
    unsafe { plan.run_raw::<f64>(1.0, a.as_ptr(), b.as_ptr(), 0.0, d.as_ptr(), d.as_mut_ptr()) };
    let want: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    assert!((d[0] - want).abs() < 1e-12, "{} vs {}", d[0], want);
}

#[test]
fn negative_strides_via_reversed_axis() {
    // A[i,k] stored with the i axis reversed: stride -1, base at the end.
    let m = 5usize;
    let k = 4usize;
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let storage: Vec<f64> = fill(m * k, &mut rng);
    // logical A[i,p] = storage[(m-1-i) + p*m]
    let la = Layout::new(vec![m as i64, k as i64], vec![-1, m as i64]).unwrap();
    let lb = Layout::col_major(&[k as i64, 3]);
    let ld = Layout::col_major(&[m as i64, 3]);
    let b: Vec<f64> = fill(k * 3, &mut rng);
    let mut d = vec![0.0f64; m * 3];

    let plan = Plan::new(
        Operand::new(&la, &[0, 2]),
        Operand::new(&lb, &[2, 1]),
        None,
        Operand::new(&ld, &[0, 1]),
    )
    .unwrap();
    // Base pointer sits at the last element of the first column.
    unsafe {
        plan.run_raw::<f64>(
            1.0,
            storage.as_ptr().add(m - 1),
            b.as_ptr(),
            0.0,
            d.as_ptr(),
            d.as_mut_ptr(),
        );
    }

    for i in 0..m {
        for j in 0..3 {
            let mut want = 0.0;
            for p in 0..k {
                want += storage[(m - 1 - i) + p * m] * b[p + j * k];
            }
            assert!(
                (d[i + j * m] - want).abs() < 1e-12,
                "({i},{j}): {} vs {want}",
                d[i + j * m]
            );
        }
    }
}

#[test]
fn conjugation_matrix_is_consistent() {
    // Check all 16 combinations of conjugation flags against the oracle.
    let mut rng = ChaCha8Rng::seed_from_u64(23);
    let la = Layout::col_major(&[3, 4]);
    let lb = Layout::col_major(&[4, 2]);
    let ld = Layout::col_major(&[3, 2]);
    type T = Complex<f64>;
    let a: Vec<T> = fill(12, &mut rng);
    let b: Vec<T> = fill(8, &mut rng);
    let c: Vec<T> = fill(6, &mut rng);
    let alpha = sample::<T>(&mut rng);
    let beta = sample::<T>(&mut rng);

    for method in ComplexMethod::ALL {
        for mask in 0..16u8 {
            let (ca, cb, cc, cd) = (mask & 1 != 0, mask & 2 != 0, mask & 4 != 0, mask & 8 != 0);
            let mut got = vec![T::zero(); 6];
            let mut want = vec![T::zero(); 6];
            let plan = Plan::new(
                Operand {
                    layout: &la,
                    idx: &[0, 2],
                    op: op(ca),
                },
                Operand {
                    layout: &lb,
                    idx: &[2, 1],
                    op: op(cb),
                },
                Some(Operand {
                    layout: &ld,
                    idx: &[0, 1],
                    op: op(cc),
                }),
                Operand {
                    layout: &ld,
                    idx: &[0, 1],
                    op: op(cd),
                },
            )
            .unwrap()
            .with_complex_method(method)
            // force multiple K blocks so the accumulate path's conjugation
            // handling is exercised
            .with_blocking(Blocking {
                mc: 1,
                kc: 1,
                nc: 1,
            });
            unsafe {
                plan.run_raw::<T>(
                    alpha,
                    a.as_ptr(),
                    b.as_ptr(),
                    beta,
                    c.as_ptr(),
                    got.as_mut_ptr(),
                )
            };
            contract_reference::<T>(
                alpha,
                &RefOperand {
                    data: &a,
                    layout: &la,
                    idx: &[0, 2],
                    op: op(ca),
                },
                &RefOperand {
                    data: &b,
                    layout: &lb,
                    idx: &[2, 1],
                    op: op(cb),
                },
                beta,
                Some(&RefOperand {
                    data: &c,
                    layout: &ld,
                    idx: &[0, 1],
                    op: op(cc),
                }),
                &mut want,
                &ld,
                &[0, 1],
                op(cd),
            )
            .unwrap();
            let err = rel_error(&got, &want);
            assert!(
                err < tol::<T>(method),
                "conj mask {mask:04b} [{}]: relative error {err:e}",
                method.name()
            );
        }
    }
}

// ----------------------------------------------------------------- threading
//
// The driver partitions the output into a `pm x pn` grid: contiguous row strips
// of whole `MR` panels by column groups of whole `NR` blocks. Every output
// element therefore has exactly one owning thread, which accumulates over the
// full `K` range in the original order, so the result must be **identical for
// every thread count and every partition** — not merely equal to a tolerance.
// That is a far stronger invariant than agreeing with the reference oracle, and
// it is what these tests check: a partition that dropped a row or a column,
// double-counted one, or let a thread read the shared `B` panel across a barrier
// would all break it, while a tolerance-based check could easily miss the last
// of those.
//
// The second thing these tests have to do is prove the partition they are named
// after is really the one that ran, so `Split` below is asserted on every case.
// A barrier-count mismatch across threads *hangs* rather than failing, so the
// cases that stress the barriers are kept separately identifiable.

/// What the partition must look like, asserted rather than hoped for.
///
/// Every variant also checks the universal invariants. They differ in what else
/// they demand, and each extra demand is *guarded* by the regime in which it is
/// meaningful, because `MR` and `NR` differ by dtype, by complex method, and
/// between the AVX-512 and the portable kernels — a shape that is two row panels
/// deep in `f32` is seven in `c64` 3m and fourteen with a
/// pinned scalar kernel, so a bare literal here would be asserting the
/// register block rather than the rule.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Split {
    /// The universal invariants only.
    Any,
    /// Wherever the threads outnumber the row panels the column axis must be in
    /// use. This is the whole point of the 2-D partition, and the reason it is
    /// asserted and not assumed is that a shape can lose it silently — put the
    /// short direction in the column role and `M` fills the threads again.
    TwoD,
    /// As `TwoD`, and wherever the row axis is saturated many times over while
    /// the column axis still has room for a thread per group, *both* factors
    /// must exceed 1 — so the suite covers a genuine grid, with the
    /// per-column-group barriers live, and not only the two 1-D degenerate
    /// cases.
    Grid,
    /// Neither axis can be split at all: one panel by one block, which must run
    /// on a single thread however many were asked for.
    Serial,
}

/// Run one `A[m,k,h] B[k,n,h] -> D[m,n,h]` product at each thread count in
/// `threads` and require every result to match the serial one exactly.
fn threaded_case<T>(
    shape: (i64, i64, i64, i64),
    blocking: Option<Blocking>,
    method: ComplexMethod,
    threads: &[usize],
    want: Split,
    seed: u64,
) where
    T: Scalar,
{
    let (m, n, k, batch) = shape;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let la = Layout::col_major(&[m, k, batch]);
    let lb = Layout::col_major(&[k, n, batch]);
    let ld = Layout::col_major(&[m, n, batch]);
    let a: Vec<T> = fill(la.storage_len() as usize, &mut rng);
    let b: Vec<T> = fill(lb.storage_len() as usize, &mut rng);
    let alpha = sample::<T>(&mut rng);
    // Non-zero starting contents, so a strip that never gets written shows up
    // as leftover garbage rather than as a plausible zero.
    let start: Vec<T> = fill(ld.storage_len() as usize, &mut rng);

    let run = |nthreads: usize| -> Vec<T> {
        let plan = Plan::new(
            Operand::new(&la, &[0, 2, 3]),
            Operand::new(&lb, &[2, 1, 3]),
            None,
            Operand::new(&ld, &[0, 1, 3]),
        )
        .expect("plan")
        .with_complex_method(method);
        let plan = match blocking {
            Some(blk) => plan.with_blocking(blk),
            None => plan,
        };
        let plan = plan.with_threads(nthreads);
        assert_eq!(plan.threads(), nthreads.max(1));
        // Assert the case really splits, and how, so a partition that quietly
        // stopped engaging cannot leave these tests green and empty.
        //
        // `plan_config` and not `selected_config`: the row-block rule can pick a
        // shape other than the kernel set's default, and the partition is
        // quantised to the shape the driver will actually run.
        let (mr, nr, _) = plan.selected_config::<T>();
        // Both axes are the *oriented* ones: when the plan computes
        // `D^T = B^T A^T` the row strips run along `n` and the column groups
        // along `m`. Getting this wrong is how the first version of this
        // assertion failed — usefully, since it means a skinny-`M` contraction
        // can still parallelise along `M`, as long as it is skinny in the
        // direction that ends up in the column role.
        let swap = plan.transposes_gemm(mr);
        let (rows, cols) = if swap { (n, m) } else { (m, n) };
        let panels = (rows as usize).div_ceil(mr).max(1);
        let blocks = (cols as usize).div_ceil(nr).max(1);
        let (pm, pn) = plan.partition(mr, nr);
        let p = nthreads.max(1);
        let what = format!(
            "{m}x{n}x{k} [{}] {mr}x{nr} {} {panels}p x {blocks}b on {p} threads -> {pm}x{pn}",
            method.name(),
            if swap { "BA" } else { "AB" }
        );
        assert!(pm >= 1 && pn >= 1, "{what}: partition is empty");
        assert!(
            pm <= panels && pn <= blocks,
            "{what}: partition exceeds the panel/block counts, so some thread gets nothing"
        );
        // Everything else here is about the *rule*; the two assertions above
        // and the bitwise comparison are the ones a pinned partition would also
        // have to satisfy.
        // The default is the domain-aware rule since D44, so the answer below may
        // be 1-D in either direction.
        // Which direction is the *row* axis is exactly what the orientation
        // switch changes, so `panels`/`blocks` below refer to the other axis
        // when it is pinned and none of these shape assertions mean what they
        // say. The bitwise-identity check above is unaffected and has run.
        {
            assert!(
                pm * pn <= p,
                "{what}: partition oversubscribes the thread count"
            );
            if panels >= p {
                // The regime the whole corpus but four cases is in: the row axis
                // fills the threads by itself, and then the partition must be
                // exactly the 1-D one, unchanged from before `N` was split — the
                // one exception being the domain-aware gate, which swaps the axes
                // when the threads span several L3s and the column axis can fill
                // them too. Which of the two it picks is decided by conditions
                // `partition_rule_is_domain_aware` covers exhaustively and this
                // test cannot see; what it pins here is that the answer is still
                // **1-D in one direction or the other, never a grid**, since a
                // grid in this regime would mean the gate had leaked into the
                // cost model.
                let want: &[(usize, usize)] = &[(p, 1), (1, p)];
                assert!(
                    want.contains(&(pm, pn)),
                    "{what}: the row axis alone fills the threads, so this must stay 1-D \
                     (one of {want:?})"
                );
            }
            if matches!(want, Split::TwoD | Split::Grid) && p > panels {
                assert!(
                    pn >= 2,
                    "{what}: more threads than row panels, so the column axis must be used"
                );
            }
            if want == Split::Grid && panels >= 2 && p >= 8 * panels && blocks >= p {
                assert!(
                    pm >= 2 && pn >= 2,
                    "{what}: both axes have room, so both must be split"
                );
            }
            if want == Split::Serial {
                assert_eq!(
                    (pm, pn),
                    (1, 1),
                    "{what}: one panel by one block cannot be split at all"
                );
            }
        }
        let mut d = start.clone();
        let exec = Exec::rayon(sweep_pool())
            .with_budget(nthreads.max(1))
            .unwrap();
        unsafe {
            plan.run_raw_with::<T>(
                &exec,
                None,
                alpha,
                a.as_ptr(),
                b.as_ptr(),
                T::zero(),
                d.as_ptr(),
                d.as_mut_ptr(),
            )
        }
        .expect("the sweep's pool serves every width it asks for");
        d
    };

    let serial = run(1);
    for &p in threads {
        let got = run(p);
        let bad = serial
            .iter()
            .zip(&got)
            .enumerate()
            .find(|(_, (s, g))| s != g)
            .map(|(i, _)| i);
        assert!(
            bad.is_none(),
            "{m}x{n}x{k} batch {batch} [{}] on {p} threads differs from serial at element {:?}",
            method.name(),
            bad.unwrap()
        );
    }

    // And the serial answer itself is right, per batch slice — so "identical to
    // serial" is anchored to something rather than to itself.
    if batch == 1 {
        let want = naive_gemm::<T>(m as usize, n as usize, k as usize, &a, &b);
        let want: Vec<T> = want.iter().map(|&v| Element::mul(alpha, v)).collect();
        let err = rel_error(&serial, &want);
        assert!(err <= tol::<T>(method), "serial anchor: rel error {err:e}");
    }
}

/// Thread counts that bracket the interesting cases: 2 (an even split), 3 (an
/// uneven one, since the panel count is not divisible by it), 8 (a realistic
/// core count) and 64 (far more threads than there are row panels, which must
/// spill onto the column axis or clamp, never produce empty cells).
const THREAD_COUNTS: &[usize] = &[2, 3, 8, 64];

/// Blocking that forces many `jc`/`pc` iterations, hence many barriers and many
/// re-packings of the shared `B` panel, on a problem small enough to stay fast.
///
/// Note what it does to a 2-D partition: `nc` is rounded up to one whole `NR`
/// sliver, so most `jc` blocks have fewer slivers than there are column groups
/// and most groups come out empty. That is the path where a thread takes both of
/// a block's barriers and then does no work at all, which is precisely the one
/// whose failure mode is a hang rather than a wrong answer.
const BARRIER_STRESS: Blocking = Blocking {
    mc: 1, // rounded up to MR
    kc: 3,
    nc: 1, // rounded up to NR
};

/// Wide in `M`: hundreds of row panels, so the partition stays 1-D at every
/// thread count here and the universal invariant pins it to `(p, 1)`.
#[test]
fn threaded_matches_serial_f64() {
    for method in ComplexMethod::ALL {
        threaded_case::<f64>(
            (301, 197, 523, 1),
            None,
            method,
            THREAD_COUNTS,
            Split::Any,
            11,
        );
        let stress = Some(BARRIER_STRESS);
        threaded_case::<f64>((200, 40, 60, 1), stress, method, &[2, 8], Split::Any, 12);
    }
}

#[test]
fn threaded_matches_serial_f32() {
    for method in ComplexMethod::ALL {
        threaded_case::<f32>(
            (401, 233, 797, 1),
            None,
            method,
            THREAD_COUNTS,
            Split::Any,
            13,
        );
        let stress = Some(BARRIER_STRESS);
        threaded_case::<f32>((200, 40, 60, 1), stress, method, &[2, 8], Split::Any, 14);
    }
}

#[test]
fn threaded_matches_serial_c64() {
    type C = Complex<f64>;
    for method in ComplexMethod::ALL {
        threaded_case::<C>(
            (211, 143, 401, 1),
            None,
            method,
            THREAD_COUNTS,
            Split::Any,
            15,
        );
        let stress = Some(BARRIER_STRESS);
        threaded_case::<C>((200, 40, 60, 1), stress, method, &[2, 8], Split::Any, 16);
    }
}

#[test]
fn threaded_matches_serial_c32() {
    type C = Complex<f32>;
    for method in ComplexMethod::ALL {
        threaded_case::<C>(
            (277, 181, 613, 1),
            None,
            method,
            THREAD_COUNTS,
            Split::Any,
            17,
        );
        let stress = Some(BARRIER_STRESS);
        threaded_case::<C>((200, 40, 60, 1), stress, method, &[2, 8], Split::Any, 18);
    }
}

/// Narrow `M`, wide `N` — the shape the 2-D partition exists for, and the one
/// the corpus has four of (`aqrs-pa-pqrs`, `ij-ikl-ljk`, `ij-kil-lkj`,
/// `ijk-il-jlk`, none of which can fill eight threads from `M`).
///
/// `m = 56` is chosen to be at least the largest `MR` in the build (48, `f32`
/// real) so the orientation rule leaves the narrow direction in the row role —
/// otherwise the swap would put the wide one there and the case would fill the
/// threads from `M` after all, testing nothing. It is also below `8 * MR` for
/// every register block in the AVX-512 set, so eight threads already overflow
/// the row axis; the portable kernels have a smaller `MR` and reach the same
/// regime at 64.
///
/// `n = 2000` exceeds the default `NC` in every dtype, so this also covers a
/// *tail* `jc` block, where the last group is narrower than the others.
#[test]
fn threaded_two_d_narrow_m_wide_n() {
    const SHAPE: (i64, i64, i64, i64) = (56, 2000, 61, 1);
    for method in ComplexMethod::ALL {
        threaded_case::<f64>(SHAPE, None, method, THREAD_COUNTS, Split::Grid, 51);
        threaded_case::<f32>(SHAPE, None, method, THREAD_COUNTS, Split::Grid, 52);
        threaded_case::<Complex<f64>>(SHAPE, None, method, THREAD_COUNTS, Split::Grid, 53);
        threaded_case::<Complex<f32>>(SHAPE, None, method, THREAD_COUNTS, Split::Grid, 54);
    }
}

/// The same 2-D partition under the barrier-stressing blocking: hundreds of
/// `(jc, pc)` iterations, one `NR` sliver each, so most column groups are empty
/// in most of them. Kept as its own test because its failure mode is a hang.
#[test]
fn threaded_two_d_barrier_stress() {
    for method in ComplexMethod::ALL {
        let stress = Some(BARRIER_STRESS);
        threaded_case::<f64>((56, 900, 7, 1), stress, method, &[8, 64], Split::TwoD, 55);
        threaded_case::<Complex<f32>>((56, 900, 5, 1), stress, method, &[8], Split::TwoD, 56);
    }
}

/// A batch (Hadamard) axis puts the whole loop nest, barriers included, inside
/// an outer loop that every thread must traverse in lockstep. If the barrier
/// counts ever went out of step across threads this deadlocks rather than
/// failing, which is worth being able to tell apart from a wrong answer — so the
/// 2-D shape is exercised here too, since it is the one that changed loop 5.
#[test]
fn threaded_matches_serial_batched() {
    for method in ComplexMethod::ALL {
        threaded_case::<f64>((157, 31, 47, 5), None, method, &[2, 3, 8], Split::Any, 21);
        let stress = Some(BARRIER_STRESS);
        threaded_case::<Complex<f64>>((157, 31, 47, 5), stress, method, &[3], Split::Any, 22);
        threaded_case::<Complex<f32>>((88, 24, 19, 3), None, method, &[2, 8], Split::Any, 23);
        threaded_case::<f64>((56, 800, 41, 3), None, method, &[3, 8], Split::TwoD, 24);
        threaded_case::<Complex<f64>>((56, 800, 41, 3), None, method, &[8], Split::TwoD, 25);
    }
}

/// Fewer cells than threads, down to a one-panel-by-one-block output: the
/// partition must clamp to what the shape has instead of handing some thread an
/// empty cell (which would be harmless) or two threads the same one (which would
/// not).
///
/// Both extents are small, because both axes are the *oriented* ones — a 1x33
/// output splits into two strips of 33 after the orientation swap, and would
/// split along the column axis even without it, so keeping only `m` small would
/// not exercise the clamp at all. The `2x2` case is within one register block in
/// both directions for *every* shape in either kernel path (the smallest are
/// `MR = 2`, `NR = 4`, from the portable complex kernels), so it must come out
/// fully serial however many threads are asked for.
#[test]
fn threaded_clamps_below_one_cell_per_thread() {
    for method in ComplexMethod::ALL {
        for m in [1, 2, 7, 25, 49] {
            let (many, one) = (&[2, 8, 64][..], &[8][..]);
            threaded_case::<f64>(
                (m, 14, 17, 1),
                None,
                method,
                many,
                Split::Any,
                31 + m as u64,
            );
            threaded_case::<Complex<f32>>(
                (m, 12, 9, 2),
                None,
                method,
                one,
                Split::Any,
                41 + m as u64,
            );
        }
        threaded_case::<f64>((2, 2, 37, 1), None, method, &[2, 8, 64], Split::Serial, 61);
        threaded_case::<Complex<f64>>((2, 2, 5, 3), None, method, &[8], Split::Serial, 62);
    }
}

/// The partition rule itself, in isolation: it is a pure function of the panel
/// count, the block count and the thread count, so it can be pinned directly
/// rather than only through its effects on a result.
///
/// The expectations are written in units of `MR` and `NR` rather than as literal
/// extents, so this test says the same thing in every dtype and on both kernel
/// paths — see [`Split`] for why that matters.
#[test]
fn thread_partition_rule() {
    // A col-major `m x n` output with `m >= MR` keeps the row role with `M`, so
    // the panel and block counts are exactly `m/MR` and `n/NR`.
    //
    // `k` is deliberately **deeper than both shallow-`k` guards** (the row-block
    // rule's 32 and the partition gate's `BANDWIDTH_BOUND_K` of 64). That makes
    // this a test of the cost-model rule alone, on any machine: with a shallow
    // `k` the gate would fire wherever the thread set spans several L3s, so
    // `case(40, 40, 8)` would answer `1x8` on a chiplet CI runner and `8x1` on
    // a one-L3-per-socket one, and the test would be a machine detector.
    let case = |panels: usize, blocks: usize, p: usize| -> (usize, usize) {
        let probe = {
            let t = Layout::col_major(&[2, 2]);
            Plan::new(
                Operand::new(&t, &[0, 2]),
                Operand::new(&t, &[2, 1]),
                None,
                Operand::new(&t, &[0, 1]),
            )
            .expect("plan")
        };
        let (mr, nr, _) = probe.selected_config::<f64>();
        let (m, n, k) = ((panels * mr) as i64, (blocks * nr) as i64, 128);
        let la = Layout::col_major(&[m, k]);
        let lb = Layout::col_major(&[k, n]);
        let ld = Layout::col_major(&[m, n]);
        let plan = Plan::new(
            Operand::new(&la, &[0, 2]),
            Operand::new(&lb, &[2, 1]),
            None,
            Operand::new(&ld, &[0, 1]),
        )
        .expect("plan")
        .with_threads(p);
        let (mr2, nr2, _) = plan.selected_config::<f64>();
        assert_eq!((mr, nr), (mr2, nr2), "row-block rule moved the shape");
        assert!(
            !plan.transposes_gemm(mr),
            "{panels}x{blocks}: unexpected swap"
        );
        plan.partition(mr, nr)
    };

    // One thread is one cell, always, whatever the shape.
    assert_eq!(case(40, 40, 1), (1, 1));
    assert_eq!(case(1, 1, 1), (1, 1));
    // The row axis fills the threads: 1-D, exactly as before `N` was split.
    assert_eq!(case(40, 40, 8), (8, 1));
    assert_eq!(case(8, 400, 8), (8, 1));
    // Nothing to split: clamp, do not oversubscribe.
    assert_eq!(case(1, 1, 8), (1, 1));
    // Narrow row axis against a wide column axis: spill onto `N`. With only
    // three panels, three strips of one panel against 400 blocks is worse
    // balanced than one strip of three against 50, so the rule takes the latter.
    assert_eq!(case(3, 400, 8), (1, 8));
    assert_eq!(case(1, 400, 8), (1, 8));
    // ... but a column axis with little in it is not worth splitting: seven
    // strips of one panel beats one strip of seven against four blocks, even
    // though it leaves a thread idle.
    assert_eq!(case(7, 4, 8), (7, 1));
    // Both axes saturated: the product is the thread count, not more.
    let (pm, pn) = case(4, 400, 64);
    assert_eq!((pm, pn), (4, 16));
    assert!(pm * pn <= 64);
}

/// Conjugation is a property of the problem, fixed when the plan is built: a
/// conjugated plan and an unconjugated one compute different contractions on
/// the same buffers, and each gives its own answer.
///
/// (The views carry no element operation, so a per-call mismatch -- which the
/// old label-based `Plan::run` had to detect and reject -- cannot be expressed.)
#[test]
fn conjugation_belongs_to_the_plan() {
    use crate::api::{CSpec, DType, Labels, LayoutSpec, Op, OperandSpec, Problem};
    use strided_view::{StridedView, StridedViewMut};

    let spec = |op| OperandSpec::new(LayoutSpec::new(&[2, 2], &[1, 2], 0).unwrap()).with_op(op);
    let (ia, ib, id) = (
        [b'i' as i64, b'k' as i64],
        [b'k' as i64, b'j' as i64],
        [b'i' as i64, b'j' as i64],
    );
    let make = |conj_a: bool| {
        let problem = Problem::from_labels(
            DType::C64,
            spec(if conj_a { Op::Conjugate } else { Op::Identity }),
            spec(Op::Identity),
            CSpec::Absent,
            spec(Op::Identity),
            &Labels::new(&ia, &ib, &id),
        )
        .unwrap();
        crate::Plan::<Complex<f64>>::new(&problem, &crate::PlanConfig::default()).unwrap()
    };

    let a = vec![
        Complex::new(1.0f64, 2.0),
        Complex::new(3.0, 4.0),
        Complex::new(5.0, 6.0),
        Complex::new(7.0, 8.0),
    ];
    let identity = vec![
        Complex::new(1.0f64, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(1.0, 0.0),
    ];
    let run = |plan: &crate::Plan<Complex<f64>>| {
        let mut d = vec![Complex::new(0.0f64, 0.0); 4];
        let av = StridedView::new(&a, &[2, 2], &[1, 2], 0).unwrap();
        let bv = StridedView::new(&identity, &[2, 2], &[1, 2], 0).unwrap();
        let mut dv = StridedViewMut::new(&mut d, &[2, 2], &[1, 2], 0).unwrap();
        plan.execute_into(&Exec::serial(), Complex::new(1.0, 0.0), &av, &bv, &mut dv)
            .unwrap();
        d
    };
    assert_eq!(run(&make(false)), a);
    assert_eq!(
        run(&make(true)),
        a.iter().map(|z| z.conj()).collect::<Vec<_>>()
    );
}
