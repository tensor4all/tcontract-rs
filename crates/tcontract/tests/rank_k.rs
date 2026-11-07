//! The rank-k plan kind (`herk`/`syrk`, tensor4all/tcontract-rs#175), step 1:
//! the reference kernel against the testkit oracle, and the written-triangle
//! contract.
//!
//! The oracle is `tcontract_testkit::oracle::rank_k_reference`, a label
//! contraction with `A` passed twice and the contracted label repeated; it
//! computes both triangles, so these tests compare the requested one and check
//! the other — and every padding element of `C` — bit for bit.

use num_complex::{Complex32, Complex64};
use strided_view::{StridedView, StridedViewMut};
use tcontract::api::{
    AliasError, DType, Error, LayoutError, LayoutSpec, RankK, RankKOp, RankKProblem, Scalar,
    ShapeError, Triangle, Unsupported, ValueError,
};
use tcontract::{Algorithm, ExecutionRoute, OutputContract, Plan, PlanConfig, Reason};
use tcontract_exec::Exec;
use tcontract_kernel::Real;
use tcontract_testkit::fixtures::seeded;
use tcontract_testkit::oracle::{rank_k_reference, RankKCase};

/// Exact bit patterns, and values that look uninitialised: NaNs with payloads,
/// infinities, extremes and subnormals.
trait Bits: Scalar {
    fn bits(self) -> (u64, u64);
    fn garbage(i: usize) -> Self;
    fn tol() -> f64;
}

fn garbage64(i: usize) -> f64 {
    match i % 4 {
        0 => f64::from_bits(0x7ff8_dead_0000_0000 | i as u64),
        1 => f64::NEG_INFINITY,
        2 => f64::MAX,
        _ => f64::from_bits(1 + i as u64),
    }
}

fn garbage32(i: usize) -> f32 {
    match i % 4 {
        0 => f32::from_bits(0x7fc0_0000 | (i as u32 & 0xffff)),
        1 => f32::NEG_INFINITY,
        2 => f32::MAX,
        _ => f32::from_bits(1 + i as u32),
    }
}

impl Bits for f64 {
    fn bits(self) -> (u64, u64) {
        (self.to_bits(), 0)
    }
    fn garbage(i: usize) -> Self {
        garbage64(i)
    }
    fn tol() -> f64 {
        1e-12
    }
}

impl Bits for f32 {
    fn bits(self) -> (u64, u64) {
        (self.to_bits().into(), 0)
    }
    fn garbage(i: usize) -> Self {
        garbage32(i)
    }
    fn tol() -> f64 {
        1e-4
    }
}

impl Bits for Complex64 {
    fn bits(self) -> (u64, u64) {
        (self.re.to_bits(), self.im.to_bits())
    }
    fn garbage(i: usize) -> Self {
        Complex64::new(garbage64(i), garbage64(i + 1))
    }
    fn tol() -> f64 {
        1e-12
    }
}

impl Bits for Complex32 {
    fn bits(self) -> (u64, u64) {
        (self.re.to_bits().into(), self.im.to_bits().into())
    }
    fn garbage(i: usize) -> Self {
        Complex32::new(garbage32(i), garbage32(i + 1))
    }
    fn tol() -> f64 {
        1e-4
    }
}

fn scalar<T: Scalar>(re: f64, im: f64) -> T {
    T::from_parts(Real::from_f64(re), Real::from_f64(im))
}

/// One case: `A` and `C` laid out with padding and an offset, so a stray write
/// outside the triangle or the matrix shows up.
struct Case<T> {
    n: usize,
    k: usize,
    op: RankKOp,
    kind: RankK,
    a: Vec<T>,
    a_dims: [usize; 2],
    a_strides: [isize; 2],
    a_off: isize,
    c: Vec<T>,
    c_strides: [isize; 2],
    c_off: isize,
}

impl<T: Bits> Case<T> {
    fn new(n: usize, k: usize, op: RankKOp, kind: RankK, seed: u64) -> Self {
        // A: row-major for NoTrans, column-major with a padded leading
        // dimension for Trans, both after one padding element.
        let a_dims = match op {
            RankKOp::NoTrans => [n, k],
            RankKOp::Trans => [k, n],
        };
        let a_strides = match op {
            RankKOp::NoTrans => [k.max(1) as isize, 1],
            RankKOp::Trans => [1, a_dims[0] as isize + 2],
        };
        let a_off = 1;
        let a_len = 1 + (a_dims[0] + 2) * a_dims[1].max(1) + a_dims[0] * a_dims[1];
        let a = seeded::<T>(seed, a_len);
        // C: column-major, leading dimension n + 3, offset 2, every element
        // garbage except the requested triangle.
        let ld = n as isize + 3;
        let c_strides = [1, ld];
        let c_off = 2;
        let c_len = 2 + (ld as usize) * n.max(1) + 1;
        let vals = seeded::<T>(seed ^ 0x5eed, n * n);
        let mut c: Vec<T> = (0..c_len).map(T::garbage).collect();
        for j in 0..n {
            for i in 0..n {
                if kind.triangle.contains(i, j) {
                    let mut v = vals[i + n * j];
                    if kind.hermitian && i == j {
                        v = T::from_parts(v.re(), Real::ZERO);
                    }
                    c[(c_off + i as isize + ld * j as isize) as usize] = v;
                }
            }
        }
        Self {
            n,
            k,
            op,
            kind,
            a,
            a_dims,
            a_strides,
            a_off,
            c,
            c_strides,
            c_off,
        }
    }

    fn c_at(&self, i: usize, j: usize) -> usize {
        (self.c_off + i as isize * self.c_strides[0] + j as isize * self.c_strides[1]) as usize
    }

    fn problem(&self) -> RankKProblem {
        RankKProblem::new(
            T::STORAGE,
            LayoutSpec::new(&self.a_dims, &self.a_strides, self.a_off).unwrap(),
            self.op,
            LayoutSpec::new(&[self.n, self.n], &self.c_strides, self.c_off).unwrap(),
            self.kind,
        )
        .unwrap()
    }

    fn execute(&mut self, plan: &Plan<T>, alpha: T, beta: T) -> tcontract::Result<()> {
        let av = StridedView::new(&self.a, &self.a_dims, &self.a_strides, self.a_off).unwrap();
        let mut cv =
            StridedViewMut::new(&mut self.c, &[self.n, self.n], &self.c_strides, self.c_off)
                .unwrap();
        plan.execute_rank_k(&mut Exec::serial(), alpha, &av, beta, &mut cv)
    }

    /// Run and check: the requested triangle against the oracle, a real
    /// Hermitian diagonal, and every other storage element bit for bit.
    fn check(mut self, alpha: T, beta: T) {
        if beta == T::zero() {
            // `beta == 0` reads no `C`: the requested triangle may be garbage
            // too, and a kernel that read it would carry a NaN into the result.
            for j in 0..self.n {
                for i in 0..self.n {
                    let at = self.c_at(i, j);
                    self.c[at] = T::garbage(at);
                }
            }
        }
        let before = self.c.clone();
        let n = self.n;
        let dense: Vec<T> = (0..n * n)
            .map(|p| {
                let (i, j) = (p % n, p / n);
                if self.kind.triangle.contains(i, j) {
                    before[self.c_at(i, j)]
                } else {
                    T::zero()
                }
            })
            .collect();
        let want = rank_k_reference(
            &RankKCase {
                data: &self.a,
                dims: self.a_dims,
                strides: self.a_strides,
                offset: self.a_off,
                transpose: self.op == RankKOp::Trans,
                hermitian: self.kind.hermitian,
            },
            alpha,
            beta,
            &dense,
        )
        .unwrap();
        let plan = Plan::<T>::new_rank_k(&self.problem(), &PlanConfig::default()).unwrap();
        self.execute(&plan, alpha, beta).unwrap();
        let label = format!(
            "{} n={} k={} {:?} {:?} alpha={alpha:?} beta={beta:?}",
            T::STORAGE.name(),
            n,
            self.k,
            self.op,
            self.kind
        );
        let scale = want.iter().map(|x| x.norm()).fold(1.0, f64::max);
        let mut written = vec![false; self.c.len()];
        for j in 0..n {
            for i in 0..n {
                if !self.kind.triangle.contains(i, j) {
                    continue;
                }
                let at = self.c_at(i, j);
                written[at] = true;
                let (got, w) = (self.c[at], want[i + n * j]);
                assert!(
                    got.sub(w).norm() <= T::tol() * scale,
                    "{label}: C[{i},{j}] = {got:?}, want {w:?}"
                );
                if self.kind.hermitian && i == j {
                    assert_eq!(
                        got.im().to_f64().to_bits(),
                        0,
                        "{label}: diagonal {i} not real"
                    );
                }
            }
        }
        for (p, (&got, &old)) in self.c.iter().zip(&before).enumerate() {
            if !written[p] {
                assert_eq!(
                    got.bits(),
                    old.bits(),
                    "{label}: storage element {p} changed"
                );
            }
        }
    }
}

const SHAPES: [(usize, usize); 7] = [(5, 3), (3, 7), (4, 4), (1, 1), (6, 1), (0, 3), (4, 0)];

fn sweep<T: Bits>() {
    let mut seed = 1;
    for (n, k) in SHAPES {
        for op in [RankKOp::NoTrans, RankKOp::Trans] {
            for triangle in [Triangle::Lower, Triangle::Upper] {
                for hermitian in [true, false] {
                    let kind = RankK {
                        triangle,
                        hermitian,
                    };
                    // Real scalars for the Hermitian form, complex ones for the
                    // symmetric one (a real dtype drops the imaginary part).
                    let im = if hermitian { 0.0 } else { 1.0 };
                    for (alpha, beta) in [
                        (scalar::<T>(1.0, 0.0), T::zero()),
                        (scalar::<T>(-0.75, 0.5 * im), scalar::<T>(1.25, -0.5 * im)),
                        (T::zero(), scalar::<T>(2.0, im)),
                        (T::zero(), T::zero()),
                    ] {
                        seed += 1;
                        Case::<T>::new(n, k, op, kind, seed).check(alpha, beta);
                    }
                }
            }
        }
    }
}

#[test]
fn sweep_matches_oracle_and_leaves_the_other_triangle_bitwise() {
    sweep::<f64>();
    sweep::<f32>();
    sweep::<Complex64>();
    sweep::<Complex32>();
}

/// The two named forms on their own, with nonzero `beta`: the Gram matrix
/// `A^H A` and `A A^H`, non-square in both directions.
#[test]
fn gram_and_outer_forms_match_the_oracle_for_non_square_a() {
    for op in [RankKOp::Trans, RankKOp::NoTrans] {
        for (n, k) in [(5, 3), (3, 8)] {
            for triangle in [Triangle::Lower, Triangle::Upper] {
                let kind = RankK {
                    triangle,
                    hermitian: true,
                };
                Case::<Complex64>::new(n, k, op, kind, 77)
                    .check(scalar(0.5, 0.0), scalar(-1.5, 0.0));
            }
        }
    }
}

/// `beta != 0` with every element outside the triangle a NaN: had the kernel
/// read one, a NaN would reach the triangle; had it written one, the bits
/// would differ.
#[test]
fn beta_accumulates_only_in_the_requested_triangle() {
    for triangle in [Triangle::Lower, Triangle::Upper] {
        let kind = RankK {
            triangle,
            hermitian: false,
        };
        let case = Case::<Complex64>::new(4, 3, RankKOp::NoTrans, kind, 9);
        case.check(scalar(1.0, 0.0), scalar(0.5, 2.0));
    }
}

fn hermitian_case(beta_im_diag: f64) -> Case<Complex64> {
    let kind = RankK {
        triangle: Triangle::Lower,
        hermitian: true,
    };
    let mut case = Case::<Complex64>::new(4, 3, RankKOp::Trans, kind, 3);
    let at = case.c_at(2, 2);
    case.c[at] = Complex64::new(1.0, beta_im_diag);
    case
}

#[test]
fn hermitian_diagonal_with_an_imaginary_part_is_rejected_and_c_is_unchanged() {
    for bad in [0.5, f64::NAN] {
        let mut case = hermitian_case(bad);
        let plan = Plan::<Complex64>::new_rank_k(&case.problem(), &PlanConfig::default()).unwrap();
        let before: Vec<_> = case.c.iter().map(|x| x.bits()).collect();
        let err = case
            .execute(&plan, scalar(1.0, 0.0), scalar(1.0, 0.0))
            .unwrap_err();
        assert!(
            matches!(
                err,
                Error::Value(ValueError::HermitianDiagonal { index: 2 })
            ),
            "{err:?}"
        );
        let after: Vec<_> = case.c.iter().map(|x| x.bits()).collect();
        assert_eq!(before, after, "a refused call wrote C");
    }
}

/// `beta == 0` reads no `C`, so an imaginary diagonal there is not an input
/// and is overwritten with a real value.
#[test]
fn hermitian_diagonal_is_not_read_with_zero_beta() {
    let case = hermitian_case(0.5);
    case.check(scalar(2.0, 0.0), Complex64::new(0.0, 0.0));
}

#[test]
fn hermitian_form_rejects_complex_alpha_and_beta() {
    let mut case = hermitian_case(0.0);
    let plan = Plan::<Complex64>::new_rank_k(&case.problem(), &PlanConfig::default()).unwrap();
    let before: Vec<_> = case.c.iter().map(|x| x.bits()).collect();
    for (alpha, beta, which) in [
        (
            Complex64::new(1.0, 1e-300),
            Complex64::new(0.0, 0.0),
            "alpha",
        ),
        (Complex64::new(1.0, 0.0), Complex64::new(1.0, -1.0), "beta"),
        (
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, f64::NAN),
            "beta",
        ),
    ] {
        let err = case.execute(&plan, alpha, beta).unwrap_err();
        assert!(
            matches!(err, Error::Value(ValueError::ComplexScalar { which: w }) if w == which),
            "{err:?}"
        );
    }
    let after: Vec<_> = case.c.iter().map(|x| x.bits()).collect();
    assert_eq!(before, after);
}

#[test]
fn route_cost_and_report_are_the_reference_kernels() {
    let kind = RankK {
        triangle: Triangle::Upper,
        hermitian: true,
    };
    let case = Case::<Complex64>::new(5, 3, RankKOp::NoTrans, kind, 1);
    let plan = Plan::<Complex64>::new_rank_k(&case.problem(), &PlanConfig::default()).unwrap();
    assert_eq!(plan.report().algorithm, Algorithm::RankKReference);
    assert_eq!(plan.report().reason, Reason::RankK);
    assert!(plan.report().packed.is_none());
    assert_eq!(plan.rank_k(), Some(&case.problem()));
    let exec = Exec::serial();
    for alpha in [scalar(1.0, 0.0), Complex64::new(0.0, 0.0)] {
        for output in [OutputContract::Initialized, OutputContract::Fresh] {
            assert_eq!(
                plan.execution_route(&exec, alpha, output).unwrap(),
                ExecutionRoute::RankKReference
            );
        }
    }
    assert_eq!(
        plan.workspace_req(&exec, OutputContract::Initialized)
            .bytes(),
        0
    );
    // n (n + 1) / 2 * k complex multiply-adds, four real ones each.
    assert_eq!(plan.muladds(), (15 * 3 * 4) as f64);
    // Serial: no width is cheaper than one.
    let rate = tcontract_exec::host::muladds_per_ns();
    assert_eq!(plan.estimate(8, rate, &exec), plan.estimate(1, rate, &exec));

    let empty = Case::<Complex64>::new(0, 3, RankKOp::NoTrans, kind, 1);
    let plan = Plan::<Complex64>::new_rank_k(&empty.problem(), &PlanConfig::default()).unwrap();
    assert_eq!(
        plan.execution_route(&exec, scalar(1.0, 0.0), OutputContract::Initialized)
            .unwrap(),
        ExecutionRoute::Empty
    );
}

#[test]
fn general_entries_refuse_a_rank_k_plan_and_the_reverse() {
    let kind = RankK {
        triangle: Triangle::Lower,
        hermitian: false,
    };
    let mut case = Case::<f64>::new(3, 3, RankKOp::Trans, kind, 4);
    let plan = Plan::<f64>::new_rank_k(&case.problem(), &PlanConfig::default()).unwrap();
    let before: Vec<_> = case.c.iter().map(|x| x.to_bits()).collect();
    {
        let av = StridedView::new(&case.a, &case.a_dims, &case.a_strides, case.a_off).unwrap();
        let mut cv =
            StridedViewMut::new(&mut case.c, &[3, 3], &case.c_strides, case.c_off).unwrap();
        let err = plan
            .execute_into(&mut Exec::serial(), 1.0, &av, &av, &mut cv)
            .unwrap_err();
        assert!(
            matches!(err, Error::Unsupported(Unsupported::Reason(_))),
            "{err:?}"
        );
    }
    let err = plan
        .execute_slices(
            &mut Exec::serial(),
            1.0,
            (&case.a, case.a_off),
            (&case.a, case.a_off),
            (&mut case.c, case.c_off),
        )
        .unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
    let after: Vec<_> = case.c.iter().map(|x| x.to_bits()).collect();
    assert_eq!(before, after);

    // A general product plan has no rank-k entry.
    let general = Plan::<f64>::new(plan.problem(), &PlanConfig::default()).unwrap();
    assert!(general.rank_k().is_none());
    let err = case.execute(&general, 1.0, 0.0).unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
}

#[test]
fn construction_errors() {
    let kind = RankK {
        triangle: Triangle::Lower,
        hermitian: true,
    };
    let l = |d: &[usize], s: &[isize]| LayoutSpec::new(d, s, 0).unwrap();
    let new = |a, op, c| RankKProblem::new(DType::C64, a, op, c, kind);
    // C must be n x n for op(A) n x k.
    assert!(matches!(
        new(l(&[4, 2], &[1, 4]), RankKOp::Trans, l(&[4, 4], &[1, 4])),
        Err(Error::Shape(ShapeError::OutputExtents { .. }))
    ));
    assert!(matches!(
        new(
            l(&[4, 2, 1], &[1, 4, 8]),
            RankKOp::NoTrans,
            l(&[4, 4], &[1, 4])
        ),
        Err(Error::Shape(ShapeError::NotMatrix { rank: 3, .. }))
    ));
    assert!(matches!(
        new(l(&[4, 2], &[1, 4]), RankKOp::NoTrans, l(&[4], &[1])),
        Err(Error::Shape(ShapeError::NotMatrix { rank: 1, .. }))
    ));
    // A C that addresses an element twice cannot be updated in place.
    assert!(matches!(
        new(l(&[4, 2], &[1, 4]), RankKOp::NoTrans, l(&[4, 4], &[1, 0])),
        Err(Error::Alias(AliasError::OutputNotInjective))
    ));
    let ok = new(l(&[4, 2], &[1, 4]), RankKOp::NoTrans, l(&[4, 4], &[1, 4])).unwrap();
    // The plan's dtype must match, and the packed driver cannot be forced.
    assert!(matches!(
        Plan::<f64>::new_rank_k(&ok, &PlanConfig::default()),
        Err(Error::Config(_))
    ));
    assert!(matches!(
        Plan::<Complex64>::new_rank_k(&ok, &PlanConfig::packed()),
        Err(Error::Unsupported(Unsupported::Reason(_)))
    ));
    // A view whose layout differs from the plan's is refused before writing.
    let plan = Plan::<Complex64>::new_rank_k(&ok, &PlanConfig::default()).unwrap();
    let a = vec![Complex64::new(1.0, 0.0); 8];
    let mut c = vec![Complex64::new(0.0, 0.0); 16];
    let av = StridedView::new(&a, &[4, 2], &[2, 1], 0).unwrap();
    let mut cv = StridedViewMut::new(&mut c, &[4, 4], &[1, 4], 0).unwrap();
    let err = plan
        .execute_rank_k(
            &mut Exec::serial(),
            Complex64::new(1.0, 0.0),
            &av,
            Complex64::new(0.0, 0.0),
            &mut cv,
        )
        .unwrap_err();
    assert!(
        matches!(err, Error::Layout(LayoutError::Mismatch { .. })),
        "{err:?}"
    );
    assert!(c.iter().all(|x| *x == Complex64::new(0.0, 0.0)));
}

/// `alpha == 0` reads no `A`: with every `A` element a NaN or an infinity, the
/// triangle is exactly `beta * C` and the rest is untouched. (The oracle reads
/// `A` regardless, so the expectation is written out here.)
#[test]
fn zero_alpha_reads_no_a() {
    for hermitian in [true, false] {
        let kind = RankK {
            triangle: Triangle::Upper,
            hermitian,
        };
        let mut case = Case::<Complex64>::new(4, 3, RankKOp::NoTrans, kind, 11);
        for (p, x) in case.a.iter_mut().enumerate() {
            *x = Complex64::garbage(p);
        }
        let before = case.c.clone();
        let beta = scalar::<Complex64>(0.5, 0.0);
        let plan = Plan::<Complex64>::new_rank_k(&case.problem(), &PlanConfig::default()).unwrap();
        case.execute(&plan, Complex64::new(0.0, 0.0), beta).unwrap();
        for (p, (&got, &old)) in case.c.iter().zip(&before).enumerate() {
            let in_triangle = (0..4).any(|j| (0..=j).any(|i| case.c_at(i, j) == p));
            if in_triangle {
                assert_eq!(got.bits(), (beta * old).bits(), "element {p}");
            } else {
                assert_eq!(got.bits(), old.bits(), "element {p}");
            }
        }
    }
}
