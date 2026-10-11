//! Shared plumbing for the harness: element traits, timing, and the
//! per-engine runners.

pub mod run;
pub mod verify;

#[cfg(feature = "tblis")]
use std::os::raw::c_int;
use std::time::Instant;

use num_complex::Complex;
use rand::Rng;
use tcontract::api::{CSpec, Labels, LayoutSpec, OperandSpec, Problem, Scalar};
use tcontract_kernel::Element;

use crate::blas::GemmScalar;
use crate::corpus::{Layout, Sized};

/// An element type the harness can drive through every engine.
#[allow(dead_code)] // `REAL_NAME` is not read by any current report
pub trait BenchElem: Scalar + GemmScalar {
    const NAME: &'static str;
    /// The same shape in the corresponding real type, for ratio reporting.
    const REAL_NAME: &'static str;
    /// The `type_t` this type maps to in the linked TBLIS.
    #[cfg(feature = "tblis")]
    const TBLIS_TYPE: c_int;
    /// A TBLIS scalar of this type, from the real value `v`.
    #[cfg(feature = "tblis")]
    fn tblis_scalar(v: f64) -> crate::tblis::tblis_scalar;

    fn sample(rng: &mut impl Rng) -> Self;
    fn from_f64(v: f64) -> Self;
}

impl BenchElem for f32 {
    const NAME: &'static str = "f32";
    const REAL_NAME: &'static str = "f32";
    #[cfg(feature = "tblis")]
    const TBLIS_TYPE: c_int = crate::tblis::TYPE_SINGLE;
    #[cfg(feature = "tblis")]
    fn tblis_scalar(v: f64) -> crate::tblis::tblis_scalar {
        crate::tblis::tblis_scalar::f32(v as f32)
    }
    fn sample(rng: &mut impl Rng) -> Self {
        rng.gen_range(-1.0..1.0)
    }
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}

impl BenchElem for f64 {
    const NAME: &'static str = "f64";
    const REAL_NAME: &'static str = "f64";
    #[cfg(feature = "tblis")]
    const TBLIS_TYPE: c_int = crate::tblis::TYPE_DOUBLE;
    #[cfg(feature = "tblis")]
    fn tblis_scalar(v: f64) -> crate::tblis::tblis_scalar {
        crate::tblis::tblis_scalar::f64(v)
    }
    fn sample(rng: &mut impl Rng) -> Self {
        rng.gen_range(-1.0..1.0)
    }
    fn from_f64(v: f64) -> Self {
        v
    }
}

impl BenchElem for Complex<f32> {
    const NAME: &'static str = "c32";
    const REAL_NAME: &'static str = "f32";
    #[cfg(feature = "tblis")]
    const TBLIS_TYPE: c_int = crate::tblis::TYPE_SCOMPLEX;
    #[cfg(feature = "tblis")]
    fn tblis_scalar(v: f64) -> crate::tblis::tblis_scalar {
        crate::tblis::tblis_scalar::c32(v as f32, 0.0)
    }
    fn sample(rng: &mut impl Rng) -> Self {
        Complex::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0))
    }
    fn from_f64(v: f64) -> Self {
        Complex::new(v as f32, 0.0)
    }
}

impl BenchElem for Complex<f64> {
    const NAME: &'static str = "c64";
    const REAL_NAME: &'static str = "f64";
    #[cfg(feature = "tblis")]
    const TBLIS_TYPE: c_int = crate::tblis::TYPE_DCOMPLEX;
    #[cfg(feature = "tblis")]
    fn tblis_scalar(v: f64) -> crate::tblis::tblis_scalar {
        crate::tblis::tblis_scalar::c64(v, 0.0)
    }
    fn sample(rng: &mut impl Rng) -> Self {
        Complex::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0))
    }
    fn from_f64(v: f64) -> Self {
        Complex::new(v, 0.0)
    }
}

fn operand(l: &Layout) -> OperandSpec {
    let dims: Vec<usize> = l.extents().iter().map(|&e| e as usize).collect();
    let strides: Vec<isize> = l.strides().iter().map(|&s| s as isize).collect();
    OperandSpec::new(LayoutSpec::new(&dims, &strides, 0).expect("corpus layouts are valid"))
}

/// The validated problem of a sized corpus case in storage type `T`: `D` is
/// overwritten (`beta = 0`), so there is no `C`.
pub fn problem_of<T: BenchElem>(s: &Sized) -> Result<Problem, tcontract::Error> {
    Problem::from_labels(
        T::STORAGE,
        operand(&s.la),
        operand(&s.lb),
        CSpec::Absent,
        operand(&s.lc),
        &Labels::new(&s.idx_a, &s.idx_b, &s.idx_c),
    )
}

/// Best-of-`reps` wall time in seconds, after time-based priming.
///
/// Priming is a wall-clock duration, not a call count: after an idle gate this
/// host reads up to 25% low for the first one to two seconds of sustained
/// AVX-512 work, and a fixed number of warm-up calls removes a different share
/// of that bias in every arm (largest in the fastest arm). `prime_ms == 0`
/// skips priming.
///
/// The 1500 ms default is not a round number: at 500 ms the *first* arm measured
/// for a case read 2.89 ms against 2.06 ms once settled — 40% higher latency, a
/// 29% lower rate — and the arm measured after it read 2.20 ms, so it looked 29%
/// faster than the same work measured on its own. At 1500 ms that case read 2.06
/// ms whether measured alone or after another arm, and so did it at 3000 ms.
/// That is those measured cases, not a guarantee: the arms are still measured one
/// after another in a fixed order, and `best` alone would hide a slow first
/// repetition, which is why the scatter is returned beside it. The diagnostic to
/// apply by hand — the harness compares outputs, not timings — is that two arms
/// whose rows carry the same family, blocking and partition policy must agree.
///
/// Returns `(best, scatter)` in seconds, scatter being `(max - min) / best` over
/// the repetitions, which is 0 for a single repetition.
pub fn timed(reps: usize, prime_ms: u64, mut f: impl FnMut()) -> (f64, f64) {
    let until = Instant::now() + std::time::Duration::from_millis(prime_ms);
    while Instant::now() < until {
        f();
    }
    let (mut best, mut worst) = (f64::INFINITY, 0.0f64);
    for _ in 0..reps.max(1) {
        let t = Instant::now();
        f();
        let secs = t.elapsed().as_secs_f64();
        best = best.min(secs);
        worst = worst.max(secs);
    }
    let scatter = if best.is_finite() && best > 0.0 {
        (worst - best) / best
    } else {
        0.0
    };
    (best, scatter)
}

/// Time what an arm prepares *before* the timed call — a plan, or the reference's
/// operand descriptors — as `(value, best seconds)`, best of `reps`.
///
/// The timing policy excludes this work, which is why the harness builds a `Plan`
/// once and times `execute`; recording it keeps the exclusion visible, because the
/// reference has nothing equivalent to hoist (its analysis is inside the call it
/// exposes) and that asymmetry is what the per-shape ratios on microsecond cases
/// actually report. Plan construction is microseconds, so the clock's own ~25 ns
/// resolution is a few percent at worst, and it costs `reps` constructions per arm.
pub fn timed_prepare<T>(reps: usize, mut prepare: impl FnMut() -> T) -> (T, f64) {
    let mut keep = None;
    let mut best = f64::INFINITY;
    for _ in 0..reps.max(1) {
        let t = Instant::now();
        let v = prepare();
        best = best.min(t.elapsed().as_secs_f64());
        keep = Some(v);
    }
    (keep.expect("at least one preparation"), best)
}

/// GFLOP/s given a multiply-accumulate count and a time.
pub fn gflops<T: Element>(macs: u64, secs: f64) -> f64 {
    (macs as f64) * (<T as Element>::FLOPS_PER_MAC as f64) / secs / 1e9
}

/// Relative Frobenius error `||got - want|| / ||want||`.
pub fn rel_error<T: Element>(got: &[T], want: &[T]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for (&g, &w) in got.iter().zip(want) {
        num += Element::sub(g, w).norm().powi(2);
        den += Element::norm(w).powi(2);
    }
    if den == 0.0 {
        num.sqrt()
    } else {
        (num / den).sqrt()
    }
}

/// Set provider budgets, and confirm the linked TBLIS ABI.
pub fn configure_threads(threads: usize) {
    let _ = threads;
    #[cfg(feature = "tblis")]
    unsafe {
        crate::tblis::tblis_set_num_threads(
            threads.try_into().expect("TBLIS thread count fits u32"),
        );
        assert_eq!(crate::tblis::tblis_get_num_threads() as usize, threads);
        if let Err(e) = crate::tblis::verify_type_tags() {
            eprintln!("FATAL: {e}");
            std::process::exit(2);
        }
    };
    // Accelerate has no equivalent: it reads `VECLIB_MAXIMUM_THREADS` once at
    // first use, so pinning it is the caller's job and cannot be done from here.
    #[cfg(all(feature = "blas", not(feature = "accelerate")))]
    unsafe {
        crate::blas::openblas_set_num_threads(
            threads.try_into().expect("BLAS thread count fits i32"),
        )
    };
}
