//! Shared plumbing for the harness: element traits, timing and the per-engine
//! runners.

pub mod run;
pub mod verify;

use std::time::Instant;

use num_complex::Complex;
use rand::Rng;
use tcontract::api::{CSpec, Labels, LayoutSpec, OperandSpec, Problem, Scalar};
use tcontract_kernel::Element;

use crate::corpus::{col_major_strides, label_ids, Program, Step};

#[cfg(feature = "tblis")]
use std::os::raw::c_int;

/// An element type the harness can drive through every engine.
pub trait BenchElem: Scalar {
    const NAME: &'static str;
    fn sample(rng: &mut impl Rng) -> Self;
    /// Multiply by a real scalar: the MPS chain's inputs are scaled.
    fn real_scaled(self, s: f64) -> Self;
    #[cfg(feature = "tblis")]
    const TBLIS_TYPE: c_int;
    #[cfg(feature = "tblis")]
    fn tblis_scalar(v: f64) -> crate::tblis::tblis_scalar;
}

impl BenchElem for f64 {
    const NAME: &'static str = "f64";
    fn sample(rng: &mut impl Rng) -> Self {
        rng.gen_range(-1.0..1.0)
    }
    fn real_scaled(self, s: f64) -> Self {
        self * s
    }
    #[cfg(feature = "tblis")]
    const TBLIS_TYPE: c_int = crate::tblis::TYPE_DOUBLE;
    #[cfg(feature = "tblis")]
    fn tblis_scalar(v: f64) -> crate::tblis::tblis_scalar {
        crate::tblis::tblis_scalar::f64(v)
    }
}

impl BenchElem for Complex<f64> {
    const NAME: &'static str = "c64";
    fn sample(rng: &mut impl Rng) -> Self {
        Complex::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0))
    }
    fn real_scaled(self, s: f64) -> Self {
        Complex::new(self.re * s, self.im * s)
    }
    #[cfg(feature = "tblis")]
    const TBLIS_TYPE: c_int = crate::tblis::TYPE_DCOMPLEX;
    #[cfg(feature = "tblis")]
    fn tblis_scalar(v: f64) -> crate::tblis::tblis_scalar {
        crate::tblis::tblis_scalar::c64(v, 0.0)
    }
}

fn operand(dims: &[usize]) -> OperandSpec {
    let strides = col_major_strides(dims);
    OperandSpec::new(LayoutSpec::new(dims, &strides, 0).expect("corpus layouts are valid"))
}

/// The validated problem of step `k` in storage type `T`: `D` is overwritten
/// (`beta = 0`), so there is no `C`.
pub fn problem_of<T: BenchElem>(p: &Program, k: usize) -> Result<Problem, tcontract::Error> {
    let st = &p.steps[k];
    Problem::from_labels(
        T::STORAGE,
        operand(&p.shapes[st.lhs]),
        operand(&p.shapes[st.rhs]),
        CSpec::Absent,
        operand(&p.shapes[p.n_in() + k]),
        &Labels::new(&label_ids(&st.a), &label_ids(&st.b), &label_ids(&st.d)),
    )
}

/// Run a program's steps in order, handing each step's two input slots (already
/// written by earlier steps, or an input tensor) and a mutable output slot to
/// `f`. Each step writes a slot no later step has read yet, so an engine that
/// keeps the intermediates preallocated reuses this one loop.
pub fn steps_with<T, F>(p: &Program, inputs: &[Vec<T>], slots: &mut [Vec<T>], mut f: F)
where
    F: FnMut(usize, &Step, &[T], &[T], &mut [T]),
{
    let n_in = p.n_in();
    for (k, st) in p.steps.iter().enumerate() {
        let (done, rest) = slots.split_at_mut(k);
        let get = |i: usize| -> &[T] {
            if i < n_in {
                &inputs[i]
            } else {
                &done[i - n_in]
            }
        };
        f(k, st, get(st.lhs), get(st.rhs), &mut rest[0]);
    }
}

/// Sample every input tensor of a program once, with the corpus's MPS scaling.
pub fn sample_inputs<T: BenchElem>(p: &Program, seed: u64) -> Vec<Vec<T>> {
    use rand::SeedableRng as _;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    p.inputs
        .iter()
        .map(|d| {
            (0..crate::corpus::numel(d))
                .map(|_| T::sample(&mut rng).real_scaled(p.scale))
                .collect()
        })
        .collect()
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

/// Maximum relative error `max|got - want| / max|want|` — the metric the
/// experiment of #61 checked every arm with. A norm ratio rather than the
/// Frobenius relative error `tcbench` uses, because these programs are chains
/// and a per-step residual is not what the corpus records.
pub fn max_rel_err<T: Element>(got: &[T], want: &[T]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want
        .iter()
        .map(|v| v.norm())
        .fold(0.0f64, f64::max)
        .max(f64::MIN_POSITIVE);
    got.iter()
        .zip(want)
        .map(|(&g, &w)| Element::sub(g, w).norm())
        .fold(0.0f64, f64::max)
        / scale
}

/// Tolerance for [`max_rel_err`], by the real type's width.
pub fn tol<T: Element>() -> f64 {
    if core::mem::size_of::<T::Real>() == 4 {
        1e-3
    } else {
        1e-10
    }
}

/// Set provider budgets and confirm the linked TBLIS ABI.
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
}
