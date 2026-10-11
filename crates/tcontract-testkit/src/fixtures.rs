//! Seeded fixtures: deterministic data and layout helpers for tests.
//!
//! Everything here is a pure function of its arguments (a seed, an index, a
//! shape), so a failing case reproduces from its printed inputs.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tcontract_kernel::{Element, Real};

/// Deterministic operand value, distinct per index and non-trivial in both
/// parts, so a conjugation or a stride error cannot cancel out.
pub fn value<T: Element>(i: usize) -> T {
    let r = T::Real::from_f64((i as f64 + 1.0) * 0.13);
    let im = T::Real::from_f64((i as f64 - 2.0) * 0.07);
    T::from_parts(r, if T::IS_COMPLEX { im } else { T::Real::ZERO })
}

/// `len` seeded values, uniform in `[-1, 1]` in each part.
pub fn seeded<T: Element>(seed: u64, len: usize) -> Vec<T> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..len)
        .map(|_| {
            let re = T::Real::from_f64(rng.gen_range(-1.0..=1.0));
            let im = T::Real::from_f64(rng.gen_range(-1.0..=1.0));
            T::from_parts(re, im)
        })
        .collect()
}

/// `len` small integers as values, so a sum of products is exact in floating
/// point and an equality check needs no tolerance.
pub fn small_ints<T: Element>(seed: usize, len: usize) -> Vec<T> {
    (0..len)
        .map(|x| {
            T::from_parts(
                T::Real::from_f64(((x * 5 + seed * 3) % 13) as f64 - 6.0),
                T::Real::ZERO,
            )
        })
        .collect()
}

/// Column-major element strides of `dims` (the first axis is fastest).
pub fn col_major(dims: &[usize]) -> Vec<isize> {
    let mut s = Vec::with_capacity(dims.len());
    let mut acc = 1isize;
    for &d in dims {
        s.push(acc);
        acc *= d.max(1) as isize;
    }
    s
}

/// Row-major element strides of `dims` (the last axis is fastest).
pub fn row_major(dims: &[usize]) -> Vec<isize> {
    let mut s = vec![0isize; dims.len()];
    let mut acc = 1isize;
    for (k, &d) in dims.iter().enumerate().rev() {
        s[k] = acc;
        acc *= d.max(1) as isize;
    }
    s
}

/// The storage length a layout needs: one past its largest addressed element,
/// for non-negative strides.
pub fn storage_len(dims: &[usize], strides: &[isize]) -> usize {
    if dims.contains(&0) {
        return 0;
    }
    1 + dims
        .iter()
        .zip(strides)
        .map(|(&d, &s)| (d - 1) * s.unsigned_abs())
        .sum::<usize>()
}

/// The largest relative error of `got` against `want`, normalised by the
/// largest magnitude in `want` (and one, so an all-zero reference is
/// meaningful).
pub fn max_rel_err<T: Element>(got: &[T], want: &[T]) -> f64 {
    let scale = want.iter().map(|x| x.norm()).fold(1.0f64, f64::max);
    got.iter()
        .zip(want)
        .map(|(g, w)| g.sub(*w).norm() / scale)
        .fold(0.0, f64::max)
}
