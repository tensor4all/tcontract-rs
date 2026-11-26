//! The scatter of a mixed-radix index group as its *structure*, not its expansion.
//!
//! [`build_scatter`](super::build_scatter) lays a group out as one `i64` per
//! element. That is `O(elements)` memory whether the entries are strides or
//! [`IRREGULAR`] -- 7.25 GiB per table on a 30-qubit circuit (tcontract-rs#221) --
//! and everything a plan needs from it is a function of the axis extents and
//! strides. A [`ScatterMap`] keeps exactly that, `O(rank)`.
//!
//! # Carry deltas
//!
//! Entry `i` of the scatter is `sum_k digit_k(i) * stride_k`, with
//! `digit_k(i) = (i / radix_k) % extent_k` and `radix_k = prod_{j<k} extent_j`.
//! Stepping from `i - 1` to `i` changes the offset by a quantity that depends
//! only on the *carry level* `k` of `i` -- the largest `k` with `radix_k | i`,
//! i.e. how many low digits wrapped:
//!
//! ```text
//! delta_k = stride_k - sum_{j<k} (extent_j - 1) * stride_j
//! ```
//!
//! Three things follow, none of which needs the expansion:
//!
//! * **Block regularity.** The steps inside the block `[lo, hi)` are those of
//!   `i` in `lo+1 ..= hi-1`. Level `k` occurs among them iff some `i` in that
//!   range is a multiple of `radix_k` but not of `radix_{k+1}`, which is two
//!   integer divisions per level. The block is an arithmetic progression iff
//!   the deltas of the levels that occur are all equal
//!   ([`ScatterMap::block_stride`]): [`run_stride`](super::run_stride)'s verdict
//!   at `O(rank)` instead of `O(block)`.
//! * **Run structure.** The first level `k >= 1` whose delta differs from
//!   `delta_0` ends the first run, at `radix_k` entries, and the vector is a
//!   concatenation of equal maximal runs iff every level from there on differs
//!   from `delta_0` too ([`ScatterMap::run_structure`]).
//! * **Gather.** [`ScatterMap::fill`] converts the first index to digits once,
//!   writes the rest of the current axis-0 run as a ramp, and advances across a
//!   carry by adding that level's delta: no division per element.
//!
//! The arithmetic wraps, exactly as the odometer and the single-axis ramp of
//! `build_scatter` do, so every value agrees with the expanded vector even for
//! strides no real layout has.

use super::IRREGULAR;

/// One axis of a [`ScatterMap`], with its derived constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MapAxis {
    extent: usize,
    stride: i64,
    /// Offset change when this axis' digit increments and every lower digit wraps.
    delta: i64,
    /// Product of the lower extents: the index distance of one step of this digit.
    radix: usize,
}

/// A mixed-radix scatter described by its axes: `O(rank)` memory for a group of
/// any number of elements. See the [module documentation](self).
///
/// Axes of extent 1 contribute nothing to any offset and are dropped on
/// construction, so two maps are equal exactly when they describe the same
/// entries.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ScatterMap {
    axes: Box<[MapAxis]>,
    len: usize,
}

impl ScatterMap {
    /// The map of a mixed-radix group, `extents`/`strides` fastest-varying first.
    ///
    /// The group has `extents.iter().product()` entries (1 for an empty group).
    /// A zero extent, or a product that overflows `usize`, gives an empty map,
    /// the cases `build_scatter` answers with an empty vector.
    ///
    /// `extents` and `strides` must have the same length (`debug_assert`); a
    /// longer `strides` has its tail ignored, as in `build_scatter`.
    ///
    /// ```
    /// use tcontract_kernel::scatter::{build_scatter, ScatterMap};
    /// let map = ScatterMap::new(&[2, 3], &[1, 10]);
    /// assert_eq!(map.len(), 6);
    /// let mut out = [0i64; 6];
    /// map.fill(0, &mut out);
    /// assert_eq!(out.to_vec(), build_scatter(&[2, 3], &[1, 10]));
    /// ```
    pub fn new(extents: &[i64], strides: &[i64]) -> ScatterMap {
        debug_assert_eq!(extents.len(), strides.len());
        let mut len: usize = 1;
        let mut axes = Vec::with_capacity(extents.len());
        let mut lower_span = 0i64; // sum_{j<k} (extent_j - 1) * stride_j, wrapping
        for (&extent, &stride) in extents.iter().zip(strides) {
            if extent == 0 {
                return ScatterMap::empty();
            }
            if extent < 0 {
                // Not a real extent; `build_scatter` would not have produced a
                // vector either.
                return ScatterMap::empty();
            }
            if extent == 1 {
                continue;
            }
            let extent = extent as usize;
            axes.push(MapAxis {
                extent,
                stride,
                delta: stride.wrapping_sub(lower_span),
                radix: len,
            });
            lower_span = lower_span.wrapping_add(((extent - 1) as i64).wrapping_mul(stride));
            len = match len.checked_mul(extent) {
                Some(l) => l,
                None => return ScatterMap::empty(),
            };
        }
        ScatterMap {
            axes: axes.into_boxed_slice(),
            len,
        }
    }

    fn empty() -> ScatterMap {
        ScatterMap {
            axes: Box::new([]),
            len: 0,
        }
    }

    /// Number of entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the group has no entry.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Axes that carry an offset (extent 1 dropped).
    pub fn rank(&self) -> usize {
        self.axes.len()
    }

    /// Heap bytes this map owns, at capacity.
    pub fn heap_bytes(&self) -> usize {
        self.axes.len() * core::mem::size_of::<MapAxis>()
    }

    /// Entry `i`. One division per axis; for a head or a single lookup, not for
    /// a loop over entries -- use [`fill`](Self::fill) for that.
    ///
    /// # Panics
    ///
    /// Panics (debug) if `i >= len()`.
    pub fn at(&self, i: usize) -> i64 {
        debug_assert!(i < self.len, "scatter index {i} out of {}", self.len);
        let mut off = 0i64;
        for a in self.axes.iter() {
            let d = (i / a.radix) % a.extent;
            off = off.wrapping_add((d as i64).wrapping_mul(a.stride));
        }
        off
    }

    /// Write entries `lo .. lo + out.len()` into `out`.
    ///
    /// The digits of `lo` are computed once; after that the offset advances by
    /// a ramp along axis 0 and by one carry delta per higher-axis step.
    ///
    /// # Panics
    ///
    /// Panics if the range exceeds `len()`.
    pub fn fill(&self, lo: usize, out: &mut [i64]) {
        let n = out.len();
        if n == 0 {
            return;
        }
        assert!(
            lo.checked_add(n).is_some_and(|hi| hi <= self.len),
            "scatter range {lo}+{n} out of {}",
            self.len
        );
        let rank = self.axes.len();
        if rank == 0 {
            out.fill(0);
            return;
        }
        // Digits above axis 0, on the stack unless the rank is unusually large.
        const STACK: usize = 32;
        let mut stack = [0usize; STACK];
        let mut heap;
        let digits: &mut [usize] = if rank <= STACK {
            &mut stack[..rank]
        } else {
            heap = vec![0usize; rank];
            &mut heap
        };
        let mut off = 0i64;
        for (k, a) in self.axes.iter().enumerate() {
            let d = (lo / a.radix) % a.extent;
            digits[k] = d;
            off = off.wrapping_add((d as i64).wrapping_mul(a.stride));
        }
        let a0 = self.axes[0];
        let mut i = 0;
        loop {
            // The rest of the current axis-0 run is a ramp.
            let run = (a0.extent - digits[0]).min(n - i);
            for (t, o) in out[i..i + run].iter_mut().enumerate() {
                *o = off.wrapping_add((t as i64).wrapping_mul(a0.stride));
            }
            i += run;
            if i == n {
                return;
            }
            // Axis 0 wrapped: carry into the first higher digit with room.
            let last = off.wrapping_add(((run - 1) as i64).wrapping_mul(a0.stride));
            let mut k = 1;
            loop {
                digits[k] += 1;
                if digits[k] < self.axes[k].extent {
                    break;
                }
                digits[k] = 0;
                k += 1;
            }
            digits[0] = 0;
            off = last.wrapping_add(self.axes[k].delta);
        }
    }

    /// The common difference of entries `lo .. hi`, or [`IRREGULAR`];
    /// `0` for a run of at most one entry. [`run_stride`](super::run_stride)
    /// of the expanded block, at `O(rank)`.
    ///
    /// # Panics
    ///
    /// Panics (debug) if `lo > hi` or `hi > len()`.
    pub fn block_stride(&self, lo: usize, hi: usize) -> i64 {
        debug_assert!(lo <= hi && hi <= self.len);
        if hi - lo <= 1 {
            return 0;
        }
        // Steps into entries `a ..= b`.
        let (a, b) = (lo + 1, hi - 1);
        let mut common: Option<i64> = None;
        let rank = self.axes.len();
        // Multiples of `radix` in `a ..= b`.
        let multiples = |radix: usize| b / radix - (a - 1) / radix;
        let mut here = b - (a - 1); // radix 1: every step
        for (k, ax) in self.axes.iter().enumerate() {
            let above = if k + 1 < rank {
                multiples(self.axes[k + 1].radix)
            } else {
                // `radix_rank` is `len`, and `b < len`.
                0
            };
            if here > above {
                match common {
                    None => common = Some(ax.delta),
                    Some(c) if c == ax.delta => {}
                    Some(_) => return IRREGULAR,
                }
            }
            here = above;
        }
        common.unwrap_or(0)
    }

    /// Append the block-scatter vector at block size `blk` to `out`, returning
    /// the `(start, end)` of what was appended. Entry `b` is
    /// [`block_stride`](Self::block_stride) of `[b*blk, min((b+1)*blk, len))`.
    ///
    /// This is [`append_block_scatter`](super::append_block_scatter) of the
    /// expanded vector, without the expansion.
    ///
    /// # Panics
    ///
    /// Panics if `blk` is 0.
    pub fn append_block_scatter(&self, out: &mut Vec<i64>, blk: usize) -> (usize, usize) {
        assert!(blk > 0);
        let start = out.len();
        let nblk = self.len.div_ceil(blk);
        out.reserve(nblk);
        // When blocks cannot straddle an axis-0 run, every block steps by it.
        let uniform = match self.axes.first() {
            None => Some(0),
            Some(a0) if a0.extent % blk == 0 => Some(a0.stride),
            Some(a0) if self.axes.len() == 1 => Some(a0.stride),
            _ => None,
        };
        match uniform {
            Some(stride) => {
                let full = self.len / blk;
                out.resize(start + full, if blk <= 1 { 0 } else { stride });
                if full < nblk {
                    // A ragged tail of `len % blk` entries.
                    out.push(self.block_stride(full * blk, self.len));
                }
            }
            None => {
                for b in 0..nblk {
                    let lo = b * blk;
                    out.push(self.block_stride(lo, (lo + blk).min(self.len)));
                }
            }
        }
        (start, out.len())
    }

    /// [`block_scatter_regular`](super::block_scatter_regular) without the
    /// expansion: whether every `blk`-block is an arithmetic progression.
    ///
    /// # Panics
    ///
    /// Panics if `blk` is 0.
    pub fn block_scatter_regular(&self, blk: usize) -> bool {
        assert!(blk > 0);
        match self.axes.first() {
            // Nothing to break: at most one entry.
            None => return true,
            // Blocks sit inside one axis-0 run, or the one axis is a ramp.
            Some(a0) if self.axes.len() == 1 || a0.extent % blk == 0 => return true,
            Some(_) => {}
        }
        (0..self.len.div_ceil(blk)).all(|b| {
            let lo = b * blk;
            self.block_stride(lo, (lo + blk).min(self.len)) != IRREGULAR
        })
    }

    /// Whether consecutive entries always differ by exactly one (a contiguous
    /// ramp), the condition `pack_b_needed` puts on `B`'s `k` scatter.
    pub fn is_unit_ramp(&self) -> bool {
        self.len <= 1 || self.block_stride(0, self.len) == 1
    }

    /// [`run_structure`](super::run_structure) of the expanded vector:
    /// `Some((len, stride))` for a concatenation of equal-length *maximal*
    /// arithmetic runs, `None` otherwise. `O(rank)`.
    pub fn run_structure(&self) -> Option<(usize, i64)> {
        if self.len < 2 {
            return None;
        }
        let stride = self.axes[0].delta;
        // The first carry level whose step breaks the progression ends the first
        // run, at `radix` entries.
        match self.axes.iter().position(|a| a.delta != stride) {
            None => Some((self.len, stride)),
            Some(k) => {
                // Every later level restarts a run too, or maximality fails.
                if self.axes[k..].iter().any(|a| a.delta == stride) {
                    None
                } else {
                    Some((self.axes[k].radix, stride))
                }
            }
        }
    }
}
