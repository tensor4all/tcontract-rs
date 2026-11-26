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
//!   ([`ScatterMap::block_stride`]): `run_stride`'s verdict
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

/// Largest tile [`ScatterMap::fill`] builds on the stack (8 KiB), and the smallest
/// that is worth building.
const TILE_MAX: usize = 1024;
const TILE_MIN: usize = 8;
/// Longest period [`ScatterMap::append_block_scatter`] computes directly.
const PERIOD_MAX: usize = 1 << 16;

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
/// any number of elements. See the module documentation.
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
        // A window at the head (every window of a call that fits one cache block)
        // has all digits zero: nothing to divide.
        if lo != 0 {
            for (k, a) in self.axes.iter().enumerate() {
                let d = (lo / a.radix) % a.extent;
                digits[k] = d;
                off = off.wrapping_add((d as i64).wrapping_mul(a.stride));
            }
        }
        if let Some((k_low, tile_len)) = self.tile_split(n) {
            self.fill_tiled(lo, out, digits, k_low, tile_len);
            return;
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

    /// How many low axes form the tile of a [`fill`](Self::fill) of `n` entries,
    /// and the tile's length (their extents' product): the most axes whose product
    /// fits `min(n, TILE_MAX)`. `None` when that is fewer than two axes or fewer
    /// than `TILE_MIN` entries, where axis 0's ramp is already long enough.
    pub(crate) fn tile_split(&self, n: usize) -> Option<(usize, usize)> {
        let cap = n.min(TILE_MAX);
        let mut len = 1usize;
        let mut k = 0;
        for a in self.axes.iter() {
            match len.checked_mul(a.extent) {
                Some(l) if l <= cap => {
                    len = l;
                    k += 1;
                }
                _ => break,
            }
        }
        (k >= 2 && len >= TILE_MIN).then_some((k, len))
    }

    /// [`fill`](Self::fill) for a map whose low axes are short (a qubit axis has
    /// extent 2, so a ramp along it is two entries and the carry logic would run
    /// per pair of entries).
    ///
    /// The offsets of the first `k_low` axes form a tile of `tile_len` entries,
    /// built by doubling (`tile[d * radix + i] = tile[i] + d * stride`), which is
    /// plain vector arithmetic. Every segment of `out` is then a slice of the tile
    /// plus the offset of the digits above it, and crossing the tile's end is one
    /// carry through the higher axes. `digits` holds every axis' digit of `lo`.
    fn fill_tiled(
        &self,
        lo: usize,
        out: &mut [i64],
        digits: &mut [usize],
        k_low: usize,
        tile_len: usize,
    ) {
        let mut tile = [0i64; TILE_MAX];
        let tile = &mut tile[..tile_len];
        let mut filled = 1usize;
        for a in &self.axes[..k_low] {
            for d in 1..a.extent {
                let add = (d as i64).wrapping_mul(a.stride);
                let (src, dst) = tile.split_at_mut(filled * d);
                for (o, &t) in dst[..filled].iter_mut().zip(&src[..filled]) {
                    *o = t.wrapping_add(add);
                }
            }
            filled *= a.extent;
        }
        debug_assert_eq!(filled, tile_len);
        // Offset of the digits above the tile.
        let mut hi = 0i64;
        for (a, &d) in self.axes[k_low..].iter().zip(&digits[k_low..]) {
            hi = hi.wrapping_add((d as i64).wrapping_mul(a.stride));
        }
        let (n, mut i, mut t) = (out.len(), 0usize, lo % tile_len);
        loop {
            let run = (tile_len - t).min(n - i);
            for (o, &v) in out[i..i + run].iter_mut().zip(&tile[t..t + run]) {
                *o = v.wrapping_add(hi);
            }
            i += run;
            if i == n {
                return;
            }
            t = 0;
            // The tile wrapped: step the digits above it.
            let mut k = k_low;
            loop {
                let a = &self.axes[k];
                digits[k] += 1;
                if digits[k] < a.extent {
                    hi = hi.wrapping_add(a.stride);
                    break;
                }
                hi = hi.wrapping_sub(((a.extent - 1) as i64).wrapping_mul(a.stride));
                digits[k] = 0;
                k += 1;
            }
        }
    }

    /// The expanded vector, one `i64` per entry: [`build_scatter`](super::build_scatter)
    /// of the same group. `O(len)` memory -- for tests and diagnostics that want
    /// the table, never for a plan.
    pub fn expand(&self) -> Vec<i64> {
        let mut out = vec![0i64; self.len];
        self.fill(0, &mut out);
        out
    }

    /// `out[t] = self.at(idx[t])` for arbitrary (not necessarily consecutive)
    /// indices, such as a blocked traversal's row membership. One division per
    /// axis per index: this is the lookup form, not the sequential gather.
    ///
    /// # Panics
    ///
    /// Panics if `out` and `idx` differ in length; (debug) on an index past `len()`.
    pub fn gather(&self, idx: &[i64], out: &mut [i64]) {
        assert_eq!(idx.len(), out.len(), "one offset per index");
        for (o, &i) in out.iter_mut().zip(idx) {
            *o = self.at(i as usize);
        }
    }

    /// Physical stride of the digit whose place value is `place` (the product of
    /// the lower extents, as in a role's logical row numbering): the offset change
    /// when that digit steps by one without a carry. `0` when no axis has that
    /// place value, which is an extent-1 axis (dropped on construction).
    pub fn place_stride(&self, place: usize) -> i64 {
        self.axes
            .iter()
            .find(|a| a.radix == place)
            .map_or(0, |a| a.stride)
    }

    /// Entries of a two-axis block of logical rows, with no division per entry.
    ///
    /// Row `(s, l)` is `base + s * outer.0 + l * inner.0` where `outer.0` and
    /// `inner.0` are the place values of two distinct axes and the block stays
    /// inside both (`digit + count <= extent`, which is what a block of an axis-aware
    /// traversal guarantees). Entry `s * inner.1 + l` of `out` is its offset; only
    /// `out.len() <= outer.1 * inner.1` entries are written.
    ///
    /// The head's digits are converted once ([`at`](Self::at)); every other entry
    /// is that offset plus `s * outer_stride + l * inner_stride`, since a step
    /// that does not carry changes the offset by the axis' stride alone.
    ///
    /// # Panics
    ///
    /// Panics if `out` is longer than the block or the head is out of range.
    pub fn fill_grid(
        &self,
        base: usize,
        outer: (usize, usize),
        inner: (usize, usize),
        out: &mut [i64],
    ) {
        assert!(out.len() <= outer.1 * inner.1, "block shorter than `out`");
        if out.is_empty() {
            return;
        }
        let head = self.at(base);
        let (so, si) = (self.place_stride(outer.0), self.place_stride(inner.0));
        let (mut s, mut l) = (0usize, 0usize);
        let mut row = head;
        for o in out.iter_mut() {
            *o = row.wrapping_add((l as i64).wrapping_mul(si));
            l += 1;
            if l == inner.1 {
                l = 0;
                s += 1;
                row = head.wrapping_add((s as i64).wrapping_mul(so));
            }
        }
    }

    /// The common difference of entries `lo .. hi`, or [`IRREGULAR`];
    /// `0` for a run of at most one entry. `run_stride`
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
            None => match self.period_for(blk) {
                // A block's verdict is a common difference, which a constant
                // shift does not change; and the entries of one tile repeat,
                // shifted, for every setting of the digits above it. So the
                // verdicts of the first tile are the verdicts of all of them.
                Some(period) => {
                    for b in 0..period / blk {
                        let lo = b * blk;
                        out.push(self.block_stride(lo, lo + blk));
                    }
                    let first = out.len() - period / blk;
                    for _ in 1..self.len / period {
                        out.extend_from_within(first..first + period / blk);
                    }
                }
                // No tile holds whole blocks: fill chunks of whole blocks and
                // read each block's stride off the entries.
                None => {
                    const CHUNK_BLOCKS: usize = 64;
                    let step = blk * CHUNK_BLOCKS;
                    let mut buf = vec![0i64; step.min(self.len)];
                    let mut lo = 0;
                    while lo < self.len {
                        let n = step.min(self.len - lo);
                        self.fill(lo, &mut buf[..n]);
                        out.extend(buf[..n].chunks(blk).map(super::run_stride));
                        lo += n;
                    }
                }
            },
        }
        (start, out.len())
    }

    /// The smallest product of a prefix of the extents that `blk` divides and that
    /// is at most [`PERIOD_MAX`]: the length after which the entries repeat, shifted.
    /// `None` when there is none, or it is the whole map (nothing to repeat).
    pub(crate) fn period_for(&self, blk: usize) -> Option<usize> {
        let mut radix = 1usize;
        for a in self.axes.iter() {
            radix = radix.checked_mul(a.extent)?;
            if radix > PERIOD_MAX {
                return None;
            }
            if radix.is_multiple_of(blk) {
                return (radix < self.len).then_some(radix);
            }
        }
        None
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

    /// [`regular_fraction`](super::regular_fraction) of the block-scatter vector at
    /// `blk`, counted block by block without materializing it.
    ///
    /// # Panics
    ///
    /// Panics if `blk` is 0.
    pub fn regular_fraction(&self, blk: usize) -> f64 {
        assert!(blk > 0);
        if self.block_scatter_regular(blk) {
            return 1.0;
        }
        let nblk = self.len.div_ceil(blk);
        let regular = (0..nblk)
            .filter(|&b| {
                let lo = b * blk;
                self.block_stride(lo, (lo + blk).min(self.len)) != IRREGULAR
            })
            .count();
        regular as f64 / nblk as f64
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

/// Axes a cursor keeps on the stack.
const CURSOR_INLINE: usize = 32;

/// A position in a [`ScatterMap`] that steps to the next entry in `O(1)`
/// amortized, without a division: the offsets of a loop over consecutive
/// indices, such as the batch entries of a contraction.
///
/// Created at an arbitrary index (one division per axis, once), then
/// [`advance`](Self::advance)d by the carry deltas.
#[derive(Clone, Debug)]
pub struct ScatterCursor<'a> {
    map: &'a ScatterMap,
    off: i64,
    inline: [usize; CURSOR_INLINE],
    // Allocated only for a map of rank above `CURSOR_INLINE`.
    spill: Vec<usize>,
}

impl ScatterMap {
    /// A cursor at entry `i` (`i <= len()`: one past the end is a valid, spent
    /// position, which [`offset`](ScatterCursor::offset) must not be asked about).
    pub fn cursor(&self, i: usize) -> ScatterCursor<'_> {
        debug_assert!(i <= self.len);
        let mut cur = ScatterCursor {
            map: self,
            off: 0,
            inline: [0; CURSOR_INLINE],
            spill: Vec::new(),
        };
        if self.axes.len() > CURSOR_INLINE {
            cur.spill = vec![0; self.axes.len()];
        }
        if i != 0 && i < self.len {
            let mut off = 0i64;
            let digits = cur.digits_mut();
            for (k, a) in self.axes.iter().enumerate() {
                let d = (i / a.radix) % a.extent;
                digits[k] = d;
                off = off.wrapping_add((d as i64).wrapping_mul(a.stride));
            }
            cur.off = off;
        }
        cur
    }
}

impl ScatterCursor<'_> {
    fn digits_mut(&mut self) -> &mut [usize] {
        let rank = self.map.axes.len();
        if rank <= CURSOR_INLINE {
            &mut self.inline[..rank]
        } else {
            &mut self.spill
        }
    }

    /// The offset at the current entry.
    #[inline]
    pub fn offset(&self) -> i64 {
        self.off
    }

    /// Step to the next entry. Stepping past the last entry leaves the cursor
    /// spent; its offset is then meaningless.
    #[inline]
    pub fn advance(&mut self) {
        let map = self.map;
        let off = self.off;
        let digits = self.digits_mut();
        for (k, a) in map.axes.iter().enumerate() {
            digits[k] += 1;
            if digits[k] < a.extent {
                self.off = off.wrapping_add(a.delta);
                return;
            }
            digits[k] = 0;
        }
        // Every digit wrapped: past the end (or a rank-0 map).
    }
}
