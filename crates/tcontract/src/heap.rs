//! Heap bytes owned by plan metadata, for [`Plan::retained_bytes`](crate::Plan::retained_bytes).
//!
//! Each count is the size of the allocation the container holds (its capacity
//! times the element size), not its length, so a caller charging a byte budget
//! sees what the allocator handed out.

use core::mem::size_of;

use strided_basic::RAW_FUSED_RANK_LIMIT;
use tinyvec::{Array, TinyVec};

/// The heap allocation of `v`.
pub(crate) fn vec<T>(v: &Vec<T>) -> usize {
    v.capacity() * size_of::<T>()
}

/// The heap allocation of `v`: zero while it is inline.
pub(crate) fn tiny<A: Array>(v: &TinyVec<A>) -> usize {
    if v.is_heap() {
        v.capacity() * size_of::<A::Item>()
    } else {
        0
    }
}

/// The heap a `strided_basic::UpdatePlan` compiled from `(dims, dst, src)` owns.
///
/// INVARIANT: strided-basic 0.9 does not report this, so it is derived from
/// `UpdatePlan::compile` and `fuse_pair_traversal` of that release: the three
/// layout vectors are rank-`INLINE_RANK` `TinyVec`s built from slices (exact
/// length once spilled), and the fused traversal keeps its rank-length scratch
/// only when the fused rank exceeds `RAW_FUSED_RANK_LIMIT`. The exact-equality
/// test in `tests/plan_retained_bytes.rs` covers both spill cases and fails if
/// a strided-basic update changes either.
pub(crate) fn update_plan(dims: &[usize], dst: &[isize], src: &[isize]) -> usize {
    let n = dims.len();
    let layout = if n > strided_view::INLINE_RANK {
        n * (size_of::<usize>() + 2 * size_of::<isize>())
    } else {
        0
    };
    let traversal = if n > RAW_FUSED_RANK_LIMIT && fused_rank(dims, dst, src) > RAW_FUSED_RANK_LIMIT
    {
        // One `FusedAxis { dim: usize, dst: isize, src: isize }` per input axis.
        n * (size_of::<usize>() + 2 * size_of::<isize>())
    } else {
        0
    };
    layout + traversal
}

/// The rank `strided_basic`'s pair fusion reaches: unit axes dropped, the rest
/// ordered by destination stride (stable), and an axis merged into its
/// predecessor when both strides continue it.
fn fused_rank(dims: &[usize], dst: &[isize], src: &[isize]) -> usize {
    if dims.contains(&0) {
        return 1;
    }
    let mut axes: Vec<(usize, isize, isize)> = (0..dims.len())
        .filter(|&i| dims[i] != 1)
        .map(|i| (dims[i], dst[i], src[i]))
        .collect();
    axes.sort_by_key(|a| a.1);
    let Some(&first) = axes.first() else {
        return 1;
    };
    let (mut rank, mut last) = (1, first);
    for &a in &axes[1..] {
        let merged = isize::try_from(last.0)
            .ok()
            .filter(|&e| last.1.checked_mul(e) == Some(a.1) && last.2.checked_mul(e) == Some(a.2))
            .and_then(|_| last.0.checked_mul(a.0));
        match merged {
            Some(m) => last.0 = m,
            None => {
                rank += 1;
                last = a;
            }
        }
    }
    rank
}
