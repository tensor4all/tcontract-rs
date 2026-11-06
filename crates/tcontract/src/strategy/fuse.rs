//! The role-group fusion both non-packed strategies plan with.
//!
//! A contraction's labels fall into four role groups (M, N, K, H); a group fuses
//! when its axes sit at one regular stride in every operand that carries them,
//! which is what makes the group addressable as one `(extent, stride)` per
//! operand instead of an index tuple.

use crate::api::{fuse_axes, sort_by_stride, OperandId, RoleAxis};

/// One fused group: `(extent, stride)` in each of A, B and D.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Fused {
    pub(crate) extent: usize,
    pub(crate) a: isize,
    pub(crate) b: isize,
    pub(crate) d: isize,
}

/// Run `f` on `0..n` in a stack buffer when `n` is small (planning allocates
/// nothing for the usual ranks) and in a heap buffer otherwise.
fn with_indices<R>(n: usize, f: impl FnOnce(&mut [usize]) -> R) -> R {
    const INLINE: usize = 16;
    if n <= INLINE {
        let mut buf = [0usize; INLINE];
        let order = &mut buf[..n];
        order.iter_mut().enumerate().for_each(|(i, x)| *x = i);
        f(order)
    } else {
        f(&mut (0..n).collect::<Vec<_>>())
    }
}

/// Fuse one group over the operands that carry it, trying the order each
/// carrier would choose and then the natural one. The candidates are tried
/// one at a time in one reused buffer.
pub(crate) fn fuse_group(group: &[RoleAxis], carriers: &[OperandId]) -> Option<Fused> {
    let fits = |order: &[usize]| {
        carriers
            .iter()
            .all(|&o| fuse_axes(group, order, o).is_some())
    };
    let fused = |order: &[usize]| {
        let get = |o: OperandId| fuse_axes(group, order, o);
        let (extent, a) = get(OperandId::A).unwrap_or((1, 1));
        // Every carrier agrees on the extent; the absent operands keep stride one.
        Fused {
            extent,
            a,
            b: get(OperandId::B).map_or(1, |x| x.1),
            d: get(OperandId::D).map_or(1, |x| x.1),
        }
    };
    with_indices(group.len(), |order| {
        let reset = |order: &mut [usize]| order.iter_mut().enumerate().for_each(|(i, x)| *x = i);
        for &o in carriers {
            // From the natural order, stable: equal strides keep that order.
            reset(order);
            sort_by_stride(group, o, order);
            if fits(order) {
                return Some(fused(order));
            }
        }
        reset(order);
        fits(order).then(|| fused(order))
    })
}
