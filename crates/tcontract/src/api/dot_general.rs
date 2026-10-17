//! The `dot_general` front end.

use crate::api::error::{ConfigError, Error, OperandId, Result, ShapeError};
use crate::api::problem::{IndexVec, OutputVec, INLINE_RANK};
use tinyvec::TinyVec;

/// Which axes of the two operands are contracted and which are batch axes.
///
/// Free axes (all others) keep their order. The output layout is
/// `[lhs_free..., rhs_free..., batch...]`, batch axes in the listed order, and
/// `C` is `D`. Repeated *axis indices* are invalid (unlike repeated *labels* in
/// [`Labels`](crate::api::Labels), which select a diagonal).
///
/// # Examples
///
/// ```
/// use tcontract::api::DotGeneral;
/// let bmm = DotGeneral::new(&[2], &[1], &[0], &[0]);
/// assert_eq!(bmm.validate(&[3, 5, 4], &[3, 4, 2]).unwrap().out_dims.as_slice(), &[5, 2, 3]);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DotGeneral {
    lhs_contract: IndexVec,
    rhs_contract: IndexVec,
    lhs_batch: IndexVec,
    rhs_batch: IndexVec,
}

/// The axes of a validated [`DotGeneral`] against concrete extents.
///
/// # Migration (0.3)
///
/// `lhs_free`, `rhs_free` and `out_dims` are `tinyvec` inline vectors, not
/// `Vec<usize>`, so a plan build at the usual ranks allocates no axis metadata
/// (see #121). They deref to `[usize]`; read them as a slice
/// (`shape.out_dims.as_slice()`) and convert to an owned `Vec` with
/// `shape.out_dims.to_vec()` where a downstream field or API still wants one.
/// `IntoIterator`-style loops keep working through the deref, but an assignment
/// to a `Vec<usize>` site needs the explicit `.to_vec()`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DotShape {
    /// Free axes of A, in order.
    pub lhs_free: IndexVec,
    /// Free axes of B, in order.
    pub rhs_free: IndexVec,
    /// Output extents `[lhs_free, rhs_free, batch]`.
    pub out_dims: OutputVec,
}

impl DotGeneral {
    /// A configuration from axis lists. Nothing is checked until
    /// [`validate`](Self::validate) (or problem construction) sees the extents.
    pub fn new(
        lhs_contract: &[usize],
        rhs_contract: &[usize],
        lhs_batch: &[usize],
        rhs_batch: &[usize],
    ) -> Self {
        Self {
            lhs_contract: lhs_contract.iter().copied().collect(),
            rhs_contract: rhs_contract.iter().copied().collect(),
            lhs_batch: lhs_batch.iter().copied().collect(),
            rhs_batch: rhs_batch.iter().copied().collect(),
        }
    }

    /// Contracted axes of A.
    pub fn lhs_contract(&self) -> &[usize] {
        self.lhs_contract.as_slice()
    }

    /// Contracted axes of B, paired with [`lhs_contract`](Self::lhs_contract).
    pub fn rhs_contract(&self) -> &[usize] {
        self.rhs_contract.as_slice()
    }

    /// Batch axes of A.
    pub fn lhs_batch(&self) -> &[usize] {
        self.lhs_batch.as_slice()
    }

    /// Batch axes of B, paired with [`lhs_batch`](Self::lhs_batch).
    pub fn rhs_batch(&self) -> &[usize] {
        self.rhs_batch.as_slice()
    }

    /// Validate against operand extents: list lengths, axis bounds, uniqueness
    /// and disjointness, then the extents of every pair.
    ///
    /// # Errors
    ///
    /// [`ConfigError`] for unequal list lengths and repeated or out-of-range
    /// axes; [`ShapeError::PairedExtent`] for mismatched extents.
    pub fn validate(&self, a: &[usize], b: &[usize]) -> Result<DotShape> {
        for (what, l, r) in [
            ("contracting", &self.lhs_contract, &self.rhs_contract),
            ("batch", &self.lhs_batch, &self.rhs_batch),
        ] {
            if l.len() != r.len() {
                return Err(ConfigError::AxisListLength {
                    what,
                    lhs: l.len(),
                    rhs: r.len(),
                }
                .into());
            }
        }
        let check = |operand: OperandId, rank: usize, c: &[usize], bt: &[usize]| {
            // A rank-sized mask, inline for the usual ranks. `resize` spills for
            // a higher rank rather than rejecting it.
            let mut seen: TinyVec<[bool; INLINE_RANK]> = TinyVec::new();
            seen.resize(rank, false);
            for &x in c.iter().chain(bt) {
                if x >= rank {
                    return Err(Error::from(ConfigError::AxisOutOfRange {
                        operand,
                        axis: x,
                        rank,
                    }));
                }
                if seen[x] {
                    return Err(ConfigError::AxisRepeated { operand, axis: x }.into());
                }
                seen[x] = true;
            }
            Ok((0..rank).filter(|&x| !seen[x]).collect::<IndexVec>())
        };
        let lhs_free = check(OperandId::A, a.len(), &self.lhs_contract, &self.lhs_batch)?;
        let rhs_free = check(OperandId::B, b.len(), &self.rhs_contract, &self.rhs_batch)?;
        for (&l, &r) in self
            .lhs_contract
            .iter()
            .zip(self.rhs_contract.iter())
            .chain(self.lhs_batch.iter().zip(self.rhs_batch.iter()))
        {
            if a[l] != b[r] {
                return Err(ShapeError::PairedExtent {
                    lhs_axis: l,
                    rhs_axis: r,
                    lhs: a[l],
                    rhs: b[r],
                }
                .into());
            }
        }
        let out_dims = lhs_free
            .iter()
            .map(|&x| a[x])
            .chain(rhs_free.iter().map(|&x| b[x]))
            .chain(self.lhs_batch.iter().map(|&x| a[x]))
            .collect();
        Ok(DotShape {
            lhs_free,
            rhs_free,
            out_dims,
        })
    }
}
