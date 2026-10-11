//! The `dot_general` front end.

use crate::api::error::{ConfigError, Error, OperandId, Result, ShapeError};

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
/// assert_eq!(bmm.validate(&[3, 5, 4], &[3, 4, 2]).unwrap().out_dims, vec![5, 2, 3]);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DotGeneral {
    lhs_contract: Vec<usize>,
    rhs_contract: Vec<usize>,
    lhs_batch: Vec<usize>,
    rhs_batch: Vec<usize>,
}

/// The axes of a validated [`DotGeneral`] against concrete extents.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DotShape {
    /// Free axes of A, in order.
    pub lhs_free: Vec<usize>,
    /// Free axes of B, in order.
    pub rhs_free: Vec<usize>,
    /// Output extents `[lhs_free, rhs_free, batch]`.
    pub out_dims: Vec<usize>,
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
            lhs_contract: lhs_contract.to_vec(),
            rhs_contract: rhs_contract.to_vec(),
            lhs_batch: lhs_batch.to_vec(),
            rhs_batch: rhs_batch.to_vec(),
        }
    }

    /// Contracted axes of A.
    pub fn lhs_contract(&self) -> &[usize] {
        &self.lhs_contract
    }

    /// Contracted axes of B, paired with [`lhs_contract`](Self::lhs_contract).
    pub fn rhs_contract(&self) -> &[usize] {
        &self.rhs_contract
    }

    /// Batch axes of A.
    pub fn lhs_batch(&self) -> &[usize] {
        &self.lhs_batch
    }

    /// Batch axes of B, paired with [`lhs_batch`](Self::lhs_batch).
    pub fn rhs_batch(&self) -> &[usize] {
        &self.rhs_batch
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
            let mut seen = vec![false; rank];
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
            Ok((0..rank).filter(|&x| !seen[x]).collect::<Vec<_>>())
        };
        let lhs_free = check(OperandId::A, a.len(), &self.lhs_contract, &self.lhs_batch)?;
        let rhs_free = check(OperandId::B, b.len(), &self.rhs_contract, &self.rhs_batch)?;
        for (&l, &r) in self
            .lhs_contract
            .iter()
            .zip(&self.rhs_contract)
            .chain(self.lhs_batch.iter().zip(&self.rhs_batch))
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
