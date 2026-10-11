//! A deliberately naive reference implementation, used as the test oracle.
//!
//! It shares no code with the planner or the engine: it walks the raw layouts
//! and label lists directly, so it independently defines the semantics of
//! repeated labels (diagonals), isolated labels (reductions), Hadamard labels,
//! conjugation and general strides. It does not use `tcontract`'s
//! lowering, roles or offsets.

use tcontract_kernel::Element;

/// A read-only operand for the oracle: plain data, no engine types.
#[derive(Clone, Copy, Debug)]
pub struct RefOperand<'a, T> {
    /// The backing allocation.
    pub data: &'a [T],
    /// Extents.
    pub dims: &'a [usize],
    /// Signed element strides.
    pub strides: &'a [isize],
    /// Element offset of index `(0, .., 0)`.
    pub offset: isize,
    /// One index label per mode.
    pub labels: &'a [i64],
    /// Whether the operand is read conjugated.
    pub conj: bool,
}

/// The output operand of the oracle.
#[derive(Debug)]
pub struct RefOutput<'a, T> {
    /// The backing allocation, written in place.
    pub data: &'a mut [T],
    /// Extents.
    pub dims: &'a [usize],
    /// Signed element strides.
    pub strides: &'a [isize],
    /// Element offset of index `(0, .., 0)`.
    pub offset: isize,
    /// One index label per mode.
    pub labels: &'a [i64],
    /// Whether the result is conjugated before it is stored.
    pub conj: bool,
}

/// Why the oracle refused its input.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum OracleError {
    /// An operand has a different number of labels than modes.
    LabelCount {
        /// Operand name.
        operand: &'static str,
        /// Modes of the operand.
        modes: usize,
        /// Labels given.
        labels: usize,
    },
    /// A label has two different extents.
    Extent {
        /// The label.
        label: i64,
    },
}

impl core::fmt::Display for OracleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            OracleError::LabelCount {
                operand,
                modes,
                labels,
            } => write!(f, "{operand} has {modes} modes but {labels} labels"),
            OracleError::Extent { label } => write!(f, "label {label} has two extents"),
        }
    }
}

impl std::error::Error for OracleError {}

/// `D = op_D(alpha * op_A(A) * op_B(B) + beta * op_C(C))`, computed by brute
/// force over every index assignment.
///
/// `beta == 0` reads no `C`. A label shared by several modes of one operand
/// selects that operand's diagonal. It walks the full index space with a linear
/// label search per access, so it is several orders of magnitude slower than
/// the engine: fine for the shapes a test uses, hopeless above a few million
/// output elements.
///
/// # Errors
///
/// [`OracleError`] for a label-count or extent inconsistency.
pub fn contract_reference<T: Element>(
    alpha: T,
    a: &RefOperand<'_, T>,
    b: &RefOperand<'_, T>,
    beta: T,
    c: Option<&RefOperand<'_, T>>,
    d: &mut RefOutput<'_, T>,
) -> Result<(), OracleError> {
    // Collect label extents, checking consistency.
    let mut labels: Vec<(i64, usize)> = Vec::new();
    let mut add = |name: &'static str, ls: &[i64], dims: &[usize]| -> Result<(), OracleError> {
        if ls.len() != dims.len() {
            return Err(OracleError::LabelCount {
                operand: name,
                modes: dims.len(),
                labels: ls.len(),
            });
        }
        for (k, &l) in ls.iter().enumerate() {
            match labels.iter().find(|x| x.0 == l) {
                Some(&(_, e0)) if e0 != dims[k] => return Err(OracleError::Extent { label: l }),
                Some(_) => {}
                None => labels.push((l, dims[k])),
            }
        }
        Ok(())
    };
    add("A", a.labels, a.dims)?;
    add("B", b.labels, b.dims)?;
    if let Some(c) = c {
        add("C", c.labels, c.dims)?;
    }
    add("D", d.labels, d.dims)?;

    // Split into output labels (those in D) and contracted labels.
    let out: Vec<(i64, usize)> = labels
        .iter()
        .copied()
        .filter(|(l, _)| d.labels.contains(l))
        .collect();
    let sum: Vec<(i64, usize)> = labels
        .iter()
        .copied()
        .filter(|(l, _)| !d.labels.contains(l))
        .collect();

    let offset = |ls: &[i64], strides: &[isize], base: isize, assign: &dyn Fn(i64) -> usize| {
        ls.iter()
            .enumerate()
            .map(|(k, &l)| assign(l) as isize * strides[k])
            .sum::<isize>()
            + base
    };

    let out_total: usize = out.iter().map(|x| x.1).product();
    let sum_total: usize = sum.iter().map(|x| x.1).product();

    let mut ovals = vec![0usize; out.len()];
    for _ in 0..out_total {
        let lookup_out = |l: i64| -> usize {
            out.iter()
                .position(|x| x.0 == l)
                .map(|p| ovals[p])
                .unwrap_or(0)
        };

        let mut acc = T::zero();
        let mut svals = vec![0usize; sum.len()];
        for _ in 0..sum_total {
            let assign = |l: i64| -> usize {
                if let Some(p) = out.iter().position(|x| x.0 == l) {
                    ovals[p]
                } else if let Some(p) = sum.iter().position(|x| x.0 == l) {
                    svals[p]
                } else {
                    0
                }
            };
            let av = a.data[offset(a.labels, a.strides, a.offset, &assign) as usize];
            let bv = b.data[offset(b.labels, b.strides, b.offset, &assign) as usize];
            let av = if a.conj { av.conj() } else { av };
            let bv = if b.conj { bv.conj() } else { bv };
            acc = acc.add(av.mul(bv));
            odometer(&mut svals, &sum);
        }

        let mut v = alpha.mul(acc);
        if beta != T::zero() {
            if let Some(c) = c {
                let cv = c.data[offset(c.labels, c.strides, c.offset, &lookup_out) as usize];
                let cv = if c.conj { cv.conj() } else { cv };
                v = v.add(beta.mul(cv));
            }
        }
        if d.conj {
            v = v.conj();
        }
        d.data[offset(d.labels, d.strides, d.offset, &lookup_out) as usize] = v;

        odometer(&mut ovals, &out);
    }
    Ok(())
}

fn odometer(vals: &mut [usize], dims: &[(i64, usize)]) {
    for (v, &(_, e)) in vals.iter_mut().zip(dims) {
        *v += 1;
        if *v < e {
            return;
        }
        *v = 0;
    }
}
