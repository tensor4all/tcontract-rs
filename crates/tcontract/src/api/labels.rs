//! The label front end.

/// Per-operand `i64` labels, with free output order and a separate C.
///
/// A label repeated within one operand selects a diagonal (extents must agree,
/// strides add). A label in only one input is a reduction. A label only in the
/// output is rejected as unsupported (TAPP case 5). `c` is present exactly when
/// the problem's C is described separately from D.
///
/// # Examples
///
/// ```
/// use tcontract::api::Labels;
/// let l = Labels::new(&[0, 2], &[2, 1], &[0, 1]);
/// assert!(l.c().is_none());
/// assert_eq!(l.with_c(&[0, 1]).c(), Some(&[0, 1][..]));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Labels {
    a: Vec<i64>,
    b: Vec<i64>,
    c: Option<Vec<i64>>,
    d: Vec<i64>,
}

impl Labels {
    /// Labels of A, B and D, with no separate C.
    pub fn new(a: &[i64], b: &[i64], d: &[i64]) -> Self {
        Self {
            a: a.to_vec(),
            b: b.to_vec(),
            c: None,
            d: d.to_vec(),
        }
    }

    /// The same labels with the labels of a separately described C.
    #[must_use]
    pub fn with_c(mut self, c: &[i64]) -> Self {
        self.c = Some(c.to_vec());
        self
    }

    /// Labels of A.
    pub fn a(&self) -> &[i64] {
        &self.a
    }

    /// Labels of B.
    pub fn b(&self) -> &[i64] {
        &self.b
    }

    /// Labels of a separate C, if any.
    pub fn c(&self) -> Option<&[i64]> {
        self.c.as_deref()
    }

    /// Labels of D.
    pub fn d(&self) -> &[i64] {
        &self.d
    }
}
