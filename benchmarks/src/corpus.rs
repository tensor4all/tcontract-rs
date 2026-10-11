//! Benchmark corpora: shapes recorded from real workloads (spec 1e, P1) or
//! written by hand, replayed by `contract --corpus`.
//!
//! ```json
//! { "source": { "tool": "...", "commit": "..." },
//!   "entries": [
//!     { "name": "...", "op": "dot_general", "dtype": "f64",
//!       "a": {"dims": [..], "strides": [..]}, "b": {..}, "c": {..},
//!       "lc": [..], "rc": [..], "lb": [..], "rb": [..], "conj": [false, false],
//!       "calls": 12, "time_share": 0.3 } ] }
//! ```
//!
//! Strides are in elements and may be negative; a replay stores each operand
//! so that its lowest addressed element is element 0 ([`Operand::span`]).
use serde::Deserialize;

/// Element type of an entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Dtype {
    /// `f32`.
    F32,
    /// `f64`.
    F64,
    /// `Complex<f32>`.
    C32,
    /// `Complex<f64>`.
    C64,
}

impl Dtype {
    /// Short tag used in case names.
    pub fn tag(self) -> &'static str {
        match self {
            Dtype::F32 => "f32",
            Dtype::F64 => "f64",
            Dtype::C32 => "c32",
            Dtype::C64 => "c64",
        }
    }
}

/// One operand layout.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct Operand {
    /// Extents.
    pub dims: Vec<usize>,
    /// Element strides, one per extent.
    pub strides: Vec<isize>,
}

impl Operand {
    /// `(offset, len)`: storing the operand in `len` elements with the view
    /// starting at `offset` puts its lowest addressed element at 0. Empty
    /// operands give `(0, 0)`.
    pub fn span(&self) -> (usize, usize) {
        if self.dims.contains(&0) {
            return (0, 0);
        }
        let (mut lo, mut hi) = (0isize, 0isize);
        for (&d, &s) in self.dims.iter().zip(&self.strides) {
            let reach = (d as isize - 1) * s;
            if reach < 0 {
                lo += reach;
            } else {
                hi += reach;
            }
        }
        ((-lo) as usize, (hi - lo + 1) as usize)
    }

    fn check(&self, what: &str) -> Result<(), String> {
        if self.dims.len() != self.strides.len() {
            return Err(format!(
                "{what}: {} dims but {} strides",
                self.dims.len(),
                self.strides.len()
            ));
        }
        Ok(())
    }
}

/// A binary contraction entry (`dot_general` semantics of tcontract).
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct DotGeneralEntry {
    /// Case name.
    pub name: String,
    /// Element type.
    pub dtype: Dtype,
    /// Left operand.
    pub a: Operand,
    /// Right operand.
    pub b: Operand,
    /// Output.
    pub c: Operand,
    /// Contracted axes of `a`.
    pub lc: Vec<usize>,
    /// Contracted axes of `b`, paired with `lc`.
    pub rc: Vec<usize>,
    /// Batch axes of `a`.
    pub lb: Vec<usize>,
    /// Batch axes of `b`, paired with `lb`.
    pub rb: Vec<usize>,
    /// Conjugate `a`, `b`.
    #[serde(default)]
    pub conj: [bool; 2],
    /// Calls in the recorded workload.
    #[serde(default)]
    pub calls: Option<u64>,
    /// Share of the recorded workload's time in this shape.
    #[serde(default)]
    pub time_share: Option<f64>,
}

/// One corpus entry.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Entry {
    /// See [`DotGeneralEntry`].
    DotGeneral(DotGeneralEntry),
}

impl Entry {
    /// Case name.
    pub fn name(&self) -> &str {
        match self {
            Entry::DotGeneral(d) => &d.name,
        }
    }
}

/// A validated corpus.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Corpus {
    /// Where the shapes came from (tool, commits, workload); kept verbatim.
    #[serde(default)]
    pub source: serde_json::Value,
    /// Entries in file order.
    pub entries: Vec<Entry>,
}

impl Corpus {
    /// Parse and validate a corpus.
    ///
    /// # Errors
    ///
    /// Malformed JSON, an unknown dtype, rank/stride disagreement, a
    /// configuration tcontract rejects, an output whose dims differ
    /// from the configuration's, or an aliased (non-injective) output.
    pub fn parse(json: &str) -> Result<Self, String> {
        let c: Corpus = serde_json::from_str(json).map_err(|e| e.to_string())?;
        for e in &c.entries {
            validate(e).map_err(|m| format!("entry {}: {m}", e.name()))?;
        }
        Ok(c)
    }

    /// [`Corpus::parse`] of a file.
    ///
    /// # Errors
    ///
    /// As [`Corpus::parse`], or the file cannot be read.
    pub fn load(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        Self::parse(&text).map_err(|e| format!("{path}: {e}"))
    }
}

/// A malformed entry, with the reason the shared lowering gives.
///
/// The entry is described to `tcontract` exactly as a replay will
/// describe it (one validation, shared with the library), so a corpus that
/// parses is a corpus that plans.
fn validate(e: &Entry) -> Result<(), String> {
    use tcontract::api::{DType, DotGeneral, LayoutSpec, Op, OperandSpec, Problem};
    let Entry::DotGeneral(d) = e;
    d.a.check("a")?;
    d.b.check("b")?;
    d.c.check("c")?;
    let spec = |o: &Operand, op: Op| -> Result<OperandSpec, String> {
        let (offset, _) = o.span();
        Ok(OperandSpec::new(
            LayoutSpec::new(&o.dims, &o.strides, offset as isize).map_err(|e| e.to_string())?,
        )
        .with_op(op))
    };
    let op = |c: bool| if c { Op::Conjugate } else { Op::Identity };
    let dtype = match d.dtype {
        Dtype::F32 => DType::F32,
        Dtype::F64 => DType::F64,
        Dtype::C32 => DType::C32,
        Dtype::C64 => DType::C64,
    };
    Problem::from_dot_general(
        dtype,
        spec(&d.a, op(d.conj[0]))?,
        spec(&d.b, op(d.conj[1]))?,
        spec(&d.c, Op::Identity)?,
        &DotGeneral::new(&d.lc, &d.rc, &d.lb, &d.rb),
    )
    .map(|_| ())
    .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests;
