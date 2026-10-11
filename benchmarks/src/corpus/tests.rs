use super::*;

const OK: &str = r#"{
  "source": {"tool": "test"},
  "entries": [
    {"name": "mm", "op": "dot_general", "dtype": "f64",
     "a": {"dims": [4, 3], "strides": [1, 4]},
     "b": {"dims": [3, 5], "strides": [5, 1]},
     "c": {"dims": [4, 5], "strides": [1, 4]},
     "lc": [1], "rc": [0], "lb": [], "rb": [], "conj": [false, true],
     "calls": 7, "time_share": 0.25},
    {"name": "bmm", "op": "dot_general", "dtype": "c32",
     "a": {"dims": [2, 4, 5], "strides": [1, 2, 8]},
     "b": {"dims": [4, 3, 5], "strides": [1, 4, 12]},
     "c": {"dims": [2, 3, 5], "strides": [1, 2, 6]},
     "lc": [1], "rc": [0], "lb": [2], "rb": [2]}
  ]
}"#;

#[test]
fn parses_contraction_entries() {
    let c = Corpus::parse(OK).unwrap();
    assert_eq!(c.entries.len(), 2);
    let Entry::DotGeneral(d) = &c.entries[0];
    assert_eq!(d.name, "mm");
    assert_eq!(d.dtype, Dtype::F64);
    assert_eq!(d.conj, [false, true]);
    assert_eq!(d.calls, Some(7));
    let Entry::DotGeneral(b) = &c.entries[1];
    assert_eq!((b.lb.as_slice(), b.rb.as_slice()), (&[2][..], &[2][..]));
    assert_eq!(b.dtype, Dtype::C32);
}

/// The GEMM corpora were rewritten as contraction cases when the batched GEMM
/// entry point went away; the op tag is gone with it.
#[test]
fn a_batched_gemm_entry_is_no_longer_an_op() {
    let old = r#"{"name": "g", "op": "gemm_batched", "dtype": "f64", "m": 2, "n": 2, "k": 2,
      "batch": 1, "a": {"dims": [2, 2, 1], "strides": [1, 2, 4]},
      "b": {"dims": [2, 2, 1], "strides": [1, 2, 4]},
      "c": {"dims": [2, 2, 1], "strides": [1, 2, 4]}}"#;
    assert!(Corpus::parse(&format!(r#"{{"source": {{}}, "entries": [{old}]}}"#)).is_err());
}

/// Every shipped corpus parses and validates under the shared lowering.
#[test]
fn the_shipped_corpora_parse() {
    for (name, text) in [
        (
            "example",
            include_str!("../../benchmarks/tcontract/corpus/example.json"),
        ),
        (
            "hadamard",
            include_str!("../../benchmarks/tcontract/corpus/hadamard.json"),
        ),
        (
            "large-batched-gemm",
            include_str!("../../benchmarks/tcontract/corpus/large-batched-gemm.json"),
        ),
        (
            "tenferro-p1-gemm",
            include_str!("../../benchmarks/tcontract/corpus/tenferro-p1-gemm.json"),
        ),
        (
            "tenferro-p1",
            include_str!("../../benchmarks/tcontract/corpus/tenferro-p1.json"),
        ),
    ] {
        Corpus::parse(text).unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}

fn with(entry: &str) -> String {
    format!(r#"{{"source": {{}}, "entries": [{entry}]}}"#)
}

#[test]
fn rejects_malformed_entries() {
    let base = r#"{"name": "x", "op": "dot_general", "dtype": "f64",
      "a": {"dims": [4, 3], "strides": [1, 4]}, "b": {"dims": [3, 5], "strides": [5, 1]},
      "c": {"dims": [4, 5], "strides": [1, 4]}, "lc": [1], "rc": [0], "lb": [], "rb": []}"#;
    assert!(Corpus::parse(&with(base)).is_ok());
    for (bad, why) in [
        (
            base.replace(r#""dtype": "f64""#, r#""dtype": "i32""#),
            "dtype",
        ),
        (
            base.replace(r#""strides": [1, 4]}, "b""#, r#""strides": [1]}, "b""#),
            "rank",
        ),
        (base.replace(r#""rc": [0]"#, r#""rc": [1]"#), "extent"),
        (
            base.replace(
                r#""c": {"dims": [4, 5], "strides": [1, 4]}"#,
                r#""c": {"dims": [4, 5], "strides": [1, 1]}"#,
            ),
            "aliased",
        ),
        (
            base.replace(r#""c": {"dims": [4, 5]"#, r#""c": {"dims": [4, 6]"#),
            "output dims",
        ),
    ] {
        assert!(Corpus::parse(&with(&bad)).is_err(), "{why} accepted");
    }
}

#[test]
fn span_offset_puts_the_lowest_element_at_zero() {
    let op = Operand {
        dims: vec![3, 2],
        strides: vec![-1, 3],
    };
    assert_eq!(op.span(), (2, 6));
    let empty = Operand {
        dims: vec![0, 2],
        strides: vec![1, 1],
    };
    assert_eq!(empty.span(), (0, 0));
}
