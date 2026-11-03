use super::*;

#[test]
fn parses_static_and_dynamic() {
    assert_eq!(parse("static"), Some(None));
    assert_eq!(
        parse("dynamic:16,32"),
        Some(Some(Partition::DynamicTiles {
            job_m: 16,
            job_n: 32
        }))
    );
    assert_eq!(parse("dynamic:16"), None);
    assert_eq!(parse("dynamic:a,b"), None);
    assert_eq!(parse("grid"), None);
    assert_eq!(
        suffix(parse("dynamic:8,24").unwrap()),
        "_dyn8x24".to_string()
    );
}
