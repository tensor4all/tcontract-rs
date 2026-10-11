//! The README shows two programs; they are the files that CI compiles and runs,
//! so the README cannot drift from code that works.
use std::path::PathBuf;

#[test]
fn the_readme_shows_the_compiled_examples_verbatim() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let readme = std::fs::read_to_string(root.join("README.md")).expect("README.md");
    for example in [
        "crates/tcontract/examples/readme.rs",
        "crates/tcontract-capi/tests/c/standard_consumer.c",
    ] {
        let source = std::fs::read_to_string(root.join(example)).expect(example);
        assert!(
            readme.contains(&source),
            "README.md does not contain {example} verbatim"
        );
    }
}
