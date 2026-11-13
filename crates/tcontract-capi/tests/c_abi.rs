//! Compiles `tests/c/abi_test.c` against the headers, links it to the built
//! `libtcontract_capi.so`, and runs it; also checks the exported symbol set.
use std::path::PathBuf;
use std::process::Command;

fn has_capi_library(dir: &std::path::Path) -> bool {
    dir.join("libtcontract_capi.so").exists() || dir.join("libtcontract_capi.dylib").exists()
}

fn target_dir() -> PathBuf {
    // Cargo may put the harness under .../target/<profile>/deps/ or, with the
    // build-dir layout, under .../target/<profile>/build/<crate>/<hash>/out/.
    // In that layout the cdylib and the harness get different hash directories,
    // so a walk up from the harness is not enough: scan the package's build
    // directories for the one that holds the library.
    let exe = std::env::current_exe().expect("exe");
    let mut dir = exe.parent().expect("exe parent").to_path_buf();
    loop {
        if has_capi_library(&dir) {
            return dir;
        }
        let pkg_build = dir.join("build").join("tcontract-capi");
        if let Ok(entries) = std::fs::read_dir(&pkg_build) {
            for entry in entries.flatten() {
                let out = entry.path().join("out");
                if has_capi_library(&out) {
                    return out;
                }
            }
        }
        if !dir.pop() {
            panic!("libtcontract_capi not found above {}", exe.display());
        }
    }
}

#[test]
fn c_program_links_and_runs() {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    if Command::new(&cc).arg("--version").output().is_err() {
        eprintln!("skipping: no C compiler ({cc})");
        return;
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let include = root.join("include");
    let lib = target_dir();
    assert!(
        lib.join("libtcontract_capi.so").exists() || lib.join("libtcontract_capi.dylib").exists(),
        "libtcontract_capi not built in {}",
        lib.display()
    );
    let out = std::env::temp_dir().join(format!("tcontract_abi_test_{}", std::process::id()));
    let st = Command::new(&cc)
        .args(["-std=c11", "-Wall", "-Werror", "-o"])
        .arg(&out)
        .arg(root.join("tests/c/abi_test.c"))
        .arg(format!("-I{}", include.display()))
        .arg(format!("-L{}", lib.display()))
        .arg(format!("-Wl,-rpath,{}", lib.display()))
        .args(["-ltcontract_capi", "-lm"])
        .status()
        .expect("cc");
    assert!(st.success(), "C compile failed");
    let run = Command::new(&out).output().expect("run");
    let _ = std::fs::remove_file(&out);
    assert!(
        run.status.success(),
        "abi_test failed: {}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
}

/// The library's file name and the `nm` arguments listing its exported,
/// defined symbols on this platform.
fn library_and_nm_args() -> (&'static str, &'static [&'static str]) {
    if cfg!(target_os = "macos") {
        ("libtcontract_capi.dylib", &["-gU"])
    } else {
        ("libtcontract_capi.so", &["-D", "--defined-only"])
    }
}

/// `tcontract_*` and `TAPP_*` names in `nm` output (`[address] type name` per line), with
/// the Mach-O leading underscore removed when `macho`.
fn tcontract_symbols(nm_output: &str, macho: bool) -> Vec<String> {
    nm_output
        .lines()
        .filter_map(|l| l.split_whitespace().last())
        .map(|s| {
            if macho {
                s.strip_prefix('_').unwrap_or(s)
            } else {
                s
            }
        })
        .filter(|s| s.starts_with("tcontract_") || s.starts_with("TAPP_"))
        .map(str::to_owned)
        .collect()
}

#[test]
fn nm_output_is_parsed_for_elf_and_macho() {
    let elf = "0000000000012340 T tcontract_abi_version\n                 U malloc\n0000000000012350 T tcontract_has_part\n";
    assert_eq!(
        tcontract_symbols(elf, false),
        ["tcontract_abi_version", "tcontract_has_part"]
    );
    let macho = "0000000000003f10 T _tcontract_abi_version\n0000000000003f20 T _tcontract_has_part\n0000000000003f30 T _other\n";
    assert_eq!(
        tcontract_symbols(macho, true),
        ["tcontract_abi_version", "tcontract_has_part"]
    );
}

#[test]
fn only_tcontract_symbols_of_the_selected_parts_are_exported() {
    let (name, args) = library_and_nm_args();
    let lib = target_dir().join(name);
    assert!(
        lib.exists(),
        "{} not built: run `cargo build -p tcontract-capi` with the same profile first",
        lib.display()
    );
    let out = Command::new("nm")
        .args(args)
        .arg(&lib)
        .output()
        .expect("nm is required to inspect the exported symbols");
    assert!(
        out.status.success(),
        "nm {args:?} {} failed: {}",
        lib.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    let syms = tcontract_symbols(
        &String::from_utf8_lossy(&out.stdout),
        cfg!(target_os = "macos"),
    );
    // The superseded proprietary surfaces are gone.
    for gone in [
        "tcontract_exec_serial",
        "tcontract_exec_rayon_create",
        "tcontract_exec_close",
        "tcontract_exec_retain",
        "tcontract_exec_release",
        "tcontract_plan_create",
        "tcontract_plan_execute",
        "tcontract_plan_destroy",
        "tcontract_blas_gemm",
        "tcontract_blas_gemm_batched",
        // #149: the entries that start a pool say `pool`, not `rayon`.
        "tcontract_tapp_executor_create_rayon",
        "tcontract_pool_create_rayon",
    ] {
        assert!(
            !syms.iter().any(|s| s == gone),
            "{gone} is still exported: {syms:?}"
        );
    }
    for want in [
        "tcontract_abi_version",
        "tcontract_pool_create",
        "tcontract_pool_destroy",
        "tcontract_tapp_executor_borrow_pool",
        "tcontract_tapp_executor_create_pool",
        "tcontract_tapp_executor_set_budget",
        "tcontract_tapp_executor_get_threads",
        "TAPP_create_executor",
        "TAPP_destroy_executor",
        "TAPP_create_tensor_product",
        "TAPP_execute_product",
        "TAPP_execute_batched_product",
        "tcontract_has_part",
    ] {
        assert!(
            syms.iter().any(|s| s == want),
            "{want} not exported: {syms:?}"
        );
    }
}

#[test]
fn headers_combine_with_a_real_dlpack_h_in_either_order() {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    if Command::new(&cc).arg("--version").output().is_err() {
        eprintln!("skipping: no C compiler ({cc})");
        return;
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let include = root.join("include");
    for src in ["headers_dlpack_first.c", "headers_tcontract_first.c"] {
        let out = std::env::temp_dir().join(format!("tcontract_{src}_{}", std::process::id()));
        let st = Command::new(&cc)
            .args(["-std=c11", "-Wall", "-Werror", "-fsyntax-only"])
            .arg(format!("-I{}", include.display()))
            .arg(root.join("tests/c").join(src))
            .status()
            .expect("cc");
        let _ = std::fs::remove_file(&out);
        assert!(st.success(), "{src} failed to compile");
    }
}

/// The standard-ABI claim: a program that includes only the pinned upstream
/// headers (`<tapp.h>`, no tcontract header) compiles as C and as C++ and performs
/// a serial contraction through the library.
#[test]
fn consumer_of_the_pinned_standard_headers_only() {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let cxx = std::env::var("CXX").unwrap_or_else(|_| "c++".into());
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let include = root.join("include");
    let lib = target_dir();
    let src = root.join("tests/c/standard_consumer.c");
    for (compiler, lang, std_flag) in [(&cc, "c", "-std=c11"), (&cxx, "c++", "-std=c++17")] {
        if Command::new(compiler).arg("--version").output().is_err() {
            eprintln!("skipping {lang}: no compiler ({compiler})");
            continue;
        }
        let out = std::env::temp_dir().join(format!(
            "tcontract_standard_consumer_{}_{}",
            lang.replace('+', "p"),
            std::process::id()
        ));
        let st = Command::new(compiler)
            .args([std_flag, "-Wall", "-Werror", "-x", lang, "-o"])
            .arg(&out)
            .arg(&src)
            // Only the upstream headers' root: `<tapp.h>` and `tapp/*.h` must
            // be enough; no `tcontract/` header is included by the program.
            .arg(format!("-I{}", include.display()))
            .arg(format!("-L{}", lib.display()))
            .arg(format!("-Wl,-rpath,{}", lib.display()))
            .args(["-ltcontract_capi", "-lm"])
            .status()
            .expect("compiler");
        assert!(
            st.success(),
            "standard_consumer.c failed to compile as {lang}"
        );
        let run = Command::new(&out).output().expect("run");
        let _ = std::fs::remove_file(&out);
        assert!(
            run.status.success(),
            "standard consumer ({lang}) failed: {}{}",
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        );
    }
}

/// The vendored headers are the pinned upstream files: the recorded commit is
/// named in `tapp/README.md`, and every file matches the SHA-256 listed there
/// (skipped when no `sha256sum`/`shasum` exists).
#[test]
fn vendored_tapp_headers_are_the_pinned_ones() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("include");
    let readme = std::fs::read_to_string(root.join("tapp/README.md")).expect("tapp/README.md");
    assert!(readme.contains("77c32d744ee6d339f504620cc80b8679601669bc"));
    let recorded: Vec<(&str, &str)> = readme
        .lines()
        .filter_map(|l| {
            let (sum, file) = l.split_once("  ")?;
            (sum.len() == 64 && sum.chars().all(|c| c.is_ascii_hexdigit())).then_some((sum, file))
        })
        .collect();
    assert_eq!(recorded.len(), 12, "{recorded:?}");
    let tool = [("sha256sum", &[][..]), ("shasum", &["-a", "256"][..])]
        .into_iter()
        .find(|(t, _)| Command::new(t).arg("--version").output().is_ok());
    for (sum, file) in recorded {
        assert!(root.join(file).exists(), "{file} is not vendored");
        let Some((tool, args)) = tool else { continue };
        let out = Command::new(tool)
            .args(args)
            .arg(root.join(file))
            .output()
            .expect("checksum tool");
        let got = String::from_utf8_lossy(&out.stdout);
        assert!(
            got.starts_with(sum),
            "{file} differs from the pinned upstream copy ({got})"
        );
    }
}

/// The declaration link test in `examples/c-consumer/main.c` must survive the
/// optimizer. A plain array of function addresses is dead code (a function's
/// address is never null), so `-O2` may delete it along with every relocation
/// and the test then passes with a symbol the library does not export. This
/// compiles the example with one pool symbol renamed to a name no library has
/// and asserts the reference is still there in the object file.
#[test]
fn the_declaration_link_test_survives_optimization() {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    if Command::new(&cc).arg("--version").output().is_err() {
        eprintln!("skipping: no C compiler ({cc})");
        return;
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let obj = std::env::temp_dir().join(format!(
        "tcontract_c_consumer_link_{}.o",
        std::process::id()
    ));
    let st = Command::new(&cc)
        .args([
            "-std=c11",
            "-O2",
            "-c",
            "-Dtcontract_pool_create=deliberately_missing_pool_symbol",
        ])
        .arg(format!("-I{}", root.join("include").display()))
        .arg(root.join("../../examples/c-consumer/main.c"))
        .arg("-o")
        .arg(&obj)
        .status()
        .expect("cc");
    assert!(st.success(), "examples/c-consumer/main.c did not compile");
    let out = Command::new("nm")
        .args(["-u"])
        .arg(&obj)
        .output()
        .expect("nm");
    let _ = std::fs::remove_file(&obj);
    let macho = cfg!(target_os = "macos");
    let undefined: Vec<String> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.split_whitespace().last())
        .map(|s| {
            if macho {
                s.strip_prefix('_').unwrap_or(s)
            } else {
                s
            }
        })
        .map(str::to_owned)
        .collect();
    for want in [
        "deliberately_missing_pool_symbol",
        "tcontract_pool_destroy",
        "tcontract_tapp_executor_borrow_pool",
    ] {
        assert!(
            undefined.iter().any(|s| s == want),
            "{want} did not survive -O2, so the declaration link test proves nothing for it: {undefined:?}"
        );
    }
}
