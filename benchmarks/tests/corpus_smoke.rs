//! Replays the example corpus through the benchmark binaries once, at one
//! thread, and requires every strategy comparison to agree.
use std::process::Command;

fn replay(bin: &str) -> String {
    let corpus = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/benchmarks/tcontract/corpus/example.json"
    );
    let out = Command::new(bin)
        .args(["--threads", "1", "--corpus", corpus])
        .env("BENCH_RUNS", "1")
        .env("BENCH_WARMUP", "0")
        .env_remove("RAYON_NUM_THREADS")
        .output()
        .expect("run");
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success(),
        "{bin} failed: {text}{}",
        String::from_utf8_lossy(&out.stderr)
    );
    text
}

fn checks(text: &str) -> Vec<&str> {
    text.lines().filter(|l| l.starts_with("CHECK")).collect()
}

#[test]
fn contract_replays_every_dot_general_entry() {
    let text = replay(env!("CARGO_BIN_EXE_contract"));
    let c = checks(&text);
    // The four contraction entries and the two batched-GEMM entries that were
    // rewritten as contraction cases.
    assert_eq!(c.len(), 6, "{text}");
    assert!(c.iter().all(|l| l.ends_with(" ok")), "{text}");
}

/// `tcbench verify` on one small corpus case: the planner's choice and the packed
/// driver agree in a real and a complex dtype, and the stress modes plan.
#[test]
fn tcbench_verifies_a_small_case_under_every_stress_mode() {
    for stress in ["none", "ragged", "padded"] {
        let out = Command::new(env!("CARGO_BIN_EXE_tcbench"))
            .args([
                "verify", "--size", "1", "--case", "ij-ik-kj", "--dtype", "f64,c64", "--stress",
                stress,
            ])
            .output()
            .expect("run");
        let text = String::from_utf8_lossy(&out.stdout).into_owned();
        assert!(
            out.status.success(),
            "stress {stress}: {text}{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(text.contains("all comparisons within tolerance"), "{text}");
    }
}

#[test]
fn tcbench_info_and_run_work_without_the_baselines() {
    let info = Command::new(env!("CARGO_BIN_EXE_tcbench"))
        .arg("info")
        .output()
        .expect("run");
    assert!(info.status.success());
    let run = Command::new(env!("CARGO_BIN_EXE_tcbench"))
        .args([
            "run",
            "--size",
            "1",
            "--reps",
            "1",
            "--case",
            "ij-ik-kj",
            "--dtype",
            "f64",
            "--engines",
            "plan,packed",
        ])
        .output()
        .expect("run");
    let text = String::from_utf8_lossy(&run.stdout).into_owned();
    assert!(run.status.success(), "{text}");
    assert!(
        text.contains("plan GF/s") && text.contains("packed GF/s"),
        "{text}"
    );
}
