//! Result collection, table formatting and CSV output.
//!
//! The timed CSV's leading columns are exactly `tcbench`'s: the campaign's
//! report reads one column set for both harnesses and a second shape would be a
//! second contract. The timing-v2 evidence columns are appended at the end, so a
//! consumer that reads the old header still works.

use std::fmt::Write as _;
use std::fs;

use crate::corpus::Program;
use crate::engines::Statistic;

/// One measurement.
#[derive(Clone, Debug)]
pub struct Row {
    pub case: String,
    pub dtype: String,
    pub engine: String,
    pub threads: usize,
    pub m: u64,
    pub n: u64,
    pub k: u64,
    pub macs: u64,
    pub secs: f64,
    /// Scatter across the timed repetitions, `(max - min) / best`. One slow first
    /// repetition — an arm measured before the machine settles — shows up here,
    /// where `secs` alone would hide it. 0 for a single repetition.
    pub spread: f64,
    /// Seconds spent constructing what this arm needs *before* the timed call: the
    /// plan for this library's arms, the operand descriptors for the reference. The
    /// timing policy excludes this work, and the reference has no equivalent to
    /// hoist, so the number makes that asymmetry legible instead of implied.
    pub prepare_s: f64,
    pub gflops: f64,
    /// Fraction of A's row blocks and B's column blocks that are regular. Not
    /// aggregated across a program's steps here, so every row carries 0.
    pub reg_a: f64,
    pub reg_b: f64,
    pub notes: String,
    /// Every timed call's wall time in seconds, in chronological order. The
    /// evidence CSV reports these so a consumer can re-derive `secs` and see the
    /// spread a single summary would hide.
    pub samples: Vec<f64>,
}

#[derive(Default)]
pub struct Results {
    pub rows: Vec<Row>,
    /// The run's regime label (`--regime`), echoed into every evidence row.
    pub regime: String,
    /// The effective unit batch, echoed into every evidence row.
    pub batch: String,
    /// Which summary of `samples` `secs` (and `total_ms`) reports.
    pub statistic: Statistic,
    /// The declared size spec (`--n`), which the evidence identity shape is
    /// derived from. Empty when the flag was not given.
    pub n_spec: String,
}

/// The identity shape the toolkit derives from a declared size spec: a 1-dim
/// spec is a square, a 2-dim spec is `m x n`, and a 3-dim spec is `m x k x n`
/// (the toolkit's own order). Only a family literally named `gemm` carries a
/// contracting `k` in the 1-dim form.
///
/// This corpus has no size knob, so the spec is only what the declared grid
/// names; the resolved extents are the harness's own report (`case_m` etc.) and
/// are not the toolkit's identity.
fn declared_shape(spec: &str, family: &str) -> (String, String, String) {
    let dims: Vec<&str> = spec.split('x').collect();
    match dims[..] {
        [m, k, n] => (m.to_string(), n.to_string(), k.to_string()),
        [m, n] => (m.to_string(), n.to_string(), String::new()),
        [m] if family == "gemm" => (m.to_string(), m.to_string(), m.to_string()),
        [m] => (m.to_string(), m.to_string(), String::new()),
        _ => (String::new(), String::new(), String::new()),
    }
}

impl Results {
    pub fn push(&mut self, r: Row) {
        self.rows.push(r);
    }

    pub fn get(&self, case: &str, dtype: &str, engine: &str) -> Option<&Row> {
        self.rows
            .iter()
            .find(|r| r.case == case && r.dtype == dtype && r.engine == engine)
    }

    /// The leading (`tcbench`-shaped) columns, with the timing-v2 evidence
    /// columns appended at the end so a consumer that reads the old header still
    /// works.
    ///
    /// The row plan is the census, not `rows`: a declared cell that was skipped
    /// still has to appear, with its status and the same note the census carries
    /// (`verify` compares the two cell sets, notes included). The rows are
    /// emitted in the census order, which is the declared rotation.
    pub fn write_csv(&self, path: &str, census: &Census) -> std::io::Result<()> {
        let mut s = String::from(
            "case,group,dtype,engine,threads,m,n,k,macs,seconds,spread,prepare_s,gflops,regular_a,regular_b,notes,regime,family,order,batch,row,status,note,total_ms,per_item_us,samples_ms,rayon_threads,vendor_threads,case_m,case_n,case_k\n",
        );
        let batch: f64 = self.batch.parse().unwrap_or(1.0);
        for c in &census.rows {
            let measured = self.get(&c.family, &c.dtype, &c.row);
            // The declared spec names the grid, so it is what the identity
            // columns carry once `--n` is given. Without it there is no declared
            // grid to name and those columns keep the resolved extents they
            // carried before the census existed; the resolved extents are in
            // `case_m`/`case_n`/`case_k` either way.
            let (m, n, k) = if self.n_spec.is_empty() {
                match measured {
                    Some(r) => (r.m.to_string(), r.n.to_string(), r.k.to_string()),
                    None => Default::default(),
                }
            } else {
                declared_shape(&self.n_spec, &c.family)
            };
            let (case_m, case_n, case_k) = (
                c.case_m.to_string(),
                c.case_n.to_string(),
                c.case_k.to_string(),
            );
            // The identity columns both a measured and a skipped row share.
            let mut fields = vec![
                c.family.clone(),
                c.group.clone(),
                c.dtype.clone(),
                c.row.clone(),
                c.threads.to_string(),
                m,
                n,
                k,
            ];
            if c.status == "ok" {
                let r = measured.expect("a census cell marked ok was measured");
                // `total_ms` is the requested statistic of the chronological
                // samples and `per_item_us` divides it by the unit batch; the
                // harness's own per-macs figure stays derivable from `macs`
                // and `seconds`. The samples keep full precision and the two
                // aggregates nine decimals: an even repetition count averages
                // two samples, and a millisecond rounded to six decimals is a
                // whole microsecond away from the median it is checked against.
                let total_ms = self.statistic.of(&r.samples) * 1000.0;
                let per_item_us = total_ms * 1000.0 / batch;
                let samples_ms = format!(
                    "[{}]",
                    r.samples
                        .iter()
                        .map(|s| format!("{:.12}", s * 1000.0))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                fields.extend([
                    r.macs.to_string(),
                    format!("{:.9}", r.secs),
                    format!("{:.4}", r.spread),
                    format!("{:.9}", r.prepare_s),
                    format!("{:.4}", r.gflops),
                    format!("{:.4}", r.reg_a),
                    format!("{:.4}", r.reg_b),
                    r.notes.clone(),
                    self.regime.clone(),
                    c.family.clone(),
                    String::new(),
                    c.batch.clone(),
                    c.row.clone(),
                    "ok".to_string(),
                    String::new(),
                    format!("{:.9}", total_ms),
                    format!("{:.9}", per_item_us),
                    samples_ms,
                    r.threads.to_string(),
                    r.threads.to_string(),
                    case_m,
                    case_n,
                    case_k,
                ]);
            } else {
                // A skipped cell carries no timing: `samples_ms` stays empty
                // rather than `[]`, because the evidence check reads a nonempty
                // value there as an unmeasured row that was timed anyway.
                fields.extend([
                    String::new(),
                    String::new(),
                    String::new(),
                    String::new(),
                    String::new(),
                    String::new(),
                    String::new(),
                    String::new(),
                    self.regime.clone(),
                    c.family.clone(),
                    String::new(),
                    c.batch.clone(),
                    c.row.clone(),
                    "skipped".to_string(),
                    c.note.clone(),
                    String::new(),
                    String::new(),
                    String::new(),
                    c.threads.to_string(),
                    c.threads.to_string(),
                    case_m,
                    case_n,
                    case_k,
                ]);
            }
            let _ = writeln!(
                s,
                "{}",
                fields.iter().map(field).collect::<Vec<_>>().join(",")
            );
        }
        fs::write(path, s)
    }
}

/// One declared cell of the run grid, in the census CSV.
struct CensusRow {
    regime: String,
    family: String,
    group: String,
    dtype: String,
    batch: String,
    row: String,
    threads: usize,
    status: &'static str,
    note: String,
    /// The case's resolved `m`/`n`/`k` (the final step's for a chain), which the
    /// timed CSV carries in `case_m`/`case_n`/`case_k` for a measured *and* a
    /// skipped cell: the case is known even when its row was not run.
    case_m: u64,
    case_n: u64,
    case_k: u64,
}

/// The element types the corpus fixes its cases to. A requested dtype outside
/// this set is `skipped` with that reason rather than a silent absence.
const KNOWN_DTYPES: [&str; 2] = ["f64", "c64"];

/// The declared run grid: every case × requested dtype × engine row, with the
/// status of the cells that were actually measured. Written by `run` and
/// `verify` alike so a consumer can tell a skipped cell from a missing one, and
/// used by `run` as the row plan for the timed CSV so both carry the same cells
/// in the same rotated order.
///
/// A case fixes its own dtype, so the wrong-dtype cells of the rectangular
/// product are declared and skipped: the toolkit compares the census against the
/// whole declared grid and then requires the timed cells and statuses to match.
pub struct Census {
    /// The declared size spec (`--n`), which the identity shape is derived from.
    n_spec: String,
    rows: Vec<CensusRow>,
}

impl Census {
    /// The full grid, in the run's execution order. Every cell starts `skipped`;
    /// `mark_ok`/`mark_skipped` move the ones the run reached.
    pub fn declared(
        regime: &str,
        n_spec: &str,
        batch: &str,
        threads: usize,
        cases: &[Program],
        dtypes: &[String],
        rows: &[String],
    ) -> Census {
        let mut cells = Vec::new();
        for case in cases {
            let (case_m, case_n, case_k) = case.report_mnk();
            for dtype in dtypes {
                let note = if !KNOWN_DTYPES.contains(&dtype.as_str()) {
                    format!("unsupported dtype {dtype}")
                } else if dtype != case.dtype {
                    format!("case dtype is {}", case.dtype)
                } else {
                    "not measured".to_string()
                };
                for row in rows {
                    cells.push(CensusRow {
                        regime: regime.to_string(),
                        family: case.name.clone(),
                        group: case.group.to_string(),
                        dtype: dtype.clone(),
                        batch: batch.to_string(),
                        row: row.clone(),
                        threads,
                        status: "skipped",
                        note: note.clone(),
                        case_m,
                        case_n,
                        case_k,
                    });
                }
            }
        }
        Census {
            n_spec: n_spec.to_string(),
            rows: cells,
        }
    }

    /// Record that `(family, dtype, row)` was measured.
    pub fn mark_ok(&mut self, family: &str, dtype: &str, row: &str) {
        for c in &mut self.rows {
            if c.family == family && c.dtype == dtype && c.row == row {
                c.status = "ok";
                c.note.clear();
            }
        }
    }

    /// Record why `(family, dtype, row)` was declared but not measured.
    pub fn mark_skipped(&mut self, family: &str, dtype: &str, row: &str, note: &str) {
        for c in &mut self.rows {
            if c.family == family && c.dtype == dtype && c.row == row {
                c.status = "skipped";
                c.note = note.to_string();
            }
        }
    }

    pub fn write_csv(&self, path: &str) -> std::io::Result<()> {
        let mut s = String::from("regime,family,dtype,order,m,n,k,batch,row,threads,status,note\n");
        for c in &self.rows {
            let (m, n, k) = declared_shape(&self.n_spec, &c.family);
            let fields = [
                c.regime.clone(),
                c.family.clone(),
                c.dtype.clone(),
                String::new(),
                m,
                n,
                k,
                c.batch.clone(),
                c.row.clone(),
                c.threads.to_string(),
                c.status.to_string(),
                c.note.clone(),
            ];
            let _ = writeln!(
                s,
                "{}",
                fields.iter().map(field).collect::<Vec<_>>().join(",")
            );
        }
        fs::write(path, s)
    }
}

pub fn print_environment() {
    println!("host        : {}", hostname());
    println!(
        "target      : {} / {}",
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    #[cfg(target_arch = "x86_64")]
    {
        let f = |n: &str, v: bool| if v { format!("{n} ") } else { String::new() };
        print!("cpu features: ");
        print!("{}", f("avx2", std::is_x86_feature_detected!("avx2")));
        print!("{}", f("fma", std::is_x86_feature_detected!("fma")));
        print!("{}", f("avx512f", std::is_x86_feature_detected!("avx512f")));
        print!(
            "{}",
            f("avx512dq", std::is_x86_feature_detected!("avx512dq"))
        );
        println!();
    }
    #[cfg(target_arch = "aarch64")]
    {
        let f = |n: &str, v: bool| if v { format!("{n} ") } else { String::new() };
        print!("cpu features: ");
        print!(
            "{}",
            f("neon", std::arch::is_aarch64_feature_detected!("neon"))
        );
        print!(
            "{}",
            f("fp16", std::arch::is_aarch64_feature_detected!("fp16"))
        );
        print!(
            "{}",
            f("bf16", std::arch::is_aarch64_feature_detected!("bf16"))
        );
        print!(
            "{}",
            f("sve", std::arch::is_aarch64_feature_detected!("sve"))
        );
        println!();
    }
    {
        use tcontract_kernel::blocking as cache;
        let h = cache::hierarchy();
        let one = |l: &cache::CacheLevel| {
            format!(
                "L{}={}KiB/{}-way/{}sh",
                l.level,
                l.size / 1024,
                l.ways,
                l.shared_by
            )
        };
        let levels: Vec<String> = [Some(h.l1d), h.l2, h.l3]
            .into_iter()
            .flatten()
            .map(|l| one(&l))
            .collect();
        println!(
            "caches      : {} (via {})",
            levels.join(" "),
            h.source.name()
        );
    }
    println!("baselines   : tblis={}", cfg!(feature = "tblis"));
    #[cfg(feature = "tblis")]
    println!(
        "tblis       : abi {}, {} thread(s)",
        crate::tblis::VERSION,
        unsafe { crate::tblis::tblis_get_num_threads() }
    );
}

/// The measurement toolkit reads this contract from `info` as `key=value` lines,
/// one per line, on top of [`print_environment`]'s human text. A key whose value
/// is unavailable is printed empty rather than omitted, so a consumer can tell
/// "not built" from "the harness changed".
///
/// This stays out of `run`/`verify`: with no new flags their output must be
/// byte-for-byte what it was, and only the toolkit's own `info` probe needs it.
pub fn print_evidence_keys(threads_requested: usize, threads_effective: usize) {
    println!("provider=tprims");
    println!("provider_version={}", provider_version());
    println!("threads={threads_requested}");
    println!("threads_effective={threads_effective}");
    #[cfg(feature = "tblis")]
    println!("tblis={}", crate::tblis::VERSION);
    #[cfg(not(feature = "tblis"))]
    println!("tblis=");
}

/// The crate version of the linked tcontract library, read through the capi's
/// own accessor (`TAPP_implementation_version`, the "crate version of the loaded
/// library" by its contract).
///
/// The measured provider itself (`tcontract` / `tcontract-kernel`) exposes no
/// runtime version API, and the harness may not add one outside this directory;
/// this is the only library version it can report without inventing a constant
/// that would drift from `Cargo.toml`.
fn provider_version() -> String {
    let p = tcontract_capi::TAPP_implementation_version();
    if p.is_null() {
        return String::new();
    }
    // SAFETY: the accessor's contract is a static, NUL-terminated string that
    // outlives the call and must not be freed.
    unsafe { std::ffi::CStr::from_ptr(p) }
        .to_string_lossy()
        .into_owned()
}

/// The corpus as this binary sees it: case, dtype, steps, MACs and the
/// `m`/`n`/`k` the rows will carry. `info` prints it so a recorded run can be
/// checked against the corpus it claims to have measured.
pub fn print_corpus() {
    let mut t = Table::new(&["case", "group", "dtype", "steps", "macs", "m", "n", "k"]);
    for p in crate::corpus::corpus() {
        let (m, n, k) = p.report_mnk();
        t.row(vec![
            p.name.clone(),
            p.group.to_string(),
            p.dtype.to_string(),
            p.steps.len().to_string(),
            p.macs.to_string(),
            m.to_string(),
            n.to_string(),
            k.to_string(),
        ]);
    }
    t.print();
}

/// This machine's name, for the `PROVENANCE.txt` convention every directory
/// under the campaign's `data/` keys on.
fn hostname() -> String {
    if let Ok(s) = fs::read_to_string("/proc/sys/kernel/hostname") {
        let s = s.trim();
        if !s.is_empty() {
            return s.to_string();
        }
    }
    posix_hostname().unwrap_or_else(|| "unknown".into())
}

/// `gethostname(2)`, declared rather than pulled in with a `libc` dependency —
/// the same call this crate's BLAS and TBLIS bindings are written as.
fn posix_hostname() -> Option<String> {
    use std::ffi::c_char;

    extern "C" {
        fn gethostname(name: *mut c_char, len: usize) -> i32;
    }

    let mut buf = [0u8; 256];
    // SAFETY: `buf` is 256 writable bytes and the length passed is 255, leaving
    // the final byte zero so the result is NUL-terminated however much of the
    // buffer the call fills.
    let rc = unsafe { gethostname(buf.as_mut_ptr().cast::<c_char>(), buf.len() - 1) };
    if rc != 0 {
        return None;
    }
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    let name = String::from_utf8_lossy(&buf[..end]).trim().to_string();
    (!name.is_empty()).then_some(name)
}

/// Right-aligned fixed-width table printer.
pub struct Table {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl Table {
    pub fn new(headers: &[&str]) -> Self {
        Table {
            headers: headers.iter().map(|s| s.to_string()).collect(),
            rows: Vec::new(),
        }
    }
    pub fn row(&mut self, cells: Vec<String>) {
        self.rows.push(cells);
    }
    pub fn print(&self) {
        let ncol = self.headers.len();
        let mut w: Vec<usize> = self.headers.iter().map(|h| h.len()).collect();
        for r in &self.rows {
            for (i, c) in r.iter().enumerate().take(ncol) {
                w[i] = w[i].max(c.len());
            }
        }
        let line: String = w
            .iter()
            .map(|&x| "-".repeat(x + 2))
            .collect::<Vec<_>>()
            .join("+");
        let fmt = |cells: &[String]| -> String {
            cells
                .iter()
                .enumerate()
                .take(ncol)
                .map(|(i, c)| format!(" {:>width$} ", c, width = w[i]))
                .collect::<Vec<_>>()
                .join("|")
        };
        println!("{}", fmt(&self.headers));
        println!("{line}");
        for r in &self.rows {
            println!("{}", fmt(r));
        }
    }
}

/// Quote a CSV field if it would otherwise break the record. `tcbench` writes
/// its fields bare and its corpus labels happen to be comma-free; a program's
/// per-step note does not have to be, and a shifted column is a silently wrong
/// row in a published report.
fn field(v: impl AsRef<str>) -> String {
    let v = v.as_ref();
    if v.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", v.replace('"', "\"\""))
    } else {
        v.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real corpus case, for a census declared from actual case ids.
    fn case(name: &str) -> Program {
        crate::corpus::corpus()
            .into_iter()
            .find(|p| p.name == name)
            .expect("corpus case")
    }

    /// One measured row for `case`, with four chronological samples whose
    /// median is 0.275 s.
    fn measured(case: &Program, engine: &str, notes: &str) -> Row {
        let (m, n, k) = case.report_mnk();
        Row {
            case: case.name.clone(),
            dtype: case.dtype.to_string(),
            engine: engine.into(),
            threads: 4,
            m,
            n,
            k,
            macs: case.macs,
            secs: 0.275,
            spread: 0.075,
            prepare_s: 0.00000042,
            gflops: 1.5,
            reg_a: 0.0,
            reg_b: 0.0,
            notes: notes.into(),
            samples: vec![0.20, 0.25, 0.30, 0.40],
        }
    }

    /// The leading columns are `tcbench`'s, byte for byte, followed by the
    /// timing-v2 evidence columns.
    #[test]
    fn csv_header_is_tcbench_s_columns_plus_evidence() {
        let census = Census::declared("default", "", "1", 1, &[], &[], &[]);
        let path = std::env::temp_dir().join("lukbench-report-test-1.csv");
        Results::default()
            .write_csv(&path.to_string_lossy(), &census)
            .unwrap();
        let body = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let mut lines = body.lines();
        assert_eq!(
            lines.next().unwrap(),
            "case,group,dtype,engine,threads,m,n,k,macs,seconds,spread,prepare_s,gflops,regular_a,regular_b,notes,regime,family,order,batch,row,status,note,total_ms,per_item_us,samples_ms,rayon_threads,vendor_threads,case_m,case_n,case_k"
        );
    }

    /// A comma in a free-text field is quoted, not left to shift the row.
    #[test]
    fn csv_quotes_a_field_that_needs_it() {
        let c = case("ij_jk_ik_f64_n64");
        let dtypes = [c.dtype.to_string()];
        let rows = ["plan".to_string()];
        let mut census =
            Census::declared("r", "", "1", 4, std::slice::from_ref(&c), &dtypes, &rows);
        census.mark_ok(&c.name, c.dtype, "plan");
        let mut r = Results {
            batch: "1".into(),
            ..Default::default()
        };
        r.push(measured(&c, "plan", "rel_err=1.00e-16, steps=2"));
        let path = std::env::temp_dir().join("lukbench-report-test-2.csv");
        r.write_csv(&path.to_string_lossy(), &census).unwrap();
        let body = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(body.contains("\"rel_err=1.00e-16, steps=2\""), "{body}");
    }

    /// The census enumerates the whole case × dtype × row product, and the timed
    /// CSV carries the same cells: a wrong-dtype cell is `skipped` in both, and a
    /// measured cell carries the statistic of its chronological samples.
    #[test]
    fn census_declares_the_dtype_product_and_the_timed_csv_matches() {
        let cases: Vec<Program> = crate::corpus::corpus()
            .into_iter()
            .filter(|p| p.name == "ij_jk_ik_c64_n32" || p.name == "ij_jk_ik_f64_n64")
            .collect();
        let dtypes = ["f64".to_string(), "c64".to_string()];
        let rows = ["plan".to_string(), "packed".to_string()];
        let mut census = Census::declared("r", "1", "1", 4, &cases, &dtypes, &rows);
        // Only the one cell `results` below actually measured; every other cell
        // of the product stays `skipped`.
        census.mark_ok("ij_jk_ik_f64_n64", "f64", "plan");

        let mut results = Results {
            regime: "r".into(),
            batch: "1".into(),
            statistic: Statistic::Median,
            n_spec: "1".into(),
            ..Default::default()
        };
        results.push(measured(
            &case("ij_jk_ik_f64_n64"),
            "plan",
            "rel_err=1.00e-16",
        ));

        let path = std::env::temp_dir().join("lukbench-report-test-3.csv");
        results.write_csv(&path.to_string_lossy(), &census).unwrap();
        let body = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 1 + cases.len() * dtypes.len() * rows.len());
        // Identity is the declared spec; the real extents are in case_m..case_k.
        assert!(body.contains(
            "ij_jk_ik_f64_n64,ij_jk_ik,f64,plan,4,1,1,,262144,0.275000000,0.0750,0.000000420,1.5000,0.0000,0.0000,rel_err=1.00e-16,r,ij_jk_ik_f64_n64,,1,plan,ok,,275.000000000,275000.000000000,\"[200.000000000000, 250.000000000000, 300.000000000000, 400.000000000000]\",4,4,64,64,64"
        ), "{body}");
        // The wrong-dtype cells are declared and skipped, in both files.
        assert!(body.contains(
            "ij_jk_ik_c64_n32,ij_jk_ik,f64,plan,4,1,1,,,,,,,,,,r,ij_jk_ik_c64_n32,,1,plan,skipped,case dtype is c64,,,,4,4,32,32,32"
        ), "{body}");
        let census_path = std::env::temp_dir().join("lukbench-report-test-3-census.csv");
        census.write_csv(&census_path.to_string_lossy()).unwrap();
        let census_body = std::fs::read_to_string(&census_path).unwrap();
        let _ = std::fs::remove_file(&census_path);
        assert!(
            census_body.contains("r,ij_jk_ik_c64_n32,f64,,1,1,,1,plan,4,skipped,case dtype is c64"),
            "{census_body}"
        );
    }
}
