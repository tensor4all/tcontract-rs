//! Result collection, table formatting and CSV output.
//!
//! The CSV columns are exactly `tcbench`'s: the campaign's report reads one
//! column set for both harnesses and a second shape would be a second
//! contract.

use std::fmt::Write as _;
use std::fs;

/// One measurement.
#[derive(Clone, Debug)]
pub struct Row {
    pub case: String,
    pub group: String,
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
}

#[derive(Default)]
pub struct Results {
    pub rows: Vec<Row>,
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

    pub fn write_csv(&self, path: &str) -> std::io::Result<()> {
        let mut s = String::from(
            "case,group,dtype,engine,threads,m,n,k,macs,seconds,spread,prepare_s,gflops,regular_a,regular_b,notes\n",
        );
        for r in &self.rows {
            let _ = writeln!(
                s,
                "{},{},{},{},{},{},{},{},{},{:.9},{:.4},{:.9},{:.4},{:.4},{:.4},{}",
                field(&r.case),
                field(&r.group),
                field(&r.dtype),
                field(&r.engine),
                r.threads,
                r.m,
                r.n,
                r.k,
                r.macs,
                r.secs,
                r.spread,
                r.prepare_s,
                r.gflops,
                r.reg_a,
                r.reg_b,
                field(&r.notes)
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
fn field(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(group: &str, notes: &str) -> Row {
        Row {
            case: "c".into(),
            group: group.into(),
            dtype: "f64".into(),
            engine: "plan".into(),
            threads: 1,
            m: 2,
            n: 3,
            k: 4,
            macs: 5,
            secs: 0.25,
            spread: 0.075,
            prepare_s: 0.00000042,
            gflops: 1.5,
            reg_a: 0.0,
            reg_b: 0.0,
            notes: notes.into(),
        }
    }

    /// The header is `tcbench`'s column set, byte for byte.
    #[test]
    fn csv_header_is_tcbench_s_columns() {
        let mut r = Results::default();
        r.push(row("g", "rel_err=1.00e-16"));
        let path = std::env::temp_dir().join("lukbench-report-test-1.csv");
        r.write_csv(&path.to_string_lossy()).unwrap();
        let body = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let mut lines = body.lines();
        assert_eq!(
            lines.next().unwrap(),
            "case,group,dtype,engine,threads,m,n,k,macs,seconds,spread,prepare_s,gflops,regular_a,regular_b,notes"
        );
        assert_eq!(
            lines.next().unwrap(),
            "c,g,f64,plan,1,2,3,4,5,0.250000000,0.0750,0.000000420,1.5000,0.0000,0.0000,rel_err=1.00e-16"
        );
    }

    /// A comma in a free-text field is quoted, not left to shift the row.
    #[test]
    fn csv_quotes_a_field_that_needs_it() {
        let mut r = Results::default();
        r.push(row("g", "rel_err=1.00e-16, steps=2"));
        let path = std::env::temp_dir().join("lukbench-report-test-2.csv");
        r.write_csv(&path.to_string_lossy()).unwrap();
        let body = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(body.contains("\"rel_err=1.00e-16, steps=2\""), "{body}");
    }
}
