//! Result collection, table formatting and CSV output.

use std::fmt::Write as _;
use std::fs;

use crate::corpus::Case;
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
    /// Fraction of A's row blocks and B's column blocks that are regular.
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
/// This harness resolves its extents from the spec (a tensor-size target), so
/// the spec is what the declared grid names; the resolved extents are the
/// harness's own report and are not the toolkit's identity.
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

/// One CSV field, quoted when it carries a comma or a quote so a note or a JSON
/// sample list cannot shift the columns.
fn csv_field(v: impl AsRef<str>) -> String {
    let v = v.as_ref();
    if v.contains(',') || v.contains('"') {
        format!("\"{}\"", v.replace('"', "\"\""))
    } else {
        v.to_string()
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

    /// The seconds/GFLOP/s columns, with the timing-v2 evidence columns appended
    /// at the end so a consumer that reads the old header still works.
    ///
    /// The row plan is the census, not `rows`: a declared cell that was skipped
    /// still has to appear, with its status and the same note the census carries,
    /// or the toolkit's timed/verified cell sets cannot be compared. The rows are
    /// emitted in the census order, which is the declared rotation.
    pub fn write_csv(&self, path: &str, census: &Census) -> std::io::Result<()> {
        let mut s = String::from(
            "case,group,dtype,engine,threads,m,n,k,macs,seconds,spread,prepare_s,gflops,regular_a,regular_b,notes,regime,family,order,batch,row,status,note,total_ms,per_item_us,samples_ms,rayon_threads,vendor_threads\n",
        );
        let batch: f64 = self.batch.parse().unwrap_or(1.0);
        for c in &census.rows {
            let (m, n, k) = declared_shape(&self.n_spec, &c.family);
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
                let r = self
                    .get(&c.family, &c.dtype, &c.row)
                    .expect("a census cell marked ok was measured");
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
                ]);
            }
            let _ = writeln!(
                s,
                "{}",
                fields.iter().map(csv_field).collect::<Vec<_>>().join(",")
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
}

/// The element types the harness can actually drive. A declared dtype outside
/// this set is a `skipped` census cell with that reason rather than a silent
/// absence.
const KNOWN_DTYPES: [&str; 4] = ["f32", "f64", "c32", "c64"];

/// The declared run grid: every case × requested dtype × engine row, with the
/// status of the cells that were actually measured. Written by `run` and
/// `verify` alike so a consumer can tell a skipped cell from a missing one, and
/// used by `run` as the row plan for the timed CSV so both carry the same cells
/// in the same rotated order.
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
        cases: &[Case],
        dtypes: &[String],
        rows: &[String],
    ) -> Census {
        let mut cells = Vec::new();
        for case in cases {
            for dtype in dtypes {
                let unsupported = !KNOWN_DTYPES.contains(&dtype.as_str());
                for row in rows {
                    cells.push(CensusRow {
                        regime: regime.to_string(),
                        family: case.name.to_string(),
                        group: case.group.to_string(),
                        dtype: dtype.clone(),
                        batch: batch.to_string(),
                        row: row.clone(),
                        threads,
                        status: "skipped",
                        note: if unsupported {
                            format!("unsupported dtype {dtype}")
                        } else {
                            "not measured".to_string()
                        },
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
                fields.iter().map(csv_field).collect::<Vec<_>>().join(",")
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
    // The aarch64 counterpart. `neon` is architecturally guaranteed on aarch64
    // rather than detected, and is printed anyway: the line's job is to say what
    // a kernel *could* use on this machine, and a missing line reads as "nothing
    // was checked". The rest are optional extensions.
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
    // The cache geometry the analytical blocking model reads, and which source
    // answered. A blocking parameter that came out of a probe should be
    // traceable to it, and a fallback to the built-in defaults — which would
    // make every derived parameter conservative — should be visible here rather
    // than inferred from a disappointing number.
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
            "caches      : {} (via {}), blocking={}",
            levels.join(" "),
            h.source.name(),
            crate::knobs::get().block_model().name()
        );
        // How many L3 domains this run's threads span, which is what the
        // domain-aware partition gate turns on (A36). Printed as a small table
        // rather than one number because the whole point is that it is a function
        // of the thread count and of the machine, and a session that quotes "16
        // domains" without saying at what width is quoting nothing. `l3_domains`
        // and not `cores_sharing` directly, so a forced domain count shows up here
        // when one is set.
        let widths: Vec<usize> = [1, 4, 8, 16, 32, 64, 128]
            .into_iter()
            .filter(|&t| {
                t == 1 || t <= 2 * std::thread::available_parallelism().map_or(1, |n| n.get())
            })
            .collect();
        let spans: Vec<String> = widths
            .iter()
            .map(|&t| {
                format!(
                    "t{t}={}",
                    cache::l3_domains(t, crate::knobs::get().l3_domains())
                )
            })
            .collect();
        println!("l3 domains  : {}", spans.join(" "));
    }
    // Which BLAS, not just whether one. Accelerate's GEMM reaches Apple's AMX
    // coprocessor and OpenBLAS's does not, so an efficiency ratio against one is
    // not the same quantity as a ratio against the other, and the difference has
    // to be legible in the run that produced the number rather than recovered
    // from a build command later.
    #[cfg(feature = "blas")]
    let blas = crate::blas::IMPL;
    #[cfg(not(feature = "blas"))]
    let blas = "false";
    println!(
        "baselines   : tblis={} blas={blas}",
        cfg!(feature = "tblis")
    );
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

/// This machine's name, for the `PROVENANCE.txt` convention every directory
/// under `bench-results/` keys on.
///
/// Linux's `/proc` file first, so the string is byte-identical to what every
/// committed run recorded, then `gethostname(2)` — which is what makes this work
/// on Darwin, where the `/proc` read fails and the host used to print as
/// `unknown`. A run labelled `unknown` is a run whose numbers cannot be traced
/// to a machine, and this project's first rule is that no number from one
/// machine may be compared with a number from another.
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

    // `_POSIX_HOST_NAME_MAX` is 255; the extra byte is the terminator the call
    // is not required to write when the name exactly fills the buffer.
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
