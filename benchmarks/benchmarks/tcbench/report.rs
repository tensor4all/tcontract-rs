//! Result collection, table formatting and CSV output.

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
    /// Fraction of A's row blocks and B's column blocks that are regular.
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
                r.case,
                r.group,
                r.dtype,
                r.engine,
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
                r.notes
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
