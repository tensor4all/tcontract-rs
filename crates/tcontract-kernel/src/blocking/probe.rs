//! Hardware cache descriptors, probed at run time: sysfs, `CPUID`, `sysctl`,
//! then a built-in fallback. The analytical blocking model they feed is in
//! [`super::model`]. These are immutable hardware facts; nothing here reads
//! the environment.
//!
//! Hardware-fact helpers shared by both: [`CacheHierarchy`], [`hierarchy`],
//! [`l3_domains`].

// ---------------------------------------------------------------------------
// Descriptors
// ---------------------------------------------------------------------------

/// One level of data cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheLevel {
    /// 1, 2 or 3.
    pub level: u8,
    /// Total size in bytes.
    pub size: usize,
    /// Line size in bytes (`C_Li` in the paper).
    pub line: usize,
    /// Associativity (`W_Li`).
    pub ways: usize,
    /// Number of sets (`N_Li`).
    pub sets: usize,
    /// How many *logical* CPUs share this cache. Load-bearing: it is what
    /// decides whether a thread count divides a budget or not.
    pub shared_by: usize,
}

impl CacheLevel {
    /// Bytes in one way, i.e. `N_Li * C_Li`. Every footprint in the model is
    /// expressed in these units, because the model reasons in whole ways.
    pub const fn bytes_per_way(&self) -> usize {
        self.sets * self.line
    }
}

/// Where the descriptors came from. Reported by `tcbench info` so a number can
/// be traced to a probe rather than to a guess.
///
/// `#[non_exhaustive]`: the set of probes grows with the targets this runs on —
/// a BSD `sysctl`, a hypervisor's topology table, a `/proc/cpuinfo` reader for
/// a machine whose sysfs is unmounted — and nothing downstream has to *service*
/// a source, only report it, so a catch-all arm is a legitimate answer here in a
/// way it is not for [`crate::ComplexMethod`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CacheSource {
    /// Linux sysfs caches of the first CPU allowed at probe time (CPU0 when
    /// affinity information is unavailable). The only source that
    /// reports cache *sharing* directly, and it needs no `unsafe`.
    Sysfs,
    /// x86 `CPUID` leaf 4, or `0x8000001D` on AMD.
    Cpuid,
    /// Darwin `sysctl`. Reports sizes, the line and L2 sharing, but **no
    /// associativity and no set count** — see `from_sysctl` in this module.
    Sysctl,
    /// The conservative built-in fallback.
    Builtin,
}

impl CacheSource {
    /// Short name for reports: `"sysfs"`, `"cpuid"`, `"sysctl"` or `"builtin"`.
    pub fn name(self) -> &'static str {
        match self {
            CacheSource::Sysfs => "sysfs",
            CacheSource::Cpuid => "cpuid",
            CacheSource::Sysctl => "sysctl",
            CacheSource::Builtin => "builtin",
        }
    }
}

/// [`CacheSource::name`]'s spelling. No [`FromStr`](core::str::FromStr) to pair
/// with it: the source is something the probe reports, never something a caller
/// asks for.
impl core::fmt::Display for CacheSource {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.pad(self.name())
    }
}

/// The data cache hierarchy of one core.
///
/// L1d is mandatory — without it there is no model — so the probe falls back to
/// [`BUILTIN`] rather than reporting nothing. L2 and L3 are optional because
/// plenty of targets lack one or both, and the model degrades level by level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheHierarchy {
    /// The L1 data cache, which fixes `kc`.
    pub l1d: CacheLevel,
    /// The L2, which fixes `mc`. `None` leaves `mc` at the model's fallback.
    pub l2: Option<CacheLevel>,
    /// The L3, which fixes `nc`. `None` on most non-server parts.
    pub l3: Option<CacheLevel>,
    /// Which probe produced these numbers, so a blocking parameter can be
    /// traced back to a measurement of the machine rather than to [`BUILTIN`].
    pub source: CacheSource,
}

/// Conservative defaults: a small 32 KiB L1d, a 256 KiB L2 and an 8 MiB L3,
/// all with plausible geometry. Deliberately smaller than any machine this is
/// likely to run on — under-blocking costs some bandwidth, over-blocking falls
/// off a cliff — and never a reason to fail a contraction.
pub const BUILTIN: CacheHierarchy = CacheHierarchy {
    l1d: CacheLevel {
        level: 1,
        size: 32 * 1024,
        line: 64,
        ways: 8,
        sets: 64,
        shared_by: 1,
    },
    l2: Some(CacheLevel {
        level: 2,
        size: 256 * 1024,
        line: 64,
        ways: 8,
        sets: 512,
        shared_by: 1,
    }),
    l3: Some(CacheLevel {
        level: 3,
        size: 8 * 1024 * 1024,
        line: 64,
        ways: 16,
        sets: 8192,
        shared_by: 4,
    }),
    source: CacheSource::Builtin,
};

impl CacheHierarchy {
    /// Logical CPUs per physical core, inferred from how many share the L1d.
    /// 2 on a hyperthreaded x86, 1 otherwise.
    pub fn threads_per_core(&self) -> usize {
        self.l1d.shared_by.max(1)
    }

    /// How many *physical cores* share `lvl`.
    ///
    /// The model assumes one thread per physical core, which is what every
    /// BLIS-shaped library assumes and what `tensorprimitives/scripts/env.sh` pins. Under that
    /// assumption a private-but-hyperthread-shared L2 (`shared_by == 2` on this
    /// machine) is one core's to itself, while a socket L3 (`shared_by == 16`)
    /// is contended by 8. Oversubscribing the siblings would halve the L2 and
    /// L1 a thread really gets; the project's own measurement rule already
    /// treats that configuration as invalid, so it is not modelled.
    pub fn cores_sharing(&self, lvl: &CacheLevel) -> usize {
        (lvl.shared_by / self.threads_per_core()).max(1)
    }

    /// How many **L3 domains** a run of `threads` threads spans.
    ///
    /// This is the input the partition rule was missing (A36): the shared packed
    /// `B` panel is sized for *an* L3, so whether "shared" means what the design
    /// assumed depends on how many separate L3s the thread set covers. One L3
    /// per socket gives 1 at every thread count up to the socket; a chiplet
    /// machine with a 4-core L3 gives 1 at 4 threads and 16 at 64.
    ///
    /// **Compact placement is assumed**: `threads` threads occupy `threads`
    /// consecutive physical cores, filling one domain before starting the next.
    /// That is what `scripts/phase4f-threads.sh` does by construction and what a
    /// whole-node run does anyway; a *scattered* placement spans more domains
    /// than this reports, and the error is in the safe direction — it under-counts,
    /// so the rule falls back to the behaviour every committed number was measured
    /// with. [`l3_domains`]'s `forced` argument overrides it for exactly that case; see
    /// [`l3_domains`].
    ///
    /// A machine with no L3 at all has nothing shared to spread, so every core is
    /// its own domain.
    pub fn l3_domains(&self, threads: usize) -> usize {
        let threads = threads.max(1);
        match &self.l3 {
            Some(l3) => threads.div_ceil(self.cores_sharing(l3)),
            None => threads,
        }
    }
}

/// How many L3 domains a `threads`-wide run spans on *this* machine.
///
/// [`CacheHierarchy::l3_domains`] over [`hierarchy`], unless `forced` overrides
/// the derivation. The override exists because the derivation assumes compact
/// placement: it is how a scattered cpuset (the same thread count spread
/// one-per-domain instead of packed) can be measured against a packed one
/// without a rebuild, and how the rule is exercised on a machine that has only
/// one domain. A `forced` of zero is ignored.
pub fn l3_domains(threads: usize, forced: Option<usize>) -> usize {
    match forced.filter(|&n| n > 0) {
        Some(n) => n.min(threads.max(1)),
        None => hierarchy().l3_domains(threads),
    }
}

/// The cache hierarchy of this machine: sysfs, then `CPUID`, then [`BUILTIN`].
///
/// Probed once per process and cached. On Linux, sysfs uses the first CPU in
/// the calling thread's initial affinity mask, not necessarily CPU0. Changing
/// affinity later does not re-probe; mixed/scattered placement still needs an
/// explicit domain override. A probe that fails is never an error: it degrades to
/// the next source, and the last source always succeeds.
pub fn hierarchy() -> CacheHierarchy {
    #[cfg(feature = "std")]
    {
        use std::sync::OnceLock;
        static H: OnceLock<CacheHierarchy> = OnceLock::new();
        *H.get_or_init(probe)
    }
    #[cfg(not(feature = "std"))]
    {
        probe()
    }
}

fn probe() -> CacheHierarchy {
    #[cfg(feature = "std")]
    if let Some(h) = probe_sysfs() {
        return h;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if let Some(h) = probe_cpuid() {
        return h;
    }
    // After `CPUID`, not before: an Intel Mac has both, and `CPUID` reports the
    // associativity and set count that `sysctl` does not. This arm is what an
    // Apple Silicon machine reaches.
    #[cfg(all(feature = "std", target_os = "macos"))]
    if let Some(h) = probe_sysctl() {
        return h;
    }
    BUILTIN
}

// ---------------------------------------------------------------------------
// Source 1: Linux sysfs
// ---------------------------------------------------------------------------

/// The attributes of one `/sys/.../cache/index*` directory, as raw text.
///
/// Parsing is expressed against this rather than against the filesystem so it
/// can be tested on fixtures — the layout of sysfs is not something a unit test
/// should need a particular kernel to exercise.
#[cfg(feature = "std")]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SysfsIndex<'a> {
    pub level: &'a str,
    pub kind: &'a str,
    pub size: &'a str,
    pub ways: &'a str,
    pub line: &'a str,
    /// `number_of_sets`, optional: derived from size/line/ways when absent.
    pub sets: &'a str,
    pub shared: &'a str,
}

/// Assemble a hierarchy from sysfs text. `None` if there is no usable L1d.
#[cfg(feature = "std")]
pub(crate) fn from_sysfs(indices: &[SysfsIndex<'_>]) -> Option<CacheHierarchy> {
    let mut h = CacheHierarchy {
        l1d: BUILTIN.l1d,
        l2: None,
        l3: None,
        source: CacheSource::Sysfs,
    };
    let mut have_l1d = false;
    for idx in indices {
        // Skip the instruction cache: only data caches hold packed panels.
        // "Unified" counts, which is what L2/L3 normally report.
        if idx.kind.trim().eq_ignore_ascii_case("Instruction") {
            continue;
        }
        let Some(lvl) = parse_level(idx) else {
            continue;
        };
        match lvl.level {
            1 if !have_l1d => {
                h.l1d = lvl;
                have_l1d = true;
            }
            2 if h.l2.is_none() => h.l2 = Some(lvl),
            3 if h.l3.is_none() => h.l3 = Some(lvl),
            _ => {}
        }
    }
    have_l1d.then_some(h)
}

#[cfg(feature = "std")]
fn parse_level(idx: &SysfsIndex<'_>) -> Option<CacheLevel> {
    let level: u8 = idx.level.trim().parse().ok()?;
    if !(1..=3).contains(&level) {
        return None;
    }
    let size = parse_size(idx.size)?;
    let line = parse_num(idx.line)?;
    let ways = parse_num(idx.ways)?;
    if size == 0 || line == 0 || ways == 0 {
        return None;
    }
    // `number_of_sets` is what the model wants; derive it when the kernel does
    // not export it, which some architectures do not.
    let sets = parse_num(idx.sets).unwrap_or(0);
    let sets = if sets > 0 { sets } else { size / (line * ways) };
    if sets == 0 {
        return None;
    }
    Some(CacheLevel {
        level,
        size,
        line,
        ways,
        sets,
        shared_by: parse_cpu_list(idx.shared).unwrap_or(1),
    })
}

#[cfg(feature = "std")]
fn parse_num(s: &str) -> Option<usize> {
    s.trim().parse().ok()
}

/// sysfs sizes carry a unit suffix: `32K`, `1024K`, `25344K`.
#[cfg(feature = "std")]
fn parse_size(s: &str) -> Option<usize> {
    let s = s.trim();
    let (digits, mult) = match s.chars().last()? {
        'K' | 'k' => (&s[..s.len() - 1], 1024),
        'M' | 'm' => (&s[..s.len() - 1], 1024 * 1024),
        'G' | 'g' => (&s[..s.len() - 1], 1024 * 1024 * 1024),
        _ => (s, 1),
    };
    digits.trim().parse::<usize>().ok()?.checked_mul(mult)
}

/// One decimal component of a CPU list, rejecting anything a CPU list cannot
/// contain. `str::parse::<usize>` alone accepts a leading `+`, which is not
/// part of the grammar, so the sign is checked explicitly.
#[cfg(feature = "std")]
fn cpu_number(s: &str) -> Option<usize> {
    let s = s.trim();
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

/// Count the CPUs in a CPU list: `0,16` is 2, `0-7,16-23` is 16.
///
/// Used for both `shared_cpu_list` and `Cpus_allowed_list`; the latter is also
/// *validated* with it, so the grammar is strict: every comma-separated
/// component must be a number or an ascending `low-high` range, with no empty
/// or sign-prefixed component. A list that does not parse is `None`, which the
/// callers turn into the CPU0 fallback rather than a guess, and the running
/// total is accumulated with `checked_add` so a hostile list cannot wrap.
///
/// This is the one piece of information no other source reports as directly,
/// and the model needs it to know whether an L3 budget is one core's or a
/// socket's.
#[cfg(feature = "std")]
fn parse_cpu_list(s: &str) -> Option<usize> {
    let mut n = 0usize;
    for part in s.trim().split(',') {
        let part = part.trim();
        if part.is_empty() {
            return None;
        }
        let count = match part.split_once('-') {
            // `low-high`, inclusive; a descending range is not a CPU list.
            Some((a, b)) => cpu_number(b)?.checked_sub(cpu_number(a)?)?,
            None => {
                cpu_number(part)?;
                0
            }
        };
        n = n.checked_add(count)?.checked_add(1)?;
    }
    (n > 0).then_some(n)
}

/// `/proc/thread-self/status` lists allowed CPUs in ascending order. Probe a
/// permitted CPU so a pinned process on a heterogeneous host does not inherit
/// CPU0's unrelated cache geometry. This is sampled at most once per process,
/// through [`hierarchy`], and never in a hot loop.
#[cfg(feature = "std")]
fn affinity_cpu(status: &str) -> Option<usize> {
    let list = status
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))?
        .trim();
    parse_cpu_list(list)?;
    cpu_number(list.split([',', '-']).next()?)
}

#[cfg(feature = "std")]
fn probe_sysfs() -> Option<CacheHierarchy> {
    let cpu = std::fs::read_to_string("/proc/thread-self/status")
        .ok()
        .and_then(|status| affinity_cpu(&status))
        .unwrap_or(0);
    let base = format!("/sys/devices/system/cpu/cpu{cpu}/cache");
    let read = |dir: &str, name: &str| -> String {
        std::fs::read_to_string(format!("{dir}/{name}")).unwrap_or_default()
    };
    let mut raw: Vec<[String; 7]> = Vec::new();
    // `index*` directories are numbered contiguously from 0; stop at the first
    // gap. The bound is a safety net, not a real limit — no CPU has 16 levels.
    for i in 0..16 {
        let dir = format!("{base}/index{i}");
        let level = read(&dir, "level");
        if level.trim().is_empty() {
            break;
        }
        raw.push([
            level,
            read(&dir, "type"),
            read(&dir, "size"),
            read(&dir, "ways_of_associativity"),
            read(&dir, "coherency_line_size"),
            read(&dir, "number_of_sets"),
            read(&dir, "shared_cpu_list"),
        ]);
    }
    let idx: Vec<SysfsIndex<'_>> = raw
        .iter()
        .map(|r| SysfsIndex {
            level: &r[0],
            kind: &r[1],
            size: &r[2],
            ways: &r[3],
            line: &r[4],
            sets: &r[5],
            shared: &r[6],
        })
        .collect();
    from_sysfs(&idx)
}

// ---------------------------------------------------------------------------
// Source 2: x86 CPUID
// ---------------------------------------------------------------------------

/// Decode one deterministic-cache-parameters leaf (`CPUID.4` on Intel,
/// `CPUID.8000001D` on AMD — the two use the same register encoding).
///
/// Returns the cache type (1 data, 2 instruction, 3 unified) and the level,
/// or `None` for the null subleaf that terminates the enumeration. Split out
/// from the `CPUID` call so the bit-twiddling can be unit-tested.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) fn decode_cache_leaf(eax: u32, ebx: u32, ecx: u32) -> Option<(u32, CacheLevel)> {
    let kind = eax & 0x1f;
    if kind == 0 {
        return None; // null subleaf: enumeration over
    }
    let level = ((eax >> 5) & 0x7) as u8;
    let shared_by = (((eax >> 14) & 0xfff) + 1) as usize;
    let ways = (((ebx >> 22) & 0x3ff) + 1) as usize;
    let partitions = (((ebx >> 12) & 0x3ff) + 1) as usize;
    let line = ((ebx & 0xfff) + 1) as usize;
    let sets = (ecx as usize) + 1;
    Some((
        kind,
        CacheLevel {
            level,
            size: ways * partitions * line * sets,
            line,
            ways,
            // The model reasons in `sets * line` bytes per way, which is only
            // the true way size when a line is one partition. Fold the
            // partition count in so `bytes_per_way * ways == size` holds.
            sets: sets * partitions,
            shared_by,
        },
    ))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn probe_cpuid() -> Option<CacheHierarchy> {
    #[cfg(target_arch = "x86")]
    use core::arch::x86 as arch;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64 as arch;

    // `__cpuid_count` became a *safe* function only in Rust 1.94 (stdarch #1935
    // — every x86 target feature postdates `CPUID`, so its availability is
    // implied). It is `unsafe` on the workspace MSRV of 1.89, so the block is
    // required there; `unused_unsafe` is allowed because the same block is
    // redundant from 1.94 on, and CI compiles with `-D warnings` on both.
    //
    // Keeping the block rather than raising the MSRV is deliberate: D20 chose
    // 1.89 because that is where the AVX-512 intrinsics stabilised, and paying
    // five Rust releases of compatibility for two braces is the wrong trade.
    //
    // When the MSRV does reach 1.94, the `allow` and the `unsafe` block come out
    // **together**. Dropping only the `allow` is a hard error on 1.89..1.94;
    // dropping only the block leaves a bare `allow` suppressing nothing, which
    // will outlast anyone's memory of why it was there.
    //
    // SAFETY: `CPUID` is unconditionally available on every x86 CPU that can
    // run this code — it predates every target feature the dispatch tests for —
    // and the instruction only reads processor identification registers.
    #[allow(unused_unsafe)]
    let leaf = |leaf: u32, sub: u32| unsafe { arch::__cpuid_count(leaf, sub) };

    let max_basic = leaf(0, 0).eax;
    let max_ext = leaf(0x8000_0000, 0).eax;
    // Leaf 4 is Intel's; AMD mirrors the same encoding at 0x8000001D. Try the
    // basic one first and fall through when it enumerates nothing.
    let candidates = [
        (max_basic >= 4).then_some(4u32),
        (max_ext >= 0x8000_001D).then_some(0x8000_001D),
    ];

    for base in candidates.into_iter().flatten() {
        let mut h = CacheHierarchy {
            l1d: BUILTIN.l1d,
            l2: None,
            l3: None,
            source: CacheSource::Cpuid,
        };
        let mut have_l1d = false;
        for sub in 0..16 {
            let r = leaf(base, sub);
            let Some((kind, lvl)) = decode_cache_leaf(r.eax, r.ebx, r.ecx) else {
                break;
            };
            if kind == 2 {
                continue; // instruction cache
            }
            match lvl.level {
                1 if !have_l1d => {
                    h.l1d = lvl;
                    have_l1d = true;
                }
                2 if h.l2.is_none() => h.l2 = Some(lvl),
                3 if h.l3.is_none() => h.l3 = Some(lvl),
                _ => {}
            }
        }
        if have_l1d {
            return Some(h);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Source 3: Darwin sysctl
// ---------------------------------------------------------------------------

/// Associativity assumed for a `sysctl`-probed level.
///
/// **This is not a probe result, and it is the one field here that is invented.**
/// Darwin exports no associativity and no set count for any cache at any
/// perflevel, so the geometry the analytical model reasons in — `sets * line`
/// bytes per way — cannot be read off the machine. 8 is assumed and the sets
/// derived from it, which preserves `bytes_per_way * ways == size`, the
/// invariant every other source maintains and every level of the sysfs fixture
/// is asserted against.
///
/// The consequence is bounded, which is why assuming is preferable to declining
/// the whole probe: `size`, `line` and `shared_by` are all real, and those are
/// the only fields the **default** (`legacy`) path consults. `ways` and `sets`
/// reach only [`analytical`], which is not the default and is refuted as a
/// portability fix on two machines (A33).
#[cfg(all(feature = "std", target_os = "macos"))]
const ASSUMED_WAYS: usize = 8;

/// What `sysctl` reports about this machine's data caches, in bytes.
///
/// Assembly is expressed against this rather than against `sysctlbyname` so it
/// can be tested on a fixture, the same way [`from_sysfs`] is — the layout of a
/// heterogeneous Apple part is not something a unit test should need that part
/// to exercise.
#[cfg(all(feature = "std", target_os = "macos"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SysctlCaches {
    /// `hw.cachelinesize`. 128 on Apple Silicon, 64 on an Intel Mac.
    pub line: Option<usize>,
    /// L1 data cache of one core.
    pub l1d: Option<usize>,
    /// L2 of one core cluster.
    pub l2: Option<usize>,
    /// How many CPUs share that L2 (`hw.perflevelN.cpusperl2`).
    pub cpus_per_l2: Option<usize>,
}

/// Assemble a hierarchy from `sysctl` numbers. `None` without a usable L1d.
///
/// **`l3` is always `None`, and that is a modelling decision rather than a gap
/// in the probe.** Darwin reports no L3 on Apple Silicon because there is no
/// conventional one: a P-core cluster's L2 is the last level the OS names, and
/// the ~48 MiB system level cache behind it is not exported at all. Reporting
/// the cluster L2 as both L2 *and* L3 would double-count it in every budget the
/// model computes, so the L2 is reported as what it is — an L2 shared by
/// `cpus_per_l2` cores — and the L3 is absent.
///
/// The consequence lands on [`CacheHierarchy::l3_domains`], which treats a
/// machine with no L3 as one domain per thread. That is the documented
/// behaviour for a no-L3 part, but on Apple Silicon it *over-counts*: six cores
/// really do share 16 MiB, so a 12-thread run spans two hardware domains and
/// this reports twelve. Nothing single-threaded can observe it — `l3_domains(1)`
/// is 1 and D44's gate never fires — and a forced domain count expresses
/// the physical count for anyone who threads. Which of the two is the right
/// input to the partition rule on this topology is unmeasured, and picking one
/// on a guess is what the override is for.
#[cfg(all(feature = "std", target_os = "macos"))]
pub(crate) fn from_sysctl(raw: &SysctlCaches) -> Option<CacheHierarchy> {
    let line = raw.line.filter(|&l| l > 0)?;
    let level = |level: u8, size: usize, shared_by: usize| -> Option<CacheLevel> {
        let ways = ASSUMED_WAYS;
        let sets = size / (line * ways);
        (sets > 0).then_some(CacheLevel {
            level,
            // Round down to what the assumed geometry can express exactly, so
            // `bytes_per_way() * ways == size` holds rather than nearly holds.
            // Every real cache size here is a power of two and divides cleanly;
            // the rounding exists so a machine reporting something odd cannot
            // put the model into an inconsistent state.
            size: sets * ways * line,
            line,
            ways,
            sets,
            shared_by: shared_by.max(1),
        })
    };
    // L1d is private to a core, and Apple Silicon has no SMT, so one CPU shares
    // it. That makes `threads_per_core()` 1, which is correct here and is what
    // `cores_sharing` divides by.
    let l1d = level(1, raw.l1d.filter(|&s| s > 0)?, 1)?;
    let l2 = raw
        .l2
        .filter(|&s| s > 0)
        .and_then(|s| level(2, s, raw.cpus_per_l2.unwrap_or(1)));
    Some(CacheHierarchy {
        l1d,
        l2,
        l3: None,
        source: CacheSource::Sysctl,
    })
}

/// Read one integer `sysctl` by name, or `None` if the key does not exist.
#[cfg(all(feature = "std", target_os = "macos"))]
fn sysctl_usize(name: &str) -> Option<usize> {
    use core::ffi::{c_char, c_int, c_void};

    // Declared rather than taken from `libc`: this crate has one
    // dependency (`num-complex`) and adding a second for four cache sizes is the
    // wrong trade. `sysctlbyname` is in libSystem, which every Darwin target
    // links unconditionally, and the bench crate already binds its C baselines
    // this way.
    extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *const c_void,
            newlen: usize,
        ) -> c_int;
    }

    // The key must be NUL-terminated. Built here rather than with `CString` so
    // this stays allocation-simple and cannot fail on an interior NUL: a
    // caller-supplied name with one would just miss the key and return `None`.
    let mut key = String::with_capacity(name.len() + 1);
    key.push_str(name);
    key.push('\0');

    // Zero-initialised and read as 8 bytes: these keys are a mix of 32- and
    // 64-bit, the kernel writes only as many as it has and reports the count in
    // `len`, and the untouched high bytes stay zero on a little-endian target.
    // Darwin is little-endian on every architecture it ships on.
    let mut value = 0u64;
    let mut len = core::mem::size_of::<u64>();

    // SAFETY: `key` is NUL-terminated and outlives the call. `oldp` points at
    // `value`, an 8-byte object, and `oldlenp` says so, which is the contract
    // `sysctlbyname` checks before writing — it writes at most `len` bytes and
    // fails with ENOMEM rather than overrunning. `newp`/`newlen` are the
    // documented null pair for a read.
    let rc = unsafe {
        sysctlbyname(
            key.as_ptr().cast::<c_char>(),
            core::ptr::addr_of_mut!(value).cast::<c_void>(),
            &mut len,
            core::ptr::null(),
            0,
        )
    };
    if rc != 0 || !(len == 4 || len == 8) {
        return None;
    }
    usize::try_from(value).ok()
}

#[cfg(all(feature = "std", target_os = "macos"))]
fn probe_sysctl() -> Option<CacheHierarchy> {
    // **Prefer `perflevel0`, and this is the trap the naive read falls into.**
    // On a heterogeneous Apple part the unprefixed `hw.l1dcachesize` and
    // `hw.l2cachesize` report the *last* perflevel — the efficiency cores. On an
    // M3 Max they answer 64 KiB and 4 MiB for a machine whose P-cores have
    // 128 KiB and 16 MiB, so reading them would under-block by 2x and 4x while
    // looking like a successful probe. `perflevel0` is the performance cores,
    // which is where a contraction runs unless something has deliberately put it
    // elsewhere; the unprefixed keys remain as the fallback for a single-perflevel
    // Mac, where they are the same numbers.
    let first = |a: &str, b: &str| sysctl_usize(a).or_else(|| sysctl_usize(b));
    let raw = SysctlCaches {
        line: sysctl_usize("hw.cachelinesize"),
        l1d: first("hw.perflevel0.l1dcachesize", "hw.l1dcachesize"),
        l2: first("hw.perflevel0.l2cachesize", "hw.l2cachesize"),
        cpus_per_l2: sysctl_usize("hw.perflevel0.cpusperl2"),
    };
    from_sysctl(&raw)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// This machine (`ccqlin038`, Xeon Gold 6244), as its sysfs reports it: a
    /// 32 KiB 8-way L1d shared by a hyperthread pair, a 1 MiB 16-way L2
    /// likewise, and a 25344 KiB 11-way L3 shared by a whole socket. Written
    /// out rather than parsed so that the model's tests do not depend on the
    /// probe's, and so that the probe has something exact to be checked against.
    pub(crate) const CASCADE: CacheHierarchy = CacheHierarchy {
        l1d: CacheLevel {
            level: 1,
            size: 32 * 1024,
            line: 64,
            ways: 8,
            sets: 64,
            shared_by: 2,
        },
        l2: Some(CacheLevel {
            level: 2,
            size: 1024 * 1024,
            line: 64,
            ways: 16,
            sets: 1024,
            shared_by: 2,
        }),
        l3: Some(CacheLevel {
            level: 3,
            size: 25344 * 1024,
            line: 64,
            ways: 11,
            sets: 36864,
            shared_by: 16,
        }),
        source: CacheSource::Sysfs,
    };

    pub(crate) fn cascade_lake() -> CacheHierarchy {
        CASCADE
    }

    #[cfg(feature = "std")]
    #[test]
    fn sysfs_probe_uses_an_allowed_cpu_not_unconditional_cpu0() {
        assert_eq!(
            affinity_cpu("Name:\ttcbench\nCpus_allowed_list:\t4-11\n"),
            Some(4)
        );
        assert_eq!(affinity_cpu("Cpus_allowed_list:\t8,10-11\n"), Some(8));
        assert_eq!(affinity_cpu("Cpus_allowed_list:\t0-23\n"), Some(0));
        for status in [
            "",
            "Cpus_allowed_list:\t",
            "Cpus_allowed_list:\tbad",
            "Cpus_allowed_list:\t11-4",
            // Malformed components must not be skipped into a plausible
            // answer: each of these previously yielded `Some(4)`.
            "Cpus_allowed_list:\t4,\n",
            "Cpus_allowed_list:\t4,,11\n",
            "Cpus_allowed_list:\t+4\n",
            "Cpus_allowed_list:\t4-\n",
            "Cpus_allowed_list:\t-4\n",
        ] {
            assert_eq!(affinity_cpu(status), None, "{status}");
        }
    }

    /// A list long enough to overflow a running total must decline rather than
    /// wrap into a plausible count; `parse_cpu_list` is the validator for
    /// `Cpus_allowed_list`, so a wrap here would silently accept junk.
    #[cfg(feature = "std")]
    #[test]
    fn cpu_list_counting_is_checked() {
        assert_eq!(parse_cpu_list("0"), Some(1));
        assert_eq!(parse_cpu_list("0,16"), Some(2));
        assert_eq!(parse_cpu_list("0-7,16-23"), Some(16));
        assert_eq!(parse_cpu_list("0-18446744073709551614,0"), None);
        assert_eq!(parse_cpu_list(""), None);
        assert_eq!(parse_cpu_list(","), None);
    }

    /// This machine's `/sys/devices/system/cpu/cpu0/cache/index*`, verbatim.
    #[cfg(feature = "std")]
    fn cascade_lake_fixture() -> Vec<[&'static str; 7]> {
        vec![
            ["1", "Data", "32K", "8", "64", "64", "0,16"],
            ["1", "Instruction", "32K", "8", "64", "64", "0,16"],
            ["2", "Unified", "1024K", "16", "64", "1024", "0,16"],
            ["3", "Unified", "25344K", "11", "64", "36864", "0-7,16-23"],
        ]
    }

    #[cfg(feature = "std")]
    fn as_indices(raw: &[[&'static str; 7]]) -> Vec<SysfsIndex<'static>> {
        raw.iter()
            .map(|r| SysfsIndex {
                level: r[0],
                kind: r[1],
                size: r[2],
                ways: r[3],
                line: r[4],
                sets: r[5],
                shared: r[6],
            })
            .collect()
    }

    #[cfg(feature = "std")]
    #[test]
    fn sysfs_fixture_parses() {
        let raw = cascade_lake_fixture();
        let h = from_sysfs(&as_indices(&raw)).expect("fixture has an L1d");
        // Including that the instruction cache at level 1 was not mistaken for
        // the data cache, and that `0,16` is two CPUs while `0-7,16-23` is 16.
        assert_eq!(h, CASCADE);
        // Every level's geometry must be self-consistent, or the model's
        // way-counting is meaningless.
        for lvl in [Some(h.l1d), h.l2, h.l3].into_iter().flatten() {
            assert_eq!(lvl.bytes_per_way() * lvl.ways, lvl.size);
        }
        // 2 logical CPUs per core, so the L2 is one core's and the L3 is eight.
        assert_eq!(h.threads_per_core(), 2);
        assert_eq!(h.cores_sharing(&h.l2.unwrap()), 1);
        assert_eq!(h.cores_sharing(&h.l3.unwrap()), 8);
    }

    /// The input the partition rule was missing (A36), on the two topologies
    /// Phase 4 measured: a socket-wide L3 spans one domain right up to the socket
    /// and a chiplet L3 starts spanning them almost immediately. That difference
    /// is the whole content of the rule, so it is pinned here rather than left to
    /// whatever machine happens to run the suite.
    #[test]
    fn l3_domains_separates_a_socket_l3_from_a_chiplet_one() {
        let socket = CASCADE; // 16 logical CPUs on the L3, SMT 2 -> 8 cores
        assert_eq!(socket.l3_domains(1), 1);
        assert_eq!(socket.l3_domains(8), 1);
        assert_eq!(socket.l3_domains(9), 2); // a second socket, or oversubscribed
        assert_eq!(socket.l3_domains(32), 4);

        // Zen2: four cores per 16 MiB L3, SMT off in the measured allocation.
        let mut chiplet = CASCADE;
        chiplet.l1d.shared_by = 1;
        chiplet.l3 = Some(CacheLevel {
            shared_by: 4,
            ..CASCADE.l3.unwrap()
        });
        assert_eq!(chiplet.l3_domains(4), 1); // the point null in all four dtypes
        assert_eq!(chiplet.l3_domains(16), 4); // 1.17-1.27x for the column axis
        assert_eq!(chiplet.l3_domains(64), 16); // up to 4.3x

        // No L3 at all: nothing is shared, so every thread is its own domain.
        let none = CacheHierarchy {
            l3: None,
            ..CASCADE
        };
        assert_eq!(none.l3_domains(8), 8);
    }

    #[cfg(feature = "std")]
    #[test]
    fn sysfs_derives_missing_set_count() {
        // Not every kernel exports `number_of_sets`.
        let raw = vec![["1", "Data", "32K", "8", "64", "", "0"]];
        let h = from_sysfs(&as_indices(&raw)).unwrap();
        assert_eq!(h.l1d.sets, 64);
        assert_eq!(h.l1d.shared_by, 1);
        assert!(h.l2.is_none() && h.l3.is_none());
    }

    #[cfg(feature = "std")]
    #[test]
    fn sysfs_rejects_junk() {
        // No data cache at all: the probe must decline rather than invent one.
        let raw = vec![["1", "Instruction", "32K", "8", "64", "64", "0"]];
        assert!(from_sysfs(&as_indices(&raw)).is_none());
        // Unparseable fields are skipped level by level, not fatal.
        let raw = vec![
            ["1", "Data", "", "8", "64", "64", "0"],
            ["2", "Unified", "1024K", "zero", "64", "1024", "0"],
        ];
        assert!(from_sysfs(&as_indices(&raw)).is_none());
        // A level out of range is ignored.
        let raw = vec![
            ["1", "Data", "32K", "8", "64", "64", "0"],
            ["4", "Unified", "128M", "16", "64", "131072", "0-63"],
        ];
        let h = from_sysfs(&as_indices(&raw)).unwrap();
        assert!(h.l2.is_none() && h.l3.is_none());
    }

    /// What this machine's `sysctl` reports, verbatim, for the *performance*
    /// cores: `hw.cachelinesize`, `hw.perflevel0.l1dcachesize`,
    /// `hw.perflevel0.l2cachesize`, `hw.perflevel0.cpusperl2` on an Apple M3 Max.
    #[cfg(all(feature = "std", target_os = "macos"))]
    fn m3_max_fixture() -> SysctlCaches {
        SysctlCaches {
            line: Some(128),
            l1d: Some(128 * 1024),
            l2: Some(16 * 1024 * 1024),
            cpus_per_l2: Some(6),
        }
    }

    #[cfg(all(feature = "std", target_os = "macos"))]
    #[test]
    fn sysctl_fixture_parses() {
        let h = from_sysctl(&m3_max_fixture()).expect("fixture has an L1d");
        assert_eq!(h.source, CacheSource::Sysctl);
        assert_eq!(h.l1d.size, 128 * 1024);
        assert_eq!(h.l1d.line, 128);
        assert_eq!(h.l1d.shared_by, 1);
        let l2 = h.l2.expect("the fixture has an L2");
        assert_eq!(l2.size, 16 * 1024 * 1024);
        assert_eq!(l2.shared_by, 6);
        // No L3, on purpose: the cluster L2 is the last level Darwin names, and
        // reporting it twice would double-count it. See `from_sysctl`.
        assert!(h.l3.is_none());
        // The same self-consistency the sysfs fixture is held to, which is what
        // makes the assumed associativity safe to feed the model.
        for lvl in [Some(h.l1d), h.l2].into_iter().flatten() {
            assert_eq!(lvl.bytes_per_way() * lvl.ways, lvl.size);
        }
        // No SMT, so one thread per core and the L2 belongs to the six cores
        // that share it rather than to twelve hyperthreads.
        assert_eq!(h.threads_per_core(), 1);
        assert_eq!(h.cores_sharing(&l2), 6);
        // And the documented over-count: twelve threads span two hardware
        // domains, this reports twelve, and only a threaded run can see it.
        assert_eq!(h.l3_domains(1), 1);
        assert_eq!(h.l3_domains(12), 12);
    }

    #[cfg(all(feature = "std", target_os = "macos"))]
    #[test]
    fn sysctl_declines_rather_than_inventing() {
        // No line size: nothing downstream can be derived, so decline.
        let raw = SysctlCaches {
            line: None,
            ..m3_max_fixture()
        };
        assert!(from_sysctl(&raw).is_none());
        // No L1d: there is no model without one, same rule as sysfs.
        let raw = SysctlCaches {
            l1d: None,
            ..m3_max_fixture()
        };
        assert!(from_sysctl(&raw).is_none());
        // A missing L2 is not fatal — the model degrades level by level.
        let raw = SysctlCaches {
            l2: None,
            ..m3_max_fixture()
        };
        let h = from_sysctl(&raw).expect("an L1d is enough");
        assert!(h.l2.is_none());
        // An L2 smaller than one way cannot be expressed and is dropped rather
        // than rounded to zero sets.
        let raw = SysctlCaches {
            l2: Some(64),
            ..m3_max_fixture()
        };
        assert!(from_sysctl(&raw).unwrap().l2.is_none());
        // Unknown sharing degrades to private, never to zero.
        let raw = SysctlCaches {
            cpus_per_l2: None,
            ..m3_max_fixture()
        };
        assert_eq!(from_sysctl(&raw).unwrap().l2.unwrap().shared_by, 1);
    }

    /// The probe must agree with the machine it is running on, which is the one
    /// thing a fixture cannot check. Asserts only what is true of every Apple
    /// part rather than of this one, so it does not become an M3-Max-only test.
    #[cfg(all(feature = "std", target_os = "macos"))]
    #[test]
    fn sysctl_probe_reads_this_machine() {
        let h = super::probe_sysctl().expect("every Darwin machine reports a line and an L1d");
        assert!(h.l1d.size >= 32 * 1024, "implausible L1d: {}", h.l1d.size);
        assert!(
            h.l1d.line == 64 || h.l1d.line == 128,
            "implausible line: {}",
            h.l1d.line
        );
        assert!(h.l3.is_none());
        // The perflevel0 preference, stated as a property rather than a number:
        // whatever the P-core L1d is, it is at least as large as the value the
        // unprefixed key reports, because that key answers for the *last*
        // perflevel — the efficiency cores on a heterogeneous part.
        if let Some(legacy) = super::sysctl_usize("hw.l1dcachesize") {
            assert!(
                h.l1d.size >= legacy,
                "probe took the efficiency cores' L1d: {} < {legacy}",
                h.l1d.size
            );
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn size_and_cpu_list_parsing() {
        assert_eq!(parse_size("32K"), Some(32 * 1024));
        assert_eq!(parse_size(" 1024K\n"), Some(1024 * 1024));
        assert_eq!(parse_size("8M"), Some(8 * 1024 * 1024));
        assert_eq!(parse_size("512"), Some(512));
        assert_eq!(parse_size(""), None);
        assert_eq!(parse_size("K"), None);
        assert_eq!(parse_cpu_list("0"), Some(1));
        assert_eq!(parse_cpu_list("0,16"), Some(2));
        assert_eq!(parse_cpu_list("0-7,16-23\n"), Some(16));
        assert_eq!(parse_cpu_list("0-3"), Some(4));
        assert_eq!(parse_cpu_list(""), None);
        assert_eq!(parse_cpu_list("7-0"), None, "a reversed range is junk");
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn cpuid_leaf_decoding() {
        // Hand-encoded 1 MiB 16-way unified L2 with 64-byte lines and 1024
        // sets, shared by 2 logical CPUs — this machine's, as CPUID.4 would
        // report it.
        let (kind_in, level_in, shared_in) = (3u32, 2u32, 2u32);
        let (line_in, partitions_in, ways_in, sets_in) = (64u32, 1u32, 16u32, 1024u32);
        let eax = kind_in | (level_in << 5) | ((shared_in - 1) << 14);
        let ebx = (line_in - 1) | ((partitions_in - 1) << 12) | ((ways_in - 1) << 22);
        let ecx = sets_in - 1;
        let (kind, lvl) = decode_cache_leaf(eax, ebx, ecx).unwrap();
        assert_eq!(kind, 3);
        assert_eq!(lvl.level, 2);
        assert_eq!(lvl.size, 1 << 20);
        assert_eq!(lvl.ways, 16);
        assert_eq!(lvl.sets, 1024);
        assert_eq!(lvl.line, 64);
        assert_eq!(lvl.shared_by, 2);
        assert_eq!(lvl.bytes_per_way() * lvl.ways, lvl.size);
        // The null subleaf terminates enumeration.
        assert!(decode_cache_leaf(0, 0, 0).is_none());
    }

    /// The source label a report prints must be the one `name()` reports,
    /// whichever of the two a caller reaches for.
    #[test]
    fn cache_source_displays_its_name() {
        for s in [
            CacheSource::Sysfs,
            CacheSource::Cpuid,
            CacheSource::Sysctl,
            CacheSource::Builtin,
        ] {
            assert_eq!(s.to_string(), s.name());
        }
    }

    /// The probe on whatever machine the tests run on must produce something
    /// self-consistent, whichever source answers.
    #[test]
    fn probe_is_self_consistent() {
        let h = hierarchy();
        for lvl in [Some(h.l1d), h.l2, h.l3].into_iter().flatten() {
            assert!(lvl.size > 0 && lvl.line > 0 && lvl.ways > 0 && lvl.sets > 0);
            assert!(lvl.shared_by >= 1);
        }
        assert!(h.l1d.level == 1);
    }
}
