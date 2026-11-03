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
