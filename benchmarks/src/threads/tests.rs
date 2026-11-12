use super::{conflicting_env, parse_threads};

fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
    move |n| {
        pairs
            .iter()
            .find(|(k, _)| *k == n)
            .map(|(_, v)| v.to_string())
    }
}

#[test]
fn conflicting_thread_variable_is_reported() {
    assert_eq!(
        conflicting_env(4, env(&[("RAYON_NUM_THREADS", "2")])),
        Some(("RAYON_NUM_THREADS".into(), "2".into()))
    );
    assert_eq!(
        conflicting_env(1, env(&[("OMP_NUM_THREADS", "8")])),
        Some(("OMP_NUM_THREADS".into(), "8".into()))
    );
}

#[test]
fn equal_or_unset_variables_are_fine() {
    assert_eq!(conflicting_env(4, env(&[("RAYON_NUM_THREADS", "4")])), None);
    assert_eq!(conflicting_env(4, env(&[])), None);
}

#[test]
fn threads_flag_parses_and_rejects_zero() {
    let a = |s: &str| s.split(' ').map(String::from).collect::<Vec<_>>();
    assert_eq!(parse_threads(&a("bin")), Ok(1));
    assert_eq!(parse_threads(&a("bin --threads 4")), Ok(4));
    assert!(parse_threads(&a("bin --threads 0")).is_err());
    assert!(parse_threads(&a("bin --threads")).is_err());
}

#[test]
fn removed_variable_prefixes_are_rejected() {
    use super::{forbidden_env, REMOVED_ENV_PREFIXES};
    let vars = |pairs: &[(&str, &str)]| {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect::<Vec<_>>()
    };
    for prefix in REMOVED_ENV_PREFIXES {
        let name = format!("{prefix}ANYTHING");
        assert_eq!(
            forbidden_env(vars(&[("PATH", "/bin"), (&name, "x")])),
            Some((name.clone(), "x".into())),
            "{name}"
        );
    }
    assert_eq!(forbidden_env(vars(&[("TCBENCH_MC", "64")])), None);
    assert_eq!(forbidden_env(vars(&[])), None);
}

/// The CPUs the calling thread may run on, from `/proc/thread-self/status`.
#[cfg(target_os = "linux")]
fn allowed_cpus() -> usize {
    let status = std::fs::read_to_string("/proc/thread-self/status").expect("thread status");
    let list = status
        .lines()
        .find_map(|l| l.strip_prefix("Cpus_allowed_list:"))
        .expect("Cpus_allowed_list")
        .trim();
    list.split(',')
        .map(|part| match part.split_once('-') {
            Some((a, b)) => b.parse::<usize>().unwrap() - a.parse::<usize>().unwrap() + 1,
            None => 1,
        })
        .sum()
}

/// A vendor library creates its thread pool from the calling thread, so the
/// harness must configure it before the benchmark team pins its leader. This
/// test fails if the hook runs after the pin (the leader would see one CPU), or
/// if the pin stops happening (the leader would keep the whole set).
#[cfg(target_os = "linux")]
#[test]
fn vendor_configuration_runs_before_the_team_pins_the_leader() {
    let available = allowed_cpus();
    if available < 4 {
        return;
    }
    let seen = std::cell::Cell::new(0usize);
    let threads = super::BenchThreads::new(4, |_| seen.set(allowed_cpus()));
    assert_eq!(threads.requested, 4);
    assert_eq!(
        seen.get(),
        available,
        "the vendor hook sees the process CPU set"
    );
    assert_eq!(
        allowed_cpus(),
        1,
        "the placed team pins its leader to one CPU"
    );
    drop(threads);
}
