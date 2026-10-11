//! `--partition static|dynamic:JM,JN` for the packed-driver benchmarks.
//!
//! The comparison of the default static grid with the opt-in
//! `Partition::DynamicTiles` is run later, with the `tcontract-benchmark`
//! protocol; this flag only makes the two separately selectable and their rows
//! separately identifiable. Absent, every benchmark behaves exactly as before.
use tcontract::{Partition, PlanConfig};

/// The requested policy, or `None` for the default (static) behaviour.
///
/// # Panics
/// On a malformed value: a benchmark run must not silently fall back.
pub fn from_args() -> Option<Partition> {
    let args: Vec<String> = std::env::args().collect();
    let i = args.iter().position(|a| a == "--partition")?;
    let value = args
        .get(i + 1)
        .unwrap_or_else(|| panic!("--partition needs a value: static or dynamic:JM,JN"));
    parse(value).unwrap_or_else(|| panic!("bad --partition {value}; use static or dynamic:JM,JN"))
}

/// Parse `static` (the default, reported as no request) or `dynamic:JM,JN`.
pub fn parse(value: &str) -> Option<Option<Partition>> {
    if value == "static" {
        return Some(None);
    }
    let (job_m, job_n) = value.strip_prefix("dynamic:")?.split_once(',')?;
    Some(Some(Partition::DynamicTiles {
        job_m: job_m.parse().ok()?,
        job_n: job_n.parse().ok()?,
    }))
}

/// The partition policy a packed plan froze, for a row's notes.
///
/// This is the *policy*, not the geometry a call ends up using. A plan is built
/// before an executor is chosen, and `driver::route` resolves the policy against
/// the problem and the executor's budget at call time: `auto` becomes a concrete
/// grid there, a pinned grid can be clamped down to the budget, a grid can be
/// reshaped for direct-B execution, and a batch axis can take the parallelism
/// instead. Two rows that carry the same policy, family and blocking therefore
/// execute the same geometry for the same problem at the same width; two rows
/// that differ here need not, and a row that says `auto` still says which grid
/// that was only together with the width it ran at.
pub fn describe(report: &tcontract::PackedReport) -> String {
    use tcontract_kernel::PartitionPolicy;
    let g = match report.partition {
        // `pm == 0` is the automatic policy's sentinel, not a 0x0 grid.
        PartitionPolicy::StaticGrid { pm: 0, pn: 0 } => "auto".into(),
        PartitionPolicy::StaticGrid { pm, pn } => format!("pin:{pm}x{pn}"),
        PartitionPolicy::DynamicTiles { job_m, job_n } => format!("dynamic:{job_m}x{job_n}"),
        _ => "other".into(),
    };
    format!("policy={g} align={}", report.align_c_lines)
}

/// Row-label suffix identifying the policy: empty for static.
pub fn suffix(policy: Option<Partition>) -> String {
    match policy {
        Some(Partition::DynamicTiles { job_m, job_n }) => format!("_dyn{job_m}x{job_n}"),
        _ => String::new(),
    }
}

/// `cfg` with the policy applied.
pub fn apply(mut cfg: PlanConfig, policy: Option<Partition>) -> PlanConfig {
    if let Some(p) = policy {
        cfg.partition = Some(p);
    }
    cfg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_static_and_dynamic() {
        assert_eq!(parse("static"), Some(None));
        assert_eq!(
            parse("dynamic:16,32"),
            Some(Some(Partition::DynamicTiles {
                job_m: 16,
                job_n: 32
            }))
        );
        assert_eq!(parse("dynamic:16"), None);
        assert_eq!(parse("dynamic:a,b"), None);
        assert_eq!(parse("grid"), None);
        assert_eq!(
            suffix(parse("dynamic:8,24").unwrap()),
            "_dyn8x24".to_string()
        );
    }
}
