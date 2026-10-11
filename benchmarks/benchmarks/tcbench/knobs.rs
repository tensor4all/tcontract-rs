//! The measurement knobs that used to be read from the environment inside the
//! library, parsed once here into explicit plan configuration (tcontract addition).
//!
//! `TCBENCH_{KERNEL,COMPLEX,MC,KC,NC,MC_PCT,NC_PCT,KC_COUPLE,BLOCKMODEL,
//! L3_DOMAINS,WRITEBACK,ORIENT,ROWBLOCK,PARTITION}` are read by this harness
//! only; the libraries read no environment. An unset variable is the library
//! baseline; an unparseable value is an error rather than a silent default.
//! Variables of the removed library knobs are rejected by prefix in
//! [`init_from_env`] (see `tcontract_bench::threads::forbidden_env`): threads come
//! from a `tcontract_exec::Exec` only.

use std::sync::OnceLock;

use tcontract::{CacheModel, Orient, Partition, PlanConfig, RowBlock, Writeback};
use tcontract_kernel::{
    blocking::BlockModel, BlockingOverride, ComplexMethod, KernelForce, Method,
};

/// The parsed knobs.
#[derive(Clone, Debug, Default)]
pub struct Knobs {
    isa: KernelForce,
    method: Option<ComplexMethod>,
    block_model: BlockModel,
    blocking: BlockingOverride,
    kc_couple: Option<usize>,
    gather: bool,
    orient: Orient,
    row_block: RowBlock,
    partition: Option<Partition>,
    l3_domains: Option<usize>,
}

fn parse_with<T>(
    get: &impl Fn(&str) -> Option<String>,
    name: &str,
    parse: impl Fn(&str) -> Option<T>,
) -> Result<Option<T>, String> {
    match get(name) {
        None => Ok(None),
        Some(v) => parse(&v)
            .map(Some)
            .ok_or_else(|| format!("{name}={v}: unrecognised value")),
    }
}

fn parse_usize(v: &str) -> Option<usize> {
    v.trim().parse().ok().filter(|&n| n > 0)
}

/// The partition spellings that survive: the default rule, or a pinned grid.
fn parse_partition(v: &str) -> Option<Option<Partition>> {
    let v = v.trim().to_ascii_lowercase();
    if v == "domain" || v == "domains" {
        return Some(None);
    }
    let (pm, pn) = v.split_once('x')?;
    Some(Some(Partition::StaticGrid {
        pin: Some((
            pm.parse::<usize>().ok()?.max(1),
            pn.parse::<usize>().ok()?.max(1),
        )),
        align_c_lines: false,
    }))
}

impl Knobs {
    /// Parse from a variable lookup (the process environment in `main`).
    pub fn parse(get: impl Fn(&str) -> Option<String>) -> Result<Knobs, String> {
        let mut k = Knobs::default();
        if let Some(f) = parse_with(&get, "TCBENCH_KERNEL", KernelForce::parse)? {
            k.isa = f;
        }
        k.method = parse_with(&get, "TCBENCH_COMPLEX", ComplexMethod::parse)?;
        if let Some(m) = parse_with(&get, "TCBENCH_BLOCKMODEL", BlockModel::parse)? {
            k.block_model = m;
        }
        let b = &mut k.blocking;
        b.mc = parse_with(&get, "TCBENCH_MC", parse_usize)?;
        b.kc = parse_with(&get, "TCBENCH_KC", parse_usize)?;
        b.nc = parse_with(&get, "TCBENCH_NC", parse_usize)?;
        b.mc_pct = parse_with(&get, "TCBENCH_MC_PCT", parse_usize)?;
        b.nc_pct = parse_with(&get, "TCBENCH_NC_PCT", parse_usize)?;
        k.kc_couple = parse_with(&get, "TCBENCH_KC_COUPLE", parse_usize)?;
        k.gather = parse_with(&get, "TCBENCH_WRITEBACK", |v: &str| {
            match v.trim().to_ascii_lowercase().as_str() {
                "gather" | "scatter" => Some(true),
                "fast" | "default" => Some(false),
                _ => None,
            }
        })?
        .unwrap_or(false);
        k.orient = parse_with(&get, "TCBENCH_ORIENT", Orient::parse)?.unwrap_or_default();
        k.row_block = parse_with(&get, "TCBENCH_ROWBLOCK", RowBlock::parse)?.unwrap_or_default();
        k.partition = parse_with(&get, "TCBENCH_PARTITION", parse_partition)?.flatten();
        k.l3_domains = parse_with(&get, "TCBENCH_L3_DOMAINS", parse_usize)?;
        Ok(k)
    }

    /// The plan configuration every knob describes. An unset knob is the library
    /// baseline; a set tuning knob forces the packed driver, as the library
    /// documents.
    pub fn config(&self) -> PlanConfig {
        PlanConfig {
            isa: self.isa,
            method: self.method.map(|m| match m {
                ComplexMethod::Planar => Method::Native,
                ComplexMethod::OneM => Method::OneM,
                ComplexMethod::ThreeM => Method::ThreeM,
            }),
            blocking: self.blocking,
            orientation: self.orient,
            row_block: self.row_block,
            partition: self.partition,
            cache_model: CacheModel {
                block_model: self.block_model,
                l3_domains: self.l3_domains,
                kc_couple: self.kc_couple,
            },
            writeback: if self.gather {
                Writeback::Gather
            } else {
                Writeback::Auto
            },
            ..PlanConfig::default()
        }
    }

    /// The same configuration with the packed driver forced, so the packed
    /// engine is measured even where the planner would pick another strategy.
    pub fn packed_config(&self) -> PlanConfig {
        let mut cfg = self.config();
        cfg.partition.get_or_insert(Partition::StaticGrid {
            pin: None,
            align_c_lines: false,
        });
        cfg
    }

    /// The block model in force.
    pub fn block_model(&self) -> BlockModel {
        self.block_model
    }

    /// The forced L3 domain count, if any.
    pub fn l3_domains(&self) -> Option<usize> {
        self.l3_domains
    }
}

static KNOBS: OnceLock<Knobs> = OnceLock::new();

/// Read the process environment once, at startup. An error is printed by the
/// caller; later [`get`] calls then see the baseline.
pub fn init_from_env() -> Result<(), String> {
    if let Some((name, _)) = tcontract_bench::threads::forbidden_env(std::env::vars()) {
        return Err(format!(
            "{name}: the old library environment knobs were removed in tcontract-rs#37; \
             the harness knobs are TCBENCH_* (see knobs.rs)"
        ));
    }
    let k = Knobs::parse(|n| std::env::var(n).ok())?;
    let _ = KNOBS.set(k);
    Ok(())
}

/// The parsed knobs (the baseline before [`init_from_env`]).
pub fn get() -> &'static Knobs {
    KNOBS.get_or_init(Knobs::default)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |n| {
            pairs
                .iter()
                .find(|(k, _)| *k == n)
                .map(|(_, v)| v.to_string())
        }
    }

    #[test]
    fn unset_is_the_baseline_and_knobs_parse() {
        let base = Knobs::parse(env(&[])).unwrap();
        assert_eq!(base.config(), PlanConfig::default());
        assert!(!base.config().requires_packed());
        let k = Knobs::parse(env(&[
            ("TCBENCH_KERNEL", "avx2"),
            ("TCBENCH_PARTITION", "4x2"),
            ("TCBENCH_MC", "64"),
            ("TCBENCH_WRITEBACK", "gather"),
            ("TCBENCH_L3_DOMAINS", "3"),
        ]))
        .unwrap();
        let cfg = k.config();
        assert_eq!(cfg.isa, KernelForce::Avx2);
        assert_eq!(
            cfg.partition,
            Some(Partition::StaticGrid {
                pin: Some((4, 2)),
                align_c_lines: false
            })
        );
        assert_eq!(cfg.blocking.mc, Some(64));
        assert_eq!(cfg.writeback, Writeback::Gather);
        assert_eq!(k.l3_domains(), Some(3));
        assert!(cfg.requires_packed());
        // The packed engine is the default configuration plus a forced grid.
        assert!(base.packed_config().requires_packed());
        assert_eq!(base.packed_config().kernel, base.config().kernel);
    }

    #[test]
    fn malformed_knobs_are_rejected() {
        assert!(Knobs::parse(env(&[("TCBENCH_KERNEL", "avx3")])).is_err());
        assert!(Knobs::parse(env(&[("TCBENCH_MC", "0")])).is_err());
        // The row/column partition modes no longer exist.
        for v in ["m", "n", "rows", "cols", "legacy"] {
            assert!(
                Knobs::parse(env(&[("TCBENCH_PARTITION", v)])).is_err(),
                "{v}"
            );
        }
    }
}
