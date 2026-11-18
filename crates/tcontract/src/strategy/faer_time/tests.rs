use super::*;

const XEON: &str = "processor\t: 0\nvendor_id\t: GenuineIntel\ncpu family\t: 6\nmodel\t\t: 85\n\
model name\t: Intel(R) Xeon(R) Platinum 9242 CPU @ 2.30GHz\n\nprocessor\t: 1\n\
vendor_id\t: GenuineIntel\ncpu family\t: 6\nmodel\t\t: 85\n";

#[test]
fn cpuinfo_gives_vendor_family_model_and_cpu_count() {
    let key = parse_cpuinfo(XEON).unwrap();
    assert_eq!(
        key,
        HostKey {
            vendor: "GenuineIntel".into(),
            family: 6,
            model: 85,
            cpus: 2
        }
    );
    assert_eq!(parse_cpuinfo("processor : 0\n"), None);
}

/// Only the fitted class has rows: another model or CPU count has none, so its
/// faer plans supply no item time and keep the ideal-division rule.
#[test]
fn only_the_fitted_class_has_rows() {
    let mut key = parse_cpuinfo(XEON).unwrap();
    assert_eq!(row_of(&key, DType::F64), None, "96 CPUs, not 2");
    key.cpus = 96;
    assert!(row_of(&key, DType::F64).is_some());
    key.model = 106;
    assert_eq!(row_of(&key, DType::F64), None);
}

/// The width `Plan::execution_width` reports for a Faer plan on a team is the
/// argmin of this host's faer time plus the team's dispatch cost over the whole
/// lane on a fitted class, and the ideal-division width the token's rule gives
/// on any other host.
#[test]
fn execution_width_is_the_argmin_of_the_host_row() {
    use crate::api::{CSpec, Labels, LayoutSpec, OperandSpec, Problem};
    use crate::{OutputContract, Plan, PlanConfig};
    use tcontract_exec::{BatchPolicy, Exec};

    let pool = threadteam::Pool::builder()
        .threads(4)
        .acquire_timeout(std::time::Duration::from_secs(10))
        .build()
        .expect("pool");
    let Ok(team) = pool.into_team(4) else {
        // The host could not reserve the members.
        return;
    };
    let exec = Exec::from_token(team.exec(4), None, BatchPolicy::default());
    for (m, n, k) in [(16, 16, 16), (64, 256, 32), (384, 384, 384)] {
        let l = |d: &[usize]| OperandSpec::new(LayoutSpec::new(d, &[1, d[0] as isize], 0).unwrap());
        let problem = Problem::from_labels(
            DType::F64,
            l(&[m, k]),
            l(&[k, n]),
            CSpec::Absent,
            l(&[m, n]),
            &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
        )
        .unwrap();
        let plan = Plan::<f64>::new(&problem, &PlanConfig::default()).unwrap();
        let muladds = (m * n * k) as f64;
        let want = match host_row(DType::F64) {
            Some(row) => {
                let time = item_time(row, m, n, muladds);
                let objective = |w: usize| {
                    let dispatch = std::num::NonZeroUsize::new(w)
                        .filter(|_| w > 1)
                        .map_or(0.0, |w| {
                            exec.token_ref().dispatch_cost(w).as_secs_f64() * 1e9
                        });
                    time.at(w) + dispatch
                };
                (1..=4)
                    .min_by(|&a, &b| objective(a).total_cmp(&objective(b)))
                    .unwrap()
            }
            None => exec.width_for_within(4, muladds / tcontract_exec::host::muladds_per_ns(), 1),
        };
        let got = plan
            .execution_width(&exec, 4, 1.0, OutputContract::Initialized)
            .unwrap()
            .map(|w| w.get());
        assert_eq!(got, Some(want), "{m}x{n}x{k}");
    }
}

/// The fitted class has a row for every storage type, and on it a large
/// product is wider than a small one, which stays at width one.
#[test]
fn the_fitted_class_has_a_row_per_dtype() {
    let mut key = parse_cpuinfo(XEON).unwrap();
    key.cpus = 96;
    for dtype in [DType::F32, DType::F64, DType::C32, DType::C64] {
        assert!(row_of(&key, dtype).is_some(), "{dtype:?}");
    }
    let row = row_of(&key, DType::F64).unwrap();
    let argmin = |t: ItemTime| {
        (1..=96)
            .min_by(|&a, &b| t.at(a).total_cmp(&t.at(b)))
            .unwrap()
    };
    assert_eq!(argmin(item_time(row, 32, 32, 32.0 * 32.0 * 32.0)), 1);
    assert!(argmin(item_time(row, 512, 512, 512.0 * 512.0 * 512.0)) >= 24);
}
