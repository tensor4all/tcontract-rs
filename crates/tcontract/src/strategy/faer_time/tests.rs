use super::*;

/// The fitted class with `cpus` logical CPUs.
fn xeon(cpus: usize) -> CpuClass {
    CpuClass {
        vendor: "GenuineIntel".into(),
        family: 6,
        model: 85,
        cpus,
    }
}

/// Only the fitted class has rows: another model or CPU count has none, so its
/// faer plans supply no item time and keep the ideal-division rule.
#[test]
fn only_the_fitted_class_has_rows() {
    let mut key = xeon(2);
    assert_eq!(row_of(&key, DType::C64), None, "96 CPUs, not 2");
    key.cpus = 96;
    assert!(row_of(&key, DType::C64).is_some());
    key.model = 106;
    assert_eq!(row_of(&key, DType::C64), None);
}

/// The width `Plan::execution_width` reports for a Faer plan on a team is the
/// argmin of this host's faer time plus the team's dispatch cost over the whole
/// lane where the host has a row for the dtype, and the ideal-division width the
/// token's rule gives otherwise (any unfitted host).
#[test]
fn execution_width_is_the_argmin_of_the_host_row() {
    let pool = threadteam::Pool::builder()
        .threads(4)
        .acquire_timeout(std::time::Duration::from_secs(10))
        .build()
        .expect("pool");
    let Ok(team) = pool.into_team(4) else {
        // The host could not reserve the members.
        return;
    };
    let exec = tcontract_exec::Exec::from_token(
        team.exec(4),
        None,
        tcontract_exec::BatchPolicy::default(),
    );
    for (m, n, k) in [(16, 16, 16), (64, 256, 32), (384, 384, 384)] {
        check_execution_width::<f64>(&exec, m, n, k);
        check_execution_width::<num_complex::Complex<f64>>(&exec, m, n, k);
    }
}

fn check_execution_width<T: crate::api::Scalar>(
    exec: &tcontract_exec::Exec<'_>,
    m: usize,
    n: usize,
    k: usize,
) {
    use crate::api::{CSpec, Labels, LayoutSpec, OperandSpec, Problem};
    use crate::{FaerLimit, OutputContract, Plan, PlanConfig};

    let l = |d: &[usize]| OperandSpec::new(LayoutSpec::new(d, &[1, d[0] as isize], 0).unwrap());
    let problem = Problem::from_labels(
        T::STORAGE,
        l(&[m, k]),
        l(&[k, n]),
        CSpec::Absent,
        l(&[m, n]),
        &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
    )
    .unwrap();
    // Every shape a Faer plan, also above the default c64 volume bound.
    let config = PlanConfig {
        faer_limit: FaerLimit::NONE,
        ..PlanConfig::default()
    };
    let plan = Plan::<T>::new(&problem, &config).unwrap();
    let muladds = plan.muladds();
    let want = match host_row(T::STORAGE) {
        Some(row) => {
            let time = item_time(row, T::STORAGE, (m, n, k), muladds);
            let objective = |w: usize| {
                let dispatch = std::num::NonZeroUsize::new(w)
                    .filter(|_| w > 1)
                    .map_or(0.0, |w| {
                        exec.token_ref().dispatch_cost(w).as_secs_f64() * 1e9
                    });
                time.at(w) + dispatch
            };
            let w = (1..=4)
                .min_by(|&a, &b| objective(a).total_cmp(&objective(b)))
                .unwrap();
            if objective(w) * (1.0 + time.gain_margin) < objective(1) {
                w
            } else {
                1
            }
        }
        None => exec.width_for_within(4, muladds / tcontract_exec::host::muladds_per_ns(), 1),
    };
    let got = plan
        .execution_width(
            exec,
            4,
            <T as num_traits::One>::one(),
            OutputContract::Initialized,
        )
        .unwrap()
        .map(|w| w.get());
    assert_eq!(got, Some(want), "{:?} {m}x{n}x{k}", T::STORAGE);
}

/// The fitted class ships a row for every dtype. On a row a large
/// product is wider than a small one, which stays at width one, and a product
/// whose largest task is most of the work stays at width one however large (f64
/// `80 x 40 x 512`: two tasks of 32 and 8 columns), while a tall one of the same
/// `n` does not (`512 x 32 x 512`: six tasks).
#[test]
fn the_fitted_class_ships_rows_for_every_dtype() {
    let key = xeon(96);
    for dtype in [DType::F32, DType::F64, DType::C32, DType::C64] {
        assert!(row_of(&key, dtype).is_some(), "{dtype:?}");
    }
    let argmin = |t: ItemTime| {
        (1..=96)
            .min_by(|&a, &b| t.at(a).total_cmp(&t.at(b)))
            .unwrap()
    };
    let fitted = |dtype| XEON_9242_2S.iter().find(|(d, _)| *d == dtype).unwrap().1;
    let at = |dtype, m: usize, n: usize, k: usize| {
        let c = if dtype == DType::C64 { 4.0 } else { 1.0 };
        argmin(item_time(
            fitted(dtype),
            dtype,
            (m, n, k),
            c * (m * n * k) as f64,
        ))
    };
    assert_eq!(at(DType::C64, 32, 32, 32), 1);
    assert!(at(DType::C64, 512, 512, 512) >= 24);
    assert_eq!(at(DType::F64, 80, 40, 512), 1);
    assert!(at(DType::F64, 512, 32, 512) > 1);
    let row = fitted(DType::C64);
    let time = item_time(row, DType::C64, (16, 16, 16), 4.0 * 4096.0);
    assert_eq!(time.gain_margin, row.fit.gain_margin);
}
