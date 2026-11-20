use super::*;

const ROW: ItemFit = ItemFit {
    fixed_ns: 100.0,
    muladds_per_ns: 50.0,
    parallel_ns: 1_000.0,
    per_width_ns: 10.0,
    slow: 0.5,
    slow_per_width_ns: 2_000.0,
    gain_margin: 0.25,
};

/// The row prices the fit's model: the work at the fitted rate in rounds of
/// its largest task, a cost per round, the slow band's fixed part a share of
/// the work and its per-participant part in nanoseconds.
#[test]
fn a_row_prices_the_fit_model() {
    // 4e5 muladds at 50 per nanosecond is 8 µs of work, as 4 tasks, the
    // largest 0.4 of it, 300 ns per round.
    let t = ROW.item_time(4e5, 4, 0.4, 300.0, 6);
    assert_eq!(t.work_ns, 8_000.0);
    assert_eq!(t.gain_margin, 0.25);
    assert_eq!(t.at(1), 8_100.0);
    // Width 3: two rounds, capped at the whole work; in the slow band.
    assert_eq!(
        t.at(3),
        100.0 + 1_000.0 + 8_000.0 * 0.8 + 2.0 * 300.0 + 10.0 * 2.0 + 0.5 * 8_000.0 + 2_000.0 * 2.0
    );
    // Width 4: one round of the largest task, still in the band.
    assert_eq!(
        t.at(4),
        100.0 + 1_000.0 + 8_000.0 * 0.4 + 300.0 + 10.0 * 3.0 + 0.5 * 8_000.0 + 2_000.0 * 3.0
    );
    // Width 6 leaves the band; width 8 only adds participants.
    assert_eq!(
        t.at(6),
        100.0 + 1_000.0 + 8_000.0 * 0.4 + 300.0 + 10.0 * 5.0
    );
    assert_eq!(
        t.at(8),
        100.0 + 1_000.0 + 8_000.0 * 0.4 + 300.0 + 10.0 * 7.0
    );
}

/// A zero slow band never adds its costs, whatever the width.
#[test]
fn a_row_without_a_slow_band_stays_off_it() {
    let row = ItemFit {
        slow: 2.0,
        slow_per_width_ns: 1e6,
        ..ROW
    };
    let t = row.item_time(4e5, 4, 0.25, 0.0, 0);
    assert_eq!(t.slow_below, 0);
    assert_eq!(t.at(4), 100.0 + 1_000.0 + 8_000.0 / 4.0 + 10.0 * 3.0);
}
