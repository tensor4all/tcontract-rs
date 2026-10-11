use super::{columns_beat_rows, local_columns};

#[test]
fn local_columns_require_saturation_balance_and_long_contiguous_run() {
    assert!(local_columns(3456, 3456, 24, 8, (288, 1), Some(8)));
    assert!(!local_columns(3456, 3456, 24, 4, (288, 1), Some(8)));
    assert!(!local_columns(3456, 3456, 24, 8, (288, 1), Some(16)));
    assert!(!local_columns(3456, 3456, 24, 8, (288, 1), None));
    assert!(!local_columns(3456, 3456, 24, 8, (192, 1), Some(8)));
    assert!(!local_columns(3456, 3456, 24, 8, (288, 2), Some(8)));
    assert!(!local_columns(3456, 3456, 24, 8, (24, 1), Some(8)));
    assert!(!local_columns(92160, 48, 16, 8, (288, 1), Some(8)));
    assert!(!local_columns(3456, 3456, 24, 1, (288, 1), Some(1)));
    // A serial run is never a column swap, however it otherwise qualifies.
    assert!(!local_columns(3456, 3456, 24, 1, (9600, 1), Some(1)));
}

/// The two thresholds are strict, and the boundary is where they flip: the run
/// must span `more than p` panels, and the column axis must be at least the row
/// axis. Both are the difference between the six recovered cases and a
/// regression on the narrow ones, so pin the exact crossing.
#[test]
fn local_columns_thresholds_are_strict_at_the_boundary() {
    // 3456 / 24 = 144 panels; p = 8 means `> 8` panels per worker.
    assert!(local_columns(3456, 3456, 24, 8, (24 * 9, 1), Some(8)));
    assert!(!local_columns(3456, 3456, 24, 8, (24 * 8, 1), Some(8)));
    // A tie on the axes is allowed (`cols >= rows`); a narrower column is not.
    assert!(local_columns(100, 100, 24, 8, (240, 1), Some(8)));
    assert!(!local_columns(101, 100, 24, 8, (240, 1), Some(8)));
    // `mr` is clamped, so a zero panel width cannot divide by zero.
    assert!(local_columns(100, 100, 0, 8, (240, 1), Some(8)));
    // Only `Some(p)` counts as saturation: more cores than threads is not it.
    assert!(!local_columns(100, 100, 24, 8, (240, 1), Some(9)));
}

#[test]
fn local_exception_is_confined_to_the_depth_guard() {
    assert!(columns_beat_rows(432, 64, 8, 1, true));
    assert!(!columns_beat_rows(432, 65, 8, 1, true));
    // The multi-domain arm is unchanged by the local exception.
    assert!(columns_beat_rows(432, 24, 8, 2, false));
    assert!(!columns_beat_rows(432, 24, 8, 1, false));
}

#[test]
fn local_exception_keeps_parallelism_and_depth_guards() {
    assert!(columns_beat_rows(432, 24, 8, 1, true));
    assert!(!columns_beat_rows(7, 24, 8, 1, true));
    assert!(!columns_beat_rows(432, 65, 8, 1, true));
    assert!(!columns_beat_rows(432, 24, 8, 1, false));
}
