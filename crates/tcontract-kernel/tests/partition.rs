//! The strip rule: strips tile the rows, land on `MR` (or alignment)
//! boundaries, and always claim the tail.
use tcontract_kernel::partition::*;
use tcontract_kernel::*;

#[test]
fn strips_tile_the_rows_on_mr_multiples() {
    for &(m, mr, pm) in &[
        (1usize, 8usize, 1usize),
        (37, 8, 3),
        (1000, 6, 7),
        (8, 8, 8),
    ] {
        let mut next = 0;
        for r in 0..pm {
            let (lo, hi) = strip(r, pm, m, mr, 0);
            assert_eq!(lo, next, "strip {r} of {pm} over {m} rows");
            assert!(lo % mr == 0 || lo == m);
            assert!(hi >= lo && hi <= m);
            next = hi;
        }
        assert_eq!(next, m);
    }
}

#[test]
fn aligned_strips_round_to_the_alignment() {
    for r in 1..4 {
        let (lo, hi) = strip(r, 4, 1000, 6, 24);
        assert_eq!(lo % 24, 0, "strip {r} starts at {lo}");
        assert!(hi >= lo);
    }
    // The tail is still claimed, whatever rounding does to the last boundary.
    assert_eq!(strip(3, 4, 1000, 6, 24).1, 1000);
    assert_eq!(strip(9, 10, 101, 8, 64).1, 101);
}

fn dynamic(
    id: &str,
    job_m: usize,
    job_n: usize,
    opts: PartitionOpts,
) -> Result<ResolvedGemm<f64>, SelectError> {
    ResolvedGemm::<f64>::resolve_with::<f64>(
        &KernelChoice::Id(id.into()),
        4,
        PartitionPolicy::DynamicTiles { job_m, job_n },
        opts,
    )
}

#[test]
fn dynamic_tiles_resolves_for_whole_register_blocks_and_survives_retargeting() {
    // ref.f64.real.4x4: logical MR = NR = 4.
    let rg = dynamic("ref.f64.real.4x4", 16, 32, PartitionOpts::default()).unwrap();
    assert_eq!(
        rg.partition,
        PartitionPolicy::DynamicTiles {
            job_m: 16,
            job_n: 32
        }
    );
    assert_eq!(rg.with_threads(2).unwrap().partition, rg.partition);
    // Complex families validate against their logical (not packed) MR/NR.
    let c = ResolvedGemm::<f64>::resolve_with::<tcontract_kernel::C64>(
        &KernelChoice::Id("ref.c64.native.4x4".into()),
        2,
        PartitionPolicy::DynamicTiles { job_m: 4, job_n: 8 },
        PartitionOpts::default(),
    );
    assert!(c.is_ok());
}

#[test]
fn dynamic_tiles_rejects_invalid_geometry_and_line_alignment() {
    let id = "ref.f64.real.4x4";
    for (jm, jn) in [(0, 8), (8, 0), (6, 8), (8, 6), (3, 3)] {
        assert!(
            matches!(
                dynamic(id, jm, jn, PartitionOpts::default()),
                Err(SelectError::Incompatible { .. })
            ),
            "{jm}x{jn}"
        );
    }
    let aligned = PartitionOpts {
        align_c_lines: true,
    };
    assert!(matches!(
        dynamic(id, 8, 8, aligned),
        Err(SelectError::Incompatible { .. })
    ));
}

#[test]
fn half_specified_static_grid_is_rejected() {
    let err = ResolvedGemm::<f64>::resolve_with::<f64>(
        &KernelChoice::Auto,
        4,
        PartitionPolicy::StaticGrid { pm: 2, pn: 0 },
        PartitionOpts::default(),
    );
    assert!(matches!(err, Err(SelectError::Incompatible { .. })));
}

/// The driver derives `pm * pn` unchecked, so a grid whose product overflows
/// must never be accepted.
#[test]
fn a_static_grid_whose_product_overflows_is_rejected() {
    let err = ResolvedGemm::<f64>::resolve_with::<f64>(
        &KernelChoice::Auto,
        4,
        PartitionPolicy::StaticGrid {
            pm: usize::MAX / 2 + 1,
            pn: 2,
        },
        PartitionOpts::default(),
    );
    assert!(matches!(err, Err(SelectError::Incompatible { .. })));
}

#[test]
fn an_explicit_grid_survives_retargeting() {
    let grid = PartitionPolicy::StaticGrid { pm: 3, pn: 2 };
    let opts = PartitionOpts {
        align_c_lines: true,
    };
    let rg = ResolvedGemm::<f64>::resolve_with::<f64>(&KernelChoice::Auto, 6, grid, opts).unwrap();
    assert_eq!(rg.partition, grid);
    assert!(rg.opts.align_c_lines);
    // Re-deriving the blocking for another width must not drop the policy.
    let serial = rg.with_threads(1).unwrap();
    assert_eq!(serial.partition, grid);
    assert_eq!(serial.opts, opts);
}
