//! The row-block menu split: Auto reads the tuning class's ordered menu, while
//! every explicit request reads the compiled one (issue #156).

use super::*;
use crate::plan::test_support::{lay, laying, problem};

type Shape = (usize, usize);
type Menu = Vec<Shape>;

/// A square `n x n x n` packed plan with the requested row-block mode. The
/// element type does not enter the row-block decision; only the menu handed to
/// it does, so an `f64` plan can exercise the `f32` menu shapes below.
fn square_plan(n: i64, row_block: RowBlock) -> PackedPlan {
    let a = lay(&[n, n]);
    let b = lay(&[n, n]);
    let d = laying(&[n, n], &[1, n]);
    let cfg = PlanConfig {
        row_block,
        ..PlanConfig::default()
    };
    PackedPlan::from_problem(&problem(&a, &[0, 2], &b, &[2, 1], &d, &[0, 1]), &cfg).unwrap()
}

/// The f32 real AVX-512 menus as the stage-3 Amd256 measurement resolved them:
/// the compiled order `[48x8, 32x8, 16x10]`, and the class order `[1, 0, 2]`
/// that puts `32x8` first (issue #156
/// `#issuecomment-6519490450`, section B, read in `#issuecomment-6520150036`).
///
/// `TuningClass` is detected from CPUID inside `tcontract-kernel` and cannot be
/// injected from this crate without widening its public API, so the test supplies
/// the two menus that row resolves to. The kernel tests pin the row itself
/// (`menu_order`) and that it turns the compiled menu into this ordered one
/// (`apply_order`).
fn sim_amd256_f32_menus() -> (Menu, Menu) {
    let compiled = vec![(48, 8), (32, 8), (16, 10)];
    let ordered = vec![compiled[1], compiled[0], compiled[2]];
    (ordered, compiled)
}

/// The shape a plan resolves to, mirroring `resolve`'s menu choice and its
/// fallback to the menu head.
fn chosen_shape(p: &PackedPlan, ordered: &[Shape], compiled: &[Shape]) -> Shape {
    let menu = menu_for_row_block(p.row_block_mode, ordered, compiled);
    p.row_block(menu).map_or(menu[0], |i| menu[i])
}

/// On a simulated Amd256 f32 host Auto takes the class's `32x8` default for a
/// square 256 problem (deep enough that the row-block rule leaves the default
/// alone), but `idx=0` still names the compiled head, `48x8`. `base` is the same
/// compiled default, and `mr=` still reads the compiled menu.
#[test]
fn auto_takes_the_class_order_while_index_keeps_the_compiled_position() {
    let (ordered, compiled) = sim_amd256_f32_menus();

    let auto = square_plan(256, RowBlock::Auto);
    assert_eq!(auto.stats.k, 256);
    assert_eq!(
        chosen_shape(&auto, &ordered, &compiled),
        (32, 8),
        "Auto starts from the class's measured default"
    );

    let index0 = square_plan(256, RowBlock::Index(0));
    assert_eq!(
        chosen_shape(&index0, &ordered, &compiled),
        (48, 8),
        "idx=0 is still compiled position 0"
    );

    let base = square_plan(256, RowBlock::Base);
    assert_eq!(chosen_shape(&base, &ordered, &compiled), (48, 8));

    let pin = square_plan(256, RowBlock::Pin(32));
    assert_eq!(chosen_shape(&pin, &ordered, &compiled), (32, 8));
}
