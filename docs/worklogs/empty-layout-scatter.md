# Empty-operand preparation without unreachable strides

## Decision

An operand with a zero extent holds no element, so no numerical access can reach
it and its per-role scatter vectors are unreachable values. A caller may
legitimately leave such an operand's strides at an unreachable magnitude — a
view over an empty slice is valid — and the packed preparation built those
scatter vectors from exactly those strides, so the odometer in
`build_scatter` overflowed (`attempt to add with overflow` for
`dims [2, 0]`, `strides [isize::MAX, 1]`, empty contraction).

The preparation now zeroes the strides of an operand whose total extent is zero
before building its `a_m`/`a_k`/`b_k`/`b_n`/`c_m`/`c_n`/`d_m`/`d_n`/`h_*`
vectors. Vector lengths, and therefore the reported plan stats, are unchanged.
`build_scatter`'s arithmetic stays exact, because real layouts must still
overflow-check.

Review of that first change found the same class of overflow earlier: `fold_axes`
evaluated `stride * extent` and `extent * extent` unchecked *before* the
normalization, so an empty view with several free axes (`dims [2, 2, 0]`,
`strides [isize::MAX, 1, 1]`) still overflowed while folding. Folding now uses
checked products and treats an overflow as "not foldable", which leaves the axes
separate and lets the empty-operand normalization own the scatter vectors. Real
layouts cannot overflow there, because their reachable address range was
validated before planning, so no fold that previously happened is lost.

A second review round found the same class once more in `build_scatter` itself:
`extents.iter().product()` was checked in `i64`, so a role whose prefix product
overflows `i64` while a later extent is zero still panicked. The product is now
computed with an early zero short-circuit and checked multiplication; a zero
extent makes the group empty whatever the other extents are, and an overflowing
product cannot describe a real group — the planner's role-size guards and the
operand validation reject an oversized role before scatter construction — so
both cases yield an empty vector instead of a panic or a wrapped length.

Unit tests cover an empty input operand, an empty output operand, the multi-axis
empty view and the overflowing-prefix scatter; each panicked before the
corresponding change.

## Verification conclusions and constraints

`cargo test --profile ci --workspace`, `cargo test -p tcontract
--release`, workspace/all-target clippy `-D warnings`, fmt, docs, the Rust 1.89
workspace/all-target check and the aarch64-apple check pass. This establishes
planning-level safety for empty operands, not a performance result: the change
only removes an overflow from unreachable values.
