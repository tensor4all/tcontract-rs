//! The vectorised micro-kernel bodies and their configuration builders, shared
//! by every instruction set.
//!
//! **This module contains no kernels and no shapes — only the two macros that
//! generate them.** `kernel::x86` and `kernel::aarch64` each invoke both, and
//! that is the whole reason this file exists: the bodies are written once, over
//! `MV`/`NR` const generics, so a comparison between two instruction sets is not
//! also a comparison between two people's hand-written kernels (D17). The same
//! argument that made the *method* a macro parameter makes the *ISA* one.
//!
//! Splitting them out of `x86.rs` was forced rather than chosen: `macro_rules!`
//! is private to the module that defines it, so a NEON sibling could not have
//! reached them, and the alternative was a second copy of ~250 lines of kernel
//! body. **The AVX-512 kernels this project's entire measurement record rests on
//! are generated from unchanged source text** — the move touched no line of
//! either macro body.
//!
//! Both macros reach the arch modules through `#[macro_use]` on this module in
//! [`super`], which is why the declaration order in `kernel/mod.rs` matters:
//! `macro_rules!` visibility follows textual order, not the module graph.
//!
//! What an ISA must supply is small, and each ISA module documents its own side:
//!
//! * `$lanes` — lanes per vector register, and therefore `MR = MV * L`.
//! * six operations — zero, load, broadcast, fused multiply-add, fused
//!   *negated* multiply-add, store. Their **argument order is the x86 one**
//!   (`fmadd(a, b, c) = a * b + c`), so an ISA whose intrinsics disagree passes
//!   thin shims rather than forking the macro; `kernel::aarch64` does exactly
//!   that, because NEON's `vfmaq` takes the accumulator first.
//! * `$feats` — target features, one `#[target_feature]` attribute each.

/// Generate the four micro-kernels for one real type on one instruction set.
///
/// The bodies are written once, over `MV`/`NR` const generics, so that a shape
/// change is a one-line edit and every method is expressed in the same style —
/// a comparison between methods should not also be a comparison between two
/// people's hand-written assembly (D17).
///
/// The same argument applies across instruction sets, so the ISA is *also* a
/// macro parameter rather than a second copy of the bodies: `$lanes` and the
/// six intrinsics are all that differ between AVX-512 and AVX2. That keeps the
/// AVX2 path from becoming a comparison between two people's kernels too, and —
/// more practically — means the AVX-512 kernels the project's whole measurement
/// record rests on are generated from *unchanged* source text.
///
/// `$feats` lists the target features the kernels need; each becomes its own
/// `#[target_feature(enable = ...)]` attribute.
///
/// Every generated function is `unsafe` twice over — raw pointers, and a
/// `#[target_feature]` the CPU may not have — so each carries a `# Safety`
/// section spelling out both. The counts differ per method because the packed
/// formats do; they are exactly the [`Ukr`] fields `a_per_k`, `b_per_k` and
/// `tile`, which is the invariant `panel_sizes_are_self_consistent` checks.
macro_rules! simd_kernels {
    (
        $modname:ident, $t:ty, $v:ty, $lanes:expr, [$($feat:literal),+ $(,)?],
        $zero:ident, $load:ident, $set1:ident, $fmadd:ident, $fnmadd:ident, $store:ident
    ) => {
        pub mod $modname {
            use super::*;

            /// Lanes per vector register.
            pub const L: usize = $lanes;

            /// `ab[j*MR + i] = sum_p a[p*MR + i] * b[p*NR + j]`, `MR = MV*L`.
            ///
            /// Also the 1m kernel: see [`onem`].
            ///
            /// # Safety
            /// The CPU must support this module's target features. `a` must be
            /// valid for `MV*L*kc` reads, `b` for `NR*kc` reads, and `ab` for
            /// `MV*L*NR` writes. Loads and stores are unaligned, so no
            /// alignment beyond `$t`'s own is required.
            $(#[target_feature(enable = $feat)])+
            pub unsafe fn real<const MV: usize, const NR: usize>(
                kc: usize,
                a: *const $t,
                b: *const $t,
                ab: *mut $t,
            ) {
                let mr = MV * L;
                let mut acc = [[$zero(); MV]; NR];
                for p in 0..kc {
                    let ap = a.add(p * mr);
                    let bp = b.add(p * NR);
                    let mut av = [$zero(); MV];
                    for u in 0..MV {
                        av[u] = $load(ap.add(u * L));
                    }
                    for j in 0..NR {
                        let bv = $set1(*bp.add(j));
                        for u in 0..MV {
                            acc[j][u] = $fmadd(av[u], bv, acc[j][u]);
                        }
                    }
                }
                for j in 0..NR {
                    for u in 0..MV {
                        $store(ab.add(j * mr + u * L), acc[j][u]);
                    }
                }
            }

            /// Planar / split-complex kernel over `MR = MV*L` complex rows.
            ///
            /// Both panels arrive as `[re plane | im plane]` per k-step, and the
            /// tile leaves as `[re plane | im plane]`, each `MR x NR`
            /// column-major. No shuffles anywhere: the four real products of a
            /// complex FMA are four `vfmadd`/`vfnmadd` on data that is already
            /// in the right lanes.
            ///
            /// # Safety
            /// The CPU must support this module's target features. `a` must be
            /// valid for `2*MV*L*kc` reads, `b` for `2*NR*kc` reads, and `ab`
            /// for `2*MV*L*NR` writes — twice the real kernel's, for the two
            /// planes. Accesses are unaligned.
            $(#[target_feature(enable = $feat)])+
            pub unsafe fn planar<const MV: usize, const NR: usize>(
                kc: usize,
                a: *const $t,
                b: *const $t,
                ab: *mut $t,
            ) {
                let mr = MV * L;
                let mut cr = [[$zero(); MV]; NR];
                let mut ci = [[$zero(); MV]; NR];
                for p in 0..kc {
                    let are = a.add(p * 2 * mr);
                    let aim = are.add(mr);
                    let bre = b.add(p * 2 * NR);
                    let bim = bre.add(NR);
                    let mut ar = [$zero(); MV];
                    let mut ai = [$zero(); MV];
                    for u in 0..MV {
                        ar[u] = $load(are.add(u * L));
                        ai[u] = $load(aim.add(u * L));
                    }
                    for j in 0..NR {
                        let br = $set1(*bre.add(j));
                        let bi = $set1(*bim.add(j));
                        for u in 0..MV {
                            cr[j][u] = $fmadd(ar[u], br, cr[j][u]);
                            cr[j][u] = $fnmadd(ai[u], bi, cr[j][u]);
                            ci[j][u] = $fmadd(ar[u], bi, ci[j][u]);
                            ci[j][u] = $fmadd(ai[u], br, ci[j][u]);
                        }
                    }
                }
                let plane = mr * NR;
                for j in 0..NR {
                    for u in 0..MV {
                        $store(ab.add(j * mr + u * L), cr[j][u]);
                        $store(ab.add(plane + j * mr + u * L), ci[j][u]);
                    }
                }
            }

            /// Van Zee's 1m: one *real* kernel of shape `2*MR x NR` over `2*kc`
            /// real steps, fed by "1e" packed `A` and "1r" packed `B`.
            ///
            /// `MV` counts the vector registers of the *real* row block, so the
            /// complex micro-tile is `MV*L/2 x NR`.
            ///
            /// # Safety
            /// The CPU must support this module's target features. `kc` is the
            /// *logical* (complex) depth and each logical step is two real ones,
            /// so `a` must be valid for `2*MV*L*kc` reads, `b` for `2*NR*kc`
            /// reads, and `ab` for `MV*L*NR` writes. Accesses are unaligned.
            $(#[target_feature(enable = $feat)])+
            pub unsafe fn onem<const MV: usize, const NR: usize>(
                kc: usize,
                a: *const $t,
                b: *const $t,
                ab: *mut $t,
            ) {
                real::<MV, NR>(2 * kc, a, b, ab)
            }

            /// Karatsuba 3m over `MR = MV*L` complex rows.
            ///
            /// Accumulates `M1 = Ar*Br`, `M2 = Ai*Bi`, `M3 = (Ar+Ai)*(Br+Bi)`
            /// one plane at a time, so only `MV` A-registers are live at once
            /// and the three accumulator planes fit alongside them.
            ///
            /// # Safety
            /// The CPU must support this module's target features. `a` must be
            /// valid for `3*MV*L*kc` reads, `b` for `3*NR*kc` reads, and `ab`
            /// for `3*MV*L*NR` writes. Accesses are unaligned.
            $(#[target_feature(enable = $feat)])+
            pub unsafe fn threem<const MV: usize, const NR: usize>(
                kc: usize,
                a: *const $t,
                b: *const $t,
                ab: *mut $t,
            ) {
                let mr = MV * L;
                let mut m1 = [[$zero(); MV]; NR];
                let mut m2 = [[$zero(); MV]; NR];
                let mut m3 = [[$zero(); MV]; NR];
                for p in 0..kc {
                    let ao = a.add(p * 3 * mr);
                    let bo = b.add(p * 3 * NR);

                    let mut av = [$zero(); MV];
                    for u in 0..MV {
                        av[u] = $load(ao.add(u * L));
                    }
                    for j in 0..NR {
                        let bv = $set1(*bo.add(j));
                        for u in 0..MV {
                            m1[j][u] = $fmadd(av[u], bv, m1[j][u]);
                        }
                    }
                    for u in 0..MV {
                        av[u] = $load(ao.add(mr + u * L));
                    }
                    for j in 0..NR {
                        let bv = $set1(*bo.add(NR + j));
                        for u in 0..MV {
                            m2[j][u] = $fmadd(av[u], bv, m2[j][u]);
                        }
                    }
                    for u in 0..MV {
                        av[u] = $load(ao.add(2 * mr + u * L));
                    }
                    for j in 0..NR {
                        let bv = $set1(*bo.add(2 * NR + j));
                        for u in 0..MV {
                            m3[j][u] = $fmadd(av[u], bv, m3[j][u]);
                        }
                    }
                }
                let plane = mr * NR;
                for j in 0..NR {
                    for u in 0..MV {
                        $store(ab.add(j * mr + u * L), m1[j][u]);
                        $store(ab.add(plane + j * mr + u * L), m2[j][u]);
                        $store(ab.add(2 * plane + j * mr + u * L), m3[j][u]);
                    }
                }
            }

            // ---- plain-fn trampolines ---------------------------------------
            //
            // A `#[target_feature]` function cannot be coerced to a function
            // pointer, so each kernel gets a one-line wrapper. The wrapper is
            // not inlined into the caller (that is the point — the feature set
            // differs), which costs one `call` per micro-tile against
            // `kc * MR * NR` FMAs of work.

            /// Function-pointer form of [`real`].
            ///
            /// # Safety
            /// As [`real`], including the target-feature requirement — the
            /// wrapper drops the `#[target_feature]` attribute, not the
            /// obligation.
            pub unsafe fn tramp_real<const MV: usize, const NR: usize>(
                kc: usize,
                a: *const $t,
                b: *const $t,
                ab: *mut $t,
            ) {
                real::<MV, NR>(kc, a, b, ab)
            }
            /// Function-pointer form of [`planar`].
            ///
            /// # Safety
            /// As [`planar`], including the target-feature requirement.
            pub unsafe fn tramp_planar<const MV: usize, const NR: usize>(
                kc: usize,
                a: *const $t,
                b: *const $t,
                ab: *mut $t,
            ) {
                planar::<MV, NR>(kc, a, b, ab)
            }
            /// Function-pointer form of [`onem`].
            ///
            /// # Safety
            /// As [`onem`], including the target-feature requirement.
            pub unsafe fn tramp_onem<const MV: usize, const NR: usize>(
                kc: usize,
                a: *const $t,
                b: *const $t,
                ab: *mut $t,
            ) {
                onem::<MV, NR>(kc, a, b, ab)
            }
            /// Function-pointer form of [`threem`].
            ///
            /// # Safety
            /// As [`threem`], including the target-feature requirement.
            pub unsafe fn tramp_threem<const MV: usize, const NR: usize>(
                kc: usize,
                a: *const $t,
                b: *const $t,
                ab: *mut $t,
            ) {
                threem::<MV, NR>(kc, a, b, ab)
            }

            /// Live-extent variants of [`real`] and [`planar`].
            ///
            /// A partial micro-tile — the last `MR` block of a row panel or the
            /// last `NR` block of a column panel — costs a whole register tile
            /// under the full kernels: `ceil(n/NR)*NR/n` of the arithmetic for a
            /// fraction of the work. The variants here run the K loop only for
            /// its live vectors and columns, read the packed slivers at the
            /// **full** family strides (`MV*L` reals per k for A and `NR` for B;
            /// doubled for the two planes of a planar panel) so that packing is
            /// untouched, and then initialise the rest of the declared
            /// `MR x NR` tile with zeros — no K-loop work there, but the
            /// full-initialisation contract of [`KernelFamily`](crate::KernelFamily)
            /// holds, which is what lets the collect-grid path
            /// (`pack::permute::permute_grid_rows`, which reads every column of
            /// every tile) keep permuting whole tiles before clipped emission.
            ///
            /// The entry point dispatches on the live extent to a
            /// monomorphised body: `ceil(m_live/L)` vector registers by
            /// `n_live` columns.
            ///
            /// Wired into the family ABI for the x86 menus only (`configs!` with
            /// `edges = yes`); a target whose families do not expose them would
            /// otherwise compile them and never run them, so they are not built
            /// there.
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            pub mod edge {
                use super::*;

                /// Widest live row extent, in vector registers, the dispatch
                /// chains below cover. `real_tramp` asserts that the shape it is
                /// instantiated for is inside the box, and
                /// `edge_entries_match_reference_at_every_live_extent` exercises
                /// every shape that has an entry, so a menu entry outside it
                /// fails loudly instead of dispatching to the wrong body.
                ///
                /// ponytail: the chains enumerate one fixed box (three vector
                /// registers by twelve columns) rather than each shape's own
                /// extent, because `macro_rules!` cannot repeat over a
                /// const-generic bound. The price is a few dead arms per narrow
                /// shape, which `real_edge`/`planar_edge` turn into a no-op
                /// rather than an out-of-range access; the ceiling is a shape
                /// wider than the box, caught by the assert.
                pub const MV_MAX: usize = 3;
                /// Widest live column extent the chains cover; see [`MV_MAX`].
                pub const NR_MAX: usize = 12;

                /// Live-extent [`real`](super::real).
                ///
                /// Computes `MVL x NRL` of the declared `MR x NR` tile (from
                /// zero) and zeroes the rest of that tile.
                ///
                /// # Safety
                /// The CPU must support this module's target features. `a` must
                /// be valid for `MV*L*kc` reads, `b` for `NR*kc` reads and `ab`
                /// for `MV*L*NR` writes — the full tile's extents, because the
                /// strides are the full tile's. Accesses are unaligned.
                $(#[target_feature(enable = $feat)])+
                pub unsafe fn real_edge<
                    const MV: usize,
                    const NR: usize,
                    const MVL: usize,
                    const NRL: usize,
                >(
                    kc: usize,
                    a: *const $t,
                    b: *const $t,
                    ab: *mut $t,
                ) {
                    // The dispatch chain has one arm per register block of its
                    // widest shape, so a shape narrower than that instantiates
                    // arms it can never reach. Returning here keeps every
                    // compiled instantiation inside the declared tile even if
                    // one were reached; for a reachable arm this folds away.
                    if MVL > MV || NRL > NR {
                        return;
                    }
                    let mr = MV * L;
                    let mut acc = [[$zero(); MVL]; NRL];
                    for p in 0..kc {
                        let ap = a.add(p * mr);
                        let bp = b.add(p * NR);
                        let mut av = [$zero(); MVL];
                        for u in 0..MVL {
                            av[u] = $load(ap.add(u * L));
                        }
                        for j in 0..NRL {
                            let bv = $set1(*bp.add(j));
                            for u in 0..MVL {
                                acc[j][u] = $fmadd(av[u], bv, acc[j][u]);
                            }
                        }
                    }
                    // The live block, at the declared column stride `MR`.
                    for j in 0..NRL {
                        for u in 0..MVL {
                            $store(ab.add(j * mr + u * L), acc[j][u]);
                        }
                    }
                    // The dead slots, without K-loop work: first the vectors
                    // past `MVL` of each live column, then the dead columns.
                    for j in 0..NRL {
                        for u in MVL..MV {
                            $store(ab.add(j * mr + u * L), $zero());
                        }
                    }
                    for j in NRL..NR {
                        for u in 0..MV {
                            $store(ab.add(j * mr + u * L), $zero());
                        }
                    }
                }

                /// Live-extent [`planar`](super::planar).
                ///
                /// Computes `MVL x NRL` complex elements of the declared
                /// `MR x NR` tile (from zero) and zeroes the rest of it, in both
                /// planes.
                ///
                /// # Safety
                /// The CPU must support this module's target features. `a` must
                /// be valid for `2*MV*L*kc` reads, `b` for `2*NR*kc` reads and
                /// `ab` for `2*MV*L*NR` writes. Accesses are unaligned.
                $(#[target_feature(enable = $feat)])+
                pub unsafe fn planar_edge<
                    const MV: usize,
                    const NR: usize,
                    const MVL: usize,
                    const NRL: usize,
                >(
                    kc: usize,
                    a: *const $t,
                    b: *const $t,
                    ab: *mut $t,
                ) {
                    // As `real_edge`: a dead arm of the dispatch chain is a
                    // no-op rather than an out-of-range access.
                    if MVL > MV || NRL > NR {
                        return;
                    }
                    let mr = MV * L;
                    let plane = mr * NR;
                    let mut cr = [[$zero(); MVL]; NRL];
                    let mut ci = [[$zero(); MVL]; NRL];
                    for p in 0..kc {
                        let are = a.add(p * 2 * mr);
                        let aim = are.add(mr);
                        let bre = b.add(p * 2 * NR);
                        let bim = bre.add(NR);
                        let mut ar = [$zero(); MVL];
                        let mut ai = [$zero(); MVL];
                        for u in 0..MVL {
                            ar[u] = $load(are.add(u * L));
                            ai[u] = $load(aim.add(u * L));
                        }
                        for j in 0..NRL {
                            let br = $set1(*bre.add(j));
                            let bi = $set1(*bim.add(j));
                            for u in 0..MVL {
                                cr[j][u] = $fmadd(ar[u], br, cr[j][u]);
                                cr[j][u] = $fnmadd(ai[u], bi, cr[j][u]);
                                ci[j][u] = $fmadd(ar[u], bi, ci[j][u]);
                                ci[j][u] = $fmadd(ai[u], br, ci[j][u]);
                            }
                        }
                    }
                    for j in 0..NRL {
                        for u in 0..MVL {
                            $store(ab.add(j * mr + u * L), cr[j][u]);
                            $store(ab.add(plane + j * mr + u * L), ci[j][u]);
                        }
                    }
                    // The dead slots, in both planes.
                    for j in 0..NRL {
                        for u in MVL..MV {
                            $store(ab.add(j * mr + u * L), $zero());
                            $store(ab.add(plane + j * mr + u * L), $zero());
                        }
                    }
                    for j in NRL..NR {
                        for u in 0..MV {
                            $store(ab.add(j * mr + u * L), $zero());
                            $store(ab.add(plane + j * mr + u * L), $zero());
                        }
                    }
                }

                /// The row half of the real chain: `MVL` vectors of `nrl`
                /// columns.
                ///
                /// # Safety
                /// As [`real_edge`], and `MVL <= MV`, `nrl <= NR`.
                $(#[target_feature(enable = $feat)])+
                unsafe fn real_cols<const MV: usize, const NR: usize, const MVL: usize>(
                    kc: usize,
                    a: *const $t,
                    b: *const $t,
                    ab: *mut $t,
                    nrl: usize,
                ) {
                    match nrl {
                        1 => unsafe { real_edge::<MV, NR, MVL, 1>(kc, a, b, ab) },
                        2 => unsafe { real_edge::<MV, NR, MVL, 2>(kc, a, b, ab) },
                        3 => unsafe { real_edge::<MV, NR, MVL, 3>(kc, a, b, ab) },
                        4 => unsafe { real_edge::<MV, NR, MVL, 4>(kc, a, b, ab) },
                        5 => unsafe { real_edge::<MV, NR, MVL, 5>(kc, a, b, ab) },
                        6 => unsafe { real_edge::<MV, NR, MVL, 6>(kc, a, b, ab) },
                        7 => unsafe { real_edge::<MV, NR, MVL, 7>(kc, a, b, ab) },
                        8 => unsafe { real_edge::<MV, NR, MVL, 8>(kc, a, b, ab) },
                        9 => unsafe { real_edge::<MV, NR, MVL, 9>(kc, a, b, ab) },
                        10 => unsafe { real_edge::<MV, NR, MVL, 10>(kc, a, b, ab) },
                        11 => unsafe { real_edge::<MV, NR, MVL, 11>(kc, a, b, ab) },
                        // `real_tramp` clamps `nrl` to `NR <= NR_MAX`.
                        _ => unsafe { real_edge::<MV, NR, MVL, NR_MAX>(kc, a, b, ab) },
                    }
                }

                /// The row half of the planar chain; as [`real_cols`].
                ///
                /// # Safety
                /// As [`planar_edge`], and `MVL <= MV`, `nrl <= NR`.
                $(#[target_feature(enable = $feat)])+
                unsafe fn planar_cols<const MV: usize, const NR: usize, const MVL: usize>(
                    kc: usize,
                    a: *const $t,
                    b: *const $t,
                    ab: *mut $t,
                    nrl: usize,
                ) {
                    match nrl {
                        1 => unsafe { planar_edge::<MV, NR, MVL, 1>(kc, a, b, ab) },
                        2 => unsafe { planar_edge::<MV, NR, MVL, 2>(kc, a, b, ab) },
                        3 => unsafe { planar_edge::<MV, NR, MVL, 3>(kc, a, b, ab) },
                        4 => unsafe { planar_edge::<MV, NR, MVL, 4>(kc, a, b, ab) },
                        5 => unsafe { planar_edge::<MV, NR, MVL, 5>(kc, a, b, ab) },
                        6 => unsafe { planar_edge::<MV, NR, MVL, 6>(kc, a, b, ab) },
                        7 => unsafe { planar_edge::<MV, NR, MVL, 7>(kc, a, b, ab) },
                        8 => unsafe { planar_edge::<MV, NR, MVL, 8>(kc, a, b, ab) },
                        9 => unsafe { planar_edge::<MV, NR, MVL, 9>(kc, a, b, ab) },
                        10 => unsafe { planar_edge::<MV, NR, MVL, 10>(kc, a, b, ab) },
                        11 => unsafe { planar_edge::<MV, NR, MVL, 11>(kc, a, b, ab) },
                        _ => unsafe { planar_edge::<MV, NR, MVL, NR_MAX>(kc, a, b, ab) },
                    }
                }

                /// Real entry: `m_live` rows by `n_live` columns of the shape's
                /// declared tile, in whatever tile format the family has.
                ///
                /// # Safety
                /// As [`real_edge`]. The extents are clamped to the shape, so a
                /// caller cannot address outside the declared panels or tile;
                /// an extent outside `1..=MR`/`1..=NR` is a caller error, not a
                /// supported mode.
                $(#[target_feature(enable = $feat)])+
                pub unsafe fn real_tramp<const MV: usize, const NR: usize>(
                    kc: usize,
                    a: *const $t,
                    b: *const $t,
                    ab: *mut $t,
                    m_live: usize,
                    n_live: usize,
                ) {
                    assert!(MV <= MV_MAX && NR <= NR_MAX);
                    let mvl = m_live.div_ceil(L).clamp(1, MV);
                    let nrl = n_live.clamp(1, NR);
                    match mvl {
                        1 => unsafe { real_cols::<MV, NR, 1>(kc, a, b, ab, nrl) },
                        2 => unsafe { real_cols::<MV, NR, 2>(kc, a, b, ab, nrl) },
                        _ => unsafe { real_cols::<MV, NR, MV_MAX>(kc, a, b, ab, nrl) },
                    }
                }

                /// Planar entry; as [`real_tramp`], with the planar kernel's
                /// extents.
                ///
                /// # Safety
                /// As [`planar_edge`], including the clamping of the extents.
                $(#[target_feature(enable = $feat)])+
                pub unsafe fn planar_tramp<const MV: usize, const NR: usize>(
                    kc: usize,
                    a: *const $t,
                    b: *const $t,
                    ab: *mut $t,
                    m_live: usize,
                    n_live: usize,
                ) {
                    assert!(MV <= MV_MAX && NR <= NR_MAX);
                    let mvl = m_live.div_ceil(L).clamp(1, MV);
                    let nrl = n_live.clamp(1, NR);
                    match mvl {
                        1 => unsafe { planar_cols::<MV, NR, 1>(kc, a, b, ab, nrl) },
                        2 => unsafe { planar_cols::<MV, NR, 2>(kc, a, b, ab, nrl) },
                        _ => unsafe { planar_cols::<MV, NR, MV_MAX>(kc, a, b, ab, nrl) },
                    }
                }

                // ---- plain-fn trampolines, as for the full kernels ----------
                //
                // A `#[target_feature]` function cannot be coerced to a
                // function pointer, so [`Ukr`](crate::Ukr)-style dispatch needs
                // one plain wrapper per entry.

                /// Function-pointer form of [`real_tramp`].
                ///
                /// # Safety
                /// As [`real_tramp`], including the target-feature requirement —
                /// the wrapper drops the attribute, not the obligation.
                pub unsafe fn tramp_real<const MV: usize, const NR: usize>(
                    kc: usize,
                    a: *const $t,
                    b: *const $t,
                    ab: *mut $t,
                    m_live: usize,
                    n_live: usize,
                ) {
                    real_tramp::<MV, NR>(kc, a, b, ab, m_live, n_live)
                }

                /// Function-pointer form of [`planar_tramp`].
                ///
                /// # Safety
                /// As [`planar_tramp`], including the target-feature requirement.
                pub unsafe fn tramp_planar<const MV: usize, const NR: usize>(
                    kc: usize,
                    a: *const $t,
                    b: *const $t,
                    ab: *mut $t,
                    m_live: usize,
                    n_live: usize,
                ) {
                    planar_tramp::<MV, NR>(kc, a, b, ab, m_live, n_live)
                }
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Configuration builders
// ---------------------------------------------------------------------------

/// The live-extent entry of one menu slot, or `None` when the menu has none.
///
/// Split out of [`configs!`] because a `macro_rules!` arm cannot branch on a
/// token at expansion time: `edges = yes|no` picks the arm, and the `no` arm is
/// the same call shape so that the two spellings stay in step.
macro_rules! edge_entry {
    (yes, real, $m:ident, $mv:literal, $nr:literal) => {
        Some($m::edge::tramp_real::<$mv, $nr>)
    };
    (yes, planar, $m:ident, $mv:literal, $nr:literal) => {
        Some($m::edge::tramp_planar::<$mv, $nr>)
    };
    (no, $kind:ident, $m:ident, $mv:literal, $nr:literal) => {{
        // A menu without edge variants: the shape exists so that this arm has
        // the same call shape as the enabled one.
        let _ = ($mv, $nr);
        None
    }};
}

/// Assemble the [`Ukr`] descriptors for one instruction set / type from a
/// *menu* of register blocks per method, the default first.
///
/// Every field is derived from `MV`/`NR` and the lane count, so a shape change
/// cannot desynchronise the descriptor from the kernel it describes — the
/// `panel_sizes_are_self_consistent` test in `kernel::tests` checks the
/// arithmetic, and `kernels_match_reference_*` checks the semantics.
///
/// # Why a menu rather than one shape
///
/// The first entry is the shape measured fastest in isolation and is what runs
/// unless something asks for otherwise. The alternates exist because peak
/// kernel throughput is not the only thing `MR` decides: it also sets the
/// granularity at which the output's row scatter is blocked, and therefore
/// whether the write-back takes its unit-stride path or its gather path. A
/// shape 10–20% off peak that moves whole block families onto the fast path
/// wins on any contraction that is nowhere near kernel-bound. See
/// `tensorcontract::Plan::row_block`.
macro_rules! configs {
    ($t:ty, $m:ident, $isa:literal,
     real   = [$(($rmv:literal, $rnr:literal)),+ $(,)?],
     planar = [$(($pmv:literal, $pnr:literal)),+ $(,)?],
     onem   = [$(($omv:literal, $onr:literal)),+ $(,)?],
     threem = [$(($tmv:literal, $tnr:literal)),+ $(,)?],
     edges  = $edges:ident $(,)?) => {
        /// Logical `(MR, NR)` with a real kernel, default first.
        ///
        /// The menu is keyed by **position**, not by `MR`: two entries may share
        /// an `MR` and differ only in `NR`, which is a shape the measurement
        /// wanted and the old `MR`-keyed menu could not express at all (A35).
        pub const REAL_ROW_BLOCKS: &[(usize, usize)] = &[$(($rmv * $m::L, $rnr)),+];
        /// Logical (complex) `(MR, NR)` with a planar kernel, default first.
        pub const PLANAR_ROW_BLOCKS: &[(usize, usize)] = &[$(($pmv * $m::L, $pnr)),+];
        /// Ditto for 1m. The complex tile is half the real row block.
        pub const ONEM_ROW_BLOCKS: &[(usize, usize)] = &[$(($omv * $m::L / 2, $onr)),+];
        /// Ditto for 3m.
        pub const THREEM_ROW_BLOCKS: &[(usize, usize)] = &[$(($tmv * $m::L, $tnr)),+];

        fn real_cfg<const MV: usize, const NR: usize>() -> KernelConfig<$t> {
            let mr = MV * $m::L;
            KernelConfig {
                ukr: Ukr {
                    mr,
                    nr: NR,
                    a_per_k: mr,
                    b_per_k: NR,
                    tile: mr * NR,
                    a_pack: PackFormat::Real,
                    b_pack: PackFormat::Real,
                    tile_fmt: TileFormat::Real,
                    func: $m::tramp_real::<MV, NR>,
                    name: concat!($isa, "-real"),
                },
                blk: Blocking::derive(core::mem::size_of::<$t>(), 1, 1),
            }
        }

        fn planar_cfg<const MV: usize, const NR: usize>() -> KernelConfig<$t> {
            let mr = MV * $m::L;
            KernelConfig {
                ukr: Ukr {
                    mr,
                    nr: NR,
                    a_per_k: 2 * mr,
                    b_per_k: 2 * NR,
                    tile: 2 * mr * NR,
                    a_pack: PackFormat::Planar,
                    b_pack: PackFormat::Planar,
                    tile_fmt: TileFormat::Planar,
                    func: $m::tramp_planar::<MV, NR>,
                    name: concat!($isa, "-planar"),
                },
                blk: Blocking::derive(core::mem::size_of::<$t>(), 2, 2),
            }
        }

        /// The real kernel is `MR2 x NR`; the complex tile is half as tall.
        /// `MR2` is even for any `MV >= 1` since `L` is a power of two.
        fn onem_cfg<const MV: usize, const NR: usize>() -> KernelConfig<$t> {
            let mr2 = MV * $m::L;
            KernelConfig {
                ukr: Ukr {
                    mr: mr2 / 2,
                    nr: NR,
                    a_per_k: 2 * mr2,
                    b_per_k: 2 * NR,
                    tile: mr2 * NR,
                    a_pack: PackFormat::OneE,
                    b_pack: PackFormat::Planar,
                    tile_fmt: TileFormat::OneM,
                    func: $m::tramp_onem::<MV, NR>,
                    name: concat!($isa, "-1m"),
                },
                // Four reals per complex element in packed A, so the same L2
                // budget buys half the rows planar gets.
                blk: Blocking::derive(core::mem::size_of::<$t>(), 4, 2),
            }
        }

        fn threem_cfg<const MV: usize, const NR: usize>() -> KernelConfig<$t> {
            let mr = MV * $m::L;
            KernelConfig {
                ukr: Ukr {
                    mr,
                    nr: NR,
                    a_per_k: 3 * mr,
                    b_per_k: 3 * NR,
                    tile: 3 * mr * NR,
                    a_pack: PackFormat::ThreeM,
                    b_pack: PackFormat::ThreeM,
                    tile_fmt: TileFormat::ThreeM,
                    func: $m::tramp_threem::<MV, NR>,
                    name: concat!($isa, "-3m"),
                },
                blk: Blocking::derive(core::mem::size_of::<$t>(), 3, 3),
            }
        }

        /// The real kernel at **menu position** `i`, or `None` past the end.
        ///
        /// Positional and not `MR`-keyed, so an entry that shares its `MR` with
        /// an earlier one is still reachable — see [`REAL_ROW_BLOCKS`] and A35.
        pub fn real_config_at(i: usize) -> Option<KernelConfig<$t>> {
            let mut n = 0usize;
            $(
                if i == n { return Some(real_cfg::<$rmv, $rnr>()); }
                n += 1;
            )+
            let _ = n;
            None
        }

        /// The complex kernel for `method` at menu position `i`.
        pub fn cplx_config_at(method: ComplexMethod, i: usize) -> Option<KernelConfig<$t>> {
            let mut n = 0usize;
            match method {
                ComplexMethod::Planar => {
                    $(
                        if i == n { return Some(planar_cfg::<$pmv, $pnr>()); }
                        n += 1;
                    )+
                }
                ComplexMethod::OneM => {
                    $(
                        if i == n { return Some(onem_cfg::<$omv, $onr>()); }
                        n += 1;
                    )+
                }
                ComplexMethod::ThreeM => {
                    $(
                        if i == n { return Some(threem_cfg::<$tmv, $tnr>()); }
                        n += 1;
                    )+
                }
            }
            let _ = n;
            None
        }

        pub fn cplx_row_blocks(method: ComplexMethod) -> &'static [(usize, usize)] {
            match method {
                ComplexMethod::Planar => PLANAR_ROW_BLOCKS,
                ComplexMethod::OneM => ONEM_ROW_BLOCKS,
                ComplexMethod::ThreeM => THREEM_ROW_BLOCKS,
            }
        }

        /// The menu for either domain, which is the shape `IsaConfigs` wants.
        pub fn row_blocks(complex: bool, method: ComplexMethod) -> &'static [(usize, usize)] {
            if complex {
                cplx_row_blocks(method)
            } else {
                REAL_ROW_BLOCKS
            }
        }

        /// The config at a chosen menu position in either domain.
        pub fn config_at(
            complex: bool,
            method: ComplexMethod,
            i: usize,
        ) -> Option<KernelConfig<$t>> {
            if complex {
                cplx_config_at(method, i)
            } else {
                real_config_at(i)
            }
        }

        /// The live-extent ("edge") entry of the real kernel at menu position
        /// `i`, or `None` past the end and when this menu has none.
        ///
        /// Its signature is [`TileEdgeFn`](crate::TileEdgeFn): the same packed
        /// panels and the same declared tile as [`real_config_at`], with the
        /// live row and column extents as two extra arguments.
        pub fn real_edge_at(i: usize) -> Option<crate::TileEdgeFn<$t>> {
            let mut n = 0usize;
            $(
                if i == n { return edge_entry!($edges, real, $m, $rmv, $rnr); }
                n += 1;
            )+
            let _ = n;
            None
        }

        /// The live-extent entry for `method` at menu position `i`.
        ///
        /// Only the native planar kernel has one. 1m's k loop is the real
        /// kernel's at twice the logical k and 3m's is three planes of it — both
        /// are wrappers around a *different* geometry, so an inherited real edge
        /// entry would bypass the transformation rather than accelerate it. They
        /// keep the full-tile entry.
        pub fn cplx_edge_at(method: ComplexMethod, i: usize) -> Option<crate::TileEdgeFn<$t>> {
            if method == ComplexMethod::Planar {
                let mut n = 0usize;
                $(
                    if i == n { return edge_entry!($edges, planar, $m, $pmv, $pnr); }
                    n += 1;
                )+
                let _ = n;
            }
            None
        }

        /// The live-extent entry at a chosen menu position in either domain.
        pub fn edge_at(
            complex: bool,
            method: ComplexMethod,
            i: usize,
        ) -> Option<crate::TileEdgeFn<$t>> {
            if complex {
                cplx_edge_at(method, i)
            } else {
                real_edge_at(i)
            }
        }

        pub fn real_config() -> KernelConfig<$t> {
            real_config_at(0).expect("the menu is never empty")
        }

        pub fn cplx_config(method: ComplexMethod) -> KernelConfig<$t> {
            cplx_config_at(method, 0).expect("the menu is never empty")
        }
    };
}
