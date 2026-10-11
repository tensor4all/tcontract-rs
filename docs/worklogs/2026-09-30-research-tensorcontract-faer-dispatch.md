# tensorcontract kernel-selection structure, faer/gemm dispatch architecture, and descriptor seams

This was a read-only study on 2026-09-30. Nothing was built or run.
It builds on `docs/worklogs/2026-09-30-gemm-strategy-faer-vs-tensorcontract.md` and does not repeat what that covers (tile shapes, the Zen 3 analysis, the cause ranking).

Path abbreviations:
- `TC` = `~/tensor4all/tcontract-rs/tensorprimitives/crates/tensorcontract/src/`
- `REG` = `~/.cargo/registry/src/index.crates.io-*/`

Items marked **[inf]** are my inferences. I did not read them verbatim or measure them.

---

## A. tensorcontract inventory

### A1. Module map

| file | what it owns |
|---|---|
| `lib.rs` | Public API: `contract`, `TensorView(Mut)`, `Plan::run`/`run_with`/`run_raw`/`run_raw_with` (lib.rs:497, 545, 593, 658, 679). Also the `env_once!` macro, which caches one variable per process in a `OnceLock` (lib.rs:313-329), and the re-exports `Element, Real, C32, C64, Error, Blocking, ComplexMethod, KernelSet, Layout, ElementOp, Operand, Plan, PlanStats` (lib.rs:349-353). |
| `plan.rs` | Index analysis: label classes, folding, diagonals, reductions, scatter vectors. Also orientation (`transposes_gemm`, :808), the row-block rule (`row_block`/`preferred_row_block`/`row_block_score`, :922-972), the thread partition (`partition_with`, :669) and the env knobs (:995-1180). **`Plan` is not generic over the dtype.** Its per-plan choices are only `blocking: Option<Blocking>`, `method: Option<ComplexMethod>` and `threads: Option<usize>` (plan.rs:298-302). |
| `layout.rs`, `scatter.rs` | Extents and strides; scatter and block-scatter vectors (`build_block_scatter`, `IRREGULAR`). |
| `element.rs` | `Real` trait (ZERO/ONE/from_f64/to_f64, element.rs:35-61), impl'd for f32/f64. `Element` trait (`type Real`, `IS_COMPLEX`, `FLOPS_PER_MAC`, re/im/from_parts/conj/mul/add/sub, element.rs:91-128), impl'd for f32/f64/C32/C64. |
| `kernel/mod.rs` | `ComplexMethod` {Planar, OneM, ThreeM} (:167-178). `PackFormat` {Real, Planar, OneE, ThreeM} (:248-261). `TileFormat` {Real, Planar, OneM, ThreeM} (:277-289). **`Ukr<T>`**, the micro-kernel descriptor (:291-329). `Blocking {mc,kc,nc}` with `derive`/`derive_at_depth`/`model` (:363-450). **`KernelConfig<T> = {ukr, blk}`** with `normalise_for`/`retarget_threads`/`with_blocking` (:459-551). Env blocking overrides (:553-606). `trait KernelSet` (:636-670). `KernelForce` and `kernel_force()` (:684-741). `impl_kernel_set!`, which wires f32/f64 to SIMD or scalar (:765-845). `selected_config`, `selected_kernel_name`, `config_for`, `config_for_plan`, `plan_config` (:850-919). Kernel-contract tests (:921-1397). |
| `kernel/simd.rs` | Macros only. `simd_kernels!` (:54-303) generates `real`/`planar`/`onem`/`threem` bodies over `<const MV, const NR>` under `#[target_feature]`, plus plain-`fn` trampolines (:243-300). `configs!` (:327-506) builds `KernelConfig`s from a per-method `(MV, NR)` menu and exposes `*_ROW_BLOCKS`, `config_at`, `real_config`, `cplx_config`. |
| `kernel/x86.rs` | Instantiates `simd_kernels!` for avx512_f64/f32 and avx2_f64/f32 (:86-144). Holds the menus `cfg_avx512_f64` (:180-191), `cfg_avx512_f32` (:230-253), `cfg_avx2_f64` (:350-361) and `cfg_avx2_f32` (:385-396). Defines `enum Isa {Avx512, Avx2}` (:406-412), feature probes cached in `OnceLock`s (:425-456), `available_isas` (:465), `selected_isa` in a `OnceLock` (:481-492), `pick_isa` (:494-509), the **`IsaConfigs<T>` fn-pointer table** (:517-524) and the `dispatch!` macro (:543-617). |
| `kernel/aarch64.rs` | The NEON equivalent. Shims turn `vfmaq` argument order into x86 order (:84-132). Instantiates `neon_f64`/`neon_f32` (:143-169). Menus at :197-240. `Isa {Neon}`, `selected_isa` and `IsaConfigs` mirror x86 (:253-367). |
| `kernel/scalar.rs` | Portable kernels generic over `T: Real` with `const MR, NR`: `real_ukr`, `planar_ukr`, `onem_ukr`, `threem_ukr` (:114-254). Builders `config_real`/`config_cplx` (:260-353). Const-fn `planar_dispatch`/`threem_dispatch` map `MR` to `MR/2` (:359-381). |
| `kernel/cache.rs` | Cache hierarchy probe (sysfs/CPUID/sysctl), cached in a `OnceLock` (`hierarchy`, :296-300). `l3_domains` (:273). `BlockModel {Legacy, Analytical}` and `block_model()` (:788-858). `analytical(PanelGeom, threads, &CacheHierarchy)` (:916). |
| `pack.rs` | `pack_panel<T: Element>(base, vscat, vbs, kscat, vr, conj, fmt, out)` (:54-107). It has a regular-block path and a gather path, zero-fills the edge lanes, and calls `emit` per element. `emit` is `#[inline(always)]` and does a **runtime `match fmt`** (:124-157). Conj is folded in here. |
| `driver.rs` | The five-loop nest. `Ctx` (:160-201). `BPart` 2-D partition (:215-258). `execute` / `execute_with` / `execute_capped` (:268-609). `run_strip` (:628-829). |
| `writeback.rs` | `writeback<T>` (:112-184) picks one of three row-addressing instantiations (gather / unit / strided). `writeback_rows` (:210-260) has a `plain` fast path. `tile_value` (:66-93) does a **runtime `match fmt`** per element. `scale_only` (:270-296). |
| `buffer.rs` | `Panel<T>`: a 64-B aligned `alloc`/`dealloc` scratch buffer allocated per call (:53-85). |
| `spmd.rs` | `trait Spmd: Sync { width(); broadcast(p, &dyn Fn(usize)+Sync) -> bool }` (:31-40). This is a tcontract addition. |
| `pool.rs` | Opt-in parked-thread pool (`TENSORCONTRACT_POOL=on`, :272-285). Bypassed when an `Spmd` is supplied (driver.rs:586-596). |
| `batch.rs` | `contract_batched(_with_threads)` (:140, :174). `run_item` calls `execute_capped(..., cap=1, ...)` (:222-236). |
| `reference.rs`, `error.rs` | Naive oracle for tests; the error type. |

There is no fused-complex "interleaved/native" path, no 4m method, no GPU code and no AVX-512-FP16/SVE code.

### A2. How a kernel is chosen today

The chain runs **on every `execute_capped` call**, at driver.rs:357 `config_for_plan::<T>(plan)`:

1. `config_for_plan` (mod.rs:893-904) gets the method from `plan.complex_method()`. That is the per-plan `Option`, or `ComplexMethod::from_env()`, which is cached (plan.rs:547-550, mod.rs:212-219).
2. It gets the menu from `<T::Real as KernelSet>::row_blocks(IS_COMPLEX, method)`.
3. `plan.row_block(menu)` returns a menu position, or `None`. That position is the output-stride-driven rule or `TENSORCONTRACT_ROWBLOCK` (plan.rs:922-955).
4. `config_at(...)` builds the config at that position. Otherwise `config_for::<T>(method)` is used (mod.rs:870-880).
5. `.retarget_threads(plan.threads())` is applied.
6. The driver then applies `plan.blocking` through `with_blocking` (driver.rs:358-361) and clamps mc and nc to m and n (driver.rs:448-450).

Inside `KernelSet` for f32/f64 (the `impl_kernel_set!` macro, mod.rs:765-845):
- Unless `force_scalar()` holds, it calls `simd_isa::config_*_f64()`.
- That calls `isa_configs_f64(selected_isa()?)`. This is a `match isa` that returns an `IsaConfigs` of *constructor* fn pointers (x86.rs:547-592).
- It then calls the constructor. The constructor instantiates `Ukr { func: tramp_<method>::<MV,NR>, ... }` and `Blocking::derive(...)` (simd.rs:346-424).
- `.normalise()` is applied.
- When no SIMD ISA is available, it falls back to `scalar::config_*::<T, 4, 4>()` (mod.rs:776, 786, 833-844).

Mechanisms, and when each is decided:

| decision | mechanism | when |
|---|---|---|
| ISA | `enum Isa` in a `OnceLock` (`selected_isa`, x86.rs:481-492, aarch64.rs:286-297); probes in their own `OnceLock`s | once per process |
| dtype | Rust generics: `execute_capped<T: Element>` with `T::Real: KernelSet`, a static trait dispatch per real type | compile time (4 monomorphs of the driver: f32, f64, C32, C64) |
| complex method | `Plan.method` override, otherwise the env default | per plan, but re-resolved each execute |
| MR×NR | position on a compiled const-generic menu, chosen by `Plan::row_block` | re-evaluated each execute (it depends only on the plan) |
| micro-kernel | `Ukr.func: unsafe fn(kc, a, b, ab)` fn pointer, called indirectly per micro-tile (driver.rs:769) | extracted each execute |
| pack / writeback format | runtime enums `Ukr.a_pack`, `b_pack`, `tile_fmt`, matched inside per-element helpers | per element; whether LLVM hoists the match out of the loops is **[inf]** unknown |
| KC/MC/NC | `Blocking::derive` (legacy constants) or `Blocking::model` (analytical), then env overrides, then `with_blocking` alignment | each execute |

**Env reads per execute.** `Blocking::derive` calls `env_usize("TENSORCONTRACT_KC_COUPLE")` (mod.rs:393). `env_usize` is explicitly **not cached** (mod.rs:595-606), and every constructor calls `Blocking::derive`. So each `execute` does one uncached `std::env::var` [inf: under std's env read lock and `getenv`, sub-µs]. That matters for tprims-blas `tblis.rs`, which calls `run_raw_with` once per batch item (tprims-blas/src/tblis.rs:71-108).

Every env var and knob:

| knob | values | effect | cached? |
|---|---|---|---|
| `TENSORCONTRACT_KERNEL` | scalar, avx2, avx512, neon, auto | ISA pin. An unavailable ISA falls back to scalar (mod.rs:707-741, x86.rs:494-509) | OnceLock |
| `TENSORCONTRACT_COMPLEX` | planar/split, 1m/onem, 3m/threem/karatsuba | default `ComplexMethod` (mod.rs:192-219) | OnceLock |
| `Plan::with_complex_method` | `ComplexMethod` | per-plan override (plan.rs:541) | stored in the plan |
| `TENSORCONTRACT_ROWBLOCK` | auto (default), base, mr=<n>, idx=<i> | MR×NR menu position (plan.rs:1009-1057) | OnceLock |
| `TENSORCONTRACT_BLOCKMODEL` | legacy (default), model | `Blocking::derive` vs the analytical model (cache.rs:852-858) | OnceLock |
| `TENSORCONTRACT_KC_COUPLE` | usize | sets kc **and** re-derives mc/nc against the budgets (mod.rs:393) | **not cached** |
| `TENSORCONTRACT_MC`, `_KC`, `_NC` | usize | absolute overrides (mod.rs:562-593) | OnceLock |
| `TENSORCONTRACT_MC_PCT`, `_NC_PCT` | percent | scale the derived value. Must not be applied twice, which is why `retarget_threads` is a no-op under Legacy (mod.rs:509-540) | OnceLock |
| `Plan::with_blocking` | `Blocking` | per-plan absolute override. Re-aligned to mr/nr (plan.rs:558, driver.rs:358) | in the plan |
| `TENSORCONTRACT_L3_DOMAINS` | usize | L3-domain count used by partition and model (cache.rs:273-288) | OnceLock |
| `TENSORCONTRACT_ORIENT` | none/ab, swap/ba, legacy | row/column orientation pin (plan.rs:997-1007) | OnceLock |
| `TENSORCONTRACT_PARTITION` | domain (default), legacy, m, n, <pm>x<pn> | thread partition (plan.rs:1128-1162). Shrunk to fit the Spmd width (driver.rs:470-478) | OnceLock |
| `TENSORCONTRACT_THREADS` / `Plan::with_threads` | usize | default thread count, 1. Ignored when an Spmd is passed (driver.rs:461-466) | OnceLock |
| `TENSORCONTRACT_POOL` | on | pooled threads when there is no Spmd (pool.rs:272-285) | OnceLock |
| `TENSORCONTRACT_WRITEBACK` | gather/scatter | forces the general write-back loop (writeback.rs:38-49) | OnceLock |

tcontract sets none of the per-plan knobs. The only exceptions are `with_threads(1)` in a test and a benchmark (tcontract-exec/tests/tensorcontract_seam.rs:41, benchmarks/.../exec_entry.rs:138).

### A3. The micro-kernel contract today

- **Signature:** `unsafe fn(kc: usize, a: *const R, b: *const R, ab: *mut R)` with `R = T::Real` (mod.rs:324). `kc` is the logical depth; for complex it counts complex steps.
- **What it reads:** the packed sliver formats (mod.rs:17-34), with `VR` = MR for A and NR for B:
  - Real: `VR` reals per k-step.
  - Planar: `[re×VR | im×VR]`.
  - 1m, A ("1e"): `4·MR` reals, i.e. two real k-steps of `[re,im,...]` and `[-im,re,...]`.
  - 1m, B: planar ("1r").
  - 3m: `[re | im | re+im]`.
  - The per-k widths are `Ukr.a_per_k` and `b_per_k`. The SIMD kernels use unaligned loads (simd.rs:69-73). Panels are 64-B aligned (buffer.rs:46).
- **What it writes:** it **overwrites** a scratch accumulator tile `ab` of `Ukr.tile` reals, in `TileFormat` (mod.rs:62-69, 313-317):
  - Real: `MR×NR` column-major.
  - Planar: a re plane, then an im plane.
  - 1m: one real `2MR×NR` tile.
  - 3m: three planes M1, M2, M3.
  - It never touches C or D.
- **Edges:** there is no edge handling in the kernel. Packing zero-fills partial slivers (pack.rs:92-104), so the kernel always runs the full MR×NR. Writeback stores only `mrem×nrem` (driver.rs:758-765, writeback.rs:236-258).
- **alpha/beta:** handled entirely in `writeback`:
  - The first kc block uses the caller's beta and C.
  - Later blocks read back D with beta=1 (driver.rs:772-816).
  - There is a `plain` fast path when alpha=1, beta=0 and there is no conj (writeback.rs:234).
- **conj:**
  - conj(A) and conj(B) are folded into packing by negating im (pack.rs:136-155).
  - conj(C) and conj(D) are applied in writeback (writeback.rs:250-255).
  - The kernel is conj-agnostic.
- **1m:** `onem = real::<MV,NR>(2*kc, ...)` (simd.rs:170-177; scalar.rs:193-201). The row and column pack formats swap under the orientation swap (driver.rs:371-375).
- **Target-feature obligation:** the trampolines drop `#[target_feature]` but not the obligation. The CPU check belongs to whoever built the `Ukr` (mod.rs:319-323). Today that is `selected_isa()`.

### A4. Generic vs per-ISA; variant count

- **Generic over dtype (T: Element):** driver (`execute_capped`, `run_strip`), `pack_panel`, `emit`, `writeback`, `writeback_rows`, `tile_value`, `scale_only`.
  - All of these are compiled with **no `#[target_feature]`**, i.e. at SSE2 baseline on this build (see the prior worklog §1).
  - `writeback` has three row-addressing instantiations per T (writeback.rs:172-183).
- **Per ISA (target_feature + const generics):** only the four kernel bodies per `simd_kernels!` instantiation.
- **Scalar portable:** generic over `T: Real` plus const MR/NR. It is also the route for foreign scalar types (the `KernelSet` doc, mod.rs:614-626).

Variant tally. The menus are `(MV, NR)`; logical MR = MV·L, or MV·L/2 for 1m.

| ISA × real type | real | planar | 1m | 3m | total shapes |
|---|---|---|---|---|---|
| avx512 f64 (L=8) | 3 | 3 | 3 | 3 | 12 |
| avx512 f32 (L=16) | 3 | 4 | 4 | 3 | 14 |
| avx2 f64 (L=4) | 3 | 2 | 3 | 2 | 10 |
| avx2 f32 (L=8) | 3 | 2 | 3 | 2 | 10 |
| neon f64 (L=2) | 4 | 3 | 3 | 3 | 13 |
| neon f32 (L=4) | 4 | 3 | 3 | 3 | 13 |
| scalar f32/f64 | 1 (4×4) | 1 (2×4) | 1 | 1 | 4 per type |

Sources: x86.rs:186-189, 236-251, 356-359, 391-394; aarch64.rs:203-206, 235-238; mod.rs:833-844.

That is 46 x86 kernel monomorphs, 26 NEON and 8 scalar. Each has its own trampoline. The 1m bodies share code with `real` (it calls it at depth 2kc) but are separate monomorphs.

There is no AVX-512 variant without FMA semantics, no AVX (non-AVX2) variant, and no SSE variant. An x86 CPU without AVX2+FMA gets scalar.

### A5. Tests and benchmarks that pin kernel behaviour

**`kernel/mod.rs` tests**
- `kernels_match_reference_f64/f32`, over every menu position (:1108-1122).
- `every_available_x86_isa_matches_reference` (:1178-1185).
- **`x86_isas_agree_on_the_pack_contract`** (:1191-1254). This asserts that pack and tile formats are identical across ISAs, and that `a_per_k`, `b_per_k`, `tile` and the mc/nc alignment are consistent.
- `row_block_menus_are_well_formed` (:1266).
- `panel_sizes_are_self_consistent` (:1293).
- `one_m_packs_a_twice_as_large_as_planar` (:1304).
- `three_m_does_fewer_flops` (:1323).
- **`legacy_blocking_is_unchanged`**, which pins exact (mc, kc, nc) per format (:1340-1359).
- `retargeting_threads_is_stable` (:1365).
- `method_parsing_roundtrips` (:1384).

**Other tests**
- `pack.rs` tests pin every format byte layout, conj, and the gather path (:190-292).
- `tests/correctness.rs`:
  - randomised over f32/f64/C32/C64 × `ComplexMethod::ALL` (:423-438, 316-319);
  - `TINY` blocking that drives every loop level (:302);
  - large cross-block GEMMs (:548-564);
  - analytical-vs-oracle (:415);
  - `threaded_matches_serial_*` bitwise (:1032-1164);
  - partition rule (:1198).
- `tests/spmd_seam.rs` (:45, 97) and `tcontract-exec/tests/tensorcontract_seam.rs`.

**Benchmarks and scripts**
- `examples/kernel_shapes.rs` imports `kernel::x86::{avx2_f32, avx2_f64, avx512_f32, avx512_f64}` directly (:56) and times kernels hot. It is gated by `scripts/kernel-shapes-compare.py`.
- `examples/blocking_model.rs`.
- `tensorprimitives-bench/src/engines/{shapes,orient}.rs` use the row-block menu and `TENSORCONTRACT_ROWBLOCK`.

---

## B. faer / gemm dispatch architecture (structure only, no code reproduced)

- **faer 0.24.4 `matmul_imp`** (REG faer-0.24.4/src/linalg/matmul/mod.rs:1176-1480):
  - Dtype routing is compile-time, through `if const { T::IS_NATIVE_F64 }`-style branches into a per-dtype macro arm (:1455-1480).
  - Inside, the backend is chosen **per call**:
    1. tiny M·N·K goes to `nano_gemm::planless` (:1331);
    2. then `is_x86_feature_detected!` is checked **every call** (:1358-1367). std caches the CPUID bits, so the check is cheap. If AVX-512 or AVX2+FMA is present, `private_gemm_x86` gets an `InstrSet` value;
    3. otherwise `gemm::gemm` (:1416).
  - Threading: a `Par` value is passed **per call** and converted to each backend's own parallelism type (`gemm::Parallelism`, :1441-1443). The thread count is gated by a size threshold (`par.degree()`, :1407).
- **private-gemm-x86 0.1.20:**
  - The **ISA and dtype are plain runtime enum arguments** (`InstrSet`, `DType`, lib.rs:1122-1135).
  - One `match (instr, dtype)` per call picks a *slice* (table) of assembly micro-kernel entry points, a matching pack-routine table, and `(mr, nr)` (lib.rs:1252-1269).
  - Kernels receive their parameters as a `#[repr(C)]` parameter block. It holds a flags word (complex, row-major, conj bits), depth, operand strides, an alpha pointer, destination pointer and strides, and optional index/diagonal pointers (lib.rs:33-64). This is one pointer argument instead of a long register signature. The kernel writes D directly.
  - The edge-size variants are entries in the table. kc is fixed per call.
  - Cache info comes from a `OnceLock` (cache.rs:356). The pack arena is thread-local and persistent (lib.rs:1286).
- **gemm-common 0.19 (reached through `gemm-f64`/`-c64`):**
  - A micro-kernel is a plain `unsafe fn` type (`MicroKernelFn<T>`, microkernel.rs:1). It takes the tile extents and depth, dst/lhs/rhs pointers with strides, alpha/beta, and flags for read-dst and conj. So **alpha/beta/conj and the store to dst are inside the kernel**.
  - A per-ISA 2-D table `[[MicroKernelFn; NR]; MR_DIV_N]`, indexed by (row-vector count − 1, cols − 1), serves edge tiles (gemm.rs:214, 726). A separate horizontal-kernel table covers thin cases (:215, 282).
  - **The whole driver is monomorphised per ISA.** `gemm_basic_generic<S: Simd, const N, MR, NR, ...>` (gemm.rs:174) is instantiated by `__inject_mod!` once per ISA module (scalar, fma=V3, avx512f=V4 behind the `x86-v4` feature, neon, simd128) (gemm.rs:851-920, 1077-1090).
  - The top-level `gemm_basic` for each ISA is chosen once and cached in an **`AtomicPtr` fn-pointer slot**, lazily initialised with Relaxed ordering (gemm.rs:1030-1075).
  - Blocking comes from `kernel_params(m,n,k,mr,nr,sizeof)` **per call** (cache.rs:514), over a cache-info once-cell (cache.rs:468-497).
  - Thresholds (threading, lhs/rhs packing) are global `AtomicUsize`s with setters (gemm.rs:120-160).
  - Parallelism is an enum argument per call (lib.rs:39-43).
- **nano-gemm 0.2.2:**
  - The closest analogue to a "descriptor". `Plan<T>` is a `Copy` value holding a 2×2 table of micro-kernels (full/edge in m and in n), a millikernel fn pointer, mr/nr, masks and m/n/k (lib.rs:28-60).
  - It is built by `Plan::new_<dtype>(m,n,k,...)`, which runs feature detection at construction (lib.rs:850-1050), and is executed with `execute_unchecked` (:781). faer uses the planless entry, so it rebuilds this each call.

**Contrasts that matter for the design**
- **Where the fn-pointer boundary sits:**
  - gemm-common puts it above the whole driver, so packing, loops and write-back all run with the ISA's target features.
  - PG and nano-gemm put it at the micro-kernel, but their kernels own alpha/beta/store/edges.
  - tensorcontract puts it at the micro-kernel with a scratch tile and leaves packing and write-back at baseline ISA.
- **How often the choice is made:** only gemm-common caches the ISA choice itself. faer and PG re-decide cheaply per call. All three cache cache sizes once.

---

## C. Proposal sketch: threading a `KernelDescriptor` through tensorcontract

### Central finding

About 70% of the descriptor already exists.
- `KernelConfig<R> = { Ukr<R>, Blocking }` is `Copy`. It carries mr, nr, a_per_k, b_per_k, tile, a_pack, b_pack, tile_fmt, `func`, name and mc/kc/nc (mod.rs:291-329, 459-465).
- `IsaConfigs<R>` is already a per-ISA fn-pointer table (x86.rs:517-524, aarch64.rs:316-323).

What is missing:
1. `pack_a`, `pack_b` and `writeback` are generic functions with runtime format matches, not pointers.
2. There is no isa/variant or method tag on the descriptor. The ISA is erased once `func` is extracted.
3. There is no validation token.
4. There is no interleaved/native or 4m variant. [inf] `Planar` is register-level 4m with split packing.
5. It is not resolved once. It is rebuilt each execute.

### Proposed seams, in order of disruption

1. **Descriptor type.** Extend or wrap `KernelConfig<R>` into `KernelDescriptor<R>`:
   - `{ dtype tag, method, a_pack, b_pack, tile_fmt, isa: Variant, mr, nr, a_per_k, b_per_k, tile, blk, ukr, pack: unsafe fn(...) per operand, writeback: unsafe fn(...), name }`.
   - Private fields, built only by a validating constructor. The constructor checks that the ISA features are present, that (mr, nr) is on the compiled menu, that the formats agree with the kernel, and that mc%mr == 0 and nc%nr == 0 (today's `with_blocking` invariant, mod.rs:545-550). Constructing it is the unsafe proof point that today lives implicitly in `selected_isa()`.
2. **Selection.** Replace the body of `config_for_plan` (mod.rs:893-904). Its only caller is driver.rs:357.
   - The global default comes from a process `OnceLock<EngineTable>`, per (dtype, method, variant), built from `IsaConfigs`.
   - Per-plan overrides become one new `Plan` field `kernel: Option<KernelChoice>` (variant, method, tile index, blocking), next to plan.rs:298-302.
   - **Design fork:** `Plan` is dtype-agnostic, so a typed descriptor cannot live in it directly. The options are:
     - (a) store a dtype-erased choice and resolve a typed descriptor per execute from the `OnceLock` table (cheap: an index lookup);
     - (b) a typed `PlanFor<T>` wrapper that caches `KernelDescriptor<T::Real>`;
     - (c) a 4-slot per-dtype cache inside `Plan`.
     - (a) is least disruptive to tcontract (`TbPlan` holds a plain `Plan`, tcontract/src/tblis.rs:145-148).
3. **Driver.** Put the descriptor in `Ctx` in place of `ukr` (driver.rs:160-201). Replace the two `pack_panel::<T>` calls (driver.rs:715-724 B, 743-752 A) with `(d.pack_b)`/`(d.pack_a)`. These run per sliver group per (jc, pc, ic), so the indirection is negligible [inf].
   - The orientation swap (driver.rs:360-386) must swap `pack_a`/`pack_b` with the operands. It already swaps formats implicitly for 1m.
4. **Write-back.** Make `writeback` a descriptor pointer specialised per `TileFormat`, with the runtime `match fmt` removed (writeback.rs:67-93). Keep the three row-addressing arms inside it. This adds one indirect call per micro-tile on top of `func`'s existing one (simd.rs:243-249) [inf: low single-digit ns against ≥kc·MR·NR FMAs].
   - This is also the hook for a later "kernel writes D directly" variant in the gemm-common/PG style, which would need the write-back contract to become optional on the descriptor.
5. **ISA-specialised pack/writeback.** With both behind pointers, `#[target_feature]` variants (trampolined like simd.rs:243-300) can live in `strided-rs`, per the workspace SIMD-placement rule. The baseline versions stay as the portable/generic variant.
6. **Blocking.** Resolve kc/mc/nc once into the descriptor through the existing `normalise_for`, applied exactly once (mod.rs:488-524). Keep per-call clamps (driver.rs:448-450) and `plan.blocking` (driver.rs:358-361) in the driver. Cache `TENSORCONTRACT_KC_COUPLE` like the other variables.

### Risks

- **MR×NR is menu-bound.** Kernels are const-generic monomorphs enumerated by `configs!`, so a runtime tile outside the menu needs new instantiations or a slow generic kernel. The scalar path is also const-matched (scalar.rs:359-381, and it is fixed at 4×4 via mod.rs:833-844). Validation must reject off-menu shapes.
- **Interleaved/native/4m is a deliberate breaking change.** `ComplexMethod`, `PackFormat` and `TileFormat` are not `#[non_exhaustive]` on purpose (mod.rs:157-166). New arms are needed in `emit`, `tile_value`, `Blocking::derive` callers, `configs!`, and the test helpers `pack_for`/`tile_value` (mod.rs:974-1027). `KernelSet` is a public trait bound used by tprims-blas (`Scalar::Re: KernelSet`, tprims-blas/src/scalar.rs:34), so changing its shape ripples into tcontract.
- **A test must be reframed.** `x86_isas_agree_on_the_pack_contract` (mod.rs:1191-1254) asserts identical formats across ISAs. A native-interleaved AVX2 variant next to a planar AVX-512 one violates it by design. It should become "formats agree *per variant family*".
- **Double-application hazard.** `_MC_PCT`/`_NC_PCT` must not be applied twice (mod.rs:526-540). A cached descriptor must be post-override and never re-normalised under Legacy.
- **Spmd interaction.** The descriptor must be resolved before `broadcast`, as it is today at driver.rs:357. It must be `Copy + Send + Sync`; fn pointers are.
  - Barrier-count equality depends on all threads sharing kc/nc (driver.rs:615-623). Resolve per call, never per thread.
  - **Existing mismatch [inf, from reading]:** `retarget_threads(plan.threads())` (mod.rs:903) uses the plan's thread count, which is 1 in tcontract. The real width is `spmd.width()`, computed *afterwards* (driver.rs:461-466). Under `TENSORCONTRACT_BLOCKMODEL=model`, nc is therefore derived for the wrong thread count on every tcontract call. The descriptor resolution should take the effective width.
- **Monomorphisation vs pointers.** Moving to a gemm-common style whole-driver-per-ISA would multiply 4 dtypes × ISAs × formats of `run_strip`. The per-seam pointer approach keeps the one generic driver. The trade-off is that loop control stays at baseline ISA [inf: minor].
- **Unsafe contracts to carry over:**
  - the panel sizes `a_per_k·kc` and `tile`;
  - write-only packing that zero-fills every lane (pack.rs:113-123);
  - a tile fully overwritten before it is read (writeback.rs:63-65);
  - the rule that `tile_fmt` must match the kernel (a mismatch reads out of bounds, writeback.rs:55-61).

  With pointers, these become descriptor invariants checked at construction.
- **Per-call costs a cached descriptor alone does not remove:** block-scatter `Vec`s (driver.rs:427-437), `Panel` allocations (driver.rs:503, 543-544, 575-576) and the `Barrier` `Vec` (driver.rs:554). A scratch arena should be carried alongside the descriptor. This is out of scope here, but the same seam.
- **Churn:** `examples/kernel_shapes.rs` and the bench engines use the `doc(hidden)` `x86::*` modules and the menus directly. Renames break them.

---

## D. Parallel partitioning boundaries

These sections were added as a follow-up. They describe structure only; no faer/gemm/PG code is reproduced.
`PG` = private-gemm-x86 0.1.20, `GC` = gemm-common 0.19, `FA` = faer 0.24.4 (all under `REG`).

### D1. tensorcontract (as used by tcontract)

**Which loops are split.** Only the **output M and N axes**. K (loop 4) is serial by design, so there is no reduction and results are bitwise identical to serial (driver.rs:83-92).
- **M: row strips, cut once, outside every loop.** Thread `t` has row index `r = t / pn` and column group `g = t % pn` (driver.rs:565). Its strip is `[(r·npanels/pm)·MR, ((r+1)·npanels/pm)·MR)` (driver.rs:573-574). It runs loops 3, 2 and 1 over that strip with its own packed-A block.
- **N: column groups, re-cut inside loop 5 for every `jc` block.** A group owns a contiguous range of NR slivers of that block, and also a sub-share of packing them (`BPart::ranges`, driver.rs:250-257).
  - Every thread walks the same `(h, jc, pc)` sequence (driver.rs:615-623, 673-687).
  - The Hadamard/batch loop `h` is **inside** each thread (driver.rs:673), so **the batch axis is never a parallel axis within one plan**.
  - `partition_with` sees only `panels = ceil(rows/MR)` and `blocks = ceil(cols/NR)` (plan.rs:669-679).
  - So a contraction with a large batch but a small m×n is capped at `panels·blocks` threads. [inf] That is a real limit for batched-Hadamard contractions going through tcontract `tblis.rs`. tprims-blas avoids it only by looping items outside (tprims-blas/src/tblis.rs:91-109).
- **How (pm, pn) is chosen** (plan.rs:669-720):
  - When `panels ≥ p`, the split is pure 1-D M, `(p, 1)`.
  - It becomes `(1, p)` only when `columns_beat_rows` holds: more than one L3 domain, `blocks ≥ p`, and `k ≤ 64` (plan.rs:1095-1097).
  - Otherwise a small cost model over pm (the `PACK_WEIGHT = 8` term prices the duplicated A packing) picks `(pm, pn)`, and ties go to the largest pm (plan.rs:708-720).
  - The partition is computed in the orientation after the swap (plan.rs:674-678).

**Boundary alignment.**
- Strips are whole MR panels and groups are whole NR slivers. Only the final panel of M and the tail sliver of each `jc` block are ragged.
- MC and NC are multiples of MR and NR (mod.rs:545-547). mc is clamped to `m.next_multiple_of(mr)` but **not to the strip**, so a strip shorter than mc just runs one `ic` pass (driver.rs:449, 740-741).
- Nothing aligns boundaries to cache lines in D.
  - [inf] For the default AVX2 shapes the byte span of one MR run is 64 B: f64 MR 8, c64 planar MR 4, f32 MR 16. So strip boundaries fall on line boundaries only if D's column base happens to be 64-B aligned.
  - Row-block alternates break even that, e.g. f64 (3,4) gives MR 12 = 96 B (x86.rs:356). So do 3m f64 MR 4 = 32 B (x86.rs:359), strided or scattered D, and the orientation swap.
  - So **the cache lines at the p−1 strip boundaries are write-shared in every column of D, on every kc pass** (writeback RMW, driver.rs:794-816). [inf] The cost is small because it is only at strip edges.
- The packed-B slices, by contrast, are deliberately one contiguous slice per group, sized at the worst-case capacity `group_cap`, so groups do not share lines (driver.rs:487-502).

**Static vs dynamic, and ragged edges.**
- Fully **static**. There is no work stealing.
- Imbalance is at most one MR panel per strip, from floor division, plus the partial last panel.
- Column groups can be **empty in a tail `jc` block**. The thread still takes both barriers and skips its compute (driver.rs:733-741). That idles `pn − nsliv` threads for that block (driver.rs:103-107).
- A thread whose strip is all zero-padded panels never happens, because `pm ≤ panels`.

**Sync points.**
- Inside `run_strip`: **two `std::sync::Barrier` waits per (h, jc, pc)**, one per column group of `pm` threads. `pm = 1` means no barrier (driver.rs:554-555, 709-728). `std::sync::Barrier` is Mutex+Condvar, so it is a futex sleep and wake with no spin phase [inf from std's implementation].
- Outside: one fork/join per `execute`, i.e. one `Spmd::broadcast` (driver.rs:586-596).
- In tcontract, `broadcast` is `Exec::broadcast`:
  - It takes a per-pool SPMD mutex (exec.rs:233-237).
  - It calls rayon `ThreadPool::broadcast` on the **whole pool**; workers with `t ≥ width` return immediately (exec.rs:239-244).
  - The calling thread must not be a pool worker, or the call is refused and tensorcontract **reruns serially** (exec.rs:224-225, driver.rs:590-595).
  - The caller blocks and does not compute.

**How the Spmd width maps onto the grid.**
1. `want = spmd.width()` replaces `plan.threads()` (driver.rs:461-466).
2. `(pm, pn) = partition_with(mr, nr, want)`, shrunk to fit if a pinned partition exceeds it (driver.rs:467-478). So `p = pm·pn ≤ width`, and it may be less when `panels·blocks < width`.
3. Rayon worker index `t` becomes cell `(t / pn, t % pn)` directly (driver.rs:565, exec.rs:240), so the cell-to-worker mapping is **stable across calls** for the same shape.
4. tcontract chooses the width with `width_for`, a cost model of 5 µs base plus 5 µs per thread plus `S/k` (exec.rs:259-275). tcontract uses `ns_per_flop` (tcontract/src/tblis.rs:226-231). tprims-blas batches choose `outer` (items spread, each item width 1) or `inner` (items serial, each an SPMD broadcast) (tprims-blas/src/batched.rs:131-148, tblis.rs:91-109).
5. As noted in C, blocking under the analytical model is retargeted with `plan.threads()`, not with this width (mod.rs:903).

### D2. faer → private-gemm-x86 (the path taken on AVX2/AVX-512 machines)

**Which loops are split.**
- The kc slab loop is serial and outermost (PG lib.rs:1455-1503 region, beta becomes "add" after the first slab).
- Within a slab there are two parallel phases. Both are `spindle::for_each_raw` over the whole team.
  - (i) **Optional B pack, split over the depth (k) axis**: thread j packs rows `[start, end)` of the slab for every NR column block (PG lib.rs:826-866). It runs only when B is packed at all, which is rare (see the prior worklog §4).
  - (ii) **Compute over a 2-D grid of jobs.** Each job is `(mf·mr) × (nf·nr)`. `mf ∈ [2, 4]` is scaled by threads (16/f for "tall", 2 for "wide") and `nf = 32/nr`, i.e. 32 columns (PG lib.rs:808-822, where the first `nf` is overwritten).
- Job id to `(row block, col block)` is row-fastest: `thd_id % n_row_jobs`, `thd_id / n_row_jobs` (PG lib.rs:451-455).
- Within a job, the micro-tiles are walked column-major unless the matrix is "tall" (PG lib.rs:465-470).

**Boundaries.** Multiples of `mf·mr` rows and 32 columns. Like tensorcontract, there is no cache-line alignment of D boundaries.
- [inf] With mr = 12 (f64 AVX2) a job is 24-48 rows (192-384 B), and row-fastest ordering puts vertically adjacent jobs on different threads **at the same time**.
- So line sharing at job boundaries is both more frequent than tensorcontract's p−1 fixed boundaries and concurrent. Each boundary is written once per slab (a register epilogue, no RMW pass per tile).

**Dynamic distribution.**
- Jobs are claimed with a relaxed `fetch_add` on a shared counter until exhausted (PG lib.rs:869-875 onward). So load balance and ragged edges are handled by granularity; the last jobs are simply smaller.
- Packed-A sharing is by per-row-panel status flags with Release/Acquire. The check-then-set is racy, so two threads may pack the same panel with identical data, costing only duplicated work (see the prior worklog §5, PG lib.rs:490-555).

**Sync points.**
- There is no barrier inside a slab.
- There are one or two fork/joins per slab through spindle, whose workers **spin** (up to about 64K pause iterations) before falling back to a futex, for the whole `gemm` call (`spindle::with_lock` at PG lib.rs:1507-1515; spindle details in the prior worklog §5).

**Width mapping.**
- faer `Par::rayon(n)`, where n comes from tcontract `exec.install` (tprims-blas/src/gemm.rs:34-37, batched.rs:234). The team size is `min(n, rayon::current_num_threads())`.
- faer drops to 1 thread below `M·N·K < 4096·sizeof` (FA matmul/mod.rs:1403-1411).
- Thread index to job is *not* stable across calls, because of the dynamic claiming.

### D3. gemm-common (fallback: no AVX2+FMA)

- **B pack:** statically split over "tasks" (NR column chunks), with the remainder spread over the first threads (GC gemm.rs:560-616). It runs through `par_for_each`, a rayon `into_par_iter` fork/join (GC gemm.rs:164-171), per (nc, kc) block.
- **Compute:** a grid of mini-jobs split into **static contiguous ranges** per thread, `floor` plus a remainder (GC gemm.rs:640-720). Each thread packs its **own** A into its thread-local L2 slab (GC gemm.rs:808-818).
- The thread count per kc slab is gated by the global `THREADING_THRESHOLD` (GC gemm.rs:511-522).
- Sync is rayon fork/join per phase; there are no barriers inside a phase.

### D4. Side-by-side

| | tensorcontract | faer/PG | gemm-common |
|---|---|---|---|
| parallel axes | M strips (outer) × N groups (per jc) | 2-D m×n jobs per kc slab | m×n mini-jobs per (nc, kc) block |
| K | serial | serial (slab) | serial |
| batch axis | serial inside each thread | n/a (faer called per item) | n/a |
| granularity | one strip per thread (long) | (2-4)·mr × 32 cols | mini-jobs |
| distribution | static | dynamic `fetch_add` | static contiguous |
| B pack | cooperative across the pm threads of a group, over N slivers | over k, all threads (rare) | over NR chunks, all threads |
| A pack | private per thread, duplicated pn times | shared per row panel, first-come (racy dup) | private per thread |
| barriers per kc | 2 × futex Barrier (pm > 1) | 0 inside the slab; 1-2 spin fork/joins | 1-2 rayon fork/joins |
| C boundary false sharing | p−1 fixed strip edges, every kc RMW [inf] | many concurrent job edges, once per slab [inf] | static edges |
| result reproducible across p | bitwise yes | yes per element [inf: K order fixed per slab] | yes [inf] |

---

## E. Buffer memory locality and NUMA first-touch

### E1. tensorcontract

All buffers are `Panel<T>`: `std::alloc` with **64-B alignment**. They are uninitialised, so the first write is the first touch, and they are freed on drop. That means **per call, no reuse, no page alignment and no huge pages** (buffer.rs:45-85).

| buffer | allocated by / when | first touched by | consumed by |
|---|---|---|---|
| packed B `bp`, `pn·b_group` reals (~3 MiB f64 at full nc) | the **calling thread**, in `execute_capped`, each call (driver.rs:500-503) | each pm thread of a group writes its own sliver share (driver.rs:712-725); page placement follows the packer [inf: Linux first-touch] | all pm threads of that group read all its slivers (driver.rs:760) |
| packed A `ap`, `ap_len` (512 KiB legacy) | **each worker, inside its cell closure**, each call (driver.rs:575) | the same worker (pack at driver.rs:743-752) | the same worker: good locality |
| tile | each worker, each call (driver.rs:576) | kernel | writeback, same worker |
| block-scatter `Vec`s `a_m_bs`, `b_n_bs`, `d_m_bs`, `c_m_bs` | calling thread, each call (driver.rs:427-437) | calling thread | read by all threads |
| `bars: Vec<Barrier>` | calling thread, each call (driver.rs:554) | — | all |
| the plan's scatter vectors (`a_m`, `b_k`, `d_m`, ...) | whoever built the `Plan` | plan builder | read by all threads per sliver and per tile |

**Concrete locality problems:**
1. **Page-fault and zeroing churn every call.** [inf: glibc behaviour]
   - The B panel is MiB-scale and the A blocks are 512 KiB. Both are above glibc's initial 128 KiB mmap threshold, so the first calls get fresh `mmap` pages that are demand-zeroed and faulted by the packers, then `munmap`ed on return.
   - glibc's dynamic threshold then moves such sizes onto arenas, but trimming can still return memory.
   - This repeats for **every batch item** in tprims-blas (`run_raw_with` per item, tprims-blas/src/tblis.rs:71-108) and for every call in tcontract.
   - This is the prior worklog's cause #3. faer's PG instead keeps a persistent page-aligned thread-local arena (below).
2. **The B panel is allocated on the caller's malloc arena and freed there, while it is touched only by workers.**
   - In tcontract the caller is never a worker (exec.rs:224-225). So the arena and its metadata live on the caller's side while the pages sit wherever the packers first-touched them.
   - On a multi-socket or multi-CCD box, a group's slivers are packed by ≈1/pm of the group per thread, so **one shared panel is physically striped across the threads' nodes/L3s** and then read by every thread of the group.
   - The domain-aware partition only moves to `1×p` (no shared B) for shallow k (plan.rs:1095-1097). For compute-bound shapes across domains, the shared B stays remote for most readers. [inf; not measured on NUMA]
3. **A is duplicated pn times, with fresh memory each call.**
   - Locality is right (packer = consumer), but capacity is wasted. On 2-D partitions every thread of a strip packs the same A into its own new 512 KiB.
   - The block is sized for the whole L2 (see the prior worklog §3), so on Zen 3 it competes with the B sliver and the C lines.
4. **No reuse despite a stable thread-to-cell mapping.** The worker index gives the same cell each call (driver.rs:565, exec.rs:240), so per-worker cached A buffers (and per-group B slices) would stay node- and L2-warm. Today that is thrown away every call.
5. **Serial-fallback double allocation.** If `broadcast` is refused, the outer frame's B panel stays alive while the serial rerun allocates another one plus A (driver.rs:586-595). That is transient ~2× memory.
6. **Scatter-vector reads are remote for every thread.** The plan's `i64` scatter vectors, and the block scatters built per call on the caller, are read by all workers through pack and writeback. They are small and L3-resident on one CCD; on NUMA they are remote-node reads. [inf]
7. **Alignment is only 64 B.** It is fine for the unaligned-load kernels (simd.rs:69-73). But slices are not page-aligned, so group slices share pages at their boundaries: first-touch splits a page between two groups' packers [inf, minor].
8. Batch "outer" mode (items spread, each serial) allocates B, A and the tile per item on each worker. Locality is good, but the churn is maximal (tprims-blas/src/tblis.rs:92-98).

### E2. faer / PG

- **One persistent thread-local arena per calling thread**, made of 4096-B aligned page units. It is sized from the (inflated) L3 for both packed A and packed B, reserved but uninitialised, so only touched pages are backed. It is reused across calls and grows by reserve. If re-entered (the `RefCell` is already borrowed), it falls back to a fresh Vec (PG lib.rs:1280-1318). Cache sizes come from a `OnceLock` (PG cache.rs:356).
- **Ownership vs touch.**
  - The arena belongs to the thread that called `gemm`. In tcontract that is a pool worker, because tprims-blas enters faer through `exec.install` (tprims-blas/src/batched.rs:234, gemm.rs:34-37).
  - Packed-A panels in it are **first-touched by whichever team thread first packs that row panel**, fused with compute. Other threads then read that panel (see the prior worklog §5).
  - So pages are placed by the first packer on the **first** call and then stay put across calls, even though the dynamic job claiming means a different thread may consume them next time. [inf: on NUMA this fixes placement to first-call history; on one CCD it is irrelevant]
- **B is normally not packed at all.** It is streamed from source, shared through L2/L3 by the row-fastest job order (see the prior worklog §4-5), so there is no shared B buffer to place.

### E3. gemm-common

- **Packed B plus a pre-packed A**: a `MemBuffer` allocated **per call** by the calling thread and shared (GC gemm.rs:420-437). It is packed cooperatively (GC gemm.rs:560-616), so first-touch is by the packers, and it is freed on return.
- **Per-thread packed A**: a **persistent thread-local "L2 slab"** sized to L2 and cache-line aligned, allocated once per worker and reused (GC gemm.rs:58-63, 808-818). The packer is the consumer, so it is node-local and warm across calls.
- With `no_std` it uses a per-call slab instead (GC gemm.rs:442-446).

### E4. Takeaways for the switchable engine (structural, [inf])

- tensorcontract already has the right *ownership* for A: the consumer packs it and it is per thread. What it lacks is **persistence**.
  - The cheapest fix is a per-worker cached scratch arena for A and the tile, keyed by the stable worker index t. gemm-common's L2 slab is the model.
  - Add a persistent shared B arena per pool, page-aligned and ideally one slice per column group, first-touched by that group's packers.
  - These scratch buffers should be carried next to the `KernelDescriptor`, and sized from its `a_per_k`, `b_per_k`, `tile`, `mc`, `kc` and `nc`.
- If buffers are cached, the fallback-to-serial and re-entrancy cases need their own fresh buffers. Both PG (the `RefCell` borrow fallback) and gemm-common handle this.
- For NUMA or multi-CCD placement, either:
  - (a) keep B per group and have its packers first-touch it (today's behaviour, but persistent); or
  - (b) replicate B per L3 domain, which is what `columns_beat_rows` approximates by changing the partition instead.
- If the batch axis is ever to be parallelised inside a plan, the per-thread buffers must be per worker, not per item.
