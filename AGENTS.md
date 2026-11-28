# AGENTS.md — tcontract-rs

## カーネルはビルド時の CPU で変わってはならない

tensor4all 共通の規則（`~/projects/tensor4all/AGENTS.md`）。SIMD カーネルは **runtime dispatch** で選ぶ：

* **ISA は `is_x86_feature_detected!`（または pulp の `dispatch`）で与え、実行時に選ぶ**。`#[target_feature]` は
  **加算的**で，書かなかった機能を除去しないため，クレート全体を `-C target-cpu=<型番>` でビルドすると
  「AVX2 カーネル」が AVX-512 命令を含みうる（`t4a-private-gemm-x86#26` で実際に起きた）。
* **配布ビルドで `-C target-cpu=<型番>` / `native` を使わない**。型番ビルドは `experiments/` 配下の実験専用で，
  ページ・リリース・CI の対象にしない。
* **カーネルの中では命令を名指しする**（intrinsics か固定幅 `std::simd`）。配列への `mul_add` ループに
  「レーンは独立」と書いて**ベクトル幅をコンパイラに選ばせる**のはこの規則に反する。
* **`cfg!(target_feature = …)` は no_std のフォールバックに限る**。現在
  `crates/tcontract-kernel/src/kernels/x86/mod.rs` にその形が1箇所あり，
  「std が無ければランタイム検出ができない」と明記されている。**ただし拒否はしていない**ので，
  std 無しでベースライン超の機能を有効にしたビルドは，ビルド機でしか正しく動かない。
  colmat-rs では `compile_error!` ＋ 明示 opt-in フィーチャで拒否する形にした（`colmat-rs#321`）。
* **性質を検査する**: AVX2 ティアのシンボルを逆アセンブルし，`zmm` / EVEX が出たら失敗させる。
  strided-rs は `scripts/check-runtime-simd-dispatch.py`，colmat-rs は `scripts/check-isa-gate.sh` を持つ。
  このリポジトリにはまだ無い。
