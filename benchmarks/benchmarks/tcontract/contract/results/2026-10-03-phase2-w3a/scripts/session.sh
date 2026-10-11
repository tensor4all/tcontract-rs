#!/usr/bin/env bash
# one pass over the W3 case list: before (c71bba4+#54 base b3bd887), after (W3a),
# and two W3b blocking arms on the W3a code (kc512 + 256 KiB A budget; analytical model).
#   session.sh OUT BIN_BEFORE BIN_AFTER   (BIN_AFTER is the experiment binary: TPRIMS_W3B selects the arm)
d=$(cd "$(dirname "$0")" && pwd); out=$1; bb=$2; ab=$3
V="before:-:$bb after:-:$ab k512:TPRIMS_W3B=k512:$ab model:TPRIMS_W3B=model:$ab"
"$d/drive.sh" "$out" phase2-profile "$V" gemm_1024_f64 gemm_1024_c64
"$d/drive.sh" "$out" tenferro-p1 "$V" dot_general_000_f64 dot_general_001_f64 dot_general_005_f64 dot_general_007_f64 dot_general_018_f64 dot_general_016_c64 dot_general_031_c64 dot_general_012_f64 dot_general_043_f64 dot_general_036_f64
"$d/drive.sh" "$out" tenferro-p1-gemm "$V" gemm_batched_008_f64 gemm_batched_012_f64 gemm_batched_022_f64 gemm_batched_023_f64 gemm_batched_009_c64
"$d/drive.sh" "$out" phase2-extra "$V" small_n4_k32_h8192_f64 small_n4_k4_h65536_c64 rank1_k1_m2048_n2048_f64
echo done > "$out/ALL-done.flag"
