#!/usr/bin/env bash
# one full before/after pass over the focused case lists
#   session.sh OUT BEFORE_BIN AFTER_BIN [only CORPUS...]
# large-batched-gemm omits the 2048^3 and 4096^3 entries (H < 4: the team path,
# unchanged by this change, and minutes per case at 1T).
d=$(cd "$(dirname "$0")" && pwd); out=$1; bb=$2; ab=$3; shift 3
only=" $* "
root=$(cd "$d/../../../../../../.." && pwd)
lst() { "$ab" --list --corpus "$root/benchmarks/benchmarks/tprims/corpus/$1.json"; }
want() { [[ $only == "  " || $only == *" $1 "* ]]; }
want tenferro-p1 && "$d/drive.sh" "$out" tenferro-p1 "$bb" "$ab" dot_general_040_f64 dot_general_041_f64 dot_general_044_f64 dot_general_045_f64 dot_general_049_f64 dot_general_008_f64 dot_general_016_c64
want tenferro-p1-gemm && "$d/drive.sh" "$out" tenferro-p1-gemm "$bb" "$ab" gemm_batched_047_f64 gemm_batched_008_f64 gemm_batched_012_f64 gemm_batched_022_f64 gemm_batched_023_f64 gemm_batched_000_f64
want phase2-extra && "$d/drive.sh" "$out" phase2-extra "$bb" "$ab" $(lst phase2-extra)
want large-batched-gemm && "$d/drive.sh" "$out" large-batched-gemm "$bb" "$ab" $(lst large-batched-gemm | grep -v -E '_b2_m2048|_b1_m4096')
want phase2-profile && "$d/drive.sh" "$out" phase2-profile "$bb" "$ab" gemm_1024_f64 gemm_1024_c64
echo done > "$out/ALL-done.flag"
