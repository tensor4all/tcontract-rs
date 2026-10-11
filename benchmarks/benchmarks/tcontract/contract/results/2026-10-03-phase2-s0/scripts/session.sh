#!/usr/bin/env bash
# one full session over the four gate corpora, sequentially
d=$(dirname "$0")
s=$1
for n in tenferro-p1 tenferro-p1-gemm large-batched-gemm phase2-extra; do "$d/drive.sh" "$d/../$s" $n; done
echo done > "$d/../$s/ALL-done.flag"
