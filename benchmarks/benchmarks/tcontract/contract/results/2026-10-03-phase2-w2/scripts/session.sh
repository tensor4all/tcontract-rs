#!/usr/bin/env bash
# One session of the W2 measurement: $1 = session dir name.
d=$(cd "$(dirname "$0")/.." && pwd); bin=/tmp/claude-2000/-home-shinaoka-tensor4all/662178ee-cfce-467b-b6ba-c7a683ca4e5b/scratchpad/contract-w2
cpus=${CPUS:-24,25,26,27,28,29,30,31}
for m in separate_b1 separate_b0 separate_same; do
  "$d/scripts/drive.sh" "$d/$1" "$bin" "$cpus" tenferro-p1-gemm $m
done
for m in separate_b1 separate_same; do
  "$d/scripts/drive.sh" "$d/$1" "$bin" "$cpus" large-batched-gemm $m
done
touch "$d/$1/ALL-done.flag"
