#!/usr/bin/env bash
# One W2b session: $1 = session dir name. b0 and b1 on both corpora.
d=$(cd "$(dirname "$0")/.." && pwd); bin=/tmp/claude-2000/-home-shinaoka-tensor4all/662178ee-cfce-467b-b6ba-c7a683ca4e5b/scratchpad/contract-w2b
cpus=${CPUS:-0,1,2,3,4,5,6,7}
for m in separate_b1 separate_b0; do
  for c in tenferro-p1-gemm large-batched-gemm; do
    "$d/scripts/drive.sh" "$d/$1" "$bin" "$cpus" $c $m
  done
done
touch "$d/$1/ALL-done.flag"
