#!/usr/bin/env bash
# W2b driver (waits only for other worktrees' pinned.sh / contract runs, never for their drive.sh): one process per (mode, corpus case), widths paired back to back.
# Before each run, wait while any other pinned.sh / drive.sh / contract
# --corpus process of another worktree is active (W1 shares this host).
#   drive.sh OUTDIR BIN CPUS CORPUSNAME MODE   (widths 1 4 8)
set -uo pipefail
out=$1; bin=$2; cpus=$3; corp=$4; mode=$5
root=$(cd "$(dirname "$0")/../../../../../../.." && pwd)
file=$root/benchmarks/benchmarks/tprims/corpus/$corp.json
here=$root/benchmarks/scripts
IFS=, read -ra all <<< "$cpus"
mkdir -p "$out"
tag=$corp-$mode
for t in 1 4 8; do
  echo "case,variant,threads,median_ns,samples" > "$out/$tag-${t}t.csv"; : > "$out/$tag-${t}t.log"
done
echo "bin=$bin cpus=$cpus mode=$mode corpus=$corp sha=$(sha256sum "$file" | cut -c1-16) commit=$(git -C "$root" rev-parse --short HEAD) $(date -u +%FT%TZ)" >> "$out/manifest-$tag.txt"
mapfile -t cases < <("$bin" --list --corpus "$file")
for c in "${cases[@]}"; do
  for t in 1 4 8; do
    while pgrep -af 'pinned.sh|contract --' | grep -v -e tprims-rs-p2w2c -e contract-w2c -e pgrep -e 'bash -c' | grep -q .; do sleep 20; done
    set_cpus=$(IFS=,; echo "${all[*]:0:$t}")
    tmp=$(mktemp)
    for attempt in $(seq 1 200); do
      BENCH_C_MODE=$mode BENCH_RUNS=${BENCH_RUNS:-5} BENCH_CASE=$c "$here/pinned.sh" "$set_cpus" -- "$bin" --threads "$t" --corpus "$file" > "$tmp" 2>>"$out/pinned.err"
      grep -q packed_exec "$tmp" && break
      echo "drive.sh: $c ${t}T no valid run (attempt $attempt)" >> "$out/pinned.err"; sleep 60
    done
    grep -v -E '^(#|case,|CHECK)' "$tmp" >> "$out/$tag-${t}t.csv" || true
    grep -E '^(#|CHECK)' "$tmp" | sed "s/^/[$c] /" >> "$out/$tag-${t}t.log" || true
    rm -f "$tmp"
  done
done
touch "$out/$tag-done.flag"
