#!/usr/bin/env bash
# Phase 2 S0 paired driver. Per case: pick 8 idle CPUs of one L3 domain, run
# 1T, 4T, 8T back to back on prefixes of that set (each through pinned.sh);
# a case whose triple is not fully valid is discarded and retried on a new pick.
#   drive.sh SESSION_DIR CORPUS_NAME   (corpus file: benchmarks/benchmarks/tprims/corpus/NAME.json)
set -uo pipefail
out=$1; name=$2
root=$(cd "$(dirname "$0")/../../../../../../.." && pwd)
bin=$root/target/release/contract
corpus=$root/benchmarks/benchmarks/tprims/corpus/$name.json
mkdir -p "$out"
for t in 1 4 8; do
  echo "case,variant,threads,median_ns,samples" > "$out/$name-${t}t.csv"; : > "$out/$name-${t}t.log"
done
echo "case,cpus,attempt" > "$out/$name-cpus.csv"
export BENCH_RUNS=${BENCH_RUNS:-5}
mapfile -t cases < <("$bin" --list --corpus "$corpus")
for c in "${cases[@]}"; do
  ok=0
  for attempt in $(seq 1 60); do
    if ! cpus=$(python3 "$root/benchmarks/scripts/idle_cpus.py" pick 8 --seconds 2 2>/dev/null); then sleep 20; continue; fi
    IFS=, read -ra all <<< "$cpus"
    tmpd=$(mktemp -d); good=1
    for t in 1 4 8; do
      set_cpus=$(IFS=,; echo "${all[*]:0:$t}")
      if ! PINNED_RETRIES=1 BENCH_CASE=$c "$root/benchmarks/scripts/pinned.sh" "$set_cpus" -- "$bin" --threads "$t" --corpus "$corpus" > "$tmpd/$t" 2>>"$out/pinned.err"; then good=0; break; fi
    done
    if (( good )); then
      for t in 1 4 8; do
        grep -v -E '^(#|case,|CHECK)' "$tmpd/$t" >> "$out/$name-${t}t.csv"
        grep -E '^(#|CHECK)' "$tmpd/$t" | sed "s/^/[$c] /" >> "$out/$name-${t}t.log"
      done
      echo "$c,\"$cpus\",$attempt" >> "$out/$name-cpus.csv"; ok=1; rm -rf "$tmpd"; break
    fi
    rm -rf "$tmpd"
  done
  (( ok )) || echo "SKIPPED $c" >> "$out/$name-skipped.txt"
done
echo done > "$out/$name-done.flag"
