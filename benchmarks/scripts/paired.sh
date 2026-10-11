#!/usr/bin/env bash
# One process per case, every thread count of a case back to back (so slow
# drift of the host cannot bias one side), each run through pinned.sh.
#
#   paired.sh BIN OUTDIR CPUS THREADS...     e.g. paired.sh target/release/contract out 8-15 1 4 8
#
# CPUS must hold at least max(THREADS) CPUs of one L3 domain (from
# `idle_cpus.py pick`); a count T uses the first T of them. BIN must support
# `--list` and `BENCH_CASE`. Environment: CORPUS (a corpus file passed as
# `--corpus`), BENCH_RUNS / BENCH_WARMUP (passed through). Writes
# OUTDIR/<bin>-<T>t.csv, OUTDIR/<bin>-<T>t.log (`#` and CHECK lines) and
# OUTDIR/manifest.txt.
set -euo pipefail
bin=$1; out=$2; cpus=$3; shift 3
threads=("$@")
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
name=$(basename "$bin")
mapfile -t all < <(python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); import idle_cpus as m; print("\n".join(map(str, m.parse_cpu_list(sys.argv[2]))))' "$here" "$cpus")
cargs=(); [[ -n ${CORPUS:-} ]] && cargs=(--corpus "$CORPUS")
mkdir -p "$out"
{
    echo "tool: benchmarks/scripts/paired.sh"
    echo "date: $(date -u +%FT%TZ)"
    echo "tcontract-rs: $(git -C "$root" rev-parse HEAD)$(git -C "$root" diff --quiet || echo ' (dirty)')"
    echo "binary: $bin"
    echo "cpu: $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | xargs || uname -m)"
    echo "cpus: $cpus"
    echo "threads: ${threads[*]}"
    echo "corpus: ${CORPUS:-built-in}$([[ -n ${CORPUS:-} ]] && echo " sha256=$(sha256sum "$CORPUS" | cut -d' ' -f1)")"
    echo "BENCH_RUNS=${BENCH_RUNS:-default} BENCH_WARMUP=${BENCH_WARMUP:-default}"
} > "$out/manifest.txt"
for t in "${threads[@]}"; do
    if (( t > ${#all[@]} )); then echo "paired.sh: $t threads but only ${#all[@]} CPUs in '$cpus'" >&2; exit 2; fi
    echo "case,variant,threads,median_ns,samples" > "$out/$name-${t}t.csv"
    : > "$out/$name-${t}t.log"
done
mapfile -t cases < <("$bin" --list "${cargs[@]}")
tmp=$(mktemp); trap 'rm -f "$tmp"' EXIT
for c in "${cases[@]}"; do
    for t in "${threads[@]}"; do
        set_cpus=$(IFS=,; echo "${all[*]:0:$t}")
        BENCH_CASE=$c "$here/pinned.sh" "$set_cpus" -- "$bin" --threads "$t" "${cargs[@]}" > "$tmp"
        grep -v -E '^(#|case,|CHECK)' "$tmp" >> "$out/$name-${t}t.csv" || true
        grep -E '^(#|CHECK)' "$tmp" >> "$out/$name-${t}t.log" || true
    done
done
echo "paired.sh: ${#cases[@]} cases x ${threads[*]} threads -> $out" >&2
