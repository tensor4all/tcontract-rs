#!/usr/bin/env bash
# Full Hadamard nonregression experiment, not selective per-case retries.
set -euo pipefail
[[ $# == 3 ]] || { echo 'usage: hadamard-pair.sh OUT BASELINE CANDIDATE' >&2; exit 2; }
out=$(realpath -m "$1"); baseline=$(realpath "$2"); candidate=$(realpath "$3")
root=$(cd "$(dirname "$0")/../../.." && pwd); cd "$root"
corpus="$root/benchmarks/benchmarks/tcontract/corpus/hadamard.json"
mkdir -p "$out"
{ date -Iseconds; sha256sum "$baseline" "$candidate" "$corpus"; printf 'native/release; median20 (10 at work>=2^24); warmup3; CPUs4/4-7\n'; } > "$out/manifest.txt"
for sample in A A2; do
    arms='baseline candidate'; [[ $sample == A2 ]] && arms='candidate baseline'
    for t in 1 4; do
        for arm in $arms; do
            bin=${!arm}
            BENCH_RUNS=20 BENCH_WARMUP=3 bash benchmarks/scripts/pinned.sh "4-$((t+3))" -- "$bin" --threads "$t" --corpus "$corpus" > "$out/$sample-$arm-${t}t.txt" 2> "$out/$sample-$arm-${t}t.guard"
        done
    done
done
