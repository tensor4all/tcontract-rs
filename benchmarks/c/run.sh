#!/usr/bin/env bash
# Build libtcontract_capi and the C bench, then run C and Rust at 1T and 4T, paired.
#   run.sh CPUS1 CPUS4 OUTDIR
set -euo pipefail
cpus1=$1; cpus4=$2; out=$3
root=$(cd "$(dirname "$0")/../.." && pwd)
cargo build --release -p tcontract-capi --manifest-path "$root/Cargo.toml" >&2
cargo build --release -p tcontract-bench --bin capi_rust --manifest-path "$root/Cargo.toml" >&2
lib=$root/target/release
pin=$root/benchmarks/scripts/pinned.sh
mkdir -p "$out"
cc -O2 -std=c11 -o "$out/bench_c" "$root/benchmarks/c/bench.c" -I"$root/crates/tcontract-capi/include" -L"$lib" -Wl,-rpath,"$lib" -ltcontract_capi -lm
for t in 1 4; do
    cpus=$cpus1; [[ $t == 4 ]] && cpus=$cpus4
    "$pin" "$cpus" -- "$out/bench_c" "$t" > "$out/c-${t}t.csv"
    "$pin" "$cpus" -- "$lib/capi_rust" --threads "$t" > "$out/rust-${t}t.csv"
done
