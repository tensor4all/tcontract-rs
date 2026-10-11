#!/usr/bin/env bash
# Every contract case in its own process, thread counts paired per case.
#   run.sh OUTDIR CPUS THREADS...      e.g. run.sh out 8-15 1 4 8
# CORPUS=file replays a recorded corpus instead of the built-in one.
set -euo pipefail
root=$(cd "$(dirname "$0")/../../../.." && pwd)
exec "$root/benchmarks/scripts/paired.sh" "$root/target/release/contract" "$@"
