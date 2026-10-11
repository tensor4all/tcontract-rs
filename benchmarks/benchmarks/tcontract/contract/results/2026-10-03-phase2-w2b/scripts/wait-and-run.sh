#!/usr/bin/env bash
# Wait for an idle L3 domain (another user's unpinned jobs keep the host busy), then run both sessions on it.
d=$(cd "$(dirname "$0")/.." && pwd); root=$(cd "$d/../../../../../.." && pwd)
until C=$(python3 "$root/benchmarks/scripts/idle_cpus.py" pick 8 2>/dev/null) && [ -n "$C" ]; do sleep 30; done
echo "start on cpus $C $(date -u +%FT%TZ)"
CPUS=$C "$d/scripts/session.sh" s1
until C=$(python3 "$root/benchmarks/scripts/idle_cpus.py" pick 8 2>/dev/null) && [ -n "$C" ]; do sleep 30; done
CPUS=$C "$d/scripts/session.sh" s2
