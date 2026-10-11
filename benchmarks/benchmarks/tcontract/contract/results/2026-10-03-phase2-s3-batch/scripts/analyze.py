#!/usr/bin/env python3
"""Phase 2 S3 before/after table.  analyze.py RESULTS_DIR SESSION... > table.txt

Per case x width: packed_exec median [us] before (c71bba4) and after (this
branch), each as the median over sessions; speedup = before/after; noise =
|log(s2/s1)| of the row's own two sessions (A/A), max of both rows. The plan
row (faer) is printed for reference, from the after binary.
"""
import math, statistics, sys
from pathlib import Path

res = Path(sys.argv[1]); sessions = [res / s for s in sys.argv[2:]]
GROUPS = [
    ("tenferro-p1", "batched/tiny"), ("tenferro-p1-gemm", "batched/tiny"),
    ("phase2-extra", "batched/tiny"), ("large-batched-gemm", "large batched (H >= width rule)"),
]
CONTROLS = {("tenferro-p1", "dot_general_008_f64"), ("tenferro-p1", "dot_general_016_c64"),
            ("tenferro-p1-gemm", "gemm_batched_000_f64"),
            ("phase2-profile", "gemm_1024_f64"), ("phase2-profile", "gemm_1024_c64")}

def load(s, name, v, t, row):
    out = {}
    f = s / f"{name}-{v}-{t}t.csv"
    if not f.exists(): return out
    for line in f.read_text().splitlines()[1:]:
        c, var, _, ns, _ = line.split(",")
        if var == row: out[c] = float(ns)
    return out

print(f"sessions: {' '.join(s.name for s in sessions)}; metric: packed_exec median (us); speedup = before/after; noise = A/A spread of the case's own rows\n")
print(f"{'corpus':<19}{'case':<40}{'T':>2} {'before':>11} {'after':>11} {'speedup':>8} {'noise':>6} {'plan(faer)':>11} {'role':>8}")
summary = {}
names = ["tenferro-p1", "tenferro-p1-gemm", "phase2-extra", "large-batched-gemm", "phase2-profile"]
for name in names:
    for t in (1, 4, 8):
        b = [load(s, name, "before", t, "packed_exec") for s in sessions]
        a = [load(s, name, "after", t, "packed_exec") for s in sessions]
        pl = [load(s, name, "after", t, "plan_exec") for s in sessions]
        cases = sorted(set.intersection(*(set(d) for d in b + a)) if b and a else [])
        for c in cases:
            bm = statistics.median(d[c] for d in b); am = statistics.median(d[c] for d in a)
            noise = max(abs(math.log(d[-1][c] / d[0][c])) for d in (b, a)) if len(sessions) > 1 else 0
            plm = statistics.median(d[c] for d in pl if c in d)
            role = "control" if (name, c) in CONTROLS else ("large" if name == "large-batched-gemm" else "batched")
            print(f"{name:<19}{c:<40}{t:>2} {bm/1e3:>11.1f} {am/1e3:>11.1f} {bm/am:>8.2f} {noise:>6.3f} {plm/1e3:>11.1f} {role:>8}")
            summary.setdefault((role, t), []).append((bm / am, noise, c))
print()
for (role, t), rows in sorted(summary.items()):
    sp = [r[0] for r in rows]
    worst = min(rows)
    print(f"{role:<8} {t}T: n={len(rows)} geomean speedup {math.exp(statistics.mean(math.log(x) for x in sp)):.2f}  min {worst[0]:.2f} ({worst[2]}, noise {worst[1]:.3f})  max {max(sp):.2f}")
