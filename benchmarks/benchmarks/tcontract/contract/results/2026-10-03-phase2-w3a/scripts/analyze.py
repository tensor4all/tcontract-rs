#!/usr/bin/env python3
"""W3 table.  analyze.py RESULTS_DIR SESSION... > table.txt

Per case x width: packed_exec median [us] for variants before, after (W3a),
k512 and model (W3b arms on the W3a code), each the median over sessions;
ratio = variant/before (<1 = faster). noise = A/A spread between sessions of the
before row; ctl = the untouched plan (faer) row after/before, a same-host control.
"""
import math, statistics, sys
from pathlib import Path
res = Path(sys.argv[1]); sessions = [res / s for s in sys.argv[2:]]
V = ["before", "after", "k512", "model"]
def load(s, name, v, t, row):
    f = s / f"{name}-{v}-{t}t.csv"; out = {}
    if f.exists():
        for l in f.read_text().splitlines()[1:]:
            c, var, _, ns, _ = l.split(",")
            if var == row: out[c] = float(ns)
    return out
print(f"sessions {' '.join(s.name for s in sessions)}; packed_exec median us; ratio = variant/before\n")
print(f"{'corpus':<17}{'case':<34}{'T':>2} {'before':>10} {'after':>7} {'k512':>7} {'model':>7} {'noise':>6} {'ctl':>6}")
agg = {}
for name in ["phase2-profile", "tenferro-p1", "tenferro-p1-gemm", "phase2-extra"]:
    for t in (1, 4, 8):
        d = {v: [load(s, name, v, t, "packed_exec") for s in sessions] for v in V}
        pl = {v: [load(s, name, v, t, "plan_exec") for s in sessions] for v in ("before", "after")}
        cases = sorted(set.intersection(*(set(x) for v in V for x in d[v]))) if all(d["before"]) else []
        for c in cases:
            m = {v: statistics.median(x[c] for x in d[v]) for v in V}
            noise = abs(math.log(d["before"][-1][c] / d["before"][0][c])) if len(sessions) > 1 else 0
            ctl = statistics.median(x[c] for x in pl["after"]) / statistics.median(x[c] for x in pl["before"])
            print(f"{name:<17}{c:<34}{t:>2} {m['before']/1e3:>10.1f} {m['after']/m['before']:>7.3f} {m['k512']/m['before']:>7.3f} {m['model']/m['before']:>7.3f} {noise:>6.3f} {ctl:>6.3f}")
            for v in V[1:]: agg.setdefault((t, v), []).append(m[v] / m["before"])
            agg.setdefault((t, "ctl"), []).append(ctl)
print()
for (t, v), xs in sorted(agg.items()):
    print(f"{t}T {v:<6} n={len(xs)} geomean {math.exp(statistics.mean(math.log(x) for x in xs)):.3f} min {min(xs):.3f} max {max(xs):.3f}")
