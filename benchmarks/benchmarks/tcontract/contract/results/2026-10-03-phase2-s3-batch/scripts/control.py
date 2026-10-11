#!/usr/bin/env python3
"""W1 control: packed-route tenferro-p1 cases, c71bba4 vs main@#54.  control.py DIR > tenferro-p1-control.txt
ratio = main/c71bba4 packed_exec median over the two sessions; noise = A/A spread of the c71bba4 row between sessions.
Flag: ratio > 1 + max(0.05, 2*noise)."""
import math, statistics, sys
from pathlib import Path
root = Path(sys.argv[1]); sess = [root / "control-s1", root / "control-s2"]
def load(s, v, t):
    f = s / f"tenferro-p1-{v}-{t}t.csv"; out = {}
    if f.exists():
        for l in f.read_text().splitlines()[1:]:
            c, var, _, ns, _ = l.split(",")
            if var == "packed_exec": out[c] = float(ns)
    return out
print("tenferro-p1 packed-route cases (default plan selects packed), packed_exec median us; ratio = main@#54 / c71bba4; flag = ratio > 1+max(0.05, 2*noise)\n")
print(f"{'case':<22}{'T':>2} {'c71bba4':>11} {'main':>11} {'ratio':>6} {'noise':>6} flag")
flags = []; agg = {}
for t in (1, 4, 8):
    b = [load(s, "c71bba4", t) for s in sess]; a = [load(s, "main54", t) for s in sess]
    for c in sorted(set.intersection(*(set(d) for d in b + a))):
        bm = statistics.median(d[c] for d in b); am = statistics.median(d[c] for d in a)
        noise = abs(math.log(b[1][c] / b[0][c]))
        r = am / bm; f = r > 1 + max(0.05, 2 * noise)
        if f: flags.append((c, t, r))
        agg.setdefault(t, []).append((r, bm))
        print(f"{c:<22}{t:>2} {bm/1e3:>11.1f} {am/1e3:>11.1f} {r:>6.3f} {noise:>6.3f} {'FLAG' if f else ''}")
print()
for t, xs in sorted(agg.items()):
    w = sum(bm * r for r, bm in xs) / sum(bm for r, bm in xs)
    print(f"{t}T: n={len(xs)} geomean {math.exp(statistics.mean(math.log(r) for r, _ in xs)):.3f} time-weighted {w:.3f} max {max(r for r,_ in xs):.3f}")
print("flagged:", flags or "none")
