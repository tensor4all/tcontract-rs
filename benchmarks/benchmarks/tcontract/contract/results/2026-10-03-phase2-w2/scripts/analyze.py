#!/usr/bin/env python3
"""W2: packed/plan exec ratio per (corpus, mode, width): workload = sum calls x median.
  analyze.py RESULTS_DIR SESSION > decision-table.txt
ratio > 1 means faer (plan) is faster than packed. Also lists the worst cases."""
import json, sys, math
from pathlib import Path
res = Path(sys.argv[1]); sess = res / sys.argv[2]
corp = Path(__file__).resolve().parents[4] / "corpus"
def rows(f):
    d = {}
    for l in f.read_text().splitlines()[1:]:
        c, v, _, ns, _ = l.split(",")
        d.setdefault(c, {})[v] = float(ns)
    return d
for f in sorted(sess.glob("*-1t.csv")):
    tag = f.name[:-7]
    cname = tag.split("-separate")[0]; mode = "separate" + tag.split("-separate")[1]
    calls = {e["name"]: e.get("calls") or 1 for e in json.load(open(corp / f"{cname}.json"))["entries"]}
    for t in (1, 4, 8):
        g = sess / f"{tag}-{t}t.csv"
        if not g.exists(): continue
        r = {c: v for c, v in rows(g).items() if "plan_exec" in v and "packed_exec" in v}
        if not r: continue
        wp = sum(calls[c] * v["plan_exec"] for c, v in r.items())
        wk = sum(calls[c] * v["packed_exec"] for c, v in r.items())
        ratios = {c: v["packed_exec"] / v["plan_exec"] for c, v in r.items()}
        gm = math.exp(sum(map(math.log, ratios.values())) / len(ratios))
        lose = sorted((x, c, r[c]["plan_exec"]) for c, x in ratios.items() if x < 0.95)
        print(f"{cname:20s} {mode:14s} {t}T n={len(r):3d} workload packed/faer={wk/wp:6.3f} geomean={gm:6.3f} faer-loses(<0.95): {len(lose)}")
        for x, c, ns in lose[:6]:
            print(f"      {c} packed/faer={x:.2f} faer={ns/1e3:.1f}us")
