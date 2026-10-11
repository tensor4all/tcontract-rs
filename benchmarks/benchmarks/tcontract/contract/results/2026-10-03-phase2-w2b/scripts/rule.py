#!/usr/bin/env python3
"""Evaluate a routing rule on the measured sessions: a case on packed under the rule has ratio 1.
  rule.py RESULTS_DIR SESSION MODE KMIN OUTMAX   (faer iff K >= KMIN or out <= OUTMAX)"""
import json, sys
from math import prod
from pathlib import Path
res, sess, mode, kmin, outmax = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
corp = Path(__file__).resolve().parents[4] / "corpus"
ents = {e["name"]: e for e in json.load(open(corp / "tenferro-p1-gemm.json"))["entries"]}
def rows(f):
    d = {}
    for l in f.read_text().splitlines()[1:]:
        n, v, _, ns, _ = l.split(","); d.setdefault(n, {})[v] = float(ns)
    return d
for t in (1, 4, 8):
    r = rows(res / sess / f"tenferro-p1-gemm-{mode}-{t}t.csv")
    wf = wp = 0.0; worst = 1e9; nf = 0
    for n, v in r.items():
        if "packed_exec" not in v or "plan_exec" not in v: continue
        e = ents[n]; K = prod(e["a"]["dims"][i] for i in e["lc"]); out = prod(e["c"]["dims"])
        faer = K >= kmin or out <= outmax
        cost = v["plan_exec"] if faer else v["packed_exec"]
        c = e["calls"]; wf += c * cost; wp += c * v["packed_exec"]
        if faer:
            nf += 1; worst = min(worst, v["packed_exec"] / v["plan_exec"])
    print(f"{mode:14s} {t}T faer cases {nf:2d} workload packed/rule={wp/wf:.3f} worst faer-routed ratio={worst:.2f}")
