#!/usr/bin/env python3
"""W2b: packed / plan exec ratio per (corpus, mode, width) for the shipped planner.

  analyze.py RESULTS_DIR SESSION...

A case whose effective route (for b0: the beta=0 route, else the algorithm) is packed
runs the same code in both rows, so its ratio is a measure of the A/A noise; a case on
faer is a real comparison. Reported per mode and width: workload (sum calls x median)
packed/plan, faer-routed cases below 0.95, and the A/A spread of the packed-routed cases
(geomean |log ratio| and the worst)."""
import json, math, re, sys
from pathlib import Path
res = Path(sys.argv[1]); sessions = sys.argv[2:]
corp = Path(__file__).resolve().parents[4] / "corpus"
def rows(f):
    d = {}
    for l in f.read_text().splitlines()[1:]:
        c, v, _, ns, _ = l.split(","); d.setdefault(c, {})[v] = float(ns)
    return d
def routes(f, mode):
    algo, b0 = {}, {}
    for l in f.read_text().splitlines():
        m = re.match(r"^\[(.+?)\] # selected \1 plan( beta0)?: (\S+)", l)
        if m: (b0 if m.group(2) else algo)[m.group(1)] = m.group(3)
    return {c: (b0.get(c, a) if mode.endswith("b0") else a) for c, a in algo.items()}
for sess in sessions:
  for f in sorted((res / sess).glob("*-1t.csv")):
    tag = f.name[:-7]
    cname, mode = tag.split("-separate"); mode = "separate" + mode
    calls = {e["name"]: e.get("calls") or 1 for e in json.load(open(corp / f"{cname}.json"))["entries"]}
    for t in (1, 4, 8):
        g = res / sess / f"{tag}-{t}t.csv"
        r = {c: v for c, v in rows(g).items() if "plan_exec" in v and "packed_exec" in v}
        if not r: continue
        rt = routes(res / sess / f"{tag}-{t}t.log", mode)
        wp = sum(calls[c] * v["plan_exec"] for c, v in r.items()); wk = sum(calls[c] * v["packed_exec"] for c, v in r.items())
        ratio = {c: v["packed_exec"] / v["plan_exec"] for c, v in r.items()}
        fa = {c: x for c, x in ratio.items() if rt.get(c) == "faer"}
        pk = {c: x for c, x in ratio.items() if rt.get(c) == "packed"}
        wf_p = sum(calls[c] * r[c]["plan_exec"] for c in fa); wf_k = sum(calls[c] * r[c]["packed_exec"] for c in fa)
        lose = sorted((x, c) for c, x in fa.items() if x < 0.95)
        aa = math.exp(sum(abs(math.log(x)) for x in pk.values()) / len(pk)) - 1 if pk else float("nan")
        aaw = max((abs(math.log(x)) for x in pk.values()), default=0)
        print(f"{sess} {cname:19s} {mode:13s} {t}T n={len(r):2d} faer={len(fa):2d} workload packed/plan all={wk/wp:5.3f} faer-routed={wf_k/wf_p if fa else float('nan'):5.3f}  faer<0.95: {len(lose)}  A/A on {len(pk)} packed-routed: mean {aa:.3f} worst x{math.exp(aaw):.2f}")
        for x, c in lose: print(f"      {c}: {x:.2f} (faer {r[c]['plan_exec']/1e3:.0f}us)")
