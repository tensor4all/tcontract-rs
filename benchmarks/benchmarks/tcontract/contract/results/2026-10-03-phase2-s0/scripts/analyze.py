#!/usr/bin/env python3
"""Phase 2 S0 analysis: packed/plan per corpus x width, with tblis_decision's metric and noise.

  analyze.py RESULTS_DIR SESSION... > decision.txt   (targets go to RESULTS_DIR/targets.txt)

Metric (tblis_decision.workload): sum over entries of calls x exec median.
Noise (tblis_decision.noise_of): mean |log(total / median over sessions)|, max of
both rows. With the two A/A sessions the noise is that between the sessions.
"""
import json, re, statistics, sys
from pathlib import Path
here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parents[5] / "scripts"))
import tblis_decision as td

root = here.parents[4]
corpdir = root / "tprims/corpus"
res = Path(sys.argv[1]); sessions = [res / s for s in sys.argv[2:]]
CORPORA = ["tenferro-p1", "tenferro-p1-gemm", "large-batched-gemm", "phase2-extra"]
WIDTHS = [1, 4, 8]
SERIAL_NS = 50_000

def load(s, name, t):
    rows = {}
    for line in (s / f"{name}-{t}t.csv").read_text().splitlines()[1:]:
        c, v, _, ns, _ = line.split(",")
        rows.setdefault(c, {})[v] = float(ns)
    return rows

def faer_cases(s, name, t):
    pat = re.compile(r"^\[(.+?)\] # selected \1 plan: (\S+)")
    out = {}
    for line in (s / f"{name}-{t}t.log").read_text().splitlines():
        m = pat.match(line)
        if m: out[m.group(1)] = m.group(2)
    return out

out = []; targets = []
P = out.append
P(f"sessions: {' '.join(s.name for s in sessions)} (A/A noise = between sessions)")
P("ratio = packed/plan workload (calls x exec median); >1 means packed is slower. noise = mean |log(total/median)| across sessions, max of both rows.")
P(f"gate (spec 5): faer-subset ratio <= max(1.05, 1+noise); no faer case above 50us with ratio > 1.5\n")
P(f"{'corpus':<20}{'T':>2} {'n':>4} {'nfaer':>5} {'faer ratio':>10} {'noise':>6} {'gate':>5} | {'whole ratio':>11} {'noise':>6} | {'>50us&>1.5x':>11}")
alltargets = {}
for name in CORPORA:
    calls = {e["name"]: e.get("calls") or 1 for e in json.loads((corpdir / f"{name}.json").read_text())["entries"]}
    for t in WIDTHS:
        data = [load(s, name, t) for s in sessions]
        cases = sorted(set.intersection(*(set(d) for d in data)) & set(calls))
        miss = set(calls) - set(cases)
        assert not miss, (name, t, miss)
        sel = faer_cases(sessions[0], name, t)
        faer = [c for c in cases if sel.get(c) == "faer"]
        def tot(cs, v): return [td.workload(d, calls, cs, v) for d in data]
        def rn(cs):
            a, b = tot(cs, "plan_exec"), tot(cs, "packed_exec")
            return statistics.median(b) / statistics.median(a), max(td.noise_of(a), td.noise_of(b))
        fr, fn = rn(faer) if faer else (float("nan"), 0)
        wr, wn = rn(cases)
        med = lambda c, v, data=data: statistics.median(d[c][v] for d in data)
        bad = [c for c in faer if med(c, "plan_exec") > SERIAL_NS and med(c, "packed_exec") / med(c, "plan_exec") > 1.5]
        ok = fr <= max(1.05, 1 + fn) and not bad
        P(f"{name:<20}{t:>2} {len(cases):>4} {len(faer):>5} {fr:>10.3f} {fn:>6.3f} {'PASS' if ok else 'FAIL':>5} | {wr:>11.3f} {wn:>6.3f} | {len(bad):>11}")
        rows = []
        for c in cases:
            p, k = med(c, "plan_exec"), med(c, "packed_exec")
            rows.append((calls[c] * (k - p), c, p, k, calls[c], sel.get(c, "?")))
        alltargets[(name, t)] = (bad, sorted(rows, reverse=True)[:10], med)
P("\n(ratio over faer subset with nfaer=0 is nan)")
T = targets.append
T("S1-S3 target list. Per corpus x width: (a) faer-selected cases above 50us with packed/plan > 1.5; (b) top 10 cases by weighted loss = calls x (packed - plan) exec median [ns]; plan route in last column.\n")
for (name, t), (bad, top, med) in alltargets.items():
    T(f"== {name} {t}T ==")
    T("(a) faer cases >50us with ratio > 1.5:")
    if not bad: T("    none")
    for c in bad:
        p, k = med(c, "plan_exec"), med(c, "packed_exec")
        T(f"    {c:<44} plan {p/1e3:>11.1f}us packed {k/1e3:>11.1f}us ratio {k/p:5.2f}")
    T("(b) top 10 weighted loss:")
    for loss, c, p, k, n, r in top:
        T(f"    {c:<44} calls {n:>5} plan {p/1e3:>11.1f}us packed {k/1e3:>11.1f}us ratio {k/p:5.2f} loss {loss/1e6:>10.2f}ms route {r}")
    T("")
(res / "targets.txt").write_text("\n".join(targets) + "\n")
print("\n".join(out))
