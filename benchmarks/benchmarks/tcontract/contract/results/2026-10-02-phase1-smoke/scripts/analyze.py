#!/usr/bin/env python3
import csv, sys, collections, os
R = sys.argv[1]
corpora = ["tenferro-p1", "hadamard", "tenferro-p1-gemm", "large-batched-gemm"]
out = []
over = []
def load(side, f):
    d = collections.defaultdict(dict)
    p = f"{R}/{side}-{f}.csv"
    if not os.path.exists(p): return d
    for r in csv.DictReader(open(p)):
        d[r["case"]][r["variant"]] = (int(r["median_ns"]), int(r["samples"]))
    return d
def calls(f):
    # calls-weighted: corpus entries may carry a 'calls' field
    import json
    j = json.load(open(f"/home/shinaoka/tensor4all/tprims-rs-smoke/benchmarks/benchmarks/tprims/corpus/{f}.json"))
    return {e["name"]: e.get("calls", 1) for e in j["entries"]}
for f in corpora:
    new, base = load("new", f), load("base", f)
    cl = calls(f)
    # default route: new plan_exec; base pg_exec for contract corpora, faer for blas corpora
    bkey = "faer" if "gemm" in f else "pg_exec"
    out.append(f"\n== {f}: new=plan_exec vs base={bkey}  (also packed_exec vs {'tblis' if 'gemm' in f else 'tblis_exec'})")
    out.append(f"{'case':28s} {'new_ns':>12s} {'base_ns':>12s} {'ratio':>7s} {'packed/tblis':>12s}")
    tn = tb = 0.0
    n = 0
    skipped = []
    for c in sorted(new):
        if c not in base: skipped.append(c); continue
        nv = new[c].get("plan_exec") or new[c].get("exec")
        bv = base[c].get(bkey)
        if not nv or not bv: skipped.append(c); continue
        r = nv[0] / bv[0]
        pk = new[c].get("packed_exec"); tb_ = base[c].get("tblis" if "gemm" in f else "tblis_exec")
        pr = f"{pk[0]/tb_[0]:.2f}" if pk and tb_ else "-"
        out.append(f"{c:28s} {nv[0]:12d} {bv[0]:12d} {r:7.2f} {pr:>12s}")
        if r > 2.0: over.append((f, c, nv[0], bv[0], r))
        w = cl.get(c, 1)
        tn += w * nv[0]; tb += w * bv[0]; n += 1
    if tb: out.append(f"-- {f}: {n} cases, calls-weighted workload ratio new/base = {tn/tb:.3f}")
    if skipped: out.append(f"-- skipped/unmapped: {skipped}")
out.append("\n== cases with ratio > 2.0")
for o in over: out.append("%s %s new=%d base=%d ratio=%.2f" % o)
if not over: out.append("none")
print("\n".join(out))
