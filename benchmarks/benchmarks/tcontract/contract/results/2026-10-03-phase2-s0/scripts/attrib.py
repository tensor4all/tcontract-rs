#!/usr/bin/env python3
"""Attribute packed-row cycles from prof.sh output. attrib.py PERF_DIR
Packed window = first..last sample of a tprims_kernel kernels::/pack:: symbol (the plan row is faer in these
cases and has no such samples). Per category: share of cycle samples in the window (all threads);
idle = 1 - samples/(T x window x F) is time threads were not on-CPU (parked/blocked: barrier/futex wait,
no cycles sampled). Page faults: count of page-fault events in the window by user symbol."""
import gzip, re, sys, collections
from pathlib import Path
F = 4999
def cat(sym, ip):
    if "tprims_kernel" in sym and "7kernels" in sym: return "kernel"
    if "tprims_kernel" in sym and "pack9writeback" in sym: return "writeback"
    if "tprims_kernel" in sym and ("4pack" in sym or "scatter" in sym): return "pack"
    if "tprims_contract" in sym or "tprims_exec" in sym: return "driver"
    if sym == "[unknown]":
        return "kernel-mode" if int(ip, 16) >> 44 == 0xfffff or ip.startswith("ffff") else "unknown-user"
    if re.search(r"futex|Barrier|park|wait|spin|yield|sched", sym, re.I): return "sync"
    if "rayon" in sym: return "rayon"
    if sym.startswith(("gemm_v0", "_RNvCs9bBgWXteRpH_16private_gemm")) or "faer" in sym or "gemm" in sym.lower(): return "faer(plan row)"
    return "other-user"
def parse(path):
    for l in gzip.open(path, "rt"):
        m = re.match(r"\s*(\d+)\s+([\d.]+):\s+(\S+)\s+(.*)$", l)
        if m: yield int(m[1]), float(m[2]), m[3], m[4].strip()
out = []
for f in sorted(Path(sys.argv[1]).glob("*.cycles.gz")):
    tag = f.name[:-len(".cycles.gz")]; T = int(re.search(r"-(\d+)t$", tag)[1])
    S = list(parse(f))
    hot = [t for _, t, ip, s in S if cat(s, ip) in ("kernel", "pack", "writeback") ]
    # window: dense span of tprims samples; take first..last
    t0, t1 = min(hot), max(hot)
    W = [(tid, ip, s) for tid, t, ip, s in S if t0 <= t <= t1]
    cnt = collections.Counter(cat(s, ip) for _, ip, s in W)
    n = sum(cnt.values()); win = t1 - t0
    # faer samples inside the window (interleaving with the plan row is not expected)
    idle = 1 - n / (T * win * F)
    pf = [ (t, s) for _, t, ip, s in parse(f.with_name(tag + ".pf.gz")) if t0 - 1e9 <= t ]
    # page-fault window uses its own timeline: take the same relative structure -> count by symbol over faults at tprims frames
    pfc = collections.Counter(cat(s, ip) for _, t, ip, s in parse(f.with_name(tag + ".pf.gz")))
    out.append(f"## {tag}  window {win*1e3:.1f} ms, {n} samples in window, threads {T}")
    out.append("  on-CPU share by category: " + ", ".join(f"{k} {v/n:.1%}" for k, v in cnt.most_common()))
    out.append(f"  off-CPU (idle/parked) fraction of T x window: {idle:.1%}")
    out.append("  page-fault events (whole process, by faulting code): " + ", ".join(f"{k} {v}" for k, v in pfc.most_common()))
    top = collections.Counter(s for _, ip, s in W if s != "[unknown]")
    out.append("  top symbols: " + "; ".join(f"{re.sub(r'^_R[A-Za-z]*?(?=tprims|private)','',s)[:60]} {v/n:.1%}" for s, v in top.most_common(6)))
    out.append("")
print("\n".join(out))
