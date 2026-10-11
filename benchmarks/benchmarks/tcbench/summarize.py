#!/usr/bin/env python3
"""Summarize the complete fixed A/A suite, retaining every case in comparison.csv."""
import csv
import math
import statistics
import sys
from pathlib import Path

out = Path(sys.argv[1])
threads_to_run = tuple(map(int, sys.argv[2].split(','))) if len(sys.argv) > 2 else (1, 4, 8, 12)
assert threads_to_run in ((1, 4, 8, 12), (1, 4, 8)), "explicit full or single-L3 protocol only"
data = {}
for sample in ("A", "A2"):
    for size in (1, 16):
        for threads in threads_to_run:
            stem = out / f"{sample}-{size}m-{threads}t"
            assert "idle before and after" in stem.with_suffix(".guard").read_text(), stem
            rows = list(csv.DictReader(stem.with_suffix(".csv").open()))
            assert len(rows) == 49 * 2 * 2, (stem, len(rows))
            keys = set()
            for r in rows:
                key = (r["case"], r["dtype"], r["engine"])
                assert key not in keys, (stem, key)
                keys.add(key)
                seconds = float(r["seconds"])
                assert math.isfinite(seconds) and seconds > 0
                assert int(r["threads"]) == threads
                assert "MISMATCH" not in r["notes"]
                data[sample, size, threads, *key] = seconds
for size in (1, 16):
    for threads in threads_to_run:
        assert "all comparisons within tolerance" in (out / f"verify-{size}m-{threads}t.txt").read_text()

cases = sorted({key[3] for key in data})
assert len(cases) == 49
engines = ("plan", "packed")
fields = ["size_mib", "threads", "dtype", "case"]
for engine in engines:
    fields += [f"{engine}_A_s", f"{engine}_A2_s", f"{engine}_AA_spread"]
fields += ["plan_scaling"]
comparison = []
summary = []
for size in (1, 16):
    for dtype in ("f64", "c64"):
        for threads in threads_to_run:
            group = []
            for case in cases:
                r = dict(size_mib=size, threads=threads, dtype=dtype, case=case)
                times = {}
                for engine in engines:
                    a, b = (data[sample, size, threads, case, dtype, engine] for sample in ("A", "A2"))
                    r[f"{engine}_A_s"], r[f"{engine}_A2_s"] = a, b
                    r[f"{engine}_AA_spread"] = abs(b / a - 1)
                    times[engine] = math.sqrt(a * b)
                serial = math.sqrt(math.prod(data[s, size, 1, case, dtype, "plan"] for s in ("A", "A2")))
                r["plan_scaling"] = serial / times["plan"]
                comparison.append(r)
                group.append(r)
            gm = lambda field: statistics.geometric_mean(r[field] for r in group)
            noises = [r[f"{e}_AA_spread"] for r in group for e in engines]
            summary.append(f"| {size} | {dtype} | {threads} | {gm('plan_scaling'):.2f} | {100*statistics.median(noises):.1f}% | {100*max(noises):.1f}% |")
with (out / "comparison.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(comparison)
print("| MiB target | dtype | T | tcontract 1T speedup | A/A median | A/A max |")
print("| --- | --- | --- | --- | --- | --- |")
print("\n".join(summary))
