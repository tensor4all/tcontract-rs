"""Acceptance of tcontract-rs#194 from probe runs of the candidate (and the base).

For every (dtype, shape, budget): chosen = the auto row's `Plan::execution_width`;
t(w) = forced-row median per round, then the median over the rounds.
Pass = t(chosen) <= 1.05 t(1) and t(chosen) <= 1.05 min_{measured w <= budget} t(w).

usage: accept.py <run dir>... (each holding round*.csv[.gz])
"""
import sys, glob, collections, statistics as st
from load import load

def main(dirs):
    paths = sorted(p for d in dirs for p in glob.glob(f'{d}/round*.csv*'))
    T, A, D = load(paths)
    out = collections.defaultdict(lambda: collections.Counter())
    worst = collections.defaultdict(list)
    for (dt, m, n, k, B), v in sorted(A.items()):
        ch = v[0][0]
        if (dt, m, n, k, ch) not in T: continue
        t1 = T[(dt, m, n, k, 1)]; tc = T[(dt, m, n, k, ch)]
        best = min(t for (d2, m2, n2, k2, w), t in T.items() if (d2, m2, n2, k2) == (dt, m, n, k) and w <= B)
        c = out[(dt, B)]; c['shapes'] += 1
        r1, rb = tc / t1, tc / best
        c['reg'] += r1 > 1.05; c['nobest'] += rb > 1.05; c['both'] += r1 <= 1.05 and rb <= 1.05
        c['serial'] += ch == 1
        worst[(dt, B)].append((r1, rb, (m, n, k), ch))
    print('| dtype | budget | shapes | pass both | > 1.05 x width 1 | > 1.05 x best | chose width 1 | max t/t(1) | max t/best | median t/best |')
    print('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for (dt, B), c in sorted(out.items()):
        w = worst[(dt, B)]
        print(f"| {dt} | {B} | {c['shapes']} | {c['both']} | {c['reg']} | {c['nobest']} | {c['serial']} | "
              f"{max(x[0] for x in w):.2f} | {max(x[1] for x in w):.2f} | {st.median(x[1] for x in w):.3f} |")
    return T, A, worst

if __name__ == '__main__':
    main(sys.argv[1:])
