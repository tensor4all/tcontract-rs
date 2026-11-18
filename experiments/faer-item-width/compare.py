"""Base against candidate on the acceptance sweep (tcontract-rs#194).

Per (dtype, budget): the regressions (> 1.05 x width 1) of each binary within its
own run, and the geometric mean over shapes of t(chosen) / t(1) (lower is
better). `best` is from the candidate run (the base run only times width 1 and
its chosen width): the base's t(chosen)/t(best) is estimated as
t_base(chosen)/t_base(1) * t_cand(1)/t_cand(best).

usage: compare.py <base dir> <candidate dir>
"""
import sys, glob, math, collections
from load import load

def runs(d):
    return load(sorted(glob.glob(f'{d}/round*.csv*')))

def main(bd, cd):
    Tb, Ab, _ = runs(bd); Tc, Ac, _ = runs(cd)
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    for key, v in Ac.items():
        dt, m, n, k, B = key
        if key not in Ab: continue
        s = (dt, m, n, k)
        best = min(t for (d2, m2, n2, k2, w), t in Tc.items() if (d2, m2, n2, k2) == s and w <= B)
        cc, bc = v[0][0], Ab[key][0][0]
        rc = Tc[s + (cc,)] / Tc[s + (1,)]
        rb = Tb[s + (bc,)] / Tb[s + (1,)]
        r = rows[(dt, B)]
        r['cand'].append(rc); r['base'].append(rb)
        r['cand_best'].append(Tc[s + (cc,)] / best)
        r['base_best'].append(rb * Tc[s + (1,)] / best)
    g = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))
    print('| dtype | budget | shapes | base: > 1.05 x w1 | base: max t/t(1) | base: geo-mean t/t(1) | base: est. > 1.05 x best | candidate: > 1.05 x w1 | candidate: max t/t(1) | candidate: geo-mean t/t(1) | candidate: > 1.05 x best |')
    print('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for (dt, B), r in sorted(rows.items()):
        print(f"| {dt} | {B} | {len(r['cand'])} | {sum(x > 1.05 for x in r['base'])} | {max(r['base']):.2f} | {g(r['base']):.3f} | "
              f"{sum(x > 1.05 for x in r['base_best'])} | {sum(x > 1.05 for x in r['cand'])} | {max(r['cand']):.2f} | {g(r['cand']):.3f} | "
              f"{sum(x > 1.05 for x in r['cand_best'])} |")

if __name__ == '__main__':
    main(*sys.argv[1:3])
