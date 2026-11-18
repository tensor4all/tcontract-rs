"""Fit the Faer item-time row of one host class from probe data (tcontract-rs#194).

Model of one rank-2 item of W multiply-adds (complex counts four) on an m x n
output at width w, excluding the token's dispatch cost D(w):

    t(1) = F + W/R
    t(w) = F + O + W/R / min(w, P) + C (w - 1) + [n > X w] (Q0 + Q1 (w - 1)) W/R
    P    = ceil(m / TM) * ceil(n / TN)

The last term is the regime in which faer's x86 kernel packs the left operand
because a participant gets more than X columns (t4a-private-gemm-x86
`pack_lhs = .. || n > 3 * nr * n_threads`). Measured, it costs a share of the
serial time that grows with the width (c32 16 x 320 x 160: 1.32x at w = 4,
3.32x at 8, 3.89x at 16), hence Q1. Tried and not kept: no split for an output narrower than one column
tile (removes the f64 narrow-n tail at 1.25-1.38x but leaves large-m narrow-n
products serial, up to 5x from the best width).

F, R: least squares on the relative error of t(1). The rest: grid search over
the (shape, budget) pairs, budgets 16 and 96, the model's argmin taken over the
measured widths <= budget of t(w) + D(w). Score = pairs within 5% of width 1 and
of the best width, minus 3 per pair above 1.05 x width 1 and 30 more per pair
above 1.25 x width 1 (the merge bound).

usage: uv run --with numpy python fit.py <dtype> results/fit-<dtype>/round*.csv.gz
The rows in crates/tcontract/src/strategy/faer_time.rs are its output.
"""
import sys, itertools
import numpy as np
from load import load, muladds

ANCHORS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96]
BUDGETS = [16, 96]

def arrays(T, dt):
    shapes = sorted({k[1:4] for k in T if k[0] == dt})
    S = np.array(shapes, dtype=float)
    W = np.array([muladds(dt, *s) for s in shapes], dtype=float)
    Tm = np.array([[T[(dt,) + s + (w,)] for w in ANCHORS] for s in shapes])
    return shapes, S, W, Tm

def serial_fit(W, t1):
    wt = 1 / t1
    A = np.stack([np.ones_like(W), W], 1) * wt[:, None]
    (F, g), *_ = np.linalg.lstsq(A, t1 * wt, rcond=None)
    return max(F, 0.0), 1 / g

def model(p, S, W):
    F, R, O, C, TM, TN, X, Q0, Q1 = p
    w = np.array(ANCHORS, dtype=float)[None, :]
    work = (W / R)[:, None]
    P = np.ceil(S[:, 0] / TM) * np.ceil(S[:, 1] / TN)
    t = F + O + work / np.minimum(w, P[:, None]) + C * (w - 1)
    t = t + (S[:, 1:2] > X * w) * (Q0 + Q1 * (w - 1)) * work
    t[:, 0] = F + W / R
    return t

def evaluate(p, S, W, Tm, D):
    tm = model(p, S, W) + np.array([D[w] for w in ANCHORS])[None, :]
    out = []
    for B in BUDGETS:
        inb = np.array(ANCHORS) <= B
        ch = np.argmin(np.where(inb[None, :], tm, np.inf), 1)
        tc = Tm[np.arange(len(W)), ch]
        out.append((tc / Tm[:, 0], tc / Tm[:, inb].min(1), np.array(ANCHORS)[ch]))
    return out

def score(res):
    both = sum(((r1 <= 1.05) & (rb <= 1.05)).sum() for r1, rb, _ in res)
    reg = sum((r1 > 1.05).sum() for r1, _, _ in res)
    tail = sum((r1 > 1.25).sum() for r1, _, _ in res)
    return both - 3 * reg - 30 * tail, both, reg, tail

def main():
    dt = sys.argv[1]; T, A, D = load(sys.argv[2:])
    D[1] = 0
    shapes, S, W, Tm = arrays(T, dt)
    F, R = serial_fit(W, Tm[:, 0])
    best = None
    for O, C, TM, TN, X, Q0, Q1 in itertools.product(
            np.arange(0, 16001, 1000), [0, 10, 25, 50, 100], [64, 128, 1e9], [12, 16, 24, 32, 48],
            [8, 12, 16, 24], [0, 0.5, 1, 2], [0, 0.05, 0.1, 0.2, 0.4]):
        p = (F, R, O, C, TM, TN, X, Q0, Q1)
        sc = score(evaluate(p, S, W, Tm, D))
        if best is None or sc[0] > best[0][0]: best = (sc, p)
    (_, both, reg, tail), p = best; tot = len(shapes) * len(BUDGETS)
    print(f"{dt}: F={p[0]:.0f} ns, R={p[1]:.3f} muladds/ns, O={p[2]:.0f} ns, C={p[3]} ns, "
          f"TM={p[4]:g}, TN={p[5]:g}, X={p[6]:g}, Q0={p[7]:g}, Q1={p[8]:g}")
    print(f"  within 5% of w1 and best: {both}/{tot}; > 1.05 x w1: {reg}; > 1.25 x w1: {tail}")
    for B, (r1, rb, ch) in zip(BUDGETS, evaluate(p, S, W, Tm, D)):
        print(f"    budget {B}: max t/t1 {r1.max():.2f}, max t/best {rb.max():.2f}, median t/best {np.median(rb):.3f}")

if __name__ == '__main__':
    main()
