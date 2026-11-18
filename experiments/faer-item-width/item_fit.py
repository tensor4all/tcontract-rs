"""Fit one host-class row of a provider's `ItemTime` from probe data.

`tcontract_exec::ItemTime` is the time of one item at width w, excluding the
token's dispatch cost D(w):

    t(1) = F + work
    t(w) = F + O + work / min(w, P) + C (w - 1) + [slow(w)] (Q0 + Q1 (w - 1)) work
    work = W / R

F and R are fitted for every provider in the same way; O, C, Q0 and Q1 are
searched on a grid. W (the item's multiply-adds), P (its split) and slow(w)
(the widths of the provider's slower path) are the provider's *features*: an
object with

    muladds(dtype, *shape) -> float      W of one shape
    grid: dict name -> values            the feature's own parameters, searched
                                         between C and Q0
    split(S, *feat) -> array             P per shape (S: one row per shape)
    slow(S, w, *feat) -> array           slow(w) per shape and width

F, R: least squares on the relative error of t(1). The rest: grid search over
the (shape, budget) pairs, budgets 16 and 96, the model's argmin taken over the
measured widths <= budget of t(w) + D(w). Score = pairs within 5% of width 1 and
of the best width, minus 3 per pair above 1.05 x width 1 and 30 more per pair
above 1.25 x width 1 (the merge bound).

The probe rows are those `load.py` reads. `fit.py` is the faer GEMM features.
"""
import itertools
import numpy as np
from load import load

ANCHORS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96]
BUDGETS = [16, 96]
O_GRID = np.arange(0, 16001, 1000)
C_GRID = [0, 10, 25, 50, 100]
Q0_GRID = [0, 0.5, 1, 2]
Q1_GRID = [0, 0.05, 0.1, 0.2, 0.4]

def arrays(features, T, dt):
    shapes = sorted({k[1:4] for k in T if k[0] == dt})
    S = np.array(shapes, dtype=float)
    W = np.array([features.muladds(dt, *s) for s in shapes], dtype=float)
    Tm = np.array([[T[(dt,) + s + (w,)] for w in ANCHORS] for s in shapes])
    return shapes, S, W, Tm

def serial_fit(W, t1):
    wt = 1 / t1
    A = np.stack([np.ones_like(W), W], 1) * wt[:, None]
    (F, g), *_ = np.linalg.lstsq(A, t1 * wt, rcond=None)
    return max(F, 0.0), 1 / g

def model(features, p, S, W):
    F, R, O, C, *feat, Q0, Q1 = p
    w = np.array(ANCHORS, dtype=float)[None, :]
    work = (W / R)[:, None]
    P = features.split(S, *feat)
    t = F + O + work / np.minimum(w, P[:, None]) + C * (w - 1)
    t = t + features.slow(S, w, *feat) * (Q0 + Q1 * (w - 1)) * work
    t[:, 0] = F + W / R
    return t

def evaluate(features, p, S, W, Tm, D):
    tm = model(features, p, S, W) + np.array([D[w] for w in ANCHORS])[None, :]
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

def fit(features, dt, paths):
    """Fit and print the row of `dt` from the probe files `paths`."""
    T, A, D = load(paths)
    D[1] = 0
    shapes, S, W, Tm = arrays(features, T, dt)
    F, R = serial_fit(W, Tm[:, 0])
    best = None
    for O, C, *feat, Q0, Q1 in itertools.product(
            O_GRID, C_GRID, *features.grid.values(), Q0_GRID, Q1_GRID):
        p = (F, R, O, C, *feat, Q0, Q1)
        sc = score(evaluate(features, p, S, W, Tm, D))
        if best is None or sc[0] > best[0][0]: best = (sc, p)
    (_, both, reg, tail), p = best; tot = len(shapes) * len(BUDGETS)
    feat = ", ".join(f"{name}={v:g}" for name, v in zip(features.grid, p[4:-2]))
    print(f"{dt}: F={p[0]:.0f} ns, R={p[1]:.3f} muladds/ns, O={p[2]:.0f} ns, C={p[3]} ns, "
          f"{feat}, Q0={p[-2]:g}, Q1={p[-1]:g}")
    print(f"  within 5% of w1 and best: {both}/{tot}; > 1.05 x w1: {reg}; > 1.25 x w1: {tail}")
    for B, (r1, rb, ch) in zip(BUDGETS, evaluate(features, p, S, W, Tm, D)):
        print(f"    budget {B}: max t/t1 {r1.max():.2f}, max t/best {rb.max():.2f}, median t/best {np.median(rb):.3f}")
