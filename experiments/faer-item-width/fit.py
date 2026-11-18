"""Fit the Faer item-time row of one host class from probe data (tcontract-rs#194).

The generic `ItemTime` fit is `item_fit.py`; this file is the features of one
rank-2 faer GEMM of W multiply-adds (complex counts four) on an m x n output:

    P       = ceil(m / TM) * ceil(n / TN)
    slow(w) = n > X w

The slow band is the regime in which faer's x86 kernel packs the left operand
because a participant gets more than X columns (t4a-private-gemm-x86
`pack_lhs = .. || n > 3 * nr * n_threads`). Measured, it costs a share of the
serial time that grows with the width (c32 16 x 320 x 160: 1.32x at w = 4,
3.32x at 8, 3.89x at 16), hence Q1. Tried and not kept: no split for an output narrower than one column
tile (removes the f64 narrow-n tail at 1.25-1.38x but leaves large-m narrow-n
products serial, up to 5x from the best width).

usage: uv run --with numpy python fit.py <dtype> results/fit-<dtype>/round*.csv.gz
The rows in crates/tcontract/src/strategy/faer_time.rs are its output.
"""
import sys
import numpy as np
from item_fit import fit
from load import muladds

class Gemm:
    muladds = staticmethod(muladds)
    grid = {"TM": [64, 128, 1e9], "TN": [12, 16, 24, 32, 48], "X": [8, 12, 16, 24]}

    @staticmethod
    def split(S, TM, TN, X):
        return np.ceil(S[:, 0] / TM) * np.ceil(S[:, 1] / TN)

    @staticmethod
    def slow(S, w, TM, TN, X):
        return S[:, 1:2] > X * w

if __name__ == '__main__':
    fit(Gemm, sys.argv[1], sys.argv[2:])
