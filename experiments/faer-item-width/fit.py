"""Fit the Faer item-time row of one host class from probe data (tcontract-rs#194, #203).

The generic `ItemTime` fit is `item_fit.py`; this file is the features of one
rank-2 faer GEMM of W multiply-adds (complex counts four) on an m x n output,
read from the partition of `t4a-private-gemm-x86` 0.1.28 (`lib.rs`) with the
AVX-512 kernels:

    mr, nr  = the microkernel tile of the dtype (`gemm_impl`: f32 96 x 4,
              f64 48 x 4 above 48 rows and 24 x 8 otherwise, c32 48 x 4,
              c64 24 x 4)
    rows    = 2 mr rows per task (`MilliPar::call`: `mf` is 2 for a wide output,
              n >= 2 m, and for every width w >= ceil(m / mr) / 4)
    cols    = 32 columns per task (`nf = 32 / nr`)
    P       = ceil(m / rows) * ceil(n / 32), taken in turn from one counter
    L       = min(m, rows) * min(n, 32) / (m n), the largest task's share
    slow(w) = n > 3 nr w

The slow band is where the kernel packs the left operand (`gemm_impl`:
`pack_lhs = .. || n > 3 * nr * n_threads`): every task of a row block waits
for the one that packs it (`claim_pack`). Measured, it costs a share of the
serial time that grows with the width (f64 80 x 512 x 40, 16 tasks: 0.82-0.90 x
width 1 at w = 5..11, 2.3x at 16; c32 16 x 320 x 160: 1.3x at 4, 3.9x at 16),
hence Q1. Nothing here is fitted; the grid of the features is empty.

usage: uv run --with numpy python fit.py <dtype> results/fit-<dtype>/round*.csv.gz
The rows in crates/tcontract/src/strategy/faer_time.rs are its output.
"""
import sys
import numpy as np
from item_fit import fit
from load import muladds

class Gemm:
    muladds = staticmethod(muladds)
    grid = {"E": [0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.2]}

    def __init__(self, dt):
        self.dt = dt

    def tile(self, S):
        """(mr, nr) per shape."""
        m = S[:, 0]
        mr = {'f32': 96, 'c32': 48, 'c64': 24}.get(self.dt)
        if mr is not None:
            return np.full_like(m, mr), np.full_like(m, 4)
        return np.where(m > 48, 48, 24), np.where(m > 48, 4, 8)

    def split(self, S, E):
        m, n, k = S[:, 0], S[:, 1], S[:, 2]
        mr, _ = self.tile(S)
        rows = 2 * mr
        P = np.ceil(m / rows) * np.ceil(n / 32)
        L = np.minimum(m, rows) * np.minimum(n, 32) / (m * n)
        return P, L, E * (np.minimum(m, rows) + np.minimum(n, 32)) * k

    def slow(self, S, w, E):
        _, nr = self.tile(S)
        return S[:, 1:2] > 3 * nr[:, None] * w

if __name__ == '__main__':
    fit(Gemm(sys.argv[1]), sys.argv[1], sys.argv[2:])
