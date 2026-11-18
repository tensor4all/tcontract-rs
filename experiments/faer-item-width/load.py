"""Load probe CSVs: forced times (median over rounds) and the team dispatch row."""
import csv, collections, gzip, statistics as st

def load(paths):
    F = collections.defaultdict(list); A = {}; D = {}
    for p in paths:
        f = gzip.open(p, "rt") if p.endswith(".gz") else open(p)
        for r in csv.reader(f):
            if r[0] == 'DISPATCH':
                w, ns = int(r[1]), int(r[2]); D[w] = 0 if w == 1 else ns
            if r[0] != 'ROW': continue
            kind, dt = r[1], r[2]; m, n, k, w, b, ch = map(int, r[3:9]); med = float(r[9])
            if kind == 'forced': F[(dt, m, n, k, w)].append(med)
            else: A.setdefault((dt, m, n, k, b), []).append((ch, med))
    T = {key: st.median(v) for key, v in F.items()}
    return T, A, D

CPLX = {'f64': 1, 'f32': 1, 'c64': 4, 'c32': 4}
def muladds(dt, m, n, k): return m * n * k * CPLX[dt]
