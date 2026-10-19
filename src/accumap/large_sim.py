"""Large shared-bead arrays with known truth (used to develop the large-array path): jittered hexagonal monolayer (26 um
spacing, random rotation) filling a W x H rectangle; C_ij ~ Poisson(A exp(q_i + a_j) (K(d_ij) + c)) with the kernel K and
background c from accumap's fit on SCOPE's real circular array; q, a ~ N(0, 0.4); A sets the median counts per bead (sent +
received). Pairs beyond 700 um receive background counts only (partners in proportion to exp(a)). Receivers are processed in
chunks, so memory stays bounded at millions of beads.

  simulate_shared_array(width_um, height_um, seed) -> (X (n, 2) true positions in um, centred; C (n, n) CSR counts)"""
import time
import numpy as np
import scipy.sparse as sp

_KD = np.array([11.7, 23.3, 46.7, 81.7, 116.7, 175.0, 233.3, 350.0, 466.7, 700.0, 1166.7, 1866.7, 3500.0])
_KV = np.array([1.0, 0.98138, 0.89151, 0.31998, 0.10499, 0.03668, 0.02133, 0.00709, 0.00310, 0.00087, 0.00019, 0.00003, 1e-6])
_BG, _A0, _RNEAR, _CH = 5.469e-5, 26.0, 700.0, 4000


def _kernel(d):
    return np.exp(np.interp(np.log(np.clip(d, _KD[0], _KD[-1])), np.log(_KD), np.log(_KV)))


def simulate_shared_array(width_um, height_um, seed=0, median_counts=1500.0, verbose=False):
    """Simulated shared-bead array in a width_um x height_um rectangle (see module docstring) -> (X, C)."""
    from scipy.spatial import cKDTree
    t0 = time.time(); rng = np.random.default_rng(seed)
    sx, sy = float(width_um), float(height_um); side = max(sx, sy)
    nx = int(side * 1.5 / _A0); ny = int(nx / (np.sqrt(3) / 2)) + 1
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    g = np.c_[(ii + 0.5 * (jj % 2)).ravel(), (jj * np.sqrt(3) / 2).ravel()] * _A0
    g -= g.mean(0); th = rng.uniform(0, np.pi); R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    X = g @ R.T + rng.normal(0, 3.0, g.shape)
    X = X[(np.abs(X[:, 0]) <= sx / 2) & (np.abs(X[:, 1]) <= sy / 2)]
    X = X[np.lexsort((X[:, 0], np.floor(X[:, 1] / 500)))]      # spatially coherent chunks (faster tree queries)
    n = len(X)
    q, ab = rng.normal(0, 0.4, n), rng.normal(0, 0.4, n); eq, ea = np.exp(q), np.exp(ab)
    tree = cKDTree(X)

    def chunk_pairs(lo):
        nb = tree.query_ball_point(X[lo:lo + _CH], _RNEAR, return_sorted=False)
        i = np.repeat(np.arange(lo, lo + len(nb)), [len(v) for v in nb]); j = np.concatenate(nb)
        k = i != j
        return i[k], j[k]

    tot = np.zeros(n); near_ea = np.zeros(n)
    for lo in range(0, n, _CH):                                # pass 1: expected counts per bead (sent + received)
        i, j = chunk_pairs(lo)
        lam = eq[i] * ea[j] * (_kernel(np.linalg.norm(X[i] - X[j], axis=1)) + _BG)
        tot += np.bincount(i, lam, n) + np.bincount(j, lam, n); near_ea += np.bincount(i, ea[j], n)
    A = float(median_counts) / np.median(tot)
    rows, cols, vals = [], [], []
    for lo in range(0, n, _CH):                                # pass 2: sample near counts
        i, j = chunk_pairs(lo)
        c = rng.poisson(A * eq[i] * ea[j] * (_kernel(np.linalg.norm(X[i] - X[j], axis=1)) + _BG))
        k = c > 0; rows.append(i[k].astype(np.int32)); cols.append(j[k].astype(np.int32)); vals.append(c[k].astype(np.float32))
    far = A * eq * _BG * (ea.sum() - near_ea - ea)             # background beyond _RNEAR
    nf = rng.poisson(np.clip(far, 0, None)); fi = np.repeat(np.arange(n, dtype=np.int32), nf)
    fj = rng.choice(n, size=len(fi), p=ea / ea.sum()).astype(np.int32)
    C = sp.coo_matrix((np.concatenate(vals + [np.ones(len(fi), np.float32)]), (np.concatenate(rows + [fi]), np.concatenate(cols + [fj]))),
                      shape=(n, n)).tocsr()
    C.setdiag(0); C.eliminate_zeros()
    if verbose:
        print(f"{n:,} beads in a {sx / 1000:g} x {sy / 1000:g} mm rectangle; {int(C.sum()):,} counts, {C.nnz:,} nonzero "
              f"({time.time() - t0:.0f} s)", flush=True)
    return X.astype(np.float32), C
