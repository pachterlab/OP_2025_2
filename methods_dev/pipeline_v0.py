"""Anchor-free reconstruction v0: local physical metric + graph geodesics.

  raw cosine similarity  ->  mutual-kNN graph (k nearest, local only)
  edge length  =  2 sigma sqrt(-log S)          [physical, near pairs only]
  all-pairs shortest path  (geodesic completion of far distances)
  ->  classical MDS  ->  2D layout

sigma comes from the assay FWHM (a chemistry constant, not per-dataset ground
truth).  No imaging positions are used by the solver.
"""
import os
import sys
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from real_data import load_hu
from common import SIGMA, classical_mds, per_bead_mse, summary


def mutual_knn_edges(S, k):
    n = S.shape[0]
    nn = np.argpartition(-S, k + 1, axis=1)[:, :k + 1]
    nnset = [set(row) for row in nn]
    edges = set()
    for i in range(n):
        for j in nn[i]:
            if j != i and i in nnset[j]:
                edges.add((i, j) if i < j else (j, i))
    return np.array(sorted(edges))


def pipeline_v0(C, sigma=SIGMA, k=15, denoise_rank=0):
    A = np.log1p(C)
    if denoise_rank:
        from sklearn.utils.extmath import randomized_svd
        U, s, Vt = randomized_svd(A, denoise_rank, random_state=0)
        A = (U * s) @ Vt
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    S = A @ A.T
    np.fill_diagonal(S, 0.0)

    E = mutual_knn_edges(S, k)
    sij = np.clip(S[E[:, 0], E[:, 1]], 1e-6, 1 - 1e-9)
    d = 2.0 * sigma * np.sqrt(np.maximum(-np.log(sij), 0.0))

    n = S.shape[0]
    rows = np.r_[E[:, 0], E[:, 1]]
    cols = np.r_[E[:, 1], E[:, 0]]
    W = csr_matrix((np.r_[d, d], (rows, cols)), shape=(n, n))
    G = shortest_path(W, method="D", directed=False)

    finite = np.isfinite(G)
    if not finite.all():
        G[~finite] = G[finite].max() * 1.5
    np.fill_diagonal(G, 0.0)

    X = classical_mds(G)
    return X, {"n_edges": int(len(E)), "frac_unreachable": float((~finite).mean())}


if __name__ == "__main__":
    rep = sys.argv[1] if len(sys.argv) > 1 else "c58_8"
    C, X_true, _ = load_hu(rep)
    cache = os.path.join(os.path.dirname(__file__), "results_real", f"{rep}_cache.npz")
    E_umap = np.load(cache)["E_umap"] if os.path.exists(cache) else None
    umse = per_bead_mse(E_umap, X_true) if E_umap is not None else None
    if umse is not None:
        r, _ = summary("UMAP", E_umap, X_true, umap_mse=umse)
        print("UMAP baseline:", {k: round(v, 3) if isinstance(v, float) else v
                                 for k, v in r.items()})
    for k in (10, 15, 25, 40):
        for rank in (0, 100):
            t = time.time()
            X, info = pipeline_v0(C, k=k, denoise_rank=rank)
            row, jac = summary(f"v0 k={k} rank={rank}", X, X_true, umap_mse=umse)
            print({**{kk: round(vv, 3) if isinstance(vv, float) else vv
                      for kk, vv in row.items()}, **info}, f"{time.time()-t:.0f}s",
                  flush=True)
