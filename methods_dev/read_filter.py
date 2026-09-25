"""Read-level ambient/crosstalk filter using EMBEDDED (b,o) distances.

The oracle (drop reads with true d_bo large) recovers rho ~0.8.  Approximate it
without positions:

  1. E_b = UMAP(C)                       -- receiver embedding (crosstalk-robust)
  2. E_o = count-weighted centroid of the receivers that capture sender o
     (95% of a sender's reads are from its near receivers, so the centroid
     survives ~5-10% contamination)
  3. dhat_bo = ||E_b - E_o||             -- distance proxy for every read
  4. drop the reads with the largest dhat_bo (the long-range / misassigned tail)
  5. iterate: re-embed the cleaned counts, re-estimate E_o, re-filter

Usage:  python read_filter.py sim
        python read_filter.py real c58_8
"""
import os
import sys
import time
import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis_scripts"))
from common import cos_sim  # noqa: E402
from decontaminate import simulate_clean, crosstalk  # noqa: E402

RADIUS = 1500.0


def umap_embed(C, epochs=400):
    import umap
    E = umap.UMAP(random_state=42, metric="cosine", n_neighbors=25, min_dist=0.99,
                  n_components=2, n_epochs=epochs).fit_transform(np.log1p(C))
    return E - E.mean(0)


def sender_centroids(C, Eb, trim=True):
    w = C / np.maximum(C.sum(0, keepdims=True), 1e-9)      # column-normalised
    Eo = w.T @ Eb                                          # O x 2
    if trim:                                               # one robust pass
        d = np.sqrt(((Eb[:, None, :] - Eo[None, :, :]) ** 2).sum(-1))  # B x O  (big)
        keep = d < np.quantile(d, 0.85, axis=0, keepdims=True)
        Cw = C * keep
        w = Cw / np.maximum(Cw.sum(0, keepdims=True), 1e-9)
        Eo = w.T @ Eb
    return Eo


def dhat(C, Eb, Eo, chunk=2000):
    B, O = C.shape
    D = np.empty((B, O), dtype=np.float32)
    for i0 in range(0, B, chunk):
        sl = slice(i0, i0 + chunk)
        D[sl] = np.sqrt(((Eb[sl, None, :] - Eo[None, :, :]) ** 2).sum(-1))
    return D


def read_filter_embed(C, n_iter=3, remove_frac=0.10, verbose=True):
    C = np.asarray(C, float)
    Chat = C.copy()
    for it in range(n_iter):
        Eb = umap_embed(Chat)
        Eo = sender_centroids(Chat, Eb)
        Dh = dhat(Chat, Eb, Eo)
        # threshold on read-weighted dhat distribution
        rr, oo = np.nonzero(Chat)
        w = Chat[rr, oo]
        dv = Dh[rr, oo]
        order = np.argsort(dv)
        cutoff = dv[order[np.searchsorted(np.cumsum(w[order]), (1 - remove_frac) * w.sum())]]
        drop = Dh > cutoff
        removed = Chat[drop].sum()
        Chat = C * (~drop)                                 # filter from original
        if verbose:
            print(f"    iter {it}: dhat cutoff={cutoff:.1f}  "
                  f"removed {100*removed/C.sum():.1f}% of reads")
    return Chat


# --------------------------------------------------------------- evaluation ---
def diag(C, td, samp, iu, n, tag):
    S = cos_sim(C); np.fill_diagonal(S, 0.0)
    rho = spearmanr(S[iu][samp], -td[iu][samp]).correlation
    k = 10
    sn = np.argpartition(-S, k + 1, axis=1)[:, :k + 1]
    tn = np.argpartition(td + np.eye(n) * 1e18, k, axis=1)[:, :k]
    j = np.mean([1 - len((set(sn[i]) - {i}) & set(tn[i])) /
                 len((set(sn[i]) - {i}) | set(tn[i])) for i in range(n)])
    print(f"  {tag:34s} reads={int(C.sum()):>11,}  rho={rho:.3f}  kNN10-Jac={j:.3f}")


def run(C, X, U, td):
    n = C.shape[0]
    Dbo = cdist(X, U)
    iu = np.triu_indices(n, 1)
    samp = np.random.RandomState(0).choice(len(iu[0]), min(10 ** 6, len(iu[0])),
                                           replace=False)
    diag(C, td, samp, iu, n, "raw")
    Co = C.copy(); Co[Dbo > 800] = 0
    diag(Co, td, samp, iu, n, "ORACLE d_bo>800 removed")

    # how good is the embedded distance proxy?
    Eb = umap_embed(C)
    Eo = sender_centroids(C, Eb)
    Dh = dhat(C, Eb, Eo)
    rr, oo = np.nonzero(C)
    sub = np.random.RandomState(1).choice(len(rr), min(500000, len(rr)), replace=False)
    print(f"  corr(dhat, d_true) over reads = "
          f"{spearmanr(Dh[rr[sub], oo[sub]], Dbo[rr[sub], oo[sub]]).correlation:.3f}")

    for rf in (0.08, 0.15, 0.25):
        Ch = read_filter_embed(C, n_iter=3, remove_frac=rf, verbose=False)
        diag(Ch, td, samp, iu, n, f"read-filter-embed remove {int(rf*100)}%")


def main_sim():
    C0, X, U = simulate_clean(n=5000)
    td = distance_matrix(X, X)
    for mu in [0.0, 0.05, 0.1]:
        print(f"mu = {mu}")
        run(C0 if mu == 0 else crosstalk(C0, mu), X, U, td)
        print()


def main_real(rep="c58_8"):
    from real_data import load_hu
    C, X, U = load_hu(rep)
    run(C, X, U, distance_matrix(X, X))


if __name__ == "__main__":
    a = sys.argv[1:]
    t0 = time.time()
    (main_real(a[1] if len(a) > 1 else "c58_8") if a and a[0] == "real"
     else main_sim())
    print(f"{time.time()-t0:.0f}s")
