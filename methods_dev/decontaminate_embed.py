"""Embedding-aware decontamination.

UMAP is robust to crosstalk, so its embedding gives a contamination-free
notion of "which beads are far".  Two ways to use it:

  A. embed-graph EM: run the count-only EM but smooth the native profile over
     the UMAP k-NN graph instead of the (contaminated) similarity k-NN graph.
  B. embed-excess: co-occurrence M_bb' vs embedding distance delta has a
     decreasing ridge (real) plus a flat second population at large delta
     (crosstalk).  Per-bead score = mean excess of M over the expected ridge
     across that bead's far pairs -> per-bead contamination fraction.

Fast: reports a_hat, Spearman(S,-d), kNN-10 Jaccard before/after.  No accumap
re-scoring (pathologically slow on real data).

Usage:  python decontaminate_embed.py sim
        python decontaminate_embed.py real c58_8
"""
import os
import sys
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis_scripts"))
from common import cos_sim  # noqa: E402
from decontaminate import decontaminate, simulate_clean, crosstalk  # noqa: E402

SIGMA = 123.0 / np.sqrt(8.0 * np.log(2.0))
RADIUS = 1500.0


def umap_embed(C, epochs=400):
    import umap
    E = umap.UMAP(random_state=42, metric="cosine", n_neighbors=25, min_dist=0.99,
                  n_components=2, n_epochs=epochs).fit_transform(np.log1p(C))
    return E - E.mean(0)


def knn_adj(P, k, metric="euclid", S=None):
    n = P.shape[0]
    if metric == "cos":
        idx = np.argpartition(-S, k + 1, axis=1)[:, :k + 1]
    else:
        d = cdist(P, P)
        idx = np.argpartition(d, k + 1, axis=1)[:, :k + 1]
    rows = np.repeat(np.arange(n), k + 1)
    return csr_matrix((np.full(n * (k + 1), 1.0 / (k + 1)), (rows, idx.ravel())),
                      shape=(n, n))


def decontam_graph(C, A, n_iter=12, a_max=0.7, chunk=2500):
    C = np.asarray(C, float)
    B, O = C.shape
    nb = C.sum(1)
    g = C.sum(0); g = g / g.sum()
    gO = g[None, :]
    s = C / np.maximum(nb[:, None], 1.0)
    a = np.full(B, 0.05)
    s_sm = None
    for _ in range(n_iter):
        s_sm = A @ s
        s_sm /= np.maximum(s_sm.sum(1, keepdims=True), 1e-12)
        contam = np.zeros(B)
        for i0 in range(0, B, chunk):
            sl = slice(i0, i0 + chunk)
            num = a[sl, None] * gO
            den = (1 - a[sl, None]) * s_sm[sl] + num + 1e-12
            z = num / den
            contam[sl] = (C[sl] * z).sum(1)
            nat = C[sl] * (1 - z)
            s[sl] = nat / np.maximum(nat.sum(1, keepdims=True), 1e-12)
        a = np.clip(contam / np.maximum(nb, 1.0), 0.0, a_max)
    Chat = np.empty_like(C)
    for i0 in range(0, B, chunk):
        sl = slice(i0, i0 + chunk)
        den = (1 - a[sl, None]) * s_sm[sl] + a[sl, None] * gO + 1e-12
        Chat[sl] = C[sl] * (1 - (a[sl, None] * gO) / den)
    return Chat, a


def embed_excess_alpha(C, E, nbin=40):
    B = C.shape[0]
    M = C @ C.T
    np.fill_diagonal(M, 0.0)
    row = np.sqrt((C ** 2).sum(1))
    Mn = M / (row[:, None] * row[None, :] + 1e-9)          # scale out depth
    d = distance_matrix(E, E)
    iu = np.triu_indices(B, 1)
    edges = np.quantile(d[iu], np.linspace(0, 1, nbin + 1))
    ctr = 0.5 * (edges[1:] + edges[:-1])
    b = np.clip(np.digitize(d, edges) - 1, 0, nbin - 1)
    ridge = np.array([np.median(Mn[b == k]) if (b == k).sum() > 200 else np.nan
                      for k in range(nbin)])
    ridge = np.array([np.nanmin(ridge[k:]) if np.isfinite(ridge[k]) else np.nan
                      for k in range(nbin)])          # non-increasing floor
    far = d > np.quantile(d[iu], 0.6)
    exp = np.interp(d, ctr, np.nan_to_num(ridge, nan=np.nanmin(ridge)))
    excess = np.clip(Mn - exp, 0, None)
    soup = np.nanmedian(Mn[b == nbin - 1])            # far-pair co-occ level
    score = (excess * far).sum(1) / np.maximum(far.sum(1), 1)
    a = np.clip(score / max(soup, 1e-9), 0, 0.7)
    a = a - a.min()
    return a


def diag(C, td, samp, iu, n, tag):
    S = cos_sim(C); np.fill_diagonal(S, 0.0)
    rho = spearmanr(S[iu][samp], -td[iu][samp]).correlation
    k = 10
    sn = np.argpartition(-S, k + 1, axis=1)[:, :k + 1]
    tn = np.argpartition(td + np.eye(n) * 1e18, k, axis=1)[:, :k]
    j = np.mean([1 - len((set(sn[i]) - {i}) & set(tn[i])) /
                 len((set(sn[i]) - {i}) | set(tn[i])) for i in range(n)])
    print(f"    {tag:16s} rho={rho:.3f}  kNN10-Jac={j:.3f}")


def run(C, X, td, mu=None):
    n = C.shape[0]
    iu = np.triu_indices(n, 1)
    samp = np.random.RandomState(0).choice(len(iu[0]), min(10 ** 6, len(iu[0])),
                                           replace=False)
    diag(C, td, samp, iu, n, "raw")

    E = umap_embed(C)
    A_emb = knn_adj(E, 20)

    Chat_c, a_c = decontaminate(C, verbose=False)
    Chat_g, a_g = decontam_graph(C, A_emb)
    a_x = embed_excess_alpha(C, E)

    tail = "" if mu is None else f" (true mu={mu})"
    print(f"    a_hat  count-only mean={a_c.mean():.3f}  embed-graph mean={a_g.mean():.3f}"
          f"  embed-excess mean={a_x.mean():.3f}{tail}")
    print(f"    per-bead corr(count-only, embed-excess) = "
          f"{np.corrcoef(a_c, a_x)[0,1]:.2f}")
    diag(Chat_c, td, samp, iu, n, "after count-only")
    diag(Chat_g, td, samp, iu, n, "after embed-graph")


def main_sim():
    C0, X, U = simulate_clean(n=5000)
    td = distance_matrix(X, X)
    for mu in [0.0, 0.05, 0.1]:
        print(f"mu = {mu}")
        C = C0 if mu == 0 else crosstalk(C0, mu)
        run(C, X, td, mu=mu)
        print()


def main_real(rep="c58_8"):
    from real_data import load_hu
    C, X_true, _ = load_hu(rep)
    td = distance_matrix(X_true, X_true)
    print(f"{rep}: C {C.shape}")
    run(C, X_true, td)


if __name__ == "__main__":
    a = sys.argv[1:]
    t0 = time.time()
    if a and a[0] == "real":
        main_real(a[1] if len(a) > 1 else "c58_8")
    else:
        main_sim()
    print(f"{time.time()-t0:.0f}s")
