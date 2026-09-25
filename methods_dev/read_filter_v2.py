"""Single-pass, calibrated embedded-distance read filter.

The iterative fixed-fraction filter (read_filter.py) is unstable: re-embedding
sparser counts distorts the embedding, d_hat inflates, and it removes far more
than requested -- destroying signal on clean AND contaminated data.

This version does ONE pass and picks the cut adaptively from the shape of the
count-vs-d_hat decay, so on clean data (decay never flattens to a background
floor) nothing is removed.

  1. E_b  = UMAP(C)                       (scaled to ~um: max pairwise = 2*RADIUS)
  2. E_o  = robust count-weighted centroid of capturing receivers
  3. d_hat_bo = ||E_b - E_o||             for every nonzero read
  4. bin reads by d_hat; per-bin mean count cbar(d_hat)
  5. background floor c_bg = mean cbar over the top d_hat decile
     cut d_hat* = smallest d_hat where cbar has decayed to (1+m)*c_bg
  6. drop reads with d_hat > d_hat*   (single pass, no re-embed)

Compared against the oracle (drop reads with TRUE d_bo > 800um) and a plain
d_hat-percentile sweep.

Usage:  python read_filter_v2.py            # sim sweep mu=0,0.05,0.1
        python read_filter_v2.py real c58_8
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
from decontaminate import simulate_clean, crosstalk, umap_embed  # noqa: E402

RADIUS = 1500.0


def sender_centroids(C, Eb, trim_q=0.85, passes=2):
    Eo = None
    Cw = C
    for _ in range(passes):
        w = Cw / np.maximum(Cw.sum(0, keepdims=True), 1e-9)
        Eo = w.T @ Eb
        d = cdist(Eb, Eo)                       # B x O
        keep = d < np.quantile(d, trim_q, axis=0, keepdims=True)
        Cw = C * keep
    return Eo


def dhat_reads(C, Eb, Eo):
    rr, oo = np.nonzero(C)
    dv = np.sqrt(((Eb[rr] - Eo[oo]) ** 2).sum(1))
    return rr, oo, C[rr, oo].astype(float), dv


def adaptive_cut(dv, w, nbin=40, margin=0.35):
    """smallest d_hat where the read-weighted mean count has decayed to the
    background floor (mean over the top d_hat decile)."""
    edges = np.quantile(dv, np.linspace(0, 1, nbin + 1))
    edges[-1] += 1e-6
    b = np.clip(np.digitize(dv, edges) - 1, 0, nbin - 1)
    cbar = np.array([w[b == k].mean() if (b == k).any() else np.nan
                     for k in range(nbin)])
    ctr = 0.5 * (edges[1:] + edges[:-1])
    c_bg = np.nanmean(cbar[int(0.9 * nbin):])
    c0 = np.nanmean(cbar[:max(1, nbin // 10)])
    if not np.isfinite(c_bg) or c0 <= c_bg * (1 + margin):
        return np.inf, cbar, ctr, c_bg          # never flattens -> keep all
    below = np.where(cbar <= c_bg * (1 + margin))[0]
    return ctr[below[0]], cbar, ctr, c_bg


def diag(C, td, samp, iu, n, tag):
    S = cos_sim(C); np.fill_diagonal(S, 0.0)
    rho = spearmanr(S[iu][samp], -td[iu][samp]).correlation
    k = 10
    sn = np.argpartition(-S, k + 1, axis=1)[:, :k + 1]
    tn = np.argpartition(td + np.eye(n) * 1e18, k, axis=1)[:, :k]
    j = np.mean([1 - len((set(sn[i]) - {i}) & set(tn[i])) /
                 len((set(sn[i]) - {i}) | set(tn[i])) for i in range(n)])
    print(f"  {tag:38s} reads={int(C.sum()):>11,}  rho={rho:.3f}  kNN10-Jac={j:.3f}")
    return rho


def run(C, X, U, td):
    n = C.shape[0]
    Dbo = cdist(X, U)
    iu = np.triu_indices(n, 1)
    samp = np.random.RandomState(0).choice(len(iu[0]), min(10 ** 6, len(iu[0])),
                                           replace=False)
    diag(C, td, samp, iu, n, "raw")
    Co = C.copy(); Co[Dbo > 800] = 0
    diag(Co, td, samp, iu, n, "ORACLE d_bo>800 removed")

    Eb = umap_embed(C)
    Eo = sender_centroids(C, Eb)
    rr, oo, w, dv = dhat_reads(C, Eb, Eo)
    dtrue = Dbo[rr, oo]
    sub = np.random.RandomState(1).choice(len(rr), min(400000, len(rr)), replace=False)
    print(f"  corr(dhat, d_true) reads = {spearmanr(dv[sub], dtrue[sub]).correlation:.3f}"
          f"   dhat p50/p90/p99 = {np.percentile(dv,[50,90,99]).round(0)}")

    # adaptive single-pass
    cut, cbar, ctr, c_bg = adaptive_cut(dv, w)
    frac = w[dv > cut].sum() / w.sum() if np.isfinite(cut) else 0.0
    print(f"  adaptive cut d_hat*={cut if np.isfinite(cut) else np.inf:.0f}um "
          f"(c_bg={c_bg:.2f})  removes {100*frac:.1f}% of reads")
    Ca = C.copy()
    if np.isfinite(cut):
        Ca[rr[dv > cut], oo[dv > cut]] = 0
    diag(Ca, td, samp, iu, n, "ADAPTIVE single-pass")

    # plain percentile sweep, single pass
    for q in (0.80, 0.90, 0.95):
        thr = np.quantile(dv, q)
        Cp = C.copy()
        m = dv > thr
        Cp[rr[m], oo[m]] = 0
        diag(Cp, td, samp, iu, n, f"d_hat > p{int(q*100)}  ({100*w[m].sum()/w.sum():.0f}% reads)")


def main_sim():
    C0, X, U = simulate_clean(n=4000)
    td = distance_matrix(X, X)
    for mu in [0.0, 0.05, 0.1]:
        print(f"\n=== mu = {mu} ===")
        run(C0 if mu == 0 else crosstalk(C0, mu), X, U, td)


def main_real(rep="c58_8"):
    from real_data import load_hu
    C, X, U = load_hu(rep)
    run(np.asarray(C, float), X, U, distance_matrix(X, X))


if __name__ == "__main__":
    a = sys.argv[1:]
    t0 = time.time()
    (main_real(a[1] if len(a) > 1 else "c58_8") if a and a[0] == "real" else main_sim())
    print(f"\n{time.time()-t0:.0f}s")
