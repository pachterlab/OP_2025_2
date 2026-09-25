"""Does the embedded-distance read filter recover accumap on the REAL Hu
c58_8 data, the way it does on the crosstalk simulation?

Sim result: oracle far-read removal restores accumap fully (pairRMSE 365 -> 24
at mu=0.05); the d_hat proxy filter at p80 gets 365 -> 69.  If real-data
failure is ALSO just crosstalk reads, d_hat-filter + accumap should drop well
below UMAP's ~41 um here.  If it stays high, real data has breakage beyond the
sim crosstalk model.

accumap_fit still calibrates the S->distance curve against imaging truth (same
as the sim benchmark) -- this experiment isolates the READ-contamination
contribution, holding the curve fixed.  It is a diagnostic, not a deployable
pipeline.

Usage:  python read_filter_real.py c58_8
"""
import os
import sys
import time
import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis_scripts"))
from common import cos_sim  # noqa: E402
from decontaminate import umap_embed, decontaminate, accumap_fit  # noqa: E402
from position_error import position_rmse, pairwise_rmse  # noqa: E402
from read_filter_v2 import sender_centroids, dhat_reads  # noqa: E402
from real_data import load_hu  # noqa: E402


def dhat_filter(C, Eb, q_cells):
    Eo = sender_centroids(C, Eb)
    rr, oo, w, dv = dhat_reads(C, Eb, Eo)
    thr = np.quantile(dv, q_cells)
    m = dv > thr
    Cf = C.copy()
    Cf[rr[m], oo[m]] = 0.0
    return Cf, w[m].sum() / w.sum()


def score(C, X, E_umap, td, tag):
    S = cos_sim(C); np.fill_diagonal(S, 0.0)
    Xr = accumap_fit(C, S, td, E_umap)
    print(f"  {tag:32s} accumap posRMSE={position_rmse(Xr, X):7.1f}  "
          f"pairRMSE={pairwise_rmse(Xr, X):7.1f}", flush=True)


def main(rep="c58_8"):
    C, X, U = load_hu(rep)
    C = np.asarray(C, float)
    td = distance_matrix(X, X)
    Dbo = cdist(X, U)
    E_umap = umap_embed(C)
    print(f"{rep}: C {C.shape}  {int(C.sum()):,} reads", flush=True)
    print(f"  {'UMAP baseline':32s} posRMSE={position_rmse(E_umap, X):7.1f}  "
          f"pairRMSE={pairwise_rmse(E_umap, X):7.1f}", flush=True)

    score(C, X, E_umap, td, "raw")

    Cdec, a = decontaminate(C, verbose=False)
    print(f"  [decontaminate a_hat mean={a.mean():.3f} median={np.median(a):.3f}]", flush=True)
    score(Cdec, X, E_umap, td, "count-only decontam")

    Eb = umap_embed(C)
    rr, oo, w, dv = dhat_reads(C, Eb, sender_centroids(C, Eb))
    dtrue = Dbo[rr, oo]
    from scipy.stats import spearmanr
    sub = np.random.RandomState(1).choice(len(rr), min(400000, len(rr)), replace=False)
    print(f"  corr(dhat, d_true) reads = {spearmanr(dv[sub], dtrue[sub]).correlation:.3f}", flush=True)

    for q in (0.75, 0.80, 0.85):
        Cf, fr = dhat_filter(C, Eb, q)
        score(Cf, X, E_umap, td, f"dhat-filter p{int(q*100)} ({100*fr:.1f}% reads)")

    Co = C.copy(); Co[Dbo > 800] = 0.0
    score(Co, X, E_umap, td, "ORACLE d_bo>800")


if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv[1] if len(sys.argv) > 1 else "c58_8")
    print(f"\n{time.time()-t0:.0f}s")
