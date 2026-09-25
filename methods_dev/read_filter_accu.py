"""Does the single-pass embedded-distance read filter recover accumap
RECONSTRUCTION (pairwise-distance RMSE), not just rho(S,-d)?

rho improves a lot (0.13 -> 0.68 at mu=0.05) but kNN-10 Jaccard does not move,
so the local neighbour graph -- what accumap actually fits -- may be unchanged.
This settles it: run accumap on raw / count-only-decontam / dhat-filter /
oracle-filter and compare pairRMSE.

Usage:  python read_filter_accu.py [mu]      # default sweeps 0.05, 0.1
"""
import os
import sys
import time
import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis_scripts"))
from common import cos_sim  # noqa: E402
from decontaminate import (simulate_clean, crosstalk, umap_embed, decontaminate,  # noqa: E402
                           accumap_fit)
from position_error import pairwise_rmse  # noqa: E402
from read_filter_v2 import sender_centroids, dhat_reads  # noqa: E402


def dhat_filter(C, Eb, q_cells):
    """zero every (b,o) cell whose embedded distance is above the q_cells
    quantile of the per-cell dhat distribution (single pass)."""
    Eo = sender_centroids(C, Eb)
    rr, oo, w, dv = dhat_reads(C, Eb, Eo)
    thr = np.quantile(dv, q_cells)
    m = dv > thr
    Cf = C.copy()
    Cf[rr[m], oo[m]] = 0.0
    return Cf, w[m].sum() / w.sum()


def score(C, X, tag):
    S = cos_sim(C); np.fill_diagonal(S, 0.0)
    td = distance_matrix(X, X)
    E = umap_embed(C)
    Xr = accumap_fit(C, S, td, E)
    pr = pairwise_rmse(Xr, X)
    pu = pairwise_rmse(E, X)
    print(f"  {tag:34s} accumap pairRMSE={pr:7.1f}   UMAP pairRMSE={pu:7.1f}")
    return pr


def run(mu):
    print(f"\n=== mu = {mu} ===")
    C0, X, U = simulate_clean(n=4000)
    Dbo = cdist(X, U)
    C = crosstalk(C0, mu)

    score(C, X, "raw (contaminated)")

    Cdec, a = decontaminate(C, verbose=False)
    print(f"  [decontaminate a_hat mean={a.mean():.3f}]")
    score(Cdec, X, "count-only decontam")

    Eb = umap_embed(C)
    for q in (0.80, 0.85, 0.90):
        Cf, fr = dhat_filter(C, Eb, q)
        score(Cf, X, f"dhat-filter p{int(q*100)} ({100*fr:.1f}% reads)")

    Co = C.copy(); Co[Dbo > 800] = 0.0
    score(Co, X, "ORACLE d_bo>800")
    score(C0, X, "clean reference")


if __name__ == "__main__":
    t0 = time.time()
    mus = [float(sys.argv[1])] if len(sys.argv) > 1 else [0.05, 0.10]
    for mu in mus:
        run(mu)
    print(f"\n{time.time()-t0:.0f}s")
