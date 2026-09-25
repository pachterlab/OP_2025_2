"""Finer k-sweep for the hard-kNN-restricted accumap on real Hu c58_8,
now that each restricted solve is known to be cheap (~1min vs ~70min for
the full-weight version) -- find where the RMSE actually bottoms out.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from knn_accumap_real import build_knn_mask, fit_curve, accumap_solve  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis_scripts"))
from common import cos_sim  # noqa: E402
from decontaminate import umap_embed  # noqa: E402
from position_error import position_rmse, pairwise_rmse  # noqa: E402
from real_data import load_hu  # noqa: E402
from scipy.spatial import distance_matrix


def main(rep="c58_8"):
    t0 = time.time()
    C, X_true, U_true = load_hu(rep)
    C = np.asarray(C, float)
    truth_dist = distance_matrix(X_true, X_true)
    print(f"{rep}: {C.shape[0]} beads", flush=True)

    S = cos_sim(C); np.fill_diagonal(S, 0.0)
    E_umap = umap_embed(C)
    print(f"  UMAP  posRMSE={position_rmse(E_umap, X_true):7.1f}  "
          f"pairRMSE={pairwise_rmse(E_umap, X_true):7.1f}  ({time.time()-t0:.0f}s)", flush=True)

    D = fit_curve(S, truth_dist)

    for k in (5, 8, 10, 12, 15, 20):
        mask = build_knn_mask(S, k)
        W = np.where(mask, S, 0.0)
        Xr = accumap_solve(E_umap, D, W, cycles=10)
        print(f"  kNN-accumap k={k:<3d}  posRMSE={position_rmse(Xr, X_true):7.1f}  "
              f"pairRMSE={pairwise_rmse(Xr, X_true):7.1f}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "c58_8")
