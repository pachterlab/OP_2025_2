"""Jaccard-distance-vs-neighbourhood-size curve for UMAP and the hard-kNN
accumap (k=5) on real c58_8 -- the exact metric reviewer M7 asked for,
fitting naturally into a Supp Fig 3 style panel.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis_scripts"))
from accumap import correct_locations  # noqa: E402
sys.path.insert(0, os.path.dirname(__file__))
from common import cos_sim, jaccard_curve  # noqa: E402
from decontaminate import umap_embed  # noqa: E402
from position_error import position_rmse, pairwise_rmse  # noqa: E402
from real_data import load_hu  # noqa: E402
from knn_accumap_real import build_knn_mask, fit_curve  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    rep, k = "c58_8", 5
    C, X_true, U_true = load_hu(rep)
    C = np.asarray(C, float)
    S = cos_sim(C); np.fill_diagonal(S, 0.0)
    truth_dist = distance_matrix(X_true, X_true)
    E_umap = umap_embed(C)
    print(f"UMAP done ({time.time()-t0:.0f}s)", flush=True)

    D = fit_curve(S, truth_dist)
    mask = build_knn_mask(S, k)
    W = np.where(mask, S, 0.0)
    df = pd.DataFrame(E_umap, columns=["xcoord", "ycoord"])
    for it in range(10):
        df = correct_locations(df, D, weight=W, gauss_seidel=True,
                               correct_order="sorted" if it < 5 else "random", verbose_step=10 ** 9)
    X_knn = np.nan_to_num(df.values)
    print(f"kNN-accumap k={k} done ({time.time()-t0:.0f}s)", flush=True)

    for name, E in [("UMAP", E_umap), (f"accumap (hard k-NN, k={k})", X_knn)]:
        pos, pair = position_rmse(E, X_true), pairwise_rmse(E, X_true)
        jac = jaccard_curve(E, X_true)
        print(f"\n{name}: posRMSE={pos:.1f} pairRMSE={pair:.1f}")
        print(f"  J@10={jac.get(10, float('nan')):.3f}")
        print(f"  full curve: {jac}")

    print(f"\ntotal {time.time()-t0:.0f}s")
