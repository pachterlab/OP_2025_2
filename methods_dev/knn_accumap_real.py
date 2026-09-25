"""Test whether the hard k-NN restriction that closed most of the gap on
SCOPE letter_E also helps on the real Hu c58_8 data (distinct sender/
receiver). Same mechanism: accumap's own weighted-MDS objective normally
touches every pair with a soft S-derived weight; here it is restricted to a
symmetric union k-NN graph built from S, weight set to EXACTLY zero outside
it (not just small), everything else (curve fit calibrated against imaging
truth, same as the paper's "accumap + fitted curve" baseline) unchanged --
isolates the one variable tested on SCOPE.

Unlike SCOPE, Hu has real imaging ground truth, so this gives a direct
quantitative answer (posRMSE/pairRMSE), not just a neighbor-agreement proxy.

Usage:  python knn_accumap_real.py c58_8
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis_scripts"))
from accumap import correct_locations  # noqa: E402
from common import cos_sim  # noqa: E402
from decontaminate import umap_embed, diff_func  # noqa: E402
from position_error import position_rmse, pairwise_rmse  # noqa: E402
from real_data import load_hu  # noqa: E402


def build_knn_mask(S, k):
    B = S.shape[0]
    idx = np.argpartition(-S, k + 1, axis=1)[:, :k + 1]
    mask = np.zeros((B, B), dtype=bool)
    rows = np.repeat(np.arange(B), k + 1)
    mask[rows, idx.ravel()] = True
    mask |= mask.T
    np.fill_diagonal(mask, False)
    return mask


def fit_curve(S, truth_dist, seed=42):
    rng = np.random.RandomState(seed)
    bars = rng.randint(0, S.shape[0], 100)
    sv, dv = S[bars].ravel(), truth_dist[bars].ravel()
    m = sv > 0
    popt, _ = curve_fit(diff_func, sv[m], dv[m],
                        p0=(-5.389e-3, 1.104, 296.2, 783.9), maxfev=20000)
    D = diff_func(np.clip(S, 1e-9, 1 - 1e-9), *popt)
    D = np.nan_to_num(D, nan=3000.0, posinf=3000.0, neginf=10.0)
    D[S <= 0] = 3000.0
    D[D <= 10] = 10.0
    np.fill_diagonal(D, 0.0)
    return D


def accumap_solve(E_init, D, W, cycles=10):
    df = pd.DataFrame(E_init, columns=["xcoord", "ycoord"])
    for it in range(cycles):
        df = correct_locations(df, D, weight=W, gauss_seidel=True,
                               correct_order="sorted" if it < cycles // 2 else "random",
                               verbose_step=10 ** 9)
    return np.nan_to_num(df.values)


def main(rep="c58_8"):
    t0 = time.time()
    C, X_true, U_true = load_hu(rep)
    C = np.asarray(C, float)
    from scipy.spatial import distance_matrix
    truth_dist = distance_matrix(X_true, X_true)
    print(f"{rep}: {C.shape[0]} beads", flush=True)

    S = cos_sim(C); np.fill_diagonal(S, 0.0)
    E_umap = umap_embed(C)
    print(f"  UMAP  posRMSE={position_rmse(E_umap, X_true):7.1f}  "
          f"pairRMSE={pairwise_rmse(E_umap, X_true):7.1f}  ({time.time()-t0:.0f}s)", flush=True)

    D = fit_curve(S, truth_dist)

    # baseline: full soft-weighted (the paper's "accumap + fitted curve")
    Xr = accumap_solve(E_umap, D, S, cycles=10)
    print(f"  accumap full-weight (soft, all pairs)  "
          f"posRMSE={position_rmse(Xr, X_true):7.1f}  pairRMSE={pairwise_rmse(Xr, X_true):7.1f}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    for k in (15, 25, 50):
        mask = build_knn_mask(S, k)
        W = np.where(mask, S, 0.0)
        Xr = accumap_solve(E_umap, D, W, cycles=10)
        print(f"  kNN-accumap k={k:<3d} (hard, {100*mask.sum()/(C.shape[0]*(C.shape[0]-1)):.2f}% pairs)  "
              f"posRMSE={position_rmse(Xr, X_true):7.1f}  pairRMSE={pairwise_rmse(Xr, X_true):7.1f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "c58_8")
