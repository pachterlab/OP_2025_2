"""Position-displacement RMSE (full Procrustes: translation + rotation +
reflection + scale) alongside the pairwise-distance RMSE we've been reporting,
for the key methods on real c58_8.
"""
import os
import sys
import numpy as np
from scipy.spatial import distance_matrix

from real_data import load_hu
from common import cos_sim, jaccard_curve


def align(recon, truth):
    T = truth - truth.mean(0)
    R = recon - recon.mean(0)
    U, s, Vt = np.linalg.svd(R.T @ T)
    Rot = U @ Vt
    Rr = R @ Rot
    scale = np.trace(Rr.T @ T) / np.trace(Rr.T @ Rr)
    return Rr * scale, T


def position_rmse(recon, truth):
    a, T = align(recon, truth)
    return float(np.sqrt(np.mean(np.sum((a - T) ** 2, axis=1))))


def pairwise_rmse(recon, truth):
    a, T = align(recon, truth)
    return float(np.sqrt(np.mean((distance_matrix(T, T) - distance_matrix(a, a)) ** 2)))


def main(rep="c58_8", extra=False):
    C, X_true, U_true = load_hu(rep)
    cache = os.path.join(os.path.dirname(__file__), "results_real", f"{rep}_cache.npz")
    z = np.load(cache)
    runs = {"UMAP": z["E_umap"],
            "accumap (perfect distances)": z["E_perf"],
            "accumap (fitted curve)": z["E_fit"]}

    if extra:
        from route2_surrogate import route2
        from variant1_marginal import variant1
        print("  (running Route 2 / Variant 1 ...)", flush=True)
        runs["Route 2 (self-consistent)"] = route2(C, n_outer=6, init=z["E_umap"])[0]
        runs["Variant 1 (marginal model)"] = variant1(C, init=z["E_umap"])[0]

    field = float((X_true.max(0) - X_true.min(0)).mean())
    print(f"{rep}: {C.shape[0]} receiver beads, field span ~{field:.0f} um\n")
    print(f"{'method':30s} {'posRMSE(um)':>12s} {'pairRMSE(um)':>12s} {'J@10':>7s}")
    for name, E in runs.items():
        pos = position_rmse(E, X_true)
        pair = pairwise_rmse(E, X_true)
        jac = jaccard_curve(E, X_true).get(10, np.nan)
        print(f"{name:30s} {pos:12.1f} {pair:12.1f} {jac:7.3f}")


if __name__ == "__main__":
    a = sys.argv[1:]
    rep = a[0] if a and not a[0].startswith("-") else "c58_8"
    main(rep, extra=("--extra" in a))
