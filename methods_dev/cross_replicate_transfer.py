"""Does the similarity->distance curve need imaging calibration for EVERY
experiment, or is it a one-time fix reusable across runs of the same assay?

Fits the curve ONCE on c58_8 (the replicate used throughout this project),
freezes it, and applies it UNMODIFIED to the other three Hu et al. 2025
replicates (c58_4, c58_5, c58_16) -- vs. a curve freshly calibrated on each
replicate's own imaging truth (the paper's current per-dataset procedure).
If the frozen curve does nearly as well as a fresh fit, calibration is a
one-time cost per assay/protocol, not a per-experiment requirement. Also
speaks to reviewer M8 (the paper currently asserts, without evidence, that
trends are "consistent across all replicates").

Uses the hard k-NN-restricted solver (k=5, the established optimum from the
real-data investigation: within ~5% of UMAP, ~85x faster than the paper's
full-weight solve) so 4 replicates x 2 curves x 1 solve each is tractable --
this test is about the CURVE, not re-litigating full-weight vs restricted.

Usage:  python cross_replicate_transfer.py
"""
import os
import sys
import time
import json
import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis_scripts"))
from accumap import correct_locations  # noqa: E402
sys.path.insert(0, os.path.dirname(__file__))
from common import cos_sim  # noqa: E402
from decontaminate import umap_embed, diff_func  # noqa: E402
from position_error import position_rmse, pairwise_rmse  # noqa: E402
from real_data import load_hu  # noqa: E402
from knn_accumap_real import build_knn_mask  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

SOURCE = "c58_8"
TARGETS = ["c58_4", "c58_5", "c58_16"]
K = 5


def fit_curve_params(S, truth_dist, seed=42):
    rng = np.random.RandomState(seed)
    bars = rng.randint(0, S.shape[0], 100)
    sv, dv = S[bars].ravel(), truth_dist[bars].ravel()
    m = sv > 0
    popt, _ = curve_fit(diff_func, sv[m], dv[m],
                        p0=(-5.389e-3, 1.104, 296.2, 783.9), maxfev=20000)
    return popt


def apply_curve(S, popt):
    D = diff_func(np.clip(S, 1e-9, 1 - 1e-9), *popt)
    D = np.nan_to_num(D, nan=3000.0, posinf=3000.0, neginf=10.0)
    D[S <= 0] = 3000.0
    D[D <= 10] = 10.0
    np.fill_diagonal(D, 0.0)
    return D


def knn_accumap_solve(E_init, D, S, k=K, cycles=10):
    mask = build_knn_mask(S, k)
    W = np.where(mask, S, 0.0)
    df = pd_dataframe(E_init)
    for it in range(cycles):
        df = correct_locations(df, D, weight=W, gauss_seidel=True,
                               correct_order="sorted" if it < cycles // 2 else "random",
                               verbose_step=10 ** 9)
    return np.nan_to_num(df.values)


def pd_dataframe(E):
    import pandas as pd
    return pd.DataFrame(np.asarray(E, float), columns=["xcoord", "ycoord"])


def main():
    t0 = time.time()
    data = {}
    for rep in [SOURCE] + TARGETS:
        C, X_true, U_true = load_hu(rep)
        C = np.asarray(C, float)
        S = cos_sim(C); np.fill_diagonal(S, 0.0)
        truth_dist = distance_matrix(X_true, X_true)
        E_umap = umap_embed(C)
        data[rep] = dict(C=C, S=S, truth_dist=truth_dist, E_umap=E_umap, X_true=X_true)
        umap_pos = position_rmse(E_umap, X_true)
        umap_pair = pairwise_rmse(E_umap, X_true)
        print(f"[{rep}] {C.shape[0]} beads  UMAP posRMSE={umap_pos:.1f} "
              f"pairRMSE={umap_pair:.1f}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\nFitting curve ONCE on {SOURCE} ...", flush=True)
    popt_source = fit_curve_params(data[SOURCE]["S"], data[SOURCE]["truth_dist"])
    print(f"  source curve params: {popt_source}", flush=True)

    results = {}
    for rep in TARGETS:
        S, truth_dist, E_umap, X_true = (data[rep]["S"], data[rep]["truth_dist"],
                                         data[rep]["E_umap"], data[rep]["X_true"])

        # (1) current practice: fresh curve calibrated on THIS replicate's own truth
        popt_fresh = fit_curve_params(S, truth_dist)
        D_fresh = apply_curve(S, popt_fresh)
        X_fresh = knn_accumap_solve(E_umap, D_fresh, S)
        pos_fresh, pair_fresh = position_rmse(X_fresh, X_true), pairwise_rmse(X_fresh, X_true)

        # (2) transferred: FROZEN curve from c58_8, no refitting, no truth used from `rep`
        D_transfer = apply_curve(S, popt_source)
        X_transfer = knn_accumap_solve(E_umap, D_transfer, S)
        pos_transfer, pair_transfer = position_rmse(X_transfer, X_true), pairwise_rmse(X_transfer, X_true)

        umap_pos, umap_pair = position_rmse(E_umap, X_true), pairwise_rmse(E_umap, X_true)

        print(f"\n[{rep}]  UMAP              posRMSE={umap_pos:7.1f}  pairRMSE={umap_pair:7.1f}")
        print(f"[{rep}]  fresh curve       posRMSE={pos_fresh:7.1f}  pairRMSE={pair_fresh:7.1f}")
        print(f"[{rep}]  transferred curve posRMSE={pos_transfer:7.1f}  pairRMSE={pair_transfer:7.1f}"
              f"  ({time.time()-t0:.0f}s)", flush=True)

        results[rep] = dict(umap=(umap_pos, umap_pair), fresh=(pos_fresh, pair_fresh),
                            transfer=(pos_transfer, pair_transfer),
                            popt_fresh=list(popt_fresh))

    json.dump({"source": SOURCE, "popt_source": list(popt_source), "k": K,
               "results": results}, open(os.path.join(OUT, "cross_replicate_transfer.json"), "w"),
              indent=2)
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
