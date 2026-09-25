"""Does softening the TREATMENT of kept edges (not just restricting WHICH
edges are kept) close more of the accumap-vs-UMAP gap on real data?

The hard k-NN restriction (k=5) fixed which pairs enter the objective. This
tests the other half: within that same k=5 graph, downweight individual
noisy edges via IRLS/Huber reweighting (a single noisy near-neighbour can no
longer dominate a bead's position update) instead of committing every kept
edge to its exact curve-predicted distance at full weight. Standard Huber
IRLS: after each solve, weight_i = 1 if |residual_i| <= delta else
delta/|residual_i|, delta = 1.345 * MAD(residuals) (the usual robust scale
estimate, ~95% efficiency under Gaussian residuals).

Usage:  python knn_accumap_robust.py
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
from common import cos_sim  # noqa: E402
from decontaminate import umap_embed  # noqa: E402
from position_error import position_rmse, pairwise_rmse  # noqa: E402
from real_data import load_hu  # noqa: E402
from knn_accumap_real import build_knn_mask, fit_curve  # noqa: E402


def robust_knn_accumap(E_init, D, S, mask, n_outer=4, cycles_per_outer=10,
                       fixed_delta=True, X_baseline=None, verbose=True):
    """fixed_delta=True: compute delta ONCE -- from X_baseline's residuals if
    given (the plain hard-kNN fit, never touched by reweighting), else from
    the first round's own fit -- and hold it fixed across outer rounds.
    Avoids the self-referential moving-target runaway of recomputing delta
    from each round's own (already-reweighted) fit."""
    W = np.where(mask, S, 0.0)
    X = np.asarray(E_init, float).copy()
    delta = None
    if fixed_delta and X_baseline is not None:
        resid0 = np.abs(distance_matrix(X_baseline, X_baseline) - D)[mask]
        mad0 = np.median(np.abs(resid0 - np.median(resid0)))
        delta = 1.345 * mad0 * 1.4826
        if verbose:
            print(f"    delta seeded from baseline fit: {delta:.1f}um", flush=True)
    for outer in range(n_outer):
        df = pd.DataFrame(X, columns=["xcoord", "ycoord"])
        for it in range(cycles_per_outer):
            df = correct_locations(df, D, weight=W, gauss_seidel=True,
                                   correct_order="sorted" if it < cycles_per_outer // 2 else "random",
                                   verbose_step=10 ** 9)
        X = np.nan_to_num(df.values)

        DX = distance_matrix(X, X)
        resid = np.abs(DX - D)
        r_kept = resid[mask]
        if delta is None or not fixed_delta:
            mad = np.median(np.abs(r_kept - np.median(r_kept)))
            delta = 1.345 * mad * 1.4826   # MAD -> robust sigma, then Huber's 1.345 sigma
        huber_w = np.where(resid <= delta, 1.0, delta / np.maximum(resid, 1e-9))
        W = np.where(mask, S * huber_w, 0.0)
        if verbose:
            frac_downweighted = float((huber_w[mask] < 1.0).mean())
            print(f"    outer {outer}: delta={delta:.1f}um  "
                  f"frac_edges_downweighted={frac_downweighted:.3f}", flush=True)
    return X


def main(rep="c58_8", k=5):
    t0 = time.time()
    C, X_true, U_true = load_hu(rep)
    C = np.asarray(C, float)
    S = cos_sim(C); np.fill_diagonal(S, 0.0)
    truth_dist = distance_matrix(X_true, X_true)
    E_umap = umap_embed(C)
    print(f"[{rep}] UMAP posRMSE={position_rmse(E_umap, X_true):.1f} "
          f"pairRMSE={pairwise_rmse(E_umap, X_true):.1f}  ({time.time()-t0:.0f}s)", flush=True)

    D = fit_curve(S, truth_dist)   # already the full postprocessed distance matrix

    mask = build_knn_mask(S, k)

    # baseline: hard k-NN, plain weight (the established k=5 result)
    df = pd.DataFrame(E_umap, columns=["xcoord", "ycoord"])
    W0 = np.where(mask, S, 0.0)
    for it in range(10):
        df = correct_locations(df, D, weight=W0, gauss_seidel=True,
                               correct_order="sorted" if it < 5 else "random", verbose_step=10 ** 9)
    X_hard = np.nan_to_num(df.values)
    print(f"[{rep}] kNN-accumap k={k} (hard, plain weight)  "
          f"posRMSE={position_rmse(X_hard, X_true):.1f}  pairRMSE={pairwise_rmse(X_hard, X_true):.1f}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    # test: hard k-NN + IRLS/Huber-reweighted kept edges, FIXED delta (computed
    # once from the plain baseline's own residuals, held constant -- no
    # self-referential moving target)
    print(f"[{rep}] robust (Huber IRLS, fixed delta) reweighting ...", flush=True)
    X_robust = robust_knn_accumap(X_hard, D, S, mask, n_outer=4, cycles_per_outer=10,
                                  fixed_delta=True, X_baseline=X_hard)
    print(f"[{rep}] kNN-accumap k={k} (hard + robust reweight, fixed delta)  "
          f"posRMSE={position_rmse(X_robust, X_true):.1f}  pairRMSE={pairwise_rmse(X_robust, X_true):.1f}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    import sys as _sys
    main(k=int(_sys.argv[1]) if len(_sys.argv) > 1 else 5)
