"""Run UMAP / accumap / Variant 1 / Route 2 / Route 1 on the Hu et al.
experimental data, scored against the in-situ-sequencing ground truth.

This is the accumap-vs-UMAP-on-real-data comparison the reviewers asked for
(M7), plus a first test of whether the marginalized model (Variant 1) and the
self-consistent fit (Route 2) hold up on real counts.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "analysis_scripts"))

from common import cos_sim, per_bead_mse, jaccard_curve, summary  # noqa: E402
from real_data import load_hu  # noqa: E402
from run_benchmark import umap_embed, fitted_distance_matrix, accumap_run  # noqa: E402
from route2_surrogate import route2  # noqa: E402
from route1_poisson import route1  # noqa: E402
from variant1_marginal import variant1  # noqa: E402

OUT = os.path.join(HERE, "results_real")
os.makedirs(OUT, exist_ok=True)


def run_rep(rep):
    t0 = time.time()
    print(f"=== {rep} ===")
    C, X_true, U_true = load_hu(rep)
    print(f"  C {C.shape}  UMI/bead {C.sum(1).mean():.0f}")
    truth_dist = distance_matrix(X_true, X_true)
    S = cos_sim(C)
    np.fill_diagonal(S, 1.0)

    cache = os.path.join(OUT, f"{rep}_cache.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        E_umap, E_perf, E_fit = z["E_umap"], z["E_perf"], z["E_fit"]
    else:
        print("  UMAP ...")
        E_umap = umap_embed(C)
        print("  accumap (perfect) ...")
        E_perf = accumap_run(E_umap, truth_dist, S, cycles=1)
        print("  accumap (fitted, imaging calib) ...")
        Dfit = fitted_distance_matrix(S, truth_dist)
        E_fit = accumap_run(E_umap, Dfit, S, cycles=10)
        np.savez(cache, E_umap=E_umap, E_perf=E_perf, E_fit=E_fit)

    umap_mse = per_bead_mse(E_umap, X_true)

    print("  Route 2 (UMAP init) ...")
    E_r2, _ = route2(C, n_outer=6, init=E_umap)
    print("  Variant 1 (UMAP init) ...")
    E_v1, v1p = variant1(C, init=E_umap)
    print("  Variant 1 (cMDS init) ...")
    E_v1c, v1cp = variant1(C, init=None)
    print("  Route 1 (oracle imaging senders) ...")
    E_r1o, _, _ = route1(C, __import__("common").SIGMA, X0=E_umap.copy(),
                         oracle_U=U_true, outer=10, steps_x=700, steps_u=0, lr=3.0)

    runs = [
        ("UMAP", E_umap), ("accumap (perfect)", E_perf), ("accumap (fitted)", E_fit),
        ("Route 2 (UMAP init)", E_r2), ("Variant 1 (UMAP init)", E_v1),
        ("Variant 1 (cMDS init)", E_v1c), ("Route 1 oracle U", E_r1o),
    ]
    rows, jacs = [], {}
    for name, E in runs:
        row, jac = summary(name, E, X_true, umap_mse=umap_mse)
        rows.append({"rep": rep, **row})
        jacs[name] = jac
        print(f"    {name:22s} meanMSE={row['mse_mean']:9.1f} "
              f"J@10={row['jaccard_10']:.3f} beatsUMAP={row.get('frac_beats_umap', np.nan):.3f}")

    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{rep}_table.csv"), index=False)
    pd.DataFrame({k: pd.Series(v) for k, v in jacs.items()}).to_csv(
        os.path.join(OUT, f"{rep}_jaccard.csv"))
    json.dump({"variant1_umap": v1p, "variant1_cmds": v1cp},
              open(os.path.join(OUT, f"{rep}_params.json"), "w"), indent=2)
    print(f"  {rep} done in {time.time() - t0:.0f}s")
    return rows


if __name__ == "__main__":
    reps = sys.argv[1:] or ["c58_8"]
    allrows = []
    for r in reps:
        allrows += run_rep(r)
    tab = pd.DataFrame(allrows)
    tab.to_csv(os.path.join(OUT, "all_table.csv"), index=False)
    print("\n" + tab.to_string(index=False))
