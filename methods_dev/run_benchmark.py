"""Benchmark the alternating-optimization methods against the current pipeline.

Runs, all on the seed-42 New_Simulation dataset (sigma = 52 um, 10k/10k beads):

  * UMAP                         - current standard
  * accumap (perfect distances)  - ground-truth distance matrix, 1 cycle
  * accumap (fitted distances)   - cosine-similarity curve fit, 10 cycles
  * Route 2  (surrogate block-coordinate + IRLS weights, physical g, no truth)
  * Route 1  (MAP-EM Poisson, joint sender positions)
  * Route 1  (MAP-EM Poisson, oracle sender positions)  - best-case bound

Writes methods_dev/results/{table.csv, jaccard.csv, compare.png}.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.optimize import curve_fit

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "analysis_scripts"))
from accumap import correct_locations  # noqa: E402

from common import (SIGMA, RADIUS, SEED, simulate, cos_sim, classical_mds,  # noqa: E402
                    per_bead_mse, summary)
from route2_surrogate import route2  # noqa: E402
from route1_poisson import route1  # noqa: E402
from variant1_marginal import variant1  # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)


def umap_embed(C):
    import umap
    reducer = umap.UMAP(random_state=42, metric="cosine", n_neighbors=25,
                        min_dist=0.99, low_memory=False, n_components=2,
                        n_epochs=10000, learning_rate=1)
    E = reducer.fit_transform(np.log1p(C))
    E = E - E.mean(axis=0)
    E = E * RADIUS * 2.0 / np.max(distance_matrix(E, E))
    return E


def diff_func(x, A, B, C, D):
    return D - C * np.log(-(A + B - x) / (A - x))


def fitted_distance_matrix(S, truth_dist):
    rng = np.random.RandomState(42)
    bars = rng.randint(0, S.shape[0], 100)
    s = S[bars].flatten()
    d = truth_dist[bars].flatten()
    keep = s > 0
    popt, _ = curve_fit(diff_func, s[keep], d[keep],
                        p0=(-5.389e-03, 1.104, 296.2, 783.9), maxfev=20000)
    D = diff_func(np.clip(S, 1e-9, 1 - 1e-9), *popt)
    D = np.nan_to_num(D, nan=3000.0, posinf=3000.0, neginf=10.0)
    D[S <= 0] = 3000.0
    D[D <= 10] = 10.0
    np.fill_diagonal(D, 0.0)
    return D


def accumap_run(init, target, weight, cycles):
    df = pd.DataFrame(np.asarray(init), columns=["xcoord", "ycoord"])
    for k in range(cycles):
        df = correct_locations(df, target, weight=weight, gauss_seidel=True,
                               correct_order="sorted" if k < max(1, cycles // 2)
                               else "random", verbose_step=10 ** 9)
    return df.values


def main():
    t0 = time.time()
    cache = os.path.join(OUT, "cache.npz")
    print("simulating ...")
    C, X_true, U_true = simulate(seed=SEED)
    truth_dist = distance_matrix(X_true, X_true)
    S = cos_sim(C)
    np.fill_diagonal(S, 1.0)

    if os.path.exists(cache):
        print("loading cached UMAP / accumap embeddings ...")
        z = np.load(cache)
        E_umap, E_perf, E_fit = z["E_umap"], z["E_perf"], z["E_fit"]
        umap_mse = per_bead_mse(E_umap, X_true)
    else:
        print("UMAP ...")
        E_umap = umap_embed(C)
        umap_mse = per_bead_mse(E_umap, X_true)
        print("accumap (perfect distances) ...")
        E_perf = accumap_run(E_umap, truth_dist, S, cycles=1)
        print("accumap (fitted distances) ...")
        Dfit = fitted_distance_matrix(S, truth_dist)
        E_fit = accumap_run(E_umap, Dfit, S, cycles=10)
        np.savez(cache, E_umap=E_umap, E_perf=E_perf, E_fit=E_fit)

    print("Route 2 (surrogate, UMAP init) ...")
    E_r2, r2_hist = route2(C, n_outer=6, init=E_umap)
    print("Route 2 (surrogate, classical-MDS init) ...")
    E_r2c, r2c_hist = route2(C, n_outer=6, init=None)

    print("Variant 1 (marginal model, classical-MDS init) ...")
    E_v1c, v1c_p = variant1(C, init=None)
    print("Variant 1 (marginal model, UMAP init) ...")
    E_v1, v1_p = variant1(C, init=E_umap)

    X0 = E_r2.copy()                          # warm-start from Route 2
    Ssend = cos_sim(C.T)
    np.fill_diagonal(Ssend, 1.0)
    U0 = classical_mds(1.0 - Ssend)
    U0 *= np.sqrt((X0 ** 2).sum(1).mean()) / np.sqrt((U0 ** 2).sum(1).mean())
    print("Route 1 (joint senders) ...")
    E_r1, U_r1, r1_hist = route1(C, SIGMA, X0=X0, U0=U0, oracle_U=None,
                                 outer=20, steps_x=500, steps_u=500, lr=3.0)
    print("Route 1 (oracle senders) ...")
    E_r1o, _, r1o_hist = route1(C, SIGMA, X0=classical_mds(1.0 - S), oracle_U=U_true,
                                outer=8, steps_x=800, steps_u=0, lr=3.0)

    runs = [
        ("UMAP", E_umap),
        ("accumap (perfect)", E_perf),
        ("accumap (fitted)", E_fit),
        ("Route 2 (UMAP init)", E_r2),
        ("Route 2 (cMDS init)", E_r2c),
        ("Variant 1 (UMAP init)", E_v1),
        ("Variant 1 (cMDS init)", E_v1c),
        ("Route 1 joint", E_r1),
        ("Route 1 oracle U", E_r1o),
    ]
    rows, jacs = [], {}
    for name, E in runs:
        row, jac = summary(name, E, X_true, umap_mse=umap_mse)
        rows.append(row)
        jacs[name] = jac
        print(f"  {name:22s} meanMSE={row['mse_mean']:8.1f} "
              f"medMSE={row['mse_median']:8.1f} J@10={row['jaccard_10']:.3f} "
              f"beatsUMAP={row.get('frac_beats_umap', float('nan')):.3f}")

    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(OUT, "table.csv"), index=False)
    jdf = pd.DataFrame({k: pd.Series(v) for k, v in jacs.items()})
    jdf.index.name = "X"
    jdf.to_csv(os.path.join(OUT, "jaccard.csv"))
    json.dump({"route2_umap": r2_hist, "route2_cmds": r2c_hist,
               "route1_joint": r1_hist, "route1_oracle": r1o_hist,
               "variant1_umap": v1_p, "variant1_cmds": v1c_p},
              open(os.path.join(OUT, "hist.json"), "w"), indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ax[0].bar(tab["method"], tab["mse_mean"])
        ax[0].set_ylabel("mean per-bead MSE ($\\mu$m$^2$)")
        ax[0].tick_params(axis="x", rotation=30)
        for name in jdf.columns:
            ax[1].plot(jdf.index, jdf[name], label=name, lw=2)
        ax[1].set_xlabel("X nearest neighbours")
        ax[1].set_ylabel("mean Jaccard distance")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "compare.png"), dpi=150)
    except Exception as e:  # pragma: no cover
        print("plot failed:", e)

    print(tab.to_string(index=False))
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
