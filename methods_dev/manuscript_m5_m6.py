"""Reviewer M5 (UMAP-init sensitivity) + M6 remainder (error bars / p-values
across simulation replicates), on the paper's own New_Simulation setup
(sigma=52um, 10k/10k beads), reusing run_benchmark.py's exact pipeline.

M5: does accumap's correction depend on starting from UMAP specifically?
Compare three initializations for the "fitted distances" accumap run
(the paper's actual deployed method): UMAP (current), random-uniform-in-disc
(structure-free), classical MDS of the fitted distance matrix (a different
but still informative global start, not derived from UMAP's topology).

M6: independent simulation replicates (new bead positions + new multinomial
count draw each time, seed=0..N_SEEDS-1; the 100-calibration-bead selection
inside fitted_distance_matrix stays at its own fixed seed=42, matching the
paper's actual calibration procedure) -> mean +/- std for MSE and the
Jaccard-vs-X curve, plus a paired test (UMAP vs accumap-fitted) on MSE
across replicates.

Usage:  python manuscript_m5_m6.py [n_seeds]
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.stats import ttest_rel, wilcoxon

sys.path.insert(0, os.path.dirname(__file__))
from common import (SIGMA, RADIUS, simulate, cos_sim, classical_mds,  # noqa: E402
                    per_bead_mse, jaccard_curve)
from run_benchmark import umap_embed, fitted_distance_matrix, accumap_run  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)


def random_init(n, radius, seed):
    rng = np.random.RandomState(seed)
    r = np.sqrt(rng.uniform(size=n)) * radius
    t = rng.uniform(size=n) * 2 * np.pi
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def run_one_seed(seed, verbose=True):
    t0 = time.time()
    C, X_true, U_true = simulate(seed=seed)
    truth_dist = distance_matrix(X_true, X_true)
    S = cos_sim(C); np.fill_diagonal(S, 1.0)

    E_umap = umap_embed(C)
    Dfit = fitted_distance_matrix(S, truth_dist)

    E_fit_umapinit = accumap_run(E_umap, Dfit, S, cycles=10)
    E_fit_randinit = accumap_run(random_init(len(X_true), RADIUS, seed=1000 + seed),
                                 Dfit, S, cycles=10)
    E_fit_cmdsinit = accumap_run(classical_mds(Dfit), Dfit, S, cycles=10)

    out = {}
    for name, E in [("UMAP", E_umap),
                    ("accumap (UMAP init)", E_fit_umapinit),
                    ("accumap (random init)", E_fit_randinit),
                    ("accumap (cMDS init)", E_fit_cmdsinit)]:
        mse = per_bead_mse(E, X_true)
        jac = jaccard_curve(E, X_true)
        out[name] = dict(mse_mean=float(mse.mean()), jac=jac)
    if verbose:
        line = "  ".join(f"{k}={v['mse_mean']:.1f}" for k, v in out.items())
        print(f"seed {seed}: {line}  ({time.time()-t0:.0f}s)", flush=True)
    return out


def main(n_seeds=8):
    t0 = time.time()
    all_results = {}
    for seed in range(n_seeds):
        all_results[seed] = run_one_seed(seed)

    methods = list(next(iter(all_results.values())).keys())
    mse_by_method = {m: np.array([all_results[s][m]["mse_mean"] for s in range(n_seeds)])
                     for m in methods}

    print("\n=== M6: mean +/- std MSE across", n_seeds, "replicates ===")
    for m in methods:
        v = mse_by_method[m]
        print(f"  {m:26s} mean={v.mean():8.1f}  std={v.std(ddof=1):7.1f}  "
              f"sem={v.std(ddof=1)/np.sqrt(n_seeds):6.1f}  values={np.round(v,1).tolist()}")

    umap_v = mse_by_method["UMAP"]
    fit_v = mse_by_method["accumap (UMAP init)"]
    t_stat, t_p = ttest_rel(fit_v, umap_v)
    w_stat, w_p = wilcoxon(fit_v, umap_v)
    print(f"\n  paired t-test  accumap(UMAP init) vs UMAP: t={t_stat:.3f} p={t_p:.4g}")
    print(f"  Wilcoxon signed-rank                       : W={w_stat:.3f} p={w_p:.4g}")

    print("\n=== M5: init sensitivity (mean MSE across", n_seeds, "replicates) ===")
    for m in ["accumap (UMAP init)", "accumap (random init)", "accumap (cMDS init)"]:
        v = mse_by_method[m]
        print(f"  {m:26s} mean={v.mean():8.1f}  std={v.std(ddof=1):7.1f}")

    ru = ttest_rel(mse_by_method["accumap (random init)"], mse_by_method["accumap (UMAP init)"])
    cu = ttest_rel(mse_by_method["accumap (cMDS init)"], mse_by_method["accumap (UMAP init)"])
    print(f"\n  paired t-test  random-init vs UMAP-init accumap: t={ru[0]:.3f} p={ru[1]:.4g}")
    print(f"  paired t-test  cMDS-init   vs UMAP-init accumap: t={cu[0]:.3f} p={cu[1]:.4g}")

    # Jaccard-vs-X curves: mean +/- std across replicates
    xs = sorted(next(iter(all_results[0].values()))["jac"].keys())
    jac_stats = {m: {"mean": [], "std": []} for m in methods}
    for m in methods:
        for x in xs:
            vals = np.array([all_results[s][m]["jac"][x] for s in range(n_seeds)])
            jac_stats[m]["mean"].append(float(vals.mean()))
            jac_stats[m]["std"].append(float(vals.std(ddof=1)))

    json.dump({"n_seeds": n_seeds,
               "mse_by_method": {m: v.tolist() for m, v in mse_by_method.items()},
               "jac_xs": xs, "jac_stats": jac_stats,
               "ttest_accumap_vs_umap": {"t": t_stat, "p": t_p},
               "wilcoxon_accumap_vs_umap": {"W": w_stat, "p": w_p},
               "ttest_randinit_vs_umapinit": {"t": ru[0], "p": ru[1]},
               "ttest_cmdsinit_vs_umapinit": {"t": cu[0], "p": cu[1]}},
              open(os.path.join(OUT, "m5_m6_results.json"), "w"), indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        for m in methods:
            mean = np.array(jac_stats[m]["mean"])
            std = np.array(jac_stats[m]["std"])
            ax[0].plot(xs, mean, label=m, lw=2)
            ax[0].fill_between(xs, mean - std, mean + std, alpha=0.2)
        ax[0].set_xlabel("X nearest neighbours"); ax[0].set_ylabel("mean Jaccard distance")
        ax[0].legend(fontsize=8); ax[0].set_title(f"Jaccard vs X, mean+/-std, n={n_seeds} reps")

        pos = np.arange(len(methods))
        means = [mse_by_method[m].mean() for m in methods]
        sems = [mse_by_method[m].std(ddof=1) / np.sqrt(n_seeds) for m in methods]
        ax[1].bar(pos, means, yerr=sems, capsize=4)
        ax[1].set_xticks(pos); ax[1].set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
        ax[1].set_ylabel("mean per-bead MSE ($\\mu$m$^2$)")
        ax[1].set_title(f"M5/M6: mean+/-SEM, n={n_seeds} reps")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "m5_m6_figure.png"), dpi=150)
    except Exception as e:
        print("plot failed:", e)

    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
