"""Reviewer M3 deep half: is the empirical 4-parameter curve
    g(s) = delta - gamma * log((alpha+beta-s)/(s-alpha))
(main.tex, "Fitting Distances") actually needed, or does the
physically-motivated inverse of the Gaussian-diffusion-overlap model,

    d(s) = 2*sigma*sqrt(-log(s))          (from S ~= exp(-d^2 / 4 sigma^2),
                                            the convolution of two receivers'
                                            Gaussian kernels, sigma known from
                                            the assay FWHM, ZERO free params)

do as well? Tested two ways on the paper's own New_Simulation setup
(sigma=52um, 10k/10k beads), same calibration procedure (100 beads, seed 42)
so only the functional FORM differs, everything else (weight=S, UMAP init,
10 cycles, same floor/threshold/cap post-processing) held fixed:

  1. Curve-fit quality: held-out prediction error of each form (empirical is
     FIT to the 100 calibration beads; physical has no fitting step at all)
     against a disjoint sample of bead pairs never used for calibration.
  2. Downstream reconstruction: run accumap to convergence with each D-tilde
     as the target and compare final MSE / Jaccard, across n replicate
     simulations (paired significance test), matching the M5/M6 protocol.

Usage:  python manuscript_m3_deep.py [n_seeds]
"""
import os
import sys
import json
import time
import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import curve_fit
from scipy.stats import ttest_rel, wilcoxon

sys.path.insert(0, os.path.dirname(__file__))
from common import SIGMA, simulate, cos_sim, per_bead_mse, jaccard_curve  # noqa: E402
from run_benchmark import umap_embed, diff_func, accumap_run  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

F_THRESH, D_MAX, D_MIN = 0.0, 3000.0, 10.0


def postprocess(D, S):
    D = np.nan_to_num(D, nan=D_MAX, posinf=D_MAX, neginf=D_MIN)
    D[S <= F_THRESH] = D_MAX
    D[D <= D_MIN] = D_MIN
    np.fill_diagonal(D, 0.0)
    return D


def empirical_curve(S, truth_dist, cal_idx):
    sv, dv = S[cal_idx].ravel(), truth_dist[cal_idx].ravel()
    m = sv > 0
    popt, _ = curve_fit(diff_func, sv[m], dv[m],
                        p0=(-5.389e-3, 1.104, 296.2, 783.9), maxfev=20000)
    D = diff_func(np.clip(S, 1e-9, 1 - 1e-9), *popt)
    return postprocess(D, S), popt


def physical_curve(S, sigma=SIGMA):
    D = 2.0 * sigma * np.sqrt(np.maximum(-np.log(np.clip(S, 1e-12, 1.0)), 0.0))
    return postprocess(D, S)


def held_out_eval(S, truth_dist, cal_idx, popt, n_beads):
    """RMSE of each functional form on bead pairs never used for calibration."""
    rng = np.random.RandomState(123)
    remaining = np.setdiff1d(np.arange(n_beads), cal_idx)
    holdout = rng.choice(remaining, size=min(100, len(remaining)), replace=False)
    sv, dv = S[holdout].ravel(), truth_dist[holdout].ravel()
    m = sv > 0
    d_emp = np.clip(diff_func(sv[m], *popt), D_MIN, D_MAX)
    d_phys = np.clip(2.0 * SIGMA * np.sqrt(np.maximum(-np.log(sv[m]), 0.0)), D_MIN, D_MAX)
    rmse_emp = float(np.sqrt(np.mean((d_emp - dv[m]) ** 2)))
    rmse_phys = float(np.sqrt(np.mean((d_phys - dv[m]) ** 2)))
    return rmse_emp, rmse_phys


def run_one_seed(seed, verbose=True):
    t0 = time.time()
    C, X_true, U_true = simulate(seed=seed)
    truth_dist = distance_matrix(X_true, X_true)
    S = cos_sim(C); np.fill_diagonal(S, 1.0)
    B = S.shape[0]

    cal_rng = np.random.RandomState(42)     # matches the paper's calibration seed
    cal_idx = cal_rng.randint(0, B, 100)

    D_emp, popt = empirical_curve(S, truth_dist, cal_idx)
    D_phys = physical_curve(S)

    rmse_emp, rmse_phys = held_out_eval(S, truth_dist, cal_idx, popt, B)

    E_umap = umap_embed(C)
    E_fit_emp = accumap_run(E_umap, D_emp, S, cycles=10)
    E_fit_phys = accumap_run(E_umap, D_phys, S, cycles=10)

    out = {}
    for name, E in [("UMAP", E_umap), ("accumap (empirical curve)", E_fit_emp),
                    ("accumap (physical inverse)", E_fit_phys)]:
        mse = per_bead_mse(E, X_true)
        jac = jaccard_curve(E, X_true)
        out[name] = dict(mse_mean=float(mse.mean()), jac=jac)
    out["rmse_emp_holdout"] = rmse_emp
    out["rmse_phys_holdout"] = rmse_phys
    if verbose:
        print(f"seed {seed}: UMAP={out['UMAP']['mse_mean']:.1f}  "
              f"empirical={out['accumap (empirical curve)']['mse_mean']:.1f}  "
              f"physical={out['accumap (physical inverse)']['mse_mean']:.1f}  "
              f"| holdout RMSE emp={rmse_emp:.1f} phys={rmse_phys:.1f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
    return out


def main(n_seeds=8):
    t0 = time.time()
    all_results = {s: run_one_seed(s) for s in range(n_seeds)}

    methods = ["UMAP", "accumap (empirical curve)", "accumap (physical inverse)"]
    mse_by_method = {m: np.array([all_results[s][m]["mse_mean"] for s in range(n_seeds)])
                     for m in methods}
    rmse_emp = np.array([all_results[s]["rmse_emp_holdout"] for s in range(n_seeds)])
    rmse_phys = np.array([all_results[s]["rmse_phys_holdout"] for s in range(n_seeds)])

    print(f"\n=== held-out curve-fit RMSE, n={n_seeds} ===")
    print(f"  empirical (4-param fit): mean={rmse_emp.mean():.1f}  std={rmse_emp.std(ddof=1):.1f}")
    print(f"  physical  (0-param):     mean={rmse_phys.mean():.1f}  std={rmse_phys.std(ddof=1):.1f}")
    t_rmse, p_rmse = ttest_rel(rmse_emp, rmse_phys)
    print(f"  paired t-test empirical vs physical RMSE: t={t_rmse:.3f} p={p_rmse:.4g}")

    print(f"\n=== downstream reconstruction MSE, n={n_seeds} ===")
    for m in methods:
        v = mse_by_method[m]
        print(f"  {m:30s} mean={v.mean():8.1f}  std={v.std(ddof=1):7.1f}  sem={v.std(ddof=1)/np.sqrt(n_seeds):6.1f}")
    t1, p1 = ttest_rel(mse_by_method["accumap (physical inverse)"], mse_by_method["accumap (empirical curve)"])
    w1 = wilcoxon(mse_by_method["accumap (physical inverse)"], mse_by_method["accumap (empirical curve)"])
    print(f"\n  paired t-test physical vs empirical (downstream MSE): t={t1:.3f} p={p1:.4g}")
    print(f"  Wilcoxon signed-rank                                 : W={w1.statistic:.3f} p={w1.pvalue:.4g}")

    xs = sorted(next(iter(all_results.values()))["UMAP"]["jac"].keys())
    jac_stats = {m: {"mean": [], "std": []} for m in methods}
    for m in methods:
        for x in xs:
            vals = np.array([all_results[s][m]["jac"][x] for s in range(n_seeds)])
            jac_stats[m]["mean"].append(float(vals.mean()))
            jac_stats[m]["std"].append(float(vals.std(ddof=1)))

    json.dump({"n_seeds": n_seeds,
               "mse_by_method": {m: v.tolist() for m, v in mse_by_method.items()},
               "rmse_emp_holdout": rmse_emp.tolist(), "rmse_phys_holdout": rmse_phys.tolist(),
               "jac_xs": xs, "jac_stats": jac_stats,
               "ttest_rmse_emp_vs_phys": {"t": t_rmse, "p": p_rmse},
               "ttest_downstream_phys_vs_emp": {"t": t1, "p": p1},
               "wilcoxon_downstream_phys_vs_emp": {"W": w1.statistic, "p": w1.pvalue}},
              open(os.path.join(OUT, "m3_deep_results.json"), "w"), indent=2)

    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
