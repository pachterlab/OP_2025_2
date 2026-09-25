"""Route 2: block-coordinate descent on the surrogate objective, with a
*self-consistently fitted* similarity -> distance map (no ground-truth
distances).

    L(theta, X) = sum_{i<j} w_ij ( ||x_i - x_j|| - g(S_ij) )^2

Each outer iteration:
  1. rescale X so its short-range distances match the physically valid
     local relation  d ~ 2 sigma sqrt(-log s)   (this is the ONE external
     scale anchor; sigma is known from the assay FWHM);
  2. fit g as a monotone-decreasing function of s to the current (S, D(X))
     pairs (binned medians + isotonic regression);
  3. update heteroscedastic weights  w_ij = 1 / (Var[resid | S_ij] + delta);
  4. weighted-MDS X-block (reuses the Rust solver).

Initialised from classical MDS of (1 - S).
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis_scripts"))
from accumap import correct_locations  # noqa: E402

from common import SIGMA, cos_sim, classical_mds  # noqa: E402


def _monotone_map(s, d, n_bins=300):
    """Fit a non-increasing g(s) ~ d from scattered (s, d); return callable."""
    lo, hi = np.quantile(s, [0.001, 0.999])
    edges = np.linspace(lo, hi, n_bins + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])
    which = np.clip(np.digitize(s, edges) - 1, 0, n_bins - 1)
    med = np.full(n_bins, np.nan)
    for k in range(n_bins):
        m = which == k
        if m.sum() > 50:
            med[k] = np.median(d[m])
    ok = ~np.isnan(med)
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    fit = iso.fit_transform(mid[ok], med[ok])
    xk, yk = mid[ok], fit

    def g(ss):
        return np.interp(np.clip(ss, xk[0], xk[-1]), xk, yk)

    return g


def route2(C, n_outer=5, sigma=SIGMA, init=None, verbose=True):
    S = cos_sim(C)
    np.fill_diagonal(S, 1.0)
    n = len(S)
    triu = np.triu_indices(n, k=1)
    s_flat = S[triu]

    # short-range pairs where d ~ 2 sigma sqrt(-log s) is trustworthy
    hi = s_flat > np.quantile(s_flat, 0.90)
    d_phys_hi = 2.0 * sigma * np.sqrt(np.maximum(-np.log(np.clip(s_flat[hi], 1e-9, 1.0)), 0.0))
    anchor = np.median(d_phys_hi)

    X = np.asarray(init, dtype=float).copy() if init is not None else classical_mds(1.0 - S)

    bins = np.linspace(0.0, 1.0, 51)
    mid = 0.5 * (bins[1:] + bins[:-1])
    hist = []
    for it in range(n_outer):
        DX = distance_matrix(X, X)
        dx_flat = DX[triu]

        scale = anchor / max(np.median(dx_flat[hi]), 1e-9)
        X = X * scale
        DX = DX * scale
        dx_flat = dx_flat * scale

        g = _monotone_map(s_flat, dx_flat)
        Dt = g(S)
        np.fill_diagonal(Dt, 0.0)

        r_flat = dx_flat - g(s_flat)
        which = np.clip(np.digitize(s_flat, bins) - 1, 0, len(mid) - 1)
        var = np.full(len(mid), np.nan)
        for k in range(len(mid)):
            m = which == k
            if m.sum() > 200:
                var[k] = r_flat[m].var()
        ok = ~np.isnan(var)
        W = 1.0 / (np.interp(S, mid[ok], var[ok]) + np.nanmedian(var))
        np.fill_diagonal(W, 0.0)

        Xdf = correct_locations(
            pd.DataFrame(X, columns=["xcoord", "ycoord"]), Dt, weight=W,
            gauss_seidel=True, correct_order="sorted", verbose_step=10 ** 9,
        )
        X = Xdf.values

        stress = float(np.mean(W[triu] * (distance_matrix(X, X)[triu] - Dt[triu]) ** 2))
        hist.append({"iter": it, "scale": float(scale), "stress": stress})
        if verbose:
            print(f"  route2 iter {it}: scale={scale:.3g} stress={stress:.4g}")

    return X, hist
