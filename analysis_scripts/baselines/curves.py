"""The original accumap (baseline): similarity-to-distance curve fitting and the weighted-MDS solve loop,
including the variant restricted to each bead's k most similar neighbors."""
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.optimize import curve_fit

from .core import correct_locations
from .simulation import RADIUS


def umap_embed(C):
    """UMAP embedding on cosine similarity of log1p(C), rescaled to the
    physical field size. Matches the simulation benchmark's UMAP setup
    (10,000 epochs); see decontaminate.umap_embed for the faster
    (400-epoch default) real-data-diagnostic variant."""
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
    """Fit the empirical similarity-to-distance curve on 100 randomly-chosen
    calibration beads (matches the paper's actual calibration procedure)."""
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


def build_knn_mask(S, k):
    """Symmetric union k-NN mask: bead pair (i,j) is kept if j is among i's
    k nearest neighbours by similarity, OR i is among j's."""
    B = S.shape[0]
    idx = np.argpartition(-S, k + 1, axis=1)[:, :k + 1]
    mask = np.zeros((B, B), dtype=bool)
    rows = np.repeat(np.arange(B), k + 1)
    mask[rows, idx.ravel()] = True
    mask |= mask.T
    np.fill_diagonal(mask, False)
    return mask


def fit_curve(S, truth_dist, seed=42):
    """Same fit as fitted_distance_matrix, with an explicit calibration seed
    (used by the real-data scripts, which reuse this exact call shape)."""
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
    """Same as accumap_run, with a `cycles=10` default (matches the real-data
    scripts' call shape)."""
    df = pd.DataFrame(E_init, columns=["xcoord", "ycoord"])
    for it in range(cycles):
        df = correct_locations(df, D, weight=W, gauss_seidel=True,
                               correct_order="sorted" if it < cycles // 2 else "random",
                               verbose_step=10 ** 9)
    return np.nan_to_num(df.values)
