"""How much barcode crosstalk (read misassignment) can reconstruction tolerate?

Start from clean Poisson counts on the New_Simulation kernel.  Then reassign a
fraction mu of every bead's reads to a random other receiver (keeping the
sender) -- the barcode-collision model: profiles shrink toward the global mean,
destroying the distinctiveness that encodes position.

For each mu: UMAP and accumap (curve fit on 100 truth-calibrated beads), scored
by pairwise-distance RMSE and J@10, plus the raw ordinal signal
(Spearman(S, -d), kNN-10 Jaccard) to anchor against c58_8 (rho ~ 0.6,
kNN-Jac ~ 0.285).
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis_scripts"))
from accumap import correct_locations  # noqa: E402
from common import cos_sim, jaccard_curve  # noqa: E402
from position_error import pairwise_rmse  # noqa: E402

SIGMA = 123.0 / np.sqrt(8.0 * np.log(2.0))
RADIUS, N, UMI, SEED = 1500.0, 8000, 2500, 42
MUS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]


def _disc(rng, n):
    r = np.sqrt(rng.uniform(size=n)) * RADIUS
    t = rng.uniform(size=n) * 2 * np.pi
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def simulate_clean(seed=SEED):
    rng = np.random.RandomState(seed)
    X = _disc(rng, N)
    U = _disc(rng, N)
    d = cdist(X, U)
    p = np.exp(-d ** 2 / (2 * SIGMA ** 2))
    p /= p.sum(1, keepdims=True)
    C = rng.poisson(UMI * p)
    return C.astype(np.float64), X, U


def crosstalk(C, mu, rng, local_um=None, X=None):
    """Count-level: each read leaves its cell w.p. mu; leavers are redistributed
    uniformly over receivers (same sender column)."""
    N_, O = C.shape
    moved = rng.binomial(C.astype(np.int64), mu)      # reads leaving each cell
    kept = C - moved
    col_moved = moved.sum(axis=0).astype(np.int64)    # leavers per sender
    add = np.zeros_like(C)
    if local_um is None:
        pu = np.full(N_, 1.0 / N_)
        for o in np.nonzero(col_moved)[0]:
            add[:, o] = rng.multinomial(col_moved[o], pu)
    else:
        # leavers from cell (b,o) go to a random receiver within local_um of b
        rr, oo = np.nonzero(moved)
        m = moved[rr, oo]
        for k in range(len(rr)):
            near = np.where(np.sum((X - X[rr[k]]) ** 2, 1) < local_um ** 2)[0]
            dest = near[rng.randint(len(near), size=int(m[k]))] if len(near) else [rr[k]]
            np.add.at(add[:, oo[k]], dest, 1)
    return kept + add


def diff_func(x, A, B, C, D):
    return D - C * np.log(-(A + B - x) / (A - x))


def accumap_fit(C, S, truth_dist, E_umap):
    rng = np.random.RandomState(42)
    bars = rng.randint(0, S.shape[0], 100)
    s = S[bars].ravel(); dd = truth_dist[bars].ravel()
    m = s > 0
    try:
        popt, _ = curve_fit(diff_func, s[m], dd[m],
                            p0=(-5.389e-3, 1.104, 296.2, 783.9), maxfev=20000)
        D = diff_func(np.clip(S, 1e-9, 1 - 1e-9), *popt)
    except Exception:
        D = np.full_like(S, 1500.0)
    D = np.nan_to_num(D, nan=3000.0, posinf=3000.0, neginf=10.0)
    D[S <= 0] = 3000.0
    D[D <= 10] = 10.0
    np.fill_diagonal(D, 0.0)
    df = pd.DataFrame(E_umap, columns=["xcoord", "ycoord"])
    for k in range(10):
        df = correct_locations(df, D, weight=S, gauss_seidel=True,
                               correct_order="sorted" if k < 5 else "random",
                               verbose_step=10 ** 9)
    return np.nan_to_num(df.values)


def knn_jac(S, td, k=10):
    n = S.shape[0]
    sim_nn = np.argpartition(-S, k + 1, axis=1)[:, :k + 1]
    tru_nn = np.argpartition(td + np.eye(n) * 1e18, k, axis=1)[:, :k]
    j = np.empty(n)
    for b in range(n):
        j[b] = 1.0 - len((set(sim_nn[b]) - {b}) & set(tru_nn[b])) / \
            len((set(sim_nn[b]) - {b}) | set(tru_nn[b]))
    return float(j.mean())


def umap_embed(C, epochs=500):
    import umap
    E = umap.UMAP(random_state=42, metric="cosine", n_neighbors=25, min_dist=0.99,
                  n_components=2, n_epochs=epochs).fit_transform(np.log1p(C))
    E = E - E.mean(0)
    return E * RADIUS * 2 / np.max(distance_matrix(E, E))


def main(local=False):
    C0, X, U = simulate_clean()
    td = distance_matrix(X, X)
    rng = np.random.RandomState(7)
    tag = "local" if local else "random"
    print(f"crosstalk model: {tag} reassignment\n")
    print(f"{'mu':>6} {'rho':>6} {'kNNjac':>7} {'UMAP':>7} {'Uj10':>6} "
          f"{'accu':>7} {'Aj10':>6}")
    iu = np.triu_indices(N, 1)
    samp = np.random.RandomState(0).choice(len(iu[0]), 2_000_000, replace=False)
    for mu in MUS:
        t0 = time.time()
        C = C0 if mu == 0 else crosstalk(C0, mu, rng,
                                         local_um=150.0 if local else None, X=X)
        S = cos_sim(C); np.fill_diagonal(S, 0.0)
        rho = spearmanr(S[iu][samp], -td[iu][samp]).correlation
        kj = knn_jac(S, td)
        Eu = umap_embed(C)
        u_r, u_j = pairwise_rmse(Eu, X), jaccard_curve(Eu, X)[10]
        Ea = accumap_fit(C, S, td, Eu)
        a_r, a_j = pairwise_rmse(Ea, X), jaccard_curve(Ea, X)[10]
        print(f"{mu:>6.2f} {rho:>6.2f} {kj:>7.3f} {u_r:>7.1f} {u_j:>6.3f} "
              f"{a_r:>7.1f} {a_j:>6.3f}   ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(local="--local" in sys.argv)
