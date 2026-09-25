"""Count-only decontamination of barcode crosstalk / ambient contamination.

Model:  C_b ~= (1 - a_b) * s_b  +  a_b * n_b * g
  g   = ambient / soup profile (sender marginals of C)
  s_b = bead b's true peaked profile over senders
  a_b = per-bead contamination fraction

EM with a k-NN-graph-smoothed native profile (sparse S-graph adjacency denoises
s_b and breaks the trivial a_b = 0 fixed point).  No embedding, no ground truth.

Usage:
  python decontaminate.py sim              # crosstalk simulation sweep
  python decontaminate.py real c58_8       # apply to experimental data
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis_scripts"))
from accumap import correct_locations  # noqa: E402
from common import cos_sim, jaccard_curve  # noqa: E402
from position_error import pairwise_rmse  # noqa: E402

SIGMA = 123.0 / np.sqrt(8.0 * np.log(2.0))
RADIUS, UMI = 1500.0, 2500


# ---------------------------------------------------------------- decontam ---
def decontaminate(C, n_iter=12, k=20, a_max=0.7, chunk=2500, verbose=True):
    C = np.asarray(C, dtype=np.float64)
    B, O = C.shape
    nb = C.sum(1)
    g = C.sum(0); g = g / g.sum()
    gO = g[None, :]

    S = cos_sim(C); np.fill_diagonal(S, 0.0)
    idx = np.argpartition(-S, k + 1, axis=1)[:, :k + 1]
    rows = np.repeat(np.arange(B), k + 1)
    A = csr_matrix((np.full(B * (k + 1), 1.0 / (k + 1)), (rows, idx.ravel())),
                   shape=(B, B))

    s = C / np.maximum(nb[:, None], 1.0)
    a = np.full(B, 0.05)
    s_sm = None
    for it in range(n_iter):
        s_sm = A @ s
        s_sm /= np.maximum(s_sm.sum(1, keepdims=True), 1e-12)
        contam = np.zeros(B)
        for i0 in range(0, B, chunk):
            sl = slice(i0, i0 + chunk)
            num = a[sl, None] * gO
            den = (1 - a[sl, None]) * s_sm[sl] + num + 1e-12
            z = num / den
            contam[sl] = (C[sl] * z).sum(1)
            nat = C[sl] * (1 - z)
            s[sl] = nat / np.maximum(nat.sum(1, keepdims=True), 1e-12)
        a = np.clip(contam / np.maximum(nb, 1.0), 0.0, a_max)
        if verbose:
            print(f"    EM {it:2d}: mean a={a.mean():.3f} med a={np.median(a):.3f}")

    Chat = np.empty_like(C)
    for i0 in range(0, B, chunk):
        sl = slice(i0, i0 + chunk)
        den = (1 - a[sl, None]) * s_sm[sl] + a[sl, None] * gO + 1e-12
        z = (a[sl, None] * gO) / den
        Chat[sl] = C[sl] * (1 - z)
    return Chat, a


# ------------------------------------------------------------- sim helpers ---
def _disc(rng, n):
    r = np.sqrt(rng.uniform(size=n)) * RADIUS
    t = rng.uniform(size=n) * 2 * np.pi
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def simulate_clean(n=5000, seed=42):
    rng = np.random.RandomState(seed)
    X, U = _disc(rng, n), _disc(rng, n)
    d = cdist(X, U)
    p = np.exp(-d ** 2 / (2 * SIGMA ** 2))
    p /= p.sum(1, keepdims=True)
    return rng.poisson(UMI * p).astype(np.float64), X, U


def crosstalk(C, mu, seed=7):
    rng = np.random.RandomState(seed)
    moved = np.minimum(rng.poisson(C * mu), C.astype(np.int64))
    kept = C - moved
    col = moved.sum(0).astype(np.int64)
    add = np.zeros_like(C)
    pu = np.full(C.shape[0], 1.0 / C.shape[0])
    for o in np.nonzero(col)[0]:
        add[:, o] = rng.multinomial(col[o], pu)
    return kept + add


# ------------------------------------------------------------ common eval ---
def diff_func(x, A, B, C, D):
    return D - C * np.log(-(A + B - x) / (A - x))


def accumap_fit(C, S, truth_dist, E_umap):
    rng = np.random.RandomState(42)
    bars = rng.randint(0, S.shape[0], 100)
    sv, dv = S[bars].ravel(), truth_dist[bars].ravel()
    m = sv > 0
    try:
        popt, _ = curve_fit(diff_func, sv[m], dv[m],
                            p0=(-5.389e-3, 1.104, 296.2, 783.9), maxfev=20000)
        D = diff_func(np.clip(S, 1e-9, 1 - 1e-9), *popt)
    except Exception:
        D = np.full_like(S, 1500.0)
    D = np.nan_to_num(D, nan=3000.0, posinf=3000.0, neginf=10.0)
    D[S <= 0] = 3000.0
    D[D <= 10] = 10.0
    np.fill_diagonal(D, 0.0)
    df = pd.DataFrame(E_umap, columns=["xcoord", "ycoord"])
    for it in range(10):
        df = correct_locations(df, D, weight=S, gauss_seidel=True,
                               correct_order="sorted" if it < 5 else "random",
                               verbose_step=10 ** 9)
    return np.nan_to_num(df.values)


def umap_embed(C, epochs=400):
    import umap
    E = umap.UMAP(random_state=42, metric="cosine", n_neighbors=25, min_dist=0.99,
                  n_components=2, n_epochs=epochs).fit_transform(np.log1p(C))
    E = E - E.mean(0)
    return E * RADIUS * 2 / np.max(distance_matrix(E, E))


def score(C, td, X):
    S = cos_sim(C); np.fill_diagonal(S, 0.0)
    iu = np.triu_indices(len(S), 1)
    ns = min(1_000_000, len(iu[0]))
    samp = np.random.RandomState(0).choice(len(iu[0]), ns, replace=False)
    rho = spearmanr(S[iu][samp], -td[iu][samp]).correlation
    Eu = umap_embed(C)
    Ea = accumap_fit(C, S, td, Eu)
    return (rho, pairwise_rmse(Eu, X), jaccard_curve(Eu, X)[10],
            pairwise_rmse(Ea, X), jaccard_curve(Ea, X)[10])


# ----------------------------------------------------------------- drivers ---
def main_sim():
    C0, X, U = simulate_clean()
    td = distance_matrix(X, X)
    print(f"{'mu':>5} {'rho0':>6} {'rho1':>6} {'a_hat':>7} "
          f"{'U0':>6} {'a0':>7} {'U1':>6} {'a1':>7}  {'Uj0':>5} {'Aj0':>5} {'Uj1':>5} {'Aj1':>5}")
    for mu in [0.0, 0.02, 0.05, 0.1, 0.2]:
        t0 = time.time()
        C = C0 if mu == 0 else crosstalk(C0, mu)
        r0, u0, uj0, aa0, aj0 = score(C, td, X)
        Chat, ah = decontaminate(C, verbose=False)
        r1, u1, uj1, aa1, aj1 = score(Chat, td, X)
        print(f"{mu:>5.2f} {r0:>6.2f} {r1:>6.2f} {ah.mean():>7.3f} "
              f"{u0:>6.1f} {aa0:>7.1f} {u1:>6.1f} {aa1:>7.1f}  "
              f"{uj0:>5.2f} {aj0:>5.2f} {uj1:>5.2f} {aj1:>5.2f}  ({time.time()-t0:.0f}s)",
              flush=True)


def main_real(rep="c58_8"):
    from real_data import load_hu
    C, X_true, U_true = load_hu(rep)
    td = distance_matrix(X_true, X_true)
    print(f"{rep}: C {C.shape}")

    print("  --- before ---")
    r0, u0, uj0, a0, aj0 = score(C, td, X_true)
    print(f"    rho={r0:.3f}  UMAP pairRMSE={u0:.1f} J@10={uj0:.3f}  "
          f"accu pairRMSE={a0:.1f} J@10={aj0:.3f}")

    t0 = time.time()
    Chat, ah = decontaminate(C, verbose=True)
    print(f"  decontam {time.time()-t0:.0f}s: a_hat mean={ah.mean():.3f} "
          f"median={np.median(ah):.3f} p90={np.percentile(ah, 90):.3f} "
          f"frac>0.1={np.mean(ah > 0.1):.2f}")

    print("  --- after ---")
    r1, u1, uj1, a1, aj1 = score(Chat, td, X_true)
    print(f"    rho={r1:.3f}  UMAP pairRMSE={u1:.1f} J@10={uj1:.3f}  "
          f"accu pairRMSE={a1:.1f} J@10={aj1:.3f}")

    np.savez(os.path.join(os.path.dirname(__file__), "results_real",
                          f"{rep}_decontam.npz"), a_hat=ah)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "real":
        main_real(a[1] if len(a) > 1 else "c58_8")
    else:
        main_sim()
