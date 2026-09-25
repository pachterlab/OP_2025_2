"""How much per-pair count overdispersion can reconstruction tolerate?

Start from the clean New_Simulation kernel; replace the multinomial draw with
Gamma-Poisson (negative binomial) counts of a given size r (r -> inf is
Poisson/clean; real c58_8 is r ~ 0.008).  For each r, score:

  * UMAP                          (moving topological baseline)
  * accumap, curve fit on 100 truth-calibrated beads, 10 cycles
  * generative MAP for X with the CORRECT r and oracle sender positions
    (best-case: is the geometric information still there?)

Reports pairwise-distance RMSE and J@10, plus the raw ordinal signal
(Spearman(S, -d), kNN-10 Jaccard) and the empirically recovered r-hat.
"""
import os
import sys
import time
import numpy as np
import torch
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
RS = [1e9, 100, 30, 10, 3, 1.0, 0.3, 0.1, 0.03, 0.01, 0.003]


def _disc(rng, n):
    r = np.sqrt(rng.uniform(size=n)) * RADIUS
    t = rng.uniform(size=n) * 2 * np.pi
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def simulate(r, seed=SEED):
    rng = np.random.RandomState(seed)
    X = _disc(rng, N)
    U = _disc(rng, N)
    d = cdist(X, U)
    p = np.exp(-d ** 2 / (2 * SIGMA ** 2))
    p /= p.sum(1, keepdims=True)
    lam = UMI * p                                     # per-pair mean rate
    if r >= 1e8:
        C = rng.poisson(lam)
    else:
        gam = rng.gamma(shape=r, scale=1.0 / r, size=lam.shape)   # mean 1, var 1/r
        C = rng.poisson(lam * gam)
    return C.astype(np.float64), X, U


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
    import pandas as pd
    df = pd.DataFrame(E_umap, columns=["xcoord", "ycoord"])
    for k in range(10):
        df = correct_locations(df, D, weight=S, gauss_seidel=True,
                               correct_order="sorted" if k < 5 else "random",
                               verbose_step=10 ** 9)
    return df.values


def gen_map_oracle(C, U, r, X0, steps=1200, lr=2.0, chunk=1500, device="cuda"):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    B = C.shape[0]
    Ct = torch.tensor(C, dtype=torch.float32, device=dev)
    Ut = torch.tensor(U, dtype=torch.float32, device=dev)
    X = torch.tensor(X0, dtype=torch.float32, device=dev).clone().requires_grad_(True)
    logA = torch.tensor(0.0, device=dev, requires_grad=True)
    inv2s2 = 1.0 / (2 * SIGMA ** 2)
    rt = float(np.clip(r, 1e-3, 500.0))          # r>=500 ~ Poisson; avoids float blowup
    lg_rt = torch.lgamma(torch.tensor(rt, device=dev))
    opt = torch.optim.Adam([X, logA], lr=lr)
    for t in range(steps):
        opt.zero_grad(set_to_none=True)
        tot = X.new_zeros(()); dn = 0.0
        A_ = torch.exp(logA)
        for i in range(0, B, chunk):
            sl = slice(i, i + chunk)
            d2 = torch.cdist(X[sl], Ut) ** 2
            mu = A_ * torch.exp(-d2 * inv2s2) + 1e-6
            c = Ct[sl]
            ll = (torch.lgamma(c + rt) - lg_rt - torch.lgamma(c + 1)
                  + rt * (np.log(rt) - torch.log(rt + mu))
                  + c * torch.log(mu / (rt + mu)))
            tot = tot - ll.sum(); dn += c.numel()
        (tot / dn).backward()
        opt.step()
        if (t + 1) % 400 == 0:
            opt.param_groups[0]["lr"] *= 0.5
    return X.detach().cpu().numpy()


def knn_jac(S, td, k=10):
    n = S.shape[0]
    sim_nn = np.argpartition(-S, k + 1, axis=1)[:, :k + 1]
    tru_nn = np.argpartition(td + np.eye(n) * 1e18, k, axis=1)[:, :k]
    j = np.empty(n)
    for b in range(n):
        a = set(sim_nn[b]) - {b}
        c = set(tru_nn[b])
        j[b] = 1.0 - len(a & c) / len(a | c)
    return float(j.mean())


def emp_r(C, X, U, n=400_000):
    rng = np.random.RandomState(1)
    bi = rng.randint(0, C.shape[0], n); oi = rng.randint(0, C.shape[1], n)
    d = np.sqrt(((X[bi] - U[oi]) ** 2).sum(1))
    c = C[bi, oi]
    b = np.digitize(d, np.quantile(d, np.linspace(0, 1, 41)))
    num = den = 0.0
    for k in np.unique(b):
        m = b == k
        if m.sum() < 200:
            continue
        mu, v = c[m].mean(), c[m].var()
        if mu > 1e-3:
            num += (v - mu) * mu ** 2; den += mu ** 4
    return den / max(num, 1e-12)


def umap_embed(C, epochs=500):
    import umap
    E = umap.UMAP(random_state=42, metric="cosine", n_neighbors=25, min_dist=0.99,
                  n_components=2, n_epochs=epochs, verbose=False).fit_transform(np.log1p(C))
    E = E - E.mean(0)
    E = E * RADIUS * 2 / np.max(distance_matrix(E, E))
    return E


def main():
    print(f"{'r_inj':>8} {'r_hat':>8} {'rho':>6} {'kNNjac':>7} "
          f"{'UMAP':>7} {'Uj10':>6} {'accu':>7} {'Aj10':>6} {'genMAP':>7} {'Gj10':>6}")
    for r in RS:
        t0 = time.time()
        C, X, U = simulate(r)
        td = distance_matrix(X, X)
        S = cos_sim(C); np.fill_diagonal(S, 0.0)
        iu = np.triu_indices(len(S), 1)
        samp = np.random.RandomState(0).choice(len(iu[0]), 2_000_000, replace=False)
        rho = spearmanr(S[iu][samp], -td[iu][samp]).correlation
        knn = knn_jac(S, td, k=10)
        rhat = emp_r(C, X, U)

        Eu = umap_embed(C)
        u_rmse, u_j = pairwise_rmse(Eu, X), jaccard_curve(Eu, X)[10]
        Ea = accumap_fit(C, S, td, Eu)
        a_rmse, a_j = pairwise_rmse(Ea, X), jaccard_curve(Ea, X)[10]
        Eg = gen_map_oracle(C, U, r, Eu)
        g_rmse, g_j = pairwise_rmse(Eg, X), jaccard_curve(Eg, X)[10]

        rlbl = "inf" if r >= 1e8 else f"{r:g}"
        print(f"{rlbl:>8} {rhat:8.3f} {rho:6.2f} {knn:7.3f} "
              f"{u_rmse:7.1f} {u_j:6.3f} {a_rmse:7.1f} {a_j:6.3f} "
              f"{g_rmse:7.1f} {g_j:6.3f}   ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
