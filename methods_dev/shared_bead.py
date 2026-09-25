"""Shared-bead modality: every bead is BOTH a sender and a receiver
(Abdulraouf 2024 / Liao 2024), as opposed to the distinct sender/receiver
Slide-tag/Hu 2025 setup the rest of methods_dev targets.

Generative model (bead b at x_b in a uniform disc of radius R):

    lambda_{b b'} = q_b * a_{b'} * K_sigma(||x_b - x_b'||)      (b != b')
    C_{b .}      ~ Multinomial( round(UMI * q_b), pi_b )   with pi_b ~ lambda_{b .}
                   (or independent Poisson / NB per cell)

  a_b  ~ LogNormal, per-bead emission strength     (CV a_cv)
  q_b  ~ LogNormal, per-bead capture efficiency    (CV q_cv)
  K    ~ Gaussian by default; cauchy / gen-gaussian for kernel misspecification
  self term C_bb dropped.

Why this case is structurally easier than distinct sender/receiver: U = X, so
there is no unobserved second point set.  Iterative proportional fitting
(Sinkhorn) strips q_b and a_{b'}, leaving an approximately symmetric kernel
matrix whose Gaussian inversion is a usable metric distance matrix with the
ONE external anchor sigma (assay FWHM) -- no ground-truth calibration.

  python shared_bead.py smoke          # fast plumbing check (B=500)
  python shared_bead.py bench          # main comparison, a few regimes
  python shared_bead.py sweep-sigma
  python shared_bead.py sweep-corrupt  # heterogeneity / crosstalk / kernel
"""
import os
import sys
import time
import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist, pdist, squareform

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis_scripts"))
from common import (SIGMA, cos_sim, classical_mds, procrustes_align,  # noqa: E402
                    per_bead_mse, jaccard_curve)

RADIUS = 1500.0
UMI = 2500


# --------------------------------------------------------------- simulate ---
def _disc(rng, n, radius):
    r = np.sqrt(rng.uniform(size=n)) * radius
    t = rng.uniform(size=n) * 2 * np.pi
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def _kernel(d, sigma, kind, p):
    if kind == "gaussian":
        return np.exp(-d ** 2 / (2 * sigma ** 2))
    if kind == "gen_gaussian":                      # p<2 heavier shoulder
        c = sigma * np.sqrt(2.0)
        return np.exp(-(np.abs(d) / c) ** p)
    if kind == "cauchy":                            # lam scaled so FWHM matches
        lam = sigma * np.sqrt(8 * np.log(2)) / 2.0
        return 1.0 / (1.0 + (d / lam) ** 2) ** 2
    raise ValueError(kind)


def simulate_shared(B=2000, sigma=SIGMA, umi=UMI, radius=RADIUS, seed=42,
                    a_cv=0.0, q_cv=0.0, kernel="gaussian", gg_p=1.4,
                    edge_narrow=0.0, noise="multinomial", nb_r=np.inf,
                    crosstalk_mu=0.0, drop_diag=True):
    rng = np.random.RandomState(seed)
    X = _disc(rng, B, radius)
    d = distance_matrix(X, X)

    sig = np.full(B, sigma)
    if edge_narrow > 0:                             # sigma shrinks toward edge
        rr = np.sqrt((X ** 2).sum(1)) / radius
        sig = sigma * (1.0 - edge_narrow * rr)
    sig_pair = 0.5 * (sig[:, None] + sig[None, :])

    a = np.exp(rng.normal(0, a_cv, B)) if a_cv > 0 else np.ones(B)
    q = np.exp(rng.normal(0, q_cv, B)) if q_cv > 0 else np.ones(B)
    a /= a.mean(); q /= q.mean()

    K = _kernel(d, sig_pair, kernel, gg_p)
    if drop_diag:
        np.fill_diagonal(K, 0.0)
    lam = q[:, None] * a[None, :] * K

    if noise == "multinomial":
        pi = lam / lam.sum(1, keepdims=True)
        C = np.empty((B, B))
        budget = np.maximum(np.round(umi * q).astype(int), 1)
        for b in range(B):
            C[b] = rng.multinomial(budget[b], pi[b])
    elif noise == "poisson":
        mu = lam * (umi / lam.sum(1, keepdims=True).mean())
        if np.isfinite(nb_r):
            mu = rng.gamma(nb_r, mu / nb_r)
        C = rng.poisson(mu).astype(float)
    else:
        raise ValueError(noise)

    if crosstalk_mu > 0:
        C = _crosstalk(C, crosstalk_mu, rng)
    if drop_diag:
        np.fill_diagonal(C, 0.0)
    return C, X, dict(a=a, q=q, sigma=sig, d=d)


def _crosstalk(C, mu, rng):
    """Reassign a fraction mu of every bead's reads to a uniformly random
    other capturing bead (barcode-collision / ambient model)."""
    C = C.astype(np.int64)
    moved = np.minimum(rng.poisson(C * mu), C)
    kept = C - moved
    col = moved.sum(0)
    add = np.zeros_like(C)
    pu = np.full(C.shape[0], 1.0 / C.shape[0])
    for o in np.nonzero(col)[0]:
        add[:, o] = rng.multinomial(col[o], pu)
    return (kept + add).astype(float)


# ------------------------------------------------------- normalise / dist ---
def ipf_normalize(C, n_iter=60, eps=1e-9):
    """Sinkhorn / iterative proportional fitting on the symmetrised matrix ->
    (approximately) doubly stochastic P.  Strips per-bead q_b and a_{b'}."""
    M = C + C.T
    np.fill_diagonal(M, 0.0)
    r = np.ones(M.shape[0])
    cc = np.ones(M.shape[0])
    for _ in range(n_iter):
        r = 1.0 / np.maximum((M * cc[None, :]).sum(1), eps)
        cc = 1.0 / np.maximum((M * r[:, None]).sum(0), eps)
    P = r[:, None] * M * cc[None, :]
    return 0.5 * (P + P.T)


def geodesic_distance(Dnear, S, k=12, big=3000.0):
    """Fill in long-range distances by shortest paths over a kNN graph whose
    edges are the trustworthy short-range Sinkhorn distances."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    B = Dnear.shape[0]
    idx = np.argpartition(-S, k + 1, axis=1)[:, :k + 1]
    rows = np.repeat(np.arange(B), k + 1)
    w = Dnear[rows, idx.ravel()]
    ok = np.isfinite(w) & (w < big) & (w > 0)
    G = csr_matrix((w[ok], (rows[ok], idx.ravel()[ok])), shape=(B, B))
    G = G.maximum(G.T)
    D = shortest_path(G, method="D", directed=False)
    D[~np.isfinite(D)] = big
    D = np.minimum(D, big)
    np.fill_diagonal(D, 0.0)
    return D


def sinkhorn_distance(P, sigma, big=3000.0):
    """Invert the Gaussian kernel on a Sinkhorn-normalised P.
    d = sigma * sqrt( -2 log( P / P_ref ) ),  P_ref = global max off-diagonal."""
    Pod = P.copy()
    np.fill_diagonal(Pod, 0.0)
    ref = Pod.max()
    with np.errstate(divide="ignore", invalid="ignore"):
        D = sigma * np.sqrt(np.maximum(-2.0 * np.log(Pod / ref), 0.0))
    D[~np.isfinite(D)] = big
    D[Pod <= 0] = big
    D = np.minimum(D, big)
    np.fill_diagonal(D, 0.0)
    return 0.5 * (D + D.T)


# ------------------------------------------------------------- methods -----
def umap_embed(C):
    import umap
    E = umap.UMAP(random_state=42, metric="cosine", n_neighbors=25,
                  min_dist=0.99, n_components=2, n_epochs=2000,
                  learning_rate=1).fit_transform(np.log1p(C))
    E = E - E.mean(0)
    return E * RADIUS * 2.0 / np.max(distance_matrix(E, E))


def accumap_run(init, target, weight, cycles=10):
    import pandas as pd
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis_scripts"))
    from accumap import correct_locations
    df = pd.DataFrame(np.asarray(init, float), columns=["xcoord", "ycoord"])
    for k in range(cycles):
        df = correct_locations(df, target, weight=weight, gauss_seidel=True,
                               correct_order="sorted" if k < max(1, cycles // 2)
                               else "random", verbose_step=10 ** 9)
    return np.nan_to_num(df.values)


def pos_rmse(recon, truth):
    a = procrustes_align(recon, truth)
    t = truth - truth.mean(0)
    return float(np.sqrt(np.mean(np.sum((a - t) ** 2, axis=1))))


def pair_rmse(recon, truth):
    a = procrustes_align(recon, truth)
    t = truth - truth.mean(0)
    return float(np.sqrt(np.mean((distance_matrix(t, t) - distance_matrix(a, a)) ** 2)))


# --------------------------------------------------------------- runner ----
def evaluate(C, X, sigma, methods, tag=""):
    B = C.shape[0]
    S = cos_sim(C); np.fill_diagonal(S, 0.0)
    P = ipf_normalize(C)
    Dsk = sinkhorn_distance(P, sigma)
    td = distance_matrix(X, X)
    E_umap = umap_embed(C)

    # sampling / distance-recovery diagnostics
    m = np.triu_indices(B, 1)
    nn = np.sort(td + np.eye(B) * 1e9, axis=1)[:, 0].mean()
    near = td[m] < 2 * sigma
    rho_all = np.corrcoef(Dsk[m], td[m])[0, 1]
    rho_near = np.corrcoef(Dsk[m][near], td[m][near])[0, 1] if near.sum() > 50 else np.nan
    frac_big = float(np.mean(Dsk[m] >= 2999.0))
    print(f"[{tag}]  B={B}  sigma={sigma:.0f}  reads/bead~{C.sum()/B:.0f}  "
          f"NN~{nn:.0f}um ({nn/sigma:.2f} sigma)  "
          f"corr(Dsk,true) all={rho_all:.3f} near={rho_near:.3f}  "
          f"frac@max={frac_big:.2f}")

    out = {}

    def rec(name, E):
        out[name] = (pos_rmse(E, X), pair_rmse(E, X),
                     jaccard_curve(E, X).get(10, np.nan))

    if "umap" in methods:
        rec("UMAP", E_umap)
    if "cmds_sinkhorn" in methods:
        rec("classical-MDS(Sinkhorn dist)", classical_mds(Dsk))
    if "accu_sinkhorn" in methods:
        Sw = S.copy(); np.fill_diagonal(Sw, 0.0)
        rec("accumap(Sinkhorn dist, UMAP init)",
            accumap_run(E_umap, Dsk, Sw, cycles=10))
    if "cmds_geodesic" in methods:
        Dgeo = geodesic_distance(Dsk, S)
        rec("classical-MDS(geodesic Sinkhorn)", classical_mds(Dgeo))
    if "accu_geodesic" in methods:
        Dgeo = geodesic_distance(Dsk, S)
        Sw = S.copy(); np.fill_diagonal(Sw, 0.0)
        rec("accumap(geodesic dist, cMDS init)",
            accumap_run(classical_mds(Dgeo), Dgeo, Sw, cycles=10))
    if "accu_perfect" in methods:
        Sw = S.copy(); np.fill_diagonal(Sw, 0.0)
        rec("accumap(perfect dist)", accumap_run(E_umap, td, Sw, cycles=2))
    if "route1" in methods:
        from route1_poisson import route1
        Xr, _ = route1(C, sigma, X0=E_umap, oracle_U=None, U0=E_umap.copy(),
                       outer=6, steps_x=400, steps_u=400, lr=2.0, verbose=False)
        rec("Route1(generative, U init=X)", Xr)
    if "route1_oracleU" in methods:
        from route1_poisson import route1
        Xr, _ = route1(C, sigma, X0=E_umap, oracle_U=X, outer=8,
                       steps_x=500, lr=2.0, verbose=False)
        rec("Route1(oracle U = X_true)", Xr)
    if "route2" in methods:
        from route2_surrogate import route2
        Xr, _ = route2(C, n_outer=6, sigma=sigma, init=E_umap, verbose=False)
        rec("Route2(self-consistent)", Xr)

    hdr = f"  {'method':38s} {'posRMSE':>8s} {'pairRMSE':>9s} {'J@10':>6s}"
    print(hdr)
    for name, (pr, qr, j) in out.items():
        print(f"  {name:38s} {pr:8.1f} {qr:9.1f} {j:6.3f}")
    print(flush=True)
    return out


ALL = ["umap", "cmds_sinkhorn", "cmds_geodesic", "accu_sinkhorn",
       "accu_geodesic", "accu_perfect", "route1", "route2"]


# regimes matched to the paper sims: NN spacing must be <~0.6 sigma for the
# diffusion kernel to overlap neighbours.  New_Simulation = B1e4/sigma52,
# Original = B1e3/sigma300.
REGIMES = {
    "tight":  dict(B=8000, sigma=SIGMA),   # ~ New_Simulation
    "mid":    dict(B=3000, sigma=150.0),
    "broad":  dict(B=1500, sigma=300.0),   # ~ Original config
}


def cmd_smoke():
    C, X, meta = simulate_shared(B=1200, sigma=300.0, a_cv=0.4, q_cv=0.4)
    print("C", C.shape, "sym err",
          np.abs(C - C.T).sum() / C.sum(), "reads/bead", C.sum() / 1200)
    evaluate(C, X, 300.0, ["umap", "cmds_sinkhorn", "cmds_geodesic",
                           "accu_sinkhorn", "accu_geodesic", "accu_perfect"],
             tag="smoke")


def cmd_bench():
    for rk, rp in REGIMES.items():
        for hetero in (False, True):
            C, X, _ = simulate_shared(a_cv=0.5 if hetero else 0.0,
                                      q_cv=0.5 if hetero else 0.0, **rp)
            evaluate(C, X, rp["sigma"], ALL,
                     tag=f"{rk}{'+het' if hetero else ''}")


def cmd_sweep_sigma():
    for B, sig in [(8000, 52.0), (5000, 100.0), (3000, 150.0),
                   (2000, 220.0), (1500, 300.0), (1200, 450.0)]:
        C, X, _ = simulate_shared(B=B, sigma=sig, a_cv=0.5, q_cv=0.5)
        evaluate(C, X, sig, ["umap", "cmds_sinkhorn", "accu_sinkhorn",
                             "accu_perfect"], tag=f"B{B}/sigma{sig:.0f}")


def cmd_sweep_corrupt():
    """Isolate each failure axis in a WELL-SAMPLED regime (B=1500, sigma=300,
    NN ~0.13 sigma) so sampling is not the confound.  Base carries realistic
    per-bead heterogeneity (CV 0.5), stripped by Sinkhorn."""
    base = dict(B=1500, sigma=300.0, a_cv=0.5, q_cv=0.5, seed=42)
    sig = base["sigma"]
    meth = ["umap", "cmds_geodesic", "accu_sinkhorn", "accu_geodesic",
            "accu_perfect"]

    print("### axis: barcode crosstalk (random read reassignment)")
    for mu in (0.0, 0.02, 0.05, 0.10, 0.20):
        C, X, _ = simulate_shared(crosstalk_mu=mu, **base)
        evaluate(C, X, sig, meth, tag=f"crosstalk mu={mu}")

    print("### axis: count overdispersion (Poisson -> NB, size r)")
    for r in (np.inf, 3.0, 1.0, 0.3, 0.1, 0.03):
        C, X, _ = simulate_shared(noise="poisson", nb_r=r, **base)
        evaluate(C, X, sig, meth, tag=f"NB r={r}")

    print("### axis: per-bead heterogeneity (emission+capture log-normal CV)")
    for cv in (0.0, 0.5, 1.0, 1.5):
        b2 = {**base, "a_cv": cv, "q_cv": cv}
        C, X, _ = simulate_shared(**b2)
        evaluate(C, X, sig, meth, tag=f"hetero CV={cv}")

    print("### axis: kernel misspecification (solver still assumes Gaussian)")
    for name, extra in [("gaussian", {}),
                        ("gen_gauss p=1.4", dict(kernel="gen_gaussian", gg_p=1.4)),
                        ("gen_gauss p=1.0", dict(kernel="gen_gaussian", gg_p=1.0)),
                        ("cauchy", dict(kernel="cauchy")),
                        ("edge_narrow 0.3", dict(edge_narrow=0.3))]:
        C, X, _ = simulate_shared(**{**base, **extra})
        evaluate(C, X, sig, meth, tag=f"kernel={name}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    t0 = time.time()
    {"smoke": cmd_smoke, "bench": cmd_bench, "sweep-sigma": cmd_sweep_sigma,
     "sweep-corrupt": cmd_sweep_corrupt}[cmd]()
    print(f"{time.time() - t0:.0f}s")
