"""Fit the real count-vs-distance relationship on c58_8.

Imaging gives BOTH receiver (x_b) and sender (u_o) positions, so d_bo is known
for every observed count C_bo.  This directly measures which generative-model
choices matter, without running any reconstruction:

  1. mean function  E[C | d]  -- Gaussian vs exponential vs generalized-Gaussian
     vs +background vs heavy-tailed, ranked by held-out weighted error
  2. noise model    Var[C | d] vs E[C | d]  -- Poisson vs negative-binomial
  3. per-bead heterogeneity  -- spread of sender-abundance / receiver-capture
     effects in a Poisson GLM with a distance offset
  4. anisotropy     -- isotropic ||d|| vs a fitted 2x2 Mahalanobis form
  5. radial variation of the kernel width (edge effect)
"""
import os
import sys
import time
import numpy as np
from scipy.optimize import curve_fit

from real_data import load_hu

OUT = os.path.join(os.path.dirname(__file__), "results_real")
os.makedirs(OUT, exist_ok=True)

# ---- distance bins (log-spaced) --------------------------------------------
DMIN, DMAX, NB = 5.0, 3500.0, 120
EDGES = np.geomspace(DMIN, DMAX, NB + 1)
CTR = np.sqrt(EDGES[1:] * EDGES[:-1])


# ---- candidate mean functions (f(0) ~ A) ---------------------------------
def m_gauss(d, A, s):            return A * np.exp(-0.5 * (d / s) ** 2)
def m_exp(d, A, lam):           return A * np.exp(-d / lam)
def m_gengauss(d, A, lam, p):   return A * np.exp(-(d / lam) ** p)
def m_gauss_bg(d, A, s, c):     return A * np.exp(-0.5 * (d / s) ** 2) + c
def m_gengauss_bg(d, A, lam, p, c): return A * np.exp(-(d / lam) ** p) + c
def m_cauchy_bg(d, A, lam, nu, c):  return A / (1 + (d / lam) ** 2) ** nu + c


MODELS = {
    "gaussian":        (m_gauss,        (1.0, 60.0)),
    "exponential":     (m_exp,          (1.0, 60.0)),
    "gen-gaussian":    (m_gengauss,     (1.0, 60.0, 2.0)),
    "gaussian+bg":     (m_gauss_bg,     (1.0, 60.0, 1e-3)),
    "gen-gaussian+bg": (m_gengauss_bg,  (1.0, 60.0, 2.0, 1e-3)),
    "cauchy+bg":       (m_cauchy_bg,    (1.0, 60.0, 1.0, 1e-3)),
}


def binned_stats(X, U, C, ecc_mask=None, chunk=400):
    """Accumulate per-bin count, sum(C), sum(C^2) over all (b,o) pairs."""
    n, cnt = C.shape[0], np.zeros(NB)
    s1, s2 = np.zeros(NB), np.zeros(NB)
    rows = np.arange(n) if ecc_mask is None else np.where(ecc_mask)[0]
    for i0 in range(0, len(rows), chunk):
        r = rows[i0:i0 + chunk]
        d = np.sqrt(((X[r, None, :] - U[None, :, :]) ** 2).sum(-1)).ravel()
        c = C[r].ravel()
        b = np.digitize(d, EDGES) - 1
        ok = (b >= 0) & (b < NB)
        np.add.at(cnt, b[ok], 1.0)
        np.add.at(s1, b[ok], c[ok])
        np.add.at(s2, b[ok], c[ok] ** 2)
    mean = np.where(cnt > 0, s1 / np.maximum(cnt, 1), np.nan)
    var = np.where(cnt > 0, s2 / np.maximum(cnt, 1) - mean ** 2, np.nan)
    return cnt, mean, var


def fit_models(mean, cnt):
    ok = np.isfinite(mean) & (cnt > 500)
    d, y, w = CTR[ok], mean[ok], np.sqrt(cnt[ok])
    # held-out: fit on even bins, score on odd
    tr = np.arange(len(d)) % 2 == 0
    res = {}
    for name, (fn, p0) in MODELS.items():
        try:
            popt, _ = curve_fit(fn, d[tr], y[tr], p0=p0, sigma=1 / w[tr],
                                maxfev=40000)
            pred = fn(d[~tr], *popt)
            rel = np.abs(pred - y[~tr]) / (y[~tr] + 1e-9)
            res[name] = (float(np.median(rel)), popt)
        except Exception as e:
            res[name] = (np.inf, str(e))
    return res


def fit_nb_dispersion(mean, var, cnt):
    ok = np.isfinite(mean) & np.isfinite(var) & (cnt > 500) & (mean > 1e-4)
    m, v = mean[ok], var[ok]
    # var = m + m^2 / r   ->  (v - m) = m^2 / r
    num = np.sum((v - m) * m ** 2)
    den = np.sum(m ** 4)
    r = den / max(num, 1e-12)
    poisson_rel = np.median(np.abs(v - m) / (v + 1e-9))
    nb_rel = np.median(np.abs(v - (m + m ** 2 / r)) / (v + 1e-9))
    return float(r), float(poisson_rel), float(nb_rel)


def per_bead_glm(X, U, C, n_sample=800_000, iters=300):
    """log E[C_bo] = a_o + q_b + log f(d) ; report sd(a_o), sd(q_b)."""
    import torch
    n, o = C.shape
    rng = np.random.RandomState(0)
    bi = rng.randint(0, n, n_sample)
    oi = rng.randint(0, o, n_sample)
    d = np.sqrt(((X[bi] - U[oi]) ** 2).sum(1))
    c = C[bi, oi].astype(np.float32)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = torch.tensor(d, dtype=torch.float32, device=dev)
    c = torch.tensor(c, dtype=torch.float32, device=dev)
    bi_t = torch.tensor(bi, device=dev)
    oi_t = torch.tensor(oi, device=dev)
    a = torch.zeros(o, device=dev, requires_grad=True)   # sender abundance
    q = torch.zeros(n, device=dev, requires_grad=True)   # receiver capture
    logA = torch.tensor(0.0, device=dev, requires_grad=True)
    loglam = torch.tensor(float(np.log(60.0)), device=dev, requires_grad=True)
    logp = torch.tensor(float(np.log(2.0)), device=dev, requires_grad=True)
    opt = torch.optim.Adam([a, q, logA, loglam, logp], lr=0.05)
    for _ in range(iters):
        opt.zero_grad()
        lam, p = torch.exp(loglam), torch.exp(logp)
        logf = logA - (d / lam) ** p
        eta = a[oi_t] + q[bi_t] + logf
        mu = torch.exp(eta.clamp(max=20))
        nll = (mu - c * eta).mean()          # Poisson
        nll = nll + 1e-3 * (a.pow(2).mean() + q.pow(2).mean())
        nll.backward()
        opt.step()
    return {"sd_sender_abundance": float(a.detach().std()),
            "sd_receiver_capture": float(q.detach().std()),
            "gengauss_p": float(torch.exp(logp)),
            "gengauss_lambda": float(torch.exp(loglam))}


def main(rep="c58_8"):
    t0 = time.time()
    C, X, U = load_hu(rep)
    C = C.astype(np.int32)
    X = X.astype(np.float32)
    U = U.astype(np.float32)
    print(f"{rep}: C {C.shape}  {C.sum():,} reads  mean count/pair {C.mean():.3f}")

    cnt, mean, var = binned_stats(X, U, C)
    print("\n--- mean function E[C|d] ---")
    res = fit_models(mean, cnt)
    for name, (score, popt) in sorted(res.items(), key=lambda kv: kv[1][0]):
        pr = np.round(popt, 4).tolist() if not isinstance(popt, str) else popt
        print(f"  {name:18s} held-out median rel.err {score:7.3f}   params {pr}")

    r, prel, nbrel = fit_nb_dispersion(mean, var, cnt)
    print(f"\n--- noise: Var(C|d) vs Mean(C|d) ---")
    print(f"  Poisson (var=mean)      median rel.err {prel:.3f}")
    print(f"  NB (var=mean+mean^2/r)  median rel.err {nbrel:.3f}   r={r:.2f} "
          f"(dispersion 1/r = {1/r:.3f})")

    print("\n--- per-bead heterogeneity (Poisson GLM) ---")
    glm = per_bead_glm(X, U, C)
    for k, v in glm.items():
        print(f"  {k:22s} {v:.4f}")

    print("\n--- radial variation of kernel width ---")
    ecc = np.sqrt((X ** 2).sum(1))
    qs = np.quantile(ecc, [0, 1 / 3, 2 / 3, 1.0])
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (ecc >= lo) & (ecc < hi + 1e-6)
        _, mn, _ = binned_stats(X, U, C, ecc_mask=m)
        rr = fit_models(mn, cnt)
        best = min(rr, key=lambda k: rr[k][0])
        p = np.round(rr[best][1], 3).tolist()
        print(f"  ecc {lo:6.0f}-{hi:6.0f} um : best={best:16s} params {p}")

    np.savez(os.path.join(OUT, f"{rep}_kernel.npz"),
             ctr=CTR, cnt=cnt, mean=mean, var=var)
    print(f"\n{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["c58_8"]))
