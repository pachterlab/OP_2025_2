"""Realistic simulation of distinct-sender/receiver bead arrays, calibrated on the
Hu et al. c58_8 experimental array (calibrated by the paper's development script realistic_sim_calibrate.py).

Counts:  C_bo ~ Poisson( mu_bo * G_bo ),  G_bo ~ Gamma(1/alpha, alpha)   (mean 1, Var mu + alpha mu^2)
         mu_bo = exp(q_b + a_o) * ( K(|x_b - u_o|) + c )
Every component can be switched to its idealized form, to isolate which one
breaks a reconstruction method:
  kernel          "fitted": transfer kernel fit to c58_8 (contact core, heavy tail)
                  "gaussian": exp(-d^2 / 2 sigma^2), sigma = 52.2 um (the idealized simulation)
  background      flat transfer c (ambient / crosstalk) as fitted, or none
  heterogeneity   per-bead capture q_b and abundance a_o drawn from their fitted
                  distributions (or taken bead-by-bead for the real layout), or constant
  overdispersion  "fitted": pair-level gamma noise with alpha a function of mu, interpolated
                  from c58_8 where the fitted mean matches the data (mu >= 0.5: alpha ~0.1-0.8)
                  and held at its mu = 0.5 value below (at lower mu the raw estimate is
                  inflated by small mean misfit, not real spikiness); "constant": the pooled
                  alpha; False: pure Poisson
  depth_scale     multiplies every expected count (1 = c58_8's depth, ~1,800 UMIs per receiver)
  collisions      fraction of receivers that also collect counts at a second, random location
                  (barcode collision / doublet); their true position stays the first one;
                  collision_strength scales the second location's expected counts. Defaults
                  (0.6% of receivers at 40% strength) match c58_8's rate of long-range
                  similarity edges (0.07-0.09% of 15-NN edges > 300 um) and split receivers
Layout: uniform random in a disc of the c58_8 array's diameter (3000 um) with c58_8's
bead numbers, or caller-supplied positions (e.g. c58_8's imaging positions).
"""
import os
import numpy as np

PARAMS = os.path.join(os.path.dirname(__file__), "data", "c58_8_sim_params.npz")
GAUSS_SIGMA = 123.0 / np.sqrt(8.0 * np.log(2.0))
GAUSS_UMI = 2500.0   # expected UMIs per receiver in the idealized (constant-depth) case


def load_params(path=PARAMS):
    Z = np.load(path)
    return {k: Z[k] for k in Z.files}


def _disc(rng, n, radius):
    r = radius * np.sqrt(rng.uniform(0, 1, n))
    t = rng.uniform(0, 2 * np.pi, n)
    return np.c_[r * np.cos(t), r * np.sin(t)].astype(np.float32)


def kernel_fn(d, params, kernel="fitted"):
    """K(d) with K = 1 at contact; d in um."""
    if kernel == "gaussian":
        return np.exp(-d ** 2 / (2 * GAUSS_SIGMA ** 2))
    lk = np.log(params["knots"])
    ld = np.log(np.clip(d, params["knots"][0], params["knots"][-1]))
    return np.exp(np.interp(ld, lk, params["logK"]))


def alpha_fn(mu, params):
    """Dispersion alpha(mu) (Var = mu + alpha mu^2), log-log interpolation of the c58_8 table."""
    T = params["alpha_table"]            # columns: bin lo, bin hi, n, mean mu, mean C, alpha
    ok = (T[:, 5] > 0) & (T[:, 0] >= 0.5)
    lm, la = np.log(T[ok, 3]), np.log(T[ok, 5])
    return np.exp(np.interp(np.log(np.maximum(mu, 1e-12)), lm, la))


def simulate_realistic(seed=0, receivers=None, senders=None, kernel="fitted", background=True,
                       heterogeneity=True, overdispersion="fitted", depth_scale=1.0, collisions=0.006,
                       collision_strength=0.4, params=None, chunk=500):
    """Return (C, X, U): float32 counts (n_receiver x n_sender), receiver and sender positions (um).

    receivers, senders: optional (n, 2) positions in um. If both are given with the calibration
    array's bead numbers, heterogeneity uses each bead's own fitted q_b / a_o (semi-synthetic
    layout); otherwise beads are placed uniformly in the calibration array's disc and q, a are
    drawn from their fitted distributions.
    """
    P = load_params() if params is None else params
    rng = np.random.default_rng(seed)
    nR, nS = int(P["n_receiver"]), int(P["n_sender"])
    real_layout = receivers is not None and senders is not None
    if real_layout:
        X, U = np.asarray(receivers, np.float32), np.asarray(senders, np.float32)
    else:
        R = float(P["array_diameter"]) / 2
        X, U = _disc(rng, nR, R), _disc(rng, nS, R)
    nR, nS = len(X), len(U)

    if heterogeneity:
        same = real_layout and nR == len(P["q"]) and nS == len(P["a"])
        q = P["q"].astype(np.float64) if same else rng.choice(P["q"], nR).astype(np.float64)
        a = P["a"].astype(np.float64) if same else rng.choice(P["a"], nS).astype(np.float64)
    else:
        q, a = None, np.zeros(nS)
    c = float(P["c"]) if background else 0.0

    ea = np.exp(a)

    def rows(pos, qi):
        d = np.sqrt(((pos[:, None, :].astype(np.float64) - U[None, :, :]) ** 2).sum(-1))
        m = (kernel_fn(d, P, kernel) + c) * ea[None, :]
        if qi is None:
            # constant capture: c58_8's mean depth for the fitted kernel, GAUSS_UMI for the Gaussian one
            target = GAUSS_UMI if kernel == "gaussian" else float(P["mean_depth"])
            mu = m * (target / m.sum(1, keepdims=True))
        else:
            mu = np.exp(qi)[:, None] * m
        mu *= depth_scale
        if overdispersion:
            al = alpha_fn(mu, P) if overdispersion == "fitted" else np.full_like(mu, float(P["alpha"]))
            mu = mu * rng.gamma(1.0 / al, al)
        return rng.poisson(mu)

    C = np.empty((nR, nS), np.float32)
    for i in range(0, nR, chunk):
        C[i:i + chunk] = rows(X[i:i + chunk], None if q is None else q[i:i + chunk])
    if collisions > 0:
        idx = rng.choice(nR, int(round(collisions * nR)), replace=False)
        second = _disc(rng, len(idx), float(P["array_diameter"]) / 2)
        qs = np.full(len(idx), np.log(collision_strength)) if q is None else q[idx] + np.log(collision_strength)
        if q is None:   # constant capture: rows() normalizes to the target depth, so scale after
            C[idx] += rng.binomial(rows(second, None).astype(np.int64), min(collision_strength, 1.0))
        else:
            C[idx] += rows(second, qs)
    return C, X, U
