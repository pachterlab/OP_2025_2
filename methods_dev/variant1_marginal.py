"""Variant 1: marginalized-sender probabilistic model (distinct sender/receiver).

The (unobserved) sender positions are assumed uniform and integrated out of the
generative model, rather than estimated.  The receiver-receiver co-occurrence
statistic

    M_bb' = sum_o C_bo * C_b'o          (= C @ C.T, diagonal dropped)

then has an expectation that depends only on the receiver-receiver distance
(convolution of two Gaussian kernels -> width sqrt(2)*sigma):

    E[M_bb'] ~= alpha * g_b * g_b' * exp(-||x_b - x_b'||^2 / (4 sigma^2)) + beta

g_b is a per-bead factor absorbing the per-receiver normalisation / capture
efficiency (and the leading edge effect).  M is modelled as negative-binomial
around that mean.  We optimise the receiver coordinates X and the scalar /
per-bead nuisances by autodiff; sigma is fixed from the assay FWHM, which pins
the global scale.  No sender coordinates are estimated.

Note: exp(-d^2/4sigma^2) ~ 0 once d >> 2 sigma, so this model constrains local
geometry well but carries little long-range information -- pair it with a
global-structure initialisation (UMAP, or a geodesic completion).
"""
import numpy as np
import torch

from common import SIGMA, classical_mds


def _classical_init(M, sigma):
    off = ~np.eye(M.shape[0], dtype=bool)
    a0 = np.percentile(M[off], 99.9)
    b0 = np.percentile(M[off], 40.0) + 1.0
    k = np.clip((M - b0) / max(a0 - b0, 1.0), 1e-4, 1.0)
    D0 = np.sqrt(np.maximum(-4.0 * sigma ** 2 * np.log(k), 0.0))
    np.fill_diagonal(D0, 0.0)
    return classical_mds(D0), a0, b0


def variant1(C, sigma=SIGMA, init=None, warmup=400, steps=3000,
             lr_x=3.0, lr_aux=0.05, lr_decay=0.5, decay_every=1000,
             device="cuda", chunk=2048, verbose=True):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    Cn = np.asarray(C, dtype=np.float64)
    M = Cn @ Cn.T
    np.fill_diagonal(M, 0.0)
    B = M.shape[0]

    X0, a0, b0 = _classical_init(M, sigma)
    if init is not None:
        X0 = np.asarray(init, dtype=float)

    Mt = torch.tensor(M, dtype=torch.float32, device=dev)
    diag_mask = torch.ones(B, B, device=dev) - torch.eye(B, device=dev)
    inv_4s2 = 1.0 / (4.0 * sigma * sigma)

    X = torch.tensor(X0, dtype=torch.float32, device=dev).clone().requires_grad_(True)
    log_g = torch.zeros(B, device=dev, requires_grad=True)
    log_a = torch.tensor(float(np.log(max(a0, 1.0))), device=dev, requires_grad=True)
    log_b = torch.tensor(float(np.log(max(b0, 1.0))), device=dev, requires_grad=True)
    log_r = torch.tensor(0.0, device=dev, requires_grad=True)          # r = 1 / dispersion
    aux = [log_g, log_a, log_b, log_r]

    def nll():
        a, bb = torch.exp(log_a), torch.exp(log_b)
        r = torch.exp(log_r) + 1e-3
        g = torch.exp(log_g)
        tot = X.new_zeros(())
        denom = 0.0
        for i in range(0, B, chunk):
            sl = slice(i, i + chunk)
            d2 = torch.cdist(X[sl], X) ** 2
            mu = a * g[sl][:, None] * g[None, :] * torch.exp(-d2 * inv_4s2) + bb + 1e-6
            m = diag_mask[sl]
            ll = (torch.lgamma(Mt[sl] + r) - torch.lgamma(r) - torch.lgamma(Mt[sl] + 1.0)
                  + r * torch.log(r / (r + mu)) + Mt[sl] * torch.log(mu / (r + mu)))
            tot = tot - (ll * m).sum()
            denom += float(m.sum())
        return tot / denom

    # phase 1: fit nuisances to the initial geometry
    opt = torch.optim.Adam([{"params": aux, "lr": lr_aux}])
    for _ in range(warmup):
        opt.zero_grad(set_to_none=True)
        L = nll(); L.backward(); opt.step()
    if verbose:
        print(f"  variant1 warmup done: nll={float(L):.4f} "
              f"alpha={float(torch.exp(log_a)):.3g} beta={float(torch.exp(log_b)):.3g}")

    # phase 2: joint
    opt = torch.optim.Adam([
        {"params": [X], "lr": lr_x},
        {"params": aux, "lr": lr_aux},
    ])
    for t in range(steps):
        opt.zero_grad(set_to_none=True)
        L = nll(); L.backward(); opt.step()
        if (t + 1) % decay_every == 0:
            for grp in opt.param_groups:
                grp["lr"] *= lr_decay
            if verbose:
                print(f"  variant1 step {t+1}: nll={float(L):.4f}")

    params = {"alpha": float(torch.exp(log_a)), "beta": float(torch.exp(log_b)),
              "phi": float(1.0 / (torch.exp(log_r) + 1e-3))}
    return X.detach().cpu().numpy(), params
