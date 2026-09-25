"""Route 1, augmented with the corrections the real count-vs-distance fit said
matter:  heavy-tailed (Cauchy) kernel with shape fixed from the fit, per-sender
abundance a_o, per-receiver capture q_b, a background floor c, and NB
overdispersion.  Oracle sender positions (imaging U).

Ceiling check on real c58_8: does a better-specified diffusion model, given true
sender positions, beat UMAP (mean MSE 1679)?

    C_bo ~ NB( mean = A * exp(a_o + q_b) * [ (1 + d^2/lam^2)^-nu + c ] , r )

lam, nu are fixed from fit_kernel.py (this also pins the global scale).
"""
import os
import sys
import time
import numpy as np
import torch

from real_data import load_hu
from common import per_bead_mse, summary

LAM, NU = 26.0, 4.0   # Cauchy kernel shape from fit_kernel.py on c58_8


def _kernel(d2, lam, nu):
    return (1.0 + d2 / (lam * lam)) ** (-nu)


def _nb_nll(C, mu, r):
    return -(torch.lgamma(C + r) - torch.lgamma(r) - torch.lgamma(C + 1.0)
             + r * torch.log(r / (r + mu)) + C * torch.log(mu / (r + mu)))


def run(C, U, X0, lam=LAM, nu=NU, warmup=500, x_blocks=12, x_steps=200,
        lr_x=2.5, lr_aux=0.02, device="cuda", chunk=1500, verbose=True):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    B, O = C.shape
    Ct = torch.tensor(C, dtype=torch.float32, device=dev)
    Ut = torch.tensor(U, dtype=torch.float32, device=dev)

    X = torch.tensor(X0, dtype=torch.float32, device=dev).clone()
    a = torch.zeros(O, device=dev)                                   # log sender abundance
    q = torch.zeros(B, device=dev)                                   # log receiver capture
    logA = torch.tensor(float(np.log(2.0)), device=dev)
    logc = torch.tensor(float(np.log(0.057)), device=dev)            # background floor
    logr = torch.tensor(float(np.log(1.4)), device=dev)             # NB size
    allp = [X, a, q, logA, logc, logr]
    for p in allp:
        p.requires_grad_(False)

    def nll():
        A_, c_, r_ = torch.exp(logA), torch.exp(logc), torch.exp(logr) + 1e-3
        tot = X.new_zeros(())
        denom = 0.0
        for i in range(0, B, chunk):
            sl = slice(i, i + chunk)
            d2 = torch.cdist(X[sl], Ut) ** 2
            mu = A_ * torch.exp(q[sl][:, None] + a[None, :]) * (_kernel(d2, lam, nu) + c_) + 1e-8
            tot = tot + _nb_nll(Ct[sl], mu, r_).sum()
            denom += Ct[sl].numel()
        return tot / denom + 1e-3 * (a.pow(2).mean() + q.pow(2).mean())

    def block(params, lr, n):
        for p in allp:
            p.requires_grad_(any(p is x for x in params))
        opt = torch.optim.Adam(params, lr=lr)
        L = None
        for _ in range(n):
            opt.zero_grad(set_to_none=True)
            L = nll()
            L.backward()
            opt.step()
        for p in params:
            p.requires_grad_(False)
        return float(L)

    aux = [a, q, logA, logc, logr]
    L0 = block(aux, lr_aux, warmup)
    if verbose:
        print(f"  warmup nll={L0:.5f}  A={float(torch.exp(logA)):.3g} "
              f"c={float(torch.exp(logc)):.3g} r={float(torch.exp(logr)):.3g}")
    lr = lr_x
    for k in range(x_blocks):
        lx = block([X], lr, x_steps)
        la = block(aux, lr_aux, 40)
        if verbose:
            print(f"  block {k:2d}: nll={la:.5f}  lr={lr:.3f}")
        lr *= 0.82

    return X.detach().cpu().numpy(), {
        "A": float(torch.exp(logA)), "c": float(torch.exp(logc)),
        "r": float(torch.exp(logr)), "sd_a": float(a.std()), "sd_q": float(q.std())}


if __name__ == "__main__":
    rep = sys.argv[1] if len(sys.argv) > 1 else "c58_8"
    t0 = time.time()
    C, X_true, U_true = load_hu(rep)
    cache = os.path.join(os.path.dirname(__file__), "results_real", f"{rep}_cache.npz")
    if os.path.exists(cache):
        X0 = np.load(cache)["E_umap"]
        print("init: cached UMAP")
    else:
        from common import classical_mds, cos_sim
        S = cos_sim(C); np.fill_diagonal(S, 1.0)
        X0 = classical_mds(1.0 - S)
        print("init: classical MDS")

    E, params = run(C.astype(np.float32), U_true.astype(np.float32), X0)
    umse = per_bead_mse(X0, X_true)
    row, jac = summary("Route1-augmented oracle-U (real)", E, X_true, umap_mse=umse)
    print("\nRESULT:", {k: (round(v, 3) if isinstance(v, float) else v)
                        for k, v in row.items()})
    print("params:", {k: round(v, 4) for k, v in params.items()})
    print("jaccard:", {k: round(v, 3) for k, v in jac.items()})
    print(f"{time.time()-t0:.0f}s")
