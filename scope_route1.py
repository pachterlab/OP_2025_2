"""Route 1 (generative MAP) adapted for the shared-bead case: U = X, so
there is no second point set to jointly infer -- the failure mode that
broke Route 1 "joint" in the distinct sender/receiver case
(methods-dev-benchmark memory) should not apply here.

SCOPE's own kernel is inverse-square (~1/r^2), not Gaussian, and there is no
known sigma for this chemistry -- so the transfer function is fit jointly
with X rather than fixed:

    K(d; lam, p) = 1 / (1 + (d/lam)^2)^p          (Cauchy-family, p=1 -> SCOPE's
                                                    reported inverse-square law)
    logits_{b,b'} = log a[b'] + p * log(1/(1+(d_{b,b'}/lam)^2))   (b' != b)
    C_{b,.} ~ Multinomial(UMI_b, softmax_{b'}(logits_{b,.}))

Free parameters: X (bead positions), lam, p (kernel shape), log_a (per-bead
emission attractiveness, shared across all rows). All optimized jointly by
Adam/autodiff (CPU -- atlas has no GPU, but B=2415 is small enough that a
dense B x B loss is cheap per step).
"""
import sys
import time
import numpy as np

sys.path.insert(0, "/home/coakes/scope_recon")
from scope_bench import load_scope, umap_embed  # imports pandas -- must come
                                                 # before torch (torch's
                                                 # bundled libstdc++ shadows
                                                 # the newer conda one that
                                                 # pandas needs otherwise)
import torch


def route1_shared(C, X0, steps=800, lr=0.05, lam0=30.0, p0=1.0, device="cpu",
                  verbose=True):
    B = C.shape[0]
    Ct = torch.tensor(C, dtype=torch.float32, device=device)
    mask = ~torch.eye(B, dtype=torch.bool, device=device)   # exclude self term

    X = torch.tensor(X0, dtype=torch.float32, device=device).clone().requires_grad_(True)
    log_lam = torch.tensor(float(np.log(lam0)), device=device).requires_grad_(True)
    log_p = torch.tensor(float(np.log(p0)), device=device).requires_grad_(True)
    log_a = torch.zeros(B, device=device).requires_grad_(True)

    opt = torch.optim.Adam([X, log_lam, log_p, log_a], lr=lr)
    hist = []
    for it in range(steps):
        opt.zero_grad(set_to_none=True)
        d2 = torch.cdist(X, X) ** 2
        lam2 = torch.exp(2 * log_lam)
        p = torch.exp(log_p)
        logK = -p * torch.log1p(d2 / lam2)             # log(1/(1+(d/lam)^2)^p)
        logits = log_a[None, :] + logK
        logits = logits.masked_fill(~mask, -1e9)
        logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        loss = -(Ct * logp).sum() / B
        loss.backward()
        opt.step()
        if verbose and (it % 100 == 0 or it == steps - 1):
            print(f"    step {it:4d}: loss={loss.item():.5f}  "
                  f"lam={torch.exp(log_lam).item():.1f}  p={torch.exp(log_p).item():.3f}",
                  flush=True)
        hist.append(float(loss.item()))
    return X.detach().numpy(), dict(lam=float(torch.exp(log_lam)), p=float(torch.exp(log_p)),
                                    hist=hist)


if __name__ == "__main__":
    t0 = time.time()
    C, bc2idx = load_scope("letter_E")
    print(f"letter_E: {C.shape}, {int(C.sum()):,} reads", flush=True)

    E_umap = umap_embed(C)
    print(f"UMAP init done ({time.time()-t0:.0f}s)", flush=True)

    X_r1, meta = route1_shared(C, E_umap, steps=800, lr=0.05)
    print(f"Route1 done ({time.time()-t0:.0f}s)  lam={meta['lam']:.1f} p={meta['p']:.3f}",
          flush=True)

    np.savez("/home/coakes/scope_recon/results/letter_E_route1_cache.npz",
            E_umap=E_umap, X_r1=X_r1, lam=meta["lam"], p=meta["p"],
            loss_hist=np.array(meta["hist"]))
    print(f"saved, total {time.time()-t0:.0f}s", flush=True)
