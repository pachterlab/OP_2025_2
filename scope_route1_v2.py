"""Route 1 v2 for the shared-bead SCOPE data: scale-anchored + background
mixture, isolating "is the right kernel shape enough" from the v1 scale
degeneracy (lam collapsed to 0.1 there because nothing pinned X's absolute
scale against lam).

Fixes vs v1 (scope_route1.py):
  1. Scale anchor: after every Adam step, rescale X (no_grad) back to a
     FIXED reference RMS-from-centroid (set once from the initial UMAP
     embedding). This pins the "ruler" so lam and p can adapt meaningfully
     instead of lam and ||X|| co-drifting to satisfy the same ratio.
  2. Background mixture: prob_b = (1-c)*softmax(signal_logits_b) + c*uniform,
     a single free (sigmoid-bounded) global contamination rate c -- same
     "ambient reads redistributed uniformly across the population" model
     used in decontaminate.py / crosstalk_sweep.py this session, but fit
     jointly with the geometry instead of estimated separately.

Same kernel family as v1: K(d;lam,p) = 1/(1+(d/lam)^2)^p, p free (SCOPE's
own reported law is p~1, inverse-square).
"""
import sys
import time
import numpy as np

sys.path.insert(0, "/home/coakes/scope_recon")
from scope_bench import load_scope, umap_embed  # noqa: E402  (import before torch, see v1 note)
import torch


def route1_shared_v2(C, X0, steps=1500, lr=0.03, lam0=30.0, p0=1.0,
                     c0=0.05, device="cpu", verbose=True):
    B = C.shape[0]
    Ct = torch.tensor(C, dtype=torch.float32, device=device)
    mask = ~torch.eye(B, dtype=torch.bool, device=device)

    X0t = torch.tensor(X0, dtype=torch.float32, device=device)
    ref_scale = float(torch.sqrt(((X0t - X0t.mean(0)) ** 2).sum(1).mean()))

    X = X0t.clone().requires_grad_(True)
    log_lam = torch.tensor(float(np.log(lam0)), device=device).requires_grad_(True)
    log_p = torch.tensor(float(np.log(p0)), device=device).requires_grad_(True)
    log_a = torch.zeros(B, device=device).requires_grad_(True)
    logit_c = torch.tensor(float(np.log(c0 / (1 - c0))), device=device).requires_grad_(True)

    log_uniform = -np.log(B - 1)   # log(1/(B-1)), constant

    opt = torch.optim.Adam([X, log_lam, log_p, log_a, logit_c], lr=lr)
    hist = []
    for it in range(steps):
        opt.zero_grad(set_to_none=True)
        d2 = torch.cdist(X, X) ** 2
        lam2 = torch.exp(2 * log_lam)
        p = torch.exp(log_p)
        logK = -p * torch.log1p(d2 / lam2)
        logits = log_a[None, :] + logK
        logits = logits.masked_fill(~mask, -1e9)
        log_signal = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        c = torch.sigmoid(logit_c)
        log_mix = torch.logsumexp(
            torch.stack([torch.log(1 - c + 1e-12) + log_signal,
                        (torch.log(c + 1e-12) + log_uniform) * torch.ones_like(log_signal)]),
            dim=0)

        loss = -(Ct * log_mix).sum() / B
        loss.backward()
        opt.step()

        with torch.no_grad():
            cur_scale = torch.sqrt(((X - X.mean(0)) ** 2).sum(1).mean())
            X.mul_(ref_scale / cur_scale.clamp_min(1e-6))

        if verbose and (it % 150 == 0 or it == steps - 1):
            print(f"    step {it:4d}: loss={loss.item():.5f}  lam={torch.exp(log_lam).item():.2f}  "
                  f"p={torch.exp(log_p).item():.3f}  c={torch.sigmoid(logit_c).item():.4f}",
                  flush=True)
        hist.append(float(loss.item()))
    return X.detach().numpy(), dict(lam=float(torch.exp(log_lam)), p=float(torch.exp(log_p)),
                                    c=float(torch.sigmoid(logit_c)), hist=hist)


if __name__ == "__main__":
    t0 = time.time()
    C, bc2idx = load_scope("letter_E")
    print(f"letter_E: {C.shape}, {int(C.sum()):,} reads", flush=True)

    E_umap = umap_embed(C)
    print(f"UMAP init done ({time.time()-t0:.0f}s)", flush=True)

    X_r1, meta = route1_shared_v2(C, E_umap, steps=1500, lr=0.03)
    print(f"Route1v2 done ({time.time()-t0:.0f}s)  lam={meta['lam']:.2f} "
          f"p={meta['p']:.3f} c={meta['c']:.4f}", flush=True)

    np.savez("/home/coakes/scope_recon/results/letter_E_route1v2_cache.npz",
            E_umap=E_umap, X_r1=X_r1, lam=meta["lam"], p=meta["p"], c=meta["c"],
            loss_hist=np.array(meta["hist"]))
    print(f"saved, total {time.time()-t0:.0f}s", flush=True)
