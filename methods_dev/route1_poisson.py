"""Route 1: MAP-EM on the generative model, with explicit sender positions.

Data model (matches the simulator): each receiver's counts are multinomial over
senders with probabilities set by a Gaussian diffusion kernel,

    p_{b o} = softmax_o( -||x_b - u_o||^2 / (2 sigma^2) )
    C_{b .} ~ Multinomial(UMI, p_{b .})

Unknowns: receiver coords X (target), sender coords U (nuisance).  sigma is
fixed from the assay FWHM, which also pins the global scale.  Hard-EM:

  E-step : argmax_X  log p(C | X, U, sigma)     (U frozen)
  M-step : argmax_U  log p(C | X, U, sigma)     (X frozen)

Both steps are full-batch Adam gradient ascent via autodiff on the GPU.
`oracle_U` fixes senders at their true positions for a best-case bound.
"""
import numpy as np
import torch


def _neg_loglik(X, U, C, inv_2s2, chunk):
    """Mean-over-receivers negative multinomial log-likelihood, chunked over
    receivers so the (chunk x O) logit matrix fits in GPU memory."""
    B = X.shape[0]
    total = X.new_zeros(())
    for i in range(0, B, chunk):
        d2 = torch.cdist(X[i:i + chunk], U) ** 2           # c x O
        logits = -d2 * inv_2s2
        logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        total = total - (C[i:i + chunk] * logp).sum()
    return total / B


def _random_disc(n, radius, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = torch.sqrt(torch.rand(n, generator=g)) * radius
    th = torch.rand(n, generator=g) * 2 * np.pi
    return torch.stack([r * torch.cos(th), r * torch.sin(th)], dim=1).numpy()


def route1(C, sigma, X0, U0=None, oracle_U=None,
           outer=10, steps_x=600, steps_u=600, lr=3.0, lr_decay=0.85,
           device="cuda", chunk=2048, verbose=True):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    Ct = torch.tensor(C, dtype=torch.float32, device=dev)
    inv_2s2 = 1.0 / (2.0 * sigma * sigma)
    n_sender = Ct.shape[1]

    X = torch.tensor(X0, dtype=torch.float32, device=dev).clone()

    fit_U = oracle_U is None
    if oracle_U is not None:
        U = torch.tensor(oracle_U, dtype=torch.float32, device=dev)
    else:
        if U0 is None:
            radius = float(np.sqrt(np.mean(np.sum(np.asarray(X0) ** 2, axis=1))) * 1.7)
            U0 = _random_disc(n_sender, radius)
        U = torch.tensor(U0, dtype=torch.float32, device=dev).clone()

    def run_block(active, steps, cur_lr):
        if steps <= 0:
            with torch.no_grad():
                return float(_neg_loglik(X, U, Ct, inv_2s2, chunk))
        for p in (X, U):
            p.requires_grad_(p is active)
        opt = torch.optim.Adam([active], lr=cur_lr)
        loss = None
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            loss = _neg_loglik(X, U, Ct, inv_2s2, chunk)
            loss.backward()
            opt.step()
        active.requires_grad_(False)
        return float(loss.detach())

    hist = []
    cur_lr = lr
    for o in range(outer):
        lx = run_block(X, steps_x, cur_lr)
        lu = run_block(U, steps_u, cur_lr) if fit_U else lx
        cur_lr *= lr_decay
        hist.append({"outer": o, "nll_after_X": lx, "nll_after_U": lu})
        if verbose:
            print(f"  route1 outer {o}: nll(afterX)={lx:.6f} nll(afterU)={lu:.6f}")

    return X.detach().cpu().numpy(), U.detach().cpu().numpy(), hist
