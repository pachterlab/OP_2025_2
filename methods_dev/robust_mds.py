"""Robust weighted MDS with far-pair softening.

  (1) reliability weights  w_ij = 1 / (sigmahat^2(S_ij) + eps)   -> far pairs ~0
  (2) far pairs enter only as one-sided lower bounds:
         penalty  max(0, D_lo(S_ij) - ||x_i - x_j||)^2
      (push apart if collapsed; no penalty for "too far")
  (3) Huber loss on the near-pair residuals

g / sigmahat^2 / D_lo are calibrated from (S, distance) pairs:
  - mode="imaging": distances from a subset of imaged beads (validation of the
    softening idea given a good calibration)
  - mode="selfcons": distances from the current embedding (no ground truth),
    optionally re-fit for a few outer passes
"""
import os
import sys
import time
import numpy as np
import torch

from real_data import load_hu
from common import cos_sim
from position_error import position_rmse, pairwise_rmse
from common import jaccard_curve


def calibrate(s, d, nbin=60, min_per=40):
    edges = np.unique(np.quantile(s, np.linspace(0, 1, nbin + 1)))
    ctr = 0.5 * (edges[1:] + edges[:-1])
    b = np.clip(np.digitize(s, edges) - 1, 0, len(ctr) - 1)
    med = np.full(len(ctr), np.nan)
    var = np.full(len(ctr), np.nan)
    lo = np.full(len(ctr), np.nan)
    for k in range(len(ctr)):
        m = b == k
        if m.sum() >= min_per:
            dk = d[m]
            med[k] = np.median(dk)
            var[k] = np.var(dk)
            lo[k] = np.quantile(dk, 0.10)
    ok = np.isfinite(med)
    g_c = np.maximum.accumulate(med[ok][::-1])[::-1]        # enforce non-increasing
    lo_c = np.maximum.accumulate(lo[ok][::-1])[::-1]
    x = ctr[ok]
    return {"x": x, "g": g_c, "var": var[ok], "lo": lo_c}


def _interp(c, key, s):
    return np.interp(np.clip(s, c["x"][0], c["x"][-1]), c["x"], c[key])


def build_pairs(S, k_near=8, s_near_q=0.90, k_far=60, seed=0, mutual=True):
    n = S.shape[0]
    rng = np.random.RandomState(seed)
    thr = np.quantile(S[np.triu_indices(n, 1)], s_near_q)
    order = np.argsort(-S, axis=1)
    knn = [set(order[i, 1:k_near + 1]) for i in range(n)]
    near = set()
    for i in range(n):
        for j in knn[i]:
            if mutual and i not in knn[j]:
                continue
            near.add((i, j) if i < j else (j, i))
    near = np.array(sorted(near)) if near else np.empty((0, 2), int)
    fi = np.repeat(np.arange(n), k_far)
    fj = rng.randint(0, n, n * k_far)
    keep = (fi != fj) & (S[fi, fj] < thr)
    far = np.stack([fi[keep], fj[keep]], 1)
    return near, far, float(thr)


def robust_mds(X0, S, near, far, cal, huber_k=None, steps=1500, lr=2.0,
               kappa=0.3, device="cuda"):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    X = torch.tensor(X0, dtype=torch.float32, device=dev).clone().requires_grad_(True)

    Dn = torch.tensor(_interp(cal, "g", S[near[:, 0], near[:, 1]]), dtype=torch.float32, device=dev)
    _v = _interp(cal, "var", S[near[:, 0], near[:, 1]])
    wn = torch.tensor(1.0 / (_v + np.percentile(_v, 10)),
                      dtype=torch.float32, device=dev)
    wn = wn / wn.mean()
    ni = torch.tensor(near[:, 0], device=dev)
    nj = torch.tensor(near[:, 1], device=dev)

    Dlo = torch.tensor(_interp(cal, "lo", S[far[:, 0], far[:, 1]]), dtype=torch.float32, device=dev)
    fi = torch.tensor(far[:, 0], device=dev)
    fj = torch.tensor(far[:, 1], device=dev)

    if huber_k is None:
        huber_k = float(np.median(np.abs(_interp(cal, "g", S[near[:, 0], near[:, 1]])
                                         - np.linalg.norm(X0[near[:, 0]] - X0[near[:, 1]], axis=1))))
        huber_k = max(huber_k, 5.0)

    opt = torch.optim.Adam([X], lr=lr)
    for t in range(steps):
        opt.zero_grad(set_to_none=True)
        dn = (X[ni] - X[nj]).norm(dim=1)
        r = dn - Dn
        hub = torch.where(r.abs() <= huber_k, 0.5 * r ** 2,
                          huber_k * (r.abs() - 0.5 * huber_k))
        loss_near = (wn * hub).mean()
        df = (X[fi] - X[fj]).norm(dim=1)
        loss_far = torch.clamp(Dlo - df, min=0.0).pow(2).mean()
        loss = loss_near + kappa * loss_far
        loss.backward()
        opt.step()
        if (t + 1) % 500 == 0:
            opt.param_groups[0]["lr"] *= 0.5
    return X.detach().cpu().numpy()


def main(rep="c58_8", mode="imaging", n_cal=3000, outer=1):
    from scipy.spatial import distance_matrix
    t0 = time.time()
    C, X_true, _ = load_hu(rep)
    S = cos_sim(C); np.fill_diagonal(S, 0.0)
    cache = os.path.join(os.path.dirname(__file__), "results_real", f"{rep}_cache.npz")
    X0 = np.load(cache)["E_umap"]

    near, far, thr = build_pairs(S)
    print(f"{rep} [{mode}]: {near.shape[0]} near edges, {far.shape[0]} far samples, "
          f"S_thresh={thr:.3f}")
    print(f"{'':22s} {'posRMSE':>9s} {'pairRMSE':>9s} {'J@10':>7s}")
    r0 = (position_rmse(X0, X_true), pairwise_rmse(X0, X_true),
          jaccard_curve(X0, X_true)[10])
    print(f"{'UMAP init':22s} {r0[0]:9.1f} {r0[1]:9.1f} {r0[2]:7.3f}")

    rng = np.random.RandomState(0)
    cal_idx = np.sort(rng.choice(S.shape[0], n_cal, replace=False))
    X = X0
    for it in range(outer):
        if mode == "imaging":
            dcal = distance_matrix(X_true[cal_idx], X_true).ravel()
        else:
            dcal = distance_matrix(X[cal_idx], X).ravel()
        cal = calibrate(S[cal_idx].ravel(), dcal)
        X = robust_mds(X, S, near, far, cal)
        rr = (position_rmse(X, X_true), pairwise_rmse(X, X_true),
              jaccard_curve(X, X_true)[10])
        print(f"{'robust-MDS pass ' + str(it):22s} {rr[0]:9.1f} {rr[1]:9.1f} {rr[2]:7.3f}")
    print(f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    a = sys.argv[1:]
    rep = a[0] if a and not a[0].startswith("-") else "c58_8"
    mode = "selfcons" if "--selfcons" in a else "imaging"
    outer = 4 if mode == "selfcons" else 1
    main(rep, mode=mode, outer=outer)
