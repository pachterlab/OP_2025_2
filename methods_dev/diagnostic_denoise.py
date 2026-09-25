"""Cheap make-or-break diagnostic: does low-rank denoising of the count matrix
restore the cosine-similarity <-> true-distance relationship on real data?

Ground-truth imaging positions are used ONLY to score the diagnostic, never in
the denoising.  If the metrics improve markedly as rank increases, the
anchor-free denoise -> graph -> layout pipeline is worth building; if not, the
noise is not low-rank-separable at this read depth.
"""
import sys
import time
import numpy as np
from scipy.spatial import distance_matrix
from scipy.stats import spearmanr
from sklearn.utils.extmath import randomized_svd

from real_data import load_hu


def row_cos(A):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    S = A @ A.T
    np.fill_diagonal(S, 0.0)
    return S


def knn_jaccard(S, Dt, k=10):
    n = S.shape[0]
    D = Dt + np.eye(n) * 1e18
    sim_nn = np.argpartition(-S, k, axis=1)[:, :k]
    tru_nn = np.argpartition(D, k, axis=1)[:, :k]
    j = np.empty(n)
    for b in range(n):
        a, c = set(sim_nn[b]), set(tru_nn[b])
        j[b] = 1.0 - len(a & c) / len(a | c)
    return j.mean()


def per_bead_rankcorr(S, Dt):
    out = []
    for b in range(S.shape[0]):
        r = spearmanr(S[b], -Dt[b]).correlation
        if np.isfinite(r):
            out.append(r)
    return float(np.mean(out))


def main(rep="c58_8", n_eval=2500):
    ranks = [0, 5, 10, 20, 30, 50, 75, 100, 150, 200]
    t0 = time.time()
    C, Xt, _ = load_hu(rep)
    B = C.shape[0]
    print(f"{rep}: C {C.shape}, UMI/bead {C.sum(1).mean():.0f}")

    A_full = np.log1p(C)
    print("randomized_svd ...", flush=True)
    U, s, Vt = randomized_svd(A_full, n_components=max(ranks), random_state=0)

    rng = np.random.RandomState(0)
    sub = np.sort(rng.choice(B, min(n_eval, B), replace=False))
    Dt = distance_matrix(Xt[sub], Xt[sub])
    iu = np.triu_indices(len(sub), 1)

    # near-pair mask for a "local only" view
    near = Dt[iu] < np.quantile(Dt[iu], 0.05)

    print(f"\n{'rank':>5} {'globalRho':>10} {'nearRho':>9} {'perBeadRho':>11} "
          f"{'kNN10_Jac':>10}")
    for r in ranks:
        Ar = A_full[sub] if r == 0 else (U[sub, :r] * s[:r]) @ Vt[:r]
        S = row_cos(Ar)
        g = spearmanr(S[iu], -Dt[iu]).correlation
        gn = spearmanr(S[iu][near], -Dt[iu][near]).correlation
        pb = per_bead_rankcorr(S, Dt)
        jac = knn_jaccard(S, Dt, k=10)
        print(f"{r:>5} {g:>10.3f} {gn:>9.3f} {pb:>11.3f} {jac:>10.3f}", flush=True)

    print(f"\n(Rho vs -distance: higher = similarity tracks closeness better. "
          f"kNN10_Jac: 0 = perfect, ~0.9 = random.)  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(*sys.argv[1:])
