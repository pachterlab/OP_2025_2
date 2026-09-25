"""Patch stitching / group synchronization (ASAP-style; Cucuringu et al.,
Singer's "global positioning from local distances").

  1. patches = each anchor bead + its k nearest (by cosine similarity)
  2. embed each patch locally from SHORT-range physical distances only
     (d = 2 sigma sqrt(-log S) is trustworthy at patch scale; sigma from FWHM)
  3. for overlapping patches, estimate the relative rigid transform (O(2)) on
     shared beads
  4. rotation/reflection synchronization: leading eigenvectors of the block
     matrix of relative transforms -> a global frame per patch
  5. translation synchronization: least squares
  6. each bead = average of its position across the patches containing it

Far-pair distances never enter -> no long-range distortion.
"""
import os
import sys
import time
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, eye as speye
from scipy.sparse.linalg import eigsh, lsqr

from real_data import load_hu
from common import SIGMA, cos_sim, classical_mds
from position_error import position_rmse, pairwise_rmse
from common import jaccard_curve


def _procrustes_rigid(A, B):
    """Best R in O(2), t with  A ~= B R^T + t  (no scale)."""
    ca, cb = A.mean(0), B.mean(0)
    M = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    t = ca - cb @ R.T
    return R, t


def patch_stitch(S, sigma=SIGMA, k=30, n_anchor=4000, min_overlap=4, seed=0):
    n = S.shape[0]
    rng = np.random.RandomState(seed)
    anchors = np.sort(rng.choice(n, min(n_anchor, n), replace=False))
    order = np.argsort(-S, axis=1)

    members = []          # bead indices per patch
    local = []            # local 2D coords per patch
    for a in anchors:
        idx = np.r_[a, order[a, 1:k + 1]]
        sub = np.clip(S[np.ix_(idx, idx)], 1e-6, 1 - 1e-9)
        d = 2.0 * sigma * np.sqrt(np.maximum(-np.log(sub), 0.0))
        np.fill_diagonal(d, 0.0)
        members.append(idx)
        local.append(classical_mds(d))

    # bead -> list of (patch, position-in-patch)
    where = [[] for _ in range(n)]
    for p, idx in enumerate(members):
        for pos, b in enumerate(idx):
            where[b].append((p, pos))

    # patch-patch edges via shared beads
    P = len(members)
    rel = {}
    for b in range(n):
        ps = where[b]
        for x in range(len(ps)):
            for y in range(x + 1, len(ps)):
                pa, _ = ps[x]
                pb, _ = ps[y]
                key = (pa, pb) if pa < pb else (pb, pa)
                rel.setdefault(key, set()).add(b)

    edges = [(pa, pb, list(sh)) for (pa, pb), sh in rel.items() if len(sh) >= min_overlap]
    if not edges:
        raise RuntimeError("no overlapping patches; raise k or n_anchor")

    # --- rotation/reflection synchronization (O(2) block eigen) ---
    Bm = lil_matrix((2 * P, 2 * P))
    deg = np.zeros(P)
    for pa, pb, sh in edges:
        Aa = local[pa][[list(members[pa]).index(b) for b in sh]]
        Ab = local[pb][[list(members[pb]).index(b) for b in sh]]
        R, _ = _procrustes_rigid(Aa, Ab)                # Aa ~= Ab R^T
        Bm[2 * pa:2 * pa + 2, 2 * pb:2 * pb + 2] = R
        Bm[2 * pb:2 * pb + 2, 2 * pa:2 * pa + 2] = R.T
        deg[pa] += 1
        deg[pb] += 1
    Dinv = speye(2 * P).tolil()
    for p in range(P):
        Dinv[2 * p, 2 * p] = 1.0 / max(deg[p], 1)
        Dinv[2 * p + 1, 2 * p + 1] = 1.0 / max(deg[p], 1)
    w, V = eigsh((Dinv @ Bm.tocsr()) + speye(2 * P) * 3.0, k=2, which="LA")
    Rots = []
    for p in range(P):
        blk = V[2 * p:2 * p + 2, :]
        U, _, Vt = np.linalg.svd(blk)
        Rp = U @ Vt
        if np.linalg.det(Rp) < 0:                       # keep orientation choice consistent
            Rp = U @ np.diag([1, -1]) @ Vt
        Rots.append(Rp)

    # rotate every patch into the common frame
    rot_local = [local[p] @ Rots[p].T for p in range(P)]

    # --- translation synchronization + bead placement (alternating LS) ---
    tvec = np.zeros((P, 2))
    Y = np.zeros((n, 2))
    for _ in range(6):
        for b in range(n):
            acc = [rot_local[p][pos] + tvec[p] for (p, pos) in where[b]]
            if acc:
                Y[b] = np.mean(acc, axis=0)
        for p in range(P):
            idx = members[p]
            tvec[p] = np.mean(Y[idx] - rot_local[p], axis=0)
    for b in range(n):
        acc = [rot_local[p][pos] + tvec[p] for (p, pos) in where[b]]
        if acc:
            Y[b] = np.mean(acc, axis=0)
    return Y


def main(rep="c58_8", k=30, n_anchor=4000):
    t0 = time.time()
    C, X_true, _ = load_hu(rep)
    S = cos_sim(C); np.fill_diagonal(S, 0.0)
    cache = os.path.join(os.path.dirname(__file__), "results_real", f"{rep}_cache.npz")
    X0 = np.load(cache)["E_umap"]

    print(f"{'':22s} {'posRMSE':>9s} {'pairRMSE':>9s} {'J@10':>7s}")
    print(f"{'UMAP':22s} {position_rmse(X0, X_true):9.1f} "
          f"{pairwise_rmse(X0, X_true):9.1f} {jaccard_curve(X0, X_true)[10]:7.3f}")

    Y = patch_stitch(S, k=k, n_anchor=n_anchor)
    print(f"{'patch-stitch':22s} {position_rmse(Y, X_true):9.1f} "
          f"{pairwise_rmse(Y, X_true):9.1f} {jaccard_curve(Y, X_true)[10]:7.3f}")
    print(f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "c58_8")
