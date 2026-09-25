"""Hard mutual/union k-NN restricted accumap: instead of a soft S-weighted
(or IRLS-reweighted) objective over ALL B*(B-1) pairs, DISCARD every pair
outside each bead's k-nearest-neighbor set entirely (weight exactly 0, not
just small) -- structurally much closer to what UMAP actually does (a graph
built only from each point's k nearest neighbors, no metric target for
anything else) than the earlier "self-consistent" accumap variants, which
still touch every pair, just with reduced influence for uncertain ones.

Isolates: is it *soft downweighting vs hard exclusion* of far pairs that
explains why UMAP recovers structure and accumap-family methods don't?
"""
import sys
import time
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

sys.path.insert(0, "/home/coakes/scope_recon")
from scope_bench import load_scope, cos_sim, umap_embed, _monotone_map  # noqa: E402
sys.path.insert(0, "/home/coakes/scope_recon/repos")
from accumap import correct_locations  # noqa: E402


def build_knn_mask(S, k=25):
    """Symmetric (union) k-NN adjacency: edge (i,j) kept if j is among i's
    top-k neighbors OR i is among j's -- the same union construction UMAP's
    fuzzy simplicial set uses, not a stricter mutual/intersection graph."""
    B = S.shape[0]
    idx = np.argpartition(-S, k + 1, axis=1)[:, :k + 1]
    mask = np.zeros((B, B), dtype=bool)
    rows = np.repeat(np.arange(B), k + 1)
    mask[rows, idx.ravel()] = True
    mask |= mask.T
    np.fill_diagonal(mask, False)
    return mask


def knn_accumap(C, S, init, k=25, n_outer=1, verbose=True):
    mask = build_knn_mask(S, k)
    deg = mask.sum(1)
    if verbose:
        print(f"  k-NN graph: k={k}  mean degree={deg.mean():.1f}  "
              f"edges={mask.sum()//2:,} / {S.shape[0]*(S.shape[0]-1)//2:,} "
              f"possible ({100*mask.sum()/2/(S.shape[0]*(S.shape[0]-1)//2):.1f}%)",
              flush=True)

    X = np.asarray(init, dtype=float).copy()
    for it in range(n_outer):
        DX = distance_matrix(X, X)
        s_flat = S[mask]
        dx_flat = DX[mask]
        g = _monotone_map(s_flat, dx_flat)
        Dt = g(S)
        np.fill_diagonal(Dt, 0.0)
        W = np.where(mask, S, 0.0)   # HARD zero outside the k-NN graph

        df = correct_locations(pd.DataFrame(X, columns=["xcoord", "ycoord"]),
                               Dt, weight=W, gauss_seidel=True,
                               correct_order="sorted", verbose_step=10 ** 9)
        X = np.nan_to_num(df.values)
        stress = float(np.mean((DX[mask] - Dt[mask]) ** 2))
        if verbose:
            print(f"    outer {it}: stress={stress:.4g}", flush=True)
    return X


if __name__ == "__main__":
    t0 = time.time()
    C, bc2idx = load_scope("letter_E")
    print(f"letter_E: {C.shape}, {int(C.sum()):,} reads", flush=True)

    S = cos_sim(C); np.fill_diagonal(S, 0.0)
    E_umap = umap_embed(C)
    print(f"UMAP done ({time.time()-t0:.0f}s)", flush=True)

    results = {}
    for k in (15, 25, 50):
        print(f"k={k}:", flush=True)
        X = knn_accumap(C, S, E_umap, k=k, n_outer=1)
        results[f"k{k}"] = X
        print(f"  done ({time.time()-t0:.0f}s)", flush=True)

    np.savez("/home/coakes/scope_recon/results/letter_E_knnaccu_cache.npz",
            E_umap=E_umap, **results)
    print(f"saved, total {time.time()-t0:.0f}s", flush=True)
