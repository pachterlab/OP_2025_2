"""Quantitative check: does kNN-accumap actually structurally agree with
UMAP (not just look similar), and by how much more than the earlier
accumap variants? All these embeddings share the same bead ordering (all
built from the same load_scope("letter_E") call), so a per-bead kNN
comparison is directly valid -- no alignment issue like SCOPE's own output
(different internal bead ordering/subset, no barcode labels to join on).

Metric: kNN-k Jaccard distance to UMAP's own neighbor graph (0 = identical
neighbor sets, ~0.9 = random) -- the same metric used throughout this
session for structural agreement without ground truth.
"""
import numpy as np
from sklearn.neighbors import KDTree


def knn_jaccard(A, B, k=10):
    ta, tb = KDTree(A), KDTree(B)
    _, na = ta.query(A, k=k + 1)
    _, nb = tb.query(B, k=k + 1)
    vals = np.empty(len(A))
    for i in range(len(A)):
        sa, sb = set(na[i, 1:]), set(nb[i, 1:])
        vals[i] = 1.0 - len(sa & sb) / len(sa | sb)
    return float(vals.mean())


z0 = np.load("/home/coakes/scope_recon/results/letter_E_cache2.npz")
zd = np.load("/home/coakes/scope_recon/results/letter_E_decon_cache.npz")
zr1 = np.load("/home/coakes/scope_recon/results/letter_E_route1_cache.npz")
zr2 = np.load("/home/coakes/scope_recon/results/letter_E_route1v2_cache.npz")
zk = np.load("/home/coakes/scope_recon/results/letter_E_knnaccu_cache.npz")

E_umap = z0["E_umap"]   # reference

methods = {
    "accumap (self-consistent, 1-pass)": z0["E_1pass"],
    "accumap + decon": zd["E_accu"],
    "Route1 v1 (unanchored)": zr1["X_r1"],
    "Route1 v2 (anchored+bg)": zr2["X_r1"],
    "kNN-accumap k=15": zk["k15"],
    "kNN-accumap k=25": zk["k25"],
    "kNN-accumap k=50": zk["k50"],
    "UMAP + decon (self-agreement sanity check)": zd["E_umap"],
}

print(f"{'method':42s} {'kNN10-Jac vs UMAP':>18s}  {'kNN25-Jac vs UMAP':>18s}")
for name, E in methods.items():
    j10 = knn_jaccard(E_umap, E, k=10)
    j25 = knn_jaccard(E_umap, E, k=25)
    print(f"{name:42s} {j10:18.3f}  {j25:18.3f}")

# random-baseline reference: shuffle UMAP's own points
rng = np.random.RandomState(0)
perm = rng.permutation(len(E_umap))
j_rand = knn_jaccard(E_umap, E_umap[perm], k=10)
print(f"\n(random permutation baseline, kNN10-Jac: {j_rand:.3f})")
