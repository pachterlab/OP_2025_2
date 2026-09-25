"""Jackpot / second-population diagnostic on real c58_8.

Uses imaging positions for BOTH receiver and sender (d_bo known for every
count) -- diagnostic only, never in a method.

1. Characterise the count distribution vs true (b,o) distance: where do the
   high-count cells sit, and how many reads are at distances where diffusion
   predicts ~zero.
2. Test hard fixes -- winsorize per-cell counts; oracle removal of long-range
   reads -- and see whether Spearman(S,-d) and kNN-10 Jaccard recover.
"""
import os
import sys
import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

from real_data import load_hu
from common import cos_sim

SIGMA = 123.0 / np.sqrt(8.0 * np.log(2.0))

rep = sys.argv[1] if len(sys.argv) > 1 else "c58_8"
C, X, U = load_hu(rep)
C = C.astype(np.float64)
B, O = C.shape
print(f"{rep}: C {C.shape}  {int(C.sum()):,} reads  mean/pair {C.mean():.3f}")

Dbo = cdist(X, U)                                   # true receiver-sender dist
td = distance_matrix(X, X)
n = B
iu = np.triu_indices(n, 1)
samp = np.random.RandomState(0).choice(len(iu[0]), 1_000_000, replace=False)


def report(Cx, tag):
    S = cos_sim(Cx); np.fill_diagonal(S, 0.0)
    rho = spearmanr(S[iu][samp], -td[iu][samp]).correlation
    k = 10
    sn = np.argpartition(-S, k + 1, axis=1)[:, :k + 1]
    tn = np.argpartition(td + np.eye(n) * 1e18, k, axis=1)[:, :k]
    j = np.mean([1 - len((set(sn[i]) - {i}) & set(tn[i])) /
                 len((set(sn[i]) - {i}) | set(tn[i])) for i in range(n)])
    print(f"  {tag:34s} reads={int(Cx.sum()):>11,}  rho={rho:.3f}  kNN10-Jac={j:.3f}")
    return rho, j


# ---- 1. count distribution vs true distance -------------------------------
print("\n[1] count vs true (b,o) distance")
vals, cnts = np.unique(C.astype(int), return_counts=True)
tot = C.sum()
print("  per-cell count histogram (value: #cells, %reads):")
for v, c in zip(vals, cnts):
    if v == 0 or v > 30:
        continue
    print(f"     {int(v):3d}: {c:>10,}  {100*v*c/tot:5.2f}%")
hi = C >= 30
print(f"     >=30: {hi.sum():>10,}  {100*C[hi].sum()/tot:5.2f}%   "
      f"(max cell {int(C.max())})")

bins = [0, 100, 250, 500, 1000, 4000]
for lo, hiB in zip(bins[:-1], bins[1:]):
    m = (Dbo >= lo) & (Dbo < hiB)
    reads = C[m].sum()
    kern = np.exp(-((lo + hiB) / 2) ** 2 / (2 * SIGMA ** 2))
    print(f"  d_bo {lo:>4}-{hiB:<4} um: {100*reads/tot:5.1f}% of reads, "
          f"mean count {C[m].mean():.4f}, cells>=2 {(C[m] >= 2).sum():>8,}, "
          f"kernel~{kern:.1e}")

far = Dbo > 300
print(f"\n  reads at d_bo > 300 um (diffusion predicts ~0): "
      f"{100*C[far].sum()/tot:.1f}% of all reads, in {(C[far] > 0).sum():,} cells")
# per-bead: fraction of a bead's reads that are 'far'
frac_far = C[far].sum() and (C * far).sum(1) / np.maximum(C.sum(1), 1)
print(f"  per-bead far-read fraction: mean {frac_far.mean():.3f} "
      f"median {np.median(frac_far):.3f} p90 {np.percentile(frac_far, 90):.3f}")


# ---- 2. hard-fix tests --------------------------------------------------
print("\n[2] fixes")
report(C, "baseline")

for T in (10, 5, 3):
    report(np.minimum(C, T), f"winsorize counts at {T}")

for Dfar in (800, 400, 200):
    Cx = C.copy()
    Cx[Dbo > Dfar] = 0.0
    report(Cx, f"ORACLE remove reads d_bo>{Dfar}um")

# promiscuous-bead removal: zero the far reads of the worst beads
prom = (C * far).sum(1) / np.maximum(C.sum(1), 1)
for q in (0.9, 0.75):
    thr = np.quantile(prom, q)
    Cx = C.copy()
    bad = prom > thr
    Cx[bad] = Cx[bad] * (~far[bad])            # keep only near reads for bad beads
    report(Cx, f"near-only for top {int((1-q)*100)}% promiscuous beads")
