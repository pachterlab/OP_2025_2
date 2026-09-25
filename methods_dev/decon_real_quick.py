"""Fast real-data check: does count-only decontamination find ambient structure
in c58_8, and does removing it recover the similarity signal?  Skips the slow
UMAP/accumap re-scoring -- reports a_hat, Spearman(S,-d), kNN-10 Jaccard
before/after.
"""
import os
import sys
import numpy as np
from scipy.spatial import distance_matrix
from scipy.stats import spearmanr

from real_data import load_hu
from common import cos_sim
from decontaminate import decontaminate

rep = sys.argv[1] if len(sys.argv) > 1 else "c58_8"
C, X_true, _ = load_hu(rep)
td = distance_matrix(X_true, X_true)
n = C.shape[0]
iu = np.triu_indices(n, 1)
samp = np.random.RandomState(0).choice(len(iu[0]), 1_000_000, replace=False)


def diag(C, tag):
    S = cos_sim(C); np.fill_diagonal(S, 0.0)
    rho = spearmanr(S[iu][samp], -td[iu][samp]).correlation
    k = 10
    sim_nn = np.argpartition(-S, k + 1, axis=1)[:, :k + 1]
    tru_nn = np.argpartition(td + np.eye(n) * 1e18, k, axis=1)[:, :k]
    j = np.mean([1.0 - len((set(sim_nn[b]) - {b}) & set(tru_nn[b])) /
                 len((set(sim_nn[b]) - {b}) | set(tru_nn[b])) for b in range(n)])
    print(f"  {tag:8s}  rho(S,-d)={rho:.3f}   kNN10-Jaccard={j:.3f}")


print(f"{rep}: C {C.shape}")
diag(C, "before")
Chat, ah = decontaminate(C, verbose=True)
print(f"  a_hat: mean={ah.mean():.3f}  median={np.median(ah):.3f}  "
      f"p10={np.percentile(ah,10):.3f}  p90={np.percentile(ah,90):.3f}  "
      f"frac>0.1={np.mean(ah>0.1):.2f}  frac>0.3={np.mean(ah>0.3):.2f}")
diag(Chat, "after")
np.savez(os.path.join(os.path.dirname(__file__), "results_real", f"{rep}_ahat.npz"),
         a_hat=ah, total=C.sum(1))
