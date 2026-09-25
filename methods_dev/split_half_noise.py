"""Split-half resampling check: is the near-neighbor identity itself
unstable under pure read-count sampling noise, independent of any
downstream method?

For each bead, thin its reads via independent binomial splitting into two
"half depth" replicates of the SAME underlying biology (same true position,
same true kernel, same everything -- only the random subset of molecules
sequenced differs). Rebuild cosine similarity from each half and find each
bead's top-k neighbors according to each half separately.

  - Jaccard overlap of the two half's neighbor SETS: pure noise-driven
    neighbor-identity instability (0 = perfectly stable, ~1 = totally
    unstable/random).
  - Mean TRUE distance (imaging) to whichever neighbors got picked from
    each half, vs the mean true distance to the actual/oracle k nearest
    neighbors: quantifies the noise-induced distance INFLATION in um,
    directly comparable to the ~49um pairRMSE floor found by every method
    tried on this data.

Usage:  python split_half_noise.py c58_8
"""
import os
import sys
import time
import numpy as np
from scipy.spatial import distance_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis_scripts"))
from common import cos_sim  # noqa: E402
from real_data import load_hu  # noqa: E402


def main(rep="c58_8", seed=0):
    t0 = time.time()
    C, X_true, U_true = load_hu(rep)
    C = np.asarray(C, dtype=np.int64)
    B = C.shape[0]
    print(f"{rep}: {B} beads, {int(C.sum()):,} reads, mean depth {C.sum(1).mean():.0f}/bead",
          flush=True)

    rng = np.random.RandomState(seed)
    half1 = rng.binomial(C, 0.5).astype(np.float64)
    half2 = (C - half1).astype(np.float64)
    print(f"split done ({time.time()-t0:.0f}s)  "
          f"mean half-depth {half1.sum(1).mean():.0f}/bead", flush=True)

    S1 = cos_sim(half1); np.fill_diagonal(S1, 0.0)
    S2 = cos_sim(half2); np.fill_diagonal(S2, 0.0)
    print(f"similarity done ({time.time()-t0:.0f}s)", flush=True)

    td = distance_matrix(X_true, X_true)
    td_ref = td + np.eye(B) * 1e18   # exclude self for the oracle ranking

    print(f"\n{'k':>4s} {'neighbor-Jaccard':>18s} {'mean-true-dist half1':>20s} "
          f"{'half2':>10s} {'oracle(true kNN)':>18s} {'corr(d1,d2)':>12s}", flush=True)
    for k in (5, 10, 15, 25):
        idx1 = np.argpartition(-S1, k, axis=1)[:, :k]
        idx2 = np.argpartition(-S2, k, axis=1)[:, :k]
        idx_oracle = np.argpartition(td_ref, k, axis=1)[:, :k]

        jac = np.mean([1 - len(set(idx1[i]) & set(idx2[i])) / len(set(idx1[i]) | set(idx2[i]))
                       for i in range(B)])
        d1 = np.array([td[i, idx1[i]].mean() for i in range(B)])
        d2 = np.array([td[i, idx2[i]].mean() for i in range(B)])
        d_oracle = np.array([td[i, idx_oracle[i]].mean() for i in range(B)])
        corr = np.corrcoef(d1, d2)[0, 1]

        print(f"{k:4d} {jac:18.3f} {d1.mean():20.1f} {d2.mean():10.1f} "
              f"{d_oracle.mean():18.1f} {corr:12.3f}", flush=True)

    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "c58_8")
