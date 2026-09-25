"""Direct test: does read depth predict the UMAP/accumap reconstruction
ceiling across the real replicates, independent of the synthetic split-half
argument? Reuses the already-computed cross-replicate transfer results
(UMAP and accumap-k5 pairRMSE for all 4 replicates) and just adds each
replicate's mean per-bead read depth.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from real_data import load_hu  # noqa: E402

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "OP_2025_2", "methods_dev", "results")

# from cross_replicate_transfer.json (c58_8's own numbers added from the
# established k=5 result on that replicate)
pair_rmse = {
    "c58_8":  {"umap": 44.8, "accumap_k5": 49.3},
    "c58_4":  {"umap": 48.20542520535246, "accumap_k5": 49.55715721545196},
    "c58_5":  {"umap": 41.94287821768313, "accumap_k5": 45.033974967363775},
    "c58_16": {"umap": 76.59275770524833, "accumap_k5": 84.12446497878504},
}

print(f"{'rep':8s} {'n_beads':>8s} {'mean_depth':>11s} {'UMAP':>7s} {'accu_k5':>8s} {'gap':>7s}")
depths, gaps = [], []
for rep in pair_rmse:
    C, X_true, U_true = load_hu(rep)
    C = np.asarray(C, float)
    depth = C.sum(1).mean()
    umap_v, accu_v = pair_rmse[rep]["umap"], pair_rmse[rep]["accumap_k5"]
    gap = accu_v - umap_v
    depths.append(depth); gaps.append(gap)
    print(f"{rep:8s} {C.shape[0]:8d} {depth:11.0f} {umap_v:7.1f} {accu_v:8.1f} {gap:7.1f}")

depths, gaps = np.array(depths), np.array(gaps)
r = np.corrcoef(depths, gaps)[0, 1]
r_umap = np.corrcoef(depths, [pair_rmse[rep]["umap"] for rep in pair_rmse])[0, 1]
print(f"\ncorr(depth, accu-umap gap) = {r:.3f}")
print(f"corr(depth, UMAP pairRMSE itself) = {r_umap:.3f}")
