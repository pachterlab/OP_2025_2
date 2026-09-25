"""Ceiling check: Route 1 (generative MAP-EM) with ORACLE imaging sender
positions on the real Hu data.  The imaging U is used to bound what the model
could do if geometry estimation were solved -- it is NOT a deployable method.
"""
import os
import sys
import numpy as np

from real_data import load_hu
from common import SIGMA, classical_mds, cos_sim, per_bead_mse, summary
from route1_poisson import route1

rep = sys.argv[1] if len(sys.argv) > 1 else "c58_8"
C, X_true, U_true = load_hu(rep)
print(f"{rep}: C {C.shape}")

cache = os.path.join(os.path.dirname(__file__), "results_real", f"{rep}_cache.npz")
if os.path.exists(cache):
    X0 = np.load(cache)["E_umap"]
    print("init: cached UMAP")
else:
    S = cos_sim(C); np.fill_diagonal(S, 1.0)
    X0 = classical_mds(1.0 - S)
    print("init: classical MDS of 1-S")

E, U, hist = route1(C, SIGMA, X0=X0, oracle_U=U_true,
                    outer=15, steps_x=900, steps_u=0, lr=3.0)

umap_mse = per_bead_mse(X0, X_true)
row, jac = summary("Route1 oracle-U (real)", E, X_true, umap_mse=umap_mse)
print("\nRESULT:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
print("jaccard curve:", {k: round(v, 3) for k, v in jac.items()})
