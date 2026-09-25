"""Fast shape/plumbing check on a tiny dataset (not a real benchmark)."""
import numpy as np
from common import SIGMA, simulate, cos_sim, classical_mds, summary
from route2_surrogate import route2
from route1_poisson import route1
from variant1_marginal import variant1

C, Xt, Ut = simulate(seed=42, n_receiver=600, n_sender=600, umi=800)
print("C", C.shape, "sum/row", C.sum(1).mean())
S = cos_sim(C); np.fill_diagonal(S, 1.0)
X0 = classical_mds(1.0 - S)
print("classical_mds ok", X0.shape)

Xr2, h2 = route2(C, n_outer=2)
print("route2 ok", Xr2.shape, h2[-1])

Xv1, pv1 = variant1(C, init=None, warmup=100, steps=300, decay_every=150,
                    device="cuda", chunk=256)
print("variant1 ok", Xv1.shape, pv1)

Xr1, Ur1, h1 = route1(C, SIGMA, X0=X0, oracle_U=None, outer=2, steps_x=60,
                      steps_u=60, lr=2.0, device="cuda", chunk=256)
print("route1 joint ok", Xr1.shape, h1[-1])

for name, E in [("r2", Xr2), ("v1", Xv1), ("r1", Xr1), ("cmds", X0)]:
    row, _ = summary(name, E, Xt)
    print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
print("SMOKE OK")
