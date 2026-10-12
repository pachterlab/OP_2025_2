"""Idealized simulation and metrics of the original accumap (baseline).

Reproduces the idealized simulation (sigma = 52 um, 10k receiver + 10k sender
beads) and the metrics used in the main benchmark: Procrustes-aligned per-bead
squared distance error, and the Jaccard-vs-X neighborhood curve.
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import procrustes, distance_matrix
from sklearn.neighbors import KDTree

SIGMA = 123.0 / np.sqrt(8.0 * np.log(2.0))   # ~= 52.23 um, from the reported FWHM
RADIUS = 1500.0
N_RECEIVER = 10000
N_SENDER = 10000
UMI = 2500
SEED = 42


def _disc(rng, n, radius):
    r = np.sqrt(rng.uniform(size=n)) * radius
    t = rng.uniform(size=n) * 2.0 * np.pi
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def simulate(seed=SEED, n_receiver=N_RECEIVER, n_sender=N_SENDER,
             radius=RADIUS, sigma=SIGMA, umi=UMI):
    """Return (C, X_true, U_true): counts (n_receiver x n_sender), receiver
    coords, sender coords. Matches New_Simulation.ipynb exactly."""
    rng = np.random.RandomState(seed)
    x_true = _disc(rng, n_receiver, radius)
    u_true = _disc(rng, n_sender, radius)
    d = cdist(x_true, u_true)
    p = np.exp(-d ** 2 / (2.0 * sigma ** 2))
    p /= p.sum(axis=1, keepdims=True)
    C = np.empty((n_receiver, n_sender), dtype=np.float64)
    for i in range(n_receiver):
        C[i] = rng.multinomial(umi, p[i])
    return C, x_true, u_true


def cos_sim(C):
    """Receiver-receiver cosine similarity of log1p counts (as in accumap)."""
    L = np.log1p(np.asarray(C, dtype=np.float64))
    L /= np.linalg.norm(L, axis=1, keepdims=True)
    S = L @ L.T
    np.clip(S, 0.0, 1.0, out=S)
    return S


def classical_mds(D, dim=2):
    """Top-`dim` classical MDS embedding of a distance matrix D."""
    D = np.asarray(D, dtype=np.float64)
    D2 = D ** 2
    n = D2.shape[0]
    rmean = D2.mean(axis=1, keepdims=True)
    gmean = D2.mean()
    B = -0.5 * (D2 - rmean - rmean.T + gmean)
    from scipy.sparse.linalg import eigsh
    w, V = eigsh(B, k=dim, which="LA")
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    return V * np.sqrt(np.maximum(w, 0.0))


def procrustes_align(recon, truth):
    """Align `recon` onto `truth` (rotation/reflection/scale/translation),
    returning coords in `truth`'s scale. Mirrors get_pa() in the notebooks."""
    truth = np.asarray(truth, dtype=np.float64)
    recon = np.asarray(recon, dtype=np.float64)
    _, m2, _ = procrustes(truth, recon)
    tc = truth - truth.mean(axis=0)
    scale = np.sqrt(np.trace(tc @ tc.T))
    scale2 = scale / np.sqrt(np.trace(m2 @ m2.T))
    return m2 * scale2


def per_bead_mse(recon, truth):
    """Per-bead mean squared error of pairwise distances, after alignment."""
    a = procrustes_align(recon, truth)
    dt = distance_matrix(truth, truth)
    dr = distance_matrix(a, a)
    return np.mean((dt - dr) ** 2, axis=0)


def jaccard_curve(recon, truth, ks=range(2, 25)):
    """Mean Jaccard distance between true and reconstructed X nearest
    neighbors, for X = k-1 over `ks`. Returns {X: mean_jaccard}."""
    tt = KDTree(np.asarray(truth))
    rt = KDTree(np.asarray(recon))
    out = {}
    for k in ks:
        _, tn = tt.query(truth, k=k)
        _, rn = rt.query(recon, k=k)
        vals = np.empty(len(truth))
        for i in range(len(truth)):
            a = set(tn[i, 1:])
            b = set(rn[i, 1:])
            vals[i] = 1.0 - len(a & b) / len(a | b)
        out[k - 1] = float(vals.mean())
    return out


def summary(name, recon, truth, umap_mse=None):
    mse = per_bead_mse(recon, truth)
    jac = jaccard_curve(recon, truth)
    row = {
        "method": name,
        "mse_mean": float(mse.mean()),
        "mse_median": float(np.median(mse)),
        "mse_max": float(mse.max()),
        "jaccard_10": jac.get(10, np.nan),
        "jaccard_meanX": float(np.mean(list(jac.values()))),
    }
    if umap_mse is not None:
        row["frac_beats_umap"] = float((mse < umap_mse).mean())
    return row, jac
