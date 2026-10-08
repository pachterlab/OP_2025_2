"""The paper's evaluation: per-bead position error after similarity alignment (median, 90th percentile),
pairwise-distance RMSE and the 10-nearest-neighbor Jaccard distance (`evaluate`), and `neighbors_kept`."""
import numpy as np
from scipy.spatial import distance_matrix


def similarity_align(recon, truth):
    """Align `recon` to `truth` by a similarity transform (translation, rotation, reflection, one global scale),
    least squares. Returns (aligned recon, centered truth). Rows with non-finite coordinates are dropped from both."""
    recon, truth = np.asarray(recon, float), np.asarray(truth, float)
    ok = np.isfinite(recon).all(1) & np.isfinite(truth).all(1)
    P, T = recon[ok] - recon[ok].mean(0), truth[ok] - truth[ok].mean(0)
    U, s, Vt = np.linalg.svd(P.T @ T)
    return P @ (U @ Vt) * (s.sum() / np.sum(P ** 2)), T


def neighbors_kept(recon, truth, k=10):
    """For each bead, how many of its k nearest neighbors in `truth` are also among its k nearest in `recon`."""
    from scipy.spatial import cKDTree
    A, T = similarity_align(recon, truth)
    nt = cKDTree(T).query(T, k + 1)[1][:, 1:]
    na = cKDTree(A).query(A, k + 1)[1][:, 1:]
    return np.array([len(set(a) & set(b)) for a, b in zip(na, nt)])


def evaluate(recon, truth, n_pairs=2_000_000, seed=0):
    """The paper's metrics after similarity alignment: median and 90th-percentile per-bead position error,
    pairwise-distance RMSE over n_pairs random bead pairs, and mean 10-nearest-neighbor Jaccard distance."""
    from scipy.spatial import cKDTree
    A, T = similarity_align(recon, truth)
    e = np.linalg.norm(A - T, axis=1)
    rng = np.random.RandomState(seed)
    i, j = rng.randint(0, len(T), n_pairs), rng.randint(0, len(T), n_pairs)
    pr = np.sqrt(np.mean((np.linalg.norm(A[i] - A[j], axis=1) - np.linalg.norm(T[i] - T[j], axis=1)) ** 2))
    nt = cKDTree(T).query(T, 11)[1][:, 1:]
    na = cKDTree(A).query(A, 11)[1][:, 1:]
    jac = np.mean([1 - len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(na, nt)])
    return {"median": float(np.median(e)), "p90": float(np.quantile(e, 0.9)), "pairwise_rmse": float(pr), "jaccard10": float(jac)}
