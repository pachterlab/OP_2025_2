from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree


# ── internals ─────────────────────────────────────────────────────────────────

def _build_tree(df: pd.DataFrame) -> tuple[cKDTree, np.ndarray]:
    coords = df.iloc[:, :2].to_numpy(dtype=float)
    return cKDTree(coords), coords


def _adjacency(tree: cKDTree, radius: float) -> sp.csr_matrix:
    """Sparse boolean adjacency matrix (self-loops excluded)."""
    D = tree.sparse_distance_matrix(tree, max_distance=radius, output_type="coo_matrix")
    A = D.tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    A.data[:] = 1
    return A


def _jaccard_from_matrices(A1: sp.csr_matrix, A2: sp.csr_matrix) -> np.ndarray:
    """Per-row Jaccard distance between two boolean adjacency matrices."""
    inter = np.asarray(A1.multiply(A2).sum(axis=1)).ravel()
    s1 = np.asarray(A1.sum(axis=1)).ravel()
    s2 = np.asarray(A2.sum(axis=1)).ravel()
    union = s1 + s2 - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, 1.0 - inter / union, 0.0)


def _threshold(D_csr: sp.csr_matrix, radius: float) -> sp.csr_matrix:
    """Threshold a precomputed distance matrix to a smaller radius."""
    A = D_csr.copy()
    A.data[A.data > radius] = 0
    A.eliminate_zeros()
    A.data[:] = 1
    return A


def _make_result(df1, df2, A1, A2, jd) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x1": df1.iloc[:, 0].values,
            "y1": df1.iloc[:, 1].values,
            "x2": df2.iloc[:, 0].values,
            "y2": df2.iloc[:, 1].values,
            "n_neighbors1": np.asarray(A1.sum(axis=1)).ravel().astype(int),
            "n_neighbors2": np.asarray(A2.sum(axis=1)).ravel().astype(int),
            "jaccard_dist": jd,
        },
        index=df1.index,
    )


# ── public API ────────────────────────────────────────────────────────────────

def compare_neighborhoods(
    df1: pd.DataFrame, df2: pd.DataFrame, radius: float
) -> pd.DataFrame:
    """
    For each point (row), compute the Jaccard distance between its neighborhood
    in df1 and its neighborhood in df2.

    Both dataframes must have the same number of rows. The first two columns are
    treated as (x, y) coordinates; any additional columns are ignored.

    Parameters
    ----------
    df1, df2 : DataFrames with columns [x, y, ...]
    radius   : Distance threshold for neighborhood membership

    Returns
    -------
    DataFrame with columns:
        x1, y1        – coordinates from df1
        x2, y2        – coordinates from df2
        n_neighbors1  – neighborhood size in df1 (excluding self)
        n_neighbors2  – neighborhood size in df2 (excluding self)
        jaccard_dist  – Jaccard distance in [0, 1]; 0 = identical neighborhoods
    """
    if len(df1) != len(df2):
        raise ValueError(
            f"Dataframes must have the same number of rows "
            f"(got {len(df1)} and {len(df2)})"
        )

    tree1, _ = _build_tree(df1)
    tree2, _ = _build_tree(df2)
    A1 = _adjacency(tree1, radius)
    A2 = _adjacency(tree2, radius)
    jd = _jaccard_from_matrices(A1, A2)
    return _make_result(df1, df2, A1, A2, jd)


def sweep_radii(
    df1: pd.DataFrame, df2: pd.DataFrame, radii: list[float]
) -> dict[float, pd.DataFrame]:
    """
    Like compare_neighborhoods, but efficiently over multiple radii.

    Builds the KD-trees once and computes a single sparse distance matrix per
    dataframe at max(radii). Each smaller radius is handled by thresholding
    that matrix, avoiding redundant tree queries.

    Parameters
    ----------
    df1, df2 : DataFrames with columns [x, y, ...]
    radii    : List of distance thresholds to evaluate

    Returns
    -------
    Dict mapping each radius to its compare_neighborhoods result DataFrame.
    """
    if len(df1) != len(df2):
        raise ValueError(
            f"Dataframes must have the same number of rows "
            f"(got {len(df1)} and {len(df2)})"
        )

    max_r = max(radii)
    tree1, _ = _build_tree(df1)
    tree2, _ = _build_tree(df2)

    # One expensive query per dataframe at max radius
    D1 = tree1.sparse_distance_matrix(tree1, max_distance=max_r, output_type="coo_matrix").tocsr()
    D2 = tree2.sparse_distance_matrix(tree2, max_distance=max_r, output_type="coo_matrix").tocsr()
    for D in (D1, D2):
        D.setdiag(0)
        D.eliminate_zeros()

    results = {}
    for r in radii:
        A1 = _threshold(D1, r)
        A2 = _threshold(D2, r)
        jd = _jaccard_from_matrices(A1, A2)
        results[r] = _make_result(df1, df2, A1, A2, jd)

    return results


# ── demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    rng = np.random.default_rng(42)
    n = 5_000
    df1 = pd.DataFrame({"x": rng.uniform(0, 100, n), "y": rng.uniform(0, 100, n)})
    df2 = df1 + rng.normal(0, 0.5, df1.shape)
    df2.columns = ["x", "y"]

    # Single radius
    t0 = time.perf_counter()
    result = compare_neighborhoods(df1, df2, radius=3.0)
    print(f"Single radius (n={n}): {time.perf_counter() - t0:.3f}s")
    print(f"  Mean Jaccard: {result['jaccard_dist'].mean():.4f}")

    # Radius sweep
    radii = [1.0, 2.0, 3.0, 5.0, 8.0]
    t0 = time.perf_counter()
    sweep = sweep_radii(df1, df2, radii)
    print(f"\nSweep over {len(radii)} radii (n={n}): {time.perf_counter() - t0:.3f}s")
    for r, df in sweep.items():
        print(f"  r={r:.1f}  mean Jaccard={df['jaccard_dist'].mean():.4f}")
