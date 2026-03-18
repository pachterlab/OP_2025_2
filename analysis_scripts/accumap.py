#Pure numpy
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
import uuid
from scipy.spatial import procrustes
import pandas as pd

try:
    import accumap_rs as _rs
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False


# ============================================================
# Classical MDS initialisation (Torgerson scaling)
# ============================================================

def classical_mds_init(truth_arr):
    """
    Return 2-D coordinates via classical MDS (Torgerson scaling).

    Gives a closed-form near-optimal starting point for SMACOF in O(N³)
    via eigendecomposition of the double-centred squared-distance matrix.

    Parameters
    ----------
    truth_arr : np.ndarray (N, N)
        Symmetric matrix of target pairwise distances.

    Returns
    -------
    X : np.ndarray (N, 2)
        Initial coordinates.
    """
    n = len(truth_arr)
    D2 = truth_arr.astype(np.float64) ** 2
    H = np.eye(n) - np.ones((n, n)) / n          # centering matrix
    B = -0.5 * H @ D2 @ H
    B = (B + B.T) / 2                             # enforce symmetry

    vals, vecs = np.linalg.eigh(B)
    # Top-2 eigenpairs (eigh returns ascending order)
    idx = np.argsort(vals)[::-1][:2]
    X = vecs[:, idx] * np.sqrt(np.maximum(vals[idx], 0.0))
    return X


# ============================================================
# SMACOF – Scaling by MAjorizing a COmplicated Function
# ============================================================

def smacof(truth_arr, init=None, n_iter=300, tol=1e-4,
           weight_arr=None, filter_counts=0, verbose_step=0):
    """
    Metric MDS via the SMACOF algorithm.

    Minimises stress = Σ w_ij (delta_ij − d_ij(X))² using the
    guaranteed-monotone Guttman-transform update.  Converges in far
    fewer iterations than per-point Powell optimisation.

    For uniform weights the update is a single matrix multiply:
        X_new = (1/n) · B(X) · X
    For non-uniform weights a pre-factored Cholesky solve is used.

    Parameters
    ----------
    truth_arr    : (N, N) target distances.
    init         : (N, 2) starting coordinates; classical MDS if None.
    n_iter       : maximum iterations.
    tol          : relative stress-change convergence threshold.
    weight_arr   : (N, N) non-negative weights (uniform 1 if None).
    filter_counts: set w_ij = 0 where weight_arr[i,j] <= filter_counts.
    verbose_step : print stress every this many iterations (0 = silent).

    Returns
    -------
    X : (N, 2) optimised coordinates.
    """
    n = truth_arr.shape[0]
    delta = truth_arr.astype(np.float64, copy=True)
    np.fill_diagonal(delta, 0.0)

    # --- Weight matrix ---
    if weight_arr is not None:
        W = np.where(weight_arr > filter_counts,
                     weight_arr.astype(np.float64), 0.0)
        np.fill_diagonal(W, 0.0)
        uniform = False
    else:
        uniform = True

    # For non-uniform weights pre-factorise V (weighted Laplacian).
    # V_ii = Σ_j w_ij,  V_ij = −w_ij  (i≠j).
    # Adding (1/n)·J makes V invertible; the solution gives V⁺·rhs
    # when rhs is centred (which B·X always is).
    if not uniform:
        row_sums = W.sum(axis=1)
        V = np.diag(row_sums) - W
        V_reg = V + 1.0 / n
        V_cho = cho_factor(V_reg)

    # --- Initialise ---
    if init is None:
        X = classical_mds_init(truth_arr)
    else:
        X = init.astype(np.float64, copy=True)
    X -= X.mean(axis=0)

    old_stress = np.inf

    for it in range(n_iter):
        # Pairwise distances  (exploit symmetry: compute upper triangle)
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]   # N×N×2
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))          # N×N

        # Stress (upper triangle × 2 = full symmetric sum, divide by 2)
        if uniform:
            stress = 0.5 * np.sum((delta - dist) ** 2)
        else:
            stress = 0.5 * np.sum(W * (delta - dist) ** 2)

        if verbose_step > 0 and it % verbose_step == 0:
            print(f"SMACOF iter {it}: stress = {stress:.6g}")

        rel_change = abs(old_stress - stress) / (old_stress + 1e-15)
        if it > 0 and rel_change < tol:
            if verbose_step > 0:
                print(f"SMACOF converged at iter {it}: stress = {stress:.6g}")
            break
        old_stress = stress

        # --- Guttman transform ---
        # ratio_ij = w_ij · delta_ij / d_ij  (0 when d_ij ≈ 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            if uniform:
                ratio = np.where(dist > 1e-10, delta / dist, 0.0)
            else:
                ratio = np.where(dist > 1e-10, W * delta / dist, 0.0)
        np.fill_diagonal(ratio, 0.0)

        # B_ij = −ratio_ij (i≠j),  B_ii = Σ_{j≠i} ratio_ij
        # Equivalently: (B·X)_i = Σ_j ratio_ij · (X_i − X_j)
        #   = diag(ratio.sum(1))·X − ratio·X
        b_diag = ratio.sum(axis=1)
        BX = b_diag[:, None] * X - ratio @ X

        if uniform:
            # X_new = (1/n)·B·X  (centering preserved automatically)
            X = BX / n
        else:
            # X_new = V⁺·B·X  via Cholesky solve
            X = cho_solve(V_cho, BX)
            X -= X.mean(axis=0)

    return X


# ============================================================
# Distance-error helpers (Powell path — kept for compatibility)
# ============================================================

def get_distance_error(positions, idx, truth_arr, b_arr, weights=None):
    """
    Vectorized version of get_distance_error that works with NumPy arrays.
    Compatible with shared-memory parallel execution.

    Parameters
    ----------
    positions : tuple(float, float)
        Candidate (x, y) coordinates being optimized.
    idx : int
        Index of the current element being optimized.
    truth_arr : np.ndarray
        Array of true distances or values (1D or 2D depending on context).
    b_arr : np.ndarray
        Array of baseline coordinates (N x 2). Updated in place.
    weights : np.ndarray, optional
        Optional weighting array matching `truth_arr` shape.

    Returns
    -------
    float
        Mean or weighted mean squared distance error.
    """
    tempx, tempy = positions

    # Update this point's coordinates (local, no side effects in parent)
    b_arr[idx, 0] = tempx
    b_arr[idx, 1] = tempy

    # Compute pairwise Euclidean distances from (tempx, tempy) to all o_arr points
    dists = np.sqrt((b_arr[:, 0] - tempx) ** 2 + (b_arr[:, 1] - tempy) ** 2)

    # Compute squared difference from truth
    diff = (truth_arr - dists) ** 2

    # Weighted or unweighted mean squared error
    if weights is not None:
        valid = weights > 0
        if np.any(valid):
            error = np.sum(np.multiply(diff[valid], weights[valid])) / np.sum(weights[valid])
        else:
            error = np.mean(diff)
    else:
        error = np.mean(diff)

    return error


# ---------- Shared memory utilities ----------

def _create_shared_array(np_array, name_prefix):
    """Create shared memory for a NumPy array and return (SharedMemory, shape, dtype)."""
    shm = shared_memory.SharedMemory(create=True, size=np_array.nbytes,
                                     name=f"{name_prefix}_{uuid.uuid4().hex[:8]}")
    shm_array = np.ndarray(np_array.shape, dtype=np_array.dtype, buffer=shm.buf)
    shm_array[:] = np_array
    return shm, np_array.shape, np_array.dtype


# ---------- Worker ----------

def _optimize_row(x,
                  truth_name, truth_shape, truth_dtype,
                  b_name, b_shape, b_dtype,
                  weight_name, weight_shape, weight_dtype,
                  filter_counts):
    """Run one minimize() optimization for a row/column."""
    truth_shm = shared_memory.SharedMemory(name=truth_name)
    truth = np.ndarray(truth_shape, dtype=truth_dtype, buffer=truth_shm.buf)

    b_shm = shared_memory.SharedMemory(name=b_name)
    b_arr = np.ndarray(b_shape, dtype=b_dtype, buffer=b_shm.buf)

    if weight_name:
        weight_shm = shared_memory.SharedMemory(name=weight_name)
        weight = np.ndarray(weight_shape, dtype=weight_dtype, buffer=weight_shm.buf)
    else:
        weight_shm = None
        weight = None

    try:
        small_truth_dist = truth[x, :]
        first_x0 = (b_arr[x, 0], b_arr[x, 1])
        if weight is not None:
            small_counts = np.multiply(weight[x, :], (weight[x, :] > filter_counts))
            res = minimize(get_distance_error, x0=first_x0,
                           args=(x, small_truth_dist, b_arr, small_counts),
                           method='Powell')
        else:
            res = minimize(get_distance_error, x0=first_x0,
                           args=(x, small_truth_dist, b_arr),
                           method='Powell')
    finally:
        truth_shm.close()
        b_shm.close()
        if weight_shm:
            weight_shm.close()

    return x, res.x[0], res.x[1]


# ---------- Main function (Powell path) ----------

def return_corrected_array(error, truth_arr, b_arr, correct_order='random',
                           weight_arr=None, filter_counts=0, n_jobs=None,
                           verbose_step=100, update_interval=None,
                           as_dataframe=False, column_names=None):
    """
    Per-point Powell optimisation (legacy / fallback path).

    Parameters
    ----------
    error : np.ndarray        Per-point error (length N); determines optimisation order.
    truth_arr, b_arr          2-D arrays (N×N and N×2).
    weight_arr                Optional (N×N) weight matrix.
    filter_counts             Exclude pairs with weight <= filter_counts.
    n_jobs                    Parallel workers (None = all CPUs).
    correct_order             'random' | 'sorted' (descending error).
    as_dataframe              Return pd.DataFrame instead of ndarray.
    column_names              Column names when as_dataframe=True.
    """
    truth_shm, truth_shape, truth_dtype = _create_shared_array(truth_arr, "truth")
    b_shm, b_shape, b_dtype = _create_shared_array(b_arr, "bdf")

    if weight_arr is not None:
        weight_shm, weight_shape, weight_dtype = _create_shared_array(weight_arr, "weight")
    else:
        weight_shm, weight_shape, weight_dtype = None, None, None

    if correct_order == 'random':
        sorted_indices = np.random.permutation(len(error))
    else:
        sorted_indices = np.argsort(error)[::-1]

    corrected_arr = b_arr.copy()
    b_arr_shared = np.ndarray(b_shape, dtype=b_dtype, buffer=b_shm.buf)

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(
                _optimize_row,
                x,
                truth_shm.name, truth_shape, truth_dtype,
                b_shm.name, b_shape, b_dtype,
                weight_shm.name if weight_shm else None, weight_shape, weight_dtype,
                filter_counts
            )
            for x in sorted_indices
        ]

        for i, future in enumerate(as_completed(futures), 1):
            x, xcoord, ycoord = future.result()
            corrected_arr[x, 0] = xcoord
            corrected_arr[x, 1] = ycoord

            if update_interval and i % update_interval == 0:
                b_arr_shared[:] = corrected_arr

            if i % verbose_step == 0:
                print(f"{i} completed")

    b_arr_shared[:] = corrected_arr

    for shm in [truth_shm, b_shm, weight_shm]:
        if shm:
            shm.close()
            shm.unlink()

    if as_dataframe:
        return pd.DataFrame(corrected_arr, columns=column_names)

    return corrected_arr


# ============================================================
# get_error  (OLS-optimal scale)
# ============================================================

def get_error(b_arr, truth_arr, filter_counts=None, counts=None):
    """
    Per-point MSE between scaled pairwise distances and truth distances.

    Uses the analytically optimal (OLS) scale factor
        α* = Σ(truth · dist) / Σ(dist²)
    which minimises the residual Σ(truth − α·dist)² over all pairs.
    """
    if _RUST_AVAILABLE:
        return _rs.get_error(
            np.asarray(b_arr, dtype=np.float64),
            np.asarray(truth_arr, dtype=np.float64),
            filter_counts,
            np.asarray(counts, dtype=np.float64) if counts is not None else None,
        )
    diff = b_arr[:, np.newaxis, :] - b_arr[np.newaxis, :, :]
    new_dist = np.sqrt(np.sum(diff ** 2, axis=-1))

    if filter_counts is not None:
        mask = counts > filter_counts
        new_dist  = new_dist  * mask
        truth_arr = truth_arr * mask

    # OLS-optimal scale (exclude diagonal, which is always 0/0)
    off = ~np.eye(len(b_arr), dtype=bool)
    denom = np.sum(new_dist[off] ** 2)
    scaler = np.sum(truth_arr[off] * new_dist[off]) / denom if denom > 0 else 1.0

    error = np.mean(np.square(truth_arr - new_dist * scaler), axis=0)
    return error


# ============================================================
# correct_locations  (SMACOF default, Powell fallback)
# ============================================================

def correct_locations(b_df, truth, weight=None,
                      filter_counts=0, n_jobs=None,
                      verbose_step=100, update_interval=None,
                      correct_order='random',
                      method='smacof', n_iter=300, tol=1e-4):
    """
    Fit 2-D coordinates to a target distance matrix.

    Parameters
    ----------
    b_df          : pd.DataFrame (N × 2)  current coordinate estimates.
    truth         : (N, N) target pairwise distances.
    weight        : (N, N) optional weights.
    filter_counts : exclude pairs with weight <= filter_counts.
    method        : 'smacof' (default) | 'powell'.
                    'smacof' uses the SMACOF algorithm (much faster for large N).
                    'powell' uses the legacy per-point Powell optimisation.
    n_iter        : max iterations (smacof only).
    tol           : convergence threshold (smacof only).
    n_jobs        : parallel workers (powell only).
    correct_order : 'random' | 'sorted' (powell only).
    """
    b_array    = np.asarray(b_df.values, dtype=np.float64)
    truth_array = np.asarray(truth, dtype=np.float64)

    if method == 'smacof':
        if _RUST_AVAILABLE:
            w = np.asarray(weight, dtype=np.float64) if weight is not None else None
            fit_b_array = _rs.smacof(
                truth_array, b_array, n_iter, tol, w,
                float(filter_counts), verbose_step,
            )
        else:
            fit_b_array = smacof(
                truth_array, init=b_array,
                n_iter=n_iter, tol=tol,
                weight_arr=weight, filter_counts=filter_counts,
                verbose_step=verbose_step,
            )
    else:
        # Legacy Powell path
        starting_error = get_error(b_array, truth_array)
        if _RUST_AVAILABLE:
            fit_b_array = _rs.return_corrected_array(
                np.asarray(starting_error, dtype=np.float64),
                truth_array, b_array, correct_order,
                np.asarray(weight, dtype=np.float64) if weight is not None else None,
                float(filter_counts), n_jobs, verbose_step, update_interval,
            )
        else:
            fit_b_array = return_corrected_array(
                starting_error, truth_array, b_array,
                correct_order=correct_order, update_interval=update_interval,
                weight_arr=weight, n_jobs=n_jobs, filter_counts=filter_counts,
                verbose_step=verbose_step,
            )

    return pd.DataFrame(fit_b_array, index=b_df.index, columns=b_df.columns)
