#Pure numpy
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory
from scipy.optimize import minimize
import uuid
import pandas as pd

try:
    import accumap_rs as _rs
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False




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
# correct_locations
# ============================================================

def correct_locations(b_df, truth, weight=None,
                      filter_counts=0, n_jobs=None,
                      verbose_step=100, update_interval=None,
                      correct_order='random', gauss_seidel=True):
    """
    Fit 2-D coordinates to a target distance matrix using per-point
    L-BFGS optimisation (Powell path).

    Parameters
    ----------
    b_df          : pd.DataFrame (N × 2)  current coordinate estimates.
    truth         : (N, N) target pairwise distances.
    weight        : (N, N) optional weights.
    filter_counts : exclude pairs with weight <= filter_counts.
    n_jobs        : parallel workers (ignored when Rust extension is available).
    correct_order : 'random' | 'sorted' — 'sorted' processes highest-error
                    points first, which propagates fixes to neighbours faster.
    gauss_seidel  : if True (default), each point's update is immediately
                    visible to subsequent points in the same pass, matching
                    the original update_interval=1 behaviour.  Set False for
                    fully parallel updates (faster, lower quality).
    update_interval: update interval for the Python fallback path.
    """
    b_array     = np.asarray(b_df.values, dtype=np.float64)
    truth_array = np.asarray(truth, dtype=np.float64)

    starting_error = get_error(b_array, truth_array)
    if _RUST_AVAILABLE:
        fit_b_array = _rs.return_corrected_array(
            np.asarray(starting_error, dtype=np.float64),
            truth_array, b_array, correct_order,
            np.asarray(weight, dtype=np.float64) if weight is not None else None,
            float(filter_counts), n_jobs, verbose_step, update_interval,
            gauss_seidel,
        )
    else:
        fit_b_array = return_corrected_array(
            starting_error, truth_array, b_array,
            correct_order=correct_order, update_interval=update_interval,
            weight_arr=weight, n_jobs=n_jobs, filter_counts=filter_counts,
            verbose_step=verbose_step,
        )

    return pd.DataFrame(fit_b_array, index=b_df.index, columns=b_df.columns)
