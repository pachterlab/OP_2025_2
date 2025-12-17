#Pure numpy
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory
from scipy.optimize import minimize
import uuid
from scipy.spatial import procrustes, distance_matrix
import pandas as pd

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

    # Update this point’s coordinates (local, no side effects in parent)
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
    shm = shared_memory.SharedMemory(create=True, size=np_array.nbytes, name=f"/{name_prefix}_{uuid.uuid4().hex}")
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
        weight = None
        
    small_truth_dist = truth[x, :]
    first_x0 = (b_arr[x, 0], b_arr[x, 1])
    #np.random.uniform(-1,1,size=2)*1500
        #(0, 0) #
    if weight is not None:
        small_counts = np.multiply(weight[x, :] ,(weight[x, :] > filter_counts))
        res = minimize(get_distance_error, x0=first_x0,
                       args=(x, small_truth_dist, b_arr, small_counts),
                       method='Powell')
    else:

        res = minimize(get_distance_error, x0=first_x0,
                       args=(x, small_truth_dist, b_arr),
                       method='Powell')
    return x, res.x[0], res.x[1]


# ---------- Main function ----------

def return_corrected_array(error, truth_arr, b_arr, correct_order='random',
                           weight_arr=None, filter_counts=0, n_jobs=None,
                                             verbose_step = 100, update_interval=None,
                           as_dataframe=False, column_names=None):
    """
    Fully NumPy + multiprocessing version of return_corrected_df.

    Parameters
    ----------
    error : np.ndarray
        Array of error values (length N).
    truth_arr, b_arr, o_arr : np.ndarray
        2D arrays containing data.
    weight_arr : np.ndarray, optional
        Optional weight matrix.
    b2o : bool
        If True, adjusts b_arr toward o_arr; else vice versa.
    filter_counts : float
        Filter threshold for weights.
    n_jobs : int, optional
        Number of parallel processes (default: all cores).
    as_dataframe : bool
        If True, return a pandas DataFrame (else a NumPy array).
    column_names : list[str], optional
        Column names for the returned DataFrame.

    Returns
    -------
    np.ndarray or pd.DataFrame
        Corrected coordinate array or DataFrame.
    """
    # --- Shared memory setup ---
    truth_shm, truth_shape, truth_dtype = _create_shared_array(truth_arr, "truth")
    b_shm, b_shape, b_dtype = _create_shared_array(b_arr, "bdf")

    if weight_arr is not None:
        weight_shm, weight_shape, weight_dtype = _create_shared_array(weight_arr, "weight")
    else:
        weight_shm, weight_shape, weight_dtype = None, None, None
    if correct_order == 'random':
        sorted_indices = np.arange(len(error)) 
    else:
        sorted_indices = np.argsort(error)[::-1]
        
    corrected_arr = b_arr.copy() 
    
    # Get shared memory view of b_arr so we can write to it mid-process
    b_arr_shared = np.ndarray(b_shape, dtype=b_dtype, buffer=b_shm.buf)
    
    # --- Parallel processing ---
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
                b_arr_shared[:] = corrected_arr  # overwrite shared array
                #if verbose_step:
                #    print(f"Shared b_arr updated at iteration {i}")
                    
            if i % verbose_step == 0:
                print(f"{i} completed")
    # Final sync so shared memory has the latest version
    b_arr_shared[:] = corrected_arr
    
    # --- Cleanup shared memory ---
    for shm in [truth_shm, b_shm, weight_shm]:
        if shm:
            shm.close()
            shm.unlink()

    if as_dataframe:
        import pandas as pd
        return pd.DataFrame(corrected_arr, columns=column_names)

    return corrected_arr

def get_error(b_arr, truth_arr, filter_counts=None,counts=None):
    new_dist = distance_matrix(b_arr, b_arr)
    scaler = np.max(truth_arr)/np.max(new_dist)
    if filter_counts is None:
        new_dist = new_dist 
    else:
        new_dist = np.multiply(new_dist, np.array(counts > filter_counts, dtype=int))
        truth_arr = np.multiply(truth_arr, np.array(counts > filter_counts, dtype=int))
    error = np.mean(np.square(np.subtract(truth_arr, new_dist*scaler)), axis=0)
    return error

def correct_locations(b_df, truth, weight=None,
                      filter_counts=0, n_jobs=None,
                     verbose_step = 100,update_interval=None,
                     correct_order = 'random'):
    # convert all arguments to arrays
    b_array = b_df.values
    truth_array = truth
    
    starting_error = get_error(b_array, truth_array)
    fit_b_array = return_corrected_array(starting_error, truth_array, b_array, correct_order = correct_order, update_interval=update_interval,
                                             weight_arr=weight, n_jobs=n_jobs, filter_counts=filter_counts,
                                             verbose_step = verbose_step)

    fit_b_df = pd.DataFrame(fit_b_array, index = b_df.index, columns = b_df.columns)
    return fit_b_df
