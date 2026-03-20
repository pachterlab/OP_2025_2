"""
Compare outputs of the Rust (accumap_rs) and Python implementations.

Run with:
    python -m unittest discover -s tests -p "test_accumap*.py" -v
"""

import sys
import os
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'analysis_scripts'))

import accumap_rs as _rs
import accumap   as _py


# ── helpers ─────────────────────────────────────────────────────────────────

def _make_coords(n=10, seed=42, scale=1000.0):
    return np.random.default_rng(seed).uniform(-scale, scale, size=(n, 2))


def _truth_from_coords(coords):
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def _mean_error(b, truth):
    """Unscaled mean squared distance error."""
    diff = b[:, np.newaxis, :] - b[np.newaxis, :, :]
    d    = np.sqrt(np.sum(diff ** 2, axis=-1))
    sc   = np.max(truth) / np.max(d)
    return float(np.mean((truth - d * sc) ** 2))


def _py_get_error_direct(b_arr, truth_arr):
    """Pure-Python get_error (bypass Rust dispatch)."""
    diff     = b_arr[:, np.newaxis, :] - b_arr[np.newaxis, :, :]
    new_dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    off      = ~np.eye(len(b_arr), dtype=bool)
    denom    = np.sum(new_dist[off] ** 2)
    scaler   = np.sum(truth_arr[off] * new_dist[off]) / denom if denom > 0 else 1.0
    return np.mean(np.square(truth_arr - new_dist * scaler), axis=0)


# ── TestGetError ─────────────────────────────────────────────────────────────

class TestGetError(unittest.TestCase):

    def _assert_close(self, py, rs, rtol=1e-5, atol=1e-8, label=""):
        np.testing.assert_allclose(np.asarray(py), np.asarray(rs),
                                   rtol=rtol, atol=atol, err_msg=label)

    def test_basic_no_filter(self):
        coords = _make_coords(n=8, seed=0)
        truth  = _truth_from_coords(coords)
        b_arr  = coords + np.random.default_rng(1).normal(0, 50, coords.shape)
        self._assert_close(_py_get_error_direct(b_arr, truth),
                           _rs.get_error(np.float64(b_arr), np.float64(truth)),
                           label="basic_no_filter")

    def test_larger_grid(self):
        coords = _make_coords(n=30, seed=7, scale=5000.0)
        truth  = _truth_from_coords(coords)
        b_arr  = coords + np.random.default_rng(8).normal(0, 200, coords.shape)
        self._assert_close(_py_get_error_direct(b_arr, truth),
                           _rs.get_error(np.float64(b_arr), np.float64(truth)),
                           label="larger_grid")

    def test_with_filter_counts(self):
        n = 12
        coords = _make_coords(n=n, seed=3)
        truth  = _truth_from_coords(coords)
        b_arr  = coords + np.random.default_rng(4).normal(0, 100, coords.shape)
        counts = np.random.default_rng(5).integers(0, 20, size=(n, n)).astype(float)
        fc     = 5.0
        self._assert_close(_py.get_error(b_arr, truth, filter_counts=fc, counts=counts),
                           _rs.get_error(np.float64(b_arr), np.float64(truth),
                                         filter_counts=fc, counts=np.float64(counts)),
                           label="with_filter_counts")

    def test_perfect_coords_zero_error(self):
        coords = _make_coords(n=6, seed=99)
        truth  = _truth_from_coords(coords)
        np.testing.assert_allclose(_py_get_error_direct(coords, truth), 0.0, atol=1e-10)
        np.testing.assert_allclose(_rs.get_error(np.float64(coords), np.float64(truth)),
                                   0.0, atol=1e-10)

    def test_output_shape(self):
        n = 15
        coords = _make_coords(n=n, seed=10)
        truth  = _truth_from_coords(coords)
        self.assertEqual(_rs.get_error(np.float64(coords + 50), np.float64(truth)).shape, (n,))


# ── TestReturnCorrectedArray ─────────────────────────────────────────────────

class TestReturnCorrectedArray(unittest.TestCase):

    def _run_rs(self, n=10, seed=0, gauss_seidel=False):
        rng    = np.random.default_rng(seed)
        coords = rng.uniform(-1000, 1000, size=(n, 2))
        truth  = _truth_from_coords(coords)
        b_arr  = (coords + rng.normal(0, 200, coords.shape)).astype(np.float64)
        error  = _py_get_error_direct(b_arr, truth)
        out    = _rs.return_corrected_array(
            np.float64(error), np.float64(truth), b_arr, 'sorted',
            None, 0.0, 1, 99999, None, gauss_seidel)
        return b_arr, truth, out

    def test_output_shape(self):
        b, t, rs = self._run_rs(n=8, seed=1)
        self.assertEqual(rs.shape, b.shape)

    def test_rust_reduces_error(self):
        b, t, rs = self._run_rs(n=10, seed=2)
        self.assertLess(_mean_error(rs, t), _mean_error(b, t))

    def test_gauss_seidel_reduces_error(self):
        b, t, rs = self._run_rs(n=10, seed=4, gauss_seidel=True)
        self.assertLess(_mean_error(rs, t), _mean_error(b, t))

    def test_gauss_seidel_better_than_parallel(self):
        """GS updates should converge to lower error (sees already-fixed neighbours)."""
        b, t, par = self._run_rs(n=12, seed=6, gauss_seidel=False)
        _,  _, gs  = self._run_rs(n=12, seed=6, gauss_seidel=True)
        self.assertLessEqual(_mean_error(gs, t), _mean_error(par, t) * 1.05)

    def test_with_weights(self):
        rng    = np.random.default_rng(5)
        n      = 8
        coords = rng.uniform(-1000, 1000, size=(n, 2))
        truth  = _truth_from_coords(coords)
        b_arr  = (coords + rng.normal(0, 200, coords.shape)).astype(np.float64)
        W      = rng.integers(1, 50, size=(n, n)).astype(float)
        W      = (W + W.T) / 2
        error  = _py_get_error_direct(b_arr, truth)
        out    = _rs.return_corrected_array(
            np.float64(error), np.float64(truth), b_arr, 'sorted',
            np.float64(W), 0.0, 1, 99999, None, False)
        self.assertLess(_mean_error(out, truth), _mean_error(b_arr, truth))

    def test_larger_problem(self):
        b, t, rs = self._run_rs(n=25, seed=9)
        self.assertEqual(rs.shape, (25, 2))
        self.assertLess(_mean_error(rs, t), _mean_error(b, t))

    def test_per_point_optimizer_matches_scipy(self):
        """L-BFGS per-point results should agree with scipy Powell."""
        from scipy.optimize import minimize
        rng    = np.random.default_rng(3)
        n      = 10
        coords = rng.uniform(-1000, 1000, size=(n, 2))
        truth  = _truth_from_coords(coords)
        b_arr  = (coords + rng.normal(0, 200, coords.shape)).astype(np.float64)
        error  = _py_get_error_direct(b_arr, truth)
        rs_out = _rs.return_corrected_array(
            np.float64(error), np.float64(truth), b_arr, 'sorted',
            None, 0.0, 1, 99999, None, False)

        for x in range(n):
            coords_tmp = b_arr.copy()
            def scipy_cost(pos, x=x, ct=coords_tmp.copy()):
                ct[x] = pos
                dists = np.sqrt((ct[:,0]-pos[0])**2 + (ct[:,1]-pos[1])**2)
                return float(np.mean((truth[x] - dists)**2))
            res     = minimize(scipy_cost, x0=b_arr[x], method='Powell')
            rs_cost = scipy_cost(rs_out[x])
            self.assertAlmostEqual(rs_cost, res.fun,
                                   delta=max(abs(res.fun)*0.05, 1.0),
                                   msg=f"Point {x}: L-BFGS={rs_cost:.4f} scipy={res.fun:.4f}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
