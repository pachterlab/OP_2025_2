use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// ============================================================
// 1-D optimisation: bracketing + Brent's method  (Powell path)
// ============================================================

#[allow(dead_code)]
/// Expand bracket from [xa, xb] until a minimum is enclosed.
/// Returns (ax, bx, cx): ax < cx with bx the current best.
fn find_bracket<F: Fn(f64) -> f64>(f: &F, xa: f64, xb: f64) -> (f64, f64, f64) {
    const GOLD: f64 = 1.618_033_988_749_895;
    let mut a = xa;
    let mut b = xb;
    let mut fa = f(a);
    let mut fb = f(b);
    if fb > fa {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }
    let mut c = b + GOLD * (b - a);
    let mut fc = f(c);
    for _ in 0..60 {
        if fb <= fc { break; }
        let tmp = c + GOLD * (c - b);
        a = b; b = c; fb = fc; c = tmp; fc = f(c);
    }
    if a < c { (a, b, c) } else { (c, b, a) }
}

#[allow(dead_code)]
/// Brent's method: 1-D minimisation seeded from the known best point bx ∈ (ax, cx).
fn brent<F: Fn(f64) -> f64>(f: F, ax: f64, bx: f64, cx: f64, tol: f64) -> f64 {
    const CGOLD: f64 = 0.381_966_011_250_105;
    const ZEPS: f64 = 1e-10;
    let (mut lo, mut hi) = if ax < cx { (ax, cx) } else { (cx, ax) };
    let mut x = bx;
    let (mut v, mut w) = (bx, bx);
    let mut fx = f(x);
    let (mut fv, mut fw) = (fx, fx);
    let mut d = 0.0_f64;
    let mut e = 0.0_f64;

    for _ in 0..500 {
        let midpt = 0.5 * (lo + hi);
        let tol1 = tol * x.abs() + ZEPS;
        let tol2 = 2.0 * tol1;
        if (x - midpt).abs() <= tol2 - 0.5 * (hi - lo) { return x; }
        if e.abs() > tol1 {
            let r = (x - w) * (fx - fv);
            let mut q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if q > 0.0 { p = -p; }
            q = q.abs();
            let etemp = e; e = d;
            if p.abs() < (0.5 * q * etemp).abs() && p > q * (lo - x) && p < q * (hi - x) {
                d = p / q;
                let u = x + d;
                if (u - lo) < tol2 || (hi - u) < tol2 {
                    d = if midpt >= x { tol1 } else { -tol1 };
                }
            } else {
                e = if x < midpt { hi - x } else { lo - x };
                d = CGOLD * e;
            }
        } else {
            e = if x < midpt { hi - x } else { lo - x };
            d = CGOLD * e;
        }
        let u = x + if d.abs() >= tol1 { d } else if d > 0.0 { tol1 } else { -tol1 };
        let fu = f(u);
        if fu <= fx {
            if u < x { hi = x; } else { lo = x; }
            v = w; fv = fw; w = x; fw = fx; x = u; fx = fu;
        } else {
            if u < x { lo = u; } else { hi = u; }
            if fu <= fw || (w - x).abs() < f64::EPSILON {
                v = w; fv = fw; w = u; fw = fu;
            } else if fu <= fv || (v - x).abs() < f64::EPSILON || (v - w).abs() < f64::EPSILON {
                v = u; fv = fu;
            }
        }
    }
    x
}

#[allow(dead_code)]
fn line_min<F: Fn([f64; 2]) -> f64>(f: &F, pt: [f64; 2], dir: [f64; 2], step: f64) -> [f64; 2] {
    let g = |a: f64| f([pt[0] + a * dir[0], pt[1] + a * dir[1]]);
    let (ax, bx, cx) = find_bracket(&g, 0.0, step);
    let alpha = brent(g, ax, bx, cx, 1e-5);
    [pt[0] + alpha * dir[0], pt[1] + alpha * dir[1]]
}

// ============================================================
// Powell's conjugate-direction method (2-D)  [legacy path]
// ============================================================

#[allow(dead_code)]
fn powell_2d<F: Fn([f64; 2]) -> f64>(f: F, x0: [f64; 2], step: f64, xtol: f64, max_iter: usize) -> [f64; 2] {
    let mut x = x0;
    let mut dirs = [[1.0_f64, 0.0], [0.0, 1.0_f64]];
    for _ in 0..max_iter {
        let x_start = x;
        let f_start = f(x);
        let mut fx = f_start;
        let mut max_decrease = 0.0_f64;
        let mut max_dir = 0usize;
        for i in 0..2 {
            let new_x = line_min(&f, x, dirs[i], step);
            let new_fx = f(new_x);
            let decrease = fx - new_fx;
            if decrease > max_decrease { max_decrease = decrease; max_dir = i; }
            x = new_x; fx = new_fx;
        }
        let dn_unnorm = [x[0] - x_start[0], x[1] - x_start[1]];
        let norm = (dn_unnorm[0].powi(2) + dn_unnorm[1].powi(2)).sqrt();
        if norm > 1e-12 {
            let dn = [dn_unnorm[0] / norm, dn_unnorm[1] / norm];
            x = line_min(&f, x, dn, step);
            dirs[max_dir] = dn;
        }
        let delta = ((x[0] - x_start[0]).powi(2) + (x[1] - x_start[1]).powi(2)).sqrt();
        if delta < xtol { break; }
    }
    x
}

// ============================================================
// L-BFGS with backtracking line search (2-D)
// ============================================================

/// Gradient of the per-point distance-error cost w.r.t. (x, y).
/// Returns (cost, [df/dx, df/dy]).
fn distance_error_with_grad(
    pos: [f64; 2],
    idx: usize,
    truth_row: &[f64],
    coords: &[[f64; 2]],
    weights: Option<&[f64]>,
) -> (f64, [f64; 2]) {
    let (tx, ty) = (pos[0], pos[1]);
    let n = coords.len();
    let mut cost = 0.0_f64;
    let mut gx   = 0.0_f64;
    let mut gy   = 0.0_f64;
    let mut wsum = 0.0_f64;

    for j in 0..n {
        if j == idx { continue; }
        let dx = tx - coords[j][0];
        let dy = ty - coords[j][1];
        let d2 = dx * dx + dy * dy;
        let d  = d2.sqrt();
        let w  = weights.map_or(1.0, |w| w[j]);
        if w <= 0.0 { continue; }
        let residual = truth_row[j] - d;
        cost  += w * residual * residual;
        wsum  += w;
        if d > 1e-12 {
            // d(residual²)/d(pos) = -2·residual · (pos−coords_j) / d
            let factor = -2.0 * w * residual / d;
            gx += factor * dx;
            gy += factor * dy;
        }
    }
    if wsum > 0.0 { cost /= wsum; gx /= wsum; gy /= wsum; }
    (cost, [gx, gy])
}

/// Dot product of two 2-vectors.
#[inline]
fn dot2(a: [f64; 2], b: [f64; 2]) -> f64 { a[0]*b[0] + a[1]*b[1] }

/// L-BFGS with m=6 memory and Armijo backtracking.
fn lbfgs_2d<FG: Fn([f64; 2]) -> (f64, [f64; 2])>(
    fg: FG,
    x0: [f64; 2],
    gtol: f64,
    max_iter: usize,
) -> [f64; 2] {
    const M: usize = 6;
    let mut x = x0;
    let (mut f_val, mut g) = fg(x);

    // Circular buffers for L-BFGS curvature pairs
    let mut s_buf = [[0.0_f64; 2]; M];   // x_{k+1} − x_k
    let mut y_buf = [[0.0_f64; 2]; M];   // g_{k+1} − g_k
    let mut rho   = [0.0_f64; M];         // 1 / (s·y)
    let mut head  = 0usize;
    let mut filled = 0usize;

    for _ in 0..max_iter {
        // Convergence: gradient norm
        if dot2(g, g).sqrt() < gtol { break; }

        // Two-loop L-BFGS recursion → search direction
        let mut q = g;
        let mut alpha = [0.0_f64; M];
        let count = filled.min(M);
        // Backward pass: iterate most-recent → oldest
        for k in 0..count {
            let idx = (head + M - 1 - k) % M;
            alpha[k] = rho[idx] * dot2(s_buf[idx], q);
            q[0] -= alpha[k] * y_buf[idx][0];
            q[1] -= alpha[k] * y_buf[idx][1];
        }
        // Scale initial Hessian
        if filled > 0 {
            let last = (head + M - 1) % M;
            let sy = dot2(s_buf[last], y_buf[last]);
            let yy = dot2(y_buf[last], y_buf[last]);
            if yy > 0.0 { let h0 = sy / yy; q[0] *= h0; q[1] *= h0; }
        }
        // Forward pass
        for k in 0..count {
            let idx = (head + M - 1 - count + 1 + k) % M;
            let beta = rho[idx] * dot2(y_buf[idx], q);
            q[0] += (alpha[count - 1 - k] - beta) * s_buf[idx][0];
            q[1] += (alpha[count - 1 - k] - beta) * s_buf[idx][1];
        }
        let dir = [-q[0], -q[1]];   // descent direction

        // Armijo backtracking line search
        let slope = dot2(g, dir);
        let mut step = 1.0_f64;
        let c1 = 1e-4;
        let x_new;
        let f_new;
        let g_new;
        loop {
            let xt = [x[0] + step * dir[0], x[1] + step * dir[1]];
            let (ft, gt) = fg(xt);
            if ft <= f_val + c1 * step * slope || step < 1e-12 {
                x_new = xt; f_new = ft; g_new = gt;
                break;
            }
            step *= 0.5;
        }

        // Update curvature pair
        let s = [x_new[0] - x[0], x_new[1] - x[1]];
        let y = [g_new[0] - g[0], g_new[1] - g[1]];
        let sy = dot2(s, y);
        if sy > 1e-20 {
            s_buf[head] = s;
            y_buf[head] = y;
            rho[head]   = 1.0 / sy;
            head = (head + 1) % M;
            filled = (filled + 1).min(M);
        }

        x = x_new; f_val = f_new; g = g_new;
    }
    x
}

// ============================================================
// Cost function (no heap allocation — accumulate directly)
// ============================================================

#[allow(dead_code)]
fn compute_distance_error(
    pos: [f64; 2],
    idx: usize,
    truth_row: &[f64],
    coords: &[[f64; 2]],
    weights: Option<&[f64]>,
) -> f64 {
    let (tx, ty) = (pos[0], pos[1]);
    let n = coords.len();
    let mut num  = 0.0_f64;
    let mut wsum = 0.0_f64;

    for j in 0..n {
        let (cx, cy) = if j == idx { (tx, ty) } else { (coords[j][0], coords[j][1]) };
        let dist = ((cx - tx).powi(2) + (cy - ty).powi(2)).sqrt();
        let sq   = (truth_row[j] - dist).powi(2);
        match weights {
            Some(w) if w[j] > 0.0 => { num += w[j] * sq; wsum += w[j]; }
            None                   => { num += sq;          wsum += 1.0; }
            _                      => {}
        }
    }
    if wsum > 0.0 { num / wsum } else { 0.0 }
}

// ============================================================
// Python-exposed: get_error  (OLS-optimal scale + symmetry)
// ============================================================

#[pyfunction]
#[pyo3(signature = (b_arr, truth_arr, filter_counts=None, counts=None))]
fn get_error<'py>(
    py: Python<'py>,
    b_arr: PyReadonlyArray2<f64>,
    truth_arr: PyReadonlyArray2<f64>,
    filter_counts: Option<f64>,
    counts: Option<PyReadonlyArray2<f64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let b     = b_arr.as_array().to_owned();
    let mut truth = truth_arr.as_array().to_owned();
    let counts_owned = counts.map(|c| c.as_array().to_owned());
    let n = b.nrows();

    let error: Array1<f64> = py.allow_threads(|| {
        // Flatten coordinates for unit-stride inner-loop access
        let bx: Vec<f64> = (0..n).map(|i| b[[i, 0]]).collect();
        let by: Vec<f64> = (0..n).map(|i| b[[i, 1]]).collect();

        // Compute distance matrix rows in parallel
        let mut dist_rows: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map(|i| {
                (0..n).map(|j| {
                    let dx = bx[i] - bx[j];
                    let dy = by[i] - by[j];
                    (dx * dx + dy * dy).sqrt()
                }).collect()
            })
            .collect();

        // Apply count filter (sequential; cheap compared to distance computation)
        if let (Some(fc), Some(cnt)) = (filter_counts, &counts_owned) {
            for i in 0..n {
                for j in 0..n {
                    if cnt[[i, j]] <= fc {
                        dist_rows[i][j] = 0.0;
                        truth[[i, j]]   = 0.0;
                    }
                }
            }
        }

        // OLS-optimal scale: α* = Σ(truth·dist) / Σ(dist²)  (off-diagonal only)
        let (num, den): (f64, f64) = (0..n)
            .into_par_iter()
            .map(|i| {
                let row = &dist_rows[i];
                (0..n)
                    .filter(|&j| i != j)
                    .fold((0.0_f64, 0.0_f64), |(acc_n, acc_d), j| {
                        let d = row[j];
                        (acc_n + truth[[i, j]] * d, acc_d + d * d)
                    })
            })
            .reduce(|| (0.0, 0.0), |(n1, d1), (n2, d2)| (n1 + n2, d1 + d2));
        let scaler = if den > 0.0 { num / den } else { 1.0 };

        // Parallel per-column MSE
        (0..n)
            .into_par_iter()
            .map(|j| {
                (0..n)
                    .map(|i| (truth[[i, j]] - dist_rows[i][j] * scaler).powi(2))
                    .sum::<f64>()
                    / n as f64
            })
            .collect::<Vec<_>>()
            .into()
    });

    Ok(error.into_pyarray_bound(py))
}

// ============================================================
// Python-exposed: return_corrected_array  (L-BFGS + Gauss-Seidel)
// ============================================================

#[pyfunction]
#[pyo3(signature = (
    error, truth_arr, b_arr,
    correct_order = "random",
    weight_arr = None,
    filter_counts = 0.0,
    n_jobs = None,
    verbose_step = 100,
    _update_interval = None,
    gauss_seidel = false,
))]
fn return_corrected_array<'py>(
    py: Python<'py>,
    error:         PyReadonlyArray1<f64>,
    truth_arr:     PyReadonlyArray2<f64>,
    b_arr:         PyReadonlyArray2<f64>,
    correct_order: &str,
    weight_arr:    Option<PyReadonlyArray2<f64>>,
    filter_counts: f64,
    n_jobs:        Option<usize>,
    verbose_step:  usize,
    _update_interval: Option<usize>,
    gauss_seidel:  bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let error_arr = error.as_array().to_owned();
    let truth     = truth_arr.as_array().to_owned();
    let b         = b_arr.as_array().to_owned();
    let weight    = weight_arr.map(|w| w.as_array().to_owned());
    let n = b.nrows();

    let sorted_indices: Vec<usize> = if correct_order == "random" {
        use rand::seq::SliceRandom;
        let mut idx: Vec<usize> = (0..n).collect();
        idx.shuffle(&mut rand::thread_rng());
        idx
    } else {
        let mut iv: Vec<(usize, f64)> = error_arr.iter().cloned().enumerate().collect();
        iv.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        iv.iter().map(|(i, _)| *i).collect()
    };

    let coords: Vec<[f64; 2]> = (0..n).map(|i| [b[[i, 0]], b[[i, 1]]]).collect();
    let max_truth = truth.iter().cloned().fold(0.0_f64, f64::max);
    let _step = (max_truth / 10.0).max(1.0);

    let truth_rows: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| truth[[i, j]]).collect())
        .collect();

    let weight_rows: Option<Vec<Vec<f64>>> = weight.map(|w| {
        (0..n).map(|i| {
            (0..n).map(|j| {
                let wi = w[[i, j]];
                if wi > filter_counts { wi } else { 0.0 }
            }).collect()
        }).collect()
    });

    let results: Vec<(usize, f64, f64)> = py.allow_threads(|| {
        if gauss_seidel {
            // Sequential Gauss-Seidel: each point sees already-updated neighbours
            let mut coords_gs = coords.clone();
            let mut out = Vec::with_capacity(n);
            for (i, &x) in sorted_indices.iter().enumerate() {
                let truth_row = &truth_rows[x];
                let weights   = weight_rows.as_ref().map(|wr| wr[x].as_slice());
                let fg = |pos: [f64; 2]| {
                    distance_error_with_grad(pos, x, truth_row, &coords_gs, weights)
                };
                let result = lbfgs_2d(fg, coords_gs[x], 1e-6, 200);
                coords_gs[x] = result;
                if verbose_step > 0 && (i + 1) % verbose_step == 0 {
                    eprintln!("{} completed (GS)", i + 1);
                }
                out.push((x, result[0], result[1]));
            }
            out
        } else {
            // Fully parallel: all points optimised against initial snapshot
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_jobs.unwrap_or(0))
                .build()
                .expect("failed to build thread pool");

            let counter = Arc::new(AtomicUsize::new(0));

            pool.install(|| {
                sorted_indices.par_iter().map(|&x| {
                    let truth_row = &truth_rows[x];
                    let weights   = weight_rows.as_ref().map(|wr| wr[x].as_slice());
                    let fg = |pos: [f64; 2]| {
                        distance_error_with_grad(pos, x, truth_row, &coords, weights)
                    };
                    let result = lbfgs_2d(fg, coords[x], 1e-6, 200);

                    if verbose_step > 0 {
                        let i = counter.fetch_add(1, Ordering::Relaxed) + 1;
                        if i % verbose_step == 0 { eprintln!("{i} completed"); }
                    }
                    (x, result[0], result[1])
                }).collect()
            })
        }
    });

    let mut corrected = b;
    for (x, xc, yc) in results { corrected[[x, 0]] = xc; corrected[[x, 1]] = yc; }
    Ok(corrected.into_pyarray_bound(py))
}

// ============================================================
// Module
// ============================================================

#[pymodule]
fn accumap_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_error, m)?)?;
    m.add_function(wrap_pyfunction!(return_corrected_array, m)?)?;
    Ok(())
}
