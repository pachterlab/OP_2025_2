"""Imaging-free reconstruction with a Poisson count model and a uniform-density prior.

Model: for receiver b at x_b and sender o at u_o,
    C_bo ~ Poisson( exp(q_b + a_o) * (K(|x_b - u_o|) + c) ),
with K a monotone transfer kernel (log K piecewise linear in log distance),
c a flat background, q_b receiver capture and a_o sender abundance.

Single pass (manuscript Methods, "Joint localization with a uniform-density prior"):
  1. rescale the receiver initialization (normally UMAP) to the known array
     diameter; place each sender at the count-weighted median of its receivers;
  2. fit K, c, a, q by Poisson maximum likelihood on that layout (no imaging);
  3. optimize a smooth displacement field applied to receivers and senders,
     maximizing  log-lik / phi  -  lam * (non-uniformity of local bead density);
  4. two rounds of per-bead refinement (grid search, likelihood only).

All length settings (kernel knots, warp grids, density bandwidths, search
windows) were set on arrays 3000 um across. Inputs are rescaled so the array
diameter equals REF_DIAMETER internally and outputs are returned in the
caller's units, so any array size can be used as long as `array_diameter` is
given in the same units as the returned coordinates.

Requires torch (CUDA strongly recommended; CPU works but is slow).
"""
import time
import numpy as np

REF_DIAMETER = 3000.0
LENGTH_REF_UM = None   # None (default, as published): every array is rescaled to REF_DIAMETER internal units and all lengths
                       # scale with it. A number (caller's units): for arrays larger than LENGTH_REF_UM, the local, kernel-scale
                       # lengths (_LOCAL_NAMES: kernel knots and table, near/far switch, meshes, refinement grid and windows)
                       # are divided by array_diameter / LENGTH_REF_UM so they keep that physical size, while
                       # array-scale settings (warp levels, density bandwidths, isotropy patches) stay relative and the isotropy
                       # prior's contact cap follows the bead spacing (ISO_MAXLEN_MED). Near pairs then grow linearly,
                       # not quadratically, with the bead count (large arrays).
_LOCAL_NAMES = ("KNOTS", "TAB_STEP", "TAB_MAX", "GRID", "REFINE_WINDOWS", "FAST_R1", "FAST_R2", "FAST_MESH", "FAST_SKIN",
                "REFINE_MESH")
_LOCAL_SAVED = None


def _local_factor(array_diameter):
    return 1.0 if LENGTH_REF_UM is None else max(1.0, float(array_diameter) / float(LENGTH_REF_UM))


class _local_scale:
    """Context: local lengths divided by _local_factor(array_diameter) (no-op if 1 or already applied by an outer call)."""
    def __init__(self, array_diameter):
        self.k = _local_factor(array_diameter); self.own = False

    def __enter__(self):
        global _LOCAL_SAVED
        if self.k == 1.0 or _LOCAL_SAVED is not None:
            return self
        g = globals(); _LOCAL_SAVED = {n: g[n] for n in _LOCAL_NAMES}; self.own = True
        for n, v in _LOCAL_SAVED.items():
            g[n] = tuple(x / self.k for x in v) if isinstance(v, tuple) else v / self.k
        return self

    def __exit__(self, *exc):
        global _LOCAL_SAVED
        if self.own:
            globals().update(_LOCAL_SAVED); _LOCAL_SAVED = None
        return False


KNOTS = np.array([5, 10, 20, 35, 50, 75, 100, 150, 200, 300, 500, 800, 1500, 4000.])
PHI = 30.0                     # likelihood temper in the warp stage
LAM = 300.0                    # density-prior weight
HS = (75.0, 150.0)             # density bandwidths
WARP_SCHEDULE = ((400, 200, 4.0), (200, 200, 3.0), (100, 200, 2.0))   # (grid spacing, steps, lr)
REFINE_WINDOWS = (15, 10)
TAB_STEP, TAB_MAX = 0.5, 4000.0
GRID = 5.0                     # normalizer grid for per-bead refinement
BUDGET = 40_000_000            # max elements per refinement batch tensor
CHUNK = 800                    # receiver rows per likelihood chunk
DENSE_FP32_BYTES = 1.5e9       # hold counts / kernel matrix densely in float32 up to this size
DENSE_FP16_BYTES = 3.5e9       # general path: kernel matrix in float16 up to this size, else recomputed
NNZ_CHUNK = 4_000_000          # nonzero count entries per likelihood chunk (general path)


def _torch():
    import torch
    return torch


def default_device():
    torch = _torch()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------- kernel --

FAST_COMPILE = True   # fuse the per-pair likelihood terms with torch.compile on CUDA (falls back to eager PyTorch)
FAST_COMPILE_CPU = True   # the same on CPU tensors


def _maybe_compiled(fn):
    """fn, compiled with torch.compile on first use (CUDA, or CPU with FAST_COMPILE_CPU); eager otherwise, and
    eager for good if compilation fails (results agree to floating-point rounding)."""
    state = {}

    def call(*args):
        torch = _torch()
        f = state.get("f")
        if f is None:
            f = fn
            if FAST_COMPILE and args[0].is_cuda:
                try:
                    import triton  # noqa: F401  (torch.compile's GPU backend)
                    f = torch.compile(fn)
                except Exception:
                    f = fn
            elif FAST_COMPILE_CPU and not args[0].is_cuda:
                try:
                    f = torch.compile(fn)        # C++ code generation; falls back below if no compiler is available
                except Exception:
                    f = fn
            state["f"] = f
        if f is fn:
            return fn(*args)
        try:
            return f(*args)
        except Exception as e:
            import warnings
            warnings.warn(f"accumap: torch.compile failed ({type(e).__name__}); using eager PyTorch")
            state["f"] = fn
            return fn(*args)
    return call


def _knot_values(w):
    torch = _torch()
    return torch.cat([w.new_zeros(1), -torch.cumsum(torch.nn.functional.softplus(w), 0)])


def _log_kernel(d, v, logk):
    torch = _torch()
    ld = torch.log(d.clamp(float(KNOTS[0]), float(KNOTS[-1])))
    idx = torch.bucketize(ld, logk[1:-1])
    t = (ld - logk[idx]) / (logk[idx + 1] - logk[idx])
    return v[idx] + t * (v[idx + 1] - v[idx])


def _log_kernel_seg(ld, v, logk):
    """_log_kernel as a sum of clamped segments (dense backward pass)."""
    out = v[0].expand_as(ld).clone()
    for k in range(len(KNOTS) - 1):
        w = logk[k + 1] - logk[k]
        out = out + (v[k + 1] - v[k]) / w * (ld - logk[k]).clamp(0.0, float(w))
    return out


def fit_kernel(X, U, C, dev, n_rows=6000, iters=300, lr=0.05, seed=0, chunk=1000, shared=False):
    """Poisson ML fit of K (knot values), c, a, q with positions fixed.
    X (receivers) and U (senders) are torch tensors in internal units; C dense or scipy sparse.
    shared=True: beads are both receivers and senders (U = X); self pairs are excluded."""
    torch = _torch()
    logk = torch.tensor(np.log(KNOTS), dtype=torch.float32, device=dev)
    n_rows = min(n_rows, len(X))
    rows = np.sort(np.random.RandomState(seed).choice(len(X), n_rows, replace=False))
    Cr = C[rows]
    CA = torch.tensor(Cr.toarray() if hasattr(Cr, "toarray") else Cr, dtype=torch.float32, device=dev)
    keep = None
    if shared:
        keep = torch.ones(CA.shape, dtype=torch.bool, device=dev)
        keep[torch.arange(n_rows, device=dev), torch.tensor(rows, device=dev)] = False
    LD = torch.log(torch.cdist(X[torch.tensor(rows, device=dev)], U).clamp(float(KNOTS[0]), float(KNOTS[-1])))
    nA, O = CA.shape
    a = torch.log(CA.sum(0) + 1.0); a = (a - a.mean()).clone().requires_grad_(True)
    q = (torch.log(CA.sum(1) + 1.0) - np.log(O)).clone().requires_grad_(True)
    v0 = -4.0 * np.log1p((KNOTS / 26.0) ** 2); v0 -= v0[0]          # Cauchy-like start
    dec = np.clip(-np.diff(v0), 1e-2, None)
    w = torch.tensor(np.log(np.expm1(dec)), dtype=torch.float32, device=dev, requires_grad=True)
    logc = torch.tensor(np.log(1e-4), dtype=torch.float32, device=dev, requires_grad=True)
    opt = torch.optim.Adam([a, q, w, logc], lr=lr)
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        for i in range(0, nA, chunk):
            sl = slice(i, i + chunk)
            v = _knot_values(w)
            mu = torch.exp(q[sl, None] + (a - a.mean())[None, :]) * (
                torch.exp(_log_kernel_seg(LD[sl], v, logk)) + torch.exp(logc)) + 1e-12
            el = CA[sl] * torch.log(mu) - mu
            if keep is not None:
                el = el * keep[sl]
            (-el.sum() / (nA * O)).backward()
        opt.step()
    with torch.no_grad():
        return _knot_values(w).detach(), float(torch.exp(logc))


def _kernel_table(v, dev, T=0.0):
    """K on a TAB_STEP radial table; T > 0: convolved with an isotropic 2D Gaussian of sd T (radial form
    via the scaled Bessel function I0, so the long tail and total mass are kept)."""
    torch = _torch()
    logk = torch.tensor(np.log(KNOTS), dtype=torch.float32, device=dev)
    r = torch.arange(0, TAB_MAX + TAB_STEP, TAB_STEP, dtype=torch.float32, device=dev)
    K = torch.exp(_log_kernel(r, v, logk))
    if T <= 0:
        return K
    from scipy.special import i0e
    rr = r.cpu().numpy().astype(np.float64); Kn = K.cpu().numpy().astype(np.float64)
    out = np.empty_like(rr)
    for i0 in range(0, len(rr), 400):
        a_ = rr[i0:i0 + 400, None]; b_ = rr[None, :]
        w = b_ / T ** 2 * np.exp(-(a_ - b_) ** 2 / (2 * T ** 2)) * i0e(a_ * b_ / T ** 2)
        out[i0:i0 + 400] = (w * Kn[None, :]).sum(1) * TAB_STEP
    return torch.tensor(out, dtype=torch.float32, device=dev)


def _logm(d, tab, c):
    torch = _torch()
    f = (d / TAB_STEP).clamp(0, len(tab) - 1.001)
    i = f.floor().long()
    t = f - i
    return torch.log(tab[i] * (1 - t) + tab[i + 1] * t + c)


def _kmat(A, B, tab, c, chunk=2000):
    torch = _torch()
    return torch.cat([torch.exp(_logm(torch.cdist(A[i:i + chunk], B), tab, c)) for i in range(0, len(A), chunk)])


def _ipf(M, Nr, Ns, a, iters=30):
    """Poisson ML of q (receivers) and a (senders) given the kernel matrix M."""
    torch = _torch()
    for _ in range(iters):
        q = torch.log(Nr) - torch.log(M @ torch.exp(a))
        a = torch.log(Ns) - torch.log(M.T @ torch.exp(q))
        a = a - a.mean()
    return q, a


# --------------------------------------------------------------- warp --

class _Warp:
    """Smooth displacement field, bicubic on a regular control grid."""

    def __init__(self, lo, hi, spacing, dev, init_field=None):
        torch = _torch()
        self.lo, self.hi = lo, hi
        l0, l1, h0, h1 = float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])
        nx = int(np.ceil((h0 - l0) / spacing)) + 1
        ny = int(np.ceil((h1 - l1) / spacing)) + 1
        if init_field is None:
            self.D = torch.zeros(1, 2, ny, nx, device=dev, requires_grad=True)
        else:   # resample the coarser warp onto this grid
            gx = torch.linspace(l0, h0, nx, device=dev); gy = torch.linspace(l1, h1, ny, device=dev)
            G = torch.stack(torch.meshgrid(gx, gy, indexing="xy"), -1).reshape(-1, 2)
            with torch.no_grad():
                d = init_field(G).T.reshape(1, 2, ny, nx)
            self.D = d.clone().requires_grad_(True)

    def __call__(self, P):
        torch = _torch()
        g = 2 * (P - self.lo) / (self.hi - self.lo) - 1
        out = torch.nn.functional.grid_sample(self.D, g[None, None], mode="bicubic", align_corners=True)
        return out[0, :, 0, :].T


def _ll_and_grad(X0, U0, warp, Ct, q, a, tab, c):
    """Poisson log-lik of the warped layout; accumulates (-ll / 1e6) grads into warp.D."""
    torch = _torch()
    for i in range(0, len(X0), CHUNK):
        U = U0 + warp(U0)
        Xc = X0[i:i + CHUNK] + warp(X0[i:i + CHUNK])
        d = torch.sqrt(((Xc[:, None, :] - U[None, :, :]) ** 2).sum(-1) + 1e-4)
        eta = q[i:i + CHUNK, None] + a[None, :] + _logm(d, tab, c)
        (-((Ct[i:i + CHUNK] * eta).sum() - torch.exp(eta).sum()) / 1e6).backward()


def density_penalty(P, footprint="disc", posed=None):
    """Squared deviation of log local bead density from its mean, summed over a grid
    inside the footprint, at bandwidths HS. Also returns each bandwidth's relative SD of density.
    footprint="disc": centroid-centred disc at the 85th-percentile bead radius (circular arrays).
    footprint="auto": any shape -- the region where the layout's own density (bandwidth max(HS))
    exceeds half its 90th percentile, holes filled, eroded by one bandwidth to stay off the edges."""
    torch = _torch()
    if isinstance(footprint, str) and footprint == "auto":
        return _density_penalty_auto(P)
    if not isinstance(footprint, str):
        return _density_penalty_polygon(P, footprint, posed)
    with torch.no_grad():
        c = P.mean(0)
        R = torch.quantile((P - c).norm(dim=1), 0.85)
    pen, sds = 0.0, []
    for h in HS:
        g = torch.arange(-float(R), float(R) + 1, h / 2, device=P.device)
        G = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
        G = G[G.norm(dim=1) < R] + c
        rho = _kde(G, P, h)            # all pairs (exact path) or FFT (fast path)
        lr = torch.log(rho)
        if EDGE_CORRECT:
            with torch.no_grad():
                r99 = float(torch.quantile((P - c).norm(dim=1), EDGE_SUPPORT))
                t = np.linspace(0, 2 * np.pi, 256, endpoint=False)
                cc = c.detach().cpu().numpy()
                lr = lr - torch.log(_edge_kappa(G, cc + r99 * np.c_[np.cos(t), np.sin(t)], h))
        pen = pen + ((lr - lr.mean()) ** 2).sum()
        sds.append(float((rho.std() / rho.mean()).detach()))
    return pen, sds


KDE_FFT = False          # set by the fast path: Gaussian KDEs by mesh spreading + FFT instead of all pairs


def _kde(G, P, h, chunk=512):
    torch = _torch()
    if KDE_FFT:
        return _kde_fft(G, P, h)
    return torch.cat([torch.exp(-torch.cdist(G[i:i + chunk], P) ** 2 / (2 * h * h)).sum(1)
                      for i in range(0, len(G), chunk)])


DENSITY_REFRESH = 50   # warp steps between recomputations of the disc density-prior geometry (1: every step)


def _disc_geometry(P):
    """Centre, 85th-percentile radius, evaluation grids and FFT meshes of the disc density prior, for reuse
    over DENSITY_REFRESH warp steps (the mesh margin leaves room for the beads to move in between)."""
    torch = _torch()
    with torch.no_grad():
        c = P.mean(0)
        R = torch.quantile((P - c).norm(dim=1), 0.85)
        lo_p = P.min(0).values; hi_p = P.max(0).values
        out = []
        for h in HS:
            g = torch.arange(-float(R), float(R) + 1, h / 2, device=P.device)
            G = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
            G = G[G.norm(dim=1) < R] + c
            lo = torch.minimum(G.min(0).values, lo_p) - 4 * h - 200
            hi = torch.maximum(G.max(0).values, hi_p) + 4 * h + 200
            M = _Mesh(lo, hi, h / 4, None, 0.0, False, P.device, fn=lambda r, h=h: torch.exp(-r * r / (2 * h * h)))
            out.append((G, M))
    return out


def _density_penalty_cached(P, geom):
    torch = _torch()
    pen = 0.0
    w = torch.ones(len(P), device=P.device, dtype=P.dtype)
    for G, M in geom:
        lr = torch.log(M.sample(M.convolve(M.spread(P, w)), G))
        pen = pen + ((lr - lr.mean()) ** 2).sum()
    return pen, []


def _kde_fft(G, P, h):
    """sum_i exp(-|g - p_i|^2 / 2h^2) via bilinear spreading onto a mesh of spacing h/4, FFT convolution
    and bilinear sampling (differentiable in P)."""
    torch = _torch()
    both = torch.cat([G.detach(), P.detach()])
    lo = both.min(0).values - 4 * h; hi = both.max(0).values + 4 * h
    M = _Mesh(lo, hi, h / 4, None, 0.0, False, P.device, fn=lambda r: torch.exp(-r * r / (2 * h * h)))
    w = torch.ones(len(P), device=P.device, dtype=P.dtype)
    return M.sample(M.convolve(M.spread(P, w)), G)


POLY_ANGLES = 360       # rotations searched when posing a polygon footprint (full circle, 1 degree)
POSE_QUANTILE = 0.98    # the rotation is chosen at this containment quantile (set by the outline's corners)
EDGE_CORRECT = False    # divide densities by those of a uniform array filling the outline (kills KDE edge bias)
EDGE_SUPPORT = 0.99     # the array outline for edge correction: the posed footprint containing this fraction
POLY_FRACTION = 0.85    # posed footprint is shrunk about the centroid until it contains this fraction of beads


def _gauge(pts, poly):
    """Scale factor by which a convex polygon (centred at the origin) must be enlarged to contain each
    point: max over edges of n.p / h (n: outward unit normal, h: edge distance from the origin)."""
    a, b = poly, np.roll(poly, -1, axis=0)
    e = b - a
    n = np.c_[e[:, 1], -e[:, 0]]
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    h = (n * a).sum(1)
    n = np.where(h[:, None] < 0, -n, n); h = np.abs(h)
    return (pts @ n.T / h).max(1)


def _is_convex(poly):
    e = np.roll(poly, -1, axis=0) - poly
    cr = e[:, 0] * np.roll(e[:, 1], -1) - e[:, 1] * np.roll(e[:, 0], -1)
    return np.all(cr >= -1e-9) or np.all(cr <= 1e-9)


def _pose_polygon(P, poly, frac=None):
    """Pose the footprint polygon on the layout: centred on the layout's centroid, at the rotation where
    POSE_QUANTILE of the beads fit inside the smallest scaled copy of the polygon (the best-aligned
    pose), then scaled so that POLY_FRACTION of beads are inside (for a disc: exactly the
    85th-percentile-radius rule)."""
    from matplotlib.path import Path
    Pn = P.detach().cpu().numpy() if hasattr(P, "detach") else np.asarray(P)
    c = Pn.mean(0)
    sub = Pn[np.random.RandomState(0).choice(len(Pn), min(len(Pn), 5000), replace=False)] - c
    poly = np.asarray(poly, np.float64)
    convex = _is_convex(poly)
    def scale(rot, frac):
        if convex:
            return np.quantile(_gauge(sub, rot), frac)
        lo_f, hi_f = 0.0, 4.0
        for _ in range(25):
            m = (lo_f + hi_f) / 2
            if Path(rot * m).contains_points(sub).mean() < frac:
                lo_f = m
            else:
                hi_f = m
        return hi_f

    best_f, best_rot = np.inf, None
    for th in np.linspace(0, 2 * np.pi, POLY_ANGLES, endpoint=False):
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        rot = poly @ R.T
        f = scale(rot, POSE_QUANTILE)
        if f < best_f:
            best_f, best_rot = f, rot
    return best_rot * scale(best_rot, POLY_FRACTION if frac is None else frac) + c


def _edge_kappa(G, support, h):
    """Unnormalized density a uniform array filling polygon `support` would have at points G (bandwidth h)."""
    torch = _torch()
    from matplotlib.path import Path
    hr = h / 4
    lo, hi = support.min(0), support.max(0)
    gx = np.arange(lo[0], hi[0] + hr, hr); gy = np.arange(lo[1], hi[1] + hr, hr)
    YX, YY = np.meshgrid(gx, gy, indexing="ij")
    Y = np.c_[YX.ravel(), YY.ravel()]
    Y = torch.tensor(Y[Path(support).contains_points(Y)], dtype=G.dtype, device=G.device)
    return _kde(G, Y, h)


def _density_penalty_polygon(P, poly, posed=None):
    """Density penalty inside a known footprint (polygon, internal units, centred), posed on the layout
    (or inside an already posed polygon)."""
    torch = _torch()
    from matplotlib.path import Path
    support = None
    with torch.no_grad():
        if posed is None:
            posed = _pose_polygon(P, np.asarray(poly, np.float64))
        elif isinstance(posed, tuple):
            posed, support = posed
        if EDGE_CORRECT and support is None:
            support = _pose_polygon(P, np.asarray(poly, np.float64), frac=EDGE_SUPPORT)
        lo, hi = posed.min(0) - 10, posed.max(0) + 10
    pen, sds = 0.0, []
    for h in HS:
        step = h / 2
        gx = np.arange(lo[0], hi[0] + step, step); gy = np.arange(lo[1], hi[1] + step, step)
        GX, GY = np.meshgrid(gx, gy, indexing="ij")
        inside = Path(posed).contains_points(np.c_[GX.ravel(), GY.ravel()]).reshape(GX.shape)
        G = torch.tensor(np.c_[GX[inside], GY[inside]], dtype=P.dtype, device=P.device)
        rho = _kde(G, P, h)
        lr = torch.log(rho)
        if EDGE_CORRECT:
            with torch.no_grad():
                lr_k = torch.log(_edge_kappa(G, support, h))
            lr = lr - lr_k
        pen = pen + ((lr - lr.mean()) ** 2).sum()
        sds.append(float((rho.std() / rho.mean()).detach()))
    return pen, sds


def _density_penalty_auto(P):
    torch = _torch()
    from scipy import ndimage
    hb = max(HS)
    with torch.no_grad():
        lo = P.min(0).values - hb; hi = P.max(0).values + hb
        step = hb / 2
        gx = torch.arange(float(lo[0]), float(hi[0]) + step, step, device=P.device)
        gy = torch.arange(float(lo[1]), float(hi[1]) + step, step, device=P.device)
        Gc = torch.stack(torch.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
        rc = _kde(Gc, P, hb).reshape(len(gx), len(gy)).cpu().numpy()
        inside = ndimage.binary_fill_holes(rc >= 0.5 * np.quantile(rc, 0.9))
        inside = ndimage.binary_erosion(inside, iterations=2)          # one bandwidth (2 grid steps)
        mask = torch.tensor(inside, device=P.device)
    pen, sds = 0.0, []
    for h in HS:
        g0 = torch.arange(float(lo[0]), float(hi[0]) + 1, h / 2, device=P.device)
        g1 = torch.arange(float(lo[1]), float(hi[1]) + 1, h / 2, device=P.device)
        G = torch.stack(torch.meshgrid(g0, g1, indexing="ij"), -1).reshape(-1, 2)
        ix = ((G[:, 0] - lo[0]) / step).round().long().clamp(0, len(gx) - 1)
        iy = ((G[:, 1] - lo[1]) / step).round().long().clamp(0, len(gy) - 1)
        G = G[mask[ix, iy]]
        rho = _kde(G, P, h)
        lr = torch.log(rho)
        pen = pen + ((lr - lr.mean()) ** 2).sum()
        sds.append(float((rho.std() / rho.mean()).detach()))
    return pen, sds


def warp_stage(X0, U0, Ct, tab, c, phi=PHI, lam=LAM, verbose=False, t0=None):
    torch = _torch()
    dev = X0.device
    Nr, Ns = Ct.sum(1), Ct.sum(0)
    allP = torch.cat([X0, U0])
    lo = allP.min(0).values - 200; hi = allP.max(0).values + 200
    a = torch.zeros(len(Ns), device=dev)
    with torch.no_grad():
        M = _kmat(X0, U0, tab, c); q, a = _ipf(M, Nr, Ns, a); del M
    warp = None
    for spacing, steps, lr in WARP_SCHEDULE:
        warp = _Warp(lo, hi, spacing, dev, init_field=warp)
        opt = torch.optim.Adam([warp.D], lr=lr)
        for it in range(steps):
            opt.zero_grad(set_to_none=True)
            _ll_and_grad(X0, U0, warp, Ct, q, a, tab, c)
            if warp.D.grad is not None:
                warp.D.grad.mul_(1e6 / phi)
            pen, _ = density_penalty(torch.cat([X0 + warp(X0), U0 + warp(U0)]))
            (lam * pen).backward()
            opt.step()
            if (it + 1) % 50 == 0:
                with torch.no_grad():
                    X = X0 + warp(X0); U = U0 + warp(U0)
                    M = _kmat(X, U, tab, c); q, a = _ipf(M, Nr, Ns, a); del M
        if verbose:
            print(f"  warp grid {spacing}: done ({time.time() - t0:.0f}s)", flush=True)
    with torch.no_grad():
        return X0 + warp(X0), U0 + warp(U0), q, a


# ---------------------------------------------------------- refinement --

def _padded(Cmat, dev):
    torch = _torch()
    nnz = (Cmat > 0).sum(1)
    K = max(int(nnz.max()), 1)
    idx = np.zeros((Cmat.shape[0], K), np.int64)
    val = np.zeros((Cmat.shape[0], K), np.float32)
    for i in range(Cmat.shape[0]):
        nz = np.nonzero(Cmat[i])[0]
        idx[i, :len(nz)] = nz
        val[i, :len(nz)] = Cmat[i, nz]
    return (torch.tensor(idx, device=dev), torch.tensor(val, device=dev),
            torch.tensor(nnz, device=dev), np.argsort(nnz))


def _density_grid(P, w, tab, c, margin):
    torch = _torch()
    lo = P.min(0).values - margin
    hi = P.max(0).values + margin
    gx = torch.arange(float(lo[0]), float(hi[0]) + GRID, GRID, device=P.device)
    gy = torch.arange(float(lo[1]), float(hi[1]) + GRID, GRID, device=P.device)
    G = torch.stack(torch.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
    Z = torch.cat([torch.exp(_logm(torch.cdist(G[i:i + 4096], P), tab, c)) @ w for i in range(0, len(G), 4096)])
    return Z.reshape(len(gx), len(gy)), lo


def _tile_near_eager(G, P, w, tab, c):
    """sum_p w_p K(|g - p|) (1 - S(|g - p|)) for grid points G (near part of the refinement normalizer)."""
    torch = _torch()
    d = torch.cdist(G, P)
    f = (d / TAB_STEP).clamp(0, tab.shape[0] - 1.001)
    i = f.floor().long()
    t = f - i
    k = torch.exp(torch.log(tab[i] * (1 - t) + tab[i + 1] * t + c)) - c       # as _kval
    x = ((d - FAST_R1) / (FAST_R2 - FAST_R1)).clamp(0, 1)
    return (k * (1 - x * x * (3 - 2 * x))) @ w


_tile_near = _maybe_compiled(_tile_near_eager)


def _density_grid_split(P, w, tab, c, margin, tile=64):
    """As _density_grid (sum_o w_o m(|g - p_o|) on a GRID-spaced mesh), with the kernel split: pairs
    within FAST_R2 summed exactly per tile of grid points, the smooth far part K*S by FFT on a
    FAST_MESH mesh, the background analytically."""
    torch = _torch()
    lo = P.min(0).values - margin
    hi = P.max(0).values + margin
    gx = torch.arange(float(lo[0]), float(hi[0]) + GRID, GRID, device=P.device)
    gy = torch.arange(float(lo[1]), float(hi[1]) + GRID, GRID, device=P.device)
    Z = torch.empty(len(gx), len(gy), device=P.device)
    ct = torch.as_tensor(float(c), dtype=torch.float32, device=P.device)
    for i0 in range(0, len(gx), tile):
        for j0 in range(0, len(gy), tile):
            tx, ty = gx[i0:i0 + tile], gy[j0:j0 + tile]
            Gt = torch.stack(torch.meshgrid(tx, ty, indexing="ij"), -1).reshape(-1, 2)
            sel = ((P[:, 0] > tx[0] - FAST_R2) & (P[:, 0] < tx[-1] + FAST_R2)
                   & (P[:, 1] > ty[0] - FAST_R2) & (P[:, 1] < ty[-1] + FAST_R2))
            if sel.any():
                nz = _tile_near(Gt, P[sel], w[sel], tab, ct)
            else:
                nz = torch.zeros(len(Gt), device=P.device)
            Z[i0:i0 + tile, j0:j0 + tile] = nz.reshape(len(tx), len(ty))
    mesh = _Mesh(lo - FAST_R2, hi + FAST_R2, FAST_MESH, tab, c, True, P.device)
    GX, GY = torch.meshgrid(gx, gy, indexing="ij")
    far = mesh.sample(mesh.convolve(mesh.spread(P, w)), torch.stack([GX.reshape(-1), GY.reshape(-1)], -1))
    return Z + far.reshape(Z.shape) + c * w.sum(), lo


SPLIT_NORMALIZER = False   # set by the fast path: refinement normalizer via _density_grid_split


def _interp(Zg, lo, pts):
    torch = _torch()
    f = (pts - lo) / GRID
    nx, ny = Zg.shape
    ix = f[:, 0].clamp(0, nx - 1.001); iy = f[:, 1].clamp(0, ny - 1.001)
    x0, y0 = ix.floor().long(), iy.floor().long()
    tx, ty = ix - x0, iy - y0
    return ((1 - tx) * (1 - ty) * Zg[x0, y0] + tx * (1 - ty) * Zg[x0 + 1, y0]
            + (1 - tx) * ty * Zg[x0, y0 + 1] + tx * ty * Zg[x0 + 1, y0 + 1])


def _step_side(Pos, pad, N, Other, w_other, W, tab, c):
    """Move every point of one side to its best position on a (2W)^2 grid, other side fixed,
    by the capture-profiled log-likelihood."""
    torch = _torch()
    idx, val, nnz, order = pad
    dev = Pos.device
    h = torch.linspace(-W, W, 41, device=dev)
    G = torch.stack(torch.meshgrid(h, h, indexing="ij"), -1).reshape(-1, 2)
    nc = len(G)
    Zg, lo = _density_grid(Other, w_other, tab, c, margin=W + 50)
    new = Pos.clone()
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and (j - i + 1) * nc * max(int(nnz[order[j]]), 1) <= BUDGET:
            j += 1
        j = max(j, i + 1)
        rows = torch.tensor(order[i:j], device=dev)
        kmax = max(int(nnz[rows].max()), 1)
        cand = Pos[rows][:, None, :] + G[None]
        nb_pts = Other[idx[rows, :kmax]]
        obs = (_logm(torch.cdist(cand, nb_pts), tab, c) * val[rows, None, :kmax]).sum(-1)
        Z = _interp(Zg, lo, cand.reshape(-1, 2)).reshape(len(rows), nc)
        ll = obs - N[rows, None] * torch.log(Z)
        best = ll.argmax(1)
        new[rows] = cand[torch.arange(len(rows), device=dev), best]
        i = j
    return new


def refine(X, U, C, tab, c, q, a):
    torch = _torch()
    dev = X.device
    Ct = torch.tensor(C, dtype=torch.float32, device=dev)
    Nr, Ns = Ct.sum(1), Ct.sum(0)
    del Ct
    padR = _padded(C, dev); padS = _padded(np.ascontiguousarray(C.T), dev)
    for W in REFINE_WINDOWS:
        M = _kmat(X, U, tab, c); q, a = _ipf(M, Nr, Ns, a); del M
        X = _step_side(X, padR, Nr, U, torch.exp(a), W, tab, c)
        M = _kmat(X, U, tab, c); q, a = _ipf(M, Nr, Ns, a); del M
        U = _step_side(U, padS, Ns, X, torch.exp(q), W, tab, c)
    return X, U


# ------------------------------------------------ general (large / shared) path --
# Used for arrays whose count matrix does not fit densely in memory, and for shared-bead
# arrays (every bead both sender and receiver, U = X, self pairs excluded). The distinct-bead
# path above is kept unchanged for arrays that fit.

class _Counts:
    """Sparse counts on the device, plus the kernel-matrix storage policy."""

    def __init__(self, Csp, dev, shared):
        torch = _torch()
        coo = Csp.tocoo()
        self.shared = shared
        self.n_r, self.n_o = Csp.shape
        self.rows = torch.tensor(coo.row, dtype=torch.long, device=dev)
        self.cols = torch.tensor(coo.col, dtype=torch.long, device=dev)
        self.vals = torch.tensor(coo.data, dtype=torch.float32, device=dev)
        self.Nr = torch.tensor(np.asarray(Csp.sum(1)).ravel(), dtype=torch.float32, device=dev)
        self.Ns = torch.tensor(np.asarray(Csp.sum(0)).ravel(), dtype=torch.float32, device=dev)
        size = self.n_r * self.n_o
        self.kernel_store = "fp32" if size * 4 <= DENSE_FP32_BYTES else ("fp16" if size * 2 <= DENSE_FP16_BYTES else "chunked")


def _kmat_fused_eager(A, B, tab, c):
    """m(|a - b|) = K + c for all pairs (dense block), as exp(_logm(cdist))."""
    torch = _torch()
    d = torch.cdist(A, B)
    f = (d / TAB_STEP).clamp(0, tab.shape[0] - 1.001)
    i = f.floor().long()
    t = f - i
    return torch.exp(torch.log(tab[i] * (1 - t) + tab[i + 1] * t + c))


_kmat_fused = _maybe_compiled(_kmat_fused_eager)


def _kblock(Xc, U, tab, c, i0, shared):
    """Kernel-matrix rows for receivers Xc (global offset i0) against all senders; self pairs zeroed."""
    torch = _torch()
    K = _kmat_fused(Xc, U, tab, torch.as_tensor(float(c), dtype=torch.float32, device=Xc.device))
    if shared:
        r = torch.arange(len(Xc), device=Xc.device)
        K[r, i0 + r] = 0.0
    return K


def _ipf_general(X, U, tab, c, cnt, a, iters=30, chunk=2000):
    torch = _torch()
    if cnt.kernel_store in ("fp32", "fp16"):
        dt = torch.float32 if cnt.kernel_store == "fp32" else torch.float16
        M = torch.empty(len(X), len(U), dtype=dt, device=X.device)    # filled in place: one copy in memory
        for i in range(0, len(X), chunk):
            M[i:i + chunk] = _kblock(X[i:i + chunk], U, tab, c, i, cnt.shared).to(dt)
        for _ in range(iters):
            ea = torch.exp(a)
            s = torch.cat([M[i:i + chunk].float() @ ea for i in range(0, len(X), chunk)])
            q = torch.log(cnt.Nr) - torch.log(s)
            eq = torch.exp(q)
            t = sum(M[i:i + chunk].float().T @ eq[i:i + chunk] for i in range(0, len(X), chunk))
            a = torch.log(cnt.Ns) - torch.log(t)
            a = a - a[torch.isfinite(a)].mean()   # a bead that never sends has a = -inf
        del M
        return q, a
    for _ in range(iters):
        ea = torch.exp(a)
        s = torch.cat([_kblock(X[i:i + chunk], U, tab, c, i, cnt.shared) @ ea for i in range(0, len(X), chunk)])
        q = torch.log(cnt.Nr) - torch.log(s)
        eq = torch.exp(q)
        t = sum(_kblock(X[i:i + chunk], U, tab, c, i, cnt.shared).T @ eq[i:i + chunk] for i in range(0, len(X), chunk))
        a = torch.log(cnt.Ns) - torch.log(t)
        a = a - a[torch.isfinite(a)].mean()   # a bead that never sends has a = -inf
    return q, a


def _ll_and_grad_general(X0, U0, warp, cnt, q, a, tab, c):
    """As _ll_and_grad, from sparse counts: sum over nonzero pairs of C log m, minus the dense
    expected-count term (self pairs excluded for shared beads); grads (-ll / 1e6) into warp.D."""
    torch = _torch()
    for i in range(0, len(X0), CHUNK):
        U = U0 + warp(U0)
        Xc = X0[i:i + CHUNK] + warp(X0[i:i + CHUNK])
        d = torch.sqrt(((Xc[:, None, :] - U[None, :, :]) ** 2).sum(-1) + 1e-4)
        E = torch.exp(q[i:i + CHUNK, None] + a[None, :] + _logm(d, tab, c))
        tot = E.sum()
        if cnt.shared:
            r = torch.arange(len(Xc), device=Xc.device)
            tot = tot - E[r, i + r].sum()
        (tot / 1e6).backward()
    for j in range(0, len(cnt.vals), NNZ_CHUNK):
        r, o, v = cnt.rows[j:j + NNZ_CHUNK], cnt.cols[j:j + NNZ_CHUNK], cnt.vals[j:j + NNZ_CHUNK]
        xr = X0[r] + warp(X0[r]); uo = U0[o] + warp(U0[o])
        d = torch.sqrt(((xr - uo) ** 2).sum(-1) + 1e-4)
        (-(v * _logm(d, tab, c)).sum() / 1e6).backward()


def warp_stage_general(X0, U0, cnt, tab, c, phi=PHI, lam=LAM, footprint="disc", verbose=False, t0=None):
    torch = _torch()
    dev = X0.device
    allP = X0 if cnt.shared else torch.cat([X0, U0])
    lo = allP.min(0).values - 200; hi = allP.max(0).values + 200
    a = torch.zeros(cnt.n_o, device=dev)
    with torch.no_grad():
        q, a = _ipf_general(X0, U0, tab, c, cnt, a)
    warp = None
    for spacing, steps, lr in WARP_SCHEDULE:
        warp = _Warp(lo, hi, spacing, dev, init_field=warp)
        opt = torch.optim.Adam([warp.D], lr=lr)
        for it in range(steps):
            opt.zero_grad(set_to_none=True)
            _ll_and_grad_general(X0, U0, warp, cnt, q, a, tab, c)
            if warp.D.grad is not None:
                warp.D.grad.mul_(1e6 / phi)
            P = X0 + warp(X0) if cnt.shared else torch.cat([X0 + warp(X0), U0 + warp(U0)])
            if not isinstance(footprint, str) and it % 50 == 0:
                posed = (_pose_polygon(P, footprint),       # footprint pose refreshed every 50 steps
                         _pose_polygon(P, footprint, frac=EDGE_SUPPORT) if EDGE_CORRECT else None)
            pen, _ = density_penalty(P, footprint, None if isinstance(footprint, str) else posed)
            (lam * pen).backward()
            opt.step()
            if (it + 1) % 50 == 0:
                with torch.no_grad():
                    X = X0 + warp(X0); U = X if cnt.shared else U0 + warp(U0)
                    q, a = _ipf_general(X, U, tab, c, cnt, a)
        if verbose:
            print(f"  warp grid {spacing}: done ({time.time() - t0:.0f}s)", flush=True)
    with torch.no_grad():
        X = X0 + warp(X0)
        return X, (X if cnt.shared else U0 + warp(U0)), q, a


def _padded_sparse(Csr, dev):
    """Per-row partner lists padded to the longest row; kept in CPU memory (the longest row can be
    ~10x the median), batches are moved to the device in _side_ll."""
    torch = _torch()
    nnz = np.diff(Csr.indptr)
    K = max(int(nnz.max()), 1)
    idx = np.zeros((Csr.shape[0], K), np.int64)
    val = np.zeros((Csr.shape[0], K), np.float32)
    for i in range(Csr.shape[0]):
        a, b = Csr.indptr[i], Csr.indptr[i + 1]
        idx[i, :b - a] = Csr.indices[a:b]
        val[i, :b - a] = Csr.data[a:b]
    return (torch.tensor(idx), torch.tensor(val), torch.tensor(nnz), np.argsort(nnz))


def _repel_lists(Pos, All, self_offset, radius):
    """For each point of Pos (whose own index in All is self_offset + i), the positions of the other beads
    in All within `radius`, padded (n, k, 2), with a validity mask (n, k)."""
    torch = _torch()
    from scipy.spatial import cKDTree
    P, A = Pos.detach().cpu().numpy(), All.detach().cpu().numpy()
    lists = cKDTree(A).query_ball_point(P, radius)
    lists = [[j for j in l if j != self_offset + i] for i, l in enumerate(lists)]
    k = max(1, max(len(l) for l in lists))
    pos = np.zeros((len(P), k, 2), np.float32); m = np.zeros((len(P), k), np.float32)
    for i, l in enumerate(lists):
        if l:
            pos[i, :len(l)] = A[l]; m[i, :len(l)] = 1
    return torch.tensor(pos, device=Pos.device), torch.tensor(m, device=Pos.device)


def _cand_obs_eager(cand, P, vv, tab, c):
    """sum_k vv_k log m(|cand_j - P_k|) for every point's candidates (n, n_cand) (refinement scoring)."""
    torch = _torch()
    d = torch.cdist(cand, P)
    f = (d / TAB_STEP).clamp(0, tab.shape[0] - 1.001)
    i = f.floor().long()
    t = f - i
    return (torch.log(tab[i] * (1 - t) + tab[i + 1] * t + c) * vv[:, None, :]).sum(-1)


_cand_obs = _maybe_compiled(_cand_obs_eager)


def _side_ll(Pos, pad, N, Other, w_other, G, tab, c, self_w=None, mG=None, repel=None, center=None, Zpre=None):
    """Capture-profiled log-likelihood of every point of one side at every candidate offset G
    (other side fixed): returns (n_points, n_candidates). self_w/mG remove each point's own
    contribution to the normalizer (shared beads: a bead is not its own partner)."""
    torch = _torch()
    idx, val, nnz, order = pad
    dev = Pos.device
    nc = len(G)
    W = float(G.abs().max())
    Zg, lo = Zpre if Zpre is not None else \
        (_density_grid_split if SPLIT_NORMALIZER else _density_grid)(Other, w_other, tab, c, margin=W + 50)
    Ctr = Pos if center is None else center
    ct = torch.as_tensor(float(c), dtype=torch.float32, device=dev)
    out = torch.empty(len(Pos), nc, device=dev)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and (j - i + 1) * nc * max(int(nnz[order[j]]), 1) <= BUDGET:
            j += 1
        j = max(j, i + 1)
        rows_c = torch.tensor(order[i:j])
        rows = rows_c.to(dev)
        kmax = max(int(nnz[rows_c].max()), 1)
        cand = Ctr[rows][:, None, :] + G[None]
        nb = idx[rows_c, :kmax].to(dev); vv = val[rows_c, :kmax].to(dev)
        obs = _cand_obs(cand, Other[nb], vv, tab, ct)
        Z = _interp(Zg, lo, cand.reshape(-1, 2)).reshape(len(rows), nc)
        if self_w is not None:
            if center is None:
                Z = (Z - self_w[rows, None] * mG[None, :]).clamp(min=1e-30)
            else:
                dself = (cand - Pos[rows][:, None, :]).norm(dim=-1)
                Z = (Z - self_w[rows, None] * torch.exp(_logm(dself, tab, c))).clamp(min=1e-30)
        ll = obs - N[rows, None] * torch.log(Z)
        if repel is not None:
            rp, rm = repel
            dd = torch.cdist(cand, rp[rows])
            ll = ll - REPEL_WEIGHT * (((REPEL_SPACING - dd).clamp(min=0) ** 2) * rm[rows, None, :]).sum(-1) / REPEL_SPACING ** 2
        out[rows] = ll
        i = j
    return out


REFINE_ROUNDS = "converge"   # refinement rounds per stage: "converge" (until fewer than REFINE_TOL of beads move) or an int
REFINE_TOL, REFINE_MAX_ROUNDS = 0.01, 12
REFINE_IPF_ITERS = 30   # IPF iterations per refinement round after the first (warm-started; the first uses 30)
REFINE_LOG = []        # per-round movement, appended by _refine_c2f (diagnostic)
REFINE_CALLBACK = None   # optional callable(X, U) after each refinement round (diagnostic)
REFINE_C2F = False   # set by the fast path: coarse (COARSE_GRID^2 over +-W) then fine (9^2) candidate search
COARSE_GRID = 13


def _c2f_grids(W, dev):
    torch = _torch()
    hc = torch.linspace(-W, W, COARSE_GRID, device=dev)
    step = 2 * W / (COARSE_GRID - 1)
    hf = torch.linspace(-step, step, 9, device=dev)
    return (torch.stack(torch.meshgrid(hc, hc, indexing="ij"), -1).reshape(-1, 2),
            torch.stack(torch.meshgrid(hf, hf, indexing="ij"), -1).reshape(-1, 2))


def _refine_c2f(X, U, Csr, cnt, tab, c, q, a, padR, padS):
    """refine_general with a coarse-to-fine candidate search; the normalizer grid of each pass is
    computed once and shared by both stages."""
    torch = _torch()
    dev = X.device
    norm = _density_grid_split if SPLIT_NORMALIZER else _density_grid
    k = 0
    while True:
        if REFINE_ROUNDS == "converge":
            if k >= REFINE_MAX_ROUNDS or (k >= len(REFINE_WINDOWS) and REFINE_LOG and REFINE_LOG[-1]["moved_frac"] < REFINE_TOL):
                break
        elif k >= max(int(REFINE_ROUNDS), 1):
            break
        W = REFINE_WINDOWS[min(k, len(REFINE_WINDOWS) - 1)]   # later rounds reuse the last (smallest) window
        Xprev, Uprev = X.clone(), U.clone()
        Gc, Gf = _c2f_grids(W, dev)
        rad = W + (REPEL_SPACING or 0.0)
        if cnt.shared:
            q, a = _ipf_general(X, X, tab, c, cnt, a, iters=30 if k == 0 else REFINE_IPF_ITERS)
            ea, eq = torch.exp(a), torch.exp(q)
            ZR = norm(X, ea, tab, c, margin=W + 50); ZC = norm(X, eq, tab, c, margin=W + 50)
            rep = _repel_lists(X, X, 0, rad) if REPEL_SPACING else None
            mGc = torch.exp(_logm(Gc.norm(dim=1), tab, c))
            llc = (_side_ll(X, padR, cnt.Nr, X, ea, Gc, tab, c, ea, mGc, repel=rep, Zpre=ZR)
                   + _side_ll(X, padS, cnt.Ns, X, eq, Gc, tab, c, eq, mGc, Zpre=ZC))
            Ctr = X + Gc[llc.argmax(1)]
            llf = (_side_ll(X, padR, cnt.Nr, X, ea, Gf, tab, c, ea, None, repel=rep, center=Ctr, Zpre=ZR)
                   + _side_ll(X, padS, cnt.Ns, X, eq, Gf, tab, c, eq, None, center=Ctr, Zpre=ZC))
            X = Ctr + Gf[llf.argmax(1)]
            U = X
        else:
            q, a = _ipf_general(X, U, tab, c, cnt, a, iters=30 if k == 0 else REFINE_IPF_ITERS)
            ea = torch.exp(a)
            Z = norm(U, ea, tab, c, margin=W + 50)
            rep = _repel_lists(X, torch.cat([X, U]), 0, rad) if REPEL_SPACING else None
            Ctr = X + Gc[_side_ll(X, padR, cnt.Nr, U, ea, Gc, tab, c, repel=rep, Zpre=Z).argmax(1)]
            X = Ctr + Gf[_side_ll(X, padR, cnt.Nr, U, ea, Gf, tab, c, repel=rep, center=Ctr, Zpre=Z).argmax(1)]
            q, a = _ipf_general(X, U, tab, c, cnt, a, iters=30 if k == 0 else REFINE_IPF_ITERS)
            eq = torch.exp(q)
            Z = norm(X, eq, tab, c, margin=W + 50)
            rep = _repel_lists(U, torch.cat([X, U]), len(X), rad) if REPEL_SPACING else None
            Ctr = U + Gc[_side_ll(U, padS, cnt.Ns, X, eq, Gc, tab, c, repel=rep, Zpre=Z).argmax(1)]
            U = Ctr + Gf[_side_ll(U, padS, cnt.Ns, X, eq, Gf, tab, c, repel=rep, center=Ctr, Zpre=Z).argmax(1)]
        with torch.no_grad():
            mv = torch.cat([(X - Xprev).norm(dim=1), (U - Uprev).norm(dim=1)]) if not cnt.shared else (X - Xprev).norm(dim=1)
            REFINE_LOG.append({"window": float(W), "moved_frac": float((mv > 0.25).float().mean()),
                               "moved_median": float(mv.median())})
            if REFINE_CALLBACK is not None:      # diagnostic hook (does not affect the result)
                REFINE_CALLBACK(X, U)
        k += 1
    return X, U


def refine_general(X, U, Csr, cnt, tab, c, q, a):
    torch = _torch()
    dev = X.device
    padR = _padded_sparse(Csr, dev)
    padS = _padded_sparse(Csr.T.tocsr(), dev)
    if REFINE_C2F:
        return _refine_c2f(X, U, Csr, cnt, tab, c, q, a, padR, padS)
    for W in REFINE_WINDOWS:
        h = torch.linspace(-W, W, 41, device=dev)
        G = torch.stack(torch.meshgrid(h, h, indexing="ij"), -1).reshape(-1, 2)
        rad = W + (REPEL_SPACING or 0.0)
        if cnt.shared:
            q, a = _ipf_general(X, X, tab, c, cnt, a)
            mG = torch.exp(_logm(G.norm(dim=1), tab, c))
            rep = _repel_lists(X, X, 0, rad) if REPEL_SPACING else None
            ll = (_side_ll(X, padR, cnt.Nr, X, torch.exp(a), G, tab, c, torch.exp(a), mG, repel=rep)
                  + _side_ll(X, padS, cnt.Ns, X, torch.exp(q), G, tab, c, torch.exp(q), mG))
            X = X + G[ll.argmax(1)]
            U = X
        else:
            q, a = _ipf_general(X, U, tab, c, cnt, a)
            rep = _repel_lists(X, torch.cat([X, U]), 0, rad) if REPEL_SPACING else None
            X = X + G[_side_ll(X, padR, cnt.Nr, U, torch.exp(a), G, tab, c, repel=rep).argmax(1)]
            q, a = _ipf_general(X, U, tab, c, cnt, a)
            rep = _repel_lists(U, torch.cat([X, U]), len(X), rad) if REPEL_SPACING else None
            U = U + G[_side_ll(U, padS, cnt.Ns, X, torch.exp(q), G, tab, c, repel=rep).argmax(1)]
    return X, U


# ------------------------------------------------------------- fast path --
# Near/far split of the kernel (particle-mesh style): pairs closer than FAST_R2 are summed exactly
# from neighbour lists; the smooth remainder K(r) S(r) (S: smoothstep from 0 at FAST_R1 to 1 at
# FAST_R2) is computed by FFT convolution on a mesh with bilinear spreading/interpolation, which is
# differentiable in both bead sets. The flat background c is summed analytically.

FAST_R1, FAST_R2 = 100.0, 200.0   # switch radii (internal units)
FAST_MESH = 10.0                  # far-field mesh spacing
FAST_SKIN = 60.0                  # neighbour-list skin; lists are rebuilt every 50 warp steps
REFINE_MESH = 5.0                 # mesh spacing of the refinement normalizer (as GRID)
IPF_WARM_ITERS = 5                # warm-started IPF iterations
COARSE_N = None                   # None: full 41x41 candidate grid per bead (as the exact path); an integer n:
                                  # coarse-to-fine n^2 + n^2 (faster, but cost ~0.006 J@10 on c58_8)
FAST_FIT, FAST_KDE, FAST_C2F = True, True, True   # components of the fast path (for ablation)
REPEL_SPACING = None              # internal-units minimum bead spacing for refinement (set from min_spacing)
REPEL_WEIGHT = 40.0               # overlap penalty: weight * sum (d_min - d)^2 / d_min^2 (nats)
FAST_REFINE = False               # mesh-normalizer refinement costs ~0.006 J@10 on c58_8 (contact-scale
                                  # smoothing of the normalizer); the exact-normalizer refinement is used


def _switch(r):
    torch = _torch()
    t = ((r - FAST_R1) / (FAST_R2 - FAST_R1)).clamp(0.0, 1.0)
    return t * t * (3 - 2 * t)


def _kval(d, tab, c):
    """K(d) (without the background)."""
    torch = _torch()
    return torch.exp(_logm(d, tab, c)) - c


class _Mesh:
    """Regular mesh over a bounding box with an FFT of a radial kernel image (the kernel K, optionally
    restricted to its far part, or any radial function `fn(r)`)."""

    def __init__(self, lo, hi, h, tab, c, far_only, dev, fn=None):
        torch = _torch()
        self.lo, self.h = lo, h
        self.nx = int(np.ceil(float(hi[0] - lo[0]) / h)) + 2
        self.ny = int(np.ceil(float(hi[1] - lo[1]) / h)) + 2
        ix = torch.arange(2 * self.nx, device=dev); iy = torch.arange(2 * self.ny, device=dev)
        dx = torch.minimum(ix, 2 * self.nx - ix).float() * h
        dy = torch.minimum(iy, 2 * self.ny - iy).float() * h
        self.r = torch.sqrt(dx[:, None] ** 2 + dy[None, :] ** 2)
        if fn is not None:
            img = fn(self.r)
        else:
            img = _kval(self.r, tab, c)
            if far_only:
                img = img * _switch(self.r)
        self.khat = torch.fft.rfft2(img)

    def _corners(self, P):
        torch = _torch()
        f = (P - self.lo) / self.h
        fx = f[:, 0].clamp(0, self.nx - 1.001); fy = f[:, 1].clamp(0, self.ny - 1.001)
        x0, y0 = fx.floor().long(), fy.floor().long()
        tx, ty = fx - x0, fy - y0
        return x0, y0, tx, ty

    def spread(self, P, w):
        torch = _torch()
        x0, y0, tx, ty = self._corners(P)
        g = torch.zeros(self.nx * self.ny, device=P.device, dtype=w.dtype)
        for ax, ay, wt in ((0, 0, (1 - tx) * (1 - ty)), (1, 0, tx * (1 - ty)), (0, 1, (1 - tx) * ty), (1, 1, tx * ty)):
            g = g.index_add(0, (x0 + ax) * self.ny + (y0 + ay), w * wt)
        return g.reshape(self.nx, self.ny)

    def convolve(self, g):
        torch = _torch()
        G = torch.zeros(2 * self.nx, 2 * self.ny, device=g.device, dtype=g.dtype)
        G[:self.nx, :self.ny] = g
        return torch.fft.irfft2(torch.fft.rfft2(G) * self.khat, s=G.shape)[:self.nx, :self.ny]

    def sample(self, F, P):
        x0, y0, tx, ty = self._corners(P)
        return ((1 - tx) * (1 - ty) * F[x0, y0] + tx * (1 - ty) * F[x0 + 1, y0]
                + (1 - tx) * ty * F[x0, y0 + 1] + tx * ty * F[x0 + 1, y0 + 1])


def _near_pairs(X, U, shared, radius, chunk=2000):
    """(receiver, sender) index pairs closer than radius (self pairs excluded when shared)."""
    torch = _torch()
    small = len(X) * len(U) > NEAR_INT32_ABOVE      # very large arrays: int32 indices (half the memory, same values)
    rr, oo = [], []
    for i in range(0, len(X), chunk):
        d = torch.cdist(X[i:i + chunk], U)
        b, o = torch.nonzero(d < radius, as_tuple=True)
        b = b + i
        if shared:
            k = b != o
            b, o = b[k], o[k]
        if small:
            b, o = b.int(), o.int()
        rr.append(b); oo.append(o)
    return torch.cat(rr), torch.cat(oo)


NEAR_INT32_ABOVE = 2e9   # receivers x senders above which near-pair indices are int32; every paper dataset is below it


class _Fast:
    """Kernel sums  s_b = sum_o w_o m(|x_b - u_o|)  (and the transpose) via near pairs + mesh."""

    def __init__(self, X, U, tab, c, shared):
        torch = _torch()
        self.tab, self.c, self.shared = tab, c, shared
        self.ct = torch.as_tensor(float(c), dtype=torch.float32, device=X.device)
        P = X if shared else torch.cat([X, U])
        margin = FAST_R2 + 400.0
        lo = P.min(0).values.detach() - margin; hi = P.max(0).values.detach() + margin
        self.mesh = _Mesh(lo, hi, FAST_MESH, tab, c, True, X.device)
        self.rebuild(X, U)

    def rebuild(self, X, U):
        self.nb, self.no = _near_pairs(X.detach(), U.detach(), self.shared, FAST_R2 + FAST_SKIN)

    def near_weight(self, X, U):
        torch = _torch()
        d = torch.sqrt(((X[self.nb] - U[self.no]) ** 2).sum(-1) + 1e-4)
        return _kval(d, self.tab, self.c) * (1 - _switch(d))

    def row_sums(self, X, U, w):
        """s_b = sum_o w_o (K(d_bo) + c), self excluded when shared (differentiable)."""
        torch = _torch()
        near = _near_sum(X, U, self.nb, self.no, w, self.tab, self.ct, len(X))
        if WIDTH is not None:
            fw = _far_w(); far = self.mesh.sample(self.mesh.convolve(self.mesh.spread(U, w * fw)), X) * fw
        else:
            far = self.mesh.sample(self.mesh.convolve(self.mesh.spread(U, w)), X)
        bg = self.c * (w.sum() - (w if self.shared else 0.0))
        return near + far + bg

    def col_sums(self, X, U, v):
        """t_o = sum_b v_b (K(d_bo) + c)."""
        torch = _torch()
        near = _near_sum(U, X, self.no, self.nb, v, self.tab, self.ct, len(U))
        if WIDTH is not None:
            fw = _far_w(); far = self.mesh.sample(self.mesh.convolve(self.mesh.spread(X, v * fw)), U) * fw
        else:
            far = self.mesh.sample(self.mesh.convolve(self.mesh.spread(X, v)), U)
        bg = self.c * (v.sum() - (v if self.shared else 0.0))
        return near + far + bg


def _obs_term_eager(X, U, rows, cols, vals, tab, c):
    """sum over nonzero counts of C * log m(d) (the observed part of the Poisson log-likelihood)."""
    torch = _torch()
    d = torch.sqrt(((X[rows] - U[cols]) ** 2).sum(-1) + 1e-4)
    if WIDTH is not None:
        d = d / _pair_scale(rows, cols)
    f = (d / TAB_STEP).clamp(0, tab.shape[0] - 1.001)
    i = f.floor().long()
    t = f - i
    return (vals * torch.log(tab[i] * (1 - t) + tab[i + 1] * t + c)).sum()


NEAR_CHUNK_ABOVE = 20_000_000   # with WIDTH set: near pairs above which the near sum is chunked + checkpointed (memory)


def _near_sum_eager(X, U, nb, no, w, tab, c, n_out):
    torch = _torch()
    if WIDTH is not None and len(nb) > NEAR_CHUNK_ABOVE:
        from torch.utils.checkpoint import checkpoint
        grad = torch.is_grad_enabled()
        out = torch.zeros(n_out, device=X.device, dtype=w.dtype)
        for j in range(0, len(nb), NNZ_CHUNK):
            args = (X, U, nb[j:j + NNZ_CHUNK], no[j:j + NNZ_CHUNK], w, tab, c, n_out)
            out = out + (checkpoint(_near_sum_eager_1, *args, use_reentrant=False) if grad else _near_sum_eager_1(*args))
        return out
    return _near_sum_eager_1(X, U, nb, no, w, tab, c, n_out)


def _near_sum_eager_1(X, U, nb, no, w, tab, c, n_out):
    """s_b = sum over near pairs of w_o K(d_bo) (1 - S(d_bo)), as _Fast.near_weight + index_add."""
    torch = _torch()
    d = torch.sqrt(((X[nb] - U[no]) ** 2).sum(-1) + 1e-4)
    dk = d / _pair_scale(nb, no) if WIDTH is not None else d   # kernel at width-scaled distance; switch at physical distance
    f = (dk / TAB_STEP).clamp(0, tab.shape[0] - 1.001)
    i = f.floor().long()
    t = f - i
    k = tab[i] * (1 - t) + tab[i + 1] * t                     # K(d) (exp(_logm) - c)
    x = ((d - FAST_R1) / (FAST_R2 - FAST_R1)).clamp(0, 1)
    sw = x * x * (3 - 2 * x)                                   # _switch(d)
    return torch.zeros(n_out, device=X.device, dtype=w.dtype).index_add(0, nb, k * (1 - sw) * w[no])


_obs_term = _maybe_compiled(_obs_term_eager)
_near_sum = _maybe_compiled(_near_sum_eager)


def _ipf_fast(X, U, fk, cnt, a, iters):
    torch = _torch()
    for _ in range(iters):
        q = torch.log(cnt.Nr) - torch.log(fk.row_sums(X, U, torch.exp(a)))
        a = torch.log(cnt.Ns) - torch.log(fk.col_sums(X, U, torch.exp(q)))
        a = a - a[torch.isfinite(a)].mean()
    return q, a


def _ll_and_grad_fast(X0, U0, warp, cnt, q, a, fk, tab, c):
    """-ll / 1e6 gradient into warp.D: expected-count term via near pairs + mesh, observed term sparse."""
    torch = _torch()
    X = X0 + warp(X0)                          # each bead warped once per step; pairs gather from these
    U = X if cnt.shared else U0 + warp(U0)
    loss = (torch.exp(q) * fk.row_sums(X, U, torch.exp(a))).sum()
    for j in range(0, len(cnt.vals), NNZ_CHUNK):
        loss = loss - _obs_term(X, U, cnt.rows[j:j + NNZ_CHUNK], cnt.cols[j:j + NNZ_CHUNK], cnt.vals[j:j + NNZ_CHUNK],
                                tab, fk.ct)
    (loss / 1e6).backward()


def _kfit_terms_eager(ldz, ldn, sn, v, logk, c, q, ac, ea, rows, cols, vals, nb, no):
    """Kernel fit: observed term over nonzero counts and near-pair expected sums (fused when compiled)."""
    torch = _torch()
    obs = (vals * (q[rows] + ac[cols] + torch.log(torch.exp(_log_kernel_seg(ldz, v, logk)) + c))).sum()
    near = torch.zeros(q.shape[0], device=q.device, dtype=q.dtype).index_add(
        0, nb, torch.exp(_log_kernel_seg(ldn, v, logk)) * sn * ea[no])
    return obs, near


_kfit_terms = _kfit_terms_eager   # compiling this costs more than it saves over 300 steps (measured)

WIDTH = None        # prototype (shared beads only): per-bead transfer width factors s_i (torch tensor on the device); pair
                    # kernels are evaluated at d / sqrt(s_i s_j). None (default) = one kernel for all pairs (unchanged).
WIDTH_BETA = 2.0    # far field: K(d/s) ~ s^beta K(d) beyond the near radius (power-law tail)
JOINT_WIDTH = False # prototype: estimate per-bead log-widths jointly with the warp (shared beads); result in WIDTH_LAM
JOINT_LR, JOINT_RIDGE, JOINT_SMOOTH = 0.01, 5.0, 500.0   # Adam rate; ridge and smoothness weights in (tempered) count units
WIDTH_LAM = None    # trainable log-widths (persist across stages; reset to None before a new reconstruction)


def _pair_scale(i, j):
    torch = _torch()
    return torch.sqrt(WIDTH[i] * WIDTH[j])


def _far_w():
    return WIDTH ** (WIDTH_BETA / 2)

KFIT_CHUNK_ABOVE = 50_000_000   # nonzero + near pairs above which the kernel-fit terms are chunked and checkpointed
                                # (same values; bounds memory on large arrays). Every paper dataset is below it.


def _kfit_obs_chunk(ldz, v, logk, c, q, ac, rows, cols, vals):
    torch = _torch()
    return (vals * (q[rows] + ac[cols] + torch.log(torch.exp(_log_kernel_seg(ldz, v, logk)) + c))).sum()


def _kfit_near_chunk(ldn, sn, v, logk, ea, no):
    torch = _torch()
    return torch.exp(_log_kernel_seg(ldn, v, logk)) * sn * ea[no]


def _kfit_terms_chunked(ldz, ldn, sn, v, logk, c, q, ac, ea, rows, cols, vals, nb, no):
    """As _kfit_terms_eager, in NNZ_CHUNK pieces whose intermediates are recomputed in the backward pass."""
    torch = _torch()
    from torch.utils.checkpoint import checkpoint
    obs = 0.0
    for j in range(0, len(vals), NNZ_CHUNK):
        s = slice(j, j + NNZ_CHUNK)
        obs = obs + checkpoint(_kfit_obs_chunk, ldz[s], v, logk, c, q, ac, rows[s], cols[s], vals[s], use_reentrant=False)
    near = torch.zeros(q.shape[0], device=q.device, dtype=q.dtype)
    for j in range(0, len(nb), NNZ_CHUNK):
        s = slice(j, j + NNZ_CHUNK)
        near = near.index_add(0, nb[s], checkpoint(_kfit_near_chunk, ldn[s], sn[s], v, logk, ea, no[s], use_reentrant=False))
    return obs, near


def fit_kernel_fast(X, U, cnt, dev, iters=300, lr=0.05, decay=False):
    """Poisson ML fit of the kernel, c, a, q with positions fixed, over ALL receivers and senders:
    observed term over nonzero pairs, expected term via near pairs + far-field mesh (the kernel image is
    re-transformed every step, differentiably)."""
    torch = _torch()
    logk = torch.tensor(np.log(KNOTS), dtype=torch.float32, device=dev)
    shared = cnt.shared
    nb, no = _near_pairs(X, U, shared, FAST_R2)
    dn = torch.sqrt(((X[nb] - U[no]) ** 2).sum(-1) + 1e-4)
    ldn = torch.log((dn / _pair_scale(nb, no) if WIDTH is not None else dn).clamp(float(KNOTS[0]), float(KNOTS[-1]))); sn = 1 - _switch(dn)
    dz = torch.sqrt(((X[cnt.rows] - U[cnt.cols]) ** 2).sum(-1) + 1e-4)
    ldz = torch.log((dz / _pair_scale(cnt.rows, cnt.cols) if WIDTH is not None else dz).clamp(float(KNOTS[0]), float(KNOTS[-1])))
    P = X if shared else torch.cat([X, U])
    mesh = _Mesh(P.min(0).values - FAST_R2 - 400, P.max(0).values + FAST_R2 + 400, FAST_MESH, None, 0.0, False,
                 dev, fn=lambda r: torch.zeros_like(r))
    lr_mesh = torch.log(mesh.r.clamp(float(KNOTS[0]), float(KNOTS[-1]))); s_mesh = _switch(mesh.r)
    a = torch.log(cnt.Ns + 1.0); a = (a - a.mean()).clone().requires_grad_(True)
    q = (torch.log(cnt.Nr + 1.0) - np.log(cnt.n_o)).clone().requires_grad_(True)
    v0 = -4.0 * np.log1p((KNOTS / 26.0) ** 2); v0 -= v0[0]
    dec = np.clip(-np.diff(v0), 1e-2, None)
    w = torch.tensor(np.log(np.expm1(dec)), dtype=torch.float32, device=dev, requires_grad=True)
    logc = torch.tensor(np.log(1e-4), dtype=torch.float32, device=dev, requires_grad=True)
    opt = torch.optim.Adam([a, q, w, logc], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters) if decay else None
    norm = float(cnt.n_r) * float(cnt.n_o)
    terms = _kfit_terms_chunked if len(cnt.vals) + len(nb) > KFIT_CHUNK_ABOVE else _kfit_terms
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        v = _knot_values(w); c = torch.exp(logc); ac = a - a.mean(); ea = torch.exp(ac)
        obs, near = terms(ldz, ldn, sn, v, logk, c, q, ac, ea, cnt.rows, cnt.cols, cnt.vals, nb, no)
        mesh.khat = torch.fft.rfft2(torch.exp(_log_kernel_seg(lr_mesh, v, logk)) * s_mesh)
        far = (mesh.sample(mesh.convolve(mesh.spread(U, ea * _far_w())), X) * _far_w() if WIDTH is not None
               else mesh.sample(mesh.convolve(mesh.spread(U, ea)), X))
        bg = c * (ea.sum() - (ea if shared else 0.0))
        expct = (torch.exp(q) * (near + far + bg)).sum()
        (-(obs - expct) / norm).backward()
        opt.step()
        if sched is not None:
            sched.step()
    with torch.no_grad():
        return _knot_values(w).detach(), float(torch.exp(logc))


def fit_kernel_nb(X, U, cnt, dev, iters=3000, lr=0.05):
    """Kernel, background, a, q and a shared negative-binomial size r fit to convergence (Adam, rate decayed
    to zero) under C ~ NB(mean mu, size r), Var = mu + mu^2 / r, with positions fixed.
    Zero-count pairs contribute -r log(1 + mu/r): computed exactly for pairs within FAST_R2 (near list), and
    as its first-order form -mu for the remaining far pairs (mu there is ~1e-3 or less), whose total comes
    from the near/far kernel sums. Returns (knot values, background, r)."""
    torch = _torch()
    logk = torch.tensor(np.log(KNOTS), dtype=torch.float32, device=dev)
    shared = cnt.shared
    nb, no = _near_pairs(X, U, shared, FAST_R2)
    dn = torch.sqrt(((X[nb] - U[no]) ** 2).sum(-1) + 1e-4)
    ldn = torch.log((dn / _pair_scale(nb, no) if WIDTH is not None else dn).clamp(float(KNOTS[0]), float(KNOTS[-1]))); sn = 1 - _switch(dn)
    dz = torch.sqrt(((X[cnt.rows] - U[cnt.cols]) ** 2).sum(-1) + 1e-4)
    ldz = torch.log((dz / _pair_scale(cnt.rows, cnt.cols) if WIDTH is not None else dz).clamp(float(KNOTS[0]), float(KNOTS[-1])))
    P = X if shared else torch.cat([X, U])
    mesh = _Mesh(P.min(0).values - FAST_R2 - 400, P.max(0).values + FAST_R2 + 400, FAST_MESH, None, 0.0, False,
                 dev, fn=lambda r: torch.zeros_like(r))
    lr_mesh = torch.log(mesh.r.clamp(float(KNOTS[0]), float(KNOTS[-1]))); s_mesh = _switch(mesh.r)
    a = torch.log(cnt.Ns + 1.0); a = (a - a.mean()).clone().requires_grad_(True)
    q = (torch.log(cnt.Nr + 1.0) - np.log(cnt.n_o)).clone().requires_grad_(True)
    v0 = -4.0 * np.log1p((KNOTS / 26.0) ** 2); v0 -= v0[0]
    dec = np.clip(-np.diff(v0), 1e-2, None)
    w = torch.tensor(np.log(np.expm1(dec)), dtype=torch.float32, device=dev, requires_grad=True)
    logc = torch.tensor(np.log(1e-4), dtype=torch.float32, device=dev, requires_grad=True)
    logr = torch.tensor(np.log(1.0), dtype=torch.float32, device=dev, requires_grad=True)
    opt = torch.optim.Adam([a, q, w, logc, logr], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    norm = float(cnt.n_r) * float(cnt.n_o)
    Cz = cnt.vals
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        v = _knot_values(w); c = torch.exp(logc); r = torch.exp(logr); ac = a - a.mean(); ea = torch.exp(ac)
        # nonzero pairs: full NB log-likelihood
        muz = torch.exp(q[cnt.rows] + ac[cnt.cols]) * (torch.exp(_log_kernel_seg(ldz, v, logk)) + c)
        ll_nz = (Cz * torch.log(muz) - (r + Cz) * torch.log(r + muz) + r * torch.log(r)
                 + torch.lgamma(Cz + r) - torch.lgamma(r)).sum()
        zero_nz = (-r * torch.log1p(muz / r)).sum()            # their zero-count term, to subtract
        # all pairs' zero-count term: exact over near pairs, first-order (-mu) over the rest
        Kn = torch.exp(_log_kernel_seg(ldn, v, logk))
        mun = torch.exp(q[nb] + ac[no]) * (Kn + c)
        zero_near = (-r * torch.log1p(mun / r)).sum()
        mesh.khat = torch.fft.rfft2(torch.exp(_log_kernel_seg(lr_mesh, v, logk)) * s_mesh)
        near_lin = torch.zeros(cnt.n_r, device=dev).index_add(0, nb, Kn * sn * ea[no])
        far = (mesh.sample(mesh.convolve(mesh.spread(U, ea * _far_w())), X) * _far_w() if WIDTH is not None
               else mesh.sample(mesh.convolve(mesh.spread(U, ea)), X))
        bg = c * (ea.sum() - (ea if shared else 0.0))
        mu_all = (torch.exp(q) * (near_lin + far + bg)).sum()
        mu_near = mun.sum()
        ll = ll_nz + (zero_near - (mu_all - mu_near)) - zero_nz
        (-ll / norm).backward()
        opt.step(); sched.step()
    with torch.no_grad():
        return _knot_values(w).detach(), float(torch.exp(logc)), float(torch.exp(logr))


ISO_SCALES = (300.0, 150.0)   # contact-isotropy patches (Gaussian sd, internal units)
ISO_TOPK, ISO_MINC, ISO_MAXLEN = 3, 3, 40.0
ISO_MAXLEN_MED = 3.0   # with LENGTH_REF_UM set: contacts kept if shorter than this multiple of the median contact length
                       # (tracks the bead spacing; ISO_MAXLEN = 40 internal units is ~3x it on the paper's datasets)
ISO_SCHEDULE = ((400, 200, 4.0), (200, 200, 3.0))   # second-stage warp levels
ISO = None                  # set per run: dict(ri, oi, weight)
ISO_WEIGHT = 3e6            # default isotropy weight (chosen on replicate c58_8)
ISO_CENTER = False          # True: penalize local anisotropy relative to the array-wide mean, ignoring a
                            # uniform directional bias of transfer (e.g. along lattice rows); set per run


def contact_pairs(Cf, X, U, shared=False, a=None, return_counts=False):
    """Contacts from the counts alone: each receiver's ISO_TOPK partners with the highest counts relative to the
    partner's fitted abundance (C / e^a, so a smooth spatial trend in abundance does not favour one direction),
    with at least ISO_MINC counts, kept if within ISO_MAXLEN in the current layout (X, U: internal units)."""
    torch = _torch()
    w = None if a is None else np.exp(np.nan_to_num(a.detach().cpu().numpy(), neginf=-50.0))
    ri, oi, cv = [], [], []
    for i in range(Cf.shape[0]):
        lo_, hi_ = Cf.indptr[i], Cf.indptr[i + 1]
        if hi_ == lo_:
            continue
        d = Cf.data[lo_:hi_]; ix = Cf.indices[lo_:hi_]
        key = d if w is None else d / np.maximum(w[ix], 1e-12)
        top = np.argsort(key)[::-1][:ISO_TOPK]
        top = top[d[top] >= ISO_MINC]
        ri += [i] * len(top); oi += ix[top].tolist(); cv += d[top].tolist()
    ri = torch.tensor(ri, device=X.device); oi = torch.tensor(oi, device=X.device)
    cv = torch.tensor(cv, dtype=torch.float32, device=X.device)
    Uo = X if shared else U
    with torch.no_grad():
        ln = (X[ri] - Uo[oi]).norm(dim=1)
        cap = ISO_MAXLEN if LENGTH_REF_UM is None else ISO_MAXLEN_MED * float(ln[ln > 1e-3].median())
        keep = (ln > 1e-3) & (ln < cap)
    return (ri[keep], oi[keep], cv[keep]) if return_counts else (ri[keep], oi[keep])


TRANSFER_WARN = (0.0, 4.0)     # warn if the transfer-anisotropy z-score exceeds 4 (amplitude not used): on the
                               # benchmarks z was 5.0-7.1 on SCOPE's simulator (directional transfer) and <= 3.2
                               # on all Hu replicates and calibrated/idealized simulations


def transfer_anisotropy(X, Uo, q, a, tab, c, ri, oi, cv):
    """Imaging-free check of the isotropy prior's assumption that transfer favours no direction. For contacts,
    r = observed / expected count under the fitted (isotropic) model; directional transfer makes r depend on the
    contact's axis, whereas a shear of the layout does not change the counts. Returns the amplitude of the
    covariance between r (normalized) and the axial direction (cos 2t, sin 2t), its z-score and its axis (degrees)."""
    torch = _torch()
    with torch.no_grad():
        v = X[ri] - Uo[oi]
        d = v.norm(dim=1).clamp(min=1e-6); u = v / d[:, None]
        comp = torch.stack([u[:, 0] ** 2 - u[:, 1] ** 2, 2 * u[:, 0] * u[:, 1]], 1)
        mu = torch.exp(q[ri] + a[oi]) * torch.exp(_logm(d, tab, c))
        r = cv / mu.clamp(min=1e-12)
        r = r / r.mean()
        dev = (r - 1)[:, None] * comp
        m = dev.mean(0); se = dev.std(0) / np.sqrt(len(r))
        amp = float(m.norm()); z = float((m / se.clamp(min=1e-12)).norm() / np.sqrt(2))
        axis = float(np.degrees(0.5 * np.arctan2(float(m[1]), float(m[0]))) % 180)
    return {"amplitude": amp, "z": z, "axis_deg": axis, "contacts": int(len(r))}



def isotropy_penalty(X, Uo, ri, oi, lo, hi, weights=None):
    """Mean squared anisotropy of contact directions in Gaussian patches (ISO_SCALES): a random packing has
    isotropic contact directions, a sheared layout does not. Per contact the traceless direction tensor
    (ux^2 - uy^2, 2 ux uy); patch anisotropy = weighted mean; penalty = contact-weighted mean of |anisotropy|^2.
    weights: patch weights from _iso_weights (recomputed from the current contact midpoints when None)."""
    torch = _torch()
    v = X[ri] - Uo[oi]
    u = v / v.norm(dim=1, keepdim=True).clamp(min=1e-6)
    comp = torch.stack([u[:, 0] ** 2 - u[:, 1] ** 2, 2 * u[:, 0] * u[:, 1]], 1)
    if weights is None:
        weights = _iso_weights(0.5 * (X[ri] + Uo[oi]), lo, hi, X.device)
    return _patch_anisotropy_w(comp, weights)


def isotropy_noise_floor(X, Uo, ri, oi, lo, hi, draws=5, seed=0):
    """Penalty expected from finite sampling alone: the same patch average with every contact's direction
    replaced by a uniformly random one (same contact locations); mean over draws."""
    torch = _torch()
    g = torch.Generator(device="cpu").manual_seed(seed)
    mid = 0.5 * (X[ri] + Uo[oi])
    vals, wts = [], _iso_weights(mid.detach(), lo, hi, X.device)
    for _ in range(draws):
        t = torch.rand(len(ri), generator=g).to(X.device) * np.pi
        comp = torch.stack([torch.cos(2 * t), torch.sin(2 * t)], 1)
        vals.append(float(_patch_anisotropy_w(comp, wts)))
    return float(np.mean(vals))


def _iso_weights(mid, lo, hi, dev):
    """Gaussian weights of contact midpoints in the isotropy patches, per scale (no gradient). The patches lie on a
    regular grid and the Gaussian factorizes, w[(ix, iy), j] = Wx[ix, j] * Wy[iy, j], so only the two factors are
    stored: per scale (Wx, Wy, ok, W) with ok the flattened patches whose total weight W exceeds 20 (W[ok] kept).
    Same values as the dense patch x contact matrix (_iso_weights_dense) in (nx + ny) x n_contact memory."""
    torch = _torch()
    out = []
    with torch.no_grad():
        for L in ISO_SCALES:
            gx = torch.arange(float(lo[0]), float(hi[0]) + L / 2, L / 2, device=dev)
            gy = torch.arange(float(lo[1]), float(hi[1]) + L / 2, L / 2, device=dev)
            Wx = torch.exp(-(gx[:, None] - mid[None, :, 0]) ** 2 / (2 * L ** 2))     # (nx, n_contact)
            Wy = torch.exp(-(gy[:, None] - mid[None, :, 1]) ** 2 / (2 * L ** 2))     # (ny, n_contact)
            W = (Wx @ Wy.T).reshape(-1)                                            # patch order as meshgrid "ij"
            ok = W > 20
            out.append((Wx, Wy, ok, W[ok]))
    return out


def _iso_weights_dense(mid, lo, hi, dev):
    """The dense patch x contact weights per scale, (w[ok], W[ok]); kept for checking _iso_weights."""
    torch = _torch()
    out = []
    with torch.no_grad():
        for L in ISO_SCALES:
            gx = torch.arange(float(lo[0]), float(hi[0]) + L / 2, L / 2, device=dev)
            gy = torch.arange(float(lo[1]), float(hi[1]) + L / 2, L / 2, device=dev)
            cen = torch.stack(torch.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
            w = torch.exp(-torch.cdist(cen, mid) ** 2 / (2 * L ** 2))     # (n_patch, n_contact)
            W = w.sum(1)
            ok = W > 20
            out.append((w[ok], W[ok]))
    return out


def _patch_means(comp, wt):
    """Weighted mean direction components per kept patch, for separable (Wx, Wy, ok, W) or dense (w, W) weights."""
    torch = _torch()
    if len(wt) == 4:
        Wx, Wy, ok, W = wt
        A = torch.stack([((Wx * comp[:, k]) @ Wy.T).reshape(-1)[ok] for k in range(comp.shape[1])], 1)
    else:
        w, W = wt
        A = w @ comp
    return A / W[:, None], W


def _patch_anisotropy_w(comp, weights):
    if ISO_CENTER:
        comp = comp - comp.mean(0, keepdim=True)
    pen = 0.0
    for wt in weights:
        A, W = _patch_means(comp, wt)
        pen = pen + ((A ** 2).sum(1) * W).sum() / W.sum()
    return pen / len(weights)


def _patch_anisotropy(mid, comp, lo, hi, dev):
    return _patch_anisotropy_w(comp, _iso_weights(mid.detach(), lo, hi, dev))


ISO_REFRESH = 50   # warp steps between recomputations of the isotropy patch weights (1: every step)


ANNEAL = 0.0     # warp levels use the kernel blurred by a Gaussian of sd ANNEAL * grid spacing (0: none)
KERNEL_V = None  # knot values of the fitted kernel (needed to build blurred tables)


WARP_COMPILE = True    # compile the whole warp-step objective (CUDA, distinct senders/receivers, disc footprint):
                       # ~7% faster per fresh run and ~22% for later runs in a session once PyTorch's on-disk
                       # compile cache is warm; the first run on a machine pays ~15 s of compilation


def _mesh_apply(P, w, lo, h, nx, ny, khat, Q):
    """Spread weights w at points P onto a mesh (bilinear), convolve with a kernel image (FFT) and sample at Q;
    functional form of _Mesh.spread / convolve / sample."""
    torch = _torch()
    f = (P - lo) / h
    fx = f[:, 0].clamp(0, nx - 1.001); fy = f[:, 1].clamp(0, ny - 1.001)
    x0, y0 = fx.floor().long(), fy.floor().long()
    tx, ty = fx - x0, fy - y0
    g = torch.zeros(nx * ny, device=P.device, dtype=w.dtype)
    for ax, ay, wt in ((0, 0, (1 - tx) * (1 - ty)), (1, 0, tx * (1 - ty)), (0, 1, (1 - tx) * ty), (1, 1, tx * ty)):
        g = g.index_add(0, (x0 + ax) * ny + (y0 + ay), w * wt)
    G = torch.zeros(2 * nx, 2 * ny, device=P.device, dtype=w.dtype)
    G[:nx, :ny] = g.reshape(nx, ny)
    F = torch.fft.irfft2(torch.fft.rfft2(G) * khat, s=(2 * nx, 2 * ny))[:nx, :ny]
    f = (Q - lo) / h
    fx = f[:, 0].clamp(0, nx - 1.001); fy = f[:, 1].clamp(0, ny - 1.001)
    x0, y0 = fx.floor().long(), fy.floor().long()
    tx, ty = fx - x0, fy - y0
    return ((1 - tx) * (1 - ty) * F[x0, y0] + tx * (1 - ty) * F[x0 + 1, y0]
            + (1 - tx) * ty * F[x0, y0 + 1] + tx * ty * F[x0 + 1, y0 + 1])


def _warp_objective(D, X0, U0, lo, hi, q, a, near, far, rows, cols, vals, tab, ct, shared, phi, lam, dens, iso):
    """The warp-step objective (minimized): -loglik / phi + lam * density penalty (+ isotropy penalty), as one
    function of the displacement-field control values D. near = (nb, no); far = (lo, h, nx, ny, khat) of the
    far-field mesh; dens = [(G, lo, h, nx, ny, khat)] per bandwidth; iso = (ri, oi, weight, [(w, W)]) or None."""
    torch = _torch()

    def warp(P):
        g = 2 * (P - lo) / (hi - lo) - 1
        return torch.nn.functional.grid_sample(D, g[None, None], mode="bicubic", align_corners=True)[0, :, 0, :].T

    X = X0 + warp(X0)
    U = X if shared else U0 + warp(U0)
    ea = torch.exp(a)
    nb, no = near
    s = (_near_sum_eager(X, U, nb, no, ea, tab, ct, X.shape[0]) + _mesh_apply(U, ea, *far, X)
         + ct * (ea.sum() - (ea if shared else 0.0)))
    loss = (torch.exp(q) * s).sum()
    for j in range(0, vals.shape[0], NNZ_CHUNK):
        loss = loss - _obs_term_eager(X, U, rows[j:j + NNZ_CHUNK], cols[j:j + NNZ_CHUNK], vals[j:j + NNZ_CHUNK], tab, ct)
    total = loss / phi
    P = X if shared else torch.cat([X, U])
    pen = 0.0
    ones = torch.ones(P.shape[0], device=P.device, dtype=P.dtype)
    for G, mlo, h, nx, ny, khat in dens:
        lr = torch.log(_mesh_apply(P, ones, mlo, h, nx, ny, khat, G))
        pen = pen + ((lr - lr.mean()) ** 2).sum()
    total = total + lam * pen
    if iso is not None:
        ri, oi, weight, wts = iso
        v = X[ri] - U[oi]
        u = v / v.norm(dim=1, keepdim=True).clamp(min=1e-6)
        comp = torch.stack([u[:, 0] ** 2 - u[:, 1] ** 2, 2 * u[:, 0] * u[:, 1]], 1)
        if ISO_CENTER:
            comp = comp - comp.mean(0, keepdim=True)
        pi = 0.0
        for wt in wts:
            A, W = _patch_means(comp, wt)
            pi = pi + ((A ** 2).sum(1) * W).sum() / W.sum()
        total = total + weight * pi / len(wts)
    return total


_WARP_OBJ = {}


def _warp_objective_compiled():
    torch = _torch()
    if "f" not in _WARP_OBJ:
        try:                                     # cache the backward graph on disk too (PyTorch >= 2.5)
            import torch._functorch.config as fc
            fc.enable_autograd_cache = True
        except Exception:
            pass
        _WARP_OBJ["f"] = torch.compile(_warp_objective, dynamic=True)
    return _WARP_OBJ["f"]


def warp_stage_fast(X0, U0, cnt, tab, c, phi=PHI, lam=LAM, footprint="disc", verbose=False, t0=None,
                    schedule=None, iso=None):
    torch = _torch()
    dev = X0.device
    allP = X0 if cnt.shared else torch.cat([X0, U0])
    lo = allP.min(0).values - 200; hi = allP.max(0).values + 200
    fk = _Fast(X0, U0, tab, c, cnt.shared)
    a = torch.zeros(cnt.n_o, device=dev)
    with torch.no_grad():
        q, a = _ipf_fast(X0, U0, fk, cnt, a, 30)
    warp = None
    tab0 = tab
    global WIDTH, WIDTH_LAM
    joint = JOINT_WIDTH and cnt.shared
    if joint:
        if WIDTH_LAM is None or WIDTH_LAM.shape[0] != len(X0):
            WIDTH_LAM = torch.zeros(len(X0), device=dev, requires_grad=True)
        WIDTH = torch.exp(WIDTH_LAM).detach()
        with torch.no_grad():                    # contacts for the smoothness penalty: pairs within 3x the median nearest-neighbour distance
            dd = torch.sqrt(((X0[fk.nb] - X0[fk.no]) ** 2).sum(-1))
            nn = torch.full((len(X0),), float("inf"), device=dev).scatter_reduce(0, fk.nb, dd, reduce="amin")
            ck = dd < 3 * nn[torch.isfinite(nn)].median()
            cb, co = fk.nb[ck], fk.no[ck]
            cdeg = torch.bincount(cb, minlength=len(X0)).clamp(min=1).float()
        lopt = torch.optim.Adam([WIDTH_LAM], lr=JOINT_LR)
    for spacing, steps, lr in (schedule or WARP_SCHEDULE):
        if ANNEAL > 0:
            tab = _kernel_table(KERNEL_V, dev, ANNEAL * spacing)
            with torch.no_grad():
                Xc = X0 + warp(X0) if warp is not None else X0
                Uc = Xc if cnt.shared else (U0 + warp(U0) if warp is not None else U0)
                fk = _Fast(Xc, Uc, tab, c, cnt.shared)
                q, a = _ipf_fast(Xc, Uc, fk, cnt, a, IPF_WARM_ITERS)
        warp = _Warp(lo, hi, spacing, dev, init_field=warp)
        opt = torch.optim.Adam([warp.D], lr=lr)
        compiled = (WARP_COMPILE and dev.type == "cuda" and not cnt.shared and isinstance(footprint, str)
                    and footprint == "disc" and KDE_FFT and not EDGE_CORRECT and ANNEAL <= 0)
        for it in range(steps):
            if compiled:
                opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    Xn = X0 + warp(X0); Un = Xn if cnt.shared else U0 + warp(U0)
                    if it % DENSITY_REFRESH == 0:
                        dgeom = _disc_geometry(Xn if cnt.shared else torch.cat([Xn, Un]))
                        dens = [(G, M.lo, M.h, M.nx, M.ny, M.khat) for G, M in dgeom]
                    if iso is not None and it % ISO_REFRESH == 0:
                        iso_w = _iso_weights(0.5 * (Xn[iso["ri"]] + Un[iso["oi"]]), lo, hi, dev)
                mh = fk.mesh
                total = _warp_objective_compiled()(
                    warp.D, X0, U0, lo, hi, q, a, (fk.nb, fk.no), (mh.lo, mh.h, mh.nx, mh.ny, mh.khat),
                    cnt.rows, cnt.cols, cnt.vals, tab, fk.ct, cnt.shared, float(phi), float(lam), dens,
                    None if iso is None else (iso["ri"], iso["oi"], float(iso["weight"]), iso_w))
                total.backward()
                opt.step()
                if (it + 1) % 50 == 0:
                    with torch.no_grad():
                        X = X0 + warp(X0); U = X if cnt.shared else U0 + warp(U0)
                        fk = _Fast(X, U, tab, c, cnt.shared)
                        q, a = _ipf_fast(X, U, fk, cnt, a, IPF_WARM_ITERS)
                continue
            opt.zero_grad(set_to_none=True)
            if joint:
                lopt.zero_grad(set_to_none=True)
                WIDTH = torch.exp(WIDTH_LAM)             # widths enter this step's likelihood graph
            _ll_and_grad_fast(X0, U0, warp, cnt, q, a, fk, tab, c)
            if warp.D.grad is not None:
                warp.D.grad.mul_(1e6 / phi)
            if joint:
                WIDTH_LAM.grad.mul_(1e6 / phi)            # same temper as the warp
                lap = ((WIDTH_LAM[cb] - WIDTH_LAM[co]) ** 2 / cdeg[cb]).sum()
                ((JOINT_RIDGE * (WIDTH_LAM ** 2).sum() + JOINT_SMOOTH * lap) / phi).backward()
                lopt.step()
                with torch.no_grad():
                    WIDTH_LAM -= WIDTH_LAM.mean()          # only relative widths are identifiable
                WIDTH = torch.exp(WIDTH_LAM).detach()
            P = X0 + warp(X0) if cnt.shared else torch.cat([X0 + warp(X0), U0 + warp(U0)])
            if not isinstance(footprint, str) and it % 50 == 0:
                posed = (_pose_polygon(P, footprint),       # footprint pose refreshed every 50 steps
                         _pose_polygon(P, footprint, frac=EDGE_SUPPORT) if EDGE_CORRECT else None)
            if isinstance(footprint, str) and footprint == "disc" and KDE_FFT and not EDGE_CORRECT:
                if it % DENSITY_REFRESH == 0:
                    dgeom = _disc_geometry(P.detach())
                pen, _ = _density_penalty_cached(P, dgeom)
            else:
                pen, _ = density_penalty(P, footprint, None if isinstance(footprint, str) else posed)
            if iso is not None:
                Xc = X0 + warp(X0)
                Uc = Xc if cnt.shared else U0 + warp(U0)
                if it % ISO_REFRESH == 0:
                    iso_w = _iso_weights((0.5 * (Xc[iso["ri"]] + Uc[iso["oi"]])).detach(), lo, hi, dev)
                pen_iso = isotropy_penalty(Xc, Uc, iso["ri"], iso["oi"], lo, hi, weights=iso_w)
                (lam * pen + iso["weight"] * pen_iso).backward()
            else:
                (lam * pen).backward()
            opt.step()
            if (it + 1) % 50 == 0:
                with torch.no_grad():
                    X = X0 + warp(X0); U = X if cnt.shared else U0 + warp(U0)
                    fk = _Fast(X, U, tab, c, cnt.shared)
                    q, a = _ipf_fast(X, U, fk, cnt, a, IPF_WARM_ITERS)
        if verbose:
            msg = ""
            if iso is not None:
                with torch.no_grad():
                    Xc = X0 + warp(X0)
                    msg = f", isotropy penalty {float(isotropy_penalty(Xc, Xc if cnt.shared else U0 + warp(U0), iso['ri'], iso['oi'], lo, hi)):.2e}"
            print(f"  warp grid {spacing}: done ({time.time() - t0:.0f}s){msg}", flush=True)
    with torch.no_grad():
        X = X0 + warp(X0)
        U = X if cnt.shared else U0 + warp(U0)
        if ANNEAL > 0:      # back to the exact kernel for refinement
            fk = _Fast(X, U, tab0, c, cnt.shared)
            q, a = _ipf_fast(X, U, fk, cnt, a, IPF_WARM_ITERS)
        return X, U, q, a


def _side_ll_fast(Center, Origin, pad, N, Other, w_other, G, tab, c, Zmesh, ZF, self_w=None):
    """Capture-profiled log-likelihood at candidates Center + G for every point (other side fixed).
    The normalizer Z comes from the refinement mesh (full kernel); self_w removes the point's own
    contribution, located at Origin (shared beads)."""
    torch = _torch()
    idx, val, nnz, order = pad
    dev = Center.device
    nc = len(G)
    out = torch.empty(len(Center), nc, device=dev)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and (j - i + 1) * nc * max(int(nnz[order[j]]), 1) <= BUDGET:
            j += 1
        j = max(j, i + 1)
        rows_c = torch.tensor(order[i:j]); rows = rows_c.to(dev)
        kmax = max(int(nnz[rows_c].max()), 1)
        cand = Center[rows][:, None, :] + G[None]
        nb = idx[rows_c, :kmax].to(dev); vv = val[rows_c, :kmax].to(dev)
        obs = (_logm(torch.cdist(cand, Other[nb]), tab, c) * vv[:, None, :]).sum(-1)
        Z = Zmesh.sample(ZF, cand.reshape(-1, 2)).reshape(len(rows), nc) + c * w_other.sum()
        if self_w is not None:
            dself = (cand - Origin[rows][:, None, :]).norm(dim=-1)
            Z = Z - self_w[rows, None] * torch.exp(_logm(dself, tab, c))
        out[rows] = obs - N[rows, None] * torch.log(Z.clamp(min=1e-30))
        i = j
    return out


def refine_fast(X, U, Csr, cnt, tab, c, q, a):
    """Two rounds of per-bead refinement, each coarse (COARSE_N^2 over +-W) then fine (COARSE_N^2
    over +-W/(COARSE_N-1) around the coarse optimum), with mesh normalizers."""
    torch = _torch()
    dev = X.device
    padR = _padded_sparse(Csr, dev)
    padS = _padded_sparse(Csr.T.tocsr(), dev)

    def grid(W, n=None):
        h = torch.linspace(-W, W, n or COARSE_N, device=dev)
        return torch.stack(torch.meshgrid(h, h, indexing="ij"), -1).reshape(-1, 2)

    def zmesh(P, w):
        lo = P.min(0).values - 100; hi = P.max(0).values + 100
        M = _Mesh(lo, hi, REFINE_MESH, tab, c, False, dev)
        return M, M.convolve(M.spread(P, w))

    if COARSE_N is None:                                    # full-grid search with mesh normalizers
        for W in REFINE_WINDOWS:
            G = grid(W, 41)
            fk = _Fast(X, U, tab, c, cnt.shared)
            q, a = _ipf_fast(X, U, fk, cnt, a, IPF_WARM_ITERS)
            ea, eq = torch.exp(a), torch.exp(q)
            if cnt.shared:
                ZmR, ZFR = zmesh(X, ea); ZmC, ZFC = zmesh(X, eq)
                ll = (_side_ll_fast(X, X, padR, cnt.Nr, X, ea, G, tab, c, ZmR, ZFR, ea)
                      + _side_ll_fast(X, X, padS, cnt.Ns, X, eq, G, tab, c, ZmC, ZFC, eq))
                X = X + G[ll.argmax(1)]; U = X
            else:
                Zm, ZF = zmesh(U, ea)
                X = X + G[_side_ll_fast(X, X, padR, cnt.Nr, U, ea, G, tab, c, Zm, ZF).argmax(1)]
                fk = _Fast(X, U, tab, c, False)
                q, a = _ipf_fast(X, U, fk, cnt, a, IPF_WARM_ITERS)
                eq = torch.exp(q)
                Zm, ZF = zmesh(X, eq)
                U = U + G[_side_ll_fast(U, U, padS, cnt.Ns, X, eq, G, tab, c, Zm, ZF).argmax(1)]
        return X, U


    def best(Pos, pad, N, Other, w, W, self_w):
        Zm, ZF = zmesh(Other, w)
        Gc = grid(W)
        off = Gc[_side_ll_fast(Pos, Pos, pad, N, Other, w, Gc, tab, c, Zm, ZF, self_w).argmax(1)]
        Gf = grid(W / (COARSE_N - 1))
        return off, Gf, Zm, ZF

    for W in REFINE_WINDOWS:
        fk = _Fast(X, U, tab, c, cnt.shared)
        q, a = _ipf_fast(X, U, fk, cnt, a, IPF_WARM_ITERS)
        if cnt.shared:
            ea, eq = torch.exp(a), torch.exp(q)
            ZmR, ZFR = zmesh(X, ea); ZmC, ZFC = zmesh(X, eq)
            Gc = grid(W)
            llc = (_side_ll_fast(X, X, padR, cnt.Nr, X, ea, Gc, tab, c, ZmR, ZFR, ea)
                   + _side_ll_fast(X, X, padS, cnt.Ns, X, eq, Gc, tab, c, ZmC, ZFC, eq))
            Cn = X + Gc[llc.argmax(1)]
            Gf = grid(W / (COARSE_N - 1))
            llf = (_side_ll_fast(Cn, X, padR, cnt.Nr, X, ea, Gf, tab, c, ZmR, ZFR, ea)
                   + _side_ll_fast(Cn, X, padS, cnt.Ns, X, eq, Gf, tab, c, ZmC, ZFC, eq))
            X = Cn + Gf[llf.argmax(1)]
            U = X
        else:
            ea = torch.exp(a)
            off, Gf, Zm, ZF = best(X, padR, cnt.Nr, U, ea, W, None)
            Cn = X + off
            X = Cn + Gf[_side_ll_fast(Cn, X, padR, cnt.Nr, U, ea, Gf, tab, c, Zm, ZF).argmax(1)]
            fk = _Fast(X, U, tab, c, False)
            q, a = _ipf_fast(X, U, fk, cnt, a, IPF_WARM_ITERS)
            eq = torch.exp(q)
            off, Gf, Zm, ZF = best(U, padS, cnt.Ns, X, eq, W, None)
            Cn = U + off
            U = Cn + Gf[_side_ll_fast(Cn, U, padS, cnt.Ns, X, eq, Gf, tab, c, Zm, ZF).argmax(1)]
    return X, U


# ------------------------------------------------------------ public --

def rescale_to_diameter(P, diameter):
    """Centre P and scale it so its largest pairwise distance equals `diameter`."""
    from scipy.spatial import ConvexHull
    from scipy.spatial.distance import pdist
    P = np.asarray(P, np.float64)
    P = P - P.mean(0)
    hull = P[ConvexHull(P).vertices]
    return (P * diameter / pdist(hull).max()).astype(np.float32)


def senders_from_receivers(C, X):
    """Each sender at the count-weighted coordinate-wise median of its receivers."""
    U = np.full((C.shape[1], 2), np.nan, np.float32)
    for o in range(C.shape[1]):
        nz = np.nonzero(C[:, o])[0]
        if len(nz) == 0:
            continue
        for j in range(2):
            srt = nz[np.argsort(X[nz, j])]
            cw = np.cumsum(C[srt, o])
            U[o, j] = X[srt[np.searchsorted(cw, cw[-1] / 2)], j]
    return U


def _reconstruct_impl(counts, receivers_init, array_diameter, phi=PHI, lam=LAM, device=None,
                fit_rows=6000, seed=0, verbose=True, shared=False, footprint="disc", force_general=False,
                exact=False, min_spacing=None, repel_weight=None, n_starts=1, fit_iters=300,
                kernel_fit="steps", anneal=0.0, kernel=None, isotropy=ISO_WEIGHT, iso_center=False, n_passes=2,
                log_passes=False, refine_rounds="converge"):
    """Imaging-free reconstruction from a receiver-by-sender count matrix.

    counts          (n_receivers, n_senders) non-negative counts (UMIs); dense array or scipy sparse.
                    shared=True: a square (n_beads, n_beads) sender-by-receiver matrix of beads that
                    are both senders and receivers (self pairs are ignored).
    receivers_init  (n_receivers, 2) starting layout, normally a UMAP embedding (shared: of the beads).
    array_diameter  largest physical distance across the array (a disc's diameter, a rectangle's
                    diagonal); fixes the global scale. Outputs are in the same units.
    fit_iters       Adam steps of the kernel fit (default 300; the fit is not run to convergence, and
                    longer fits give a sharper kernel). Not used by the exact distinct-bead path.
    kernel_fit      "steps" (default): fit_iters Adam steps at a fixed rate, not converged;
                    "converged": Adam with the rate decayed to zero over fit_iters (default 3000) steps.
    anneal          > 0: warp levels use the kernel blurred by a Gaussian of sd anneal x grid spacing,
                    refinement the unblurred kernel (fast path only). Default 0.
    kernel          optional precomputed (knot values, background) in internal units, skipping the fit.
    isotropy        weight of the contact-isotropy prior in a second warp + refinement stage (default ISO_WEIGHT).
                    After the first refinement, contacts are identified from the counts (each receiver's top
                    partners, short in the current layout); a second warp (coarse levels) adds this weight times
                    the anisotropy of contact directions relative to the array-wide mean, followed by a second
                    refinement. 0 runs the second stage without the prior; None skips the second stage (the
                    original single-stage method). Fast path only (ignored with exact=True).
    log_passes      if True, the result has "pass_log": after every pass, the isotropy penalty, its sampling-noise floor
                    (random contact directions), the median bead movement since the previous pass (internal units) and
                    the receivers (input units, before the final rescaling). Does not change the result.
    refine_rounds   per-bead refinement rounds per stage: "converge" (default) repeats rounds until fewer than 1
                    percent of beads move (at least 2, at most 12; windows +-15 then +-10), or an
                    integer number of rounds.
    n_passes        number of warp + refinement passes (default 2: the first stage and one second stage); each
                    extra pass repeats the second stage (contacts, isotropy warp, refinement) on the current layout.
    iso_center      False (default): the prior assumes transfer is isotropic. True: local anisotropy is measured
                    relative to the array-wide mean, so a uniform directional bias of transfer (as in lattice
                    arrays whose transfer favours one lattice direction) is not mistaken for shear; this also
                    leaves an array-wide shear uncorrected.
    n_starts        number of initializations (default 1). With n_starts > 1, accumap is also run from
                    UMAP embeddings with further seeds (umap_layout) and the layout with the highest
                    maximized Poisson log-likelihood (log_likelihood) is returned; this guards against
                    regions that UMAP occasionally embeds mirror-imaged, which the smooth warp cannot undo.
                    The result then also has "start_logliks" and "chosen_start".
    min_spacing     physical minimum distance between bead centres (about one bead diameter), in the
                    caller's units; if given, per-bead refinement penalizes closer placements
                    (weight repel_weight, default REPEL_WEIGHT). Not available with exact=True.
    exact           True: all-pairs kernel sums and full-grid refinement (the computation used for the
                    published results); False (default): near/far kernel split, warm-started IPF and
                    coarse-to-fine refinement -- much faster and lighter.
    footprint       "disc" for circular arrays; otherwise the array outline as an (k, 2) polygon in the
                    caller's units (e.g. a rectangle's corners), whose rotation and position are fitted
                    to the layout. "auto" estimates the outline from the layout itself (not recommended:
                    it forgoes the shape information, and was worse on c58_8).
    Returns a dict: receivers, senders (centred coordinates, caller's units; shared: identical),
    receivers_warp, senders_warp (after the warp stage, before per-bead refinement),
    kernel_distance, kernel (K at the knots, distances in caller's units), background.
    Beads with no counts are not reconstructed (NaN rows).
    """
    import scipy.sparse as sp
    if n_starts > 1:
        inits = [receivers_init] + [umap_layout(counts, seed=1000 + k, shared=shared) for k in range(1, n_starts)]
        kw = dict(phi=phi, lam=lam, device=device, fit_rows=fit_rows, seed=seed, verbose=verbose, shared=shared,
                  footprint=footprint, force_general=force_general, exact=exact, min_spacing=min_spacing,
                  repel_weight=repel_weight, fit_iters=fit_iters, kernel_fit=kernel_fit, anneal=anneal,
                  isotropy=isotropy, iso_center=iso_center, n_passes=n_passes, log_passes=log_passes,
                  refine_rounds=refine_rounds)
        results, scores = [], []
        for k, E in enumerate(inits):
            R = reconstruct(counts, E, array_diameter, n_starts=1, **kw)
            scores.append(log_likelihood(counts, R["receivers"], array_diameter,
                                         senders=None if shared else R["senders"], shared=shared, device=device))
            results.append(R)
            if verbose:
                print(f"  start {k}: log-likelihood {scores[-1]:.6g}", flush=True)
        best = int(np.argmax(scores))
        out = results[best]
        out["start_logliks"], out["chosen_start"] = scores, best
        return out
    n_r, n_o = counts.shape
    global REPEL_WEIGHT
    if repel_weight is not None:
        REPEL_WEIGHT = float(repel_weight)
    if (not exact) or shared or force_general or min_spacing is not None or not (isinstance(footprint, str) and footprint == "disc") \
            or n_r * n_o * 4 > DENSE_FP32_BYTES:
        return _reconstruct_general(sp.csr_matrix(counts, dtype=np.float32), receivers_init, array_diameter,
                                    phi, lam, device, fit_rows, seed, verbose, shared, footprint, fast=not exact,
                                    min_spacing=min_spacing, fit_iters=fit_iters, kernel_fit=kernel_fit,
                                    anneal=anneal, kernel=kernel, isotropy=isotropy, iso_center=iso_center,
                                    n_passes=n_passes, log_passes=log_passes, refine_rounds=refine_rounds)
    if sp.issparse(counts):
        counts = counts.toarray()
    global KDE_FFT, SPLIT_NORMALIZER, REFINE_C2F
    KDE_FFT = False                                   # exact path: all-pairs KDEs, as published
    SPLIT_NORMALIZER = False
    REFINE_C2F = False
    torch = _torch()
    t0 = time.time()
    dev = torch.device(device) if device is not None else default_device()
    C = np.asarray(counts, np.float32)
    r_ok, o_ok = C.sum(1) > 0, C.sum(0) > 0
    Cf = np.ascontiguousarray(C[r_ok][:, o_ok])
    s = REF_DIAMETER / float(array_diameter)          # caller units -> internal units

    Xi = rescale_to_diameter(np.asarray(receivers_init)[r_ok], REF_DIAMETER)
    Ui = senders_from_receivers(Cf, Xi)
    X0 = torch.tensor(Xi, device=dev); U0 = torch.tensor(Ui, device=dev)
    if verbose:
        print(f"accumap count model: {Cf.shape[0]} receivers x {Cf.shape[1]} senders, "
              f"{Cf.sum():.3g} counts, device {dev}", flush=True)

    v, c = fit_kernel(X0, U0, Cf, dev, n_rows=fit_rows, seed=seed)
    tab = _kernel_table(v, dev)
    if verbose:
        print(f"  kernel fit on initial layout, background {c:.2e} ({time.time() - t0:.0f}s)", flush=True)
    Ct = torch.tensor(Cf, device=dev)
    Xw, Uw, q, a = warp_stage(X0, U0, Ct, tab, c, phi=phi, lam=lam, verbose=verbose, t0=t0)
    del Ct
    X, U = refine(Xw, Uw, Cf, tab, c, q, a)
    with torch.no_grad():   # known-array-size anchor: restore the initial RMS radius
        P, P0 = torch.cat([X, U]), torch.cat([X0, U0])
        c0, c1 = P0.mean(0), P.mean(0)
        f = ((P0 - c0).pow(2).sum(1).mean() / (P - c1).pow(2).sum(1).mean()).sqrt()
        X = (X - c1) * f + c0; U = (U - c1) * f + c0
    if verbose:
        print(f"  done ({time.time() - t0:.0f}s)", flush=True)

    Xo = np.full((C.shape[0], 2), np.nan, np.float32); Xo[r_ok] = X.cpu().numpy() / s
    Uo = np.full((C.shape[1], 2), np.nan, np.float32); Uo[o_ok] = U.cpu().numpy() / s
    Xwo = np.full((C.shape[0], 2), np.nan, np.float32); Xwo[r_ok] = Xw.cpu().numpy() / s
    Uwo = np.full((C.shape[1], 2), np.nan, np.float32); Uwo[o_ok] = Uw.cpu().numpy() / s
    return {"receivers": Xo, "senders": Uo, "receivers_warp": Xwo, "senders_warp": Uwo,
            "kernel_distance": KNOTS / s,
            "kernel": np.exp(v.cpu().numpy()), "background": c}


def _senders_from_receivers_sparse(Csc, X):
    """senders_from_receivers for a scipy CSC matrix (same result)."""
    U = np.full((Csc.shape[1], 2), np.nan, np.float32)
    for o in range(Csc.shape[1]):
        a, b = Csc.indptr[o], Csc.indptr[o + 1]
        nz, w = Csc.indices[a:b], Csc.data[a:b]
        if len(nz) == 0:
            continue
        for j in range(2):
            srt = np.argsort(X[nz, j])
            cw = np.cumsum(w[srt])
            U[o, j] = X[nz[srt[np.searchsorted(cw, cw[-1] / 2)]], j]
    return U


def _reconstruct_general(Csr, receivers_init, array_diameter, phi, lam, device, fit_rows, seed, verbose,
                         shared, footprint, fast=False, min_spacing=None, fit_iters=300, kernel_fit="steps",
                         anneal=0.0, kernel=None, isotropy=None, iso_center=False, n_passes=2, log_passes=False,
                         refine_rounds="converge"):
    torch = _torch()
    t0 = time.time()
    dev = torch.device(device) if device is not None else default_device()
    rs, cs = np.asarray(Csr.sum(1)).ravel(), np.asarray(Csr.sum(0)).ravel()
    if shared:
        assert Csr.shape[0] == Csr.shape[1], "shared=True needs a square bead-by-bead matrix"
        Csr = Csr.tolil(); Csr.setdiag(0); Csr = Csr.tocsr(); Csr.eliminate_zeros()
        r_ok = o_ok = (rs > 0) | (cs > 0)
    else:
        r_ok, o_ok = rs > 0, cs > 0
    Cf = Csr[r_ok][:, o_ok].tocsr()
    s = REF_DIAMETER / float(array_diameter)
    global REPEL_SPACING
    REPEL_SPACING = None if min_spacing is None else float(min_spacing) * s
    if not isinstance(footprint, str):
        fp = np.asarray(footprint, np.float64)
        footprint = (fp - fp.mean(0)) * s
    Xi = rescale_to_diameter(np.asarray(receivers_init)[r_ok], REF_DIAMETER)
    Ui = Xi if shared else _senders_from_receivers_sparse(Cf.tocsc(), Xi)
    X0 = torch.tensor(Xi, device=dev)
    U0 = X0 if shared else torch.tensor(Ui, device=dev)
    cnt = _Counts(Cf, dev, shared)
    if verbose:
        kind = f"{Cf.shape[0]} shared beads" if shared else f"{Cf.shape[0]} receivers x {Cf.shape[1]} senders"
        print(f"accumap count model ({'fast' if fast else 'exact general'} path): {kind}, {Cf.sum():.3g} counts, {Cf.nnz} nonzero, "
              f"kernel matrix {cnt.kernel_store}, footprint {footprint if isinstance(footprint, str) else 'polygon'}, "
              f"device {dev}", flush=True)
    global REFINE_ROUNDS
    REFINE_ROUNDS = refine_rounds   # set before the first refinement
    REFINE_LOG.clear()
    global KDE_FFT, SPLIT_NORMALIZER, REFINE_C2F
    KDE_FFT = bool(fast) and FAST_KDE
    SPLIT_NORMALIZER = bool(fast)
    REFINE_C2F = bool(fast) and FAST_C2F
    global ANNEAL, KERNEL_V
    if kernel is not None:
        v, c = torch.as_tensor(kernel[0], dtype=torch.float32, device=dev), float(kernel[1])
    elif fast and FAST_FIT and kernel_fit == "converged":
        v, c = fit_kernel_fast(X0, U0, cnt, dev, iters=fit_iters if fit_iters != 300 else 3000, decay=True)
    elif fast and FAST_FIT:
        v, c = fit_kernel_fast(X0, U0, cnt, dev, iters=fit_iters)
    else:
        v, c = fit_kernel(X0, U0, Cf, dev, n_rows=fit_rows, seed=seed, shared=shared, iters=fit_iters)
    tab = _kernel_table(v, dev)
    ANNEAL, KERNEL_V = (float(anneal) if fast else 0.0), v
    if verbose:
        print(f"  kernel fit on initial layout, background {c:.2e} ({time.time() - t0:.0f}s)", flush=True)
    stage, ref = (warp_stage_fast, refine_fast if FAST_REFINE else refine_general) if fast else (warp_stage_general, refine_general)
    Xw, Uw, q, a = stage(X0, U0, cnt, tab, c, phi=phi, lam=lam, footprint=footprint, verbose=verbose, t0=t0)
    X, U = ref(Xw, Uw, Cf, cnt, tab, c, q, a)
    global ISO_CENTER
    ISO_CENTER = bool(iso_center)
    plog, prevX = ([] if ((log_passes or verbose) and fast) else None), None

    def _log_pass():
        nonlocal prevX
        with torch.no_grad():
            rr, oo = contact_pairs(Cf, X, U, shared, a=a)
            Pall = X if shared else torch.cat([X, U])
            lo_, hi_ = Pall.min(0).values - 200, Pall.max(0).values + 200
            pen = float(isotropy_penalty(X, X if shared else U, rr, oo, lo_, hi_))
            floor = isotropy_noise_floor(X, X if shared else U, rr, oo, lo_, hi_)
            mv = float((X - prevX).norm(dim=1).median()) if prevX is not None else float("nan")
            Xo = np.full((Csr.shape[0], 2), np.nan, np.float32); Xo[r_ok] = X.cpu().numpy() / s
            plog.append({"penalty": pen, "floor": floor, "moved": mv, "contacts": int(len(rr)), "receivers": Xo})
            prevX = X.clone()
            if verbose:
                print(f"  pass {len(plog)}: contact anisotropy / sampling floor = {pen / floor:.2f} "
                      "(a correct layout is about 1)", flush=True)

    if plog is not None:
        _log_pass()
    for _pass in range(max(0, n_passes - 1) if (isotropy is not None and fast) else 0):
        ri, oi = contact_pairs(Cf, X, U, shared, a=a)
        iso = {"ri": ri, "oi": oi, "weight": float(isotropy)} if isotropy > 0 else None
        if verbose:
            with torch.no_grad():
                P = torch.cat([X, U]) if not shared else X
                p0 = float(isotropy_penalty(X, X if shared else U, ri, oi, P.min(0).values - 200, P.max(0).values + 200))
            print(f"  isotropy stage: {len(ri)} contacts, weight {isotropy}, penalty before {p0:.2e}", flush=True)
        Xw, Uw, q, a = warp_stage_fast(X, U, cnt, tab, c, phi=phi, lam=lam, footprint=footprint, verbose=verbose,
                                       t0=t0, schedule=ISO_SCHEDULE, iso=iso)
        X, U = ref(Xw, Uw, Cf, cnt, tab, c, q, a)
        if plog is not None:
            _log_pass()
    tdiag = None
    if fast:   # check the isotropic-transfer assumption on the final layout (imaging-free)
        rf, of, cf = contact_pairs(Cf, X, U, shared, a=a, return_counts=True)
        tdiag = transfer_anisotropy(X, X if shared else U, q, a, tab, c, rf, of, cf)
        if tdiag["amplitude"] > TRANSFER_WARN[0] and tdiag["z"] > TRANSFER_WARN[1]:
            import warnings
            warnings.warn(f"accumap: counts suggest direction-dependent transfer (amplitude {tdiag['amplitude']:.3f}, "
                          f"z {tdiag['z']:.0f}, axis {tdiag['axis_deg']:.0f} deg); the isotropy prior assumes none. "
                          "Consider iso_center=True or isotropy=0.")
        if verbose:
            print(f"  transfer anisotropy check: amplitude {tdiag['amplitude']:.3f}, z {tdiag['z']:.1f}", flush=True)
    with torch.no_grad():   # known-array-size anchor: restore the initial RMS radius
        P, P0 = (X, X0) if shared else (torch.cat([X, U]), torch.cat([X0, U0]))
        c0, c1 = P0.mean(0), P.mean(0)
        f = ((P0 - c0).pow(2).sum(1).mean() / (P - c1).pow(2).sum(1).mean()).sqrt()
        X = (X - c1) * f + c0; U = X if shared else (U - c1) * f + c0
    if verbose:
        print(f"  done ({time.time() - t0:.0f}s)", flush=True)
    n_r, n_o = Csr.shape
    Xo = np.full((n_r, 2), np.nan, np.float32); Xo[r_ok] = X.cpu().numpy() / s
    Uo = np.full((n_o, 2), np.nan, np.float32); Uo[o_ok] = U.cpu().numpy() / s
    Xwo = np.full((n_r, 2), np.nan, np.float32); Xwo[r_ok] = Xw.cpu().numpy() / s
    Uwo = np.full((n_o, 2), np.nan, np.float32); Uwo[o_ok] = Uw.cpu().numpy() / s
    return {"receivers": Xo, "senders": Uo, "receivers_warp": Xwo, "senders_warp": Uwo,
            "kernel_distance": KNOTS / s, "kernel": np.exp(v.cpu().numpy()), "background": c,
            "transfer_anisotropy": tdiag, "pass_log": plog if log_passes else None, "refine_log": list(REFINE_LOG)}


def _fixed_layout(counts, receivers, senders, array_diameter, device):
    """Counts and positions in internal units, empty rows/columns and non-finite positions dropped."""
    torch = _torch()
    import scipy.sparse as sp
    dev = torch.device(device) if device is not None else default_device()
    C = sp.csr_matrix(counts, dtype=np.float32)
    r_ok = (np.asarray(C.sum(1)).ravel() > 0) & np.isfinite(receivers).all(1)
    o_ok = (np.asarray(C.sum(0)).ravel() > 0) & np.isfinite(senders).all(1)
    s = REF_DIAMETER / float(array_diameter)
    X = torch.tensor(np.asarray(receivers, np.float32)[r_ok] * s, device=dev)
    U = torch.tensor(np.asarray(senders, np.float32)[o_ok] * s, device=dev)
    Cf = C[r_ok][:, o_ok].tocsr()
    return Cf, X, U, _Counts(Cf, dev, False), dev


def _fit_kernel_at_impl(counts, receivers, senders, array_diameter, device=None, iters=300):
    """Fit the transfer kernel and background with receivers and senders held at the given positions (e.g. a
    reconstruction, or ground truth), as accumap does on its starting layout. Returns (knot values, background)
    in internal units, which reconstruct(kernel=...) accepts."""
    _, X, U, cnt, dev = _fixed_layout(counts, receivers, senders, array_diameter, device)
    v, c = fit_kernel_fast(X, U, cnt, dev, iters=iters)
    return v.cpu().numpy(), float(c)


def _contact_anisotropy_impl(counts, receivers, senders, array_diameter, device=None):
    """Anisotropy of contact directions in a layout, relative to the level expected from sampling alone (random
    directions). About 1 for a correct layout of an isotropic array; well above 1 indicates shear. Contacts are
    chosen as accumap's isotropy prior chooses them (kernel and abundances fit at these positions).
    Returns {"penalty", "floor", "ratio", "contacts"}."""
    torch = _torch()
    Cf, X, U, cnt, dev = _fixed_layout(counts, receivers, senders, array_diameter, device)
    v, c = fit_kernel_fast(X, U, cnt, dev)
    tab = _kernel_table(v, dev)
    a0 = torch.log(cnt.Ns + 1.0); a0 = a0 - a0.mean()
    q, a = _ipf_fast(X, U, _Fast(X, U, tab, c, False), cnt, a0, 30)
    ri, oi = contact_pairs(Cf, X, U, False, a=a)
    P = torch.cat([X, U]); lo, hi = P.min(0).values - 200, P.max(0).values + 200
    with torch.no_grad():
        pen = float(isotropy_penalty(X, U, ri, oi, lo, hi))
    fl = isotropy_noise_floor(X, U, ri, oi, lo, hi)
    return {"penalty": pen, "floor": fl, "ratio": pen / fl, "contacts": int(len(ri))}


def umap_layout(counts, seed=0, shared=False, n_epochs=400, fast=False, backend="umap-learn"):
    """UMAP initialization as used in the paper: log(1 + C) (symmetrized C + C^T for shared beads),
    cosine metric, 25 neighbours, minimum distance 0.99.
    fast       for large arrays (only the starting layout changes; accumap still uses every count): pairs with a
               single count are dropped from UMAP's input (rows left empty keep their counts), 200 epochs, and
               umap-learn runs multi-threaded without a fixed seed, so starts are not exactly reproducible.
    backend    "umap-learn" (default, as in the paper) or "cuml" (RAPIDS GPU UMAP; same settings, different
               implementation, so its layouts differ from umap-learn's)."""
    import scipy.sparse as sp
    C = sp.csr_matrix(counts, dtype=np.float32)
    if shared:
        C = (C + C.T).tocsr()
    C = C.copy()
    if fast:
        n_epochs = min(n_epochs, 200)
        keep = C.copy(); keep.data[keep.data < 2] = 0; keep.eliminate_zeros()
        empty = np.diff(keep.indptr) == 0
        if empty.any():
            keep = (keep + sp.diags(empty.astype(np.float32)) @ C).tocsr()
        C = keep
    C.data = np.log1p(C.data)
    if backend == "cuml":
        from cuml.manifold import UMAP
        return np.asarray(UMAP(random_state=None if fast else seed, metric="cosine", n_neighbors=25, min_dist=0.99,
                               n_components=2, n_epochs=n_epochs).fit_transform(C), dtype=np.float32)
    import umap
    return umap.UMAP(random_state=None if fast else seed, metric="cosine", n_neighbors=25, min_dist=0.99,
                     n_components=2, n_epochs=n_epochs, n_jobs=-1 if fast else 1).fit_transform(C).astype(np.float32)


def _log_likelihood_impl(counts, receivers, array_diameter, senders=None, shared=False, device=None):
    """Maximized Poisson log-likelihood of a layout (no ground truth): the kernel, background, sender
    abundances and receiver capture are refit to the counts at the given positions. Higher is better;
    comparable between layouts of the same counts (e.g. to choose among several initializations)."""
    torch = _torch()
    import scipy.sparse as sp
    dev = torch.device(device) if device is not None else default_device()
    s = REF_DIAMETER / float(array_diameter)
    C = sp.csr_matrix(counts, dtype=np.float32)
    if shared:
        C = C.tolil(); C.setdiag(0); C = C.tocsr(); C.eliminate_zeros()
        ok = (np.asarray(C.sum(1)).ravel() > 0) | (np.asarray(C.sum(0)).ravel() > 0)
        r_ok = o_ok = ok & np.isfinite(np.asarray(receivers)).all(1)
    else:
        r_ok = (np.asarray(C.sum(1)).ravel() > 0) & np.isfinite(np.asarray(receivers)).all(1)
        o_ok = (np.asarray(C.sum(0)).ravel() > 0) & np.isfinite(np.asarray(senders)).all(1)
    Cf = C[r_ok][:, o_ok].tocsr()
    cnt = _Counts(Cf, dev, shared)
    X = torch.tensor(np.asarray(receivers)[r_ok] * s, dtype=torch.float32, device=dev)
    U = X if shared else torch.tensor(np.asarray(senders)[o_ok] * s, dtype=torch.float32, device=dev)
    v, c = fit_kernel_fast(X, U, cnt, dev)
    tab = _kernel_table(v, dev)
    with torch.no_grad():
        fk = _Fast(X, U, tab, c, shared)
        q, a = _ipf_fast(X, U, fk, cnt, torch.zeros(cnt.n_o, device=dev), 30)
        d = torch.sqrt(((X[cnt.rows] - U[cnt.cols]) ** 2).sum(-1) + 1e-4)
        obs = (cnt.vals * (q[cnt.rows] + a[cnt.cols] + _logm(d, tab, c))).sum()
        exp_ = (torch.exp(q) * fk.row_sums(X, U, torch.exp(a))).sum()
    return float(obs - exp_)



def _with_local_scale(impl, diameter_pos):
    """Public entry point: impl run under _local_scale(array_diameter); diameter_pos = position of array_diameter."""
    def f(*args, **kwargs):
        d = args[diameter_pos] if len(args) > diameter_pos else kwargs["array_diameter"]
        with _local_scale(d):
            return impl(*args, **kwargs)
    f.__name__ = impl.__name__[1:-5]; f.__doc__ = impl.__doc__
    return f


reconstruct = _with_local_scale(_reconstruct_impl, 2)
log_likelihood = _with_local_scale(_log_likelihood_impl, 2)
fit_kernel_at = _with_local_scale(_fit_kernel_at_impl, 3)
contact_anisotropy = _with_local_scale(_contact_anisotropy_impl, 3)
