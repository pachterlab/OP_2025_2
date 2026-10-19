"""Multilevel starting layout for large arrays (with count_model.LENGTH_REF_UM set), and relocation of misplaced beads.

Coarsening: heavy-edge matching (as in multilevel graph partitioning) on the symmetrized counts, restricted to each node's
MATCH_K strongest size-normalized partners (optionally mutual pairs only, MATCH_MUTUAL), roughly halving the number of nodes
per round until at most `target` super-beads remain; super-bead counts are summed.
Coarsest level: UMAP (paper settings) + accumap. For elongated polygon footprints (aspect >= COARSE_ASPECT_MIN), UMAP's
default spectral initialization folds the array (rectangles longer than ~2:1), so COARSE_RANDOM_TRIES random-init UMAPs are
also tried, each followed by accumap, and the layout with the least density excess (folds crowd beads) is kept.
Back up the levels: each node starts at its coarse node's position and accumap is re-run on every level of at most `finest`
nodes, with LENGTH_REF_UM scaled to that level's node spacing (level_scale); finer levels are prolonged only.
Relocation (polygon footprints): beads outside the posed footprint, or far from all their strongest partners, are moved to the
median of their well-placed strongest partners, from the edge of each misplaced patch inward (relocate, relocate_passes);
applied at the end of the start, and usable on accumap's result.

  multilevel_layout(C, footprint=polygon or diam=...)   shared beads -> (n, 2) start, info
  multilevel_layout_distinct(C, diam, footprint=None)   distinct receivers x senders (joint nodes) -> (n_receivers, 2), info
  relocate_passes(C, Y, footprint)                      relocation passes on any layout -> (Y, beads flagged)
Positions are in the caller's units (those of the footprint or diameter)."""
import time, numpy as np, scipy.sparse as sp
from . import count_model as cm


class level_scale:
    """Context: count_model.LENGTH_REF_UM scaled for a level of n_nodes super-beads of an n_beads array by
    sqrt(n_beads / n_nodes) (the node spacing relative to the bead spacing), so local lengths match the nodes:
    coarse levels fall back to the published relative scaling (reference larger than the array)."""
    def __init__(self, n_beads, n_nodes):
        self.f = float(np.sqrt(n_beads / n_nodes))

    def __enter__(self):
        self.old = cm.LENGTH_REF_UM
        if self.old is not None:
            cm.LENGTH_REF_UM = self.old * self.f
        return self

    def __exit__(self, *exc):
        cm.LENGTH_REF_UM = self.old
        return False


COARSE_ASPECT_MIN = 1.5    # footprints at least this elongated: the coarsest level tries the default UMAP (spectral init) and
COARSE_RANDOM_TRIES = 16   # this many random-init UMAPs (UMAP's default epochs), each followed by accumap; keeps the layout with the
COARSE_ELONGATED_SELECT = "density"   # lowest density excess (folds/overlaps crowd beads). Spectral init folds rectangles longer than
                           # ~2:1 (the 2nd long-axis harmonic precedes the short-axis mode). Eye chart 2.4:1 coarsest level
                           # (2,778 nodes), two pools (12 at 500 epochs, 24 at 400): minimum density excess picked the lowest-error
                           # layout both times (255 / 273 um; spectral ~2,350); log-likelihood only once (Spearman 0.79 / 0.43);
                           # near-zero excess in 3/12 random inits at 500 epochs vs 2/24 at 400.


def _aspect(P):
    """Principal-axis aspect ratio (1st-99th percentile extents) of a point set."""
    P = np.asarray(P, float); P = P - P.mean(0)
    _, V = np.linalg.eigh(np.cov(P.T)); Z = P @ V
    ext = np.percentile(Z, 99, 0) - np.percentile(Z, 1, 0)
    return float(max(ext) / max(min(ext), 1e-12))


def footprint_aspect(sq):
    """Aspect ratio of a polygon footprint (points sampled inside it), as _aspect; None for "disc" or no footprint."""
    if sq is None or isinstance(sq, str):
        return None
    from matplotlib.path import Path
    sq = np.asarray(sq, float); lo, hi = sq.min(0), sq.max(0)
    g = np.stack(np.meshgrid(np.linspace(lo[0], hi[0], 200), np.linspace(lo[1], hi[1], 200)), -1).reshape(-1, 2)
    return _aspect(g[Path(sq).contains_points(g)])


def coarse_candidates(Cs, seed, target_aspect=None):
    """Coarsest-level UMAP starts: [(name, layout)], the default (spectral init), plus COARSE_RANDOM_TRIES random-init
    UMAPs (cm.umap_layout settings, UMAP's default epochs) for elongated footprints."""
    out = [(f"spectral s{seed}", cm.umap_layout(Cs, seed=seed, shared=True))]
    if target_aspect is not None and target_aspect >= COARSE_ASPECT_MIN and COARSE_RANDOM_TRIES:
        import umap
        S = (Cs + Cs.T).tocsr().astype(np.float32); S.data = np.log1p(S.data)
        for t in range(COARSE_RANDOM_TRIES):
            out.append((f"random s{seed + 1 + t}", umap.UMAP(random_state=seed + 1 + t, metric="cosine", n_neighbors=25, min_dist=0.99,
                                                             n_components=2, init="random").fit_transform(S).astype(np.float32)))
    return out


RELOCATE_K = 10            # relocation: strongest partners per node used to place misplaced nodes
RELOCATE_MIN_GOOD = 3      # a misplaced node is placed once this many of its strongest partners are well placed
RELOCATE_POSE_FRAC = 0.97  # footprint posed to contain this fraction of nodes, then widened by RELOCATE_MARGIN of its short side;
RELOCATE_MARGIN = 0.01     # nodes outside it are misplaced (eye chart, after accumap: margin 0.03 / 0.01 / 0.005 -> beads > 1 mm
                           # 0.47 / 0.07 / 0.01%, robust median 50.0 / 50.8 / 50.9 um: edge beads pulled to their partners)
RELOCATE_DIST = 4.0        # ... as are nodes farther than this many median partner distances from their partners' median
RELOCATE_FLOOR = 0.02      # (and at least this fraction of the footprint's short side, if a polygon footprint is given)
RELOCATE_AFTER = True          # for callers (e.g. run scripts): also relocate accumap's result (polygon footprints)
RELOCATE_PASSES = (4.0, 3.0)   # multilevel start, polygon footprints: relocation passes on the beads at the end (RELOCATE_DIST per
                               # pass; footprint test in the first). Eye chart run-6 result: beads > 1 mm 2.53% -> 0.84% -> 0.52%
                               # (flagged beads 16.8 mm -> 126 um); () = off


def relocate(C, Y, footprint=None, verbose=True, outside=True):
    """Move misplaced nodes (outside the posed footprint, or far from all their strongest partners) to the median of their
    well-placed strongest partners, from the boundary of each misplaced patch inward (a node is placed once
    RELOCATE_MIN_GOOD of its RELOCATE_K strongest partners are well placed; placed nodes count as well placed in later
    rounds), so coherent misplaced patches are moved piece by piece. -> (Y, number relocated)"""
    C = sp.csr_matrix(C)
    r, c = _topk_nb(C.indptr.astype(np.int64), C.indices.astype(np.int64), C.data.astype(np.float64), int(RELOCATE_K))
    n = C.shape[0]
    order = np.argsort(r, kind="stable"); r, c = r[order], c[order]
    ptr = np.zeros(n + 1, np.int64); np.add.at(ptr, r + 1, 1); ptr = np.cumsum(ptr)
    Y = np.array(Y, float)
    med = np.full((n, 2), np.nan)
    for i in range(n):                                  # partners' median position per node
        if ptr[i + 1] > ptr[i]:
            med[i] = np.median(Y[c[ptr[i]:ptr[i + 1]]], 0)
    d = np.linalg.norm(Y - med, axis=1)
    thr = RELOCATE_DIST * np.nanmedian(d)
    poly = footprint is not None and not isinstance(footprint, str)
    if poly:
        from matplotlib.path import Path
        posed = cm._pose_polygon(Y, np.asarray(footprint, float) - np.asarray(footprint, float).mean(0), frac=RELOCATE_POSE_FRAC)
        ctr = posed.mean(0); V = posed - ctr
        short = float(np.sqrt(np.linalg.eigvalsh(np.cov(V.T))[0]) * 2 * np.sqrt(3))
        thr = max(thr, RELOCATE_FLOOR * short)          # prolonged starts: beads of a node coincide, so the median is ~0
    bad = np.isfinite(d) & (d > thr)
    if poly and outside:
        grown = ctr + V * (1 + 2 * RELOCATE_MARGIN * short / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9))
        bad |= ~Path(grown).contains_points(Y)
    n_bad = int(bad.sum())
    good = ~bad
    for _ in range(1000):
        idx = np.nonzero(~good)[0]
        if not len(idx):
            break
        moved = []
        for i in idx:
            nb = c[ptr[i]:ptr[i + 1]]; g = nb[good[nb]]
            if len(g) >= RELOCATE_MIN_GOOD:
                moved.append((i, np.median(Y[g], 0)))
        if not moved:
            break
        for i, p in moved:
            Y[i] = p; good[i] = True
    if verbose:
        print(f"  relocate: {n_bad:,} misplaced nodes ({100 * n_bad / n:.2f}%), {int(good.sum() - (n - n_bad)):,} relocated", flush=True)
    return Y, n_bad


COARSE_STARTS = 1  # UMAP starts at the coarsest level (accumap n_starts: the highest-likelihood layout is kept)
COARSE_SEEDS = 1   # UMAP seeds tried at the coarsest level, each followed by accumap; the best by COARSE_SELECT is kept
COARSE_SELECT = "density"   # "density": least density excess (fraction of nodes above 1.6x the median 10-NN density;
                            # folded regions overlap their neighbours); "loglik": highest maximized log-likelihood


def density_excess(Y):
    """Fraction of nodes whose local density (10-NN) exceeds 1.6x the median (overlap from folds)."""
    from scipy.spatial import cKDTree
    Y = np.asarray(Y, float); Y = Y[np.isfinite(Y).all(1)]
    rho = 1 / cKDTree(Y).query(Y, 11)[0][:, -1] ** 2
    return float((rho > 1.6 * np.median(rho)).mean())
SPARSIFY_K = 30   # coarse levels: keep each node's SPARSIFY_K strongest partners (union over both directions) before
                  # reconstructing; super-beads pool background counts, so coarse graphs are nearly dense (18 mm: 61%
                  # of pairs at 17k nodes, 100% at 4.3k) and the neighbour signal is diluted. None: keep all.


def sparsify(Cc, k=None, mutual=False):
    """Counts of each node's k strongest partners (by symmetrized count), union over both directions (mutual=True:
    only pairs each in the other's k strongest)."""
    k = SPARSIFY_K if k is None else k
    if k is None:
        return Cc
    S = (Cc + Cc.T).tocsr(); S.setdiag(0); S.eliminate_zeros()
    r, c = _topk_nb(S.indptr.astype(np.int64), S.indices.astype(np.int64), S.data.astype(np.float64), int(k))
    M = sp.csr_matrix((np.ones(len(r), bool), (r, c)), shape=Cc.shape)
    M = M.multiply(M.T).astype(bool) if mutual else (M + M.T).astype(bool)
    return Cc.multiply(M).tocsr()


try:                                                   # compiled per-node loops (numba); pure Python otherwise
    from numba import njit
except Exception:                                      # pragma: no cover
    def njit(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]


@njit(cache=True)
def _heavy_edge_nb(indptr, indices, data, order):
    n = len(indptr) - 1; match = np.full(n, -1, np.int64)
    for i in order:
        if match[i] >= 0:
            continue
        best, bw = -1, -1.0
        for p in range(indptr[i], indptr[i + 1]):
            j = indices[p]
            if match[j] < 0 and j != i and data[p] > bw:
                bw = data[p]; best = j
        if best >= 0:
            match[i] = best; match[best] = i
        else:
            match[i] = i
    return match


@njit(cache=True)
def _topk_nb(indptr, indices, data, k):
    n = len(indptr) - 1; cnt = np.zeros(n, np.int64)
    for i in range(n):
        cnt[i] = min(k, indptr[i + 1] - indptr[i])
    off = np.zeros(n + 1, np.int64)
    for i in range(n):
        off[i + 1] = off[i] + cnt[i]
    rows = np.empty(off[n], np.int64); cols = np.empty(off[n], np.int64)
    for i in range(n):
        a, b = indptr[i], indptr[i + 1]
        if b > a:
            sel = np.argsort(-data[a:b])[:cnt[i]]
            for t in range(cnt[i]):
                rows[off[i] + t] = i; cols[off[i] + t] = indices[a + sel[t]]
    return rows, cols


@njit(cache=True)
def _argmax_rows_nb(indptr, indices, data):
    n = len(indptr) - 1; out = np.full(n, -1, np.int64)
    for i in range(n):
        a, b = indptr[i], indptr[i + 1]
        if b > a:
            best = a
            for p in range(a + 1, b):
                if data[p] > data[best]:
                    best = p
            out[i] = indices[best]
    return out


def _footprint(side=None, diam=None, footprint=None):
    """(diameter, footprint for count_model.reconstruct) from a square side, or an explicit footprint: "disc" (needs diam) or a
    polygon (k, 2) in caller units, whose diameter (largest vertex distance) is used when diam is not given."""
    if side is not None:
        return side * np.sqrt(2), np.array([[0, 0], [side, 0], [side, side], [0, side]], float) - side / 2
    if footprint is None or (isinstance(footprint, str) and footprint == "disc"):
        assert diam is not None, "a disc footprint needs diam"
        return float(diam), "disc"
    fp = np.asarray(footprint, float); fp = fp - fp.mean(0)
    d = float(np.max(np.linalg.norm(fp[:, None] - fp[None], axis=-1))) if diam is None else float(diam)
    return d, fp


def heavy_edge_round(S, rng):
    """One matching round on symmetric S (CSR, zero diagonal): cluster label per node."""
    n = S.shape[0]
    match = _heavy_edge_nb(S.indptr.astype(np.int64), S.indices.astype(np.int64), S.data.astype(np.float64), rng.permutation(n).astype(np.int64))
    rep = np.minimum(np.arange(n), match)
    _, lab = np.unique(rep, return_inverse=True)
    return lab


FULL_BELOW = 70000   # with PRUNE_K0: levels with at most this many nodes (the ones multilevel_layout reconstructs) carry full
                     # aggregated counts; larger levels carry the pruned matching graph only
PRUNE_K0 = 30   # coarsen() first keeps each node's PRUNE_K0 strongest partners (count matrix symmetrized); the scipy operations
                # on the full matrix (443M nonzeros at 18 mm) dominated coarsening time. None: no pruning
MATCH_WEIGHT = "sqrt"   # edge weight for MATCH_K: "sqrt" (count / sqrt(size_i size_j)) or "excess" (count minus the
                        # background expectation k_i k_j / sum k, as in modularity)
MATCH_MUTUAL = False    # True: mutual_round (pairs only of mutual MATCH_K-strongest neighbours; leftovers join their
                        # strongest neighbour's group)
MATCH_K = 6   # heavy-edge matching only among each node's MATCH_K strongest size-normalized neighbours (count /
              # sqrt(size_i size_j)); None: all neighbours (the original rule). Unrestricted matching merged distant pieces
              # through pooled background counts (coarsest nodes with members > 1 mm away: Hu 12 mm 91%, shared 18 mm 42%).


def mutual_round(W, rng):
    """One coarsening round on symmetric weights W (each node's MATCH_K strongest kept, union): heavy-edge matching
    restricted to mutual pairs (each among the other's MATCH_K strongest), then every node left unmatched joins the
    group of its single strongest neighbour (instead of pairing with whatever free node remains)."""
    n = W.shape[0]
    Wm = W.multiply(sparsify(W, MATCH_K, mutual=True) > 0).tocsr()        # mutual edges only
    lab = heavy_edge_round(Wm, rng)                                          # pairs; unmatched nodes are singletons
    cnt = np.bincount(lab)
    single = np.flatnonzero(cnt[lab] == 1)
    strongest = _argmax_rows_nb(W.indptr.astype(np.int64), W.indices.astype(np.int64), W.data.astype(np.float64))
    ok = strongest[single] >= 0
    lab[single[ok]] = lab[strongest[single[ok]]]                             # join the strongest neighbour's group
    _, lab = np.unique(lab, return_inverse=True)
    return lab


def coarsen(C, target=5000, seed=0, verbose=True, keep_levels=False):
    """Labels (bead -> super-bead) and the coarse count matrix; keep_levels: also the list of (labels, counts) of
    every level (labels map level k-1 nodes to level k nodes; level 0 = beads)."""
    rng = np.random.default_rng(seed)
    C = C.tocsr(); lab = np.arange(C.shape[0]); Cc = C; levels = []
    Cfull = None                                        # full counts of the current level (once at or below FULL_BELOW nodes)
    if PRUNE_K0 is not None:                            # matching only uses a node's strongest partners: prune once
        Cc = sparsify(C, PRUNE_K0)                      # (symmetric union) so the matching levels are cheap
        if verbose:
            print(f"  coarsen: pruned to each node's {PRUNE_K0} strongest partners: {C.nnz:,} -> {Cc.nnz:,} nonzero", flush=True)
    size = np.ones(C.shape[0])                          # beads per node
    while Cc.shape[0] > target:
        t0 = time.time()
        Sc = (Cc + Cc.T).tocsr(); Sc.setdiag(0); Sc.eliminate_zeros()
        if MATCH_K is not None:
            r_ = np.repeat(np.arange(Sc.shape[0]), np.diff(Sc.indptr))
            if MATCH_WEIGHT == "excess":               # count above its expectation from node totals alone (background)
                k_ = np.asarray(Sc.sum(1)).ravel(); w = Sc.data - k_[r_] * k_[Sc.indices] / k_.sum()
                w = np.maximum(w, 0.0)
            else:                                      # "sqrt": count / sqrt(size_i size_j)
                w = Sc.data / np.sqrt(size[r_] * size[Sc.indices])
            Sc = sp.csr_matrix((w, Sc.indices, Sc.indptr), shape=Sc.shape); Sc.eliminate_zeros()
            Sc = sparsify(Sc, MATCH_K)
        if MATCH_MUTUAL:
            l = mutual_round(Sc, rng)
        else:
            l = heavy_edge_round(Sc, rng)
        size = np.bincount(l, size)
        P = sp.csr_matrix((np.ones(len(l), np.float32), (np.arange(len(l)), l)))
        Cc = (P.T @ Cc @ P).tocsr(); Cc.setdiag(0); Cc.eliminate_zeros()
        lab = l[lab]
        if PRUNE_K0 is not None:
            # the coarse reconstructions need full counts (pruned graphs keep only adjacent nodes): aggregate the full
            # matrix once, directly to the first level at or below FULL_BELOW nodes, then level by level from there
            if Cfull is None and Cc.shape[0] <= FULL_BELOW:
                Pt = sp.csr_matrix((np.ones(len(lab), np.float32), (np.arange(len(lab)), lab)), shape=(len(lab), Cc.shape[0]))
                Cfull = (Pt.T @ C @ Pt).tocsr(); Cfull.setdiag(0); Cfull.eliminate_zeros()
            elif Cfull is not None:
                Cfull = (P.T @ Cfull @ P).tocsr(); Cfull.setdiag(0); Cfull.eliminate_zeros()
        levels.append((l, Cfull if Cfull is not None else Cc))
        if verbose:
            print(f"  coarsen: {len(l):,} -> {Cc.shape[0]:,} nodes ({time.time() - t0:.0f} s)", flush=True)
        if Cc.shape[0] > 0.9 * len(l):
            break
    Cout = Cfull if Cfull is not None else Cc
    return (lab, Cout, levels) if keep_levels else (lab, Cout)


PROLONG = "own"   # "own": each node at its own coarse node's position (diag_prolong.py: 3x lower error than "mean" at 18 mm);
                  # "mean": count-weighted mean position of all partners' coarse nodes (pulled to the centre by background counts)


def prolong(C_fine, labels, Y_coarse, how=None):
    """Positions for the nodes of a finer level from the coarse positions (PROLONG)."""
    if (how or PROLONG) == "own":
        return np.asarray(Y_coarse, np.float32)[labels]
    S = (C_fine + C_fine.T).tocsr(); S = S + sp.identity(S.shape[0], format="csr") * float(np.median(S.data))
    P = sp.csr_matrix((np.ones(len(labels)), (np.arange(len(labels)), labels)), shape=(len(labels), len(Y_coarse)))
    W = S @ P
    W = sp.diags(1 / np.asarray(W.sum(1)).ravel()) @ W
    return np.asarray(W @ Y_coarse, np.float32)


def multilevel_layout(C, side=None, target=5000, finest=70000, seed=0, device="cuda:0", verbose=True, target_beads=None, on_level=None,
                      diam=None, footprint=None, n_beads=None):
    """Multilevel start: coarsen to <= target nodes; reconstruct the coarsest level (UMAP + accumap); then, level by
    level back up, prolong and re-run accumap on that level's counts, while the level has <= finest nodes; finally
    prolong to the beads. Each level runs with LENGTH_REF_UM scaled to its node spacing (level_scale).
    side: square array; or footprint: "disc" (with diam) or a polygon in caller units (diam derived if not given).
    n_beads: beads behind C's nodes (default C.shape[0]).
    -> (n, 2) start in caller units, info."""
    if target_beads is not None:                       # coarsest nodes of about target_beads beads each
        target = C.shape[0] / target_beads
    lab, Cc, levels = coarsen(C, target, seed, verbose, keep_levels=True)
    diam, sq = _footprint(side, diam, footprint)        # square side, disc + diam, or any polygon (caller units)
    t0 = time.time(); n = C.shape[0] if n_beads is None else n_beads   # beads behind the nodes (level scaling)
    with level_scale(n, Cc.shape[0]):
        Cs = sparsify(Cc)
        best = None
        fa = footprint_aspect(sq)
        cands = [c for k in range(COARSE_SEEDS) for c in coarse_candidates(Cs, seed + 1000 * k, fa)]
        select = COARSE_ELONGATED_SELECT if (fa is not None and fa >= COARSE_ASPECT_MIN and COARSE_RANDOM_TRIES) else COARSE_SELECT
        for k, (name, Ec) in enumerate(cands):        # selection at the coarsest level (seeds and/or random inits)
            if on_level is not None and k == 0:        # diagnostic hook: coarsest UMAP before accumap
                on_level(f"{len(levels)} umap", Ec, levels)
            Yk = np.asarray(cm.reconstruct(Cs, Ec, diam, shared=True, footprint=sq,
                                           verbose=False, device=device, n_starts=COARSE_STARTS)["receivers"], float)
            if len(cands) > 1:
                sc = (density_excess(Yk) if select == "density" else
                      -cm.log_likelihood(Cs, Yk, diam, shared=True, device=device))
                if verbose:
                    print(f"  coarsest {name}: {select} score {sc:.6g}", flush=True)
            else:
                sc = 0.0
            if best is None or sc < best[0]:
                best = (sc, Yk, name)
        Y = best[1]
        if verbose and len(cands) > 1:
            print(f"  coarsest: kept {best[2]} ({select})", flush=True)
    if verbose:
        print(f"  level {len(levels)}: {Cc.shape[0]:,} nodes reconstructed ({time.time() - t0:.0f} s)", flush=True)
    if on_level is not None:                           # diagnostic hook: (level, layout, levels)
        on_level(len(levels), Y, levels)
    for lev in range(len(levels) - 1, -1, -1):          # levels[i] = (labels level i -> i+1, counts of level i+1)
        C_lev = C.tocsr() if lev == 0 else levels[lev - 1][1]
        Y = prolong(C_lev, levels[lev][0], Y)
        if 0 < lev and C_lev.shape[0] <= finest:       # intermediate level small enough: re-run accumap on it
            t0 = time.time()
            with level_scale(n, C_lev.shape[0]):
                Y = np.asarray(cm.reconstruct(sparsify(C_lev), Y, diam, shared=True, footprint=sq, verbose=False, device=device)["receivers"], float)
            if verbose:
                print(f"  level {lev}: {C_lev.shape[0]:,} nodes reconstructed ({time.time() - t0:.0f} s)", flush=True)
            if on_level is not None:
                on_level(lev, Y, levels)
    Y, n_reloc = relocate_passes(C, Y, sq, verbose)    # misplaced patches (beads) moved next to their well-placed partners
    return Y.astype(np.float32), {"coarse_nodes": int(Cc.shape[0]), "relocated": int(n_reloc)}


def relocate_passes(C, Y, footprint, verbose=True):
    """RELOCATE_PASSES relocation passes (RELOCATE_DIST per pass; footprint test in the first) for polygon footprints.
    Used at the end of the multilevel start and, optionally, on accumap's result. -> (Y, nodes flagged)"""
    n_reloc = 0
    if not RELOCATE_PASSES or footprint is None or isinstance(footprint, str):
        return Y, 0
    t0 = time.time(); keep = RELOCATE_DIST
    try:
        for k, dist in enumerate(RELOCATE_PASSES):
            globals()["RELOCATE_DIST"] = dist
            Y, nb = relocate(C, Y, footprint=footprint, verbose=verbose, outside=k == 0)
            n_reloc += nb
    finally:
        globals()["RELOCATE_DIST"] = keep
    if verbose:
        print(f"  relocation ({len(RELOCATE_PASSES)} passes): {time.time() - t0:.0f} s", flush=True)
    return Y, n_reloc


def coarse_layout(C, side=None, target=5000, seed=0, device="cuda:0", verbose=True, diam=None, footprint=None):
    lab, Cc = coarsen(C, target, seed, verbose)
    diam, sq = _footprint(side, diam, footprint)
    Cs = sparsify(Cc)
    Ec = cm.umap_layout(Cs, seed=seed, shared=True)
    with level_scale(C.shape[0], Cc.shape[0]):
        Rc = cm.reconstruct(Cs, Ec, diam, shared=True, footprint=sq, verbose=False, device=device)
    Y = np.asarray(Rc["receivers"], float)
    S = (C + C.T).tocsr(); S = S + sp.identity(S.shape[0], format="csr") * float(np.median(S.data))
    P = sp.csr_matrix((np.ones(len(lab)), (np.arange(len(lab)), lab)), shape=(len(lab), Cc.shape[0]))
    W = S @ P                                          # bead x super-bead count weights
    W = sp.diags(1 / np.asarray(W.sum(1)).ravel()) @ W
    return np.asarray(W @ Y, np.float32), {"coarse_nodes": int(Cc.shape[0]), "coarse_layout": Y, "labels": lab}


def joint_nodes(C, return_kept=False):
    """Distinct-bead counts (receivers x senders) -> shared-bead counts over joint nodes: node o = sender o plus the
    receivers whose strongest partner it is. Returns (receiver -> node labels, square node x node counts: summed
    counts of a node's receivers to each sender; within-node counts sit on the diagonal and are dropped as self pairs)."""
    C = C.tocsr()
    lab = _argmax_rows_nb(C.indptr.astype(np.int64), C.indices.astype(np.int64), C.data.astype(np.float64))
    ok = lab >= 0
    P = sp.csr_matrix((np.ones(ok.sum(), np.float32), (lab[ok], np.flatnonzero(ok))), shape=(C.shape[1], C.shape[0]))
    S = (P @ C).tocsr(); S.setdiag(0); S.eliminate_zeros()
    # drop isolated nodes (senders with no counts and no attached receivers): they never merge, survive to the coarsest
    # level (15% of its nodes at 12 mm) and come back unplaced
    live = (np.asarray(S.sum(1)).ravel() > 0) | (np.asarray(S.sum(0)).ravel() > 0)
    new_id = np.full(len(live), -1, np.int64); new_id[live] = np.arange(live.sum())
    S = S[live][:, live].tocsr()
    lab = np.where(lab >= 0, new_id[np.maximum(lab, 0)], -1)
    return (lab, S, np.flatnonzero(live)) if return_kept else (lab, S)


def multilevel_layout_distinct(C, diam, footprint=None, seed=0, device="cuda:0", verbose=True, **kw):
    """Multilevel start for distinct receivers and senders: joint nodes (joint_nodes), the shared-bead multilevel start
    on them, then each receiver at its node's position (receivers without counts at the array centre).
    -> (n_receiver, 2) receiver start in caller units, info."""
    t0 = time.time()
    lab, S = joint_nodes(C)
    if verbose:
        print(f"  joint nodes: {C.shape[0]:,} receivers + {C.shape[1]:,} senders -> {S.shape[0]:,} nodes, {S.nnz:,} nonzero ({time.time() - t0:.0f} s)", flush=True)
    Y, info = multilevel_layout(S, diam=diam, footprint=footprint, seed=seed, device=device, verbose=verbose,
                                n_beads=C.shape[0] + C.shape[1], **kw)
    E = np.zeros((C.shape[0], 2), np.float32); ok = lab >= 0
    E[ok] = Y[lab[ok]]
    info["joint_nodes"] = int(S.shape[0])
    return E, info
