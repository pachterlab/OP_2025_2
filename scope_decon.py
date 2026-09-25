"""Apply count-only EM decontamination to the SCOPE letter_E data (shared-bead:
every bead is both sender and receiver, so the same decontaminate() model
applies row-wise without modification) and re-check whether it changes the
UMAP-vs-accumap structural-agreement finding.
"""
import sys
import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, "/home/coakes/scope_recon")
from scope_bench import load_scope, cos_sim, umap_embed, accumap_self_consistent


def decontaminate(C, n_iter=12, k=20, a_max=0.7, chunk=1000, verbose=True):
    C = np.asarray(C, dtype=np.float64)
    B, O = C.shape
    nb = C.sum(1)
    g = C.sum(0); g = g / g.sum()
    gO = g[None, :]

    S = cos_sim(C); np.fill_diagonal(S, 0.0)
    idx = np.argpartition(-S, k + 1, axis=1)[:, :k + 1]
    rows = np.repeat(np.arange(B), k + 1)
    A = csr_matrix((np.full(B * (k + 1), 1.0 / (k + 1)), (rows, idx.ravel())),
                   shape=(B, B))

    s = C / np.maximum(nb[:, None], 1.0)
    a = np.full(B, 0.05)
    s_sm = None
    for it in range(n_iter):
        s_sm = A @ s
        s_sm /= np.maximum(s_sm.sum(1, keepdims=True), 1e-12)
        contam = np.zeros(B)
        for i0 in range(0, B, chunk):
            sl = slice(i0, i0 + chunk)
            num = a[sl, None] * gO
            den = (1 - a[sl, None]) * s_sm[sl] + num + 1e-12
            z = num / den
            contam[sl] = (C[sl] * z).sum(1)
            nat = C[sl] * (1 - z)
            s[sl] = nat / np.maximum(nat.sum(1, keepdims=True), 1e-12)
        a = np.clip(contam / np.maximum(nb, 1.0), 0.0, a_max)
        if verbose:
            print(f"    EM {it:2d}: mean a={a.mean():.3f} med a={np.median(a):.3f}", flush=True)

    Chat = np.empty_like(C)
    for i0 in range(0, B, chunk):
        sl = slice(i0, i0 + chunk)
        den = (1 - a[sl, None]) * s_sm[sl] + a[sl, None] * gO + 1e-12
        z = (a[sl, None] * gO) / den
        Chat[sl] = C[sl] * (1 - z)
    return Chat, a


if __name__ == "__main__":
    C, bc2idx = load_scope("letter_E")
    print(f"letter_E: {C.shape}, {int(C.sum()):,} reads", flush=True)

    Chat, a = decontaminate(C, n_iter=12)
    print(f"a_hat: mean={a.mean():.3f} median={np.median(a):.3f} "
          f"p10={np.percentile(a,10):.3f} p90={np.percentile(a,90):.3f} "
          f"frac>0.3={np.mean(a>0.3):.3f}", flush=True)

    E_umap = umap_embed(Chat)
    S = cos_sim(Chat); np.fill_diagonal(S, 0.0)
    E_accu = accumap_self_consistent(Chat, S, E_umap, n_outer=1)

    np.savez("/home/coakes/scope_recon/results/letter_E_decon_cache.npz",
            E_umap=E_umap, E_accu=E_accu, a=a)
    print("saved", flush=True)
