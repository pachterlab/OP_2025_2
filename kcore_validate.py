"""Validate a vectorized k-core replacement for reconstruction.py's
process_dist() against the actual networkx implementation, using the real
pickled inputs from the letter_E run (no re-running the 29-min original
pipeline stage more than once).

nx.Graph edge-weight tie-break when dist_mat is asymmetric: edges are added
in row-major .nonzero() order (row i, then row j>i), and nx.Graph.add_edge
overwrites on duplicate (i,j) -- so for an unordered pair with BOTH A[i,j]
and A[j,i] present, the surviving weight is A[j,i] (from the higher-indexed
row, added later). Reproduced here as W[i,j] = A[max(i,j), min(i,j)] if
present else A[min(i,j), max(i,j)].
"""
import pickle
import time
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix

d = pickle.load(open("/home/coakes/scope_recon/repos/SCOPE/process_dist_inputs.pkl", "rb"))
A = d["dist_mat"]
n = A.shape[0]
K = 20

# ---------------- reference: original networkx approach -------------------
t0 = time.time()
rows, cols = A.nonzero()
weights = A.data
G = nx.Graph()
G.add_nodes_from(range(n))
edges = [(rows[i], cols[i], {"weight": weights[i]}) for i in range(len(weights))]
G.add_edges_from(edges)
t_build = time.time() - t0

t0 = time.time()
G_100 = nx.k_core(G, K)
t_kcore = time.time() - t0
ref_indices = np.array(sorted(G_100.nodes()))
ref_dist100 = nx.adjacency_matrix(G_100).astype(np.float32)
print(f"[reference] build={t_build:.1f}s k_core={t_kcore:.1f}s "
      f"survivors={len(ref_indices)}", flush=True)


# ---------------- vectorized replacement -----------------------------------
def vectorized_kcore(A, k):
    """Return sorted surviving-node indices for the k-core (unweighted
    degree, undirected via OR-symmetrization -- matches nx.Graph semantics
    when A is not perfectly symmetric)."""
    n = A.shape[0]
    Ab = (A != 0)
    B = (Ab + Ab.T) > 0          # boolean symmetric adjacency (structure only)
    B = csr_matrix(B, dtype=np.int8)
    B.setdiag(0)
    B.eliminate_zeros()
    alive = np.ones(n, dtype=bool)
    deg = np.asarray(B.sum(axis=1)).ravel()
    while True:
        dead_now = alive & (deg < k)
        if not dead_now.any():
            break
        idx = np.where(dead_now)[0]
        # degree contribution removed from each remaining neighbor of idx
        sub = B[idx][:, alive]
        deg[np.where(alive)[0]] -= np.asarray(sub.sum(axis=0)).ravel()
        alive[idx] = False
    return np.where(alive)[0]


def build_weighted_submatrix(A, survivors):
    """W[i,j] = A[max(i,j),min(i,j)] if present else A[min(i,j),max(i,j)] --
    reproduces nx.Graph's overwrite-on-duplicate-edge tie-break, restricted
    to the surviving node set, in survivor order."""
    sub = A[survivors][:, survivors].tocoo()
    subT = A.T.tocsr()[survivors][:, survivors].tocoo()
    # start from the "upper" (A[i,j]) values, then overwrite with "lower"
    # (A[j,i], i.e. subT's A[i,j]==orig A[j,i]) wherever present
    W = csr_matrix((sub.data, (sub.row, sub.col)), shape=sub.shape)
    Wt = csr_matrix((subT.data, (subT.row, subT.col)), shape=subT.shape)
    # tie-break: wherever Wt has an entry, it wins (came from higher row idx)
    W = W.tolil()
    r, c = Wt.nonzero()
    W[r, c] = Wt[r, c]
    return W.tocsr().astype(np.float32)


t0 = time.time()
fast_indices = vectorized_kcore(A, K)
t_fast = time.time() - t0
fast_dist100 = build_weighted_submatrix(A, fast_indices)
t_fast_total = time.time() - t0
print(f"[vectorized] total={t_fast_total:.3f}s (kcore peel {t_fast:.3f}s) "
      f"survivors={len(fast_indices)}", flush=True)

print("\n=== comparison ===")
print("index sets identical:", np.array_equal(ref_indices, fast_indices))
if not np.array_equal(ref_indices, fast_indices):
    print("  only in ref:", sorted(set(ref_indices) - set(fast_indices)))
    print("  only in fast:", sorted(set(fast_indices) - set(ref_indices)))

diff = (ref_dist100 - fast_dist100)
diff.eliminate_zeros()
print("weighted submatrix identical:", diff.nnz == 0)
if diff.nnz:
    print("  n_diff_entries:", diff.nnz, "max abs diff:", np.abs(diff.data).max())

print(f"\nSPEEDUP: {t_kcore / t_fast_total:.0f}x  ({t_kcore:.1f}s -> {t_fast_total:.3f}s)")
