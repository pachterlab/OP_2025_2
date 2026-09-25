import sys
sys.path.insert(0, "/home/coakes/scope_recon")
import numpy as np
from scope_bench import load_scope

C, bc2idx = load_scope("letter_E")
B = C.shape[0]

row_sorted = -np.sort(-C, axis=1)
for k in (5, 10, 25, 50):
    frac = row_sorted[:, :k].sum(1) / np.maximum(C.sum(1), 1)
    print(f"  top-{k} neighbors carry {100*np.median(frac):.1f}% (median) of a bead's reads")

# how many neighbors needed to reach 50% / 90% of a bead's reads (typical)
csum = np.cumsum(row_sorted, axis=1) / np.maximum(C.sum(1, keepdims=True), 1)
n50 = (csum < 0.5).sum(1) + 1
n90 = (csum < 0.9).sum(1) + 1
print(f"\n  median # neighbors to reach 50% of reads: {np.median(n50):.0f}")
print(f"  median # neighbors to reach 90% of reads: {np.median(n90):.0f}  (out of {B-1} possible)")
