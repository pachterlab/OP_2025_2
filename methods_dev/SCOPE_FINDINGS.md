# SCOPE (Srivatsan lab, GSE313591) vs UMAP vs accumap

SCOPE (https://elifesciences.org/reviewed-preprints/110436, "Optics-free
reconstruction of shapes, images and volumes with DNA barcode proximity
graphs") is the shared-bead modality (every barcode is both sender and
receiver in one square interaction matrix) with the deposited real data the
`shared-bead-simulation-todo` memory flagged as absent -- this is the first
real dataset for that modality. Setup lives on `atlas`
(`~/scope_recon/`, envs `ofst`/`scope`), not `bio` -- CPU only, no GPU needed.
Data: GEO GSE313591, 8 real bead-array samples (circle, asymmetric "swoosh",
Snellen eye chart, bear/star/butterfly/letter-E silicone-mold shapes).
Ground truth is the *known macroscopic shape*, not per-bead imaging like Hu
-- no exact coordinates are deposited, so scoring is shape-recovery
(qualitative), not posRMSE.

## Performance fixes to SCOPE's own `reconstruction.py`

Both validated bit-identical / behavior-identical before deploying.

**1. k-core: 1412x speedup (1643.8s -> 1.16s), zero result change.**
`process_dist()` rebuilt a `networkx.Graph` from the (dense, 44%-connected)
distance matrix and called `nx.k_core(G, 20)` -- a pure-Python per-edge
peeling algorithm that scales badly on a graph this dense (1.0M edges on
2297 nodes). Replaced with a vectorized scipy-sparse k-core peel operating
directly on the adjacency matrix. Validated against the real letter_E data
(pickled inputs, `kcore_validate.py`): identical surviving-node index set,
identical weighted submatrix (0 differing entries after reproducing
networkx's edge-weight tie-break for the asymmetric `dist_mat`), confirmed
again in the live pipeline (same downstream crash point, ~29min -> ~1s for
this step). This one line was >98% of a 29-minute pipeline stage.

**2. Delaunay crash on small arrays -- data-dependent bug, not perf.**
`knn_paga()` triangulates cluster-level PAGA positions to decide which
clusters to stitch together; with a small array like letter_E (2415 beads)
the Leiden clustering step produces only ~2 clusters, and
`scipy.spatial.Delaunay` needs >=4 points in 2D. Guarded: `if len(points) >=
4` uses the original Delaunay path unchanged; else falls back to a complete
graph over the (few) clusters, since there's no meaningful "nearby cluster"
topology to prune with only 1-3 of them anyway. Fixed downstream code
(spring layout, plotting) is unchanged and runs correctly off either path.

**Net effect:** full end-to-end reconstruction on letter_E, unpatched: never
completed (crashed ~29min in after the k-core step, at Delaunay). Patched:
**6m19s, zero crashes**, including the full 16-point UMAP min_dist/
repulsion_strength grid search and shape/image analysis. Patches are local
to `~/scope_recon/repos/SCOPE/reconstruction.py` on atlas (not upstreamed).

## letter_E comparison (2415 beads, ~1M interactions, 44% pairwise density)

Three pipelines, same input data:
- **UMAP**: cosine sim on log1p counts, standard embed.
- **accumap (1-pass self-consistent)**: no imaging truth and no assay sigma
  exist for this chemistry, so the fitted-curve approach (Hu-style) doesn't
  apply. Used the Route-2-style self-consistent monotone similarity->distance
  fit (`scope_bench.py`); anchored to a fixed reference scale (the UMAP
  init's own near-pair distance) to avoid the scale-runaway divergence a
  multi-outer-iteration version hit (stress rose monotonically 0.33->0.64
  across 6 outer iterations even after adding the anchor -- root cause not
  fully diagnosed, single-pass sidesteps it, consistent with this session's
  other findings that iterative self-consistent refitting is fragile).
- **SCOPE's own pipeline** (`reconstruction.py`, RF distance predictor +
  Leiden clustering + PAGA + per-cluster UMAP + boundary stitching).

Result: **UMAP and SCOPE's own pipeline independently agree** -- both
recover a boxy/L-shaped outline with two enclosed voids in similar relative
positions, despite unrelated methods. **accumap (self-consistent) is the
outlier**: collapses to a smooth, featureless disc, no internal structure.
SCOPE's own shape-legibility check (`binary_img_final.png`, one binary image
per grid point) does not recover a clean letter "E" at ANY of the 16
parameter settings either -- the "accepted" one is chosen by their own
heuristic (exactly one connected foreground object), not by resemblance to a
letter, so this is not itself strong evidence the true shape was recovered;
what IS strong evidence is the UMAP/SCOPE structural agreement.

This is the **third independent case this session** of the same pattern
(after Hu real data and the shared-bead simulation corruption sweeps):
accumap's self-consistent/fitted-curve reconstruction washes out real local
structure that topology-preserving embeddings (UMAP; here also SCOPE's own
RF+clustering+UMAP method) preserve, whenever there is no reliable external
distance calibration to anchor the similarity->distance map.

## Does the finding survive accumap's other fixes/alternates? (decon, Route 1)

Checked whether count-only decontamination or the shared-bead Route 1
(generative MAP, U=X, kernel fit jointly since SCOPE's real transfer is
inverse-square not Gaussian) change the verdict. Neither does.

- **Decontamination** (`scope_decon.py`, reused `decontaminate.py`'s EM
  unmodified -- it is row-wise and does not assume distinct sender/receiver
  structure): letter_E carries modest real contamination,
  `a_hat` mean 4.3% / median 1.6% / p90 10.5% (same order as Hu c58_8's 7%,
  smaller). "UMAP + decon" is visually near-identical to plain UMAP (same
  two-void shape) -- decontamination barely touches UMAP's output at this
  contamination level. **"accumap + decon" is STILL a featureless blob** --
  decontamination does not rescue the lost structure.
- **Route 1, shared-bead** (`scope_route1.py`, U=X so no joint
  sender-localization problem; kernel `K(d;lam,p)=1/(1+(d/lam)^2)^p` fit
  jointly with X via Adam/autodiff since no sigma exists for this chemistry
  -- converged `p~1.17`, consistent with SCOPE's reported inverse-square
  law): result is genuinely different from the self-consistent fit -- shows
  two denser internal clumps rather than a uniform disc, so it is not fully
  collapsing structure. But it does **not** reproduce the two clean voids
  UMAP/SCOPE agree on either; output is noisier/streakier with a few
  flyaway points. `lam` collapsed to 0.1 during fitting -- an unanchored
  absolute-scale degeneracy between `lam` and X's overall size (same failure
  family as the self-consistent loop's scale-runaway risk) -- so this may be
  an optimization/identifiability artifact rather than evidence the
  generative-model idea itself is wrong on this data.

**Verdict stands**: none of self-consistent / decontaminated / Route-1
accumap reproduces the UMAP/SCOPE structural agreement on letter_E.
6-way comparison: `results/letter_E_6way.png` (atlas).

## Follow-up: is it the kernel equation, or something deeper?

Tested directly: built Route 1 v2 (`scope_route1_v2.py`) to isolate "is the
right kernel shape enough" from the v1 scale-degeneracy artifact --
(1) scale anchor: rescale X back to a fixed reference RMS-from-centroid
after every Adam step, so `lam`/`p` can't just co-drift with `||X||` to
satisfy the same ratio; (2) background mixture
`prob = (1-c)*softmax(signal) + c*uniform`, a global contamination rate
fit jointly (same "ambient reads redistributed uniformly" model as
`decontaminate.py`, here fit rather than estimated separately).

Converged cleanly this time -- monotonic loss throughout, stable end state
(`lam=0.10, p=1.255, c=0.029`; c is in the same ballpark as the count-only
decontamination estimate of 4.3%). **But the resulting geometry is not
meaningfully closer to the UMAP/SCOPE structure** -- still two-ish density
clumps in the core (similar to v1), plus now a diffuse halo of scattered
outlier points at the edges (beads whose reads the background term explains
well enough that the optimizer doesn't need their position to be
geometrically meaningful).

**Conclusion: fixing the optimization pathology did not fix the
reconstruction.** This rules out "unstable/degenerate fit" as the
explanation and points at something structural: a single deterministic
map from counts to geometry -- however well-specified the kernel, however
cleanly it converges -- has to get every pairwise distance right, while
UMAP only needs each bead's local neighbor *ranking* to survive. That
asymmetry, not the choice of kernel equation, looks like the real reason
UMAP (and SCOPE's own clustering+UMAP-refinement pipeline) succeeds where
every accumap-family method tried here does not. Matches the Hu real-data
finding (S carries ordinal information, not reliable metric information).
Comparison: `results/letter_E_route1_compare.png` (atlas).

## Reversal: hard k-NN restriction (not soft reweighting) closes the gap

Every accumap variant above -- self-consistent isotonic, decontaminated,
Route 1 v1/v2 -- still assigns a metric-distance target and nonzero
influence to EVERY pair, just downweighted for uncertain ones (S itself as
weight, or `W = 1/(Var[resid|S]+delta)` IRLS reweighting). That is not the
same thing UMAP does: UMAP builds its graph from each bead's literal k
nearest neighbors and *discards every other pair completely* -- no
distance target, zero influence, not just a small one.

Tested the hard version directly (`scope_knn_accumap.py`): build a
symmetric (union) k-NN adjacency mask from S (k=15/25/50, ~1-3% of all
pairs kept), fit the isotonic `g(s)->d` map using ONLY the kept pairs, set
weight to exactly **zero** outside the graph (not small -- zero), single
weighted-MDS solve from the UMAP init.

**Result: kNN-accumap (k=15, k=25) visibly recovers the same two-void
boxy topology as UMAP and SCOPE's own pipeline** -- the first accumap
variant tried on this data to do so. k=50 is visibly worse (voids less
distinct) as more far/noisy pairs re-enter the graph -- consistent with the
mechanism, not a coincidence. `results/letter_E_knnaccu_compare.png`.

**Quantified** (`scope_quant_check.py`, kNN-k Jaccard distance to UMAP's own
neighbor graph, 0=identical/~1.0=random; all embeddings share the same bead
ordering so this is a valid per-bead comparison -- SCOPE's own output could
not be included, it carries no barcode labels to align its internal
ordering to ours):

| method | kNN10-Jac vs UMAP | kNN25-Jac vs UMAP |
|---|--:|--:|
| accumap (self-consistent) | 0.938 | 0.893 |
| accumap + decon | 0.929 | 0.883 |
| Route1 v1 | 0.764 | 0.657 |
| Route1 v2 | 0.759 | 0.653 |
| **kNN-accumap k=15** | **0.669** | **0.500** |
| kNN-accumap k=25 | 0.705 | 0.545 |
| kNN-accumap k=50 | 0.757 | 0.618 |
| *(same-method sanity check: UMAP+decon vs UMAP)* | *0.471* | *0.314* |
| *(random-permutation baseline)* | *0.997* | -- |

Ranking matches the visual: self-consistent/decontaminated ~= random ->
Route 1 (real but partial improvement) -> kNN-accumap k=15 (clearly best)
-> k=25 -> k=50 (degrading monotonically as k grows, exactly as the
mechanism predicts). **Calibration matters**: the same-method sanity check
(UMAP on decontaminated vs raw data -- same method, slightly different
input) only reaches 0.47/0.31, not 0 -- that is the realistic ceiling for
"good agreement" on this noisy real dataset. kNN-accumap k=15 at 0.67/0.50
falls short of that ceiling, so "recovers essentially the same structure"
(the initial visual read) overstates it -- the accurate statement is
**hard k-NN restriction closes most, not all, of the gap**, and is a real,
substantial improvement over every other accumap variant tried (roughly
half the remaining distance from random down to the same-method floor).
Also revises Route 1 upward: it looked like unstructured density clumps
visually, but at 0.76 (vs 0.997 random) it is capturing real, if partial,
local structure -- more than the visual comparison gave it credit for.

**This overturns the earlier framing.** The finding is not "accumap can't
recover this structure on real data" -- it's "accumap's *global, softly
reweighted* objective can't, but a *hard, locally-restricted* one can."
Soft downweighting reduces a far pair's influence but still (a) lets many
weak-weight far pairs add up across millions of pairs, and (b) still
assigns near pairs an explicit metric target from the same global
similarity->distance curve, which is no more reliable for being upweighted
if that curve is itself miscalibrated. Hard exclusion removes both
problems: far pairs contribute nothing, and the surviving local structure
is fit on its own terms, close to what UMAP already does structurally
(build a k-NN graph, lay it out) -- except now via accumap's own
weighted-MDS solver instead of UMAP's fuzzy-simplicial-set optimizer.

Practical upshot: for real (contaminated/heterogeneous/non-Gaussian-kernel)
count data, restricting accumap's own solver to a hard k-NN graph appears
to be the actual fix, not fitting a better similarity->distance curve or a
better generative model. Worth carrying back to the real Hu (distinct
sender/receiver) case too, and to the paper's discussion of when accumap
should be expected to work.

## Files
| path | what |
|---|---|
| `~/scope_recon/` (atlas) | data (`data/GSE313591/`), repos (`repos/SCOPE`, `repos/sci-space-v2`, `repos/accumap_rs`), envs `ofst`/`scope` |
| `scope_bench.py` (atlas, `~/scope_recon/`) | loader, cos_sim/UMAP, self-consistent accumap, evaluate() |
| `~/scope_recon/repos/SCOPE/reconstruction.py` | SCOPE's own pipeline, patched (k-core + Delaunay fixes, see above) |
| `~/scope_recon/repos/SCOPE/kcore_validate.py` | validation script proving the k-core replacement is bit-identical |
| `~/scope_recon/repos/SCOPE/results/letter_E/` | SCOPE's own reconstruction output (`final_solution.csv`, figures) |
| `~/scope_recon/results/letter_E_cache2.npz`, `letter_E_3way.png` | UMAP/accumap outputs + 3-way comparison figure |

## Not yet done
- Only letter_E (smallest sample) run end-to-end. butterfly/star/bear/asym/
  circ not yet attempted -- larger arrays should produce more Leiden
  clusters (>=4), so the Delaunay guard likely won't even trigger there;
  worth checking whether SCOPE's own pipeline holds up at that scale, and
  whether the UMAP/accumap gap seen here persists.
- No fixed random seed in SCOPE's own pipeline (RF, Leiden clustering) or in
  the accumap self-consistent fit here -- run-to-run variation not
  characterized.
- True ground-truth outline for letter_E (exact mold geometry) not obtained
  -- comparison is method-agreement, not verified-correct shape recovery.
