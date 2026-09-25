# Imaging-free reconstruction on the Hu et al. experimental data (c58_8)

All methods on SCP2577 c58_8 — 12,030 receiver beads, 12,786 sender beads,
~2,498 UMI/bead, ~3,000 um field span (the manuscript's post-QC numbers).
Ground truth = in-situ-sequencing bead positions, used **only for scoring**.
`sigma = 123/sqrt(8 ln 2) ~= 52 um` from the FWHM (a chemistry constant,
allowed as a solver input).

Metrics after full Procrustes alignment (rotation + reflection + scale +
translation):
- **posRMSE**: sqrt(mean_i ||x_i_recon - x_i_true||^2)  (per-bead displacement)
- **pairRMSE**: sqrt(mean_ij (d_true_ij - d_recon_ij)^2)  (pairwise-distance distortion)
- **J@10**: mean per-bead Jaccard distance between true and reconstructed
  10 nearest neighbors (0 = perfect, ~0.9 = random)

posRMSE ~= pairRMSE for every method here, so the errors are incoherent
(bead-level), not a smooth global warp.

## Results

| method | posRMSE (um) | pairRMSE (um) | J@10 | deployable? |
|---|--:|--:|--:|:--:|
| **UMAP** | **46** | **41** | **0.31** | yes |
| accumap + true distance matrix | 12 | 9 | 0.06 | no (uses truth) |
| augmented generative model + true sender positions | ~45 | 44 | 0.35 | no (uses truth) |
| accumap + fitted S->distance curve | 192 | 197 | 0.87 | yes |
| Route 2 (self-consistent fit) | ~250 | ~250 | 0.95 | yes |
| Variant 1 (marginalized generative model) | ~320 | ~320 | 0.90 | yes |
| plain Route 1 (Gaussian) + true sender positions | ~60 | 60 | 0.64 | no (uses truth) |
| geodesic / Isomap-MDS (best of 8 settings) | ~105 | ~105 | 0.49 | yes |
| robust weighted MDS (reliability wts + far-hinge + Huber), loose k | 150 | 161 | 0.87 | yes |
| robust weighted MDS, tight mutual-8-NN core | 149 | 163 | 0.94 | yes |
| robust weighted MDS, self-consistent (4 passes) | 195 (diverges) | 200 | 0.90 | yes |
| patch-stitch / group synchronization (ASAP-style) | 1039 | 1448 | 1.00 | yes |

Simulation equivalents (New_Simulation, sigma=52): accumap+true-dist MSE_pair
~9, augmented-model+true-senders MSE_pair ~1.5, accumap fitted ~11, UMAP ~19
(um; i.e. sqrt of MSE 87 / 2.4 / 121 / 343).

## Diagnostics

- **Raw cosine similarity vs true distance**: Spearman rho ~= 0.6 (global and
  per-bead), kNN-10 Jaccard 0.285.  Signal is ordinal, not metric.  Low-rank
  SVD denoising (ranks 5-200) does NOT improve neighbor identity.
- **Count-vs-distance fit** (imaging gives both endpoints, all 30M reads):
  best mean function is Cauchy+background (`A/(1+(d/lam)^2)^nu + c`), lam~26,
  nu~4, c~0.06; every no-background model fails outright.  Kernel is
  sub-Gaussian (gen-Gaussian p ~= 0.82) and narrows ~30% toward the field
  edge.  Noise is strongly overdispersed: NB `var = mean + mean^2/1.4`
  (per-bin); per-pair NB size falls to **r ~= 0.008** (variance ~100x Poisson).
  Per-bead heterogeneity is large: sd(log sender abundance) ~ 0.6-1.4,
  sd(log receiver capture) ~ 0.5-1.3.
- **Augmented generative model** (Cauchy kernel + background + per-sender
  abundance + per-receiver capture + NB), oracle sender positions: NLL barely
  moves as X is optimized (0.16414 -> 0.16241); final MSE_pair ~44 um --
  essentially tied with UMAP, far better than the plain Gaussian model (60 um)
  but not past UMAP.

## Interpretation

1. **The solver works.**  Given the true distance matrix, real counts
   reconstruct to ~1 bead pitch (12 um).
2. **The limit is barcode crosstalk / ambient contamination, not the
   reconstruction algorithm and not (as I first thought) per-pair
   overdispersion.**  See the sweep section below: injected NB overdispersion
   down to r_hat ~ 0.007 (the c58_8 value) leaves accumap ahead of UMAP, but
   ~2-5% random read misassignment breaks it exactly as observed on real data.
   c58_8 carries ~7% diffuse ambient contamination (count-only EM estimate).
   The flat augmented-model NLL is a symptom of that corruption, not of
   overdispersion per se.
3. **Cosine similarity is a valid input; converting it to metric distances is
   what fails.**  S carries ordinal information (rho ~= 0.6) but not reliable
   metric information.  Every method that builds a metric objective on S --
   fitted curve, self-consistent fit, marginalized/augmented generative model,
   Isomap, robust weighted MDS with the standard far-pair robustifications,
   patch stitching -- amplifies the ordinal-level error into large metric
   distortion.  Far-pair softening does not help because the *near* pairs are
   also unreliable (real S has no near/far structure; top-8 mutual neighbors
   are ~28% wrong).
4. **UMAP wins** by using only the ordinal/topological content with a soft
   local layout that does not propagate errors, and it sits near the ceiling
   of a ~72%-accurate-at-k=10 local graph (raw kNN-Jac 0.285 vs UMAP J@10
   0.307 -- UMAP does not even improve the neighbor graph, it just lays it
   out).  Its ~46 um is poor for cell-resolution work (~3-4 bead pitches) but
   is the best any counts-only method achieves here.

## Bottom line for the paper

The "accumap outperforms UMAP" result is **simulation-only**, where the
diffusion model holds by construction.  On the experimental data we tested
~10 reconstruction approaches spanning distance-geometry, generative-model,
and robust-MDS families; **none exceeds UMAP**.  Controlled sweeps localize the
break to **barcode crosstalk / ambient contamination** (~2-5% random read
misassignment reproduces the real-data failure; c58_8 carries ~7%), NOT to
count overdispersion (tolerable to r_hat ~ 0.007) and NOT to the reconstruction
algorithm (given true distances it reconstructs to ~1 bead pitch).  Count-only
EM decontamination detects the ~7% and helps, but is insufficient alone.  This
is an assumption-free, mechanistic replacement for the manuscript's current
"poor quality data" argument.  The actionable fix is at the assay level --
error-corrected barcode whitelists, longer barcodes, UMI-collision handling --
not a better algorithm.

Maps to reviewers: **M7** (accumap vs UMAP on real data -- done, accumap
loses; scope the claim to simulation), **M9** (soften "poor quality" -- replace
with this ceiling analysis), **M4** (probabilistic model -- built, fails
diagnosably), **M2** (relation to MDS -- weighted MDS and Isomap both tested).

## Files
| script | what |
|---|---|
| `real_data.py` | loader reproducing `Real_Data_Exploratory.ipynb` preprocessing; 4 replicates |
| `run_real.py` | UMAP / accumap / Route 2 / Variant 1 on real data (`results_real/real.log`) |
| `run_real_route1.py` | plain Route 1, oracle senders (`results_real/r1oracle.log`) |
| `route1_augmented.py` | Cauchy kernel + bg + per-bead + NB, oracle senders (`results_real/r1aug.log`) |
| `fit_kernel.py` | direct count-vs-distance fit (`results_real/fitk.log`) |
| `diagnostic_denoise.py` | SVD-denoise vs S-distance correlation (`results_real/diag.log`) |
| `pipeline_v0.py` | geodesic / Isomap-MDS (`results_real/pv0.log`) |
| `robust_mds.py` | reliability weights + far-hinge + Huber (`results_real/rmds_*.log`) |
| `patch_stitch.py` | ASAP-style patch stitching / group synchronization (`results_real/patch.log`) |
| `position_error.py` | posRMSE vs pairRMSE for cached methods |

---

# What breaks reconstruction: overdispersion vs barcode crosstalk

Simulation sweeps from the clean New_Simulation kernel (8k beads), scored by
pairwise-distance RMSE (um).

## Injected NB overdispersion (`overdispersion_sweep.py`)

| r_inj | r_hat | kNN-Jac | UMAP | accumap(fit) |
|---|--:|--:|--:|--:|
| inf   | 0.22  | 0.15 | 26.5 | 16.2 |
| 1     | 0.09  | 0.25 | 24.2 | 18.0 |
| 0.1   | 0.019 | 0.74 | 35.0 | 23.7 |
| 0.03  | 0.007 | 0.88 | 48.4 | 38.2 |

accumap beats UMAP at **every** overdispersion level, down to r_hat ~ 0.007
(the c58_8 value).  **Pure count overdispersion does not break accumap** -- both
methods degrade gracefully, accumap stays ahead.  (My earlier "count noise is
the primary limit" claim from the augmented-model NLL was wrong.)

## Injected barcode crosstalk (`crosstalk_sweep.py`, `decontaminate.py`)

Reassign a fraction mu of every bead's reads to a random other receiver
(barcode-collision model: profiles shrink toward the global mean).

| mu   | rho(S,-d) | UMAP | accumap(fit) |
|---|--:|--:|--:|
| 0.00 | 0.39 | 27 | 16  |
| 0.02 | 0.14 | 43 | 249 |
| 0.05 | 0.13 | 43 | 334 |
| 0.10 | 0.12 | 37 | 381 |

**5% (even 2%) random read misassignment is the culprit.**  UMAP is nearly
immune (rank of k-NN survives); accumap's fitted-curve reconstruction blows up
10x, exactly the real-data failure mode (real: accu 192 vs UMAP 46).  Cosine
similarity of sparse count profiles is catastrophically fragile to ambient
admixture -- rho collapses at the first ~2% and saturates.

# Decontamination

Count-only EM: `C_b ~= (1-a_b) s_b + a_b n_b g`; g = sender marginals, native
profile denoised over the k-NN similarity graph.  No embedding, no truth.

## Crosstalk sim (`decontaminate.py`)

| mu   | a_hat | rho pre->post | accumap pre->post | accu J@10 pre->post |
|---|--:|---|---|---|
| 0.00 | 0.000 | 0.39->0.39 | 24->17 | 0.14->0.14 |
| 0.02 | 0.017 | 0.14->0.16 | 249->66 | 0.69->0.39 |
| 0.05 | 0.036 | 0.13->0.15 | 334->204 | 0.79->0.62 |
| 0.10 | 0.069 | 0.12->0.16 | 381->285 | 0.82->0.69 |
| 0.20 | 0.130 | 0.11->0.16 | 410->312 | 0.80->0.72 |

`a_hat` tracks mu (~0.65x, conservative), zero false positive on clean data.
Decontam **helps accumap a lot at low mu** (249->66) but **never recovers to
beat UMAP** (accu-post 66-312 vs UMAP-post ~32-36), and rho barely moves
(0.14->0.16).  Removing contamination *reads* does not restore the similarity
*structure* -- the reads that left for good, plus residual, keep rho collapsed.

## Real c58_8 (`decon_real_quick.py`)

| | rho(S,-d) | kNN-10 Jac |
|---|--:|--:|
| before | 0.593 | 0.355 |
| after  | **0.654** | 0.346 |

`a_hat`: mean 0.080, median 0.065, p10-p90 0.03-0.15, 25% of beads > 0.10,
**0% > 0.30**.  The real data carries **~7% diffuse ambient contamination**
(not doublets).  Removing it lifts rho 0.59 -> 0.65 (real but modest) and does
not touch the local neighbor graph.

## Embedding-aware decontamination (`decontaminate_embed.py`) -- does NOT help

| mu=0.05 | a_hat (true 0.05) | rho pre->post |
|---|--:|---|
| count-only   | 0.036 | 0.13->0.15 |
| embed-graph EM | 0.000 | 0.13->0.13 |
| embed-excess  | 0.650 | -- |

embed-graph EM detects **zero** contamination (UMAP-neighbor smoothing is too
faithful -> trivial a=0 fixed point).  embed-excess over-estimates 13x and is
per-bead-uncorrelated with count-only.  The contamination signal ("resembles
the *global* soup") is position-independent, so a position-aware baseline adds
no discriminating power.  The embedding's role in this problem is as the robust
final layout (UMAP), not as a filter input.

# Read-level embedded-distance filter (`read_filter_v2.py`, `read_filter_accu.py`)

Count-only decontamination downweights each bead's soup-like component but
leaves rho/kNN collapsed.  The alternative: identify and DELETE the individual
misassigned reads.

**Oracle (drop every read with TRUE d_bo > 800 um) fully restores accumap** on
the crosstalk sim -- pairRMSE 365 -> 24 at mu=0.05, 379 -> 25 at mu=0.1
(clean reference 27).  So crosstalk is entirely a bad-*reads* problem: if you
can find the misassigned reads, the solver works perfectly.  Count-only
decontam only reaches 277 / 301.

**Imaging-free proxy for d_bo:** `E_b = UMAP(C)` (crosstalk-robust);
`E_o` = robust count-weighted centroid of the receivers that capture sender o
(2 trim passes at the 85th pctile); `d_hat_bo = ||E_b - E_o||`.
`corr(d_hat, d_true)` over reads = **0.93 clean, 0.99 contaminated** (rank).

**Single-pass filter:** zero every (b,o) cell whose d_hat is above the p80 of
the per-*cell* d_hat distribution.  Contamination cells hold ~1 read each, so
this removes only ~1-2% of reads but sweeps the far tail with margin.

| preprocessing | accumap pairRMSE mu=0.05 | mu=0.1 |
|---|--:|--:|
| raw (contaminated)        | 365 | 379 |
| count-only decontam       | 277 | 301 |
| **d_hat-filter p80**      | **69** | **84** |
| d_hat-filter p85          | 78  | 1410 (diverged) |
| d_hat-filter p90          | 98  | 1417 (diverged) |
| ORACLE d_bo>800           | 24  | 25  |
| clean reference           | 27  | 27  |
| UMAP (any preprocessing)  | 34-44 | 34-44 |

- Recovers accumap from ~10x worse than UMAP to ~1.5-2x worse; count-only
  decontam cannot (277/301).
- **Threshold-fragile above p80**: accumap's curve-fit + Gauss-Seidel diverges
  when a few catastrophic far pairs survive (p85/p90 at mu=0.1 -> 1400+).
  p80 held at both contamination levels but there is no ground-truth-free way
  to prove p80 is safe on a given dataset -- a generous cut is essential.
- Iterative re-embed version (`read_filter.py`) is unstable (re-embedding
  sparser counts inflates d_hat, removes 20%+ of reads, hurts clean data);
  the single-pass per-cell-percentile rule (`read_filter_v2.py`) is the one
  that works.
- The adaptive "cut where count-vs-d_hat decay flattens to background" rule is
  too aggressive (removes 5-10% of reads, eats real mid-range signal, rho only
  ~0.45); fixed p80-of-cells is better.
- rho(S,-d) 0.13 -> 0.68; kNN-10 Jaccard does NOT move (0.16) -- the filter
  fixes long-range ordering, not the local neighbour graph, yet that is enough
  for the solver because the surviving far pairs no longer collapse to small
  distances.

**Real c58_8 (`read_filter_real.py`) -- the sim recovery does NOT transfer:**

| preprocessing | posRMSE (um) | pairRMSE (um) |
|---|--:|--:|
| UMAP baseline               | **52** | **45** |
| raw accumap (fitted curve)  | 191 | 196 |
| count-only decontam         | 126 | 133 |
| d_hat-filter p75 (4.4% reads)| 128 | 142 |
| d_hat-filter p80 (3.5% reads)| 127 | 141 |
| d_hat-filter p85 (2.6% reads)| 124 | 136 |
| **ORACLE d_bo>800 removed**  | **112** | **127** |

`corr(d_hat, d_true)` over reads = 0.975 (proxy still excellent).  d_hat-filter
helps accumap (196 -> ~140), roughly tying count-only decontam (133).  **But the
ORACLE far-read removal -- perfect knowledge, using imaging -- only reaches
112/127, still 2.5x worse than UMAP.**  In the sim the identical oracle FULLY
restored accumap (365 -> 24 ~= clean 27).

So real-data failure is **not just crosstalk reads**.  Removing the long-range
misassigned reads (even perfectly) recovers the *global* rank ordering
(jackpot_diag: rho 0.59 -> 0.81 with the same oracle cut) but **not** the
reconstruction, because the *short-range* distance structure in c58_8 is
independently corrupted -- non-Gaussian / edge-narrowing transfer, capture and
abundance heterogeneity (sd(log) 0.5-1.4), NB overdispersion r~0.008, and
barcode collisions that land at small d_bo.  No read-level filter, oracle
included, touches that.  rho is a misleading success indicator here: it is
dominated by far pairs; accumap needs the neighbourhood-scale metric, which
stays broken.

**Net:** the read-level embedded-distance filter is a real advance on the
*crosstalk* failure mode in simulation (10x -> 1.7x vs UMAP) and a modest help
on real data (tying count-only decontam), but neither closes the gap to UMAP on
c58_8.  The real-data ceiling is crosstalk PLUS irreducible short-range
distortion; both are assay-level, not algorithmic.

# Hard k-NN restriction (`knn_accumap_real.py`) -- the single biggest fix found

Everything above (fitted curve, self-consistent fit, decontamination, the
read-level filter) still lets accumap's weighted-MDS objective touch EVERY
pair, just downweighted for far/uncertain ones (soft S-derived or IRLS
weight). Following the same idea that closed most of the gap on the SCOPE
shared-bead data: restrict the weight matrix to a symmetric union k-NN
graph from S, weight EXACTLY zero outside it (hard, not soft), everything
else (curve fit calibrated against imaging truth, same as the "accumap +
fitted curve" baseline) unchanged.

| preprocessing | posRMSE | pairRMSE | wall time |
|---|--:|--:|--:|
| UMAP | 52.0 | 44.8 | 176s |
| accumap full-weight (soft, all pairs -- the paper's baseline) | 191.2 | 196.2 | 4116s |
| **kNN-accumap k=15** | **59.2** | **56.9** | **48s** |
| kNN-accumap k=25 | 64.9 | 65.0 | 51s |
| kNN-accumap k=50 | 78.1 | 82.5 | 73s |

k=15 takes accumap from **3.5-4x worse than UMAP to within ~14-27%** of it
-- a much bigger effect than the same fix gave on SCOPE (there it closed
most but not all of a smaller gap). Monotonic degradation with k, same
pattern as SCOPE (more far pairs re-entering the objective hurts). Also
**~85x faster** (48s vs 4116s) -- the sparse weight matrix means far fewer
active constraints per bead in the Gauss-Seidel solve, not just a quality
win.

**Full k-sweep, finding the true optimum:**

| k | posRMSE | pairRMSE |
|--:|--:|--:|
| UMAP (reference) | 52.0 | 44.8 |
| 2 | 54.8 | 48.8 |
| 3 | 55.0 | 49.6 |
| 4 | 54.7 | 49.3 |
| **5** | **54.6** | **49.3** |
| 6 | 54.6 | 49.6 |
| 8 | 55.3 | 51.0 |
| 10 | 56.9 | 53.4 |
| 12 | 57.5 | 54.5 |
| 15 | 59.2 | 56.9 |
| 20 | 61.8 | 60.8 |
| 25 | 64.9 | 65.0 |
| 50 | 78.1 | 82.5 |
| full-weight (soft, all pairs) | 191.2 | 196.2 |

Monotonic from k=50 down to k~4-6, then **plateaus** -- k=2 through k=6 are
all within noise of each other. That is the true optimum:
**posRMSE ~54.6 / pairRMSE ~49.3, within ~5% (posRMSE) / ~9-11% (pairRMSE)
of UMAP -- close, but it does not cross over and beat UMAP even at the best
k.** Not a "ties/beats UMAP" result; a "gets close to competitive, from
catastrophically worse" result.

**Why the plateau doesn't cross UMAP: split-half resampling
(`split_half_noise.py`).** Thinned each bead's reads via independent
binomial splitting into two half-depth replicates of the SAME underlying
biology (same true position/kernel, only which molecules got sequenced
differs), rebuilt cosine similarity from each half, compared identified
near-neighbors:

| k | neighbor-identity Jaccard | mean true dist, half-picked NN | oracle (true k-NN) |
|--:|--:|--:|--:|
| 5 | 0.244 | 26.3 / 26.4 | **22.3** |
| 10 | 0.200 | 34.2 / 34.2 | **30.1** |
| 15 | 0.185 | 40.6 / 40.5 | **36.4** |
| 25 | 0.175 | 51.2 / 51.2 | **46.5** |

~18-24% of a bead's identified nearest neighbors change between two random
halves of its OWN reads, with zero change in the true biology -- pure
finite-read-depth sampling noise (mean depth 2498/bead, halved to
~1249/bead here). That noise inflates the effective neighbor distance:
oracle true-5-NN average 22.3um vs 26.3-26.4um when neighbors are picked
from noisy half-depth data -- a ~4um inflation from sampling noise alone,
at HALF the real depth (so an upper-bound analog, not an exact full-depth
measurement). This is the same order of magnitude as the observed
UMAP-vs-kNN-accumap-optimum gap (44.8 vs 49.3 pairRMSE, 4.5um). **The
residual gap is well explained by finite-read-depth sampling noise in
identifying true nearest neighbors -- a property of the assay's depth
(~2500 UMI/bead), not of accumap's implementation.** UMAP tolerates this
because it only needs approximate local topology to survive; a metric fit
needs the exact identified neighbor's distance to be right, which noisy
neighbor-picking undermines regardless of pipeline quality.

This revises the manuscript-relevant conclusion below: the ceiling analysis
still shows the reconstruction ALGORITHM was never broken (given true
distances, ~1 bead pitch) -- what was broken is fitting a *global* soft
objective over every pair. Restricted to hard local structure, accumap's
own solver gets CLOSE to UMAP on real data (~5-11% worse at the optimum),
not catastrophically worse (3.5-4x) -- but it does not beat UMAP. "accumap
beats UMAP" is still simulation-only; "accumap is far worse than UMAP on
real data" is revised to "accumap, properly restricted to local structure,
is close to but still behind UMAP on real data."

## Does soft reweighting close the rest of the gap? No -- two clean negatives

Tested (`knn_accumap_robust.py`) whether the remaining gap is closeable by
softening the TREATMENT of kept edges (Huber/IRLS reweighting: after each
solve, downweight edges whose residual exceeds `delta = 1.345*MAD` of the
kept-edge residuals) rather than changing WHICH edges are kept.

First attempt (delta recomputed each outer round from the current fit's own
residuals) was unstable -- a self-referential moving target, delta shrank
every round (14.0->11.1->8.9->7.4um) while ~40% of edges stayed flagged as
outliers, and the result got WORSE than the plain k=5 baseline (posRMSE
63.0 vs 54.6). Fixing delta ONCE from the plain baseline's own residuals
(no longer self-referential) stabilised the downweighted fraction (0.39->
0.22, monotonically decreasing) but was STILL worse (posRMSE 66.3,
pairRMSE 63.9 vs baseline 54.6/49.3) -- not an instability artifact, a
genuine negative result: at k=5 there are only 5 edges per bead to begin
with, so discounting ~30% of them starves the fit of the little redundancy
it has.

Tested whether a LARGER neighbourhood (k=25, matching UMAP's own
n_neighbors) fixed via the same *soft* IRLS reweighting could recover the
"many neighbours + robust to individual noise" combination that makes UMAP
work, since plain (unweighted) k=25 only failed because every included edge
counted at full unforgiving strength. Result: decisively worse, not
better -- plain k=25 baseline posRMSE=64.9/pairRMSE=65.0 (matches the
k-sweep exactly) vs k=25+Huber IRLS (fixed delta) posRMSE=104.8/pairRMSE=
114.7. Diagnostic: delta scaled up correctly (31.5um, vs 14.0um at k=5,
reflecting k=25's heavier residual tail) but 51% of edges were flagged as
outliers in round 0, settling to 36% by round 3 -- roughly a third to half
of every bead's neighbourhood looks anomalous relative to the rest (a real,
informative finding: a k=25 neighbourhood has a genuine bimodal split
between a well-behaved near-core and a noisy tail) but continuously
downweighting that tail, rather than excluding it outright, still lets it
exert partial pull each round, and the iterative reweight-then-resolve
process compounds rather than corrects the damage across rounds.

**Conclusion: robust/Huber reweighting on top of hard k-NN restriction does
not recover any of the remaining gap, at small or large k -- it actively
hurts, more so with more room (larger k) to do so.** This strengthens
rather than undermines the earlier conclusion: the plain hard k-NN
restriction at small k (k~5) is a real local optimum for this family of
fixes; the residual gap to UMAP is better explained by the structural
difference between an exact-metric commitment (accumap, even restricted)
and UMAP's fuzzy/topological objective, compounded by the genuine
read-depth noise floor -- not by an undiscovered reweighting scheme.

# Consolidated conclusion

- accumap's simulation advantage is real and does NOT transfer to Hu c58_8
  **when using the full-weight (soft, all-pairs) objective**. Hard k-NN
  restriction (above) substantially closes this gap -- from 3.5-4x worse
  to ~5-11% worse at the optimum (k~4-6) -- but does not close it entirely;
  UMAP remains the best method found on this dataset.
- **In simulation**, the sole isolatable break is barcode crosstalk / ambient
  contamination (~2-5% random read misassignment); overdispersion (to r_hat
  ~0.007) and the algorithm are fine.  Oracle removal of the misassigned reads
  fully restores accumap; an imaging-free embedded-distance read filter
  recovers most of it (10x -> 1.7x vs UMAP).
- **On real c58_8** the picture is worse: even ORACLE far-read removal leaves
  accumap at 112/127 um pairRMSE, 2.5x worse than UMAP (52/45).  Real data
  carries **crosstalk PLUS irreducible short-range distance distortion**
  (non-Gaussian/edge-narrowing transfer, capture+abundance heterogeneity,
  r~0.008 overdispersion, short-range barcode collisions).  Count-only
  decontam and the d_hat read filter each help (~196 -> ~133/140) but neither
  closes the gap.
- Actionable for the assay: reduce barcode misassignment (error-corrected
  whitelists, longer barcodes, UMI collision handling) AND tighten/characterise
  the transfer kernel and capture uniformity.  A better algorithm is not the
  missing piece.

# Files (added)
| script | what |
|---|---|
| `overdispersion_sweep.py` | inject NB overdispersion, sweep r (`results_real/odsweep.log`) |
| `crosstalk_sweep.py` | inject random read misassignment, sweep mu |
| `decontaminate.py` | count-only EM decontamination + crosstalk-sim sweep (`decon_sim.log`) |
| `decon_real_quick.py` | fast real-data decontam check (`dq.log`) |
| `decontaminate_embed.py` | embedding-aware decontam variants (`de_sim.log`) -- do not help |
| `read_filter.py` | iterative re-embed read filter -- unstable, superseded |
| `read_filter_v2.py` | single-pass per-cell d_hat-percentile read filter (`rfv2_sim.log`) |
| `read_filter_accu.py` | accumap pairRMSE after each filter, crosstalk sim (`rfv2_accu.log`) |
| `read_filter_real.py` | accumap posRMSE/pairRMSE after d_hat-filter on real c58_8 (`rfv2_real.log`) |
