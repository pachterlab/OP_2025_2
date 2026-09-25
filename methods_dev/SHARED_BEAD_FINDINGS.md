# Shared-bead modality (every bead is BOTH sender and receiver)

The Abdulraouf 2024 / Liao 2024 setup, as opposed to the distinct
sender/receiver Slide-tag / Hu 2025 case the rest of `methods_dev/` targets.
No experimental ground-truth dataset exists for this modality yet, so this is
simulation-only (`shared_bead.py`).

Model: bead b at x_b in a uniform disc, R = 1500 um.
`lambda_bb' = q_b a_{b'} K_sigma(||x_b - x_b'||)`, self term dropped;
`C_{b.} ~ Multinomial(round(UMI q_b), lambda_{b.}/sum)`.  a_b, q_b log-normal
per-bead emission / capture.  UMI = 2500, matching the paper sims.

**Deployable imaging-free pipeline** (`accumap(Sinkhorn dist)`):
1. `ipf_normalize` -- Sinkhorn / IPF on `C + C.T` -> doubly-stochastic P
   (strips per-bead q_b, a_{b'}).
2. `sinkhorn_distance` -- invert the Gaussian kernel:
   `d = sigma sqrt(-2 log(P / P_ref))`, sigma from the assay FWHM, P_ref the
   global max off-diagonal.  Far pairs (P ~ 0) -> filler constant 3000.
3. accumap (Rust weighted-MDS), UMAP init, cosine-similarity weights S.

Metric: posRMSE / pairRMSE in um after full Procrustes; J@10 (0 = perfect,
~0.9 = random).  `accumap(perfect dist)` = solver ceiling (true distance
matrix in, same solver).

## Headline

On **clean, broad-kernel** data the imaging-free method **beats UMAP**
(posRMSE 48 vs 57, sigma = 300, B = 1500) -- unlike the distinct/real case
where nothing beats UMAP.  But it has one dominant failure mode plus the same
count-corruption axes as the real data, and **`accumap(perfect dist)` is
3-19 um in every single condition tested** -- the solver is never the
bottleneck; recovering a metric from the counts is.

## Failure mode 1 (dominant): kernel span, sigma / field radius

`sweep-sigma`, B chosen so local overlap (NN spacing / sigma) stays < 0.3
throughout -- so this is NOT a local-sampling problem, it is long-range.

| B / sigma | NN/sigma | frac pairs, no count signal | UMAP | accu(Sinkhorn) | perfect |
|---|--:|--:|--:|--:|--:|
| 8000 / 52   (~ New_Simulation) | 0.29 | 0.99 | **21** | 787 | 11 |
| 5000 / 100  | 0.19 | 0.96 | **33** | 745 | 19 |
| 3000 / 150  | 0.16 | 0.92 | **34** | 554 | 11 |
| 2000 / 220  | 0.14 | 0.84 | 47 | **81** | 14 |
| 1500 / 300  (~ Original config) | 0.12 | 0.74 | 57 | **48** | 5 |
| 1200 / 450  | 0.09 | 0.56 | 77 | **65** | 5 |

Crossover at **sigma ~ R/6-R/5** (fraction of pairs with any count signal
~ 0.8, `corr(Sinkhorn dist, true dist)` ~ 0.65).  When sigma is small relative
to the field, the count matrix carries **zero information about pairs more than
~3 sigma apart**; those unmeasured long-range distances cannot be recovered
from the noisy local graph well enough for a global metric embedding, and the
imaging-free method is 15-40x worse than UMAP.  This is exactly the regime of
the paper's flagship New_Simulation (sigma = 52).  UMAP's local-topology-only
layout is immune -- it never tries to place far beads at a metric distance.

`classical-MDS(Sinkhorn dist)` and `accumap(geodesic-filled dist)` were also
tested: both fail across the board (posRMSE 280-1200) -- a full-metric MDS on a
mostly-filler or geodesic-accumulated distance matrix never competes.

## Failure modes 2-5: counts -> distance corruption

All at the favourable regime (sigma = 300, B = 1500, heterogeneity CV = 0.5),
so kernel span is not the confound.  Every row: `accumap(perfect dist)` stays
2-7 um, i.e. the corruption breaks the Sinkhorn distance recovery, not the
solver.

### Barcode crosstalk (fraction mu of reads randomly reassigned)
| mu | UMAP | accu(Sinkhorn) | perfect |
|--:|--:|--:|--:|
| 0.00 | 57 | **48** | 5 |
| 0.02 | 59 | 56 | 6 |
| 0.05 | 62 | 73 | 4 |
| 0.10 | 64 | 111 | 2 |
| 0.20 | 70 | 197 | 2 |
Crosses UMAP at **mu ~ 3-5%**, 4x worse by mu = 10%.  Same threshold as the
distinct case.  Earliest / worst of the corruption axes.

### Count overdispersion (Poisson -> NB, size r)
| r | UMAP | accu(Sinkhorn) | perfect |
|--:|--:|--:|--:|
| inf | 57 | **40** | 7 |
| 1.0 | 65 | **41** | 6 |
| 0.3 | 86 | **53** | 7 |
| 0.1 | 122 | **83** | 4 |
| 0.03 | 196 | 1091 (diverged) | 3 |
**Best tolerated.**  accumap stays ahead to r ~ 0.1; only at r ~ 0.03
(~30x Poisson variance) does the Sinkhorn inversion destabilise -- and UMAP is
already wrecked there.  `corr(Sinkhorn dist, true)` for near pairs collapses
0.61 -> 0.28 -> 0.11 -> 0.065 as r drops: the mechanism is noise in the
inversion, not the geometry.

### Per-bead heterogeneity (emission + capture log-normal CV; Sinkhorn strips the mean)
| CV | UMAP | accu(Sinkhorn) | perfect |
|--:|--:|--:|--:|
| 0.0 | 54 | **41** | 4 |
| 0.5 | 57 | **48** | 5 |
| 1.0 | 68 | 74 | 7 |
| 1.5 | 87 | 133 | 6 |
Sinkhorn removes the mean per-bead scale but not its interaction with sampling
noise -- low-emission beads get sparse, unrecoverable rows.  Breaks past
**CV ~ 0.8-1.0**.  Real Hu data had sd(log abundance/capture) 0.5-1.4.

### Kernel shape misspecification (solver assumes Gaussian)
| kernel | UMAP | accu(Sinkhorn) | perfect |
|---|--:|--:|--:|
| gaussian | 57 | **48** | 5 |
| gen-Gaussian p=1.4 | 69 | 80 | 3 |
| gen-Gaussian p=1.0 | 94 | 154 | 2 |
| cauchy | 64 | 96 | 3 |
| edge-narrowing 0.3 | 58 | **34** | 14 |
The Gaussian inversion `d = sigma sqrt(-2 log P/P_ref)` is simply the wrong
formula for a heavier-tailed transfer; **p = 1.4 already loses to UMAP**.
Real Hu data was sub-Gaussian (gen-Gaussian p ~ 0.82).  Fixable in principle
by fitting the kernel form, but not done here.  (Edge-narrowing alone does not
hurt -- it is a mild per-bead sigma change Sinkhorn partly absorbs.)

## Comparison to the distinct sender/receiver (real-data) findings

- The shared-bead case is genuinely easier: there **is** a regime (broad sigma,
  clean counts) where an imaging-free method beats UMAP.  U = X removes the
  joint sender-localisation problem, and Sinkhorn strips the per-bead nuisance
  scales cleanly.
- It fails on the **same axes** as the real data, in the same order:
  crosstalk (mu ~ 3-5%) first, then kernel shape / heterogeneity, overdispersion
  last.
- Plus a new dominant axis -- **kernel span** -- that bites hardest exactly in
  the tight-sigma regime the paper's main simulation uses.
- In every single condition, `accumap(perfect dist)` reconstructs to 3-19 um.
  The reconstruction algorithm is never the limiting factor; turning
  noisy/contaminated/misspecified counts into a metric distance matrix is.

## Not yet run in this modality
Route 1 (generative MAP-EM, single X since U = X -- the memory's hypothesis
that it should approach the oracle) and Route 2 (self-consistent surrogate).
The `bench` run that includes them was killed to prioritise the failure-mode
sweeps; `shared_bead.py bench` runs all eight methods across the three regimes.

## Files
| script | what |
|---|---|
| `shared_bead.py` | simulator + Sinkhorn/geodesic distance recovery + benchmark; subcommands `smoke` / `bench` / `sweep-sigma` / `sweep-corrupt` |
| `sb_sigma.log` | kernel-span sweep |
| `sb_corrupt.log` | crosstalk / overdispersion / heterogeneity / kernel-shape sweeps |
