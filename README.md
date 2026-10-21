# OP_2025_2

Code for **accumap**, imaging-free reconstruction of bead positions in diffusion-based, optics-free spatial transcriptomics (Oakes & Pachter, *Computational array reconstruction with accumap for spatial transcriptomics*).

accumap models each receiver-by-sender count as Poisson, with the expected count set by bead capture, sender abundance and a transfer kernel of the distance between the two beads. Starting from a UMAP embedding, it fits the kernel and then warps and refines the layout to maximize the likelihood. Two physical priors supply what the counts leave undetermined: beads are deposited at uniform density, and contacts between neighboring beads have no preferred direction. Imaging is never used by the method; in the paper it is used only to score reconstructions.

## Repository layout

- **`src/accumap/`**: the installable package.
  - `count_model`: the method (`reconstruct`, `umap_layout`; diagnostics `contact_anisotropy`, `fit_kernel_at`).
  - `metrics`: the paper's evaluation (`evaluate`, `neighbors_kept`, `similarity_align`).
  - `realistic_sim`: the simulator calibrated to experimental data (`simulate_realistic`).
  - `real_data`: loads the Hu et al. ground-truth replicates (`load_hu`).
- **`analysis_scripts/`**: notebooks that walk through the paper and regenerate its figures and tables (below). `paper_results/` holds the cached results they read; `baselines/` holds the original method, similarity-weighted multidimensional scaling (similarity MDS), which the paper uses as a baseline and which is not part of the package.
- **`final_figures/`**: the paper's figures (`Fig#_final.png`, `FigS#_final.png`), as written by the notebooks.
- **`data/`**: input data (gitignored; see below).

## Installation

accumap needs Python 3.9 or later. It runs on CPU, but a CUDA GPU is strongly recommended.

**1. PyTorch (for GPU use).** `pip` installs the newest PyTorch build, which may require a newer NVIDIA driver than your
machine has; PyTorch then silently runs on the CPU. To use a GPU, first install the PyTorch build that matches your CUDA
driver, using the selector at https://pytorch.org/get-started/locally/ (check your driver's CUDA version with `nvidia-smi`).
Check with `python -c "import torch; print(torch.cuda.is_available())"`.

**2. accumap only:**

```
pip install git+https://github.com/pachterlab/OP_2025_2.git
```

**To reproduce the paper** (package, notebooks and their cached results):

```
git clone https://github.com/pachterlab/OP_2025_2.git
cd OP_2025_2
pip install -e .                    # accumap (installs PyTorch if not already installed)
pip install -r requirements.txt     # notebook dependencies
```

## Usage

```python
from accumap.count_model import reconstruct, umap_layout

# C: (n_receivers, n_senders) UMI counts, dense or scipy sparse
start = umap_layout(C, seed=0)
result = reconstruct(C, start, array_diameter=3000.0)   # the array's physical diameter sets the scale (here um)
receivers, senders = result["receivers"], result["senders"]
```

For arrays in which every bead both sends and receives, pass the square bead-by-bead matrix with `shared=True` (and `umap_layout(C, shared=True)`). `n_starts=3` runs several UMAP initializations and keeps the most likely layout. accumap warns if the counts indicate that transfer favors one direction; for such arrays use `iso_center=True` (see `SCOPE_Simulator.ipynb`). The `reconstruct` docstring documents every option.

## Reproducing the paper

Each notebook draws its figures and tables from the cached results in `analysis_scripts/paper_results/` and runs a small live example. Setting `RECOMPUTE = True` in a notebook reruns its analyses from scratch (CUDA GPU required) and writes to a separate folder, so the cache is never overwritten. Each notebook states its runtime and the data it needs.

| Notebook | Produces |
|---|---|
| `Method_Overview.ipynb` | Fig. 1 (the method, step by step on a toy array) |
| `Experimental_Ground_Truth.ipynb` | Fig. 2, Supp Figs S2–S3, Supp Tables S3–S4 |
| `Simulation_Benchmark.ipynb` | Supp Fig S1, Supp Tables S1–S2 |
| `Design_Choices.ipynb` | Supp Fig S4, Supp Table S5, the number of passes |
| `SCOPE_Simulator.ipynb` | the SCOPE results (Supplementary Note 1; Supp Table S2) |
| `Original_Method.ipynb` | Supp Fig S5 and Supp Table S6 (the original method, similarity MDS; Supplementary Note 3), and the reviewer-response Figs R1–R3 and Table R1 |

`Original_Method.ipynb` draws its figures and tables from cached results only; the runs behind them were development scripts.

## Data

The experimental ground truth (four replicates of Hu et al., 2025) is available from the Single Cell Portal study [SCP2577](https://singlecell.broadinstitute.org/single_cell/study/SCP2577/computational-reconstruction-for-imaging-free-spatial-transcriptomics). Place the files in `data/` or point `$ACCUMAP_DATA` at their folder. Cached results let every notebook draw its figures without the data. Concordex domain analyses also need the Slide-seq cerebellum section (`data/cerebellum_processed.h5ad`).

## License

See `LICENSE`.
