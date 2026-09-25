"""Shared plotting helpers for the accumap figure notebooks.

Centralizes the Fig 1 / Supp Fig panels so that, per reviewer feedback:
  * the UMAP-vs-accumap scatters use a single shared colour scale and one
    colour bar (optionally on a log10 scale),
  * axis / colour-bar / title fonts are legible,
  * every panel carries a bold letter (A / B / C).

Used by New_Simulation.ipynb, New_Simulation_Old_Diffusion.ipynb and
Original_Simulation.ipynb.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Consistent method colours across every panel.
METHOD_PALETTE = {
    "UMAP Embedding": "crimson",
    "Perfect Information Accumap": "navy",
    "Cosine Similarity of Bead Counts": "limegreen",
    "Distance Curve-Fit Accumap": "royalblue",
}

_TITLE_FS = 13
_LABEL_FS = 12
_TICK_FS = 10
_LETTER_FS = 18


def add_panel_letter(fig, letter, x=0.01, y=0.99):
    """Stamp a bold panel letter in the top-left corner of a figure."""
    fig.text(x, y, letter, fontsize=_LETTER_FS, fontweight="bold",
             va="top", ha="left")


def make_summary_fig(old1, new1, color1, color2,
                     name1="", name2="", cbar_label="MSE",
                     log=False, panel_letter="A", cmap="viridis",
                     point_size=2, clip_percentile=(1, 99)):
    """Two bead-location scatters (UMAP vs accumap) sharing one colour scale.

    Parameters
    ----------
    old1, new1 : DataFrame with 'xcoord' / 'ycoord' columns.
    color1, color2 : per-bead scalar (e.g. MSE) for each embedding.
    name1, name2 : panel subtitles.
    cbar_label : label for the shared colour bar.
    log : colour by log10(value) instead of value.
    panel_letter : bold letter drawn in the corner.
    clip_percentile : pooled percentile range mapped onto the colour scale
        (keeps a few outliers from flattening the rest).
    """
    c1 = np.asarray(color1, dtype=float)
    c2 = np.asarray(color2, dtype=float)
    if log:
        pos = np.concatenate([c1[c1 > 0], c2[c2 > 0]])
        floor = pos.min() if pos.size else 1.0
        c1 = np.log10(np.clip(c1, floor, None))
        c2 = np.log10(np.clip(c2, floor, None))
        cbar_label = f"log10 {cbar_label}"

    pooled = np.concatenate([c1, c2])
    vmin, vmax = np.nanpercentile(pooled, clip_percentile)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6),
                                   sharex=True, sharey=True)
    im = None
    for ax, x, y, c, title in (
        (ax1, old1["xcoord"].values, old1["ycoord"].values, c1, name1),
        (ax2, new1["xcoord"].values, new1["ycoord"].values, c2, name2),
    ):
        im = ax.scatter(x, y, c=c, s=point_size, cmap=cmap,
                        vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=_TITLE_FS)
        ax.set_aspect("equal")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="both", which="major", labelsize=_TICK_FS)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.06, right=0.88, wspace=0.12)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=_TICK_FS)
    cbar.set_label(cbar_label, fontsize=_LABEL_FS)

    add_panel_letter(fig, panel_letter)
    return fig


def fit_curve_panel(test_df, curve_x, curve_y, panel_letter="B",
                    xlabel="Cosine similarity",
                    ylabel="Receiver bead-to-bead distance (µm)",
                    title="Distance vs. cosine similarity", logy=True):
    """Scatter of receiver--receiver cosine similarity vs distance, plus the
    fitted curve used to build the accumap distance matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(data=test_df, x="cosine similarity", y="distance",
                    alpha=0.35, s=12, edgecolor=None, ax=ax)
    ax.plot(curve_x, curve_y, color="black", lw=2)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=_LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=_LABEL_FS)
    ax.set_title(title, fontsize=_TITLE_FS)
    ax.tick_params(labelsize=_TICK_FS)
    fig.tight_layout()
    add_panel_letter(fig, panel_letter)
    return fig


def jaccard_panel(long_df, panel_letter="C", ylim=(-0.01, 0.5),
                  xlabel="Number of nearest neighbours (X)",
                  ylabel="Mean Jaccard distance"):
    """Mean Jaccard distance vs neighbourhood size X, one line per method.

    `long_df` has columns 'method', 'nn', 'jac' (per-bead rows); this averages
    over beads within each (method, nn).
    """
    summary = long_df.groupby(["method", "nn"]).mean(numeric_only=True).reset_index()
    present = list(summary["method"].unique())
    palette = {k: v for k, v in METHOD_PALETTE.items() if k in present}
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.lineplot(data=summary, x="nn", y="jac", hue="method", lw=2,
                 palette=palette or None, ax=ax)
    sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1.02, 1),
                    fontsize=_TICK_FS, title=None)
    ax.set_xlabel(xlabel, fontsize=_LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=_LABEL_FS)
    ax.tick_params(labelsize=_TICK_FS)
    ax.set_ylim(*ylim)
    fig.tight_layout()
    add_panel_letter(fig, panel_letter)
    return fig
