"""One-shot patcher: point the Fig 1 / Supp Fig panel cells at fig_style.py.

Rewrites the three plotting cells (make_summary_fig def, the Fig?b scatter, the
Fig?c Jaccard line plot) in each simulation notebook so they call the shared
helpers in fig_style.py (shared colour scale + one colour bar, larger fonts,
bold panel letters). Idempotent: matches cells by content, not index.

Run from analysis_scripts/:  python _patch_fig_cells.py
"""
import nbformat

# Prepended to every patched caller cell so each is self-sufficient regardless
# of notebook run order (the plotting cells do not all run consecutively).
_IMPORTS = (
    "import sys\n"
    "if '.' not in sys.path:\n"
    "    sys.path.insert(0, '.')\n"
    "from fig_style import make_summary_fig, fit_curve_panel, jaccard_panel  # shared panel styling\n"
)

IMPORT_CELL = (
    "# Shared panel-plotting helpers (see fig_style.py): single shared colour\n"
    "# scale + one colour bar, larger fonts, bold panel letters.\n"
) + _IMPORTS


def summary_caller(fig_name):
    return _IMPORTS + (
        "\n"
        "_umap_mse = new_error_df[new_error_df.method == 'UMAP Embedding']['error'].values\n"
        "_accu_mse = new_error_df[new_error_df.method == 'Distance Curve-Fit Accumap']['error'].values\n"
        "_med = new_error_df.groupby('method')['error'].median()\n\n"
        "p = make_summary_fig(\n"
        "    pd.DataFrame(embedding, columns=['xcoord', 'ycoord']),\n"
        "    fit_accumap,\n"
        "    _umap_mse, _accu_mse,\n"
        "    name1=f\"UMAP, median MSE: {int(_med['UMAP Embedding'])}\",\n"
        "    name2=f\"accumap, median MSE: {int(_med['Distance Curve-Fit Accumap'])}\",\n"
        "    cbar_label='MSE', log=False, panel_letter='A',   # set log=True for log10(MSE)\n"
        ")\n"
        f"p.savefig('../final_figures/{fig_name}a.png', dpi=200, bbox_inches='tight')\n"
        "plt.show()\n\n"
        "# --- reproducibility: numbers quoted in the manuscript ---\n"
        "print('MSE mean   :', new_error_df.groupby('method')['error'].mean().round(1).to_dict())\n"
        "print('MSE median :', new_error_df.groupby('method')['error'].median().round(1).to_dict())\n"
        "print('MSE max    :', new_error_df.groupby('method')['error'].max().round(1).to_dict())\n"
        "_u = new_error_df[new_error_df.method == 'UMAP Embedding'].reset_index(drop=True)['error'].values\n"
        "_c = new_error_df[new_error_df.method == 'Distance Curve-Fit Accumap'].reset_index(drop=True)['error'].values\n"
        "print(f'curve-fit accumap beats UMAP on {(_c < _u).mean():.1%} of beads')\n"
    )


def curve_caller(fig_name):
    return _IMPORTS + (
        "\n"
        "_xx = np.arange(0, 1, 0.01)\n"
        "fig_b = fit_curve_panel(test_df, _xx, diff_func(_xx, *popt), panel_letter='B')\n"
        f"fig_b.savefig('../final_figures/{fig_name}b.png', dpi=200, bbox_inches='tight')\n"
        "plt.show()\n"
    )


def jaccard_caller(fig_name, ylim):
    return _IMPORTS + (
        "\n"
        f"fig_c = jaccard_panel(new_median_nn, panel_letter='C', ylim={ylim})\n"
        f"fig_c.savefig('../final_figures/{fig_name}c.png', dpi=200, bbox_inches='tight')\n"
        "plt.show()\n\n"
        "# --- reproducibility: mean Jaccard distance by neighbourhood size X ---\n"
        "_j = new_median_nn.groupby(['method', 'nn'])['jac'].mean().unstack('method').round(3)\n"
        "print(_j.reindex([1, 5, 10, 15, 20]))\n"
        "print('mean over all X:', new_median_nn.groupby('method')['jac'].mean().round(3).to_dict())\n"
    )


# notebook -> (figure basename, Jaccard ylim upper)
NOTEBOOKS = {
    "New_Simulation.ipynb": ("Fig1", (-0.01, 0.5)),
    "New_Simulation_Old_Diffusion.ipynb": ("Supp1", (-0.01, 1.1)),
    "Original_Simulation.ipynb": ("Supp2", (-0.1, 1.1)),
}


def patch(path, fig_name, ylim):
    nb = nbformat.read(path, as_version=4)
    n_summary = n_curve = n_jac = n_umap = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source
        if "umap.UMAP(" in src and "random_state" not in src:
            # pin UMAP so the figures and quoted numbers are reproducible
            cell.source = src.replace("umap.UMAP(", "umap.UMAP(random_state=42, ", 1)
            cell.outputs = []
            cell.execution_count = None
            n_umap += 1
            continue
        if "def make_summary_fig" in src:
            cell.source = IMPORT_CELL
            cell.outputs = []
            cell.execution_count = None
        elif "make_summary_fig(" in src and "savefig" in src:
            cell.source = summary_caller(fig_name)
            cell.outputs = []
            cell.execution_count = None
            n_summary += 1
        elif "Curve To Fit" in src or ("diff_func(np.array" in src and "savefig" in src):
            cell.source = curve_caller(fig_name)
            cell.outputs = []
            cell.execution_count = None
            n_curve += 1
        elif "new_median_nn" in src and "sns.lineplot" in src and "savefig" in src:
            cell.source = jaccard_caller(fig_name, ylim)
            cell.outputs = []
            cell.execution_count = None
            n_jac += 1
    nbformat.write(nb, path)
    print(f"{path}: umap_seed={n_umap} summary={n_summary} curve={n_curve} jaccard={n_jac}")


if __name__ == "__main__":
    for nb_path, (fig, yl) in NOTEBOOKS.items():
        patch(nb_path, fig, yl)
