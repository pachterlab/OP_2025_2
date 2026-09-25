"""Execute a figure notebook headless, minus the Colab-only setup cells.

Drops cells that clone the repo / pip-install / build Rust (they assume a Colab
/content path and a fresh machine), then runs the rest with a long timeout.
Writes <name>.executed.ipynb next to the source and exits non-zero on error.

Usage (from analysis_scripts/, in an env with the deps installed):
    python _run_notebook.py New_Simulation.ipynb
"""
import sys
import nbformat
from nbclient import NotebookClient

SKIP_MARKERS = (
    "git clone",
    "/content/",
    "pip install",
    "maturin",
    "rustup.rs",
    "apt-get",
    "apt install",
)


def main(src):
    nb = nbformat.read(src, as_version=4)
    kept = []
    for cell in nb.cells:
        if cell.cell_type == "code" and any(m in cell.source for m in SKIP_MARKERS):
            continue
        kept.append(cell)
    dropped = len(nb.cells) - len(kept)
    nb.cells = kept
    print(f"{src}: running {len(kept)} cells ({dropped} setup cells skipped)")

    client = NotebookClient(
        nb, timeout=7200, kernel_name="python3",
        resources={"metadata": {"path": "."}},
    )
    client.execute()

    out = src.replace(".ipynb", ".executed.ipynb")
    nbformat.write(nb, out)
    print(f"OK -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
