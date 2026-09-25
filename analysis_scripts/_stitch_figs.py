"""Stitch panel PNGs into the composite figures the manuscript includes.

Layout matches FigSmasher3000.ipynb: panel A on top, panels B and C side by
side beneath it. Writes final_figures/<name>.png.

Usage (from analysis_scripts/):  python _stitch_figs.py Fig1 Supp1 Supp2
"""
import sys
import os
from PIL import Image

FIGDIR = "../final_figures"


def h_concat(ims, pad=0, bg=(255, 255, 255)):
    h = min(i.height for i in ims)
    ims = [i.resize((round(i.width * h / i.height), h), Image.LANCZOS) for i in ims]
    w = sum(i.width for i in ims) + pad * (len(ims) - 1)
    out = Image.new("RGB", (w, h), bg)
    x = 0
    for i in ims:
        out.paste(i, (x, 0))
        x += i.width + pad
    return out


def v_concat(ims, pad=0, bg=(255, 255, 255)):
    w = max(i.width for i in ims)
    ims = [i.resize((w, round(i.height * w / i.width)), Image.LANCZOS) for i in ims]
    h = sum(i.height for i in ims) + pad * (len(ims) - 1)
    out = Image.new("RGB", (w, h), bg)
    y = 0
    for i in ims:
        out.paste(i, (0, y))
        y += i.height + pad
    return out


def stitch(name):
    a = Image.open(os.path.join(FIGDIR, f"{name}a.png")).convert("RGB")
    b = Image.open(os.path.join(FIGDIR, f"{name}b.png")).convert("RGB")
    c = Image.open(os.path.join(FIGDIR, f"{name}c.png")).convert("RGB")
    row_bc = h_concat([b, c], pad=20)
    composite = v_concat([a, row_bc], pad=20)
    dest = os.path.join(FIGDIR, f"{name}.png")
    composite.save(dest, dpi=(200, 200))
    print(f"{dest}  {composite.size}")


if __name__ == "__main__":
    for n in sys.argv[1:]:
        stitch(n)
