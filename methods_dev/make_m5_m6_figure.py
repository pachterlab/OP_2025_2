"""Manuscript-ready figure for reviewer items M5 (UMAP-init sensitivity) and
M6 (error bars / significance across simulation replicates), matching the
established fig_style.py conventions (font sizes, bold panel letters,
UMAP=crimson / accumap=royalblue palette, spines removed top/right).
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TITLE_FS, LABEL_FS, TICK_FS, LETTER_FS = 13, 12, 10, 18
C_UMAP, C_ACCU, C_RAND, C_CMDS = "crimson", "royalblue", "darkorange", "seagreen"

HERE = os.path.dirname(__file__)
d = json.load(open(os.path.join(HERE, "results", "m5_m6_results.json")))
n = d["n_seeds"]
mse = {k: np.array(v) for k, v in d["mse_by_method"].items()}
xs = np.array(d["jac_xs"])
jac = d["jac_stats"]


def sig_stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def bracket(ax, x1, x2, y, h, text, fontsize=TICK_FS):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.1, c="black")
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=fontsize)


fig = plt.figure(figsize=(14, 4.6))
axA = fig.add_subplot(1, 3, 1)
axB = fig.add_subplot(1, 3, 2)
axC = fig.add_subplot(1, 3, 3)

# ---------------------------------------------------------------- Panel A --
# M6: UMAP vs accumap(UMAP init), mean +/- SEM, paired-test annotation
methA = ["UMAP", "accumap (UMAP init)"]
means = [mse[m].mean() for m in methA]
sems = [mse[m].std(ddof=1) / np.sqrt(n) for m in methA]
colsA = [C_UMAP, C_ACCU]
xpos = np.arange(2)
axA.bar(xpos, means, yerr=sems, capsize=5, color=colsA, width=0.6,
       error_kw=dict(lw=1.3))
axA.set_xticks(xpos)
axA.set_xticklabels(["UMAP", "accumap"], fontsize=LABEL_FS)
axA.set_ylabel("Mean per-bead MSE ($\\mu$m$^2$)", fontsize=LABEL_FS)
axA.tick_params(axis="both", labelsize=TICK_FS)
p = d["ttest_accumap_vs_umap"]["p"]
ytop = max(means[0] + sems[0], means[1] + sems[1])
bracket(axA, 0, 1, ytop * 1.08, ytop * 0.05,
       f"paired t-test\np = {p:.1e} ({sig_stars(p)})")
axA.set_ylim(0, ytop * 1.35)
axA.set_title(f"Simulation replicates (n={n})", fontsize=TITLE_FS)
axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)

# ---------------------------------------------------------------- Panel B --
# M6: Jaccard vs X, mean +/- std shading, UMAP + accumap(UMAP init) only
for m, c in [("UMAP", C_UMAP), ("accumap (UMAP init)", C_ACCU)]:
    mean = np.array(jac[m]["mean"]); std = np.array(jac[m]["std"])
    axB.plot(xs, mean, lw=2, color=c, label="UMAP" if m == "UMAP" else "accumap")
    axB.fill_between(xs, mean - std, mean + std, color=c, alpha=0.18)
axB.set_xlabel("Number of nearest neighbours (X)", fontsize=LABEL_FS)
axB.set_ylabel("Mean Jaccard distance", fontsize=LABEL_FS)
axB.tick_params(axis="both", labelsize=TICK_FS)
axB.legend(fontsize=TICK_FS, frameon=False)
axB.set_title(f"Mean $\\pm$ s.d., n={n} replicates", fontsize=TITLE_FS)
axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)

# ---------------------------------------------------------------- Panel C --
# M5: initialization sensitivity, log scale, all 4 conditions
methC = ["UMAP", "accumap (UMAP init)", "accumap (cMDS init)", "accumap (random init)"]
labelsC = ["UMAP", "accumap\n(UMAP init)", "accumap\n(cMDS init)", "accumap\n(random init)"]
colsC = [C_UMAP, C_ACCU, C_CMDS, C_RAND]
meansC = np.array([mse[m].mean() for m in methC])
semsC = np.array([mse[m].std(ddof=1) / np.sqrt(n) for m in methC])
xposC = np.arange(4)
axC.bar(xposC, meansC, yerr=semsC, capsize=5, color=colsC, width=0.6,
       error_kw=dict(lw=1.3))
axC.set_yscale("log")
axC.set_xticks(xposC)
axC.set_xticklabels(labelsC, fontsize=TICK_FS - 1)
axC.set_ylabel("Mean per-bead MSE ($\\mu$m$^2$, log scale)", fontsize=LABEL_FS)
axC.tick_params(axis="both", labelsize=TICK_FS)
p_rand = d["ttest_randinit_vs_umapinit"]["p"]
p_cmds = d["ttest_cmdsinit_vs_umapinit"]["p"]
axC.set_ylim(70, meansC.max() * 25)
for xi, mi, p in [(2, meansC[2], p_cmds), (3, meansC[3], p_rand)]:
    axC.annotate(f"p={p:.1e}\nvs UMAP-init", xy=(xi, mi), xytext=(0, 20),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=TICK_FS - 1)
axC.set_title("Initialization sensitivity", fontsize=TITLE_FS)
axC.spines["top"].set_visible(False); axC.spines["right"].set_visible(False)

for ax, letter in [(axA, "A"), (axB, "B"), (axC, "C")]:
    ax.text(-0.12, 1.08, letter, transform=ax.transAxes, fontsize=LETTER_FS,
           fontweight="bold", va="top", ha="left")

fig.tight_layout()
out = os.path.join(HERE, "results", "Fig_M5_M6.png")
fig.savefig(out, dpi=300)
print("saved", out)
