"""Poster figures, sized for one column of the SciLifeLab 70 x 100 cm template (18.4 cm wide) with poster type
(2026-10-13): the planted-dispersal identifiability result, and SpOTr against Tangram on the paper's simulations
(sim_spotr_tangram.pdf, from spotr_out/sim_benchmark.json of 81_sim_benchmark.py).

Same data and arms as slide 3 of the deck (72_slide_figs.py, s3_identifiability): half of PEtracer M3-4's clades get
extra random dispersal sigma, spots are rebuilt from the moved cells, and the anchor (clade mix at 20% or 5% of the
spots) is asked which group is dispersed, through SpOTr's per-clade mixing length or through the earlier per-clade
weights alpha and theta.  Table: spotr_out/planted_identifiability.csv, written by 69_compare_figs.py from the
grids of 66_planted_control.py and 68_paper_formulation.py.
Usage: python 80_poster_figures.py [out.pdf] [--png]   (default: SpOTr-overleaf/poster/planted_identifiability.pdf)
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
OUT = Path(ARGS[0]) if ARGS else HERE.parents[2] / "SpOTr-overleaf" / "poster" / "planted_identifiability.pdf"
plt.rcParams.update({"font.family": "Liberation Sans", "mathtext.fontset": "cm", "pdf.fonttype": 42, "font.size": 18,
                     "axes.edgecolor": "#8a8984", "axes.linewidth": 1.2, "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a",
                     "axes.labelcolor": "#111111", "xtick.major.size": 6, "ytick.major.size": 6})
INK2, GRID = "#4a4a4a", "#e6e5e0"
R = pd.read_csv(HERE / "spotr_out" / "planted_identifiability.csv")
S = R.groupby(["arm", "sigma"]).recovered.mean().unstack(0)
fig, ax = plt.subplots(figsize=(18.37 / 2.54, 13.0 / 2.54))
for arm, col, lab, mk in [("Law: mixing length L", "#2a78d6", "SpOTr: per-clade mixing length", "o"),
                          ("Paper: tree weight α", "#eb6834", "FGW: per-clade tree weight α", "s"),
                          ("Paper: within-clade weight θ", "#8a8984", "FGW: within-clade weight θ", "D")]:
    ax.plot(S.index, 100 * S[arm], "-", color=col, lw=3.2, marker=mk, ms=12, label=lab, clip_on=False, zorder=3)
ax.set_xticks([0, 300, 500]); ax.set_xlim(-25, 525); ax.set_ylim(0, 100); ax.set_yticks([0, 25, 50, 75, 100])
ax.set_xlabel("planted extra dispersal, σ (µm)", fontsize=20)
ax.set_ylabel("dispersed clades found (%)", fontsize=20)
ax.grid(color=GRID, lw=1.0); ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(loc="upper left", fontsize=17, frameon=False, handlelength=1.6, borderaxespad=0.1, labelspacing=0.35)
ax.text(520, 33, "20 draws per point;\nties count as not found", ha="right", va="bottom", fontsize=15, color=INK2,
        style="italic", linespacing=1.25)
fig.tight_layout(pad=0.3)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
if "--png" in sys.argv:
    fig.savefig(OUT.with_suffix(".png"), dpi=200)
print("wrote", OUT, (100 * S).round(0).to_dict())


# ---------------------------------------------------------------- simulation: SpOTr against Tangram (81_sim_benchmark.py)
import json, numpy as np
SIM = json.load(open(HERE / "spotr_out" / "sim_benchmark.json"))
MODES = [("main_clades", "clones"), ("main_and_subclades", "clones +\nsubclones"), ("subclades", "subclones"),
         ("unstructured", "none")]
ARMS = [("spotr_select", "SpOTr", "#2a78d6"), ("tangram", "Tangram", "#6250d6")]
fig, ax = plt.subplots(figsize=(18.37 / 2.54, 11.5 / 2.54))
rng = np.random.default_rng(0)
for i, (mode, _) in enumerate(MODES):
    for j, (arm, lab, col) in enumerate(ARMS):
        v = np.array([r[arm]["pearson_heldout"] for r in SIM if r["mode"] == mode])
        x = i + (j - 0.5) * 0.36
        ax.boxplot([v], positions=[x], widths=0.28, patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor=col, alpha=0.25, edgecolor=col, lw=2), medianprops=dict(color=col, lw=3),
                   whiskerprops=dict(color=col, lw=2), capprops=dict(color=col, lw=2))
        ax.scatter(x + rng.uniform(-0.07, 0.07, len(v)), v, s=40, color=col, lw=0, zorder=3)
ax.set_xticks(range(len(MODES))); ax.set_xticklabels([m[1] for m in MODES], fontsize=17)
ax.set_xlabel("spatial structure of the simulated subclones", fontsize=19)
ax.set_ylabel("Pearson r", fontsize=19)
ax.set_ylim(0.7, 1.005); ax.set_yticks([0.7, 0.8, 0.9, 1.0])
ax.grid(axis="y", color=GRID, lw=1.0); ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.6, edgecolor=c) for _, _, c in ARMS],
          labels=[a[1] for a in ARMS], loc="lower left", fontsize=17, frameon=False, ncol=2, borderaxespad=0.1)
fig.tight_layout(pad=0.3)
OUT2 = OUT.parent / "sim_spotr_tangram.pdf"
fig.savefig(OUT2)
if "--png" in sys.argv:
    fig.savefig(OUT2.with_suffix(".png"), dpi=200)
print("wrote", OUT2)
