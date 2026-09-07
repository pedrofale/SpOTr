"""The phylogeny, and cell x cell tree vs space distance matrices."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list, cophenet
from scipy.spatial.distance import squareform, pdist
from scipy.stats import spearmanr, pearsonr

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d8d4"
RAMP = "Blues"                      # sequential: one hue, light -> dark
plt.rcParams.update({"axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
                     "axes.titlesize": 10.5})

d = np.load("petracer_xy.npz", allow_pickle=True)
clades, D = d["clades"].astype(str), np.asarray(d["D"], float)
K = len(clades)

# ---------------- figure A: the phylogeny ----------------
Z = linkage(squareform(D, checks=False), "average")
coph = pearsonr(cophenet(Z), squareform(D, checks=False))[0]
order = leaves_list(Z)

fig = plt.figure(figsize=(13.2, 5.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.26,
                      top=0.845, bottom=0.13, left=0.055, right=0.965)

ax = fig.add_subplot(gs[0, 0])
dendrogram(Z, labels=list(clades), ax=ax, color_threshold=0,
           above_threshold_color=INK2, leaf_font_size=10)
ax.set_ylabel("patristic distance between clades")
ax.set_title(f"Phylogeny of the 18 clades\naverage-linkage summary; cophenetic r = {coph:.3f}")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)

ax = fig.add_subplot(gs[0, 1])
Dm = D[np.ix_(order, order)]
im = ax.imshow(Dm, cmap=RAMP)
ax.set_xticks(range(K)); ax.set_xticklabels([clades[i] for i in order], fontsize=8)
ax.set_yticks(range(K)); ax.set_yticklabels([clades[i] for i in order], fontsize=8)
ax.set_title("The distances themselves (exact)\nclade × clade patristic distance")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=8)
off = D[np.triu_indices(K, 1)]
ax.set_xlabel(f"range {off.min():.2f}–{off.max():.2f}, median {np.median(off):.2f}", fontsize=8.5)
fig.suptitle("PEtracer M3: evolutionary distances between clades", fontsize=13.5, y=0.95, color=INK)
fig.savefig("petracer_phylogeny.png", dpi=155, facecolor="white")
print("wrote petracer_phylogeny.png")

# ---------------- figure B: cell x cell, tree vs space ----------------
SEC = "M3-4"
p = np.load(f"petracer_{SEC}.npz", allow_pickle=True)
CT = np.asarray(p["C_tree"], float)
xy = np.asarray(p["xy_sc"], float)
cc = np.asarray(p["cell_clade"]).astype(str)
cl = list(np.asarray(p["clades"]).astype(str))
phylo = [clades[i] for i in order]
key = {c: i for i, c in enumerate(phylo)}
idx = np.argsort([key.get(c, 99) for c in cc], kind="stable")
CT, xy, cc = CT[np.ix_(idx, idx)], xy[idx], cc[idx]
CS = squareform(pdist(xy))
bounds = np.cumsum([np.sum(cc == c) for c in phylo])[:-1]
n = len(cc)

fig = plt.figure(figsize=(15.4, 5.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.92], wspace=0.30,
                      top=0.83, bottom=0.13, left=0.045, right=0.975)
for k, (M, lab) in enumerate([(CT, "tree (patristic) distance"),
                              (CS, "spatial distance (μm)")]):
    ax = fig.add_subplot(gs[0, k])
    im = ax.imshow(M, cmap=RAMP, interpolation="nearest")
    for b in bounds:
        ax.axhline(b, color="white", lw=0.5); ax.axvline(b, color="white", lw=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"cell × cell {lab}\n{n} cells, ordered by clade (phylogenetic order)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=8)

ax = fig.add_subplot(gs[0, 2])
iu = np.triu_indices(n, 1)
same = cc[iu[0]] == cc[iu[1]]
v = CT[iu]
bins = np.linspace(v.min(), v.max(), 70)
ax.hist(v[same], bins=bins, color="#2a78d6", alpha=0.9, label="within clade", log=True)
ax.hist(v[~same], bins=bins, color="#eb6834", alpha=0.75, label="between clades", log=True)
frac_max = np.mean(v >= v.max() - 1e-9)
ax.annotate(f"{frac_max:.0%} of ALL pairs\nsit at exactly {v.max():.0f}",
            xy=(v.max(), 3e5), xytext=(-118, -6), textcoords="offset points",
            fontsize=9, color=INK,
            arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))
cv = v[~same].std() / v[~same].mean()
ax.set_xlabel("tree (patristic) distance"); ax.set_ylabel("cell pairs (log scale)")
ax.set_title(f"The metric is nearly binary\nbetween-clade CV = {cv:.3f}; overlap with within-clade ~0")
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle(f"PEtracer {SEC}: does tree distance predict spatial distance?",
             fontsize=13.5, y=0.945, color=INK)
fig.savefig("petracer_tree_vs_space.png", dpi=155, facecolor="white")
print("wrote petracer_tree_vs_space.png")

# ---------------- figure C: does the topology help, or only the clade label? ------
import pandas as pd
f = pd.read_csv("flat_tree_control.csv")
SECS3 = ["M3-1", "M3-2", "M3-4"]
TREES = [("real", "#2a78d6"), ("flat", "#eb6834"), ("binary", "#1baf7a"), ("shuffled", GRID)]
LAB = {"real": "real", "flat": "flat", "binary": "binary", "shuffled": "shuffled"}

FULL = {"real": "real tree", "flat": "flat: no between-clade topology",
        "binary": "binary: clade label only", "shuffled": "shuffled: identity destroyed"}
fig = plt.figure(figsize=(13.6, 4.9))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.15], wspace=0.30,
                      top=0.80, bottom=0.14, left=0.055, right=0.985)
for k, s_ in enumerate(SECS3):
    ax = fig.add_subplot(gs[0, k])
    g = f[f.section == s_]
    a0 = float(g[g.tree == "real"].set_index("alpha").soft[0.0])
    vals = [float(g[g.tree == tn].soft.max()) for tn, _ in TREES]
    ax.bar(range(4), vals, 0.66, color=[c for _, c in TREES], edgecolor="white", linewidth=1.4)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=8.5, color=INK)
    ax.axhline(a0, color=INK2, ls=":", lw=1.4)
    ax.text(3.45, a0 - 0.018, r"no tree ($\alpha$=0)", ha="right", fontsize=8, color=INK2)
    ax.set_xticks(range(4)); ax.set_xticklabels([LAB[t] for t, _ in TREES], fontsize=9)
    ax.set_ylim(0, 0.36); ax.set_title(s_, fontsize=11.5, color=INK)
    if k == 0: ax.set_ylabel("best accuracy over the $\\alpha$ sweep")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

ax = fig.add_subplot(gs[0, 3])
for tn, c in TREES:
    m = [float(f[(f.section == s_) & (f.tree == tn)].soft.max()) for s_ in SECS3]
    ax.plot(range(3), m, marker="o", ms=8, lw=2, c=c, label=FULL[tn],
            markeredgecolor="white", markeredgewidth=1)
ax.set_xticks(range(3)); ax.set_xticklabels(SECS3)
ax.set_ylabel("best accuracy"); ax.set_ylim(0, 0.36)
ax.set_title("The sections disagree", fontsize=11.5, color=INK)
ax.legend(frameon=False, fontsize=7.6, loc="upper left")
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Is claim (a) about the phylogeny, or only about clade membership?",
             fontsize=13.5, y=0.935, color=INK)
fig.savefig("petracer_flat_tree_control.png", dpi=155, facecolor="white")
print("wrote petracer_flat_tree_control.png")
