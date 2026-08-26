"""Evidence figure: does the phylogeny help map SS3 cells onto Visium spots?"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tree_evidence import clone_tree_distances  # noqa: E402

d = np.load(HERE / "bcsa3_B1_inputs.npz", allow_pickle=True)
xy, truth = d["xy"], d["spot_clone_numbat"]
nodes, Dtree = clone_tree_distances(HERE / "bcsa3_clone_edges.csv")
pos = {c: i for i, c in enumerate(nodes)}
keep = np.array([c in pos for c in truth])
xy, truth = xy[keep], truth[keep]
ti = np.array([pos[c] for c in truth])
rng = np.random.default_rng(0)

fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))

# --- A: tree distance vs spatial distance, binned ---
n = len(truth)
ii, jj = rng.integers(0, n, 400_000), rng.integers(0, n, 400_000)
ok = ii != jj; ii, jj = ii[ok], jj[ok]
dt, ds = Dtree[ti[ii], ti[jj]], np.linalg.norm(xy[ii] - xy[jj], axis=1)
bins = np.unique(dt)
mean = [ds[dt == b].mean() for b in bins]
sem = [ds[dt == b].std() / np.sqrt((dt == b).sum()) for b in bins]
axs[0].errorbar(bins, mean, yerr=sem, fmt="o-", color="steelblue", capsize=3)
axs[0].set_xlabel("phylogenetic distance between the spots' clones")
axs[0].set_ylabel("mean spatial distance (px)")
axs[0].set_title(f"Tree vs space agreement\nr = +0.024 (p=0.008, but $r^2$=0.06%)", fontsize=11)

# --- B: clone territories ---
cl = [c for c in np.unique(truth) if (truth == c).sum() >= 20]
cent = np.stack([xy[truth == c].mean(0) for c in cl])
Ds = np.linalg.norm(cent[:, None] - cent[None, :], axis=-1)
Dt = Dtree[np.ix_([pos[c] for c in cl], [pos[c] for c in cl])]
iu = np.triu_indices(len(cl), 1)
axs[1].scatter(Dt[iu], Ds[iu], s=40, color="darkorange", edgecolor="k", linewidth=0.4)
axs[1].set_xlabel("phylogenetic distance between clones")
axs[1].set_ylabel("distance between clone centroids (px)")
axs[1].set_title(f"Clone territories ({len(cl)} clones)\nMantel r = +0.278, p = 0.11 (n.s.)", fontsize=11)

# --- C: shuffled-tree ablation ---
labels = ["argmax acc", "ARI", "clone r"]
real = [0.1198, 0.0100, 0.0050]
shuf = [0.1052, 0.0146, 0.0243]
notree = [0.100, 0.024, 0.037]
x = np.arange(3); w = 0.26
axs[2].bar(x - w, real, w, label="real Numbat tree", color="seagreen")
axs[2].bar(x, shuf, w, label="shuffled tree (3 seeds)", color="silver")
axs[2].bar(x + w, notree, w, label=r"no tree ($\alpha$=0)", color="lightcoral")
axs[2].set_xticks(x); axs[2].set_xticklabels(labels)
axs[2].set_title("Ablation: the real tree does not beat\na randomly shuffled one", fontsize=11)
axs[2].legend(fontsize=8)

fig.suptitle("BCSA3 / B1 — is there evidence the phylogeny improves the mapping?", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = HERE / "bcsa3_tree_evidence.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
