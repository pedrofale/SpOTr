"""Fig. S5 (fig_options), weight 0 on the dispersed clones (environment benchmark_env, CPU).

In each simulated tumor with one or two of its five clones dispersed over the section, and with heritable programs of strength 0.06 (as in the
benchmark) and 0.3, we fit SpOTr with the default clade weights and with weight 0 on the dispersed clones. (A, B) The Pearson correlation with
the truth of each dispersed clone in the tumors with one (A) and two (B) dispersed clones, under both settings side by side and joined by a line.
Open markers show a clone that remained partly clustered after the dispersal, with a clustering above 0.1. The clustering of a clone is the
fraction of the ten nearest tumor neighbors of its cells that belong to the clone, above the fraction of the clone in the tumor, and scaled to 1
for a fully clustered clone. (C) The same for the other clones of these tumors, whose weights we do not change.

Usage: python fig_weights.py   (reads config.yaml and the results of rule spotr_weights, and writes to common.paths.FIGURES)
"""
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
from common import paths, metrics, figlayout  # noqa: E402
import world  # noqa: E402
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

GRID, ORANGE, BROWN = "#e6e5e0", "#F47B20", "#7a3a10"


def clustering(xy, clone, k, neighbours=10):
    """Return the clustering of the clone in the tissue, which is 0 for a clone dispersed at random and 1 when the nearest tumor neighbors of every
    cell of the clone belong to the clone.
    """
    m = clone == k; s = m.mean(); nn = cKDTree(xy).query(xy[m], k=neighbours + 1)[1][:, 1:]
    return float((m[nn].mean() - s) / (1.0 - s))


def changes(d):
    """Return (dispersed, clustering, PCC with the default weights, PCC with weight 0 on the dispersed clones) for each scored clone."""
    z = np.load(d / "spotr_weights.npz"); q = np.load(d / "problem.npz"); s = np.load(d / "sim.npz"); K = int(q["K"]); tr = q["traced"].astype(bool)
    oh = np.eye(K)[q["clone"][tr]]; Ncl = q["Ncl"].astype(float); h = Ncl.sum(1) > 0; Q = Ncl[h] / Ncl[h].sum(1, keepdims=True)
    ks = [k for k in range(K) if Ncl[:, k].sum() >= metrics.MIN_CELLS and Q[:, k].std() > 0]
    a, b = (metrics.per_clone(z[f"phi_{arm}"].astype(float).T @ oh, Ncl)["PCC"] for arm in ("default", "weights"))
    disp = set(int(x) for x in z["dispersed"])
    return [(k in disp, clustering(s["xy"], s["clone"], k), x, y) for k, x, y in zip(ks, a, b)]


def main():
    cfg = yaml.safe_load(open(HERE.parent / "config.yaml")); root = HERE.parent / cfg["out"]
    sig, mig = sorted(cfg["weights"]["sigma_tree"]), min(cfg["weights"]["migrate"]); rows = {"one": [], "two": [], "other": []}
    for st, m, w, _ in world.weight_worlds(cfg):
        for s in cfg["seeds"]:
            for disp, c, x, y in changes(root / "worlds" / w / f"seed{s}"):
                rows[("one" if m == mig else "two") if disp else "other"].append((ORANGE if st == sig[0] else BROWN, disp and c > 0.1, x, y))
    fig, axs = plt.subplots(1, 3, figsize=(10.6, 3.8), sharey=True)
    for ax, key, title in ((axs[0], "one", "One clone dispersed"), (axs[1], "two", "Two clones dispersed"), (axs[2], "other", "Other clones")):
        for col, clustered, x, y in rows[key]:
            ax.plot([0, 1], [x, y], color=col, lw=1.2, alpha=0.45 if key == "other" else 0.8, ls=(0, (3, 2)) if clustered else "-", zorder=2)
            ax.scatter([0, 1], [x, y], s=36, marker="o", zorder=3, facecolor="white" if clustered else col, edgecolor=col, lw=1.2)
        ax.set_xticks([0, 1], ["default weights", "weight 0"]); ax.set_xlim(-0.35, 1.35); ax.set_title(title, fontsize=11)
        ax.grid(color=GRID, axis="y"); ax.set_axisbelow(True); [ax.spines[k].set_visible(False) for k in ("top", "right")]
    axs[0].set_ylabel("Clone PCC")
    fmt = lambda v: f"{v:g}"
    leg = axs[2].legend(handles=[Line2D([], [], ls="", marker="o", color=ORANGE, label=fmt(sig[0])), Line2D([], [], ls="", marker="o", color=BROWN, label=fmt(sig[1]))],
                        title="Heritable program\nstrength", frameon=False, fontsize=8, title_fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    leg._legend_box.align = "left"; axs[2].add_artist(leg)
    axs[2].legend(handles=[Line2D([], [], ls="", marker="o", markerfacecolor="white", markeredgecolor="#777777", label="partly clustered")],
                  frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 0.7))
    figlayout.finish(fig, [(axs[0], "A"), (axs[1], "B"), (axs[2], "C")])
    f = paths.figure("fig_options.pdf"); fig.savefig(f, bbox_inches="tight"); figlayout.compress_pdf(f); fig.savefig(f.with_suffix(".png"), dpi=110, bbox_inches="tight"); plt.close(fig)
    for key in ("one", "two"):
        d = np.array([(cl, x, y) for _, cl, x, y in rows[key]], float); r = d[d[:, 0] == 0]
        print(f"{key} dispersed, at random: improved {int((r[:, 2] > r[:, 1]).sum())} of {len(r)}; partly clustered: {d[d[:, 0] == 1][:, 1:].round(3).tolist()}")
    o = np.array([(x, y) for *_, x, y in rows["other"]]); print(f"other clones: {len(o)}, largest |change| {np.abs(o[:, 1] - o[:, 0]).max():.3f}")


if __name__ == "__main__":
    main()
