"""Fig. S5 (fig_options): weight 0 on the dispersed clones (environment benchmark_env; CPU).

In each simulated tumor with one or two of its five clones dispersed over the section, with weak (the benchmark's) and with distinct clone
expression, SpOTr is fitted with the default clade weights and with weight 0 on the dispersed clones.  For each dispersed clone, the change in its
Pearson correlation with the truth against how clustered it is in the tissue: the share of each of its cells' ten nearest tumor neighbours that
belong to the clone, above the clone's share of the tumor and scaled to 1 for a fully clustered clone (0: dispersed at random).  The other
clones keep their weights; their change is printed.

Usage: python fig_weights.py   (reads config.yaml and the results of rule spotr_weights; writes to common.paths.FIGURES)
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
    """the clone's clustering in the tissue: 0 dispersed at random, 1 every cell's nearest tumor neighbours in the clone"""
    m = clone == k; s = m.mean(); nn = cKDTree(xy).query(xy[m], k=neighbours + 1)[1][:, 1:]
    return float((m[nn].mean() - s) / (1.0 - s))


def changes(d):
    """per scored clone: (dispersed, clustering, PCC with the default weights, PCC with weight 0 on the dispersed clones)"""
    z = np.load(d / "spotr_weights.npz"); q = np.load(d / "problem.npz"); s = np.load(d / "sim.npz"); K = int(q["K"]); tr = q["traced"].astype(bool)
    oh = np.eye(K)[q["clone"][tr]]; Ncl = q["Ncl"].astype(float); h = Ncl.sum(1) > 0; Q = Ncl[h] / Ncl[h].sum(1, keepdims=True)
    ks = [k for k in range(K) if Ncl[:, k].sum() >= metrics.MIN_CELLS and Q[:, k].std() > 0]
    a, b = (metrics.per_clone(z[f"phi_{arm}"].astype(float).T @ oh, Ncl)["PCC"] for arm in ("default", "weights"))
    disp = set(int(x) for x in z["dispersed"])
    return [(k in disp, clustering(s["xy"], s["clone"], k), x, y) for k, x, y in zip(ks, a, b)]


def main():
    cfg = yaml.safe_load(open(HERE.parent / "config.yaml")); root = HERE.parent / cfg["out"]; sig = sorted(cfg["weights"]["sigma_tree"])
    fig, ax = plt.subplots(figsize=(5.6, 4.0)); dd, other = [], []
    for st, m, w, _ in world.weight_worlds(cfg):
        for s in cfg["seeds"]:
            for disp, c, x, y in changes(root / "worlds" / w / f"seed{s}"):
                if disp:
                    dd.append((c, y - x)); col = ORANGE if st == sig[0] else BROWN; mk = "o" if m == min(cfg["weights"]["migrate"]) else "s"
                    ax.scatter([c], [y - x], s=40, color=col, marker=mk, zorder=3, edgecolor="white", lw=0.5)
                else:
                    other.append(y - x)
    dd = np.array(dd); ax.axhline(0, color="#9a9a9a", lw=1, ls=(0, (4, 3)))
    ax.set_xlabel("clustering of the dispersed clone in the tissue"); ax.set_ylabel("change in per-clone PCC"); ax.set_title("Zero weight on dispersed clones", fontsize=11)
    ax.legend(handles=[Line2D([], [], ls="", marker="o", color=ORANGE, label="weak clone expression"), Line2D([], [], ls="", marker="o", color=BROWN, label="distinct clone expression"),
                       Line2D([], [], ls="", marker="o", color="#777777", label="1 clone dispersed"), Line2D([], [], ls="", marker="s", color="#777777", label="2 clones dispersed")],
              frameon=False, fontsize=8, loc="lower left")
    ax.grid(color=GRID); ax.set_axisbelow(True); [ax.spines[k].set_visible(False) for k in ("top", "right")]
    f = paths.figure("fig_options.pdf"); fig.savefig(f, bbox_inches="tight"); figlayout.compress_pdf(f); fig.savefig(f.with_suffix(".png"), dpi=80, bbox_inches="tight"); plt.close(fig)
    other = np.array(other)
    print(f"dispersed clones: improved {int((dd[:, 1] > 0).sum())} of {len(dd)}, median change {np.median(dd[:, 1]):+.3f}")
    for c, v in sorted(dd.tolist()):
        print(f"   clustering {c:.2f}: {v:+.3f}")
    print(f"other clones: median change {np.median(other):+.3f}, largest |change| {np.abs(other).max():.3f}")


if __name__ == "__main__":
    main()
