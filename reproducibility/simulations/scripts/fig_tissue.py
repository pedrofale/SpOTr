"""Fig. S2 (fig_sim_tissue), the example tumor in space (environment benchmark_env, CPU).

(A) Every cell of the section, with the tumor cells colored by clone and the non-tumor cells by type. (B) A window of the measured half across
the edge of the tumor, with the 55 um Visium spots over its cells.

Usage: python fig_tissue.py   (reads config.yaml and its results, and writes to common.paths.FIGURES)
"""
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
from common import paths, figlayout  # noqa: E402
import world  # noqa: E402
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

TYPE_COL = {1: "#d8cfbd", 2: "#8fa9c4", 3: "#c9908f"}


def main():
    import warnings; warnings.filterwarnings("ignore")
    from spotr.plotting import cells
    cfg = yaml.safe_load(open(HERE.parent / "config.yaml")); q = np.load(world.example(cfg, HERE.parent) / "problem.npz", allow_pickle=True)
    K = int(q["K"]); tab = plt.get_cmap("tab10"); ccol = [to_hex(tab(k)) for k in range(K - 1)] + ["#bdbdbd"]
    xy, ty, cl, sec = q["tissue_xy"].astype(float), q["tissue_type"], q["tissue_clone"], q["tissue_section"].astype(bool); tum = ty == 0
    col = np.array([ccol[c] if t == 0 else TYPE_COL[t] for c, t in zip(cl, ty)], dtype=object); o = np.argsort(tum, kind="stable")
    ext = (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max()); dm = float(q["side"]); panels = []
    fig = plt.figure(figsize=(17, 8)); gs = fig.add_gridspec(1, 2, wspace=0.08)
    ax = fig.add_subplot(gs[0, 0]); panels.append((ax, "A")); cells(ax, xy[o], list(col[o]), "Simulated tissue", size=2.0, fontsize=12, extent=ext)
    r = 300.0; c0 = np.median(xy[tum], 0) + np.array([-0.6, 0.0]) * (np.ptp(xy[tum][:, 0]) / 2); win = (np.abs(xy - c0) <= r).all(1) & sec
    ax = fig.add_subplot(gs[0, 1]); panels.append((ax, "B"))
    cells(ax, xy[win], list(col[win]), "Visium capture", size=40, fontsize=12, extent=(c0[0] - r, c0[0] + r, c0[1] - r, c0[1] + r))
    sp = np.asarray(q["spots"], float)
    for s in sp[(np.abs(sp - c0) <= r + dm).all(1)]:
        ax.add_patch(Circle(s, dm / 2, fill=False, lw=1.6, ec="#333333", zorder=5))
    ax.plot([c0[0] + r - 120, c0[0] + r - 20], [c0[1] + r - 25] * 2, color="#111111", lw=3); ax.text(c0[0] + r - 70, c0[1] + r - 40, "100 µm", ha="center", va="bottom", fontsize=10)
    hd = [Line2D([], [], ls="", marker="o", ms=8, color=ccol[k], label=f"clone {k + 1}") for k in range(K - 1)]
    hd += [Line2D([], [], ls="", marker="o", ms=8, color=ccol[K - 1], label="smaller clones")]
    hd += [Line2D([], [], ls="", marker="o", ms=8, color=TYPE_COL[t], label=f"non-tumor type {t}") for t in (1, 2, 3)]
    hd += [Line2D([], [], ls="", marker="o", ms=10, mfc="none", mec="#333333", label="Visium spot")]
    ax.legend(handles=hd, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=10)
    figlayout.finish(fig, panels); f = paths.figure("fig_sim_tissue.pdf"); fig.savefig(f, bbox_inches="tight", dpi=200); figlayout.compress_pdf(f)
    fig.savefig(f.with_suffix(".png"), dpi=70, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
