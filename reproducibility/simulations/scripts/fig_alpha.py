"""Fig. S4 (fig_alpha_elbow): choosing the structural importance alpha at the elbow, on the example world (environment benchmark_env; CPU).

(A) SpOTr's expression cost and structural cost over alpha, each normalised between the smallest and the largest alpha, and their sum, whose
    minimum is the elbow; one line per seed.  (B) The per-spot clone error of each map (the mean over spots with tumor cells of the total
    variation distance between the true and the mapped clone composition), with the elbow and the default alpha marked.

Usage: python fig_alpha.py   (reads config.yaml and the results of rule spotr_grid; writes to common.paths.FIGURES)
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
from common import paths, figlayout  # noqa: E402
import world  # noqa: E402
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402

INK, GRID, ORANGE = "#111111", "#e6e5e0", "#F47B20"


def clone_error(P, Ncl):
    has = Ncl.sum(1) > 0; T = Ncl[has] / Ncl[has].sum(1, keepdims=True); Q = P[has] / np.maximum(P[has].sum(1, keepdims=True), 1e-300)
    return float(0.5 * np.abs(T - Q).sum(1).mean())


def main():
    cfg = yaml.safe_load(open(HERE.parent / "config.yaml")); base = world.example(cfg, HERE.parent).parent
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6)); fig.subplots_adjust(wspace=0.3)
    for s in cfg["seeds"]:
        d = base / f"seed{s}"; j = json.load(open(d / "spotr_grid.json")); z = np.load(d / "spotr_grid.npz"); q = np.load(d / "problem.npz")
        al = [float(a) for a in j["alphas"]]; tr = q["traced"].astype(bool); oh = np.eye(int(q["K"]))[q["clone"][tr]]; ls = "-" if s == cfg["seeds"][0] else (0, (4, 2))
        E = np.array([j["costs"][str(a)]["expression"] for a in al]); S = np.array([j["costs"][str(a)]["structure"] for a in al])
        e = (E - E[0]) / max(E[-1] - E[0], 1e-30); c = (S - S[-1]) / max(S[0] - S[-1], 1e-30); el = al[int(np.argmin(e + c))]; first = s == cfg["seeds"][0]
        axs[0].plot(al, e, color="#4c78a8", ls=ls, lw=2, label="expression cost" if first else None)
        axs[0].plot(al, c, color="#b279a2", ls=ls, lw=2, label="structural cost" if first else None)
        axs[0].plot(al, e + c, color=INK, ls=ls, lw=2.5, label="sum" if first else None); axs[0].scatter([el], [(e + c)[al.index(el)]], s=90, color=ORANGE, edgecolor=INK, zorder=5)
        err = [clone_error(z[f"phi_{a}"].astype(float).T @ oh, q["Ncl"].astype(float)) for a in al]
        axs[1].plot(al, err, color=ORANGE, ls=ls, lw=2.5, marker="o", label="SpOTr" if first else None)
        axs[1].scatter([el], [err[al.index(el)]], s=130, marker="D", color=ORANGE, edgecolor=INK, zorder=5); print(f"seed {s}: elbow {el}")
    axs[0].set_title("Elbow", fontsize=11); axs[0].set_xlabel("alpha"); axs[0].set_ylabel("normalised cost"); axs[0].legend(frameon=False, fontsize=9)
    ax = axs[1]; ax.axvline(cfg["spotr"]["alpha"], color="#9a9a9a", lw=1, ls=(0, (4, 3)))
    ax.text(cfg["spotr"]["alpha"] + 0.005, ax.get_ylim()[1], "default", va="top", fontsize=9, color="#7a7a7a")
    ax.set_title("Clone error", fontsize=11); ax.set_xlabel("alpha"); ax.set_ylabel("per-spot clone error"); ax.legend(frameon=False, fontsize=9)
    for ax in axs:
        ax.grid(color=GRID); ax.set_axisbelow(True); [ax.spines[k].set_visible(False) for k in ("top", "right")]
    figlayout.finish(fig, [(axs[0], "A"), (axs[1], "B")]); f = paths.figure("fig_alpha_elbow.pdf"); fig.savefig(f, bbox_inches="tight"); figlayout.compress_pdf(f)
    fig.savefig(f.with_suffix(".png"), dpi=80, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
