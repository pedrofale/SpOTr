"""Fig. S5 (fig_options): weight 0 on the dispersed clones (environment benchmark_env; CPU).

For each simulated tumor with one or two of its five clones dispersed over the section, with weak (the benchmark's) and with distinct clone
expression, the change in the mean per-clone Pearson correlation with the truth when SpOTr's clade weights of the dispersed clones are set to 0;
bars the mean over the seeds, points the seeds.

Usage: python fig_weights.py   (reads config.yaml and the results of rule spotr_weights; writes to common.paths.FIGURES)
"""
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
from common import paths, metrics, figlayout  # noqa: E402
import world  # noqa: E402
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402

GRID, ORANGE, BROWN = "#e6e5e0", "#F47B20", "#7a3a10"


def change(d):
    """the change in the mean per-clone PCC from the default weights to weight 0 on the dispersed clones, and the number of dispersed clones"""
    z = np.load(d / "spotr_weights.npz"); q = np.load(d / "problem.npz"); tr = q["traced"].astype(bool); oh = np.eye(int(q["K"]))[q["clone"][tr]]
    pcc = {a: np.mean(metrics.per_clone(z[f"phi_{a}"].astype(float).T @ oh, q["Ncl"].astype(float))["PCC"]) for a in ("default", "weights")}
    return float(pcc["weights"] - pcc["default"]), len(z["dispersed"])


def main():
    cfg = yaml.safe_load(open(HERE.parent / "config.yaml")); root = HERE.parent / cfg["out"]; rows = world.weight_worlds(cfg)
    fig, ax = plt.subplots(figsize=(5.6, 4.0)); xt, xl = [], []
    for g, (sig, label) in enumerate(zip(sorted(cfg["weights"]["sigma_tree"]), ("weak", "distinct"))):
        res = {}
        for s_, m_, w, _ in rows:
            if s_ == sig:
                for s in cfg["seeds"]:
                    dl, nd = change(root / "worlds" / w / f"seed{s}"); res.setdefault(nd, []).append(dl)
        for h, nd in enumerate(sorted(res)):
            x0 = g * 2.6 + h; dl = res[nd]
            ax.bar([x0], [np.mean(dl)], width=0.7, color=ORANGE, alpha=0.45, edgecolor=ORANGE)
            ax.scatter(x0 + np.linspace(-0.08, 0.08, len(dl)), dl, s=28, color=BROWN, zorder=3); xt.append(x0); xl.append("1 clone" if nd == 1 else f"{nd} clones")
            print(f"{label} expression, {nd} dispersed: {np.round(dl, 3).tolist()}")
        ax.text(g * 2.6 + 0.5, -0.11, f"{label} clone expression", ha="center", va="top", fontsize=10, transform=ax.get_xaxis_transform())
    ax.axhline(0, color="#9a9a9a", lw=1, ls=(0, (4, 3))); ax.set_xticks(xt, xl, fontsize=9); ax.tick_params(axis="x", pad=2)
    ax.set_ylabel("change in mean per-clone PCC"); ax.set_title("Zero weight on dispersed clones", fontsize=11)
    ax.grid(color=GRID, axis="y"); ax.set_axisbelow(True); [ax.spines[k].set_visible(False) for k in ("top", "right")]
    fig.subplots_adjust(bottom=0.2); f = paths.figure("fig_options.pdf"); fig.savefig(f, bbox_inches="tight"); figlayout.compress_pdf(f)
    fig.savefig(f.with_suffix(".png"), dpi=80, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
