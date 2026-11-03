"""Supplementary figure: choosing the structural importance alpha at the elbow, on a simple example (Pedro, 2026-11-02: "Proceed with a default
alpha and maybe add a supplementary figure on a simple example showing how we might use the elbow to choose it").

The example world of Fig. 2 (heritable expression, sigma_tree 1, seeds 1 and 2), SpOTr over the grid alpha 0, 0.1, 0.2, 0.4, 0.95 with clone labels
and with the tree (the workflow's rule spotr_grid; spotr_grid.npz / .json in each world-seed folder).
(A) For SpOTr with the lineage tree, the expression cost and the structural cost of the map, each normalised between alpha 0 and the largest alpha, and their
    sum, whose minimum is the elbow (450_.elbow / fit_spotr.elbow); one line per seed.
(B) The per-spot clone error (total variation, as score.py) of each map against alpha, with the elbow and the default alpha 0.2 marked.
Short titles (writing-style).

Usage: python 463_fig_alpha_elbow.py   (benchmark_env, CPU; writes SpOTr-overleaf/figures/fig_alpha_elbow.{pdf,png})
"""
import sys, json, importlib.util
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; FIG = ROOT.parent / "SpOTr-overleaf" / "figures"; sys.path.insert(0, str(ROOT / "src"))
INK, GRID = "#111111", "#e6e5e0"; ARMS = [("tree", "SpOTr", "#F47B20")]; PANELS = []                              # one mode, the lineage tree (Pedro, 2026-11-02)


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def letter(ax, s):
    PANELS.append((ax, s))


def tv(P, Ncl):
    has = Ncl.sum(1) > 0; T = Ncl[has] / Ncl[has].sum(1, keepdims=True); Q = P[has] / np.maximum(P[has].sum(1, keepdims=True), 1e-300)
    return float(0.5 * np.abs(T - Q).sum(1).mean())


def main():
    from spotr.figlayout import finish, compress_pdf
    wd = mod("m455", "455_fig_sim_benchmark.py").example_world().parent.parent; seeds = sorted(d for d in wd.glob("seed*") if (d / "spotr_grid.npz").exists())
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6)); fig.subplots_adjust(wspace=0.3); el_all = {}
    for sd in seeds:
        j = json.load(open(sd / "spotr_grid.json")); al = [float(a) for a in j["alphas"]]; z = np.load(sd / "spotr_grid.npz"); q = np.load(sd / "problem.npz")
        K = int(q["K"]); tr = q["traced"].astype(bool); oh = np.eye(K)[q["clone"][tr]]; ls = "-" if sd.name == "seed1" else (0, (4, 2))
        for i, (arm, nm, cc) in enumerate(ARMS):
            E = np.array([j["costs"][f"{arm} {a}"]["expression"] for a in al]); S = np.array([j["costs"][f"{arm} {a}"]["structure"] for a in al])
            e = (E - E[0]) / max(E[-1] - E[0], 1e-30); s = (S - S[-1]) / max(S[0] - S[-1], 1e-30); el = al[int(np.argmin(e + s))]; el_all.setdefault(arm, []).append(el)
            ax = axs[i]; ax.plot(al, e, color="#4c78a8", ls=ls, lw=2, label="expression cost" if sd.name == "seed1" else None)
            ax.plot(al, s, color="#b279a2", ls=ls, lw=2, label="structural cost" if sd.name == "seed1" else None)
            ax.plot(al, e + s, color=INK, ls=ls, lw=2.5, label="sum" if sd.name == "seed1" else None); ax.scatter([el], [(e + s)[al.index(el)]], s=90, color=cc, edgecolor=INK, zorder=5)
            err = [tv((z["phi_expr"] if a == 0.0 else z[f"phi_{arm}_{a}"]).astype(float).T @ oh, q["Ncl"].astype(float)) for a in al]
            axs[1].plot(al, err, color=cc, ls=ls, lw=2.5, marker="o", label=nm if sd.name == "seed1" else None)
            axs[1].scatter([el], [err[al.index(el)]], s=130, marker="D", color=cc, edgecolor=INK, zorder=5)
    for i, (arm, nm, cc) in enumerate(ARMS):
        ax = axs[i]; letter(ax, "A"); ax.set_title("Elbow", fontsize=11); ax.set_xlabel("alpha"); ax.set_ylabel("normalised cost"); ax.legend(frameon=False, fontsize=9)
    ax = axs[1]; letter(ax, "B"); ax.axvline(0.2, color="#9a9a9a", lw=1, ls=(0, (4, 3))); ax.text(0.205, ax.get_ylim()[1], "default", va="top", fontsize=9, color="#7a7a7a")
    ax.set_title("Clone error", fontsize=11); ax.set_xlabel("alpha"); ax.set_ylabel("per-spot clone error"); ax.legend(frameon=False, fontsize=9)
    for ax in axs:
        ax.grid(color=GRID); ax.set_axisbelow(True); [ax.spines[s_].set_visible(False) for s_ in ("top", "right")]
    finish(fig, PANELS); FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_alpha_elbow.pdf", bbox_inches="tight"); compress_pdf(FIG / "fig_alpha_elbow.pdf"); fig.savefig(FIG / "fig_alpha_elbow.png", dpi=80, bbox_inches="tight"); plt.close(fig)
    print("seeds", [d.name for d in seeds], "elbows", el_all)


if __name__ == "__main__":
    main()
