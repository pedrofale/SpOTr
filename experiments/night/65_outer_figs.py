"""Figures for 64_outer_loss.py output (light and dark).  Usage: python 65_outer_figs.py <dir of 64_ output>"""
import sys, json, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
D = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("m60", HERE / "60_anchor_learn.py")
m60 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m60)
P = m60.prep("M3-4", 0)
path = pd.read_csv(D / "path.csv"); traj = pd.read_json(D / "trajectories.json"); meta = json.load(open(D / "meta.json"))
J = np.array(meta["J"]); Jfit, Jval = J[: len(J) // 2], J[len(J) // 2:]
z = np.load(HERE / "anchor_fast" / "M3-4_s0_c20_a-only.npz"); n = P["M"].shape[0]
T = m60.m55.solve(P["M"], z["W"] * z["sc"], z["WPhi"] * z["sc"], P["D"], P["D"] ** 2, np.ones(n) / n, P["q"], float(z["a"]))
Y = np.eye(P["K"])[P["cl"]]
obs = P["Nspot"] / P["Nspot"].sum(1, keepdims=True); pred = (T.T @ Y); pred = pred / pred.sum(1, keepdims=True)
sizes = np.array(meta["sizes"]); top = list(np.argsort(-sizes)[:8])
pitch = float(np.load(HERE.parent / "petracer" / "petracerFAIR_M3-4.npz", allow_pickle=True)["pitch"])
fgw_best = None

THEMES = {
    "light": dict(surf="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e4e3df", other="#c9c8c2", faint="#efeeea",
                  s=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]),
    "dark": dict(surf="#1a1a19", ink="#f0efec", ink2="#c3c2b7", grid="#34342f", other="#55554f", faint="#262624",
                 s=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]),
}
ARMS = {"weight only": 0, "per-clade, anchor": 1, "random groups": 2, "per-clade, supervised": 3}
ARM_LABEL = {"weight only": "Tree weight only (anchor)", "per-clade, anchor": "+ per-clade L (anchor)",
             "random groups": "+ per-group L, random groups (anchor)", "per-clade, supervised": "+ per-clade L (supervised, 30% of cells)"}


def style(t):
    plt.rcParams.update({"figure.facecolor": t["surf"], "axes.facecolor": t["surf"], "savefig.facecolor": t["surf"],
                         "axes.edgecolor": t["grid"], "axes.labelcolor": t["ink2"], "xtick.color": t["ink2"],
                         "ytick.color": t["ink2"], "text.color": t["ink"], "axes.grid": True, "grid.color": t["grid"],
                         "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
                         "font.size": 10.5, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
                         "axes.titlecolor": t["ink"], "legend.frameon": False, "lines.linewidth": 2, "axes.axisbelow": True})


def glyph_map(ax, t, comp, title):
    xy = P["xy_st"]; h = pitch * 0.94; colors = [t["s"][i] for i in range(8)]
    for j, (x, y) in enumerate(xy):
        if j not in set(J):
            ax.add_patch(Rectangle((x - h / 2, y - h / 2), h, h, color=t["faint"], lw=0))
            continue
        x0 = x - h / 2
        fr = [comp[j, k] for k in top] + [1 - comp[j, top].sum()]
        for f, c in zip(fr, colors + [t["other"]]):
            if f > 1e-3:
                ax.add_patch(Rectangle((x0, y - h / 2), h * f, h, color=c, lw=0)); x0 += h * f
        if j in set(Jval):
            ax.add_patch(Rectangle((x - h / 2, y - h / 2), h, h, fill=False, ec=t["ink"], lw=0.8))
    pad = pitch
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad); ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title)


for th, t in THEMES.items():
    style(t); s = t["s"]

    # A. the outer loss: observed vs predicted clade composition at the anchor spots
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    glyph_map(axes[0], t, obs, "Observed clade mix at the anchor spots")
    glyph_map(axes[1], t, pred, "Clade mix the mapping predicts there")
    names = sorted(set(m60.obs_clade().loc[np.asarray(np.load(HERE.parent / "petracer" / "petracerFAIR_M3-4.npz", allow_pickle=True)["cell_names"])]) |
                   set(m60.obs_clade().loc[np.asarray(np.load(HERE.parent / "petracer" / "petracerFAIR_M3-4.npz", allow_pickle=True)["spot_cell_names"])]))
    hd = [Rectangle((0, 0), 1, 1, color=s[i]) for i in range(8)] + [Rectangle((0, 0), 1, 1, color=t["other"]),
          Rectangle((0, 0), 1, 1, fill=False, ec=t["ink"])]
    fig.legend(hd, [f"Clade {names[k]}" for k in top] + ["smaller clades", "validation spot (outlined)"],
               loc="lower center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    fig.savefig(D / f"A_anchor_{th}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # B. does the loss help: along the tree weight, and over learning steps
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.6), sharex="col")
    x = path.w.replace(0, 0.01)
    axes[0, 0].plot(x, path.err, "-o", color=t["ink"], ms=4); axes[0, 0].set_ylabel("Placement error (µm)")
    axes[1, 0].plot(x, path.anchor_loss, "-o", color=s[0], ms=4); axes[1, 0].set_ylabel("Outer loss (anchor NLL)")
    axes[1, 0].set_xscale("log"); axes[1, 0].set_xlabel("Tree weight w (0 drawn at 0.01)")
    axes[0, 0].set_title("Along the tree weight, exact solver")
    for arm in ["weight only", "per-clade, anchor", "random groups"]:
        g = traj[traj.arm == arm]; c = s[ARMS[arm]]; ls = "--" if arm == "random groups" else "-"
        axes[0, 1].plot(g.step, g.err, ls, color=c, lw=1.6, label=ARM_LABEL[arm])
        axes[1, 1].plot(g.step, g.anchor_fit, ls, color=c, lw=1.6)
        axes[1, 1].plot(g.step, g.anchor_val, ":", color=c, lw=1.2)
    axes[0, 1].set_title("Over learning steps (fit = solid, validation spots = dotted)")
    axes[1, 1].set_xlabel("Learning step"); axes[0, 1].set_ylabel("Placement error (µm)"); axes[1, 1].set_ylabel("Outer loss")
    h_, l_ = axes[0, 1].get_legend_handles_labels()
    fig.tight_layout(); fig.legend(h_, l_, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.05))
    fig.savefig(D / f"B_helping_{th}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # C. per-clade parameters: trajectories and learned vs measured
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    g = traj[traj.arm == "per-clade, anchor"]; L = np.array(g.L.tolist())
    for k in range(L.shape[1]):
        c = s[top.index(k)] if k in top else t["other"]
        axes[0].plot(g.step, L[:, k], color=c, lw=1.6 if k in top else 0.9)
    axes[0].axhline(305, color=t["ink2"], ls=":", lw=1)
    axes[0].set_xlabel("Learning step"); axes[0].set_ylabel("Mixing length L_k (µm)")
    axes[0].set_title("Per-clade mixing lengths while learning from the anchor")
    for arm in ["per-clade, anchor", "per-clade, supervised"]:
        gg = traj[traj.arm == arm]; Lend = np.array(gg.L.iloc[-1]); Lm = np.array(meta["measured"][arm], dtype=float)
        ok = np.isfinite(Lm)
        rho = spearmanr(Lend[ok], Lm[ok])[0]
        axes[1].scatter(Lm[ok], Lend[ok], s=34, color=s[[1, 3, 2][["per-clade, anchor", "per-clade, supervised", "random groups"].index(arm)]] if False else
                        s[{"per-clade, anchor": 0, "per-clade, supervised": 1, "random groups": 2}[arm]],
                        label=f"{ARM_LABEL[arm]}: Spearman {rho:+.2f}", alpha=0.85, lw=0)
    axes[1].set_xlabel("Mixing length measured from true positions (µm)"); axes[1].set_ylabel("Learned (µm)")
    axes[1].set_title("Learned vs measured, groups with 30+ cells"); axes[1].legend(fontsize=8.5, loc="upper left")
    fig.tight_layout(); fig.savefig(D / f"C_perclade_{th}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

stats = dict(path_best_w=float(path.loc[path.err.idxmin(), "w"]), path_best_err=float(path.err.min()),
             path_loss_min_w=float(path.loc[path.anchor_loss.idxmin(), "w"]),
             rho_path=float(spearmanr(path.err, path.anchor_loss)[0]),
             end={arm: dict(a=float(g.a.iloc[-1]), err=float(g.err.iloc[-1]), err_heldout=float(g.err_heldout.iloc[-1]),
                            fit=float(g.anchor_fit.iloc[-1]), val=float(g.anchor_val.iloc[-1])) for arm, g in traj.groupby("arm")},
             rho_L={arm: float(spearmanr(np.array(traj[traj.arm == arm].L.iloc[-1])[np.isfinite(np.array(meta["measured"][arm], float))],
                                         np.array(meta["measured"][arm], float)[np.isfinite(np.array(meta["measured"][arm], float))])[0])
                    for arm in ["per-clade, anchor", "per-clade, supervised", "random groups"]},
             n_anchor=len(J))
json.dump(stats, open(D / "stats.json", "w"), indent=1); print(json.dumps(stats, indent=1))
