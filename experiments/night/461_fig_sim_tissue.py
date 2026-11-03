"""Supplementary figure: one simulated tumor in space (Pedro, 2026-11-02: "I think the current figs1 is outdated?" -- it showed the earlier
simulator's microenvironment; Fig. 2A is now a UMAP, so the tissue itself is shown here).

The example world of Fig. 2 (the base world, seed 1) as the Cassiopeia workflow simulates it (config.yaml, Visium spots; since 2026-11-02, night,
the tumour cells in nested territories along the tree, simulate.py spatial: territories).
The section's cells are rebuilt exactly as workflows/cassiopeia_benchmark/scripts/express.py builds them (the tumour cells onto scCube's
tumour sites by exact OT, the microenvironment from the layout, the random half measured, the Visium lattice) and checked against the
stored problem (spots and true clone counts identical), since express.py stores the measured half's spots but not its cells.
(A) Every cell of the section: cancer cells by clone, the microenvironment by type.  (B) A window of the measured half, with the 55 um Visium
spots over its cells.  Short titles (writing-style).

Usage: python 461_fig_sim_tissue.py   (benchmark_env, CPU; writes SpOTr-overleaf/figures/fig_sim_tissue.{pdf,png})
"""
import sys, json, importlib.util
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; FIG = ROOT.parent / "SpOTr-overleaf" / "figures"; sys.path.insert(0, str(ROOT / "src"))
PANELS = []


def mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def letter(ax, s, x=None, y=None):
    PANELS.append((ax, s))


def section(wd, p, seed):
    """express.py's main up to the measurement, verbatim in its random draws: every cell's position, kind, type and clone, the measured half
    and the Visium spots"""
    import ot
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(int(seed) + 7); z = np.load(wd / "sim.npz")
    xy_t, clone_t, K = z["xy"], z["clone"], int(z["K"]); n_t = len(xy_t); G, P = int(p["genes"]), int(p["programs"])
    parent, time = z["parent"], z["time"]; h = np.zeros((len(parent), P))
    for v in range(len(parent)):
        if parent[v] >= 0:
            h[v] = h[parent[v]] + rng.normal(0.0, np.sqrt(max(time[v] - time[parent[v]], 0.0) / float(z["depth"])), P)
    L = np.load(wd / "layout.npz"); lt = L["type"]; site = L["xy"][lt == 0]
    nz = lambda a: (a - a.mean(0)) / np.sqrt(((a - a.mean(0)) ** 2).sum(1).mean()); w = np.full(n_t, 1.0 / n_t)
    xy_t = site[np.asarray(ot.emd(w, w, ot.dist(nz(xy_t), nz(site)), numItermax=10 ** 8)).argmax(1)]
    xy_me = L["xy"][lt > 0]; me_type = lt[lt > 0] - 1; n_me = len(xy_me)
    xy = np.concatenate([xy_t, xy_me]); kind = np.r_[np.zeros(n_t, int), np.ones(n_me, int)]; ctype = np.r_[np.zeros(n_t, int), 1 + me_type]
    b = rng.normal(0, 1, G); c = rng.normal(0, 1, (4, G)); A = rng.normal(0, 1 / np.sqrt(P), (G, P)); Bm = rng.normal(0, 1 / np.sqrt(P), (G, P))
    diam = float(np.ptp(xy, 0).max()); W = rng.normal(0, 1 / (float(p["length_scale"]) * diam), (P, 2)); u = rng.uniform(0, 2 * np.pi, P)
    F = np.sqrt(2) * np.cos(xy @ W.T + u); Hall = np.r_[h[z["leaf_node"]] / max(h[z["leaf_node"]].std(), 1e-12), np.zeros((n_me, P))]
    eta = b[None, :] + c[ctype] + float(p["sigma_tree"]) * Hall @ A.T + float(p["sigma_space"]) * F @ Bm.T
    pr = np.exp(eta - eta.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True); rng.poisson(float(p["library"]) * pr)
    sec = rng.random(len(xy)) < 0.5
    a, dm = float(p["spot_spacing"]), float(p["spot_diameter"]); lo, hi = xy.min(0), xy.max(0); rows = np.arange(lo[1] + a / 4, hi[1], a * np.sqrt(3) / 2)
    grid = np.array([(x, y) for i, y in enumerate(rows) for x in np.arange(lo[0] + a / 4 + (a / 2) * (i % 2), hi[0], a)])
    dist, near = cKDTree(grid).query(xy[sec]); hit = dist <= dm / 2; idx = np.where(sec)[0][hit]; used, inv = np.unique(near[hit], return_inverse=True)
    Ncl = np.zeros((len(used), K)); tum = kind[idx] == 0; np.add.at(Ncl, inv.ravel()[tum], np.eye(K)[clone_t[idx[tum]]])
    return dict(xy=xy, kind=kind, ctype=ctype, clone=np.r_[clone_t, -np.ones(n_me, int)], sec=sec, spots=grid[used], Ncl=Ncl, K=K, dm=dm, types=[str(t) for t in L["types"]])


def main():
    import warnings; warnings.filterwarnings("ignore")
    from spotr.plot_spatial import cells
    from spotr.figlayout import finish
    m455 = mod("m455", "455_fig_sim_benchmark.py"); fp = m455.example_world(); wd = fp.parent; p = json.load(open(wd.parent / "params.json")); q = np.load(fp)
    S = section(wd, p, 1); assert np.allclose(S["spots"], q["spots"]) and np.array_equal(S["Ncl"], q["Ncl"]), "the rebuilt section differs from the stored problem"
    K = S["K"]; tab = plt.get_cmap("tab10"); ccol = [to_hex(tab(k)) for k in range(K - 1)] + ["#bdbdbd"]
    mecol = {1: "#d8cfbd", 2: "#8fa9c4", 3: "#c9908f"}; names = {1: "non-tumor type 1", 2: "non-tumor type 2", 3: "non-tumor type 3"}   # synthetic types: random expression; scCube's roles (stroma, aggregates, vessels) set only where they sit
    col = np.array([ccol[c] if k == 0 else mecol[t] for c, k, t in zip(S["clone"], S["kind"], S["ctype"])], dtype=object); o = np.argsort(S["kind"] == 0, kind="stable")
    xy = S["xy"]; ext = (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max())
    fig = plt.figure(figsize=(17, 8)); gs = fig.add_gridspec(1, 2, wspace=0.08)
    ax = fig.add_subplot(gs[0, 0]); letter(ax, "A"); cells(ax, xy[o], list(col[o]), "Simulated tissue", size=2.0, fontsize=12, extent=ext)
    # (B) a window across the tumor's edge: the measured half's cells and the spots
    tum = S["kind"] == 0; c0 = np.median(xy[tum], 0); r = 300.0; c0 = c0 + np.array([-0.6, 0.0]) * (np.ptp(xy[tum][:, 0]) / 2)
    win = (np.abs(xy - c0) <= r).all(1) & S["sec"]
    ax = fig.add_subplot(gs[0, 1]); letter(ax, "B"); cells(ax, xy[win], list(col[win]), "Visium capture", size=40, fontsize=12, extent=(c0[0] - r, c0[0] + r, c0[1] - r, c0[1] + r))
    for s_ in S["spots"][(np.abs(S["spots"] - c0) <= r + S["dm"]).all(1)]:
        ax.add_patch(Circle(s_, S["dm"] / 2, fill=False, lw=1.6, ec="#333333", zorder=5))
    ax.plot([c0[0] + r - 120, c0[0] + r - 20], [c0[1] + r - 25] * 2, color="#111111", lw=3); ax.text(c0[0] + r - 70, c0[1] + r - 40, "100 µm", ha="center", va="bottom", fontsize=10)
    hd = [Line2D([], [], ls="", marker="o", ms=8, color=ccol[k], label=f"clone {k + 1}") for k in range(K - 1)] + [Line2D([], [], ls="", marker="o", ms=8, color=ccol[K - 1], label="smaller clones")]
    hd += [Line2D([], [], ls="", marker="o", ms=8, color=mecol[t], label=names[t]) for t in (1, 2, 3)] + [Line2D([], [], ls="", marker="o", ms=10, mfc="none", mec="#333333", label="Visium spot")]
    ax.legend(handles=hd, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=10)
    finish(fig, PANELS); FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_sim_tissue.pdf", bbox_inches="tight", dpi=200); __import__('spotr.figlayout', fromlist=['compress_pdf']).compress_pdf(FIG / "fig_sim_tissue.pdf"); fig.savefig(FIG / "fig_sim_tissue.png", dpi=70, bbox_inches="tight"); plt.close(fig)
    print("section rebuilt and checked:", int(S["sec"].sum()), "cells in the measured half,", len(S["spots"]), "spots")


if __name__ == "__main__":
    main()
