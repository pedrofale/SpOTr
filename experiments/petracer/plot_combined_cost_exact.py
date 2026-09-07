"""Yesterday's per-cell cost figure, regenerated with the EXACT solver (POT's FGW).

Same three cells, same alpha grid, same layout. The cost shown is the linearised FGW cost at
convergence:  G = (1-alpha)*M + alpha*gwgrad(T),  built with POT's own helpers.
"""
import numpy as np, ot
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d8d4"
ALPHAS = [0.0, 0.10, 0.30, 0.45, 0.60, 0.90]
PICKS = [775, 2111, 1236]
SEC = "M3-4"

d = np.load(HERE / f"petracer_{SEC}.npz", allow_pickle=True)
X_sc, X_st = np.asarray(d["X_sc"], np.float64), np.asarray(d["X_st"], np.float64)
CT = np.asarray(d["C_tree"], np.float64); CS = np.asarray(d["C_space"], np.float64)
xy_sc, xy_st = np.asarray(d["xy_sc"], float), np.asarray(d["xy"], float)
cc = np.asarray(d["cell_clade"]).astype(str)
clades = list(np.asarray(d["clades"]).astype(str))
cidx = np.array([clades.index(c) for c in cc])
COMP = np.load(HERE / f"comp_{SEC}.npy").astype(np.float64)
n, m = len(cc), X_st.shape[0]
p, q = np.ones(n)/n, np.ones(m)/m
M = ot.dist(X_sc, X_st); M = M/M.max()
C1, C2 = CT/CT.mean(), CS/CS.mean()
constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, "square_loss")

COST, GAM = {}, {}
print(f"{'alpha':>7}{'median placement error (um)':>30}{'soft accuracy':>16}")
print("-"*53)
for a in ALPHAS:
    T = ot.gromov.fused_gromov_wasserstein(M, C1, C2, p, q, loss_fun="square_loss", alpha=a,
                                           max_iter=200, tol_rel=1e-9, tol_abs=1e-9)
    GAM[a] = T
    COST[a] = (1-a)*M + a*ot.gromov.gwggrad(constC, hC1, hC2, T)
    j = T.argmax(1)
    err = np.linalg.norm(xy_st[j]-xy_sc, axis=1)
    print(f"{a:>7.2f}{np.median(err):>30.1f}{float(COMP[j, cidx].mean()):>16.4f}")
print(f"\nrandom-spot floor: {float(COMP.mean(0)[cidx].mean()):.4f}")

fig, axes = plt.subplots(3, len(ALPHAS), figsize=(3.05*len(ALPHAS), 9.6))
for r, i0 in enumerate(PICKS):
    e0 = np.linalg.norm(xy_st[GAM[0.0][i0].argmax()] - xy_sc[i0])
    for c, a in enumerate(ALPHAS):
        ax = axes[r, c]
        v = COST[a][i0].copy(); v = (v-v.min())/(v.max()-v.min()+1e-12)
        ax.scatter(xy_st[:,0], xy_st[:,1], s=26, c=v, cmap="Blues_r", vmin=0, vmax=1, linewidth=0)
        j = int(GAM[a][i0].argmax())
        ax.scatter(*xy_st[j], s=155, facecolor="none", edgecolor="#eb6834", linewidth=2.1, zorder=5)
        ax.scatter(*xy_sc[i0], s=200, marker="*", c="#1baf7a", edgecolor="white",
                   linewidth=1.1, zorder=6)
        err = np.linalg.norm(xy_st[j]-xy_sc[i0])
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color(GRID)
        if r == 0: ax.set_title(f"$\\alpha$ = {a:g}", fontsize=12, color=INK, pad=8)
        ax.set_xlabel(f"error {err:.0f} µm", fontsize=9,
                      color="#1baf7a" if err < 300 else ("#eb6834" if err < 900 else "#c05621"))
        if c == 0: ax.set_ylabel(f"cell {i0}", fontsize=10, color=INK)

h = [plt.Line2D([],[],marker="*",ls="",ms=13,c="#1baf7a",mec="white",label="where the cell really is"),
     plt.Line2D([],[],marker="o",ls="",ms=11,mfc="none",mec="#eb6834",mew=2,label="where SpOTr puts it")]
fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=10.5,
           bbox_to_anchor=(0.5, 0.012))
fig.suptitle("The combined cost for one cell, over all 429 spots, as $\\alpha$ varies\n"
             "EXACT solver (POT fused_gromov_wasserstein) — dark = low cost = preferred",
             fontsize=13, y=0.972, color=INK)
fig.tight_layout(rect=[0, 0.045, 1, 0.925])
fig.savefig("petracer_combined_cost_exact.png", dpi=145, facecolor="white")
print("\nwrote petracer_combined_cost_exact.png")
