"""What SpOTr actually sees when it places one cell, as alpha varies.

The cost is cell x SPOT:   C[i,j] = (1-a) * C_feature[i,j]  +  a * L_gw[i,j]
L_gw depends on the current coupling, so the cost is taken at convergence of the real solver
(semi-relaxed, tau = 0.01 -- the headline configuration).

For a chosen cell we colour every spot by that cell's cost row. The star is where the cell
ACTUALLY is (known, since sc-side and spot-side cells are disjoint halves of the same section);
the circle is the spot the model picks. The gap between them is the error.
"""
import sys
from pathlib import Path
import numpy as np, anndata as ad, jax.numpy as jnp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S
from spotr.perclade_v2 import build_perclade_alpha_cost_v2
from spotr.semirelaxed import solve_perclade_semirelaxed

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d8d4"
EPS, T, J, TAU = 0.02, 50, 3, 0.01
ALPHAS = [0.0, 0.10, 0.30, 0.45, 0.60, 0.90]
SEC = "M3-4"

d = np.load(HERE / f"petracer_{SEC}.npz", allow_pickle=True)
X_sc, X_st = np.asarray(d["X_sc"], np.float32), np.asarray(d["X_st"], np.float32)
xy_sc, xy_st = np.asarray(d["xy_sc"], float), np.asarray(d["xy"], float)
cc = np.asarray(d["cell_clade"]).astype(str)
clades = list(np.asarray(d["clades"]).astype(str))
cidx = np.array([clades.index(c) for c in cc])
CT = np.asarray(d["C_tree"], float)
n, m, K = len(cc), X_st.shape[0], len(clades)
A = np.zeros((n, K), np.float32); A[np.arange(n), cidx] = 1.0

Cf, C_T, C_S, a, b = S.prepare_ot_inputs(ad.AnnData(X_sc), ad.AnnData(X_st),
                                         CT.astype(np.float32),
                                         np.asarray(d["C_space"], np.float32))
omega = jnp.asarray(A); g0 = a[:, None] * b[None, :]
sF, sL = S.make_cladexfgw_scales(Cf, C_T, C_S, a, b, g0, eps=EPS)

COST, GAM = {}, {}
for av in ALPHAS:
    def build(g):
        return build_perclade_alpha_cost_v2(jnp.full((K,), av, dtype=jnp.float32), Cf, C_T,
                                            C_S, a, b, g, omega, sF, sL, equalize="measured")
    g, _ = solve_perclade_semirelaxed(build, a, b, EPS, T_sinkhorn=T, J_alt=J, tau=TAU)
    GAM[av] = np.asarray(g); COST[av] = np.asarray(build(g))
    err = np.linalg.norm(xy_st[GAM[av].argmax(1)] - xy_sc, axis=1)
    print(f"  alpha {av:.2f}: median placement error {np.median(err):7.1f} um", flush=True)

# pick three illustrative cells: rescued by the tree, already fine, still wrong
e0 = np.linalg.norm(xy_st[GAM[0.0].argmax(1)] - xy_sc, axis=1)
e6 = np.linalg.norm(xy_st[GAM[0.60].argmax(1)] - xy_sc, axis=1)
rescued = np.argsort(-(e0 - e6))[:40]
fine    = np.where((e0 < 250) & (e6 < 250))[0]
broken  = np.where((e0 > 900) & (e6 > 900))[0]
rng = np.random.default_rng(0)
PICKS = [("rescued by the tree", int(rescued[3])),
         ("expression already right", int(rng.choice(fine))),
         ("wrong either way", int(rng.choice(broken)))]
print("\ncells chosen:", [(lab, i, f"{e0[i]:.0f}->{e6[i]:.0f} um") for lab, i in PICKS])

fig, axes = plt.subplots(3, len(ALPHAS), figsize=(3.05*len(ALPHAS), 9.6))
for r, (lab, i0) in enumerate(PICKS):
    for c, av in enumerate(ALPHAS):
        ax = axes[r, c]
        v = COST[av][i0].copy()
        v = (v - v.min()) / (v.max() - v.min() + 1e-12)
        ax.scatter(xy_st[:, 0], xy_st[:, 1], s=26, c=v, cmap="Blues_r", vmin=0, vmax=1,
                   linewidth=0)
        j = int(GAM[av][i0].argmax())
        ax.scatter(*xy_st[j], s=155, facecolor="none", edgecolor="#eb6834", linewidth=2.1,
                   zorder=5)
        ax.scatter(*xy_sc[i0], s=200, marker="*", c="#1baf7a", edgecolor="white",
                   linewidth=1.1, zorder=6)
        err = np.linalg.norm(xy_st[j] - xy_sc[i0])
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color(GRID)
        if r == 0: ax.set_title(f"$\\alpha$ = {av:g}", fontsize=12, color=INK, pad=8)
        ax.set_xlabel(f"error {err:.0f} µm", fontsize=9,
                      color="#1baf7a" if err < 300 else ("#eb6834" if err < 900 else "#c05621"))
        if c == 0:
            ax.set_ylabel(f"cell {i0}\n{lab}", fontsize=9.5, color=INK)

h = [plt.Line2D([], [], marker="*", ls="", ms=13, c="#1baf7a", mec="white",
                label="where the cell really is"),
     plt.Line2D([], [], marker="o", ls="", ms=11, mfc="none", mec="#eb6834", mew=2,
                label="where SpOTr puts it")]
fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=10.5,
           bbox_to_anchor=(0.5, 0.012))
fig.suptitle("The combined cost for one cell, over all 429 spots, as $\\alpha$ varies\n"
             "dark = low cost = preferred.  $\\alpha$=0 is expression only; "
             "$\\alpha$=0.9 is almost entirely tree-vs-space",
             fontsize=13, y=0.972, color=INK)
fig.tight_layout(rect=[0, 0.045, 1, 0.925])
fig.savefig("petracer_combined_cost.png", dpi=145, facecolor="white")
print("wrote petracer_combined_cost.png")
