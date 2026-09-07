import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parent
L = pd.read_csv(HERE / "two_branch_theta_landscape.csv")
U = sorted(L.u1.unique()); C = sorted(L.c.unique())
BEST_C = {"M3-4": 0.20, "M3-1": 0.30}
fig = plt.figure(figsize=(14.6, 4.8))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2], wspace=.42,
                      left=.05, right=.985, top=.78, bottom=.14)

for i, s in enumerate(["M3-4", "M3-1"]):
    ax = fig.add_subplot(gs[0, i]); c = BEST_C[s]
    d = L[(L.section == s) & (L.c == c)]
    Z = d.pivot(index="u2", columns="u1", values="acc").reindex(index=U, columns=U).values
    im = ax.imshow(Z, origin="lower", cmap="viridis", extent=[-.5, 4.5, -.5, 4.5])
    for a in range(5):
        for b in range(5):
            if np.isfinite(Z[a, b]):
                ax.text(b, a, f"{Z[a,b]:.3f}", ha="center", va="center", fontsize=7.6,
                        color="white" if Z[a, b] < np.nanmax(Z)*.85 else "black")
    ax.plot([-.5, 4.5], [-.5, 4.5], "w--", lw=1.2, alpha=.75)
    bi = np.unravel_index(np.nanargmax(Z), Z.shape)
    ax.scatter([bi[1]], [bi[0]], s=180, facecolors="none", edgecolors="r", lw=2.2, zorder=5)
    ax.set_xticks(range(5)); ax.set_xticklabels([f"{u:g}" for u in U], fontsize=8)
    ax.set_yticks(range(5)); ax.set_yticklabels([f"{u:g}" for u in U], fontsize=8)
    ax.set_xlabel(r"$u_1$ (branch 1)"); ax.set_ylabel(r"$u_2$ (branch 2)")
    dg = np.array([Z[k, k] for k in range(5)])
    ax.set_title(f"{s},  c = {c:g}\nbest {np.nanmax(Z):.4f}   uniform {np.nanmax(dg):.4f}"
                 f"   →  {np.nanmax(Z)-np.nanmax(dg):+.4f}", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=.046)

ax = fig.add_subplot(gs[0, 2])
for s, col in [("M3-4", "#1f77b4"), ("M3-1", "#d62728")]:
    hi, un = [], []
    for c in C:
        d = L[(L.section == s) & (L.c == c)]
        hi.append(d.acc.max()); un.append(d[d.u1 == d.u2].acc.max())
    ax.plot(C, hi, "o-", color=col, lw=2.2, ms=6, label=f"{s}  two-branch")
    ax.plot(C, un, "s--", color=col, lw=2, ms=6, alpha=.55, label=f"{s}  uniform u")
    ax.fill_between(C, un, hi, color=col, alpha=.12)
ax.set_xlabel("c  =  shared cross-clade weight  $α_τ(1-θ_τ)$")
ax.set_ylabel("best soft accuracy")
ax.set_title("c = 0 is θ=1 (decoupled).  Shaded = what\nper-branch $u$ adds over a uniform $u$.",
             fontsize=10)
ax.legend(frameon=False, fontsize=8.5); ax.grid(alpha=.25)
fig.suptitle("The well-posed family: $u_τ = α_τθ_τ$ free per branch, "
             "$c = α_τ(1-θ_τ)$ shared", fontsize=11.5, y=.95)
fig.savefig(HERE / "two_branch_theta_grid.png", dpi=155)
print("wrote two_branch_theta_grid.png")
