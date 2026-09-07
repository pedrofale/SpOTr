import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parent
Lm = pd.read_csv(HERE / "two_branch_theta1_landscape.csv")
Lu = pd.read_csv(HERE / "two_branch_exact_landscape.csv")
G = sorted(Lm.a1.unique())
fig = plt.figure(figsize=(15.0, 5.1))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=.45,
                      left=.05, right=.985, top=.78, bottom=.13)

for i, s in enumerate(["M3-4", "M3-1"]):
    ax = fig.add_subplot(gs[0, i])
    Z = Lm[Lm.section == s].pivot(index="a2", columns="a1", values="acc").reindex(
        index=G, columns=G).values
    im = ax.imshow(Z, origin="lower", cmap="viridis", extent=[-.5, 5.5, -.5, 5.5])
    for a in range(6):
        for b in range(6):
            ax.text(b, a, f"{Z[a,b]:.3f}", ha="center", va="center", fontsize=7.2,
                    color="white" if Z[a, b] < Z.max()*.82 else "black")
    ax.plot([-.5, 5.5], [-.5, 5.5], "w--", lw=1.2, alpha=.75)
    bi = np.unravel_index(np.nanargmax(Z), Z.shape)
    ax.scatter([bi[1]], [bi[0]], s=180, facecolors="none", edgecolors="r", lw=2.2, zorder=5)
    ax.set_xticks(range(6)); ax.set_xticklabels([f"{g:g}" for g in G], fontsize=8)
    ax.set_yticks(range(6)); ax.set_yticklabels([f"{g:g}" for g in G], fontsize=8)
    ax.set_xlabel(r"$\alpha_1$ (branch 1)"); ax.set_ylabel(r"$\alpha_2$ (branch 2)")
    dg = np.array([Z[k, k] for k in range(6)])
    ax.set_title(f"{s}, θ=1   best {Z[bi]:.4f}\nbest uniform {dg.max():.4f}  →  "
                 f"gain {Z[bi]-dg.max():+.4f}", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=.046)

ax = fig.add_subplot(gs[0, 2])
for s, c in [("M3-4", "#1f77b4"), ("M3-1", "#d62728")]:
    for tag, L, ls, mk in [("θ=1 (masked)", Lm, "-", "o"), ("unmasked", Lu, "--", "s")]:
        d = L[L.section == s]
        y = [float(d[(d.a1 == g) & (d.a2 == g)].acc.iloc[0]) for g in G]
        ax.plot(G, y, ls, marker=mk, color=c, lw=2, ms=6, alpha=1 if "mask" in tag else .55,
                label=f"{s}  {tag}")
ax.set_xlabel("uniform α"); ax.set_ylabel("soft accuracy")
ax.set_title("Uniform α: masked vs unmasked", fontsize=10)
ax.legend(frameon=False, fontsize=8.5); ax.grid(alpha=.25)
fig.suptitle("Two-branch α with the paper's Ω decomposition at θ=1 "
             "(branches fully decoupled, valid gradient, exact OT)", fontsize=11, y=.95)
fig.savefig(HERE / "two_branch_theta1.png", dpi=155)
print("wrote two_branch_theta1.png")
