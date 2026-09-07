import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parent
L = pd.read_csv(HERE / "two_branch_exact_landscape.csv")
G = sorted(L.a1.unique())
secs = ["M3-4", "M3-1"]
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9))
for ax, s in zip(axes, secs):
    d = L[L.section == s]
    Z = d.pivot(index="a2", columns="a1", values="acc").reindex(index=G, columns=G).values
    im = ax.imshow(Z, origin="lower", cmap="viridis",
                   extent=[-0.5, len(G)-0.5, -0.5, len(G)-0.5])
    for i, a2 in enumerate(G):
        for j, a1 in enumerate(G):
            ax.text(j, i, f"{Z[i,j]:.3f}", ha="center", va="center", fontsize=7.5,
                    color="white" if Z[i,j] < Z.max()*0.82 else "black")
    ax.plot([-0.5, len(G)-0.5], [-0.5, len(G)-0.5], color="w", lw=1.2, ls="--", alpha=.75)
    bi = np.unravel_index(np.nanargmax(Z), Z.shape)
    ax.scatter([bi[1]], [bi[0]], s=190, facecolors="none", edgecolors="r", lw=2.2, zorder=5)
    dg = np.array([Z[i, i] for i in range(len(G))])
    bd = int(np.nanargmax(dg))
    ax.scatter([bd], [bd], s=190, marker="s", facecolors="none", edgecolors="w", lw=2.0, zorder=5)
    ax.set_xticks(range(len(G))); ax.set_xticklabels([f"{g:g}" for g in G])
    ax.set_yticks(range(len(G))); ax.set_yticklabels([f"{g:g}" for g in G])
    ax.set_xlabel(r"$\alpha_1$  (branch 1: clades 1–4)")
    ax.set_ylabel(r"$\alpha_2$  (branch 2: the other 14)")
    ax.set_title(f"{s}   best {Z[bi]:.4f} at "
                 rf"$(\alpha_1,\alpha_2)$=({G[bi[1]]:g},{G[bi[0]]:g})" "\n"
                 f"best on diagonal {dg[bd]:.4f} at α={G[bd]:g}   →  gain {Z[bi]-dg[bd]:+.4f}",
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=.046, label="soft accuracy")
fig.suptitle("Two-branch α on the CORRECTED per-clade cost, exact OT   "
             "(dashed = uniform α; red circle = grid best; white square = best uniform)",
             fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(HERE / "two_branch_exact.png", dpi=155)
print("wrote two_branch_exact.png")
