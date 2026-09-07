import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parent
PE = HERE.parent / "petracer"
L = pd.read_csv(HERE / "sim_branch_alpha.csv")
SN = pd.read_csv(HERE / "sim_branch_null.csv")
RN = pd.read_csv(PE / "branch_null_M3-1.csv")
U = sorted(L.u1.unique())
fig, ax = plt.subplots(1, 3, figsize=(14.2, 4.4))

d = L[(L.seed == 0) & (L.c == 0.0)]
Z = d.pivot(index="u2", columns="u1", values="acc").reindex(index=U, columns=U).values
im = ax[0].imshow(Z, origin="lower", cmap="viridis", extent=[-.5, 4.5, -.5, 4.5])
for a in range(5):
    for b in range(5):
        ax[0].text(b, a, f"{Z[a,b]:.3f}", ha="center", va="center", fontsize=7.6,
                   color="white" if Z[a, b] < np.nanmax(Z)*.85 else "black")
ax[0].plot([-.5, 4.5], [-.5, 4.5], "w--", lw=1.2, alpha=.75)
bi = np.unravel_index(np.nanargmax(Z), Z.shape)
ax[0].scatter([bi[1]], [bi[0]], s=190, facecolors="none", edgecolors="r", lw=2.4, zorder=5)
ax[0].set_xticks(range(5)); ax[0].set_xticklabels([f"{u:g}" for u in U], fontsize=8)
ax[0].set_yticks(range(5)); ax[0].set_yticklabels([f"{u:g}" for u in U], fontsize=8)
ax[0].set_xlabel(r"$u_1$   branch planted TREE-INFORMATIVE")
ax[0].set_ylabel(r"$u_2$   branch planted TREE-BLIND")
ax[0].set_title("Simulation, planted structure (c=0)\n"
                f"best {np.nanmax(Z):.4f} off the diagonal", fontsize=10)
fig.colorbar(im, ax=ax[0], fraction=.046)

for a, R, ttl in [(ax[1], SN[SN.seed == 0], "Simulation: planted structure"),
                  (ax[2], RN.rename(columns={"gain_full": "gain"}), "PEtracer M3-1: real data")]:
    o = float(R[R.real].iloc[0]["gain"]); v = R[~R.real]["gain"].values
    a.hist(v, bins=np.linspace(-0.004, 0.042, 24), color="0.72", edgecolor="white",
           label=f"{len(v)} random splits\n(same sizes)")
    a.axvline(o, color="#d62728", lw=2.8, label=f"true branch split {o:+.4f}")
    a.axvline(v.mean(), color="0.35", ls="--", lw=1.6, label=f"null mean {v.mean():+.4f}")
    z = (o - v.mean()) / max(v.std(), 1e-9)
    a.set_title(f"{ttl}\nz = {z:+.2f}", fontsize=10)
    a.set_xlabel("two-branch gain  (best − best uniform)")
    a.set_ylabel("random splits")
    a.set_xlim(-0.004, 0.042)
    a.legend(frameon=False, fontsize=7.6, loc="upper left")
fig.suptitle("Same cost, same solver, same test: the method finds planted per-branch α "
             "and finds nothing in PEtracer", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(HERE / "sim_vs_real.png", dpi=155)
print("wrote sim_vs_real.png")
