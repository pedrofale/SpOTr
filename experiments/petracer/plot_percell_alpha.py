import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID_C = "#0b0b0b", "#52514e", "#d8d8d4"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
GRID = [0.0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90]
SEC = "M3-4"
E = np.load(f"percell_err_{SEC}.npy")
d = np.load(f"petracer_{SEC}.npz", allow_pickle=True)
cc = np.asarray(d["cell_clade"]).astype(str)
clades = list(np.asarray(d["clades"]).astype(str))
cidx = np.array([clades.index(c) for c in cc])
best = np.array(GRID)[E.argmin(1)]

fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3))

ax = axes[0]
cnt = [int((best == v).sum()) for v in GRID]
ax.bar(range(len(GRID)), cnt, 0.7, color=[C2 if v == 0 else C1 for v in GRID],
       edgecolor="white", linewidth=1.2)
ax.set_xticks(range(len(GRID))); ax.set_xticklabels([f"{v:g}" for v in GRID])
ax.set_xlabel(r"the $\alpha$ that places this cell best"); ax.set_ylabel("cells")
ax.set_title(f"{cnt[0]/len(best):.0%} of cells are placed best with\nNO tree at all "
             r"($\alpha$=0, orange)", fontsize=10.5)
ax.grid(axis="y", color=GRID_C, lw=0.6); ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)

ax = axes[1]
for k, c_ in enumerate(clades):
    m = cidx == k
    if m.sum() < 30: continue
    mu = E[m].mean(0)
    ax.plot(GRID, mu/mu.mean(), c=GRID_C, lw=1.3, zorder=1)
mu_all = E.mean(0)
ax.plot(GRID, mu_all/mu_all.mean(), c=C1, lw=3, zorder=3, label="all cells")
ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("mean placement error (scaled per clade)")
ax.set_title("Each grey line is one clade.\nThey do differ — but not by much", fontsize=10.5)
ax.legend(frameon=False, fontsize=9)
ax.grid(color=GRID_C, lw=0.6); ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)

ax = axes[2]
rng = np.random.default_rng(0)
def eta2(v, lab):
    gm = v.mean()
    ssb = sum((v[lab==k].mean()-gm)**2*(lab==k).sum() for k in np.unique(lab))
    return ssb/max(((v-gm)**2).sum(),1e-12)
null = np.array([eta2(best, rng.permutation(cidx)) for _ in range(999)])
obs = eta2(best, cidx)
ax.hist(null, bins=40, color=GRID_C, edgecolor="white", linewidth=0.6, label="clade labels shuffled")
ax.axvline(obs, color=C2, lw=2.6, label=f"observed  $\\eta^2$ = {obs:.3f}")
ax.set_xlabel(r"$\eta^2$: variance in best $\alpha$ explained by clade")
ax.set_ylabel("permutations")
ax.set_title(f"Real, but small:\n{1-obs:.0%} of the variation is WITHIN clades", fontsize=10.5)
ax.legend(frameon=False, fontsize=8.5)
ax.grid(axis="y", color=GRID_C, lw=0.6); ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)

fig.suptitle(f"PEtracer {SEC}: does the $\\alpha$ a cell needs depend on its clade?",
             fontsize=13, y=0.985, color=INK)
fig.tight_layout(rect=[0,0,1,0.90])
fig.savefig("petracer_percell_alpha.png", dpi=150, facecolor="white")
print("wrote petracer_percell_alpha.png")
