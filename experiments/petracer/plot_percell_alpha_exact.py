"""Yesterday's per-cell alpha figure, regenerated with the EXACT solver."""
import numpy as np, ot, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
INK, INK2, GRIDC = "#0b0b0b", "#52514e", "#d8d8d4"
C1c, C2c = "#2a78d6", "#eb6834"
GRID = np.array([round(0.1*i, 1) for i in range(10)])
SEC = "M3-4"

d = np.load(HERE / f"petracer_{SEC}.npz", allow_pickle=True)
X_sc, X_st = np.asarray(d["X_sc"], np.float64), np.asarray(d["X_st"], np.float64)
CT = np.asarray(d["C_tree"], np.float64); CS = np.asarray(d["C_space"], np.float64)
xy_sc, xy_st = np.asarray(d["xy_sc"], float), np.asarray(d["xy"], float)
cc = np.asarray(d["cell_clade"]).astype(str)
clades = list(np.asarray(d["clades"]).astype(str))
cidx = np.array([clades.index(c) for c in cc])
n, m = len(cc), X_st.shape[0]
p, q = np.ones(n)/n, np.ones(m)/m
M = ot.dist(X_sc, X_st); M = M/M.max()
C1, C2 = CT/CT.mean(), CS/CS.mean()

f = HERE / f"percell_err_exact_{SEC}.npy"
if f.exists():
    E = np.load(f)
else:
    E = np.zeros((n, len(GRID))); t0 = time.time()
    for i, a in enumerate(GRID):
        T = ot.gromov.fused_gromov_wasserstein(M, C1, C2, p, q, loss_fun="square_loss",
                                               alpha=a, max_iter=200, tol_rel=1e-9, tol_abs=1e-9)
        E[:, i] = np.linalg.norm(xy_st[T.argmax(1)] - xy_sc, axis=1)
        print(f"  alpha {a:.1f} done ({time.time()-t0:.0f}s)", flush=True)
    np.save(f, E)
best = GRID[E.argmin(1)]

def eta2(v, lab):
    gm = v.mean()
    return sum((v[lab==k].mean()-gm)**2*(lab==k).sum() for k in set(lab))/max(((v-gm)**2).sum(),1e-12)

rng = np.random.default_rng(0)
null = np.array([eta2(best, rng.permutation(cidx)) for _ in range(999)])
obs = eta2(best, cidx)
pv = (np.sum(null >= obs) + 1)/1000
print(f"\nfrac of cells best at alpha=0: {np.mean(best==0):.3f}")
print(f"eta2 = {obs:.4f}   null {null.mean():.4f} +- {null.std():.4f}   p = {pv:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3))
ax = axes[0]
cnt = [int((best == v).sum()) for v in GRID]
ax.bar(range(len(GRID)), cnt, 0.7, color=[C2c if v == 0 else C1c for v in GRID],
       edgecolor="white", linewidth=1.2)
ax.set_xticks(range(len(GRID))); ax.set_xticklabels([f"{v:g}" for v in GRID])
ax.set_xlabel(r"the $\alpha$ that places this cell best"); ax.set_ylabel("cells")
ax.set_title(f"{cnt[0]/len(best):.0%} of cells are placed best with\nNO tree at all "
             r"($\alpha$=0, orange)", fontsize=10.5)
ax.grid(axis="y", color=GRIDC, lw=0.6); ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)

ax = axes[1]
for k in range(len(clades)):
    mk = cidx == k
    if mk.sum() < 30: continue
    mu = E[mk].mean(0); ax.plot(GRID, mu/mu.mean(), c=GRIDC, lw=1.3, zorder=1)
mu = E.mean(0)
ax.plot(GRID, mu/mu.mean(), c=C1c, lw=3, zorder=3, label="all cells")
ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("mean placement error (scaled per clade)")
ax.set_title("Each grey line is one clade", fontsize=10.5)
ax.legend(frameon=False, fontsize=9)
ax.grid(color=GRIDC, lw=0.6); ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)

ax = axes[2]
ax.hist(null, bins=40, color=GRIDC, edgecolor="white", linewidth=0.6, label="clade labels shuffled")
ax.axvline(obs, color=C2c, lw=2.6, label=f"observed  $\\eta^2$ = {obs:.3f}")
ax.set_xlabel(r"$\eta^2$: variance in best $\alpha$ explained by clade")
ax.set_ylabel("permutations")
ax.set_title(f"{1-obs:.0%} of the variation is WITHIN clades\n(p = {pv:.3f})", fontsize=10.5)
ax.legend(frameon=False, fontsize=8.5)
ax.grid(axis="y", color=GRIDC, lw=0.6); ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)

fig.suptitle(f"PEtracer {SEC}, EXACT solver: does the $\\alpha$ a cell needs depend on its clade?",
             fontsize=13, y=0.985, color=INK)
fig.tight_layout(rect=[0,0,1,0.90])
fig.savefig("petracer_percell_alpha_exact.png", dpi=150, facecolor="white")
print("wrote petracer_percell_alpha_exact.png")
