"""The result that matters: where does the model actually put the cells?

Everything else in this project is a summary statistic. This is the mapping itself -- for a few
clades, the cells' TRUE positions against the positions the model assigns them, with and
without the tree.
"""
import numpy as np, ot, pickle
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d8d4"
SEC, ALPHAS = "M3-4", [0.0, 0.75]
plt.rcParams.update({"axes.edgecolor": GRID, "text.color": INK, "font.size": 9})

d = np.load(HERE / f"petracerFAIR_{SEC}.npz", allow_pickle=True)
X_sc, X_st = np.asarray(d["X_sc"], np.float64), np.asarray(d["X_st"], np.float64)
CT = np.asarray(d["C_tree"], np.float64); CS = np.asarray(d["C_space"], np.float64)
xy_sc, xy_st = np.asarray(d["xy_sc"], float), np.asarray(d["xy"], float)
cc = np.asarray(d["cell_clade"]).astype(str)
n, m = len(cc), X_st.shape[0]
p, q = np.ones(n)/n, np.ones(m)/m
M = ot.dist(X_sc, X_st); M = M/M.max()
C1, C2 = CT/CT.mean(), CS/CS.mean()

cache = HERE / "where_cells_go.pkl"
if cache.exists():
    T = pickle.loads(cache.read_bytes())
else:
    T = {}
    for a in ALPHAS:
        T[a] = ot.gromov.fused_gromov_wasserstein(M, C1, C2, p, q, loss_fun="square_loss",
                                                  alpha=a, max_iter=200, tol_rel=1e-9,
                                                  tol_abs=1e-9)
        print(f"  solved alpha={a}", flush=True)
    cache.write_bytes(pickle.dumps(T))

# three clades that hold clear territories, largest first
sizes = {c: (cc == c).sum() for c in np.unique(cc)}
spread = {c: np.median(np.linalg.norm(xy_sc[cc == c] - xy_sc[cc == c].mean(0), axis=1))
          for c in sizes if sizes[c] >= 120}
PICK = sorted(spread, key=lambda c: spread[c])[:3]

fig, ax = plt.subplots(len(PICK), 3, figsize=(11.4, 3.6 * len(PICK)))
for r, cl in enumerate(PICK):
    s = cc == cl
    for k, (ttl, pts) in enumerate([
            ("where they really are", xy_sc[s]),
            (r"expression only ($\alpha$=0)", xy_st[np.asarray(T[0.0])[s].argmax(1)]),
            (r"expression + tree ($\alpha$=0.75)", xy_st[np.asarray(T[0.75])[s].argmax(1)])]):
        a = ax[r, k]
        a.scatter(xy_sc[:, 0], xy_sc[:, 1], s=3, c="#e8e8e6", edgecolors="none", zorder=0)
        a.scatter(pts[:, 0], pts[:, 1], s=14, c="#2a78d6" if k == 0 else "#eb6834",
                  edgecolors="none", alpha=.8, zorder=2)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        if r == 0:
            a.set_title(ttl, fontsize=11)
        if k == 0:
            a.set_ylabel(f"clade {cl}\n{int(s.sum())} cells", fontsize=10)
        if k > 0:
            err = np.median(np.linalg.norm(pts - xy_sc[s], axis=1))
            a.text(.5, -.06, f"median error {err:.0f} µm", transform=a.transAxes,
                   ha="center", fontsize=9.5,
                   color="#a83c10" if k == 1 else "#1a7a3a", fontweight="bold")
fig.suptitle("Where the model puts each clade's cells, PEtracer M3-4\n"
             "grey = all cells in the section; the tree pulls each clade back into its own "
             "territory", fontsize=12.5, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(HERE / "where_cells_go.png", dpi=150, facecolor="white")
print("wrote where_cells_go.png")
