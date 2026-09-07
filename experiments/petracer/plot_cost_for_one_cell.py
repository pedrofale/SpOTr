"""What the two costs look like as maps over the tissue, for a cell the tree RESCUES and a
cell the tree MISLEADS. Both happen; the sweep result is the average of the two."""
import numpy as np, ot, pickle
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parent
INK, GRID = "#0b0b0b", "#d8d8d4"
SEC = "M3-4"
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
T = pickle.loads((HERE / "where_cells_go.pkl").read_bytes())
constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, "square_loss")
Lgw = ot.gromov.gwggrad(constC, hC1, hC2, np.asarray(T[0.75]))

e0 = np.linalg.norm(xy_st[np.asarray(T[0.0]).argmax(1)] - xy_sc, axis=1)
e6 = np.linalg.norm(xy_st[np.asarray(T[0.75]).argmax(1)] - xy_sc, axis=1)
delta = e0 - e6                                   # positive = tree helped
frac = float((delta > 0).mean())
GOOD = int(np.argsort(-delta)[0])
BAD = int(np.argsort(delta)[0])

def norm(v): return (v - v.min()) / (v.max() - v.min() + 1e-12)

fig, ax = plt.subplots(2, 4, figsize=(14.6, 8.0))
for r, (cell, tag, col) in enumerate([(GOOD, "the tree RESCUES this cell", "#1a7a3a"),
                                      (BAD, "the tree MISLEADS this cell", "#a83c10")]):
    # the spot the model ACTUALLY assigns, from the coupling -- not the argmin of the cost row.
    # OT's marginal constraints mean those differ, and the assignment is what the result is made of.
    j0 = int(np.asarray(T[0.0])[cell].argmax())
    j6 = int(np.asarray(T[0.75])[cell].argmax())
    panels = [("expression cost " + r"$C^F$", norm(M[cell]), j0, r"assigned at $\alpha$=0"),
              ("tree cost " + r"$L$", norm(Lgw[cell]), None, None),
              (r"combined, $\alpha$=0.75", norm(0.4*M[cell] + 0.6*Lgw[cell]), j6,
               r"assigned at $\alpha$=0.75")]
    for k, (ttl, v, jj, jlab) in enumerate(panels):
        a = ax[r, k]
        sc = a.scatter(xy_st[:, 0], xy_st[:, 1], c=v, s=20, cmap="magma_r", edgecolors="none")
        a.scatter(*xy_sc[cell], s=210, marker="*", c="#1a9e4b", edgecolors="white", lw=1.2, zorder=5)
        if jj is not None:
            a.scatter(*xy_st[jj], s=130, facecolors="none", edgecolors="#2a78d6", lw=2.4, zorder=5)
            er = np.linalg.norm(xy_st[jj] - xy_sc[cell])
            a.set_xlabel(f"{jlab}: {er:.0f} µm from truth", fontsize=9,
                         color="#a83c10" if er > 400 else "#1a7a3a", fontweight="bold")
        else:
            a.set_xlabel("(enters only through the sum)", fontsize=8.5, color="#666")
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        if r == 0: a.set_title(ttl, fontsize=10.5)
        fig.colorbar(sc, ax=a, fraction=.046, pad=.03).ax.tick_params(labelsize=6.5)
    a = ax[r, 3]
    sib = np.argsort(CT[cell])[1:26]
    a.scatter(xy_sc[:, 0], xy_sc[:, 1], s=2.5, c="#e8e8e6", edgecolors="none")
    a.scatter(xy_sc[sib, 0], xy_sc[sib, 1], s=28, c="#eb6834", edgecolors="none")
    a.scatter(*xy_sc[cell], s=210, marker="*", c="#1a9e4b", edgecolors="white", lw=1.2, zorder=5)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    if r == 0: a.set_title("its 25 nearest relatives\n(green star = the cell)", fontsize=10.5)
    sp = np.median(np.linalg.norm(xy_sc[sib] - xy_sc[cell], axis=1))
    a.set_xlabel(f"relatives sit {sp:.0f} µm away (median)", fontsize=9)
    ax[r, 0].set_ylabel(tag, fontsize=11, color=col, fontweight="bold")

fig.suptitle("The two costs as maps over the tissue.  Green star = the cell's true location, "
             "blue ring = the spot the model assigns it.\n"
             f"The tree improves {frac:.0%} of cells and worsens the rest — the sweep result "
             "is the average of these two pictures.", fontsize=11.5, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(HERE / "cost_for_one_cell.png", dpi=150, facecolor="white")
print(f"wrote cost_for_one_cell.png  (rescued {GOOD}, misled {BAD}, tree helps {frac:.1%})")
