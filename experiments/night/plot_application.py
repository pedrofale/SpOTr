"""Cross-section application: one section's cells placed into another section's spots."""
import pickle, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parent
OUT = pickle.load(open(HERE / "application.pkl", "rb"))
A = pd.read_csv(HERE / "application.csv")

SRC, TGT = "M3-2", "M3-1"
d = OUT[(SRC, TGT)]
xy = d["xy_st"]; clades = d["clades"]; TRUE = d["TRUE"]
lab = d["labels"]
maps = {k: v[1] for k, v in d["best"].items()}

order = np.argsort(-TRUE.sum(1))[:4]
fig, ax = plt.subplots(3, 4, figsize=(14.5, 10.4))
for col, k in enumerate(order):
    dens = {"the target's own cells\n(ground truth)": TRUE[k] / TRUE[k].sum()}
    for nm, key in [("expression only", "expression only"), ("expression + tree\n(learned loss)",
                                                             "learned")]:
        P = maps[key][lab == clades[k]].sum(0)
        dens[nm] = P / P.sum()
    for row, (nm, v) in enumerate(dens.items()):
        a = ax[row, col]
        a.scatter(xy[:, 0], xy[:, 1], s=26, c=v, cmap="magma_r",
                  vmin=0, vmax=np.quantile(v, .985), edgecolors="none")
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_visible(False)
        if col == 0:
            a.set_ylabel(nm, fontsize=10)
        if row == 0:
            a.set_title(f"clade {clades[k]}   ({int(TRUE[k].sum())} cells in {TGT})", fontsize=10.5)

r_expr = A.query("src==@SRC and tgt==@TGT and arm=='expression only'").r.max()
r_learn = A.query("src==@SRC and tgt==@TGT and arm=='learned'").r.max()
r_null = A.query("src==@SRC and tgt==@TGT and arm.str.contains('NULL')", engine="python").r.mean()
fig.suptitle(f"{SRC}'s cells placed onto {TGT}'s spots — no cell is shared, no position in "
             f"{TGT} is used to build the cost.\nAgreement with {TGT}'s own clade territories:  "
             f"expression only r = {r_expr:.2f}   |   expression + tree r = {r_learn:.2f}   |   "
             f"clade labels permuted r = {r_null:.2f}", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(HERE / "application.png", dpi=135, facecolor="white")
print("wrote application.png")

# summary table
S = A[A.w != -1].groupby(["src", "tgt", "arm"]).r.max().reset_index()
piv = S.pivot_table(index=["src", "tgt"], columns="arm", values="r")
print(piv.round(3).to_string())
print("\nnulls:", A[A.w == -1].groupby(["src", "tgt"]).r.mean().round(3).to_dict())
