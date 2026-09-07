"""Cross-section application figure under the clone-spread cost."""
import pickle, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parent
OUT = pickle.load(open(HERE / "application_cs.pkl", "rb"))
A = pd.read_csv(HERE / "application_cs.csv")
SRC, TGT = "M3-2", "M3-1"
d = OUT[(SRC, TGT)]
xy = d["xy_st"]; clades = d["clades"]; TRUE = d["TRUE"]; lab = d["labels"]
maps = {k: v[1] for k, v in d["best"].items()}
order = np.argsort(-TRUE.sum(1))[:4]
fig, ax = plt.subplots(3, 4, figsize=(14.5, 10.2))
for col, k in enumerate(order):
    dens = {f"{TGT}'s own cells\n(ground truth)": TRUE[k] / TRUE[k].sum()}
    for nm, key in [("expression only", "expression only"),
                    ("expression + tree\n(clone spread)", "clone spread")]:
        P = maps[key][lab == clades[k]].sum(0)
        dens[nm] = P / P.sum()
    for row, (nm, v) in enumerate(dens.items()):
        a = ax[row, col]
        a.scatter(xy[:, 0], xy[:, 1], s=26, c=v, cmap="magma_r", vmin=0,
                  vmax=np.quantile(v, .985), edgecolors="none")
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_visible(False)
        if col == 0: a.set_ylabel(nm, fontsize=10)
        if row == 0: a.set_title(f"clade {clades[k]}   ({int(TRUE[k].sum())} cells)", fontsize=10.5)
r_e = A.query("src==@SRC and tgt==@TGT and arm=='expression only'").r.max()
r_c = A.query("src==@SRC and tgt==@TGT and arm=='clone spread'").r.max()
r_n = A.query("src==@SRC and tgt==@TGT and w==-1").r.mean()
fig.suptitle(f"{SRC}'s cells placed onto {TGT}'s spots — no shared cell, no position in {TGT} used "
             f"to build the cost\nagreement with {TGT}'s own clade territories:  expression only "
             f"r = {r_e:.2f}   |   with the tree r = {r_c:.2f}   |   clade labels permuted "
             f"r = {r_n:.2f}", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(HERE / "application_cs.png", dpi=135, facecolor="white", bbox_inches="tight")
print("wrote application_cs.png")
