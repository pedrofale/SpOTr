import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import cKDTree
import sys
sys.path.insert(0, ".")
from premise import load_coords

NB = Path("/mnt/data/pedro/BCSA/BCSA2/numbat_out_visium_both")
SP = {"TumC2": "/mnt/frisen_nas/shared/shared/20251112_BCSA3_for_pedro/BCSA2/BCSA2_TumC2_visium/spatial",
      "TumE2": "/mnt/frisen_nas/shared/shared/20251112_BCSA3_for_pedro/BCSA2/BCSA2_TumE2_visium/spatial"}
cp = pd.read_csv(NB/"clone_post_2.tsv", sep="\t")
cid = np.array([1,2,3,4,5])
yc=[f"p_y_{i}" for i in cid]
COL = {1:"#9e9e9e",2:"#4C72B0",3:"#DD8452",4:"#55A868",5:"#C44E52"}

fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.6))
res = {}
for row, sec in enumerate(["TumC2","TumE2"]):
    v = cp[cp["cell"].str.startswith(sec+"_")].copy()
    v["bc"] = v["cell"].str.replace(f"^{sec}_", "", regex=True)
    co = load_coords(SP[sec]); v = v[v["bc"].isin(co.index)].reset_index(drop=True)
    xy = co.loc[v["bc"]].to_numpy()
    Py = v[yc].to_numpy()
    lab_y = cid[Py.argmax(1)]; lab_j = v["clone_opt"].to_numpy()
    for col, (lab, ttl) in enumerate([(lab_y, "ALLELE-only (argmax p_y)"), (lab_j, "JOINT (clone_opt)")]):
        ax = axes[row, col]
        for c in cid:
            m = lab == c
            if m.sum():
                ax.scatter(xy[m,1], -xy[m,0], s=3.2, c=COL[c], label=f"clone {c} (n={m.sum()})", linewidths=0)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{sec} — {ttl}", fontsize=11)
        ax.legend(markerscale=4, fontsize=7, loc="upper right", framealpha=.85)
    # coherence panel
    _, nbr = cKDTree(xy).query(xy, k=13); nbr = nbr[:,1:]
    ax = axes[row,2]
    vc = pd.Series(lab_y).value_counts()
    big = [c for c in cid if vc.get(c,0) >= 20]
    rng = np.random.default_rng(3)
    exc, lo, hi = [], [], []
    for c in big:
        m = lab_y==c
        exc.append((lab_y[nbr][m]==c).mean() - m.mean())
        nl = np.empty(999)
        for t in range(999):
            pl = rng.permutation(lab_y); nb2 = pl[nbr]; mm = pl==c
            nl[t] = (nb2[mm]==c).mean() - mm.mean()
        lo.append(np.percentile(nl,2.5)); hi.append(np.percentile(nl,97.5))
    x = np.arange(len(big))
    ax.bar(x, exc, color=[COL[c] for c in big], width=.62)
    ax.fill_between([-.6, len(big)-.4], min(lo), max(hi), color="k", alpha=.12, zorder=0)
    ax.axhline(0, color="k", lw=.8)
    ax.axhline(0.309, color="#C44E52", ls="--", lw=1.2)
    ax.axhline(0.205, color="#C44E52", ls=":", lw=1.2)
    ax.text(len(big)-.45, 0.309, " eTRACER late escape 0.309", fontsize=7, va="bottom", ha="right", color="#C44E52")
    ax.text(len(big)-.45, 0.205, " eTRACER baseline 0.205", fontsize=7, va="bottom", ha="right", color="#C44E52")
    ax.set_xticks(x); ax.set_xticklabels([f"clone {c}" for c in big], fontsize=8)
    ax.set_ylim(-0.03, 0.36); ax.set_ylabel("coherence (excess over chance)", fontsize=9)
    ax.set_title(f"{sec} — allele-only coherence, k=12\n(grey band = 95% of 999-permutation null)", fontsize=10)
    ax.set_xlim(-.6, len(big)-.4)
    res[sec] = float(np.mean(exc))
fig.suptitle("BCSA2 — SpOTr premise on Visium spots, ALLELE-ONLY clone calls (numbat p_y_*, shared 5-clone tree across both sections)",
             fontsize=12.5, y=.985)
fig.tight_layout(rect=[0,0,1,.96])
fig.savefig("bcsa2_premise.png", dpi=155)
print(res)
