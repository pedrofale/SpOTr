"""The headline result figure."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parent
SECS = ["M3-4", "M3-1", "M3-2"]

S = pd.concat([pd.read_csv(HERE / "split_estimation.csv"),
               pd.read_csv(HERE / "split_estimation_more.csv")], ignore_index=True)
SH = pd.read_csv(HERE / "split_shuffled.csv")
# expression-only baseline measured on the SAME half-section subsets the split protocol maps,
# not on the full section -- otherwise the bars are not comparable
EXPR = {sec: float(S[(S.section == sec) & (S.w == 0)].err.mean()) for sec in SECS}

fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.9))

# ---- A: the ladder ---------------------------------------------------------------------------
rowsA = []
for sec in SECS:
    g = S[(S.section == sec) & (S.w > 0)]
    fg = g[g.arm == "FGW"].groupby(["seed", "half"]).err.min()
    l2 = g[g.arm == "learned rank 2"].groupby(["seed", "half"]).err.min()
    sh = SH[(SH.section == sec) & (SH.w > 0)].groupby(["seed", "half"]).err.min()
    rowsA.append(dict(sec=sec, expr=EXPR[sec], shuf=sh.mean(), fgw=fg.mean(), learned=l2.mean(),
                      fgw_v=fg.values, l_v=l2.values))
A = pd.DataFrame(rowsA)
x = np.arange(3); wd = 0.2
ax[0].bar(x - 1.5*wd, A.expr, wd, color="#c9ced3", label="expression only")
ax[0].bar(x - 0.5*wd, A.shuf, wd, color="#e6d3c0", label="tree, transform fitted on\na destroyed relationship")
ax[0].bar(x + 0.5*wd, A.fgw, wd, color="#7a99b5", label="tree, square loss (FGW)")
ax[0].bar(x + 1.5*wd, A.learned, wd, color="#1b6ca8", label="tree, learned loss")
for i in range(3):
    for j, (v, c) in enumerate([(A.expr[i], "#6b7278"), (A.shuf[i], "#9a7f63"),
                                (A.fgw[i], "#3f5f7d"), (A.learned[i], "#0f4a76")]):
        ax[0].text(x[i] + (j - 1.5) * wd, v + 8, f"{v:.0f}", ha="center", fontsize=8, color=c)
ax[0].set_xticks(x); ax[0].set_xticklabels(SECS)
ax[0].set_ylabel("median placement error (um)")
ax[0].set_title("A.  Placing cells with no position information\n(transform fitted on a disjoint "
                "half of the cells)", fontsize=10.5)
ax[0].legend(frameon=False, fontsize=8, loc="lower left")
ax[0].set_ylim(0, 720)

# ---- B: paired splits ------------------------------------------------------------------------
CL = {"M3-4": "#1b6ca8", "M3-1": "#d1852a", "M3-2": "#2f7d4f"}
for i, sec in enumerate(SECS):
    r = A[A.sec == sec].iloc[0]
    for a_, b_ in zip(r.fgw_v, r.l_v):
        ax[1].plot([0, 1], [a_, b_], "-o", color=CL[sec], lw=1.1, ms=3.5, alpha=.7,
                   label=sec if a_ == r.fgw_v[0] else None)
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["FGW", "learned loss"])
ax[1].set_ylabel("median placement error (um)")
n_better = sum(int(b_ < a_) for _, r in A.iterrows() for a_, b_ in zip(r.fgw_v, r.l_v))
n_tot = sum(len(r.fgw_v) for _, r in A.iterrows())
ax[1].set_title(f"B.  Every split, paired\nlearned loss wins {n_better} of {n_tot}", fontsize=10.5)
ax[1].legend(frameon=False, fontsize=8.5)

# ---- C: sensitivity to the trade-off weight --------------------------------------------------
for sec, ls in zip(SECS, ["-", "--", ":"]):
    g = S[(S.section == sec) & (S.w > 0)].groupby(["arm", "w"], as_index=False).err.mean()
    for arm, c in [("FGW", "#7a99b5"), ("learned rank 2", "#1b6ca8")]:
        h = g[g.arm == arm].sort_values("w")
        ax[2].plot(h.w / h.w.iloc[3], h.err, ls, color=c, lw=2.0, marker="o", ms=3.5,
                   label=(arm.replace("learned rank 2", "learned loss") if sec == "M3-4" else None))
    ax[2].plot([], [], color="0.5", ls=ls, label=sec)
ax[2].set_xscale("log")
ax[2].set_xlabel("weight on the tree term, relative to FGW's optimum")
ax[2].set_ylabel("median placement error (um)")
ax[2].set_title("C.  The learned loss can be trusted further,\nand degrades far more gently",
                fontsize=10.5)
ax[2].legend(frameon=False, fontsize=8, ncol=2)

for a_ in ax:
    a_.grid(alpha=.22); a_.set_axisbelow(True)
    for sp in ("top", "right"): a_.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(HERE / "result.png", dpi=140, facecolor="white")
print(A[["sec", "expr", "shuf", "fgw", "learned"]].round(1).to_string(index=False))
print(f"\nlearned beats FGW in {n_better}/{n_tot} splits")
print("wrote result.png")
