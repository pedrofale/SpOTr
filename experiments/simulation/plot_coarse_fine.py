import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
FG, BG, MID = "#1b1b1f", "#ffffff", "#8a8a94"
ACC, WARN, OK = "#c0392b", "#d68910", "#2e7d5b"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MID,
                     "axes.labelcolor": FG, "text.color": FG, "xtick.color": MID,
                     "ytick.color": MID, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": BG, "axes.facecolor": BG, "font.size": 9})

fig, ax = plt.subplots(1, 4, figsize=(15.5, 3.6))

# --- 1. alpha sweep, seed 0 ---
q1 = pd.read_csv(HERE / "q1_coarse_fine.csv")
a = ax[0]
a.plot(q1.alpha, q1.err, "o-", color=ACC, lw=2, ms=5, label="expected distance")
a.axhline(q1.err[0], color=MID, ls=":", lw=1)
a.set_xlabel("global alpha (structure weight)"); a.set_ylabel("expected transport distance")
a.set_title("1. Seed 0: the tree helps\n(interior optimum at alpha 0.20)", fontsize=10, loc="left")
i = int(q1.err.idxmin())
a.annotate(f"-12.4%", xy=(q1.alpha[i], q1.err[i]), xytext=(q1.alpha[i]+.12, q1.err[i]+.012),
           color=OK, fontsize=9, arrowprops=dict(arrowstyle="->", color=OK, lw=1.2))

# --- 2. replication across seeds ---
rp = pd.read_csv(HERE / "replicate_q1.csv")
a = ax[1]
x = np.arange(len(rp)); w = 0.38
a.bar(x - w/2, rp.gain, w, color=ACC, label="all clades")
a.bar(x + w/2, rp.gain_tf, w, color=MID, label="tree-fine clades only")
a.axhline(0, color=FG, lw=.8)
a.set_xticks(x); a.set_xticklabels([f"seed {s}" for s in rp.seed], fontsize=8)
a.set_ylabel("subclade-accuracy gain")
a.set_title("2. It does not replicate\nand the mechanism fails", fontsize=10, loc="left")
a.legend(frameon=False, fontsize=8, loc="upper right")
a.text(.30, .80, f"mean gain  all +{rp.gain.mean():.4f}\n       tree-fine +{rp.gain_tf.mean():.4f}\nzero in 2 of 5 seeds",
       transform=a.transAxes, fontsize=8, color=WARN, va="top")

# --- 3. per-clade phi: signal but no headroom ---
q4 = pd.read_csv(HERE / "q4_headroom.csv").set_index("metric")
a = ax[2]
row = q4.loc["acc_sub"]
names = ["uniform\nphi = 1", "ORACLE\nper-clade", "shuffled", "inverted"]
vals = [row.glob, row.oracle, row.shuf, row.inv]
cols = [OK, ACC, MID, "#d5d5dc"]
a.bar(names, vals, color=cols)
a.errorbar([2], [row.shuf], yerr=[row.shuf_sd], color=FG, capsize=4, lw=1, fmt="none")
a.set_ylim(0.30, 0.46); a.set_ylabel("subclade accuracy")
a.set_title("3. phi carries signal, but a\nuniform maximum still wins", fontsize=10, loc="left")
a.text(.02, .92, "oracle > shuffled (z=+2.1) > inverted", transform=a.transAxes,
       fontsize=8, color=MID)

# --- 4. learned phi does not recover the split ---
a = ax[3]
q5 = pd.read_csv(HERE / "q5_learn_phi.csv")[["init", "auc"]].assign(loss="clade (wrapper)")
q6 = pd.read_csv(HERE / "q6_fineloss.csv")
q6 = q6.assign(loss=q6.gran.map({"coarse (clade)": "clade (direct)",
                                 "fine (subclade)": "subclade"}))[["init", "auc", "loss"]]
al = pd.concat([q5, q6])
for j, (nm, sub) in enumerate(al.groupby("loss", sort=False)):
    a.scatter(sub.init + (j - 1) * 0.02, sub.auc, s=55, label=nm,
              color=[ACC, WARN, MID][j], zorder=3)
a.axhline(0.5, color=FG, ls="--", lw=1)
a.text(0.82, 0.515, "no signal", fontsize=8, color=FG)
a.set_ylim(-0.03, 1.0); a.set_xticks([0.2, 0.5, 0.8])
a.set_xlabel("phi initialisation"); a.set_ylabel("AUC(learned phi -> tree-fine)")
a.set_title("4. The learned phi does not\nrecover the split, from any init", fontsize=10, loc="left")
a.legend(frameon=False, fontsize=8, loc="upper left")
a.annotate("diverged\n(NaN)", xy=(0.8, 0.61), xytext=(0.66, 0.80), fontsize=7.5, color=WARN,
           ha="center", arrowprops=dict(arrowstyle="->", color=WARN, lw=.9))

fig.tight_layout()
fig.savefig(HERE / "coarse_fine_summary.png", dpi=170, bbox_inches="tight", facecolor=BG)
print("wrote coarse_fine_summary.png")
