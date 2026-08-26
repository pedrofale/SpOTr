"""Cross-system comparison of SpOTr's core premise."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
prem = pd.read_csv(HERE / "etracer_premise.csv")
coh = pd.read_csv(HERE / "etracer_coherence.csv")

fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.8))

# --- A: clone spatial coherence across every system tested ---
systems = [
    ("BCSA3/B1\n(CNV)", 0.107, "#c44"), ("BCSA3/D1\n(CNV)", 0.169, "#c44"),
    ("BCSA6\n(CNV)", 0.035, "#c44"),
    ("GBM ZH916bulk\n(CNV)", 0.799, "#e8a33d"), ("GBM ZH916T1\n(CNV)", 0.671, "#e8a33d"),
    ("PEtracer\n(lineage)", 0.310, "#3a7d44"),
    ("eTRACER BL\n(lineage)", 0.205, "#3a7d44"),
    ("eTRACER EE\n(lineage)", 0.258, "#3a7d44"),
    ("eTRACER LE\n(lineage)", 0.309, "#3a7d44"),
]
x = np.arange(len(systems))
axs[0].bar(x, [s[1] for s in systems], color=[s[2] for s in systems], edgecolor="k", linewidth=.5)
axs[0].axhline(0.30, ls="--", c="k", lw=1.1)
axs[0].text(len(x) - .4, .31, "target > 0.3", ha="right", va="bottom", fontsize=8)
axs[0].set_xticks(x); axs[0].set_xticklabels([s[0] for s in systems], fontsize=7.5, rotation=35, ha="right")
axs[0].set_ylabel("clone spatial coherence")
axs[0].set_title("Do clones form spatial territories?", fontsize=11)

# --- B: eTRACER per-clone tree-vs-space correlation vs its own null ---
sig = prem["p"] < 0.05
axs[1].hist(prem.loc[~sig, "r"], bins=np.arange(-.15, .30, .02), color="silver",
            edgecolor="k", linewidth=.4, label=f"n.s. ({(~sig).sum()})")
axs[1].hist(prem.loc[sig, "r"], bins=np.arange(-.15, .30, .02), color="#3a7d44",
            edgecolor="k", linewidth=.4, label=f"p<0.05 ({sig.sum()})")
axs[1].axvline(0, c="k", lw=1)
axs[1].axvline(prem["r"].mean(), c="crimson", lw=1.6, ls="--",
               label=f"mean r = {prem['r'].mean():.3f}")
axs[1].set_xlabel("r(tree distance, spatial distance) within a clone")
axs[1].set_ylabel("clones")
axs[1].set_title("eTRACER: premise holds, but weakly\n"
                 "18/54 significant vs 2.7 expected (p=6e-11)", fontsize=11)
axs[1].legend(fontsize=8)

# --- C: coherence rises as immune escape progresses ---
axs[2].plot(range(3), coh["coherence"], "o-", color="#3a7d44", lw=2, ms=9)
for i, row in coh.iterrows():
    axs[2].annotate(f"{row['coherence']:.3f}\n{int(row['clones'])} clones",
                    (i, row["coherence"]), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=8)
axs[2].set_xticks(range(3))
axs[2].set_xticklabels(["baseline", "early escape", "late escape"])
axs[2].set_ylim(0.15, 0.38)
axs[2].set_ylabel("clone spatial coherence")
axs[2].set_title("Clones segregate as immune escape proceeds", fontsize=11)

fig.suptitle("SpOTr's premise across four systems: real in genuine lineage data, "
             "but a small effect everywhere", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.90])
out = HERE / "premise_across_systems.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
