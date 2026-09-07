"""Why the learned loss wins: it is sharper at the displacements that matter."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parent
G = pd.read_csv(HERE / "margin.csv"); L = pd.read_csv(HERE / "local_margin.csv")
SECS = ["M3-4", "M3-1", "M3-2"]
KEEP = {"square loss (FGW)": ("#9aa7b1", "square loss (FGW)"),
        "HGR rank 2": ("#1b6ca8", "learned loss")}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for k, (c, lab) in KEEP.items():
    h = G[(G.section == "M3-4") & (G.loss == k)].sort_values("eps")
    ax[0].plot(h.eps * 100, h.norm, "o-", color=c, lw=2.3, ms=5, label=lab)
ax[0].set_xlabel("% of cells moved to a random spot anywhere in the section")
ax[0].set_ylabel("loss, scaled:  0 = truth,  1 = random map")
ax[0].set_title("A.  Against gross corruption the two losses\nare indistinguishable  (M3-4)",
                fontsize=10.5)
ax[0].legend(frameon=False, fontsize=9)

for k, (c, lab) in KEEP.items():
    h = L[(L.section == "M3-4") & (L.loss == k)].sort_values("disp")
    ax[1].plot(h.disp, h.norm, "o-", color=c, lw=2.3, ms=5, label=lab)
ax[1].axvline(364, color="0.5", ls=":", lw=1.6)
ax[1].text(372, .05, "the error we are\ntrying to reduce", fontsize=8, color="0.4")
ax[1].set_xlabel("median displacement of the corrupted map (um)")
ax[1].set_ylabel("loss, scaled")
ax[1].set_title("B.  Against LOCAL corruption — the regime\nthat sets the placement error — it is not",
                fontsize=10.5)
ax[1].legend(frameon=False, fontsize=9)

sl = []
for sec in SECS:
    for k, (c, lab) in KEEP.items():
        h = L[(L.section == sec) & (L.loss == k)].sort_values("disp")
        h = h[h.disp < 700]
        sl.append(dict(section=sec, loss=lab, slope=np.polyfit(h.disp, h.norm, 1)[0] * 100))
SL = pd.DataFrame(sl)
x = np.arange(3); wd = .35
for i, (k, (c, lab)) in enumerate(KEEP.items()):
    v = SL[SL.loss == lab].set_index("section").loc[SECS, "slope"].values
    ax[2].bar(x + (i - .5) * wd, v, wd, color=c, label=lab)
    for j, vv in enumerate(v):
        ax[2].text(x[j] + (i - .5) * wd, vv + .002, f"{vv:.3f}", ha="center", fontsize=8.5)
ax[2].set_xticks(x); ax[2].set_xticklabels(SECS)
ax[2].set_ylabel("loss increase per 100 um of displacement")
ax[2].set_title("C.  The learned loss is 42-65% sharper\nat short range, in every section",
                fontsize=10.5)
ax[2].legend(frameon=False, fontsize=9)
for a in ax:
    a.grid(alpha=.22); a.set_axisbelow(True)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(HERE / "margin_fig.png", dpi=140, facecolor="white")
print(SL.pivot_table(index="section", columns="loss", values="slope").round(4).to_string())
print("wrote margin_fig.png")
