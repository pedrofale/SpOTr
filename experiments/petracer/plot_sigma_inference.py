"""What the per-cell bandwidth estimate does and does not deliver."""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.neighbourhood as nb
INK, GRID = "#0b0b0b", "#d8d8d4"
plt.rcParams.update({"axes.edgecolor": GRID, "text.color": INK, "font.size": 9})
d = np.load(HERE / "petracerFAIR_M3-4.npz", allow_pickle=True)
DT = np.asarray(d["C_tree"], float); XY = np.asarray(d["xy_sc"], float)
X = XY / np.median(np.linalg.norm(XY - np.median(XY, 0), axis=1))
SIG = np.array([1, 1.5, 2, 3, 4, 5, 7, 10, 14, 20, 30, 45, 70])
sig, L = nb.estimate_sigma(DT, X, SIG)
DTd = DT.copy(); np.fill_diagonal(DTd, np.inf)
ess = np.empty(len(sig))
for s in np.unique(sig):
    m = sig == s; W = np.exp(-DTd[m] / s)
    ess[m] = W.sum(1) ** 2 / (W ** 2).sum(1)
o = np.argsort(DTd, 1)[:, :15]; cen = XY[o].mean(1)
spr = np.sqrt(((XY[o] - cen[:, None]) ** 2).sum(2).mean(1))
mot = np.linalg.norm(XY - cen, axis=1) / np.maximum(spr, 1e-9)
grp = np.digitize(mot, np.quantile(mot, [.25, .5, .75]))
COL = ["#1a6b3c", "#6aa84f", "#e0a53c", "#a83c10"]; LAB = ["settled", "q2", "q3", "motile"]
gl = SIG[L.mean(0).argmax()]; gain = L.max(1) - L[:, list(SIG).index(gl)]

fig, ax = plt.subplots(1, 4, figsize=(16.4, 4.3))
a = ax[0]
for k in range(4):
    a.plot(SIG, np.median(L[grp == k], 0), "o-", color=COL[k], lw=2.2, ms=4, label=LAB[k])
a.axvline(gl, color="0.4", ls="--", lw=1.5)
a.text(gl * 1.08, a.get_ylim()[0] + 1.2, f"best global\n$\\sigma$={gl:g}", fontsize=8.5,
       color="0.35")
a.set_xscale("log"); a.set_ylim(-8, 4)
a.set_xlabel("bandwidth $\\sigma$"); a.set_ylabel("median leave-one-out loglik")
a.set_title("There is an interior optimum,\nand it shifts right with motility", fontsize=10.5)
a.legend(frameon=False, fontsize=8.5, loc="lower right")

a = ax[1]
parts = a.violinplot([np.log10(sig[grp == k]) for k in range(4)], showmedians=True, widths=.85)
for b, c in zip(parts["bodies"], COL):
    b.set_facecolor(c); b.set_alpha(.6); b.set_edgecolor(c)
for key in ("cbars", "cmins", "cmaxes", "cmedians"):
    parts[key].set_color("0.3"); parts[key].set_linewidth(1.2)
for k in range(4):
    a.text(k + 1, np.log10(np.median(sig[grp == k])) + .06, f"{np.median(sig[grp==k]):g}",
           ha="center", fontsize=10, fontweight="bold", color=COL[k])
r = spearmanr(mot, sig)
a.set_xticks(range(1, 5)); a.set_xticklabels(LAB)
a.set_ylabel("fitted $\\hat\\sigma$  (log$_{10}$)")
a.set_title(f"Motile cells take a wider bandwidth,\nbut only just: $\\rho$ = "
            f"{r.statistic:+.2f}, medians 3 vs 5", fontsize=10.5)

a = ax[2]
bins = np.logspace(0, np.log10(max(ess.max(), 10)), 30)
a.hist(ess, bins=bins, color="#6a7f8f", edgecolor="white", lw=.6)
a.axvspan(1, 3, color="#a83c10", alpha=.18)
a.axvspan(500, ess.max() * 1.1, color="#a83c10", alpha=.18)
a.set_xscale("log")
a.set_xlabel("effective neighbourhood size at $\\hat\\sigma$  (Kish ESS)")
a.set_ylabel("cells")
a.set_title(f"40% of fits are degenerate\n{np.mean(ess<3):.0%} use 1-3 relatives, "
            f"{np.mean(ess>500):.0%} use the whole section", fontsize=10.5)

a = ax[3]
a.bar(range(4), [gain[grp == k].mean() for k in range(4)], .62, color=COL,
      edgecolor="white", lw=1.4)
for k in range(4):
    a.text(k, gain[grp == k].mean() + .03, f"{gain[grp==k].mean():.2f}", ha="center",
           fontsize=10.5, fontweight="bold")
a.set_xticks(range(4)); a.set_xticklabels(LAB); a.set_ylim(0, 1.45)
a.set_ylabel(f"loglik gained over the best global $\\sigma$={gl:g}")
a.set_title("The gain is ~1 nat and FLAT:\nnot a motile-cell effect", fontsize=10.5)
for a in ax:
    a.grid(alpha=.25); a.set_axisbelow(True); a.spines[["top", "right"]].set_visible(False)
fig.suptitle("Per-cell bandwidth inferred from lineage-traced ground truth "
             "(PEtracer M3-4, n = 2281): what it does and does not deliver", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(HERE / "sigma_inference.png", dpi=150, facecolor="white")
print("wrote sigma_inference.png")
