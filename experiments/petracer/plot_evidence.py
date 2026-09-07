"""Six plots, one claim each, all showing the raw points."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"      # validated categorical slots 1-3
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d8d4"
SEC = ["M3-1", "M3-2", "M3-4"]
COL = dict(zip(SEC, [C1, C2, C3]))

t = pd.concat([pd.read_csv(f"alpha_eps_grid_{s}.csv").assign(section=s) for s in SEC])

plt.rcParams.update({"axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
                     "axes.titlesize": 10, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True})
fig = plt.figure(figsize=(13.6, 8.4))
gs = fig.add_gridspec(2, 3, hspace=0.44, wspace=0.30, top=0.885, bottom=0.085,
                      left=0.062, right=0.982)

# --- 1. within-slice rho, the claim itself -----------------------------------
ax = fig.add_subplot(gs[0, 0])
groups = [("expression\nloss", "expr_loss", C1), ("composition\nloss", "comp_loss", C2),
          ("true\naccuracy", "acc", C3)]
rng = np.random.default_rng(0)
for i, (lab, col, c) in enumerate(groups):
    rs = np.array([spearmanr(g[col], g.ent)[0] for _, g in t.groupby(["section", "eps"])])
    ax.scatter(i + rng.uniform(-.13, .13, len(rs)), rs, s=34, c=c, alpha=0.8,
               edgecolor="white", linewidth=0.6, zorder=3)
    ax.plot([i - .26, i + .26], [rs.mean()] * 2, c=INK, lw=2.4, zorder=4)
    ax.text(i, 1.14, f"{rs.mean():+.2f}", ha="center", fontsize=9, color=INK)
ax.axhline(0, color=INK2, lw=1); ax.set_ylim(-1.05, 1.28)
ax.set_xticks(range(3)); ax.set_xticklabels([g[0] for g in groups], fontsize=8.5)
ax.set_ylabel(r"$\rho$ with coupling entropy, within slice")
ax.set_title("1. Each dot is one of 24 (section, $\epsilon$) slices.\nBoth losses track entropy; accuracy does not")

# --- 2. the same thing as trajectories ---------------------------------------
ax = fig.add_subplot(gs[0, 1])
for (s_, e_), g in t.groupby(["section", "eps"]):
    g = g.sort_values("alpha")
    ax.plot(g.ent, g.expr_loss, c=COL[s_], lw=1.3, alpha=0.7, zorder=2)
for s_ in SEC:
    ax.plot([], [], c=COL[s_], lw=2, label=s_)
ax.set_xlabel("coupling entropy"); ax.set_ylabel("expression loss")
ax.set_title("2. One line per slice, $\\alpha$ swept along it.\nPooled the trend hides ($\\rho$=+0.33); per slice it is +0.85")
ax.legend(frameon=False, fontsize=8, loc="upper left")

# --- 3. one slice in detail ---------------------------------------------------
ax = fig.add_subplot(gs[0, 2])
g = t[(t.section == "M3-4") & (np.isclose(t.eps, 0.02))].sort_values("alpha")
nrm = lambda v: (v - v.min()) / (v.max() - v.min())
for y, c, lab, mk in [(g.expr_loss, C1, "expression loss", "o"),
                      (g.ent, C2, "coupling entropy", "s"),
                      (g.acc, C3, "true accuracy", "D")]:
    ax.plot(g.alpha, nrm(y.values), c=c, lw=2, marker=mk, ms=6.5, label=lab,
            markeredgecolor="white", markeredgewidth=0.8)
ax.axvline(0.6, color=INK2, ls=":", lw=1.2)
ax.annotate("accuracy peaks\nhere", (0.6, 0.30), fontsize=8, color=INK2,
            xytext=(-58, 0), textcoords="offset points")
ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("min–max scaled within the slice")
ax.set_title("3. M3-4 at $\\epsilon$=0.02: loss and entropy\nrise together; accuracy peaks elsewhere")
ax.legend(frameon=False, fontsize=8, loc="lower right")

# --- 4. nothing beats the constant -------------------------------------------
ax = fig.add_subplot(gs[1, 0])
crit = [("expression\nloss", 0.1120), ("coupling\nentropy", 0.1246),
        ("composition\nloss", 0.1459), ("entropy\nmatching", 0.1530),
        ("constant\n" + r"$\alpha$=0.2", 0.1547)]
cols = [GRID] * 4 + [C1]
bars = ax.bar(range(5), [v for _, v in crit], 0.66, color=cols,
              edgecolor="white", linewidth=1.4)
for i, (_, v) in enumerate(crit):
    ax.text(i, v + 0.0018, f"{v:.4f}", ha="center", fontsize=8, color=INK)
ax.axhline(0.1722, color=INK2, ls="--", lw=1.4)
ax.text(4.42, 0.1734, "oracle 0.1722", ha="right", fontsize=8, color=INK2)
ax.axhline(0.1120, color=INK2, ls=":", lw=1.4)
ax.text(4.42, 0.1064, r"no tree ($\alpha$=0) 0.1120", ha="right", fontsize=8, color=INK2)
ax.set_xticks(range(5)); ax.set_xticklabels([k for k, _ in crit], fontsize=8)
ax.set_ylim(0.10, 0.182); ax.set_ylabel("accuracy at the selected $\\alpha$")
ax.set_title("4. No label-free criterion beats a constant\n24 PEtracer configurations")

# --- 5. how much alpha is worth, by source -----------------------------------
ax = fig.add_subplot(gs[1, 1])
v = pd.read_csv("../simulation/varalpha_oracle.csv"); v = v[v.in_band]
AS = [0.0, 0.1, 0.2, 0.35, 0.5, 0.7]
for _, r in v.iterrows():
    ax.plot(AS, [r[f"acc_{a}"] - r["acc_0.0"] for a in AS], c=GRID, lw=1.1, zorder=1)
ax.plot(AS, [v[f"acc_{a}"].mean() - v["acc_0.0"].mean() for a in AS],
        c=INK2, lw=2.8, zorder=4, label=f"simulator mean ({len(v)})")
for s_ in SEC:
    sub = t[t.section == s_]
    for e_, g2 in sub.groupby("eps"):
        g2 = g2.sort_values("alpha")
        ax.plot(g2.alpha, g2.acc - g2.acc.iloc[0], c=COL[s_], lw=1.0, alpha=0.35, zorder=2)
    mn = sub.groupby("alpha").acc.mean(); mn = mn - mn.iloc[0]
    ax.plot(mn.index, mn.values, c=COL[s_], lw=2.8, zorder=4, label=f"{s_} mean")
ax.axhline(0, color=INK2, lw=1)
ax.set_xlabel(r"$\alpha$"); ax.set_ylabel(r"accuracy gain over $\alpha$=0")
ax.set_title("5. Only M3-4 has large $\\alpha$-sensitivity.\nM3-1 and M3-2 are no better than the simulator")
ax.legend(frameon=False, fontsize=7.5, loc="lower left")

# --- 6. composition informativeness ------------------------------------------
ax = fig.add_subplot(gs[1, 2])
lab = ["simulator", "M3-1", "M3-2", "M3-4"]
val = [0.086, 0.566, 0.553, 0.586]
ax.bar(range(4), val, 0.62, color=[GRID, C1, C2, C3], edgecolor="white", linewidth=1.4)
for i, x in enumerate(val):
    ax.text(i, x + 0.014, f"{x:.3f}", ha="center", fontsize=8.5, color=INK)
ax.set_xticks(range(4)); ax.set_xticklabels(lab)
ax.set_ylim(0, 0.70); ax.set_ylabel("composition → position informativeness")
ax.set_title("6. My 'composition is uninformative on real\ndata' claim was backwards")

fig.suptitle("Why the losses cannot select $\\alpha$ — the evidence, with every data point shown",
             fontsize=13.5, y=0.958, color=INK)
fig.savefig("alpha_evidence.png", dpi=155, facecolor="white")
print("wrote alpha_evidence.png")
