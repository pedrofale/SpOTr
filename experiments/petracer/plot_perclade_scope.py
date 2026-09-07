import numpy as np, pandas as pd, itertools
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

rep = pd.read_csv("petracer_perclade_replication.csv")
tm  = pd.read_csv("petracer_transfer_matrix.csv")
sh  = pd.read_csv("petracer_split_half_oracle.csv")
SECS = ["M3-1", "M3-2", "M3-4"]

fig = plt.figure(figsize=(13.5, 8.2))
gs = fig.add_gridspec(2, 3, hspace=0.46, wspace=0.46, top=0.88, bottom=0.09, left=0.065, right=0.975)

# --- A/B: per-clade coherence replication, within block vs across blocks
for i, (a, b) in enumerate([("M3-1", "M3-2"), ("M3-1", "M3-4")]):
    ax = fig.add_subplot(gs[0, i])
    x, y = rep[f"coh_{a}"], rep[f"coh_{b}"]
    r, p = pearsonr(x, y)
    ax.scatter(x, y, s=46, c="#2b6cb0" if i == 0 else "#c05621",
               edgecolor="white", linewidth=0.8, zorder=3)
    for _, q in rep.iterrows():
        ax.annotate(int(q.clade), (q[f"coh_{a}"], q[f"coh_{b}"]), fontsize=6.5,
                    xytext=(3.5, 3.5), textcoords="offset points", color="#4a5568")
    lim = [-0.03, 0.75]
    ax.plot(lim, lim, ls=":", c="#a0aec0", lw=1, zorder=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"per-clade coherence, {a}"); ax.set_ylabel(f"per-clade coherence, {b}")
    tvd = 0.113 if i == 0 else 0.449
    ax.set_title(f"{'same tissue block' if i==0 else 'different region'}\n"
                 f"r = {r:+.3f} (p = {p:.3f}),  composition TVD = {tvd:.3f}",
                 fontsize=10.5, color="#1a202c")
    ax.grid(alpha=0.25, zorder=0)

# --- C: split-half held-out gain
ax = fig.add_subplot(gs[0, 2])
w, xs = 0.26, np.arange(3)
cols = {"pc - best global": "#2b6cb0", "pc - uniform@mean": "#38a169", "pc - shuffled": "#a0aec0"}
for j, (lab, key) in enumerate([("pc - best global", "global_held"),
                                ("pc - uniform@mean", "uniform_held"),
                                ("pc - shuffled", "shuffled_held")]):
    m = [ (sh[sh.section==s].perclade_held - sh[sh.section==s][key]).mean() for s in SECS ]
    e = [ (sh[sh.section==s].perclade_held - sh[sh.section==s][key]).std()  for s in SECS ]
    ax.bar(xs + (j-1)*w, m, w, yerr=e, capsize=3, label=lab, color=cols[lab],
           edgecolor="white", linewidth=0.7)
ax.axhline(0, c="#2d3748", lw=1)
ax.set_xticks(xs); ax.set_xticklabels(SECS)
ax.set_ylabel("held-out soft-accuracy gain")
ax.set_ylim(0, 0.175)
ax.set_title("per-clade α, fitted on half the spots,\nscored on the other half (3 splits)",
             fontsize=10.5, color="#1a202c")
ax.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=1)
ax.grid(alpha=0.25, axis="y")

# --- D: transfer matrix heatmap (vs uniform@mean -- controls for the global level)
ax = fig.add_subplot(gs[1, 0])
M = np.full((3, 3), np.nan)
for _, q in tm.iterrows():
    M[SECS.index(q.source), SECS.index(q.target)] = q.transferred - q.uniform_at_mean
v = np.nanmax(np.abs(M))
im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{M[i,j]:+.3f}", ha="center", va="center", fontsize=10,
                color="#1a202c", fontweight="bold" if i == j else "normal")
        if i == j:
            ax.add_patch(plt.Rectangle((j-.5, i-.5), 1, 1, fill=False, ec="#1a202c", lw=2))
ax.set_xticks(range(3)); ax.set_xticklabels(SECS); ax.set_yticks(range(3)); ax.set_yticklabels(SECS)
ax.set_xlabel("applied to (target)"); ax.set_ylabel("α fitted on (source)")
ax.set_title("transfer of oracle per-clade α\n(gain over flat α at the same mean)",
             fontsize=10.5, color="#1a202c")
cb = plt.colorbar(im, ax=ax, fraction=0.040, shrink=0.80, pad=0.04)
cb.ax.tick_params(labelsize=7.5)

# --- E: transfer summary, within block vs across
ax = fig.add_subplot(gs[1, 1])
off = tm[tm.is_diag == 0].copy()
off["within"] = off.apply(lambda x: {x.source, x.target} <= {"M3-1", "M3-2"}, axis=1)
lbl = [f"{q.source}→{q.target}" for _, q in off.iterrows()]
val = (off.transferred - off.uniform_at_mean).values
c = ["#2b6cb0" if w else "#c05621" for w in off["within"]]
ax.barh(range(len(val)), val, color=c, edgecolor="white", linewidth=0.7)
ax.set_yticks(range(len(val))); ax.set_yticklabels(lbl, fontsize=8.5)
ax.axvline(0, c="#2d3748", lw=1); ax.invert_yaxis()
ax.set_xlabel("gain over flat α at the same mean")
ax.set_title("same block (blue) transfers;\nacross regions (orange) does not", fontsize=10.5)
ax.grid(alpha=0.25, axis="x")

# --- F: the global premise, per section (what claim (a) rests on)
ax = fig.add_subplot(gs[1, 2])
pre = pd.read_csv("petracer_premise_persection.csv")
xs = np.arange(3)
ax.bar(xs - 0.2, pre.coherence, 0.4, label="spatial coherence", color="#2b6cb0",
       edgecolor="white", linewidth=0.7)
ax.bar(xs + 0.2, pre.mantel_r, 0.4, label="tree–space Mantel r", color="#38a169",
       edgecolor="white", linewidth=0.7)
for i, (c_, m_) in enumerate(zip(pre.coherence, pre.mantel_r)):
    ax.text(i-0.2, c_+0.012, f"{c_:.2f}", ha="center", fontsize=8)
    ax.text(i+0.2, m_+0.012, f"{m_:.2f}", ha="center", fontsize=8)
ax.set_xticks(xs); ax.set_xticklabels(SECS); ax.set_ylim(0, 0.62)
ax.set_ylabel("value"); ax.legend(fontsize=8, frameon=False)
ax.set_title("the GLOBAL tree–space map is clear\nin every section (all p < 0.002)",
             fontsize=10.5, color="#1a202c")
ax.grid(alpha=0.25, axis="y")

fig.suptitle("PEtracer: per-clade α buys real held-out accuracy, but it is a property of clade × tissue region",
             fontsize=13.5, y=0.955, color="#1a202c")
fig.savefig("perclade_scope.png", dpi=160, facecolor="white")
print("wrote perclade_scope.png")
