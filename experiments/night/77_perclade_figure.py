"""Figure: the learned per-clade mixing lengths and what they relate to (PEtracer, 2026-10-13).

Reads spotr_out/perclade_features.{csv,json} from 76_perclade_features.py.
  a  every learned L_tau (shared start, the per-clade arm of the slides), mean over a section's 8 splits with the
     spread across them, against the shared L the section starts from;
  b  learned L against the clone's EMT score (the primary, pre-specified test);
  c  the clone's true spread against the same EMT score (does the tissue show what L shows?);
  d  every pre-specified clone feature: within-section Spearman for both learned L and for the true spread.
Same width as figures/overview.pdf.
Usage: python 77_perclade_figure.py [out.pdf] [--png]   (default: SpOTr-overleaf/figures/perclade_lengths.pdf)
"""
import sys, json, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
OUT = Path(ARGS[0]) if ARGS else HERE.parents[2] / "SpOTr-overleaf" / "figures" / "perclade_lengths.pdf"
spec = importlib.util.spec_from_file_location("m70", HERE / "70_spotr.py")
m70 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m70)
D = pd.read_csv(HERE / "spotr_out" / "perclade_features.csv")
J = json.load(open(HERE / "spotr_out" / "perclade_features.json"))
TESTS = pd.DataFrame(J["tests"])

plt.rcParams.update({"font.family": "Liberation Sans", "mathtext.fontset": "cm", "pdf.fonttype": 42,
                     "axes.unicode_minus": False, "font.size": 9.5, "axes.edgecolor": "#8a8984", "axes.linewidth": 0.8,
                     "xtick.color": "#4a4a4a", "ytick.color": "#4a4a4a", "axes.labelcolor": "#222222"})
INK, INK2, GRID = "#111111", "#4a4a4a", "#e6e5e0"
LIN_COL = ["#2a78d6", "#eb6834", "#1baf7a", "#a9a8a2"]        # as petracer_in_action.pdf (grey a step darker on dots)
LINEAGE = {**{c: 0 for c in ["2", "3", "4"]}, **{c: 1 for c in ["13", "14", "15"]}, **{c: 2 for c in ["16", "17", "18"]}}
D["lin"] = D.clade.astype(str).map(LINEAGE).fillna(3).astype(int)
SECS = ["M3-4", "M3-1", "M3-2"]; MARK = {"M3-4": "o", "M3-1": "s", "M3-2": "^"}
FIG_W, FIG_H = 911 / 72, 5.7
fig = plt.figure(figsize=(FIG_W, FIG_H))
ax_at = lambda x, y, w, h: fig.add_axes([x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H])


def letter(ax, s, title, sub=None, x=-0.02):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=18 if sub else 6)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=8.8, color=INK2, va="bottom", style="italic")
    ax.text(x, 1.0, s, transform=ax.transAxes, fontsize=14, fontweight="bold", ha="right", va="bottom")


def clean(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- a: every learned L_tau
ax = ax_at(0.95, 1.05, 2.55, 3.97)
y, ticks, labels, heads = 0.0, [], [], []
for sec in SECS[::-1]:                                          # bottom-up, so M3-4 reads first
    d = D[D.section == sec].sort_values("logL_shared")
    heads.append((y + len(d) - 0.35, sec, m70.LAWS[sec][0]))
    for _, r in d.iterrows():
        mu, sd = r.logL_shared, r.logL_shared_sd
        ax.plot([np.exp(mu - sd), np.exp(mu + sd)], [y, y], color=LIN_COL[r.lin], lw=1.4, solid_capstyle="round")
        ax.scatter([np.exp(mu)], [y], s=26, color=LIN_COL[r.lin], zorder=3, lw=0, marker=MARK[sec])
        ticks.append(y); labels.append(f"clade {r.clade}"); y += 1
    L0 = m70.LAWS[sec][0]
    ax.plot([L0, L0], [y - len(d) - 0.5, y - 0.5], color=INK2, lw=0.9, ls=(0, (3, 2.5)), zorder=0)
    y += 2.3
for yy, sec, L0 in heads:
    ax.text(0, yy + 0.75, f"{sec}  (shared L {L0:.0f} µm, dashed)", transform=ax.get_yaxis_transform(),
            fontsize=8.8, color=INK, fontweight="bold", va="bottom", ha="left")
ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=7.6); ax.set_ylim(-0.8, y - 0.9)
ax.set_xlabel("learned mixing length $L_\\tau$ (µm)\nmean and spread over 8 splits")
ax.grid(axis="x", color=GRID, lw=0.7); ax.set_axisbelow(True); clean(ax)
letter(ax, "a", "Learned mixing length per clade", sub="18 parameters per section; clades with 30+ cells", x=-0.30)

# ---------------------------------------------------------------- b, c: against the clone's EMT score
def rel_axis(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.25, 1.5]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"×{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True); clean(ax)


def stat(est, feat):
    r = TESTS[(TESTS.estimate == est) & (TESTS.feature == feat)].iloc[0]
    return f"Spearman {r.rho:+.2f}, p = {r.p:.2f}" if r.p >= 0.01 else f"Spearman {r.rho:+.2f}, p = {r.p:.3f}"


for i, (col, est, title, sub, letter_) in enumerate([
        ("logL_shared_rel", "learned L, shared start", "Learned L against EMT", "the primary, pre-specified test", "b"),
        ("log_true_spread_rel", "true spread", "True spread against EMT", "the same clones, measured in the tissue", "c")]):
    ax = ax_at(4.45, 3.33 - i * 2.28, 2.55, 1.69)
    for sec in SECS:
        d = D[D.section == sec]
        ax.scatter(d.emt, np.exp(d[col]), s=30, c=[LIN_COL[k] for k in d.lin], marker=MARK[sec], lw=0, zorder=3)
    rel_axis(ax)
    v = np.exp(D[col].values); ax.set_ylim(v.min() / 1.08, v.max() * 1.3)      # headroom for the statistic
    ax.set_xlabel("clone EMT score (mesenchymal − epithelial, z)" if i == 1 else "")
    ax.set_ylabel("relative to the section", fontsize=8.8)
    ax.text(0.02, 0.97, stat(est, "EMT score (primary)"), transform=ax.transAxes, fontsize=8.6, va="top", color=INK)
    letter(ax, letter_, title, sub=sub, x=-0.2)

# ---------------------------------------------------------------- d: every clone feature
ax = ax_at(8.75, 1.05, 3.6, 3.97)
rows = [("EMT score (primary)", "EMT score\n(primary)"), ("fitness (LBI)", "fitness (LBI)"), ("clade age", "clade age"),
        ("expression plasticity", "expression plasticity"), ("distance to tumour boundary", "distance to\ntumour boundary"),
        ("clade size", "clade size\n(control)"), ("true spatial spread", "true spatial spread\n(validation)")]
ests = [("learned L, shared start", "#184f95", "o", "learned L, shared start"),
        ("learned L, anchor start", "#5598e7", "D", "learned L, anchor start"),
        ("true spread", INK, "o", "true spread (tissue)")]
for j, (feat, _) in enumerate(rows):
    yj = len(rows) - 1 - j
    for k, (est, col, mk, _) in enumerate(ests):
        t = TESTS[(TESTS.estimate == est) & (TESTS.feature == feat)]
        if t.empty:
            continue
        r = t.iloc[0]; yy = yj + 0.24 - 0.24 * k
        sig = r.p < 0.05
        ax.plot([r.lo, r.hi], [yy, yy], color=col, lw=1.0, alpha=0.55, solid_capstyle="butt")
        ax.scatter([r.rho], [yy], s=30, marker=mk, facecolor=col if sig else "white", edgecolor=col, lw=1.2, zorder=3)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_xlim(-1, 1); ax.set_ylim(-1.15, len(rows) - 0.4)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[1] for r in rows][::-1], fontsize=8.6)
ax.set_xlabel("Spearman correlation, within section")
ax.text(-0.02, -0.17, "Only EMT was pre-specified. Corrected for the six secondary features,\n"
        "what holds is the true spread (validation) and clade age; fitness\n"
        "against the true spread misses narrowly (p = 0.009).",
        transform=ax.transAxes, fontsize=7.8, color=INK2, style="italic", va="top")
ax.grid(axis="x", color=GRID, lw=0.7); ax.set_axisbelow(True); clean(ax)
handles = [Line2D([], [], marker=mk, color="none", markerfacecolor="white", markeredgecolor=col,
                  markersize=6, label=lab) for e, col, mk, lab in ests]
handles.append(Line2D([], [], marker="o", color="none", markerfacecolor="#777777", markeredgecolor="#777777", markersize=6,
                      label="filled: p < 0.05; line: 95% interval"))
ax.legend(handles=handles, loc="lower left", fontsize=7.6, frameon=False, handletextpad=0.3, borderaxespad=0.2,
          labelspacing=0.35)
letter(ax, "d", "Every clone feature tested", sub=f"{J['n_clades']} clades in 3 sections; permutation p within section", x=-0.36)

# lineage and section key, bottom left
keys = [Line2D([], [], marker="s", color="none", markerfacecolor=c, markeredgecolor=c, markersize=7, label=l)
        for c, l in zip(LIN_COL, ["clades 2, 3, 4", "clades 13, 14, 15", "clades 16, 17, 18", "other clades"])]
keys += [Line2D([], [], marker=MARK[s], color="none", markerfacecolor="#777777", markeredgecolor="#777777", markersize=6,
                label=s) for s in SECS]
fig.legend(handles=keys, loc="lower left", bbox_to_anchor=(0.6 / FIG_W, 0.0), ncol=7, frameon=False, fontsize=8.6,
           handletextpad=0.3, columnspacing=1.1)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
if "--png" in sys.argv:
    fig.savefig(OUT.with_suffix(".png"), dpi=200)
print("wrote", OUT)
