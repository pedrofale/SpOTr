import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d8d4"
L = pd.read_csv("two_branch_landscape.csv")
H = pd.read_csv("two_branch_heldout.csv")
SECS = ["M3-1", "M3-2", "M3-4"]
G = sorted(L.a1.unique())

fig = plt.figure(figsize=(14.6, 4.9))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.95], wspace=0.34,
                      top=0.79, bottom=0.16, left=0.052, right=0.985)
for k, s_ in enumerate(SECS):
    ax = fig.add_subplot(gs[0, k])
    M = L[L.section == s_].pivot(index="a1", columns="a2", values="acc").values
    im = ax.imshow(M, cmap="Blues", origin="lower", aspect="auto")
    for i in range(len(G)):
        ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False, ec=INK2, lw=1.1, ls=":"))
    bi = np.unravel_index(np.nanargmax(M), M.shape)
    ax.add_patch(plt.Rectangle((bi[1] - .5, bi[0] - .5), 1, 1, fill=False, ec="#eb6834", lw=2.4))
    ax.set_xticks(range(len(G))); ax.set_xticklabels([f"{v:g}" for v in G], fontsize=7.5)
    ax.set_yticks(range(len(G))); ax.set_yticklabels([f"{v:g}" for v in G], fontsize=7.5)
    ax.set_xlabel(r"$\alpha_2$  (branch 2, 14 clades)", fontsize=9)
    if k == 0: ax.set_ylabel(r"$\alpha_1$  (branch 1: clades 1,2,3,4)", fontsize=9)
    ax.set_title(f"{s_}\nbest ({G[bi[0]]:g}, {G[bi[1]]:g}) = {np.nanmax(M):.3f}",
                 fontsize=10.5, color=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=7.5)

ax = fig.add_subplot(gs[0, 3])
w = 0.36; xs = np.arange(3)
g2 = [H[H.section == s_].held_2branch.mean() for s_ in SECS]
gg = [H[H.section == s_].held_global.mean() for s_ in SECS]
e2 = [H[H.section == s_].held_2branch.std() for s_ in SECS]
eg = [H[H.section == s_].held_global.std() for s_ in SECS]
ax.bar(xs - w/2, g2, w, yerr=e2, capsize=3, color="#2a78d6", edgecolor="white",
       linewidth=1.2, label="two-branch $\\alpha$")
ax.bar(xs + w/2, gg, w, yerr=eg, capsize=3, color=GRID, edgecolor="white",
       linewidth=1.2, label="best single $\\alpha$")
for i, (v2, vg) in enumerate(zip(g2, gg)):
    ax.text(i, max(v2, vg) + 0.014, f"{v2-vg:+.3f}", ha="center", fontsize=9,
            color="#2a78d6" if v2 > vg else "#c05621", fontweight="bold")
ax.set_xticks(xs); ax.set_xticklabels(SECS)
ax.set_ylabel("held-out soft accuracy"); ax.set_ylim(0, 0.28)
ax.set_title("Held out: chosen on half the spots,\nscored on the other half (3 splits)",
             fontsize=10.5, color=INK)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Two-branch $\\alpha$: the full landscape (dotted diagonal = a single global $\\alpha$; "
             "orange = best cell)", fontsize=12.5, y=0.925, color=INK)
fig.savefig("two_branch_alpha.png", dpi=155, facecolor="white")
print("wrote two_branch_alpha.png")
