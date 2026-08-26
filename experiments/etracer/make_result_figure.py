"""The eTRACER result: the lineage tree helps, but only under matched composition."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
import spotr.spotr as S

fine = pd.read_csv(HERE / "etracer_fine_sweep.csv")
abl = pd.read_csv(HERE / "etracer_ablation.csv")
d = np.load(HERE / "etracer_inputs.npz", allow_pickle=True)

fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.9))

# --- A: fine alpha sweep, real vs shuffled ---
piv = fine.pivot_table(index="alpha", columns="variant", values="ari")
sh = [c for c in piv.columns if c.startswith("shuf")]
axs[0].plot(piv.index, piv["real"], "o-", color="#3a7d44", lw=2.2, ms=7, label="real lineage tree")
axs[0].plot(piv.index, piv[sh].mean(1), "s--", color="#999", lw=1.8, ms=6, label="shuffled tree (3 seeds)")
axs[0].fill_between(piv.index, piv[sh].mean(1) - piv[sh].std(1), piv[sh].mean(1) + piv[sh].std(1),
                    color="#999", alpha=.3)
axs[0].axhline(piv.loc[0.0, "real"], ls=":", c="k", lw=1.2, label="no tree (α=0)")
b = piv["real"].idxmax()
axs[0].plot([b], [piv.loc[b, "real"]], "*", ms=20, color="gold", markeredgecolor="k", zorder=5)
axs[0].annotate(f"α={b}\nARI {piv.loc[b,'real']:.3f}", (b, piv.loc[b, "real"]),
                textcoords="offset points", xytext=(12, -4), fontsize=9)
axs[0].set_xlabel(r"structure weight $\alpha$"); axs[0].set_ylabel("adjusted Rand index")
axs[0].set_title("The real tree beats shuffled at 8/9 α\n(18.5 sd at the optimum)", fontsize=11)
axs[0].legend(fontsize=8)

# --- B: composition matters ---
mism = dict(real=0.1511, shuf=0.2103, none=0.1847)     # from the first, mismatched run
match = dict(real=float(piv["real"].max()), shuf=float(piv[sh].mean(1).max()),
             none=float(piv.loc[0.0, "real"]))
x = np.arange(2); w = 0.26
for i, (k, lab, col) in enumerate([("real", "real tree", "#3a7d44"),
                                   ("shuf", "shuffled tree", "#999"),
                                   ("none", "no tree", "#c46")]):
    axs[1].bar(x + (i - 1) * w, [mism[k], match[k]], w, label=lab, color=col, edgecolor="k", lw=.5)
axs[1].set_xticks(x)
axs[1].set_xticklabels(["mismatched\n(TVD 0.63)", "matched\n(TVD 0.011)"])
axs[1].set_xlabel("cell vs bin clone composition")
axs[1].set_ylabel("adjusted Rand index")
axs[1].set_title("The tree only helps when the reference\ncomposition matches the tissue", fontsize=11)
axs[1].legend(fontsize=8)

# --- C: spatial map at the optimum ---
A, clones, truth, xy = d["A"], d["clones"], d["bin_clone"], d["xy"]
asc, ast = ad.AnnData(d["X_sc"]), ad.AnnData(d["X_st"])
Cf, C_T, C_S, a, b_ = S.prepare_ot_inputs(asc, ast, d["C_tree"], d["C_space"])
g = np.asarray(S.sinkhorn_fgw(Cf, C_T, C_S, a, b_, 0.02, T_sinkhorn=100, J_alt=5, alpha=float(b))[0])
prop = g.T @ A; prop = prop / np.maximum(prop.sum(1, keepdims=True), 1e-12)
pred = np.asarray(clones)[prop.argmax(1)]
top = pd.Series(truth).value_counts().head(4).index.tolist()
cmap = {c: plt.get_cmap("tab10")(i) for i, c in enumerate(top)}
for lab, mk, ttl in [(truth, "o", "true"), (pred, "o", "SpOTr")]:
    pass
axs[2].scatter(xy[:, 0], -xy[:, 1], c=[cmap.get(c, (.85, .85, .85, 1)) for c in truth], s=9)
axs[2].scatter(xy[:, 0] + (xy[:, 0].max() - xy[:, 0].min()) * 1.12, -xy[:, 1],
               c=[cmap.get(c, (.85, .85, .85, 1)) for c in pred], s=9)
axs[2].set_xticks([]); axs[2].set_yticks([]); axs[2].set_aspect("equal")
axs[2].set_title(f"Numbat-free ground truth (left) vs SpOTr (right)\n"
                 f"α={b}, top-4 clones coloured", fontsize=11)
handles = [plt.Line2D([], [], marker="o", ls="", color=cmap[c], label=c) for c in top]
axs[2].legend(handles=handles, fontsize=8, loc="lower right", ncol=2)

fig.suptitle("SpOTr on eTRACER late-escape: the lineage tree gives a real but modest gain",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.90])
out = HERE / "etracer_spotr_result.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
