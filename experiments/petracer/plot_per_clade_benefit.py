import numpy as np, pickle, ot, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)
d = np.load(HERE / "petracerFAIR_M3-4.npz", allow_pickle=True)
xy_sc, xy_st = np.asarray(d["xy_sc"], float), np.asarray(d["xy"], float)
X_sc, X_st = np.asarray(d["X_sc"], float), np.asarray(d["X_st"], float)
cc = np.asarray(d["cell_clade"]).astype(str)
T = pickle.loads((HERE / "where_cells_go.pkl").read_bytes())
e0 = np.linalg.norm(xy_st[np.asarray(T[0.0]).argmax(1)] - xy_sc, axis=1)
e6 = np.linalg.norm(xy_st[np.asarray(T[0.75]).argmax(1)] - xy_sc, axis=1)
delta = e0 - e6
glob = np.median(np.linalg.norm(xy_sc - np.median(xy_sc, 0), axis=1))
big = [c for c in np.unique(cc) if (cc == c).sum() >= 30]
R = pd.DataFrame([dict(
        clade=c, n=int((cc == c).sum()),
        cohesion=np.median(np.linalg.norm(xy_sc[cc == c] - np.median(xy_sc[cc == c], 0),
                                          axis=1)) / glob,
        err_a0=float(np.median(e0[cc == c])),
        mean_delta=float(delta[cc == c].mean()),
        se=float(delta[cc == c].std() / np.sqrt((cc == c).sum()))) for c in big])

lab = np.array([big.index(c) if c in big else -1 for c in cc]); m = lab >= 0
def eta2(v, l):
    gm = v.mean()
    return sum((v[l == k].mean()-gm)**2 * (l == k).sum() for k in np.unique(l)) / ((v-gm)**2).sum()
obs = eta2(delta[m], lab[m])
null = [eta2(delta[m], rng.permutation(lab[m])) for _ in range(2000)]
p_eta = (1 + sum(x >= obs for x in null)) / 2001

fig, ax = plt.subplots(1, 3, figsize=(14.2, 4.5))
S = R.sort_values("mean_delta")
col = ["#a83c10" if v < 0 else "#1a6b3c" for v in S.mean_delta]
ax[0].barh(range(len(S)), S.mean_delta, xerr=S.se, color=col, edgecolor="white", lw=1.2,
           error_kw=dict(lw=1, ecolor="0.45"))
ax[0].axvline(0, color="0.3", lw=1.2)
ax[0].set_yticks(range(len(S))); ax[0].set_yticklabels(S.clade, fontsize=8.5)
ax[0].set_xlabel("mean µm of error the tree removes")
ax[0].set_ylabel("clade")
ax[0].set_title(f"Clades really do differ\n$\\eta^2$ = {obs:.3f}, p = {p_eta:.4f} "
                f"vs permutation", fontsize=10.5)
ax[0].spines[["top", "right"]].set_visible(False); ax[0].grid(axis="x", alpha=.25)
ax[0].set_axisbelow(True)

for a, col_, ttl in [(ax[1], "cohesion", "…but NOT because some clades\nare more spatially cohesive"),
                     (ax[2], "err_a0", "Expression headroom no longer\nexplains it on a fair cost")]:
    r = spearmanr(R[col_], R.mean_delta)
    a.scatter(R[col_], R.mean_delta, s=np.sqrt(R.n) * 9, c="#2a78d6", alpha=.75,
              edgecolors="white", lw=1.2)
    for _, q in R.iterrows():
        a.annotate(q.clade, (q[col_], q.mean_delta), fontsize=7.5, color="#555",
                   textcoords="offset points", xytext=(7, -3))
    a.axhline(0, color="0.5", lw=1, ls=":")
    a.set_xlabel({"cohesion": "clade spread / section spread  (low = compact)",
                  "err_a0": "median placement error with expression alone (µm)"}[col_])
    a.set_ylabel("mean µm the tree removes")
    a.set_title(f"{ttl}\nρ = {r.statistic:+.3f}, p = {r.pvalue:.3f}", fontsize=10.5)
    a.spines[["top", "right"]].set_visible(False); a.grid(alpha=.25); a.set_axisbelow(True)

fig.suptitle("Does per-clade α have anything to find?  Clades differ significantly, but no "
             "candidate explains WHY.\nMarker size = clade size.", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.86])
fig.savefig(HERE / "per_clade_benefit.png", dpi=150, facecolor="white")
print(f"eta2={obs:.4f} p={p_eta:.4f}; wrote per_clade_benefit.png")
