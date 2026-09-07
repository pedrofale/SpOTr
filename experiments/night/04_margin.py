"""WHICH STRUCTURAL LOSS CAN EVEN TELL A GOOD MAP FROM A BAD ONE?

A structural loss is a function L(tree distance, spatial distance).  For a hard assignment of
cells to spots it is evaluated as

    S(assignment) = sum over cell pairs of  L( t_ik , distance between their assigned spots )
                  = < L , H(assignment) >

where H is the 2-D histogram of (tree distance, mapped spatial distance).  So a loss is nothing
but a set of weights on the cells of that histogram, and its ability to guide a mapping is
exactly its ability to score the TRUE assignment below corrupted ones.

That can be measured without solving any OT problem: corrupt the true assignment by shuffling a
fraction eps of the cells and watch S climb.  Slope = usable signal.  Flat = the loss cannot see
the corruption, whatever the solver does.

Losses compared, all built from the same reference statistics:
    square loss            (t - s)^2 on standardised distances          <- what FGW uses
    HGR rank-R             sum_r rho_r phi_r(t) psi_r(s)                <- maximal correlation
    -PMI                   -log[ p(t,s) / p(t)p(s) ]                    <- likelihood ratio
    contrast               p(t)p(s) - p(t,s)                            <- linear discriminant
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
from spotr.genfgw import fit_kernel, tree_bins, space_bins

SECS = ["M3-4", "M3-1", "M3-2"]
EPS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
NDRAW = 12
rng = np.random.default_rng(0)


def losses(t, s_ref, te, se):
    """Build every candidate loss as a matrix over the (tree bin) x (space bin) grid."""
    P, _, _ = np.histogram2d(t, s_ref, bins=[te, se]); P /= P.sum()
    a = P.sum(1)[:, None]; b = P.sum(0)[None, :]
    tc = 0.5 * (te[:-1] + te[1:]); sc = 0.5 * (se[:-1] + se[1:])
    ts = (tc - (tc * P.sum(1)).sum()) / np.sqrt(max(((tc ** 2) * P.sum(1)).sum()
                                                    - ((tc * P.sum(1)).sum()) ** 2, 1e-12))
    ss = (sc - (sc * P.sum(0)).sum()) / np.sqrt(max(((sc ** 2) * P.sum(0)).sum()
                                                    - ((sc * P.sum(0)).sum()) ** 2, 1e-12))
    out = {"square loss (FGW)": (ts[:, None] - ss[None, :]) ** 2}
    K = fit_kernel(t, s_ref, ncomp=3, te=te, se=se)
    for R in (1, 2, 3):
        out[f"HGR rank {R}"] = -sum(K["rho"][r] * np.outer(K["phi"][r], K["psi"][r])
                                    for r in range(R))
    ratio = P / np.maximum(a * b, 1e-300)
    out["-PMI"] = -np.log(np.where(P > 0, ratio, 1e-6))
    out["contrast (pq - P)"] = (a * b - P)
    for k in out:                                        # standardise so slopes are comparable
        L = out[k]
        w = (a * b)
        mu = float((L * w).sum()); sd = float(np.sqrt(max((L ** 2 * w).sum() - mu ** 2, 1e-30)))
        out[k] = (L - mu) / sd
    return out, te, se


rows = []
for SEC in SECS:
    d = np.load(HERE.parent / "petracer" / f"petracerFAIR_{SEC}.npz", allow_pickle=True)
    CT = np.asarray(d["C_tree"], float)
    xy_sc = np.asarray(d["xy_sc"], float); xy_st = np.asarray(d["xy"], float)
    n, m = len(xy_sc), len(xy_st)
    CSpot = squareform(pdist(xy_st))
    iu = np.triu_indices(n, 1)
    t = CT[iu]
    te = tree_bins(t); se = space_bins(pdist(xy_sc))
    L, te, se = losses(t, pdist(xy_sc), te, se)
    true_spot = np.argmin(((xy_sc[:, None, :] - xy_st[None, :, :]) ** 2).sum(2), axis=1)

    def score(assign):
        sm = CSpot[assign[iu[0]], assign[iu[1]]]
        H, _, _ = np.histogram2d(t, sm, bins=[te, se]); H /= H.sum()
        return {k: float((v * H).sum()) for k, v in L.items()}

    base = score(true_spot)
    rnd = np.mean([[score(rng.permutation(true_spot))[k] for k in L] for _ in range(6)], axis=0)
    rnd = dict(zip(L, rnd))
    for e in EPS:
        acc = {k: [] for k in L}
        for _ in range(NDRAW if e > 0 else 1):
            a2 = true_spot.copy()
            idx = rng.random(n) < e
            a2[idx] = rng.permutation(a2[idx])
            sc_ = score(a2)
            for k in L:
                acc[k].append(sc_[k])
        for k in L:
            # 0 = the truth, 1 = a random assignment
            rows.append(dict(section=SEC, loss=k, eps=e,
                             norm=(np.mean(acc[k]) - base[k]) / (rnd[k] - base[k])))
    print(f"{SEC}: done", flush=True)

R = pd.DataFrame(rows); R.to_csv(HERE / "margin.csv", index=False)
piv = R.pivot_table(index="eps", columns=["section", "loss"], values="norm")
print("\nHow fast the loss rises as the assignment is corrupted (1.0 = a random map)")
for SEC in SECS:
    print(f"\n{SEC}")
    print(piv[SEC].round(3).to_string())

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
CL = {"square loss (FGW)": "#9aa7b1", "HGR rank 1": "#1b6ca8", "HGR rank 2": "#5aa0d8",
      "HGR rank 3": "#a8cbe8", "-PMI": "#b8320f", "contrast (pq - P)": "#2f7d4f"}
for ax, SEC in zip(axes, SECS):
    g = R[R.section == SEC]
    for k, c in CL.items():
        h = g[g.loss == k].sort_values("eps")
        ax.plot(h.eps, h.norm, "o-", color=c, lw=2.2 if k in ("square loss (FGW)", "-PMI") else 1.5,
                ms=5, label=k)
    ax.set_title(SEC); ax.set_xlabel("fraction of cells shuffled"); ax.grid(alpha=.25)
    ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
axes[0].set_ylabel("loss, scaled:  0 = truth,  1 = random map")
axes[0].legend(frameon=False, fontsize=8.5)
fig.suptitle("Every candidate structural loss separates the truth from a corrupted map — "
             "the question is how steeply", fontsize=12)
fig.tight_layout()
fig.savefig(HERE / "margin.png", dpi=140, facecolor="white")
print("\nwrote margin.png")
