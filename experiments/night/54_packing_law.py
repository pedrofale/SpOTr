"""FIRST LOOK (2026-10-12): does clone SIZE plus cell DENSITY predict the tree-space law r(a)?

If a clone founded a days ago now holds n(a) cells and grows as a compact patch, it covers area
n(a)/rho in the section (rho = cells per um^2), so its radius is R(a) = sqrt(n(a) / (pi rho)) and the
spread of a pair inside it is r_pack(a) = R(a)/sqrt(2) (uniform disc: E|X-Y|^2 = R^2, r = sqrt(E/2)).

n(a) comes from the TREE ALONE (cells sharing an ancestor at age <= a), rho from the SPATIAL SIDE
ALONE.  Both are counted on the same sample here, so the sampling fraction cancels.  In an unmatched
dataset they come from different samples and only the RELATIVE r_pack across ages survives -- which
is all relative decomposition weights need, the global scale being absorbed by alpha.

Compared against the measured r(a) (48_'s fit_spread).  If measured / packed is roughly constant in
a, one mixing factor and a tree-side count predict the whole curve, with no ground truth.
"""
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform

HERE = Path(__file__).resolve().parent


def measured(age, sep, nbin=22, min_pairs=200):
    cap = age.max(); below = age[age < cap - 1e-9]
    edges = np.r_[np.linspace(below.min(), below.max(), nbin), cap - 1e-9, cap + 1e-9]
    out = []
    for i in range(len(edges) - 1):
        m = (age >= edges[i]) & (age < edges[i + 1])
        if m.sum() >= min_pairs:
            out.append((age[m].mean(), np.sqrt((sep[m] ** 2).mean() / 2.0), m.sum()))
    return np.array(out)


for sec in ["M3-4", "M3-1", "M3-2"]:
    z = np.load(HERE.parent / "petracer" / f"petracerFAIR_{sec}.npz", allow_pickle=True)
    xy = np.asarray(z["xy_sc"], float); AGE = np.asarray(z["C_tree"], float) / 2.0
    n = len(xy)
    rho = n / ConvexHull(xy).volume                       # cells per um^2 in this sample
    iu = np.triu_indices(n, 1)
    sep = squareform(pdist(xy))[iu]; age = AGE[iu]
    M = measured(age, sep)
    pack = []
    for a, _, _ in M:
        nc = (AGE <= a + 1e-9).sum(1)                     # size of each cell's clone founded at age a
        # pairs with LCA at ~a live in the clone founded at a: average r over cells, size-weighted
        R = np.sqrt(nc / (np.pi * rho))
        pack.append(np.sqrt(np.average(R ** 2, weights=nc) / 2.0))
    pack = np.array(pack)
    ratio = M[:, 1] / pack
    sat = M[:, 0] >= age.max() - 1e-6
    print(f"\n{sec}: n={n}, density {rho*1e4:.2f} cells/100um^2, founding age {age.max():.1f} d")
    print("   age(d)   pairs    r_meas(um)  r_pack(um)  meas/pack")
    for (a, r, c), p, q in zip(M, pack, ratio):
        print(f"   {a:6.1f}  {int(c):7d}   {r:9.1f}   {p:9.1f}   {q:6.2f}")
    res = ~sat
    lr = np.corrcoef(np.log(M[res, 1]), np.log(pack[res]))[0, 1]
    print(f"   resolved ages: dynamic range measured {M[:,1].max()/M[:,1].min():.2f}x, "
          f"packed {pack.max()/pack.min():.2f}x;  corr(log meas, log pack) {lr:+.3f};  "
          f"meas/pack {ratio[res].min():.2f}-{ratio[res].max():.2f} (cv {ratio[res].std()/ratio[res].mean():.2f})")
