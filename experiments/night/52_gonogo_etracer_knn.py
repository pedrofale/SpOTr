"""GO/NO-GO (2026-10-12): does FGW beat a spatial neighbour vote among barcoded spots on eTRACER?

The application is assigning lineage to spatial units whose barcodes were not read.  On eTRACER the
tree-free way to do that is to look at the barcoded spots nearby.  22_etracer_map.py never ran that
baseline.  This script scores it on exactly the same footing: same k-d spots, same seeds, same
reference/held-out halves (perm[:m//2] / perm[m//2:]), same clone-density metric (`score` below is
22_'s, applied to a predicted spot x clone density instead of a coupling).

The vote uses the reference half's clone composition, which FGW does not use -- that is the point:
in a real experiment those barcoded spots exist, so a method has to beat what they already give.
k is chosen by leave-one-out on the reference half only.  FGW numbers are read from etracer_map.csv.
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

HERE = Path(__file__).resolve().parent
KGRID = [1, 2, 3, 5, 8, 12, 20, 30]


def score_density(P, comp_held, clones):
    """22_etracer_map.score with P[held_spot, clone] in place of T[lab == c][:, held].sum(0)."""
    rs, ws = [], []
    for k, _ in enumerate(clones):
        tru = comp_held[:, k]
        if tru.sum() < 20 or tru.std() == 0:
            continue
        pred = P[:, k]
        if pred.sum() <= 0 or pred.std() == 0:
            continue
        rs.append(pearsonr(pred / pred.sum(), tru / tru.sum()).statistic); ws.append(tru.sum())
    return float(np.average(rs, weights=ws))


def knn_density(xy_from, comp_from, xy_to, k, exclude_self=False):
    D = cdist(xy_to, xy_from)
    if exclude_self:
        np.fill_diagonal(D, np.inf)
    nn = np.argsort(D, axis=1)[:, :k]
    frac = comp_from / np.maximum(comp_from.sum(1, keepdims=True), 1e-12)
    return frac[nn].mean(1)


rows = []
for tag in ["LE", "EE"]:
    d = np.load(HERE / f"etracer_spots_{tag}.npz", allow_pickle=True)
    comp = d["spot_comp"].astype(float); xy = d["xy_st"].astype(float)
    clones = d["clones"].astype(str); m = len(xy)
    for seed in (0, 1, 2):
        perm = np.random.default_rng(seed).permutation(m)
        ref, held = perm[:m // 2], perm[m // 2:]
        lab_ref = comp[ref].sum(1) > 0                     # reference spots with any barcoded bin
        r_ref, c_ref = xy[ref][lab_ref], comp[ref][lab_ref]
        # k by leave-one-out on the reference half only
        loo = {k: score_density(knn_density(r_ref, c_ref, r_ref, k, True), c_ref, clones)
               for k in KGRID}
        k_sel = max(loo, key=loo.get)
        for k in KGRID:
            rows.append(dict(tag=tag, seed=seed, arm=f"kNN k={k}", k=k, selected=(k == k_sel),
                             r=score_density(knn_density(r_ref, c_ref, xy[held], k), comp[held], clones)))
        prior = np.tile(c_ref.sum(0), (len(held), 1))
        rows.append(dict(tag=tag, seed=seed, arm="clone prior (flat)", k=0, selected=False,
                         r=np.nan))  # a flat map has zero variance -> undefined correlation

df = pd.DataFrame(rows)
df.to_csv(HERE / "gonogo_etracer_knn.csv", index=False)

fgw = pd.read_csv(HERE / "etracer_map.csv")
print("clone-density agreement r on held-out spots (mean over 3 seeds; higher is better)\n")
for tag in ["LE", "EE"]:
    g = fgw[fgw.tag == tag]
    e = g[g.w == 0].groupby("seed").r.first()
    out = {"expression-only OT": e.values}
    for arm in ["FGW", "learned"]:
        h = g[(g.arm == arm) & (g.w > 0)]
        loso = []
        for s in sorted(h.seed.unique()):
            wsel = h[h.seed != s].groupby("w").r.mean().idxmax()
            loso.append(h[(h.seed == s) & (h.w == wsel)].r.iloc[0])
        out[f"{arm} (w by other seeds)"] = np.array(loso)
        out[f"{arm} (best w, oracle)"] = h.groupby("seed").r.max().values
    k = df[(df.tag == tag) & df.arm.str.startswith("kNN")]
    out["kNN vote (k by LOO on reference)"] = k[k.selected].sort_values("seed").r.values
    out["kNN vote (best k, oracle)"] = k.groupby("seed").r.max().values
    print(tag)
    for name, v in out.items():
        print(f"   {name:<36} {v.mean():.3f}   per seed " + " ".join(f"{x:.3f}" for x in v))
    print(f"   k selected per seed: {k[k.selected].sort_values('seed').k.tolist()}\n")
