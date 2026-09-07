"""Proper controls for the granularity sweep. Reuses the saved split-half solves.

Fixes two flaws in the first pass:
  * the size-matched random control used ONE draw per split -- far too noisy (it gave 0.449 at
    18 groups where a better-controlled run gave 0.006). Now 200 draws.
  * correlation over 2-4 points is meaningless. Exact agreement of the selected alpha is the
    primary metric, with its own permutation null.
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr

HERE = Path(__file__).resolve().parent
GRID = np.array([0.0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90])
LEVELS = [2, 3, 4, 6, 9, 18]
NDRAW = 200
SEC = "M3-4"
rng = np.random.default_rng(1)

HALVES = np.load(HERE / f"granularity_halves_{SEC}.npy")
d = np.load(HERE / f"petracer_{SEC}.npz", allow_pickle=True)
cc = np.asarray(d["cell_clade"]).astype(str)
n = len(cc)
dd = np.load(HERE / "petracer_xy.npz", allow_pickle=True)
Dcl = np.asarray(dd["D"], float); allcl = list(dd["clades"].astype(str))
Zc = linkage(squareform(Dcl, checks=False), "average")
MAP = {L: np.array([(fcluster(Zc, L, "maxclust")-1)[allcl.index(c)] for c in cc]) for L in LEVELS}

def gopt(E, lab):
    return np.array([GRID[E[lab == k].mean(0).argmin()] for k in sorted(set(lab))])
def rand_like(lab):
    sizes = np.bincount(lab); perm = rng.permutation(n); out = np.full(n, -1); p0 = 0
    for gi, sz in enumerate(sizes): out[perm[p0:p0+sz]] = gi; p0 += sz
    return out

rows = []
for L in LEVELS:
    lab = MAP[L]; G = len(set(lab))
    ex, rr, ex_n, rr_n = [], [], [], []
    for EA, EB in HALVES:
        oA, oB = gopt(EA, lab), gopt(EB, lab)
        ex.append(np.mean(oA == oB))
        if G >= 5: rr.append(pearsonr(oA, oB)[0])
        for _ in range(NDRAW):
            rl = rand_like(lab); a_, b_ = gopt(EA, rl), gopt(EB, rl)
            ex_n.append(np.mean(a_ == b_))
            if G >= 5 and a_.std() > 0 and b_.std() > 0: rr_n.append(pearsonr(a_, b_)[0])
    ex_m, exn_m, exn_s = np.mean(ex), np.mean(ex_n), np.std(ex_n)
    p_ex = (np.sum(np.array(ex_n) >= ex_m) + 1) / (len(ex_n) + 1)
    rows.append(dict(groups=G, min_size=int(np.bincount(lab).min()),
                     exact=ex_m, exact_null=exn_m, exact_null_sd=exn_s, p_exact=p_ex,
                     z=(ex_m - exn_m)/max(exn_s, 1e-9),
                     rel=np.mean(rr) if rr else np.nan,
                     rel_null=np.mean(rr_n) if rr_n else np.nan))
r = pd.DataFrame(rows); r.to_csv(HERE / f"granularity_controlled_{SEC}.csv", index=False)
print(f"M3-4, {len(HALVES)} independent spot-half splits, {NDRAW} size-matched random groupings each\n")
print(f"{'groups':>7}{'min size':>10}{'exact agree':>13}{'random null':>14}{'z':>7}{'p':>8}"
      f"{'corr':>8}{'corr null':>11}")
print("-"*78)
for _, x in r.iterrows():
    cs = f"{x.rel:>8.3f}" if np.isfinite(x.rel) else f"{'--':>8}"
    cn = f"{x.rel_null:>11.3f}" if np.isfinite(x.rel_null) else f"{'--':>11}"
    print(f"{int(x.groups):>7}{int(x.min_size):>10}{x.exact:>13.3f}"
          f"{x.exact_null:>9.3f}±{x.exact_null_sd:<4.3f}{x.z:>7.1f}{x.p_exact:>8.3f}{cs}{cn}")
