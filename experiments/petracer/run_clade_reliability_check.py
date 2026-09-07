"""Settle it: was the +0.473 clade advantage real, or an undersampled control?

Reuses the saved split-half solves. Matches the earlier analysis exactly (clades with >=30
cells, correlation of per-clade optimal alpha between halves), but with 500 size-matched
random groupings instead of the 3 the earlier run used.
"""
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

HERE = Path(__file__).resolve().parent
GRID = np.array([0.0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90])
SEC = "M3-4"; NDRAW = 500
rng = np.random.default_rng(2)
HALVES = np.load(HERE / f"granularity_halves_{SEC}.npy")
d = np.load(HERE / f"petracer_{SEC}.npz", allow_pickle=True)
cc = np.asarray(d["cell_clade"]).astype(str)
clades = list(np.asarray(d["clades"]).astype(str))
cidx = np.array([clades.index(c) for c in cc]); n = len(cc)
keep = [k for k in range(len(clades)) if (cidx == k).sum() >= 30]
sizes = [int((cidx == k).sum()) for k in keep]
print(f"{len(keep)} clades with >=30 cells, sizes {sorted(sizes, reverse=True)}")

def gopt(E, lab, groups):
    return np.array([GRID[E[lab == k].mean(0).argmin()] for k in groups])

obs, null = [], []
for EA, EB in HALVES:
    oA, oB = gopt(EA, cidx, keep), gopt(EB, cidx, keep)
    obs.append(pearsonr(oA, oB)[0])
    for _ in range(NDRAW):
        perm = rng.permutation(n); rl = np.full(n, -1); p0 = 0
        for gi, sz in enumerate(sizes): rl[perm[p0:p0+sz]] = gi; p0 += sz
        a_, b_ = gopt(EA, rl, range(len(sizes))), gopt(EB, rl, range(len(sizes)))
        if a_.std() > 0 and b_.std() > 0:
            v = pearsonr(a_, b_)[0]
            if np.isfinite(v): null.append(v)
null = np.array(null); null = null[np.isfinite(null)]; o = float(np.mean(obs))
p = (np.sum(null >= o) + 1) / (len(null) + 1)
print(f"\n  observed clade reliability r = {o:+.3f}   (per split: "
      f"{', '.join(f'{v:+.3f}' for v in obs)})")
print(f"  size-matched RANDOM null, {len(null)} draws: mean {null.mean():+.3f} "
      f"sd {null.std():.3f}, 95th pct {np.percentile(null,95):+.3f}")
print(f"  z = {(o-null.mean())/null.std():+.2f}   p = {p:.3f}")
print(f"\n  the earlier run used 3 random draws and happened to draw "
      f"{'a low' if null.mean() > 0.2 else 'an unremarkable'} value; "
      f"P(a single draw <= 0.006) = {np.mean(null <= 0.006):.3f}")
