"""SCOPING: does lineage predict position differently at the tumour edge than in its core?

If clones at the invasive front have dispersed more, the lineage should predict their positions
LESS well there.  Stratify cell pairs by how deep in the tumour both cells sit
(obs['tumor_boundary_dist'], 40-696 um), and measure within-stratum corr(divergence age,
separation).  The null shuffles which cell carries which tree row WITHIN the stratum, which
preserves that stratum's geometry exactly and destroys only the lineage link.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np, pandas as pd, pycea
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
HERE = Path(__file__).resolve().parent

t = pycea.datasets.koblan25("tumor")
obs = t.obs
rng = np.random.default_rng(0)
rows = []
for SEC in ["M3-4", "M3-1", "M3-2"]:
    d = np.load(HERE.parent / "petracer" / f"petracerFAIR_{SEC}.npz", allow_pickle=True)
    names = np.asarray(d["cell_names"]).astype(str)
    CT = np.asarray(d["C_tree"], float); xy = np.asarray(d["xy_sc"], float)
    bd = obs.loc[names, "tumor_boundary_dist"].to_numpy(float)
    fit = obs.loc[names, "fitness"].to_numpy(float)
    q = np.quantile(bd, [1/3, 2/3])
    strata = {"edge  (outer third)": bd <= q[0],
              "middle": (bd > q[0]) & (bd <= q[1]),
              "core  (inner third)": bd > q[1]}
    for nm, mask in strata.items():
        idx = np.where(mask)[0]
        iu = np.triu_indices(len(idx), 1)
        a = CT[np.ix_(idx, idx)][iu] / 2.0
        dd = squareform(pdist(xy[idx]))[iu]
        obs_r = np.corrcoef(a, dd)[0, 1]
        nulls = []
        for _ in range(40):
            perm = rng.permutation(len(idx))
            an = CT[np.ix_(idx[perm], idx[perm])][iu] / 2.0
            nulls.append(np.corrcoef(an, dd)[0, 1])
        z = (obs_r - np.mean(nulls)) / max(np.std(nulls), 1e-9)
        rows.append(dict(section=SEC, stratum=nm, n=len(idx), corr=obs_r,
                         null=np.mean(nulls), z=z,
                         med_bd=float(np.median(bd[mask])),
                         med_fit=float(np.median(fit[mask]))))
R = pd.DataFrame(rows)
print("Lineage predictivity by depth in the tumour")
print("(corr between divergence age and separation; null shuffles the tree within the stratum)\n")
print(R.round(3).to_string(index=False))
print("\nmean over sections:")
print(R.groupby("stratum")[["corr", "null", "z", "med_fit"]].mean().round(3).to_string())
R.to_csv(HERE / "boundary.csv", index=False)
