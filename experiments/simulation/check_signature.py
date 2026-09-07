"""Does the branching simulator reproduce PEtracer's signature?"""
import sys, numpy as np, ot
from pathlib import Path
from scipy.stats import pearsonr
from scipy.spatial.distance import squareform, pdist
from scipy.cluster.hierarchy import linkage, fcluster
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "petracer"))
from local_concordance import local_score
import sim_branching

def signature(CT, xy, grp, clade, tag):
    n = len(CT)
    iu = np.triu_indices(n, 1)
    t = CT[iu]
    same_cl = clade[iu[0]] == clade[iu[1]]
    frac_max = float(np.mean(t >= t.max() * 0.999))
    cv = float(t[~same_cl].std() / t[~same_cl].mean())
    gl, lo = [], []
    for g in np.unique(grp):
        s = np.where(grp == g)[0]
        if len(s) > 1200:
            s = np.random.default_rng(0).choice(s, 1200, replace=False)
        i2 = np.triu_indices(len(s), 1)
        gl.append(pearsonr(CT[np.ix_(s, s)][i2], squareform(pdist(xy[s]))[i2])[0])
    lo = 1 - local_score(CT, xy, 5, restrict=grp)
    print(f"{tag:<22}{np.mean(gl):>9.3f}{lo.mean():>9.3f}{cv:>9.3f}{frac_max:>9.1%}")

print(f"{'':<22}{'global':>9}{'local':>9}{'btwn CV':>9}{'@max':>9}")
print("-" * 58)
for SEC in ["M3-4", "M3-1", "M3-2"]:
    d = np.load(HERE.parent / "petracer" / f"petracer_{SEC}.npz", allow_pickle=True)
    CT = np.asarray(d["C_tree"], float)
    cc = np.asarray(d["cell_clade"]).astype(str)
    _, ci = np.unique(cc, return_inverse=True)
    br = fcluster(linkage(squareform(CT, checks=False), "average"), 2, "maxclust") - 1
    signature(CT, np.asarray(d["xy_sc"], float), br, ci, f"PEtracer {SEC}")
print()
for sg in [0.35, 0.55, 0.85]:
    s_ = sim_branching.simulate(0, sigma=sg)
    signature(s_["C_tree"], s_["xy_sc"], s_["branch"], s_["clade"], f"branching sigma={sg}")

print()
for bs in [1.6, 8.0, 40.0, 300.0]:
    s_ = sim_branching.simulate(0, sigma=0.55, bl_shape=bs)
    signature(s_["C_tree"], s_["xy_sc"], s_["branch"], s_["clade"], f"branching bl_shape={bs:g}")
