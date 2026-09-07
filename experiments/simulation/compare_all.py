"""All five statistics, PEtracer vs both generators."""
import sys, numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from scipy.spatial.distance import squareform, pdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.neighbors import NearestNeighbors
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "petracer"))
from local_concordance import local_score
import sim_branching, sim_eden


def stats(CT, xy, clade, branch):
    n = len(CT); iu = np.triu_indices(n, 1)
    t = CT[iu]; D = squareform(pdist(xy))[iu]
    same = clade[iu[0]] == clade[iu[1]]
    coh = D[same].mean() / D.mean()
    gl = np.mean([pearsonr(CT[np.ix_(s, s)][np.triu_indices(len(s), 1)],
                           squareform(pdist(xy[s]))[np.triu_indices(len(s), 1)])[0]
                  for s in [np.where(branch == g)[0] for g in np.unique(branch)]
                  if len(s) > 30])
    lo = (1 - local_score(CT, xy, 5, restrict=branch)).mean()
    nb = NearestNeighbors(n_neighbors=16).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]
    obs = float(np.mean(clade[nb] == clade[:, None]))
    _, cnt = np.unique(clade, return_counts=True)
    terr = obs - float(((cnt / cnt.sum()) ** 2).sum())
    return coh, gl, lo, terr, float(np.mean(t >= t.max() * 0.999)), \
        float(t[~same].std() / t[~same].mean())


hdr = f"{'':<26}{'coher':>7}{'global':>8}{'local':>7}{'terr':>8}{'@max':>7}{'btwnCV':>8}"
print(hdr); print("-" * len(hdr))
for SEC in ["M3-4", "M3-1", "M3-2"]:
    d = np.load(HERE.parent / "petracer" / f"petracer_{SEC}.npz", allow_pickle=True)
    CT = np.asarray(d["C_tree"], float); xy = np.asarray(d["xy_sc"], float)
    cc = np.asarray(d["cell_clade"]).astype(str); _, ci = np.unique(cc, return_inverse=True)
    br = fcluster(linkage(squareform(CT, checks=False), "average"), 2, "maxclust") - 1
    v = stats(CT, xy, ci, br)
    print(f"{'PEtracer ' + SEC:<26}{v[0]:>7.3f}{v[1]:>8.3f}{v[2]:>7.3f}{v[3]:>+8.3f}"
          f"{v[4]:>7.1%}{v[5]:>8.3f}")
print()
d = sim_branching.simulate(0, bl_shape=8.0, sat_q=0.5)
v = stats(d["C_tree"], d["pos"], d["clade"], d["branch"])
print(f"{'free diffusion':<26}{v[0]:>7.3f}{v[1]:>8.3f}{v[2]:>7.3f}{v[3]:>+8.3f}"
      f"{v[4]:>7.1%}{v[5]:>8.3f}")
for lm in [0.0, 0.15, 0.30]:
    d = sim_eden.simulate(0, local_mix=lm)
    v = stats(d["C_tree"], d["pos"], d["clade"], d["branch"])
    print(f"{f'EDEN local_mix={lm}':<26}{v[0]:>7.3f}{v[1]:>8.3f}{v[2]:>7.3f}{v[3]:>+8.3f}"
          f"{v[4]:>7.1%}{v[5]:>8.3f}")
print("\ncoher=clade coherence (low=territories)  global=corr(tree,space)  "
      "local=1-d(treeNN)/d(rand)\nterr=kNN same-clade excess  @max=pairs at tree maximum")
