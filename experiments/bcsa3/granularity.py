"""How do the preconditions behave as a function of tree granularity?

Numbat on Visium resolves only a handful of clones, but the scRNA-seq side can carry a
much richer phylogeny. The question that matters is then: at what granularity does the
tree still map to spatial locations? We cut the clone tree into k groups (k = 2..K),
relabel the spots by their coarse clade, and re-measure both preconditions at each k.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tree_evidence_all import RUNS, clone_tree_distances, load_coords  # noqa: E402

rng = np.random.default_rng(0)


def coherence(xy, lab, k=12):
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]
    same = (lab[nbrs] == lab[:, None]).mean()
    p = np.array([(lab == c).mean() for c in np.unique(lab)])
    chance = (p ** 2).sum()
    return (same - chance) / (1 - chance)


def mantel(Dt, xy, lab, groups, n_perm=5000):
    g = [c for c in groups if (lab == c).sum() >= 20]
    if len(g) < 4:
        return np.nan, np.nan
    cent = np.stack([xy[lab == c].mean(0) for c in g])
    Ds = np.linalg.norm(cent[:, None] - cent[None, :], axis=-1)
    sub = Dt[np.ix_([groups.index(c) for c in g], [groups.index(c) for c in g])]
    iu = np.triu_indices(len(g), 1)
    r = np.corrcoef(sub[iu], Ds[iu])[0, 1]
    null = np.array([np.corrcoef(sub[np.ix_(p, p)][iu], Ds[iu])[0, 1]
                     for p in (rng.permutation(len(g)) for _ in range(n_perm))])
    return r, (np.sum(np.abs(null) >= abs(r)) + 1) / (n_perm + 1)


def analyse(label, nbdir, source, tree_csv, spatial_dir):
    nbdir = Path(nbdir)
    nb = pd.read_csv(nbdir / "clone_post_2.tsv", sep="\t").merge(
        pd.read_csv(nbdir / "cell_metadata.csv"), on="cell")
    v = nb[nb["source"] == source].copy()
    pre = source.replace("Visium_", "")
    v["bc"] = v["cell"].str.replace(f"^{pre}_", "", regex=True)
    coords = load_coords(spatial_dir)
    v = v[v["bc"].isin(coords.index)]
    xy = coords.loc[v["bc"]].to_numpy()
    truth = v["clone_opt"].to_numpy()

    nodes, Dtree = clone_tree_distances(HERE / tree_csv)
    pos = {c: i for i, c in enumerate(nodes)}
    keep = np.array([c in pos for c in truth])
    xy, truth = xy[keep], truth[keep]

    # agglomerate clones by tree distance -> coarse clades
    Z = linkage(squareform(Dtree, checks=False), method="average")
    out = []
    for k in range(2, len(nodes) + 1):
        grp = fcluster(Z, k, criterion="maxclust")
        cmap = {c: grp[pos[c]] for c in nodes}
        lab = np.array([cmap[c] for c in truth])
        gs = sorted(set(grp))
        # tree distance between coarse clades = mean pairwise distance between members
        Dg = np.zeros((len(gs), len(gs)))
        for i, a in enumerate(gs):
            for j, bb in enumerate(gs):
                ia = [pos[c] for c in nodes if cmap[c] == a]
                ib = [pos[c] for c in nodes if cmap[c] == bb]
                Dg[i, j] = Dtree[np.ix_(ia, ib)].mean()
        r, p = mantel(Dg, xy, lab, gs)
        out.append(dict(dataset=label, k=k, n_groups=len(set(lab)),
                        coherence=coherence(xy, lab), mantel_r=r, mantel_p=p))
    return out


if __name__ == "__main__":
    rows = [r for spec in RUNS for r in analyse(*spec)]
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for ds, sub in df.groupby("dataset", sort=False):
        print(f"\n=== {ds}")
        print(sub[["k", "n_groups", "coherence", "mantel_r", "mantel_p"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    df.to_csv(HERE / "granularity.csv", index=False)
