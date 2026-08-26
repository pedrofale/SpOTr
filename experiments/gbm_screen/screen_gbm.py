"""Screen the Greenwald et al. glioblastoma Visium cohort for SpOTr's preconditions.

For each section: call CNV clones from the spot expression, build a tree over those
clones, then ask the two questions that decided BCSA:
  (1) do the clone labels form spatial territories?   (coherence)
  (2) does tree distance predict territory distance?  (Mantel)
Both are measured across tree granularities, and both are computed on MALIGNANT spots
only -- the normal-like cluster is dropped, because a tumour/normal boundary inflates
coherence without saying anything about subclonal structure (the BCSA2 lesson).
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.spatial.distance import squareform, pdist
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)

ROLL = 101          # genomic rolling window (genes)
N_CLONES = 8        # finest clone partition to build the tree from
MIN_SPOTS = 20      # a clade must hold this many spots to enter the Mantel test
CNV_SCORE_MIN = 0.02  # chr7 - chr10 log2 signal required to call a cluster malignant


def load_sample(d):
    pre = None
    for f in d.glob("*_matrix.mtx"):
        if not f.name.startswith("._"):
            pre = f.name[:-len("_matrix.mtx")]
    if pre is None:
        return None
    X = sio.mmread(d / f"{pre}_matrix.mtx").T.tocsr()          # spots x genes
    feats = pd.read_csv(d / f"{pre}_features.tsv", sep="\t", header=None)
    bcf = d / f"{pre}_barcodes.tsv"
    if not bcf.exists():
        bcf = d / f"{pre}_barcodes.tsv.gz"
    bcs = pd.read_csv(bcf, header=None)[0].to_numpy()
    posf = d / f"{pre}_tissue_positions_list.csv"
    if not posf.exists():
        return None                      # no coordinates -> cannot be screened
    pos = pd.read_csv(posf, header=None,
                      names=["bc", "in_tissue", "row", "col", "pxl_row", "pxl_col"]).set_index("bc")
    pos = pos.reindex(bcs)
    keep = pos["in_tissue"].to_numpy() == 1
    return X[keep], feats[0].to_numpy(), pos.loc[bcs[keep], ["pxl_row", "pxl_col"]].to_numpy(float)


def cnv_profile(X, genes, gpos):
    """Per-spot relative CNV profile: log ratio to the section mean, smoothed along the genome."""
    g = pd.DataFrame({"gene_id": genes}).merge(gpos, on="gene_id", how="left")
    ok = g["chrom"].notna().to_numpy()
    X, g = X[:, ok], g[ok].reset_index(drop=True)
    order = np.lexsort((g["start"].to_numpy(), g["chrom"].to_numpy()))
    X, g = X[:, order], g.iloc[order].reset_index(drop=True)

    tot = np.asarray(X.sum(1)).ravel()
    keep = tot > 500                                  # drop near-empty spots
    X, tot = X[keep], tot[keep]
    E = np.asarray(X.todense(), dtype=np.float32) / tot[:, None] * 1e4

    expressed = E.mean(0) >= 0.5                      # quiet genes give meaningless ratios
    E, g = E[:, expressed], g[expressed].reset_index(drop=True)
    L = np.log2(E / np.maximum(E.mean(0), 1e-3) + 1.0)
    L = np.clip(L, -1.5, 1.5)

    chrom = g["chrom"].to_numpy()
    S = np.empty_like(L)
    for c in np.unique(chrom):
        idx = np.where(chrom == c)[0]
        S[:, idx] = pd.DataFrame(L[:, idx].T).rolling(
            ROLL, center=True, min_periods=1).mean().to_numpy().T
    S -= S.mean(1, keepdims=True)                     # centre each spot
    return S, chrom, keep


def coherence(xy, lab, k=12):
    nb = NearestNeighbors(n_neighbors=min(k, len(xy) - 1) + 1).fit(xy)
    nbrs = nb.kneighbors(xy, return_distance=False)[:, 1:]
    same = (lab[nbrs] == lab[:, None]).mean()
    p = np.array([(lab == c).mean() for c in np.unique(lab)])
    chance = (p ** 2).sum()
    return (same - chance) / (1 - chance) if chance < 1 else np.nan


def mantel(Dt, xy, lab, groups, n_perm=5000):
    g = [c for c in groups if (lab == c).sum() >= MIN_SPOTS]
    if len(g) < 4:
        return np.nan, np.nan, len(g)
    cent = np.stack([xy[lab == c].mean(0) for c in g])
    Ds = squareform(pdist(cent))
    sub = Dt[np.ix_([groups.index(c) for c in g], [groups.index(c) for c in g])]
    iu = np.triu_indices(len(g), 1)
    r = np.corrcoef(sub[iu], Ds[iu])[0, 1]
    null = np.array([np.corrcoef(sub[np.ix_(p, p)][iu], Ds[iu])[0, 1]
                     for p in (rng.permutation(len(g)) for _ in range(n_perm))])
    return r, (np.sum(np.abs(null) >= abs(r)) + 1) / (n_perm + 1), len(g)


def screen(d, gpos):
    got = load_sample(d)
    if got is None:
        return []
    X, genes, xy = got
    S, chrom, keep = cnv_profile(X, genes, gpos)
    xy = xy[keep]

    Zs = linkage(S, method="ward")
    clone = fcluster(Zs, N_CLONES, criterion="maxclust")
    cl = np.unique(clone)
    means = np.stack([S[clone == c].mean(0) for c in cl])

    # Malignant = clusters carrying the canonical IDH-wt GBM axis (chr7 gain + chr10 loss).
    # The reference here is the section mean, so normal spots appear *inverted* on that axis
    # (chr7 down, chr10 up) rather than flat -- picking the flattest cluster as "normal"
    # therefore leaves real normal spots in, and a tumour/normal boundary would inflate
    # coherence without saying anything about subclones.
    score = np.array([means[i][chrom == 7].mean() - means[i][chrom == 10].mean()
                      for i in range(len(cl))])
    malignant_clusters = cl[score > CNV_SCORE_MIN]
    mal = np.isin(clone, malignant_clusters)
    if mal.sum() < 100 or len(malignant_clusters) < 3:
        return []

    # tree over the malignant clones only
    keep_c = np.isin(cl, malignant_clusters)
    cl_m, means_m = cl[keep_c], means[keep_c]
    Zc = linkage(means_m, method="average")
    Dt_full = squareform(cophenet(Zc))

    rows = []
    for k in range(2, len(cl_m) + 1):
        grp = fcluster(Zc, k, criterion="maxclust")
        cmap = {c: grp[i] for i, c in enumerate(cl_m)}
        lab = np.array([cmap.get(c, -1) for c in clone])
        gs = sorted(set(grp))
        Dg = np.zeros((len(gs), len(gs)))
        for i, a in enumerate(gs):
            for j, b in enumerate(gs):
                ia = [x for x, c in enumerate(cl_m) if cmap[c] == a]
                ib = [x for x, c in enumerate(cl_m) if cmap[c] == b]
                Dg[i, j] = Dt_full[np.ix_(ia, ib)].mean()
        r, p, ng = mantel(Dg, xy[mal], lab[mal], gs)
        rows.append(dict(sample=d.name, k=k, n_clades=len(set(lab[mal])),
                         coherence_malignant=coherence(xy[mal], lab[mal]),
                         n_malignant_clones=len(cl_m),
                         mantel_r=r, mantel_p=p, n_groups_tested=ng,
                         n_spots=int(mal.sum()), cnv_score_max=float(score.max())))
    return rows


if __name__ == "__main__":
    gpos = pd.read_csv(HERE / "gene_positions.csv")
    out = []
    for d in sorted((HERE / "ext").iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        try:
            r = screen(d, gpos)
            out += r
            if r:
                best = max(r, key=lambda x: x["coherence_malignant"])
                print(f"{d.name:<22} best coherence {best['coherence_malignant']:.3f} "
                      f"(k={best['k']}, {best['n_spots']} malignant spots)")
        except Exception as e:
            print(f"{d.name:<22} FAILED {type(e).__name__}: {str(e)[:80]}")
    pd.DataFrame(out).to_csv(HERE / "gbm_screen.csv", index=False)
    print(f"\nwrote gbm_screen.csv ({len(out)} rows)")
