"""Can we build a CNV clone tree from the B178 scRNA-seq?

B178 is the one patient in GSE318564 (scRNA-seq) that also has a Visium section in
GSE318562 -- i.e. the unmatched scRNA-seq + Visium pair SpOTr needs.

Reference cells are immune (PTPRC+), which are non-malignant by definition, so the CNV
ratio is anchored on real diploid cells rather than on the sample mean.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.io as sio
from scipy.sparse import vstack
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import silhouette_score

HERE = Path(__file__).resolve().parent
ROLL = 101
SAMPLES = ["B178_1", "B178_2", "B178_3"]


def load_10x(d):
    X = sio.mmread(d / "matrix.mtx.gz").T.tocsr()
    feats = pd.read_csv(d / "features.tsv.gz", sep="\t", header=None)
    bcs = pd.read_csv(d / "barcodes.tsv.gz", header=None)[0].to_numpy()
    return X, feats[0].to_numpy(), feats[1].to_numpy(), bcs


def main():
    gpos = pd.read_csv(HERE / "gene_positions.csv")
    mats, ids, syms, origin = [], None, None, []
    for s in SAMPLES:
        X, gid, sym, bc = load_10x(HERE / "b178" / "ext" / f"{s}_filtered_feature_bc_matrix")
        if ids is None:
            ids, syms = gid, sym
        assert list(gid) == list(ids)
        mats.append(X); origin += [s] * X.shape[0]
    X = vstack(mats).tocsr()
    origin = np.array(origin)
    print(f"[sc] loaded {X.shape[0]} cells x {X.shape[1]} genes from {len(SAMPLES)} samples")

    tot = np.asarray(X.sum(1)).ravel()
    ngene = np.asarray((X > 0).sum(1)).ravel()
    keep = (tot > 1000) & (ngene > 500)
    X, origin = X[keep], origin[keep]
    print(f"[sc] after QC (>1000 UMI, >500 genes): {X.shape[0]} cells")

    sym2i = {s: i for i, s in enumerate(syms)}
    def expr(gene):
        i = sym2i.get(gene)
        return np.asarray(X[:, i].todense()).ravel() if i is not None else np.zeros(X.shape[0])

    E_all = np.asarray(X.sum(1)).ravel()
    cpm = lambda v: v / E_all * 1e4
    immune = cpm(expr("PTPRC")) > 1.0
    print(f"[sc] PTPRC+ immune reference cells: {immune.sum()}")

    # --- CNV profile, referenced on immune cells ---
    g = pd.DataFrame({"gene_id": ids}).merge(gpos, on="gene_id", how="left")
    ok = g["chrom"].notna().to_numpy()
    Xc, g = X[:, ok], g[ok].reset_index(drop=True)
    order = np.lexsort((g["start"].to_numpy(), g["chrom"].to_numpy()))
    Xc, g = Xc[:, order], g.iloc[order].reset_index(drop=True)
    E = np.asarray(Xc.todense(), dtype=np.float32) / E_all[:, None] * 1e4
    expressed = (E > 0).mean(0) >= 0.05
    E, g = E[:, expressed], g[expressed].reset_index(drop=True)
    ref = np.maximum(E[immune].mean(0), 1e-3)
    L = np.clip(np.log2(E / ref + 1.0), -1.5, 1.5)

    chrom = g["chrom"].to_numpy()
    S = np.empty_like(L)
    for c in np.unique(chrom):
        idx = np.where(chrom == c)[0]
        S[:, idx] = pd.DataFrame(L[:, idx].T).rolling(ROLL, center=True, min_periods=1).mean().to_numpy().T
    S -= S[immune].mean(0)                       # centre on the immune reference

    score = S[:, chrom == 7].mean(1) - S[:, chrom == 10].mean(1)
    thr = np.percentile(score[immune], 99)
    malignant = (score > thr) & ~immune
    print(f"[sc] malignant cells (chr7-chr10 above 99th pct of immune): {malignant.sum()}")
    print(f"[sc]   immune score {score[immune].mean():+.3f}  malignant {score[malignant].mean():+.3f}")

    Sm = S[malignant]
    Z = linkage(Sm, method="ward")
    print("\n[tree] clone structure at increasing k:")
    print(f"  {'k':>3} {'silhouette':>11} {'min clone':>10} {'chr7':>7} {'chr10':>7}  sizes")
    best = None
    for k in range(2, 9):
        lab = fcluster(Z, k, criterion="maxclust")
        sizes = np.bincount(lab)[1:]
        if sizes.min() < 20:
            continue
        sil = silhouette_score(Sm, lab, sample_size=min(3000, len(Sm)), random_state=0)
        means = np.stack([Sm[lab == c].mean(0) for c in np.unique(lab)])
        c7 = means[:, chrom == 7].mean(1); c10 = means[:, chrom == 10].mean(1)
        print(f"  {k:>3} {sil:>11.3f} {sizes.min():>10} "
              f"{c7.min():+.2f}/{c7.max():+.2f} {c10.min():+.2f}/{c10.max():+.2f}  {sizes.tolist()}")
        if best is None or sil > best[1]:
            best = (k, sil)
    if best:
        print(f"\n[tree] best k = {best[0]} (silhouette {best[1]:.3f})")

    np.savez_compressed(HERE / "b178_sc_cnv.npz", S=Sm, chrom=chrom, score=score[malignant],
                        origin=origin[malignant],
                        genes=g["gene_id"].to_numpy()[:], starts=g["start"].to_numpy())
    print(f"[sc] wrote b178_sc_cnv.npz  ({Sm.shape[0]} malignant cells x {Sm.shape[1]} genes)")


if __name__ == "__main__":
    main()
