"""Copy-number read-outs of the HT260C1 nuclei and spots, from their expression and (for the nuclei) their ATAC peaks.

Arms.  A gene or a peak belongs to the p or q arm of its chromosome (1-22) by its start against the centromere (GRCh38, CENTROMERE_MB).
Smoothed RNA arm scores (inferCNV-style), per row (nucleus or spot): log1p of the counts per 10^4 on the genes detected in at least
`min_det` of the rows, centred on the reference rows' mean, smoothed by a running mean over 101 genes in genomic order within each
chromosome, recentred by the row's median; each arm's mean over its genes, for the arms with at least 20 genes.
Unsmoothed RNA arm means: each arm's mean log1p counts per 10^4 over its genes detected in at least 5% of the nuclei, less the non-tumour
nuclei's mean.
ATAC arm coverage: each arm's share of a nucleus's peak counts (arms with at least 200 peaks), as log2 of its ratio to the non-tumour
nuclei's mean.
"""
import gzip

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse
from scipy.ndimage import uniform_filter1d

CENTROMERE_MB = {"1": 123.4, "2": 93.9, "3": 90.9, "4": 50.0, "5": 48.8, "6": 59.8, "7": 60.1, "8": 45.2, "9": 43.0, "10": 39.8, "11": 53.4,
                 "12": 35.5, "13": 17.7, "14": 17.2, "15": 19.0, "16": 36.8, "17": 25.1, "18": 18.5, "19": 26.2, "20": 28.1, "21": 12.0,
                 "22": 15.0}


def read10x(prefix):
    """a 10x matrix (rows x features, CSR), its features table and its barcodes"""
    M = scipy.sparse.csr_matrix(scipy.io.mmread(gzip.open(f"{prefix}-matrix.mtx.gz")).T)
    f = pd.read_csv(f"{prefix}-features.tsv.gz", sep="\t", header=None); b = pd.read_csv(f"{prefix}-barcodes.tsv.gz", header=None)[0].to_numpy()
    return M, f, b


def arm(chrom, start):
    c = str(chrom); return f"chr{c}{'p' if start < CENTROMERE_MB[c] * 1e6 else 'q'}"


def gene_positions(path):
    return pd.read_csv(path).drop_duplicates("gene_id").set_index("gene_id")


def arm_scores(X, ids, gp, ref, min_det, allowed=None, total="all", min_genes=20, window=101):
    """smoothed RNA arm scores (rows x arms); X: rows x genes counts (CSR) with gene ids `ids`; ref: the reference rows; allowed: the genes
    that may be used (default all); total: the counts per 10^4 against every gene of X ("all") or against the genes used ("used")"""
    det = np.asarray((X > 0).mean(0)).ravel() >= min_det
    ok = det & np.array([i in gp.index and str(gp.loc[i, "chrom"]) in CENTROMERE_MB and (allowed is None or i in allowed) for i in ids])
    j = np.where(ok)[0]; ch = np.array([str(gp.loc[ids[k], "chrom"]) for k in j]); st = np.array([gp.loc[ids[k], "start"] for k in j], float)
    o = np.lexsort((st, ch.astype(int))); j, ch, st = j[o], ch[o], st[o]; arms = np.array([arm(c, s) for c, s in zip(ch, st)])
    Xj = X[:, j]; tot = np.asarray(X.sum(1), float) if total == "all" else np.asarray(Xj.sum(1), float)
    L = np.log1p(np.asarray(Xj.todense(), float) / tot * 1e4); L -= L[ref].mean(0)
    for c in np.unique(ch):
        m = ch == c; L[:, m] = uniform_filter1d(L[:, m], size=min(window, m.sum()), axis=1, mode="nearest")
    L -= np.median(L, axis=1, keepdims=True)
    keep = [a for a in pd.unique(arms) if (arms == a).sum() >= min_genes]
    return pd.DataFrame({a: L[:, arms == a].mean(1) for a in keep})


def arm_means(G, ids, gp, ref):
    """unsmoothed RNA arm means (rows x arms), arms in the order the genes first name them"""
    umi = np.asarray(G.sum(1)).ravel().astype(float); det = np.asarray((G > 0).mean(0)).ravel() >= 0.05
    arm_g = np.array([arm(gp.loc[g, "chrom"], gp.loc[g, "start"]) if (g in gp.index and str(gp.loc[g, "chrom"]) in CENTROMERE_MB and d) else ""
                      for g, d in zip(ids, det)])
    Lg = G.multiply(1.0 / np.maximum(umi, 1)[:, None]).tocsc() * 1e4; Lg.data = np.log1p(Lg.data)
    arms = [a for a in pd.unique(arm_g) if a and (arm_g == a).sum() >= 20]
    R = np.column_stack([np.asarray(Lg[:, np.where(arm_g == a)[0]].mean(1)).ravel() for a in arms])
    return pd.DataFrame(R - R[ref].mean(0), columns=arms), umi


def peak_arms(f):
    """each peak's arm ('' off chromosomes 1-22) and the arms with at least 200 peaks"""
    chrom = f[3].astype(str).str.replace("chr", ""); start = f[4].to_numpy(float)
    a = np.array([arm(c, s) if c in CENTROMERE_MB else "" for c, s in zip(chrom, start)])
    return a, [x for x in pd.unique(a) if x and (a == x).sum() >= 200]


def atac_arms(P, f, ref):
    """ATAC arm coverage (rows x arms) and each row's peak total; P: rows x peaks counts, f: the peaks' features rows"""
    a, arms = peak_arms(f); P = P.tocsc(); tot = np.asarray(P.sum(1)).ravel().astype(float)
    F = pd.DataFrame(np.column_stack([np.asarray(P[:, np.where(a == x)[0]].sum(1)).ravel() for x in arms]) / np.maximum(tot[:, None], 1.0), columns=arms)
    return np.log2(F.divide(F[ref].mean(0), axis=1).clip(lower=1e-3)), tot


def regress_out(X, covariate):
    """X less its least-squares fit on [1, covariate], each column then standardised"""
    D = np.column_stack([np.ones(len(covariate)), covariate]); R = X - D @ np.linalg.lstsq(D, X, rcond=None)[0]
    return (R - R.mean(0)) / np.maximum(R.std(0), 1e-12)


def euclidean(Z):
    """Euclidean distances between the rows of Z"""
    q = (Z ** 2).sum(1); return np.sqrt(np.maximum(q[:, None] + q[None, :] - 2.0 * Z @ Z.T, 0.0))


# ------------------------------------------------------------------------------------------------------------------- the copy-number tree
def profile_distances(Z):
    """the cancer nuclei's copy-number distances: Euclidean between their profiles, in single precision"""
    return euclidean(np.asarray(Z, float)).astype(np.float32).astype(float)


def clone_distances(d, labels):
    """the mean distance between the nuclei of each pair of clones (0 on the diagonal); labels 0 .. K - 1"""
    K = int(labels.max()) + 1
    return np.array([[d[np.ix_(labels == a, labels == b)].mean() if a != b else 0.0 for b in range(K)] for a in range(K)])


def clone_tree(d, labels):
    """the tree SpOTr is given, as distances between the nuclei: Ward linkage within each clone, its cophenetic distances scaled so that the
    deepest within a clone equals the shallowest distance between clones; between clones, the clones' mean distances"""
    from scipy.cluster.hierarchy import linkage, cophenet
    from scipy.spatial.distance import squareform
    Dk = clone_distances(d, labels); out = Dk[np.ix_(labels, labels)].astype(float).copy(); off = Dk[~np.eye(len(Dk), dtype=bool)].min()
    for k in range(len(Dk)):
        i = np.where(labels == k)[0]; co = squareform(cophenet(linkage(squareform(d[np.ix_(i, i)], checks=False), "ward")))
        out[np.ix_(i, i)] = co / max(co.max(), 1e-12) * off
    return out


def clone_linkage(d, labels, order):
    """the same tree as a scipy linkage, for drawing: Ward within each clone (its root at 95% of the shallowest distance between clones), the
    clones joined by their mean distances, `order` (three clone indices) giving the first two clones joined and then the third"""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    n = len(labels); Dk = clone_distances(d, labels); off = Dk[~np.eye(len(Dk), dtype=bool)].min(); rows, nid, root = [], n, {}
    for k in order:
        idx = np.where(labels == k)[0]; Lk = linkage(squareform(d[np.ix_(idx, idx)], checks=False), "ward"); sc = 0.95 * off / Lk[:, 2].max()
        g = list(idx) + [None] * (len(idx) - 1)
        for r, (a, b, h, c) in enumerate(Lk):
            rows.append([g[int(a)], g[int(b)], h * sc, c]); g[len(idx) + r] = nid; nid += 1
        root[k] = (nid - 1, len(idx))
    a, b, c = order
    rows.append([root[a][0], root[b][0], Dk[a, b], root[a][1] + root[b][1]]); nid += 1
    rows.append([nid - 1, root[c][0], (Dk[a, c] + Dk[b, c]) / 2, n]); return np.array(rows, float)
