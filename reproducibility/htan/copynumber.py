"""CNA scores of the HT260C1 nuclei and spots, from their expression and, for the nuclei, their chromatin accessibility (ATAC peaks).

Chromosome arms. A gene or a peak is on the p or q chromosome arm of its chromosome (1-22) by its start relative to the centromere (GRCh38,
CENTROMERE_MB).
Smoothed CNA scores from expression, as in inferCNV, for each row (nucleus or spot). We take log1p of the counts per 10^4 on the genes detected
in at least `min_det` of the rows, subtract the mean of the reference rows, smooth by a moving average over 101 genes in genomic order within
each chromosome, and subtract the median of each row. The CNA score of a chromosome arm is the mean over its genes, for the chromosome arms with
at least 20 genes.
Unsmoothed CNA scores from expression. The CNA score of a chromosome arm is the mean log1p counts per 10^4 over its genes detected in at least 5%
of the nuclei, minus the mean of the non-tumor nuclei.
CNA scores from ATAC. The CNA score of a chromosome arm (with at least 200 peaks) is the log2 ratio between the fraction of the peak counts of a
nucleus on the chromosome arm and the mean of this fraction in the non-tumor nuclei.
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
    """Read a 10x matrix (rows x features, CSR), its table of features and its barcodes."""
    M = scipy.sparse.csr_matrix(scipy.io.mmread(gzip.open(f"{prefix}-matrix.mtx.gz")).T)
    f = pd.read_csv(f"{prefix}-features.tsv.gz", sep="\t", header=None); b = pd.read_csv(f"{prefix}-barcodes.tsv.gz", header=None)[0].to_numpy()
    return M, f, b


def arm(chrom, start):
    c = str(chrom); return f"chr{c}{'p' if start < CENTROMERE_MB[c] * 1e6 else 'q'}"


def gene_positions(path):
    return pd.read_csv(path).drop_duplicates("gene_id").set_index("gene_id")


def arm_scores(X, ids, gp, ref, min_det, allowed=None, total="all", min_genes=20, window=101):
    """Return the smoothed CNA scores from expression (rows x chromosome arms).

    Args:
        X: Counts (rows x genes, CSR) with gene ids `ids`.
        ref: The reference rows.
        allowed: The genes that may be used (default all).
        total: Normalize the counts per 10^4 against every gene of X ("all") or against the genes used ("used").
    """
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
    """Return the unsmoothed CNA scores from expression (rows x chromosome arms), with the chromosome arms in the order of their first gene."""
    umi = np.asarray(G.sum(1)).ravel().astype(float); det = np.asarray((G > 0).mean(0)).ravel() >= 0.05
    arm_g = np.array([arm(gp.loc[g, "chrom"], gp.loc[g, "start"]) if (g in gp.index and str(gp.loc[g, "chrom"]) in CENTROMERE_MB and d) else ""
                      for g, d in zip(ids, det)])
    Lg = G.multiply(1.0 / np.maximum(umi, 1)[:, None]).tocsc() * 1e4; Lg.data = np.log1p(Lg.data)
    arms = [a for a in pd.unique(arm_g) if a and (arm_g == a).sum() >= 20]
    R = np.column_stack([np.asarray(Lg[:, np.where(arm_g == a)[0]].mean(1)).ravel() for a in arms])
    return pd.DataFrame(R - R[ref].mean(0), columns=arms), umi


def peak_arms(f):
    """Return the chromosome arm of each peak ('' outside chromosomes 1-22) and the chromosome arms with at least 200 peaks."""
    chrom = f[3].astype(str).str.replace("chr", ""); start = f[4].to_numpy(float)
    a = np.array([arm(c, s) if c in CENTROMERE_MB else "" for c, s in zip(chrom, start)])
    return a, [x for x in pd.unique(a) if x and (a == x).sum() >= 200]


def atac_arms(P, f, ref):
    """Return the CNA scores from ATAC (rows x chromosome arms) and the total peak counts of each row, from the peak counts P (rows x peaks) and the
    rows f of the features table for the peaks.
    """
    a, arms = peak_arms(f); P = P.tocsc(); tot = np.asarray(P.sum(1)).ravel().astype(float)
    F = pd.DataFrame(np.column_stack([np.asarray(P[:, np.where(a == x)[0]].sum(1)).ravel() for x in arms]) / np.maximum(tot[:, None], 1.0), columns=arms)
    return np.log2(F.divide(F[ref].mean(0), axis=1).clip(lower=1e-3)), tot


def regress_out(X, covariate):
    """Return X minus its least-squares fit on [1, covariate], with each column then standardized."""
    D = np.column_stack([np.ones(len(covariate)), covariate]); R = X - D @ np.linalg.lstsq(D, X, rcond=None)[0]
    return (R - R.mean(0)) / np.maximum(R.std(0), 1e-12)


def euclidean(Z):
    """Return the Euclidean distances between the rows of Z."""
    q = (Z ** 2).sum(1); return np.sqrt(np.maximum(q[:, None] + q[None, :] - 2.0 * Z @ Z.T, 0.0))


# ------------------------------------------------------------------------------------------------------------------- the copy number tree
def profile_distances(Z):
    """Return the copy number distances between the cancer nuclei, which are the Euclidean distances between their CNA profiles in single precision.
    """
    return euclidean(np.asarray(Z, float)).astype(np.float32).astype(float)


def clone_distances(d, labels):
    """Return the mean distance between the nuclei of each pair of clones (0 on the diagonal), for labels 0 .. K - 1."""
    K = int(labels.max()) + 1
    return np.array([[d[np.ix_(labels == a, labels == b)].mean() if a != b else 0.0 for b in range(K)] for a in range(K)])


def clone_tree(d, labels):
    """Return the tree that SpOTr is given, as distances between the nuclei.

    Within each clone, the distances are the cophenetic distances of a Ward linkage, scaled so that the deepest distance within a clone equals
    the shallowest distance between clones. Between clones, the distances are the mean distances between the clones.
    """
    from scipy.cluster.hierarchy import linkage, cophenet
    from scipy.spatial.distance import squareform
    Dk = clone_distances(d, labels); out = Dk[np.ix_(labels, labels)].astype(float).copy(); off = Dk[~np.eye(len(Dk), dtype=bool)].min()
    for k in range(len(Dk)):
        i = np.where(labels == k)[0]; co = squareform(cophenet(linkage(squareform(d[np.ix_(i, i)], checks=False), "ward")))
        out[np.ix_(i, i)] = co / max(co.max(), 1e-12) * off
    return out


def clone_linkage(d, labels, order):
    """Return the same tree as a scipy linkage for drawing, with a Ward linkage within each clone whose root is at 95% of the shallowest distance
    between clones, and the clones joined by their mean distances. `order` (three clone indices) gives the first two clones to join and then the
    third.
    """
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
