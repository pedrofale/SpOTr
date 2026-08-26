"""BCSA3 premise re-test using ALLELE-ONLY clone posteriors (p_y_*).

No p_*, no clone_opt, no phi_mle, no exp_post anywhere.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

NB = Path("/mnt/data/pedro/BCSA/BCSA3/numbat_out")
HERE = Path("/home/pedro/projects/spotr/phylospatial_mapping/experiments/bcsa3")
SPATIAL = {"B1": "/mnt/data/pedro/BCSA/BCSA3/V19T26-031_B1/outs/spatial",
           "D1": "/mnt/data/pedro/BCSA/BCSA3/V19T26-031_D1/outs/spatial"}
K = 12
NPERM = 999
rng = np.random.default_rng(0)


def load_coords(d):
    p = Path(d)
    f = p / "tissue_positions.csv"
    if f.exists():
        df = pd.read_csv(f); df.columns = [c.strip() for c in df.columns]
        df = df.set_index("barcode")
        return df[["pxl_row_in_fullres", "pxl_col_in_fullres"]].astype(float)
    df = pd.read_csv(p / "tissue_positions_list.csv", header=None,
                     names=["barcode", "in_tissue", "row", "col", "pxl_row", "pxl_col"]).set_index("barcode")
    return df[["pxl_row", "pxl_col"]].astype(float)


def tree_dist_from_edges(csv):
    e = pd.read_csv(csv)
    nodes = sorted(set(e["from_clone"]) | set(e["to_clone"]))
    idx = {c: i for i, c in enumerate(nodes)}
    D = np.full((len(nodes), len(nodes)), np.inf); np.fill_diagonal(D, 0.0)
    for _, r in e.iterrows():
        i, j = idx[r["from_clone"]], idx[r["to_clone"]]
        D[i, j] = D[j, i] = float(r["n_events"])
    for k in range(len(nodes)):
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    return np.array(nodes), D


def hamming_from_GT(cp):
    """Clone -> set of CNV events from GT_opt; Hamming distance between clones."""
    gt = cp.dropna(subset=["GT_opt"]).groupby("clone_opt")["GT_opt"].agg(lambda s: s.value_counts().index[0])
    clones = sorted(gt.index)
    sets = {c: set(x for x in str(gt[c]).split(",") if x) for c in clones}
    allev = sorted(set().union(*sets.values()))
    M = np.array([[1 if e in sets[c] else 0 for e in allev] for c in clones])
    D = (M[:, None, :] != M[None, :, :]).sum(-1).astype(float)
    return np.array(clones), D


def coherence(labels, nbrs, clones):
    """per-clone: mean over spots of clone c of (frac of k-NN == c) - share(c)."""
    out = {}
    n = len(labels)
    for c in clones:
        m = labels == c
        if m.sum() == 0:
            out[c] = np.nan; continue
        share = m.mean()
        frac = (labels[nbrs[m]] == c).mean()
        out[c] = frac - share
    return out


def mantel(Dt, Ds, nperm=NPERM, rng=rng):
    n = Dt.shape[0]
    iu = np.triu_indices(n, 1)
    a, b = Dt[iu], Ds[iu]
    if a.std() == 0 or b.std() == 0:
        return np.nan, np.nan, None
    r = np.corrcoef(a, b)[0, 1]
    null = np.empty(nperm)
    for i in range(nperm):
        p = rng.permutation(n)
        null[i] = np.corrcoef(Dt[np.ix_(p, p)][iu], b)[0, 1]
    pval = (np.sum(np.abs(null) >= abs(r)) + 1) / (nperm + 1)
    return r, pval, null


cp = pd.read_csv(NB / "clone_post_2.tsv", sep="\t")
meta = pd.read_csv(NB / "cell_metadata.csv")
nb = cp.merge(meta, on="cell")
PY = [f"p_y_{i}" for i in range(1, 13)]

nodes_tree, Dtree_full = tree_dist_from_edges(HERE / "trees" / "bcsa3_B1D1.csv")
nodes_ham, Dham_full = hamming_from_GT(cp)
print("tree nodes", nodes_tree, "\nhamming nodes", nodes_ham)
print("tree D:\n", pd.DataFrame(Dtree_full, index=nodes_tree, columns=nodes_tree).to_string())
print("hamming D:\n", pd.DataFrame(Dham_full, index=nodes_ham, columns=nodes_ham).to_string())

RES = {}
for sec in ["B1", "D1"]:
    v = nb[nb["source"] == f"Visium_{sec}"].copy()
    v["bc"] = v["cell"].str.replace(f"^{sec}_", "", regex=True)
    coords = load_coords(SPATIAL[sec])
    v = v[v["bc"].isin(coords.index)].reset_index(drop=True)
    xy = coords.loc[v["bc"]].to_numpy()

    P = v[PY].to_numpy(float)
    P = P / P.sum(1, keepdims=True)
    lab_allele = np.array([int(PY[i][4:]) for i in P.argmax(1)])
    conf = P.max(1)
    ent = -(np.where(P > 0, P * np.log(P), 0)).sum(1) / np.log(P.shape[1])
    lab_expr = v["clone_opt"].to_numpy()

    print(f"\n{'='*70}\n{sec}: {len(v)} spots")
    print("allele max-posterior confidence: mean %.3f median %.3f" % (conf.mean(), np.median(conf)))
    qs = [0, 5, 10, 25, 50, 75, 90, 95, 100]
    print("  quantiles:", {q: round(float(np.percentile(conf, q)), 3) for q in qs})
    for t in [1/12, 0.2, 0.5, 0.8, 0.9, 0.99]:
        print(f"  frac spots with maxp > {t:.3f}: {(conf > t).mean():.3f}")
    print("  normalised entropy: mean %.3f  (1 = uniform over 12)" % ent.mean())
    print("  frac near-uniform (entropy > 0.95): %.3f" % (ent > 0.95).mean())
    print("  agreement allele-argmax vs expression clone_opt: %.3f" % (lab_allele == lab_expr).mean())
    print("  allele clone counts:", dict(pd.Series(lab_allele).value_counts().sort_index()))
    print("  expr   clone counts:", dict(pd.Series(lab_expr).value_counts().sort_index()))

    nnb = NearestNeighbors(n_neighbors=K + 1).fit(xy)
    nbrs = nnb.kneighbors(xy, return_distance=False)[:, 1:]

    for tag, labels in [("ALLELE", lab_allele), ("EXPR(clone_opt, for contrast)", lab_expr)]:
        clones = np.array(sorted(set(labels)))
        counts = pd.Series(labels).value_counts()
        obs = coherence(labels, nbrs, clones)
        # permutation null
        nullmat = np.empty((NPERM, len(clones)))
        for i in range(NPERM):
            perm = labels[rng.permutation(len(labels))]
            o = coherence(perm, nbrs, clones)
            nullmat[i] = [o[c] for c in clones]
        rows = []
        for j, c in enumerate(clones):
            nl = nullmat[:, j]
            pval = (np.sum(nl >= obs[c]) + 1) / (NPERM + 1)
            rows.append(dict(clone=c, n_spots=int(counts[c]), share=float((labels == c).mean()),
                             coherence=obs[c], null_mean=float(nl.mean()), null_sd=float(nl.std()),
                             p_emp=pval))
        df = pd.DataFrame(rows)
        big = df[df.n_spots >= 20]
        print(f"\n-- {sec} coherence [{tag}] --")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"  mean coherence over clones with >=20 spots (n={len(big)}): {big.coherence.mean():.4f}")
        print(f"  spot-weighted mean: {(big.coherence*big.n_spots).sum()/big.n_spots.sum():.4f}")
        print(f"  clones with p<0.05: {int((big.p_emp<0.05).sum())}/{len(big)}")
        if tag == "ALLELE":
            RES[f"{sec}_coh"] = df

    # ---- Mantel: tree vs spatial centroid, ALLELE labels only
    labels = lab_allele
    counts = pd.Series(labels).value_counts()
    for dname, nodes_d, D_d in [("mut_graph n_events", nodes_tree, Dtree_full),
                                ("GT_opt Hamming", nodes_ham, Dham_full)]:
        pos = {c: i for i, c in enumerate(nodes_d)}
        cl = [c for c in sorted(set(labels)) if counts[c] >= 20 and c in pos]
        if len(cl) < 4:
            print(f"  Mantel [{dname}]: too few clones ({len(cl)})"); continue
        cent = np.stack([xy[labels == c].mean(0) for c in cl])
        Ds = np.linalg.norm(cent[:, None] - cent[None, :], axis=-1)
        Dt = D_d[np.ix_([pos[c] for c in cl], [pos[c] for c in cl])]
        r, p, _ = mantel(Dt, Ds)
        print(f"\n-- {sec} Mantel [{dname}] on {len(cl)} clones {cl}: r = {r:+.4f}, p = {p:.4f}")
        RES[f"{sec}_mantel_{dname}"] = dict(r=float(r), p=float(p), clones=[int(c) for c in cl])
        RES[f"{sec}_cent_{dname}"] = dict(cl=[int(c) for c in cl], Dt=Dt.tolist(), Ds=Ds.tolist())

np.save("/tmp/claude-1005/-home-pedro-projects/fd7cdd18-6dfc-4868-9c2a-fc644da4f924/scratchpad/res.npy", RES, allow_pickle=True)
