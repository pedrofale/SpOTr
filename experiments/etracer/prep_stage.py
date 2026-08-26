"""Build SpOTr inputs for eTRACER late-escape: cells (with lineage tree) -> Stereo-seq bins."""
from pathlib import Path
import numpy as np, pandas as pd, scipy.io as sio
import scanpy as sc
from ete3 import Tree
from scipy.spatial.distance import pdist, squareform

HERE = Path(__file__).resolve().parent
B = HERE / "Tumor_Immune_Evasion_eTRACER" / "LineageTracing"
N_HVG = 2000


def load(d):
    X = sio.mmread(d / "matrix.mtx").T.tocsr()               # obs x genes
    g = [l.strip() for l in open(d / "features.tsv")]
    b = [l.strip() for l in open(d / "barcodes.tsv")]
    return X, np.array(g), np.array(b)


def cophenetic_bl(t, leaves):
    """Pairwise distance with branch lengths, over the given leaves only."""
    idx = {l: i for i, l in enumerate(leaves)}
    D = np.zeros((len(leaves), len(leaves)))
    for node in t.traverse("postorder"):
        if node.is_leaf():
            node._acc = [(idx[node.name], 0.0)] if node.name in idx else []
            continue
        kids = [c._acc for c in node.children]
        blen = [max(c.dist, 0.0) for c in node.children]
        for a in range(len(kids)):
            for b_ in range(a + 1, len(kids)):
                for i, di in kids[a]:
                    for j, dj in kids[b_]:
                        d = di + blen[a] + dj + blen[b_]
                        D[i, j] = D[j, i] = d
        node._acc = [(i, d + blen[k]) for k, kk in enumerate(kids) for i, d in kk]
    return D


def main(stage="LE"):
    cells = pd.read_csv(HERE / f"sel_cells_{stage}.csv")
    bins_ = pd.read_csv(HERE / f"sel_bins_{stage}.csv")

    Xsc, gsc, bsc = load(HERE / f"export/sc_sub_{stage}")
    Xst, gst, bst = load(HERE / f"export/spatial_sub_{stage}")
    cells = cells.set_index("matrix_id").loc[bsc].reset_index()
    bins_ = bins_.set_index("matrix_id").loc[[int(x) for x in bst]].reset_index()
    print(f"[prep] {Xsc.shape[0]} cells, {Xst.shape[0]} bins")

    shared = sorted(set(gsc) & set(gst))
    asc = sc.AnnData(Xsc[:, [list(gsc).index(g) for g in shared]])
    ast = sc.AnnData(Xst[:, [list(gst).index(g) for g in shared]])
    asc.var_names = ast.var_names = shared
    for a in (asc, ast):
        sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    sc.pp.highly_variable_genes(ast, n_top_genes=min(N_HVG, len(shared)))
    hvg = ast.var_names[ast.var.highly_variable]
    asc, ast = asc[:, hvg].copy(), ast[:, hvg].copy()
    print(f"[prep] shared genes {len(shared)} -> HVG {len(hvg)}")

    # ---- tree distances: block-diagonal, one block per clone ----
    n = len(cells)
    D = np.zeros((n, n), dtype=np.float32)
    pos = {cb: i for i, cb in enumerate(cells.cellBC)}
    within_max = 0.0
    for clone, g in cells.groupby("clone"):
        tf = sorted((B / f"single_cell/sc_trees_{stage}").glob(f"{clone}-*.phy.treefile"))[0]
        t = Tree(str(tf), format=1)
        leaves = [c for c in g.cellBC if c in {l.name for l in t.iter_leaves()}]
        sub = cophenetic_bl(t, leaves)
        ii = [pos[c] for c in leaves]
        D[np.ix_(ii, ii)] = sub
        within_max = max(within_max, sub.max())
    # clones are independent lineages -> no shared ancestry; set cross-clone to a constant
    same = (cells.clone.to_numpy()[:, None] == cells.clone.to_numpy()[None, :])
    D[~same] = 2.0 * within_max
    np.fill_diagonal(D, 0.0)
    print(f"[prep] tree distance: within-clone max {within_max:.3f}, cross-clone {2*within_max:.3f}")

    xy = bins_[["x", "y"]].to_numpy(float)
    C_space = squareform(pdist(xy)).astype(np.float32)

    clones = sorted(cells.clone.unique(), key=lambda c: int(c[1:]))
    A = np.zeros((n, len(clones)), dtype=np.float32)
    for i, c in enumerate(cells.clone):
        A[i, clones.index(c)] = 1.0

    np.savez_compressed(
        HERE / f"etracer_inputs_{stage}.npz",
        X_sc=np.asarray(asc.X.todense() if hasattr(asc.X, "todense") else asc.X, dtype=np.float32),
        X_st=np.asarray(ast.X.todense() if hasattr(ast.X, "todense") else ast.X, dtype=np.float32),
        C_tree=D, C_space=C_space, xy=xy, A=A,
        clones=np.array(clones), cell_clone=cells.clone.to_numpy(),
        bin_clone=bins_.lineageGrp.to_numpy(),
    )
    print(f"[prep] wrote etracer_inputs_{stage}.npz")
    print(f"[prep] bin clone composition: {pd.Series(bins_.lineageGrp).value_counts().head(5).to_dict()}")


if __name__ == "__main__":
    import sys; main(sys.argv[1] if len(sys.argv)>1 else "LE")
