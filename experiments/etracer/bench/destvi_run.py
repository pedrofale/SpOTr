"""DestVI adapter for the eTRACER unmatched clone-mapping benchmark.

Common interface:

    run(X_sc, X_st, cell_clone, clones, xy=None, seed=0) -> [n_bins, n_clones]

DestVI is a deconvolution method: it learns a conditional generative model of the
single-cell data (CondSCVI) with a discrete label per cell, then fits a spatial model
in which every spot is a mixture over those labels.  Here the label is the CLONE, so
DestVI's per-spot cell-type proportions are directly the per-bin clone composition
the benchmark asks for.

It uses only X_sc, X_st and cell_clone.  It never sees C_tree, xy, or any spatial
clone label (xy is accepted and ignored, to satisfy the shared signature).

COUNTS.  DestVI/CondSCVI are negative-binomial models and need integer counts, but the
benchmark hands over log-normalised matrices.  The normalisation is invertible here:
for a row that was `log1p(count * s_i)`, expm1 gives `count * s_i`, and the smallest
non-zero entry of the row is s_i itself (some gene had exactly one count).  Dividing
by it recovers the integer counts exactly -- verified on this dataset, max deviation
from an integer is 0.0 for every row of both matrices.  No information is invented.
"""

from __future__ import annotations

import numpy as np

__all__ = ["run", "recover_counts"]


def recover_counts(X_log: np.ndarray) -> np.ndarray:
    """Invert log1p(count * per-row-scale) back to integer counts."""
    E = np.expm1(np.asarray(X_log, dtype=np.float64))
    E[E < 0] = 0.0
    out = np.zeros_like(E)
    for i in range(E.shape[0]):
        row = E[i]
        nz = row > 0
        if not nz.any():
            continue
        out[i] = row / row[nz].min()
    return np.rint(out).astype(np.float32)


def run(X_sc, X_st, cell_clone, clones, xy=None, seed=0,
        sc_epochs=300, st_epochs=2500, batch_size=128):
    import anndata
    import pandas as pd
    import scvi
    import torch
    from scvi.model import CondSCVI, DestVI

    scvi.settings.seed = int(seed)
    torch.manual_seed(int(seed))

    clones = np.asarray(clones).astype(str)
    cell_clone = np.asarray(cell_clone).astype(str)

    # Only clones that actually have cells can be modelled.  A label with zero
    # cells makes DestVI's VampPrior normalise 0/0 and the entire fit goes NaN,
    # so such clones are dropped here and simply left at proportion 0 in the
    # output.  On the full eTRACER data every clone has >=12 cells, so this
    # drops nothing and `present` is `clones` in the same order.
    present = [c for c in clones if (cell_clone == c).any()]

    # ---- single cell ----------------------------------------------------
    sc_counts = recover_counts(X_sc)
    sc_adata = anndata.AnnData(
        X=sc_counts,
        obs=pd.DataFrame(
            {"clone": pd.Categorical(cell_clone, categories=present)},
            index=[f"cell{i}" for i in range(sc_counts.shape[0])],
        ),
        var=pd.DataFrame(index=[f"g{j}" for j in range(sc_counts.shape[1])]),
    )
    sc_adata.layers["counts"] = sc_adata.X.copy()

    CondSCVI.setup_anndata(sc_adata, layer="counts", labels_key="clone")
    sc_model = CondSCVI(sc_adata, weight_obs=False)
    sc_model.train(max_epochs=sc_epochs, batch_size=batch_size,
                   enable_progress_bar=False)

    # ---- spatial --------------------------------------------------------
    st_counts = recover_counts(X_st)
    st_adata = anndata.AnnData(
        X=st_counts,
        obs=pd.DataFrame(index=[f"bin{i}" for i in range(st_counts.shape[0])]),
        var=pd.DataFrame(index=sc_adata.var_names.copy()),
    )
    st_adata.layers["counts"] = st_adata.X.copy()

    DestVI.setup_anndata(st_adata, layer="counts")
    st_model = DestVI.from_rna_model(st_adata, sc_model)
    st_model.train(max_epochs=st_epochs, batch_size=batch_size,
                   enable_progress_bar=False)

    prop = st_model.get_proportions()          # DataFrame [n_bins, n_labels]

    # ---- align to the benchmark's clone order ---------------------------
    P = np.zeros((st_counts.shape[0], len(clones)), dtype=np.float64)
    col = {str(c): k for k, c in enumerate(prop.columns)}
    for k, c in enumerate(clones):
        if c in col:
            P[:, k] = prop.values[:, col[c]]

    P = np.clip(P, 0.0, None)
    rs = P.sum(1, keepdims=True)
    bad = (rs[:, 0] <= 0) | ~np.isfinite(rs[:, 0])
    P[bad] = 1.0 / len(clones)
    rs = P.sum(1, keepdims=True)
    return (P / rs).astype(np.float64)


if __name__ == "__main__":  # standalone verification on the real npz
    import sys, time, collections
    npz = sys.argv[1] if len(sys.argv) > 1 else (
        "/home/pedro/projects/spotr/phylospatial_mapping/experiments/etracer/etracer_LE_full.npz")
    d = np.load(npz, allow_pickle=True)
    t0 = time.time()
    P = run(d["X_sc"], d["X_st"], d["cell_clone"], d["clones"], xy=d["xy"], seed=0)
    dt = time.time() - t0
    clones = np.asarray(d["clones"]).astype(str)
    lab = d["is_labelled"]
    truth = np.asarray(d["bin_clone"]).astype(str)[lab]
    pred = clones[P[lab].argmax(1)]
    print("shape", P.shape, "rowsum min/max %.6f %.6f" % (P.sum(1).min(), P.sum(1).max()),
          "nan", int(np.isnan(P).sum()))
    print("wall_time_s %.1f" % dt)
    print("barcoded-bin argmax accuracy %.4f  (n=%d)" % ((pred == truth).mean(), lab.sum()))
    maj = collections.Counter(truth).most_common(1)[0]
    print("majority-clone baseline %.4f (%s)" % (maj[1] / lab.sum(), maj[0]))
    print("n distinct predicted clones over all bins",
          len(np.unique(clones[P.argmax(1)])))
    np.save("/home/pedro/projects/spotr/phylospatial_mapping/experiments/etracer/bench/destvi_P.npy", P)
