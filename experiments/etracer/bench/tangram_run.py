"""
Tangram adapter for the eTRACER unmatched clone-mapping benchmark.

Method
------
Tangram (Biancalani et al., Nature Methods 2021), `tangram-sc==1.0.4`.

Tangram learns a cell-by-spot mapping matrix M (softmax-normalised over spots,
so each cell's row sums to 1) purely from expression, by maximising the cosine
similarity between the spatial matrix G and the mapping-predicted spatial
matrix M^T S, plus a KL density term.  We then transfer the clone label the
canonical way (`tg.project_cell_annotations`, i.e. M^T @ onehot(clone)) and
row-normalise to get per-bin clone proportions.

Information used
----------------
X_sc, X_st, cell_clone only.  Tangram is spatially agnostic: `xy` is accepted
for interface compatibility and never used.  C_tree is never touched, and no
spatial clone label is used anywhere.

Notes / deviations
------------------
* tangram 1.0.4 has a bug in `map_cells_to_space`: for a dense ndarray
  `adata_sc.X` it still calls `.toarray()`.  We therefore hand it CSR matrices,
  which take the (correct) sparse branch.  This does not change the numerics.
* `Mapper.__init__` seeds with `if self.random_state:` which is falsy for
  seed=0, so we additionally seed numpy/torch ourselves before the call.  With
  that, seed=0 is reproducible like any other seed.
* Defaults are stock Tangram: mode='cells', 1000 epochs, lr=0.1,
  density_prior='rna_count_based', lambda_g1=1, all other lambdas 0.
"""

import logging
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = ["run"]


def _quiet():
    # tangram sets the root logger to INFO at import time
    logging.getLogger().setLevel(logging.WARNING)


def _as_adata(X, prefix, gene_names):
    import anndata as ad

    obs = pd.DataFrame(index=[f"{prefix}{i}" for i in range(X.shape[0])])
    var = pd.DataFrame(index=list(gene_names))
    # CSR on purpose: see module docstring (tangram 1.0.4 dense-branch bug)
    return ad.AnnData(X=sp.csr_matrix(np.asarray(X, dtype=np.float32)), obs=obs, var=var)


def run(
    X_sc,
    X_st,
    cell_clone,
    clones,
    xy=None,
    seed=0,
    *,
    device=None,
    num_epochs=1000,
    learning_rate=0.1,
    density_prior="rna_count_based",
    verbose=False,
):
    """Map cells to spatial bins with Tangram and aggregate to clone proportions.

    Parameters
    ----------
    X_sc : (n_cells, n_genes) log-normalised scRNA-seq.
    X_st : (n_bins,  n_genes) log-normalised spatial bins, same gene space.
    cell_clone : (n_cells,) str, clone id of each cell.
    clones : (n_clones,) str, clone ids; defines the column order of the output.
    xy : ignored (Tangram is spatially agnostic); accepted for interface parity.
    seed : int.

    Returns
    -------
    (n_bins, n_clones) float64 array, rows sum to 1.
    """
    import torch
    import tangram as tg

    _quiet()

    X_sc = np.asarray(X_sc, dtype=np.float32)
    X_st = np.asarray(X_st, dtype=np.float32)
    cell_clone = np.asarray(cell_clone).astype(str)
    clones = list(np.asarray(clones).astype(str))

    n_cells, n_genes = X_sc.shape
    n_bins = X_st.shape[0]
    if X_st.shape[1] != n_genes:
        raise ValueError(f"gene space mismatch: {X_sc.shape} vs {X_st.shape}")
    if cell_clone.shape[0] != n_cells:
        raise ValueError("cell_clone length != n_cells")

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    gene_names = [f"g{i:05d}" for i in range(n_genes)]  # already lowercase
    ad_sc = _as_adata(X_sc, "cell", gene_names)
    ad_sp = _as_adata(X_st, "bin", gene_names)
    ad_sc.obs["clone"] = pd.Series(cell_clone, index=ad_sc.obs_names, dtype=object)

    # gene filtering + training-gene bookkeeping + density priors
    tg.pp_adatas(ad_sc, ad_sp, genes=None)

    # seed before the call: Mapper's own `if self.random_state:` skips seed=0
    np.random.seed(seed)
    torch.manual_seed(seed)

    ad_map = tg.map_cells_to_space(
        ad_sc,
        ad_sp,
        mode="cells",
        device=device,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        density_prior=density_prior,
        random_state=seed,
        verbose=verbose,
    )

    # canonical Tangram label transfer: M^T @ onehot(clone) -> (n_bins, n_clones)
    tg.project_cell_annotations(ad_map, ad_sp, annotation="clone")
    df = ad_sp.obsm["tangram_ct_pred"]
    df = df.reindex(columns=clones, fill_value=0.0)  # clone order == `clones`

    P = np.asarray(df.values, dtype=np.float64)
    P = np.clip(P, 0.0, None)
    row = P.sum(axis=1, keepdims=True)
    dead = (row[:, 0] <= 0) | ~np.isfinite(row[:, 0])
    P[dead] = 1.0 / len(clones)
    row[dead] = 1.0
    P = P / row

    assert P.shape == (n_bins, len(clones)), P.shape
    return P


if __name__ == "__main__":
    NPZ = (
        "/home/pedro/projects/spotr/phylospatial_mapping/experiments/etracer/"
        "etracer_LE_full.npz"
    )
    d = np.load(NPZ, allow_pickle=True)

    t0 = time.time()
    P = run(
        d["X_sc"],
        d["X_st"],
        d["cell_clone"],
        d["clones"],
        xy=d["xy"],
        seed=0,
        verbose=True,
    )
    wall = time.time() - t0

    clones = list(np.asarray(d["clones"]).astype(str))
    bin_clone = np.asarray(d["bin_clone"]).astype(str)
    lab = np.asarray(d["is_labelled"], dtype=bool)

    idx = {c: i for i, c in enumerate(clones)}
    truth = np.array([idx.get(c, -1) for c in bin_clone])
    ok = lab & (truth >= 0)
    acc = float((P[ok].argmax(axis=1) == truth[ok]).mean())

    print(f"\nwall_time_s      {wall:.1f}")
    print(f"output_shape     {P.shape}")
    print(f"rowsum min/max   {P.sum(1).min():.6f} {P.sum(1).max():.6f}")
    print(f"barcoded_bins    {int(ok.sum())}")
    print(f"argmax_accuracy  {acc:.4f}")
    print(f"n_distinct_argmax {len(np.unique(P.argmax(1)))}")
