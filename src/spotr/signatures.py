"""Reference signatures for the deconvolution of SpOTr, which are the expected counts per cell of each label over the genes.

SpOTr accepts three kinds of signatures. "mean" (the default) uses the mean raw counts per cell of each label and needs only numpy.
"cell2location" uses the regression model of the cell2location package, as in the paper, and needs `pip install cell2location`. A pandas
DataFrame (genes x labels) provides signatures computed elsewhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mean_signatures(X: np.ndarray, labels: np.ndarray, genes) -> pd.DataFrame:
    """Return the mean counts per cell of each label (genes x labels)."""
    labels = np.asarray(labels).astype(str); cats = sorted(set(labels.tolist()))
    return pd.DataFrame(np.stack([np.asarray(X[labels == c].mean(0)).ravel() for c in cats], axis=1), index=list(genes), columns=cats)


def cell2location_signatures(X: np.ndarray, labels: np.ndarray, genes, max_epochs: int = 250) -> pd.DataFrame:
    """Return the signatures of the RegressionModel of cell2location, trained as in its tutorials (genes x labels)."""
    import anndata as ad
    from cell2location.models import RegressionModel
    a = ad.AnnData(np.asarray(X, np.float32)); a.var_names = [str(g) for g in genes]; a.obs["label"] = pd.Categorical(np.asarray(labels).astype(str))
    RegressionModel.setup_anndata(adata=a, labels_key="label"); m = RegressionModel(a); m.train(max_epochs=max_epochs)
    a = m.export_posterior(a, sample_kwargs={"num_samples": 1000, "batch_size": 2500})
    cols = [f"means_per_cluster_mu_fg_{c}" for c in a.uns["mod"]["factor_names"]]
    out = a.varm["means_per_cluster_mu_fg"][cols].copy(); out.columns = list(a.uns["mod"]["factor_names"]); return out


def signatures(X, labels, genes, method="mean") -> pd.DataFrame:
    """Return the signatures (genes x labels) for `method` "mean" or "cell2location", or the DataFrame `method` restricted to `genes`."""
    if isinstance(method, pd.DataFrame):
        missing = sorted(set(np.asarray(labels).astype(str)) - set(map(str, method.columns)))
        if missing:
            raise ValueError(f"the signatures lack the labels {missing}")
        return method.loc[list(genes)]
    if method == "mean":
        return mean_signatures(X, labels, genes)
    if method == "cell2location":
        return cell2location_signatures(X, labels, genes)
    raise ValueError('signatures must be "mean", "cell2location" or a DataFrame (genes x labels)')
