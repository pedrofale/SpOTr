"""Tests of the quick start of the README on the toy tumor. We check that SpOTr runs with and without the tree, that its outputs are valid, and that
the proportions of the clones in the spots are more accurate with the tree.
"""
import numpy as np
import pytest

import spotr


@pytest.fixture(scope="module")
def toy():
    sc_adata, st_adata, tree, truth = spotr.datasets.toy_tumor(seed=0)
    spotr.SpOTr.setup_anndata(sc_adata, labels_key="cell_type", clone_key="clone", tree=tree)
    return sc_adata, st_adata, truth


def per_clone_r(props, truth):
    """Return the mean over clones of the Pearson correlation between the true and estimated proportions of each clone among the tumor cells, over
    the spots that contain tumor cells (as in notebooks/spotr_toy_example.ipynb).
    """
    T = truth["clone_counts"]; clones = [c for c in T.columns if c.startswith("clone")]
    true = T[clones].div(T[clones + ["smaller clones"]].sum(axis=1).clip(lower=1), axis=0)
    est = props[clones].div(props[clones].sum(axis=1).clip(lower=1e-12), axis=0)
    h = T[clones].sum(axis=1) > 0
    return float(np.mean([np.corrcoef(true.loc[h, c], est.loc[h, c])[0, 1] for c in clones]))


def test_toy_tumor(toy):
    sc_adata, st_adata, truth = toy
    r = {}
    for alpha in (0.2, 0.0):
        model = spotr.SpOTr(sc_adata, st_adata).train(alpha=alpha)
        P, G = model.get_proportions(), model.get_mapping()
        assert P.shape[0] == st_adata.n_obs and np.isfinite(P.to_numpy()).all() and (P.to_numpy() >= 0).all()
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)
        assert G.shape[1] == st_adata.n_obs and np.isfinite(G.to_numpy()).all() and (G.to_numpy() >= 0).all()
        np.testing.assert_allclose(G.sum(axis=1), 1.0, atol=1e-4)
        assert np.isfinite(sc_adata.obsm["spotr_spatial"][sc_adata.obs_names.isin(G.index)]).all()
        r[alpha] = per_clone_r(P, truth)
    assert r[0.2] > r[0.0], r
