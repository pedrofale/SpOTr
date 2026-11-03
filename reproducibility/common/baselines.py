"""The methods SpOTr is compared with, run from their published packages with their documented settings (environment benchmark_env).

destvi(Xc, labels, Y, seed)        scvi-tools: CondSCVI on the labelled dissociated cells (300 epochs), then DestVI on the spots (1,500 epochs);
                                   each spot's proportion of every label (spots x labels) and the labels.
scdot(Xc, labels, Y, seed)         scDOT from its repository (github.com/namtk/scDOT, commit da74a8a), run as its README documents: Adam at
                                   learning rate 0.1 for 10 iterations on the cosine embedding loss between its transport plan and the
                                   non-negative least squares deconvolution lifted to cells; returns its map (cells x spots).
tangram(Xc, Y, mass, epochs, seed) Tangram's map_cells_to_space in mode "cells", with the spots' masses as its density prior (cells x spots).
exact_ot(Xc, Y, mass)              exact optimal transport (POT's emd) under the cosine distance between log-normalised profiles (cells x spots).

scDOT's inputs are those of its data loader: the cells normalised to 1e5 and log1p, the spots to 1e4 and log1p, each label's mean over its
cells as the marker table, the labels one-hot.  Its net.py imports OptimalTransportLayer from transport.py, which does not define it, and uses
nls_projgrad without importing it; both are supplied from the repository's own ddn.py and deconv.py when net.py is loaded, with no line of it
changed.  Clone the repository to reproducibility/external/scDOT, or point $SPOTR_SCDOT at a clone.
"""
import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np

SCDOT = Path(os.environ.get("SPOTR_SCDOT", Path(__file__).resolve().parents[1] / "external" / "scDOT")) / "src"


def destvi(Xc, labels, Y, seed=0, sc_epochs=300, st_epochs=1500):
    import anndata, scvi, pandas as pd
    scvi.settings.seed = int(seed)
    sc = anndata.AnnData(np.asarray(Xc, np.float32)); sc.obs["label"] = pd.Categorical(np.asarray(labels).astype(str))
    scvi.model.CondSCVI.setup_anndata(sc, labels_key="label"); msc = scvi.model.CondSCVI(sc, weight_obs=False)
    msc.train(max_epochs=sc_epochs, enable_progress_bar=False)
    st = anndata.AnnData(np.asarray(Y, np.float32)); scvi.model.DestVI.setup_anndata(st); mst = scvi.model.DestVI.from_rna_model(st, msc)
    mst.train(max_epochs=st_epochs, enable_progress_bar=False)
    names = list(msc.adata_manager.get_state_registry("labels").categorical_mapping); own = mst.get_proportions()
    return np.asarray(own[names].values if hasattr(own, "values") else own, float), names


def _scdot_net():
    sys.path.insert(0, str(SCDOT))
    import ddn, deconv
    shim = types.ModuleType("transport"); shim.OptimalTransportLayer = ddn.OptimalTransportLayer
    saved = sys.modules.get("transport"); sys.modules["transport"] = shim
    try:
        spec = importlib.util.spec_from_file_location("scdot_net", SCDOT / "net.py"); net = importlib.util.module_from_spec(spec)
        net.nls_projgrad = deconv.nls_projgrad; spec.loader.exec_module(net)
    finally:
        if saved is None:
            sys.modules.pop("transport", None)
        else:
            sys.modules["transport"] = saved
    return net


def scdot(Xc, labels, Y, seed=0, iters=10, lr=0.1):
    import anndata, scanpy as sc, pandas as pd, torch
    torch.manual_seed(int(seed)); net = _scdot_net()
    labels = np.asarray(labels).astype(str); cats = sorted(set(labels.tolist()))
    a_sc = anndata.AnnData(np.asarray(Xc, np.float64)); sc.pp.normalize_total(a_sc, target_sum=1e5); sc.pp.log1p(a_sc)
    a_st = anndata.AnnData(np.asarray(Y, np.float64)); sc.pp.normalize_total(a_st, target_sum=1e4); sc.pp.log1p(a_st)
    a_sc.X, a_st.X = np.asarray(a_sc.X, np.float64), np.asarray(a_st.X, np.float64)
    a_sc.obsm["cell_type"] = pd.DataFrame((labels[:, None] == np.array(cats)[None, :]).astype(np.float64), columns=cats, index=a_sc.obs_names)
    a_st.uns["markers"] = pd.DataFrame(np.stack([a_sc.X[labels == c].mean(0) for c in cats]), index=cats)
    model = net.scDOT(a_sc, a_st); opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(iters):
        NNLS_output, OT_output = model(a_sc, a_st)
        P_true = NNLS_output @ torch.tensor(a_sc.obsm["cell_type"].values.T).float()
        loss = torch.nn.CosineEmbeddingLoss()(OT_output, P_true, torch.ones(P_true.shape[0]))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        _, OT_output = model(a_sc, a_st)
    return np.asarray(OT_output.detach().cpu().numpy(), float).T


def tangram(Xc, Y, mass, epochs=1000, seed=0, device=None):
    import anndata as ad, tangram as tg
    genes = [f"g{i}" for i in range(Xc.shape[1])]
    a_sc = ad.AnnData(np.asarray(Xc, np.float32)); a_sc.var_names = genes; a_sp = ad.AnnData(np.asarray(Y, np.float32)); a_sp.var_names = genes
    tg.pp_adatas(a_sc, a_sp, genes=None); a_sp.obs["rna_count_based_density"] = np.asarray(mass, float) / np.sum(mass)
    device = device or os.environ.get("TANGRAM_DEVICE", "cuda")
    return np.asarray(tg.map_cells_to_space(a_sc, a_sp, mode="cells", density_prior="rna_count_based", lambda_d=1.0, num_epochs=int(epochs),
                                            device=device, random_state=int(seed), verbose=False).X, float)


def lognorm(X):
    X = np.asarray(X, float); return np.log1p(X / np.maximum(X.sum(1, keepdims=True), 1) * 1e4)


def exact_ot(Xc, Y, mass):
    import ot
    mass = np.asarray(mass, float) / np.sum(mass); n = len(Xc)
    return ot.emd(np.full(n, 1.0 / n), mass, ot.dist(lognorm(Xc), lognorm(Y), metric="cosine"), numItermax=10 ** 7) * n
