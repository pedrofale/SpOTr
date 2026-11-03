"""The competing methods as their authors ship them, shared by the simulation benchmark and the real-data figures (decision-paper-results-structure:
"NO REIMPLEMENTATIONS -- THEIR ACTUAL PACKAGES").

destvi(Xc, labels, Y, seed)   scvi-tools: CondSCVI on the labelled dissociated cells (300 epochs), then DestVI on the spots (1,500 epochs), as the
                              Cassiopeia workflow's fit_destvi.py; returns each spot's proportion of every label (spots x labels) and the labels.
scdot(Xc, labels, Y, seed)    scDOT from its upstream repository unmodified (github.com/namtk/scDOT at da74a8a, cloned to benchmark/external/
                              scDOT_upstream), run as its README documents: scDOT(sc_adata, st_adata), Adam at lr 0.1 for 10 iterations on the
                              cosine embedding loss between its OT plan and the NNLS deconvolution lifted to cells.  Returns scDOT's own map
                              (cells x spots, each cell's distribution over the spots), NOT the broadcast deconvolution the vendored copy returned
                              (related-work/rw-scdot).
  Inputs as upstream's data loader prepares them: the cells normalised to 1e5 and log1p, the spots to 1e4 and log1p; the marker table st.uns
  ["markers"] is each label's mean over its cells (labels x genes, as nls_projgrad(markers.T, st.T) requires); obsm["cell_type"] the one-hot
  labels.  Upstream as released does not import: net.py takes OptimalTransportLayer from transport.py, which does not define it (ddn.py does),
  and calls nls_projgrad without importing it (deconv.py defines it).  We supply exactly those two names from upstream's own files when loading
  net.py and change no line of it.
"""
import sys, types, importlib.util
from pathlib import Path
import numpy as np

UPSTREAM = Path(__file__).resolve().parent / "external" / "scDOT_upstream" / "src"


def destvi(Xc, labels, Y, seed=0, sc_epochs=300, st_epochs=1500):
    import anndata, scvi, pandas as pd
    scvi.settings.seed = int(seed)
    sc = anndata.AnnData(np.asarray(Xc, np.float32)); sc.obs["label"] = pd.Categorical(np.asarray(labels).astype(str))
    scvi.model.CondSCVI.setup_anndata(sc, labels_key="label"); msc = scvi.model.CondSCVI(sc, weight_obs=False); msc.train(max_epochs=sc_epochs, enable_progress_bar=False)
    st = anndata.AnnData(np.asarray(Y, np.float32)); scvi.model.DestVI.setup_anndata(st); mst = scvi.model.DestVI.from_rna_model(st, msc)
    mst.train(max_epochs=st_epochs, enable_progress_bar=False)
    names = list(msc.adata_manager.get_state_registry("labels").categorical_mapping); own = mst.get_proportions()
    return np.asarray(own[names].values if hasattr(own, "values") else own, float), names


def _net():
    """upstream net.py, with the two names it uses but does not import supplied from upstream's ddn.py and deconv.py"""
    sys.path.insert(0, str(UPSTREAM))
    import ddn, deconv
    shim = types.ModuleType("transport"); shim.OptimalTransportLayer = ddn.OptimalTransportLayer
    saved = sys.modules.get("transport"); sys.modules["transport"] = shim
    try:
        spec = importlib.util.spec_from_file_location("scdot_net", UPSTREAM / "net.py"); net = importlib.util.module_from_spec(spec)
        net.nls_projgrad = deconv.nls_projgrad; spec.loader.exec_module(net)
    finally:
        if saved is None:
            sys.modules.pop("transport", None)
        else:
            sys.modules["transport"] = saved
    return net


def scdot(Xc, labels, Y, seed=0, iters=10, lr=0.1):
    import anndata, scanpy as sc, pandas as pd, torch
    torch.manual_seed(int(seed)); net = _net()
    labels = np.asarray(labels).astype(str); cats = sorted(set(labels.tolist()))
    a_sc = anndata.AnnData(np.asarray(Xc, np.float64)); sc.pp.normalize_total(a_sc, target_sum=1e5); sc.pp.log1p(a_sc)
    a_st = anndata.AnnData(np.asarray(Y, np.float64)); sc.pp.normalize_total(a_st, target_sum=1e4); sc.pp.log1p(a_st)
    a_sc.X, a_st.X = np.asarray(a_sc.X, np.float64), np.asarray(a_st.X, np.float64)
    onehot = pd.DataFrame((labels[:, None] == np.array(cats)[None, :]).astype(np.float64), columns=cats, index=a_sc.obs_names)
    a_sc.obsm["cell_type"] = onehot
    a_st.uns["markers"] = pd.DataFrame(np.stack([a_sc.X[labels == c].mean(0) for c in cats]), index=cats)
    model = net.scDOT(a_sc, a_st); opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(iters):                                                                           # README's training loop
        NNLS_output, OT_output = model(a_sc, a_st)
        ct = torch.tensor(a_sc.obsm["cell_type"].values.T)
        P_true = NNLS_output @ ct.float()
        loss = torch.nn.CosineEmbeddingLoss()(OT_output, P_true, torch.ones(P_true.shape[0]))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        _, OT_output = model(a_sc, a_st)
    return np.asarray(OT_output.detach().cpu().numpy(), float).T                                       # cells x spots, each cell's column summing to 1
