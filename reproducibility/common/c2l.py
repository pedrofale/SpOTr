"""cell2location, the package, with its documented workflow. The models run in their own environment (c2l). The functions below write their inputs
to a file, run this script in that environment, and read its outputs back.

    signatures(X, labels, path)     Reference signatures from RegressionModel on the labeled dissociated cells (250 epochs), on every gene given.
                                    Returns the posterior mean expression of each label (means_per_cluster_mu_fg), labels x genes. SpOTr uses
                                    them in its deconvolution on every gene it fits.
    deconvolve(Xc, labels, Y, n_cells, path, epochs)
                                    cell2location as a method of its own, with genes from filter_genes (cell_count_cutoff 5,
                                    cell_percentage_cutoff2 0.03, nonz_mean_cutoff 1.12), signatures from RegressionModel (250 epochs), and
                                    Cell2location with N_cells_per_location n_cells and detection_alpha 20, trained on the whole data for
                                    `epochs` epochs. Returns the posterior mean abundance of each label in each spot (means_cell_abundance_w_sf),
                                    spots x labels.

Both functions cache their outputs, and read an existing output file instead of fitting again.

Usage (in the c2l environment): python c2l.py signatures|deconvolve <in.npz> <out.npz>
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def _run(stage, inputs, path):
    path = Path(path)
    if not path.exists():
        sys.path.insert(0, str(Path(__file__).resolve().parent)); from paths import c2l_python
        path.parent.mkdir(parents=True, exist_ok=True); f_in = path.with_name(path.stem + "_in.npz")
        np.savez_compressed(f_in, **inputs)
        subprocess.run(c2l_python() + [str(Path(__file__).resolve()), stage, str(f_in), str(path)], check=True, env=dict(os.environ, PYTHONNOUSERSITE="1"))
        os.remove(f_in)
    return np.load(path, allow_pickle=True)


def signatures(X, labels, path):
    """Return {label: signature over the genes of X}."""
    z = _run("signatures", dict(X=np.asarray(X, np.float32), labels=np.asarray(labels).astype(str)), path)
    return dict(zip([str(l) for l in z["labels"]], np.asarray(z["signatures"], float)))


def deconvolve(Xc, labels, Y, n_cells, path, epochs=30000):
    """Return the abundances (spots x labels) and the labels."""
    z = _run("deconvolve", dict(Xc=np.asarray(Xc, np.float32), labels=np.asarray(labels).astype(str), Y=np.asarray(Y, np.float32),
                                n_cells=float(n_cells), epochs=int(epochs)), path)
    return np.asarray(z["w"], float), [str(t) for t in z["types"]]


def _reference(X, labels, genes):
    import anndata as ad, pandas as pd
    return ad.AnnData(X=np.asarray(X, np.float32), obs=pd.DataFrame({"label": labels}, index=[f"cell{i}" for i in range(len(labels))]),
                      var=pd.DataFrame(index=genes))


def _regression(ref):
    from cell2location.models import RegressionModel
    RegressionModel.setup_anndata(adata=ref, labels_key="label"); rm = RegressionModel(ref); rm.train(max_epochs=250)
    ref = rm.export_posterior(ref, sample_kwargs={"num_samples": 1000, "batch_size": 2500}); names = list(ref.uns["mod"]["factor_names"])
    S = ref.varm["means_per_cluster_mu_fg"][[f"means_per_cluster_mu_fg_{c}" for c in names]].copy(); S.columns = names
    return S


def _main(stage, f_in, f_out):
    import warnings; warnings.filterwarnings("ignore")
    z = np.load(f_in, allow_pickle=True); labels = [str(l) for l in z["labels"]]
    if stage == "signatures":
        X = np.asarray(z["X"], np.float32); S = _regression(_reference(X, labels, [f"g{j}" for j in range(X.shape[1])]))
        np.savez_compressed(f_out, signatures=S.to_numpy().T.astype(np.float32), labels=np.array(S.columns))
        return
    import anndata as ad, pandas as pd, cell2location
    from cell2location.models import Cell2location
    from cell2location.utils.filtering import filter_genes
    Xc = np.asarray(z["Xc"], np.float32); genes = [f"g{j}" for j in range(Xc.shape[1])]; ref = _reference(Xc, labels, genes)
    ref = ref[:, filter_genes(ref, cell_count_cutoff=5, cell_percentage_cutoff2=0.03, nonz_mean_cutoff=1.12)].copy(); S = _regression(ref)
    vis = ad.AnnData(X=np.asarray(z["Y"], np.float32), obs=pd.DataFrame(index=[f"spot{i}" for i in range(len(z["Y"]))]), var=pd.DataFrame(index=genes))
    shared = np.intersect1d(vis.var_names, S.index); vis = vis[:, shared].copy(); S = S.loc[shared, :]
    Cell2location.setup_anndata(adata=vis); m = Cell2location(vis, cell_state_df=S, N_cells_per_location=float(z["n_cells"]), detection_alpha=20)
    m.train(max_epochs=int(z["epochs"]), batch_size=None, train_size=1)
    vis = m.export_posterior(vis, sample_kwargs={"num_samples": 1000, "batch_size": m.adata.n_obs}); W = vis.obsm["means_cell_abundance_w_sf"]
    np.savez_compressed(f_out, w=W.to_numpy(np.float32), types=np.array([c.replace("meanscell_abundance_w_sf_", "") for c in W.columns]),
                        version=cell2location.__version__)


if __name__ == "__main__":
    _main(*sys.argv[1:4])
