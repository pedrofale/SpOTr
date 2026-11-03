"""cell2location, the package, on one simulated problem (run in the c2l environment with PYTHONNOUSERSITE=1, GPU).

The documented workflow, as experiments/night/355_c2l_package.py: reference = the dissociated cells, labelled by clone (the clones of the tree
given to SpOTr; pooled traced cells and untraced tumour cells as one type each) and microenvironment type; genes by filter_genes; signatures by
RegressionModel (250 epochs); Cell2location (N_cells_per_location = the mean number of cells per spot, detection_alpha 20); the posterior
mean abundance of each type.

Usage: python fit_c2l.py <params.json> <problem.npz> <trees.npz> <epochs> <out.npz>
"""
import sys, json
import numpy as np


def main(pfile, prob, trees, epochs, out):
    import anndata as ad, pandas as pd, cell2location
    from cell2location.models import RegressionModel, Cell2location
    from cell2location.utils.filtering import filter_genes
    p = json.load(open(pfile)); q, t = np.load(prob), np.load(trees); rec = p["tree"] == "reconstructed"
    clone = (t["clone_rec_tree"] if rec else t["clone_true_tree"]).astype(int); kind, me, tr = q["kind"], q["me_type"], q["traced"]
    labs = np.array([f"me{e}" for e in me], dtype=object); labs[(kind == 0) & ~tr] = "untraced"; labs[tr] = [f"c{c}" if c >= 0 else "pooled" for c in clone]
    genes = [f"g{j}" for j in range(q["Xc"].shape[1])]
    ref = ad.AnnData(X=q["Xc"].astype(np.float32), obs=pd.DataFrame({"label": labs.astype(str)}, index=[f"cell{i}" for i in range(len(labs))]), var=pd.DataFrame(index=genes))
    vis = ad.AnnData(X=q["Y"].astype(np.float32), obs=pd.DataFrame(index=[f"spot{i}" for i in range(len(q["Y"]))]), var=pd.DataFrame(index=genes))
    ref = ref[:, filter_genes(ref, cell_count_cutoff=5, cell_percentage_cutoff2=0.03, nonz_mean_cutoff=1.12)].copy()
    RegressionModel.setup_anndata(adata=ref, labels_key="label"); rm = RegressionModel(ref); rm.train(max_epochs=250)
    ref = rm.export_posterior(ref, sample_kwargs={"num_samples": 1000, "batch_size": 2500}); fn = ref.uns["mod"]["factor_names"]
    inf = ref.varm["means_per_cluster_mu_fg"][[f"means_per_cluster_mu_fg_{c}" for c in fn]].copy(); inf.columns = fn
    shared = np.intersect1d(vis.var_names, inf.index); vis = vis[:, shared].copy(); inf = inf.loc[shared, :]
    Cell2location.setup_anndata(adata=vis); m = Cell2location(vis, cell_state_df=inf, N_cells_per_location=float(q["ncell"].mean()), detection_alpha=20)
    m.train(max_epochs=int(epochs), batch_size=None, train_size=1); vis = m.export_posterior(vis, sample_kwargs={"num_samples": 1000, "batch_size": m.adata.n_obs})
    W = vis.obsm["means_cell_abundance_w_sf"]
    np.savez_compressed(out, w=W.to_numpy(np.float32), types=np.array([c.replace("meanscell_abundance_w_sf_", "") for c in W.columns]), version=cell2location.__version__)
    print(f"cell2location {cell2location.__version__}: {len(fn)} types, {len(shared)} genes")


if __name__ == "__main__":
    main(*sys.argv[1:6])
