"""DestVI on one simulated problem (run in the spotr environment, which carries scvi-tools; GPU), as published and as the paper's runs used it
(experiments/night/250_spotr_core.py fit_destvi): CondSCVI on the dissociated cells (300 epochs), labelled as for cell2location (clones of the
tree given to SpOTr, pooled traced cells, untraced tumour cells, microenvironment types), then DestVI on the spots (1,500 epochs); each label's
proportion per spot.

Usage: python fit_destvi.py <params.json> <problem.npz> <trees.npz> <seed> <out.npz>
"""
import sys, json
import numpy as np


def main(pfile, prob, trees, seed, out):
    import anndata, scvi, pandas as pd
    p = json.load(open(pfile)); q, t = np.load(prob), np.load(trees); rec = p["tree"] == "reconstructed"
    clone = (t["clone_rec_tree"] if rec else t["clone_true_tree"]).astype(int); kind, me, tr = q["kind"], q["me_type"], q["traced"]
    labs = np.array([f"me{e}" for e in me], dtype=object); labs[(kind == 0) & ~tr] = "untraced"; labs[tr] = [f"c{c}" if c >= 0 else "pooled" for c in clone]
    scvi.settings.seed = int(seed)
    sc = anndata.AnnData(q["Xc"].astype(np.float32)); sc.obs["label"] = pd.Categorical(labs.astype(str))
    scvi.model.CondSCVI.setup_anndata(sc, labels_key="label"); msc = scvi.model.CondSCVI(sc, weight_obs=False); msc.train(max_epochs=300, enable_progress_bar=False)
    st = anndata.AnnData(q["Y"].astype(np.float32)); scvi.model.DestVI.setup_anndata(st); mst = scvi.model.DestVI.from_rna_model(st, msc); mst.train(max_epochs=1500, enable_progress_bar=False)
    names = list(msc.adata_manager.get_state_registry("labels").categorical_mapping); own = mst.get_proportions(); own = own[names].values if hasattr(own, "values") else own
    np.savez_compressed(out, w=np.asarray(own, np.float32), types=np.array(names), version=scvi.__version__); print(f"DestVI (scvi-tools {scvi.__version__}): {len(names)} labels")


if __name__ == "__main__":
    main(*sys.argv[1:6])
