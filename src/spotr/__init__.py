"""SpOTr, phylospatial optimal transport for single-cell and spatial transcriptomics data integration.

    import spotr
    spotr.SpOTr.setup_anndata(sc_adata, labels_key="cell_type", clone_key="clone", tree=newick)
    model = spotr.SpOTr(sc_adata, st_adata).train()
    model.get_proportions()
"""
from .model import SpOTr
from . import datasets, plotting, tree

__all__ = ["SpOTr", "datasets", "plotting", "tree"]
