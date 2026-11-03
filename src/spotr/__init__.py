"""SpOTr: tree and space-aware optimal transport for mapping dissociated cancer cells to spatial transcriptomics spots.

    import spotr
    spotr.SpOTr.setup_anndata(sc_adata, labels_key="cell_type", clone_key="clone", tree=newick)
    model = spotr.SpOTr(sc_adata, st_adata).train()
    model.get_proportions()
"""
from .model import SpOTr
from . import datasets, plotting, tree

__all__ = ["SpOTr", "datasets", "plotting", "tree"]
