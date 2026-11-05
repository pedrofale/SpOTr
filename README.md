<div align="left">
  <img src="https://raw.githubusercontent.com/pedrofale/spotr/main/assets/spotr.png" width="200px">
</div>
<p></p>

Tree and space-aware optimal transport for single-cell and spatial transcriptomics data. SpOTr maps dissociated cancer cells to the spots of spatial transcriptomics data, using the concordance between the evolutionary tree of the cells and their spatial distribution. It combines the spot deconvolution model of cell2location with Fused Gromov–Wasserstein optimal transport: the tree draws related cells to nearby spots, and the deconvolution determines how many cancer cells each spot holds. SpOTr recovers the spatial locations of clones that are hard to distinguish by their expression, places sub-lineages that no reference signature describes, and draws the phylogeny of a tumor in the tissue. When no tree is available, it can use clone labels alone.

## Installation

```bash
pip install clonespotr
```

The package is named `clonespotr` on PyPI, where `spotr` is an unrelated project, and is imported as `spotr`.
Optional extras: `[tree]` (networkx, to read trees as graphs), `[plot]` (scanpy and squidpy, for the notebook) and `[c2l]` (cell2location, for its reference signatures), as in `pip install "clonespotr[plot]"`.
The development version installs with `pip install git+https://github.com/pedrofale/spotr.git`.

## Quick start

SpOTr follows the interface of scvi-tools models such as DestVI. The dissociated cells (raw counts) carry each cell's type and each cancer cell's clone; a tree over the cancer cells is optional, as a Newick string, a networkx graph, a TreeData object or an `.obsp` matrix of tree distances. The spots carry raw counts and their positions.

```python
import spotr

spotr.SpOTr.setup_anndata(
    sc_adata,
    labels_key="cell_type",   # every cell's type
    clone_key="clone",        # each cancer cell's clone (NaN for the other cells)
    tree=newick,              # optional: leaves named by sc_adata.obs_names
)
model = spotr.SpOTr(sc_adata, st_adata, spatial_key="spatial")
model.train(alpha=0.2)

model.get_proportions()       # spots x (cell types + clones), also in st_adata.obsm["spotr_proportions"]
model.get_mapping()           # cancer cells x spots
model.get_cell_locations()    # cancer cells x 2, also in sc_adata.obsm["spotr_spatial"]
model.choose_alpha([0, 0.1, 0.2, 0.4, 0.95])   # the elbow of the expression and structural costs
```

Without a tree, SpOTr draws the cells of each clone together. `model.train(clade_weights={"clone 3": 0.0})` maps a clone known to be dispersed through the tissue by its expression alone.

## Example

[`notebooks/spotr_toy_example.ipynb`](https://github.com/pedrofale/spotr/blob/main/notebooks/spotr_toy_example.ipynb) runs SpOTr on a toy tumor (`spotr.datasets.toy_tumor()`) whose clones are hard to distinguish by their expression but each hold their own territory, and compares it with SpOTr without the tree.

## Reproducibility

The simulation benchmark, the analyses of the PEtracer and HTAN data, and every figure of the paper are in [`reproducibility/`](https://github.com/pedrofale/spotr/tree/main/reproducibility).

## Citation

Pedro F. Ferreira, Oskar Kviman, Jens Lagergren. Tree and space-aware optimal transport for single-cell and spatial transcriptomics data integration (2026).
