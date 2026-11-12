<div align="left">
  <img src="https://raw.githubusercontent.com/pedrofale/spotr/main/assets/spotr.png" width="200px">
</div>
<p></p>

[![pypi](https://img.shields.io/pypi/v/phylospatial-spotr.svg?style=flat)](https://pypi.org/project/phylospatial-spotr/)
[![CI](https://github.com/pedrofale/spotr/actions/workflows/tests.yml/badge.svg)](https://github.com/pedrofale/spotr/actions/workflows/tests.yml)

Phylospatial optimal transport for single-cell and spatial transcriptomics data integration.

SpOTr maps dissociated cancer cells to the spots of spatial transcriptomics data by using the concordance between the evolutionary tree of the cells and their spatial locations. It combines the spot deconvolution model of cell2location with Fused Gromov–Wasserstein optimal transport, so that closely related cells are mapped to nearby spots and the number of cancer cells mapped to each spot follows the deconvolution. SpOTr recovers the spatial locations of clones that are hard to distinguish by their expression, places sub-lineages that are not given as labels, and positions the clades of the tree in the tissue. It can also use clone labels in place of a tree.

## Installation

```bash
pip install phylospatial-spotr
```

The package is named `phylospatial-spotr` on PyPI, where `spotr` is an unrelated project, and is imported as `spotr`.
The optional extras are `[tree]` (networkx, to read trees as graphs), `[plot]` (scanpy and squidpy, for the notebook) and `[c2l]` (cell2location, for its reference signatures), installed as in `pip install "phylospatial-spotr[plot]"`.
To install the development version, run `pip install git+https://github.com/pedrofale/spotr.git`.

## Quick start

SpOTr follows the interface of scvi-tools models such as DestVI. The AnnData of the dissociated cells contains their raw counts, the cell type of every cell and the clone of each cancer cell. A tree over the cancer cells is optional, and can be given as a Newick string, a networkx graph, a TreeData object, a DataFrame of tree distances or an `.obsp` matrix of tree distances. The AnnData of the spots contains their raw counts and positions.

```python
import spotr

spotr.SpOTr.setup_anndata(
    sc_adata,
    labels_key="cell_type",   # the cell type of every cell
    clone_key="clone",        # the clone of each cancer cell (NaN for the other cells)
    tree=newick,              # optional, with leaves named by sc_adata.obs_names
)
model = spotr.SpOTr(sc_adata, st_adata, spatial_key="spatial")
model.train(alpha=0.2)

model.get_proportions()       # spots x (cell types + clones), also in st_adata.obsm["spotr_proportions"]
model.get_mapping()           # cancer cells x spots
model.get_cell_locations()    # cancer cells x 2, also in sc_adata.obsm["spotr_spatial"]
model.choose_alpha([0, 0.1, 0.2, 0.4, 0.95])   # the elbow of the expression and structural costs
```

SpOTr maps the cells of each clone close together when no tree is given. To map a clone known to be dispersed through the tissue by its expression alone, set its weight in the structural term to zero, as in `model.train(clade_weights={"clone 3": 0.0})`.

## Example

We run SpOTr on a toy tumor (`spotr.datasets.toy_tumor()`) in [`notebooks/spotr_toy_example.ipynb`](https://github.com/pedrofale/spotr/blob/main/notebooks/spotr_toy_example.ipynb). The clones of the toy tumor are hard to distinguish by their expression but occupy separate territories, and we compare SpOTr with and without the tree.

## Reproducibility

The code for the simulation benchmark, the analyses of the PEtracer and HTAN data and every figure of the paper is in [`reproducibility/`](https://github.com/pedrofale/spotr/tree/main/reproducibility).

## Citation

Pedro F. Ferreira, Oskar Kviman, Jens Lagergren. Phylospatial optimal transport for single-cell and spatial transcriptomics data integration (2026).
