# HT260C1, a colorectal cancer liver metastasis (Figs. 4 and S3)

The data are multiome nuclei and one Visium section of the same tumor block (Mo et al. 2024), from the HTAN data portal.

```bash
python download.py                      # Synapse (free account; ~/.synapseConfig) -> $SPOTR_DATA/htan/raw
Rscript metadata.R $SPOTR_DATA/htan/raw  # the authors' cell types and spot purity, from their Seurat objects (R, no packages needed)
python prepare.py                       # copy number clones, sc.h5ad and st.h5ad                     (benchmark_env, CPU, ~10 min)
python fit.py spotr                     # SpOTr at alpha 0.2 and 0                                     (benchmark_env, GPU)
python fit.py c2l                       # signatures and deconvolution of cell2location                (its own environment, called from here)
python figure.py                        # fig_htan_beyond and fig_htan_purity                          (benchmark_env, CPU)
```

`prepare.py` calls the clones from the CNA scores of the nuclei, which it computes for each chromosome arm from expression and from chromatin
accessibility (ATAC). It calls clones A and B by a likelihood ratio on the chromosome arms that separate two Ward clusters of the nuclei with the
most counts, and splits B into B1 and B2 along the first principal component of its CNA scores. It then calls the remaining nuclei by adding the
evidence of their chromatin accessibility, and splits B1 into the sub-lineages B1a and B1b, which no method is given and whose placement we score
on held-out genes. `fit.py` builds the tree of SpOTr from the distances between the CNA profiles of the cancer nuclei. The Supplementary Methods
of the paper (Clones and tree of HT260C1) describe these steps. `gene_positions.csv.gz` contains the GRCh38 coordinates of the measured genes,
which we use to assign each gene to its chromosome arm.
