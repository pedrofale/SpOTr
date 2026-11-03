# HT260C1: colorectal cancer liver metastasis (Fig. 4, Fig. S3)

Multiome nuclei and one Visium section of the same tumour block (Mo et al. 2024), from the HTAN data portal.

```bash
python download.py                      # Synapse (free account; ~/.synapseConfig) -> $SPOTR_DATA/htan/raw
Rscript metadata.R $SPOTR_DATA/htan/raw  # the authors' cell types and spot purity, from their Seurat objects (R, no packages needed)
python prepare.py                       # copy-number clones, sc.h5ad and st.h5ad                     (benchmark_env, CPU, ~10 min)
python fit.py spotr                     # SpOTr at alpha 0.2 and 0                                     (benchmark_env, GPU)
python fit.py c2l                       # cell2location's signatures and deconvolution                  (its own environment, called from here)
python figure.py                        # fig_htan_beyond and fig_htan_purity                          (benchmark_env, CPU)
```

`prepare.py` calls the clones from the nuclei's copy number: arm scores from expression, two clones (A, B) by a likelihood ratio on the arms
that separate the deepest nuclei's two Ward clusters, B split into B1 and B2 along its own copy-number axis, the nuclei the expression leaves
uncalled called with their ATAC arm coverage added, and B1's two sub-lineages (B1a, B1b), which no method is given and which score SpOTr's
maps on held-out genes.  Each cancer nucleus keeps its copy-number profile; `fit.py` builds SpOTr's tree from their distances.
`gene_positions.csv.gz` holds the GRCh38 coordinates of the genes the files measure, which place each gene on its chromosome arm.
