# Reproducibility

This folder contains the code for every result and figure of the paper. We run SpOTr through the package (`spotr.SpOTr`) with the settings of the
paper, namely the tree of each data set, $\alpha = 0.2$, $\beta = 1/2$ and reference signatures from the regression model of cell2location.

```
common/        code shared by the analyses (paths, metrics, figure layout, the compared methods in common/baselines.py and cell2location in common/c2l.py)
simulations/   the simulation benchmark as a Snakemake workflow (Figs. 2, S1, S2, S4, S5)
petracer/      PEtracer lineage-traced tumors (Fig. 3)
htan/          HT260C1 from HTAN, a colorectal cancer liver metastasis (Figs. 4, S3)
envs/          the conda environments
```

## Environments

| Environment | File | For |
|---|---|---|
| `benchmark_env` | `envs/benchmark_env.yaml` | SpOTr, DestVI, Tangram, scDOT, POT, scCube, the data loaders and the figures |
| `cassiopeia` | `envs/cassiopeia.yaml` | the simulated lineages |
| `c2l` | `envs/c2l.yaml` | cell2location |
| `snakemake` | `envs/snakemake.yaml` | the simulation workflow |

```bash
for e in benchmark_env cassiopeia c2l snakemake; do conda env create -f envs/$e.yaml; done
git clone https://github.com/namtk/scDOT external/scDOT && git -C external/scDOT checkout da74a8a
```

The scripts import SpOTr from the installed package or, when it is not installed, from `src/` in this repository. cell2location runs in its own
environment, whose Python is given by `SPOTR_C2L_PYTHON` (by default `conda run -n c2l python`). The scripts read scDOT from `external/scDOT` or
from the path in `SPOTR_SCDOT`, and write data, results and figures to `data/`, `results/` and `figures/` in this folder, or to the paths in
`SPOTR_DATA`, `SPOTR_RESULTS` and `SPOTR_FIGURES`.

## Simulations (Figs. 2, S1, S2, S4, S5)

```bash
cd simulations
# first set python: in config.yaml to the Python of benchmark_env and of cassiopeia
snakemake --cores 8 --resources gpu=1
```

Each simulated tumor is a lineage of 12,000 cells simulated with Cassiopeia and placed in the tumor nests of a scCube layout, with every clade in a
territory within that of its parent. We simulate the gene expression of the cells from heritable, positional and cell type programs, and measure
one half of the section with Visium-like spots and the other half as dissociated cells. `config.yaml` contains the base scenario (the base world
in the code) and the three studies of Fig. 2, which vary the strength of the heritable programs, the number of dispersed clones and the number of
clones, each with seeds 1 and 2. We chose the settings of SpOTr on seed 3 of the base scenario (rule `spotr_tuning`). Every method runs on every
seed of every scenario (`scripts/fit_spotr.py`, `scripts/fit_baselines.py`), and the figure scripts compute the per-clone scores from the saved
maps. Simulating one seed of a scenario takes about 20 s, and the 30,000 epochs of cell2location take about 15 minutes on a GPU.

## PEtracer (Fig. 3)

`petracer/prepare.py` downloads the PEtracer tumor data (Koblan et al. 2025) with `pycea.datasets.koblan25(experiment="tumor")` (pycea 0.2.0,
Zenodo record 15750529). It splits the cells of section M3-4 at random into two halves, measures one half with synthetic Visium spots (55 µm
spots on a hexagonal lattice 100 µm apart) and samples 8,000 dissociated cells from the other half. It writes `sc.h5ad`, `st.h5ad` and the truth
to `data/petracer/M3-4/`. `petracer/fit.py` runs SpOTr with the lineage tree and the compared methods (cell2location, DestVI, Tangram and
scDOT), each from its package. `petracer/figure.py` draws Fig. 3 and prints the per-clone PCC, SSIM, RMSE and JS divergence of each method.

```bash
cd petracer
python prepare.py && python fit.py && python figure.py      # benchmark_env, GPU
```

## HTAN (Figs. 4, S3)

The HTAN data are multiome nuclei and a Visium section of one colorectal cancer liver metastasis (Mo et al. 2024), from the HTAN data portal.
`htan/download.py` fetches the twelve files from Synapse (free account), and `Rscript htan/metadata.R` reads the cell types and the tumor purity
of the spots that the authors annotated, from their Seurat objects. `htan/prepare.py` (CPU, about 10 minutes) calls the copy number clones A and
B, the sub-lineages B1 and B2 of B, and the sub-lineages B1a and B1b of B1 from the expression and chromatin accessibility of the nuclei, and
writes `sc.h5ad` and `st.h5ad`. No method is given B1a and B1b, and we score their placement against the CNA signal of the spots on held-out
genes. `htan/fit.py` fits SpOTr with the copy number tree at $\alpha = 0.2$ and $\alpha = 0$, and cell2location, and `htan/figure.py` draws
Figs. 4 and S3. [`htan/README.md`](htan/README.md) lists the steps.
