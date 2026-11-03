# Reproducibility

The code behind every result and figure of the paper. SpOTr runs through the package (`spotr.SpOTr`), with the paper's settings: the lineage
tree, $\alpha = 0.2$, $\beta = 1/2$, and reference signatures from cell2location's regression model.

```
common/        shared by the analyses: paths, metrics, figure layout, the compared methods (common/baselines.py) and cell2location (common/c2l.py)
simulations/   the simulation benchmark, a Snakemake workflow (Figs. 2, S1, S2, S4, S5)
petracer/      PEtracer lineage-traced tumors (Fig. 3)
htan/          HTAN HT260C1, a colorectal liver metastasis (Figs. 4, S3)
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

The scripts import SpOTr from the installed package or, when it is not installed, from this repository's `src/`. cell2location runs in its
own environment: set `SPOTR_C2L_PYTHON` to that environment's Python (by default `conda run -n c2l python`). scDOT is read from
`external/scDOT`, or from `SPOTR_SCDOT`. Data, results and figures go to `data/`, `results/` and `figures/` here, or to `SPOTR_DATA`,
`SPOTR_RESULTS` and `SPOTR_FIGURES`.

## Simulations (Figs. 2, S1, S2, S4, S5)

```bash
cd simulations
# set python: in config.yaml to the Python of benchmark_env and of cassiopeia
snakemake --cores 8 --resources gpu=1
```

Each simulated tumor is a lineage of 12,000 cells simulated by Cassiopeia, placed in the tumor nests of a scCube layout with every clade in a
territory within its parent's, given gene expression from heritable, positional and cell type programs, and measured with Visium-like spots on
one half and as dissociated cells on the other. `config.yaml` holds the base world and the three studies of Fig. 2 (the strength of the
heritable programs, the number of dispersed clones, the number of clones), each with seeds 1 and 2; seed 3 of the base world chose SpOTr's
settings (rule `spotr_tuning`). Every method runs on every world-seed (`scripts/fit_spotr.py`, `scripts/fit_baselines.py`), and the figure
scripts compute the per-clone scores from the saved maps. One world-seed takes about 20 s to simulate; cell2location's 30,000 epochs take
about 15 minutes on a GPU.

## PEtracer (Fig. 3)

`petracer/prepare.py` downloads the PEtracer tumor data (Koblan et al. 2025) with `pycea.datasets.koblan25(experiment="tumor")` (pycea 0.2.0,
Zenodo record 15750529). It splits the cells of section M3-4 at random into two halves, measures one half with synthetic Visium spots (55 µm
spots on a hexagonal lattice 100 µm apart) and draws 8,000 dissociated cells from the other, writing `sc.h5ad`, `st.h5ad` and the truth to
`data/petracer/M3-4/`. `petracer/fit.py` runs SpOTr with the lineage tree and the compared methods, cell2location, DestVI, Tangram and scDOT,
each from its package. `petracer/figure.py` draws Fig. 3 and prints each method's per-clone PCC, SSIM, RMSE and JS divergence.

```bash
cd petracer
python prepare.py && python fit.py && python figure.py      # benchmark_env, GPU
```

## HTAN (Figs. 4, S3)

Multiome nuclei and a Visium section of one colorectal liver metastasis (Mo et al. 2024) from the HTAN data portal: `htan/download.py`
fetches the twelve files from Synapse (free account), and `Rscript htan/metadata.R` reads the authors' cell types and spot purity from their
Seurat objects. `htan/prepare.py` (CPU, about 10 minutes) calls the copy-number clones from the nuclei's expression and ATAC: A and B, B's B1
and B2, and B1's two sub-lineages, which no method is given and which score the maps on held-out genes. It writes `sc.h5ad` and `st.h5ad`.
`htan/fit.py` fits SpOTr with the copy-number tree at $\alpha$ 0.2 and 0, and cell2location; `htan/figure.py` draws Figs. 4 and S3. The
steps are in [`htan/README.md`](htan/README.md).
