# SpOTr simulation benchmark on Cassiopeia lineages

A Snakemake workflow that simulates lineage-traced tumours with [Cassiopeia](https://github.com/YosefLab/Cassiopeia), adds a deliberately simple
expression layer, measures them as spots and as dissociated cells, and benchmarks SpOTr (no tree, clone labels, clone tree, lineage tree)
against cell2location, DestVI, Tangram and exact optimal transport, with ablations of the parts of SpOTr.

## Design

**Lineage and space (Cassiopeia).**  The tree is a birth-death process with weak selection (`BirthDeathFitnessSimulator`) grown to
`n_cells` leaves; a lineage that dies out is redrawn.  Cas9 barcodes (`Cas9LineageTracingDataSimulator`, 13 cassettes of 3 sites) are
edited at a rate that leaves half of the sites edited at the end.  Positions come from `ClonalSpatialDataSimulator` (each clade fills a
contiguous territory; lineage and space agree) or `BrownianSpatialDataSimulator` (cells move from their parent by Gaussian steps), rescaled
so cells lie 12 um apart.  The clones are the clades of the tree's first 64 lineages holding at least 2% of the cells; the rest are pooled.
To make lineage and space disagree, the cells of a share `migrate` of the clones swap places with random cells of other clones.

**The expression layer (ours, kept minimal).**  A cell's expected counts are `L * softmax(eta)`, with Poisson noise, where

    eta = baseline + cell-type shift + sigma_tree * (heritable programs) + sigma_space * (positional programs)

The baseline and the four cell-type shifts (tumour and three microenvironment types) are standard normal per gene.  The heritable programs
(5) perform Brownian motion along the true tree, scaled to unit variance at the leaves; the positional programs (5) are smooth random
functions of position (random Fourier features at a quarter of the section's diameter).  Both reach the genes through fixed random
loadings.  The tissue is scCube's tumour-microenvironment layout (Qian et al. 2024, Nat Commun 15:5021; `scripts/layout.py`): 9 oval
tumour nests (90% tumour cells, the rest infiltrating T cells and endothelial cells) in a fibroblast stroma with 3 T-cell aggregates and
15% endothelial cells; Cassiopeia's tumour cells are moved onto its tumour sites by exact optimal transport, which keeps their arrangement.
(Until 2026-10 the microenvironment cells had random positions over the tumour; `results_v1_randomME/` is that run.)  `sigma_tree` is how much the subclones differ
in expression; `sigma_space` how much cell states vary across the section.

**Measurement.**  The cells are split at random in two halves, as in PEtracer's half design.  The section's half is measured as by Visium
(`spots: visium`, since 2026-11-02): spots of 55 um on a hexagonal lattice 100 um apart, each holding the cells whose centres lie in it (about
three, since cells are 10 um apart and only half of them are in the section), the cells between spots not measured.  `results/` is the run
with `spots: square`, square bins of about ten cells that tile the section.  Of the other half, 50% is sequenced as dissociated cells, and the barcodes of 70% of the sequenced tumour cells are
read.  SpOTr receives the true tree of the traced dissociated cells, or the tree Cassiopeia's `VanillaGreedySolver` reconstructs from their
barcodes (`tree: reconstructed`).

**Scores.**  The per-spot subclone error (total variation between the true and the inferred clone composition of each spot holding tumour
cells); the same at the resolution of the sub-clades of each clone; SpOTr's learned against true spreads; and the lineage-resolution
readout's precision and recall against the cells' true positions.

## Studies (`config.yaml`)

| study | varies | compares |
|---|---|---|
| heritable | `sigma_tree` 0, 0.5, 1, 2, 4 | SpOTr's arms, cell2location, DestVI, Tangram, exact OT |
| positional | `sigma_space` 0.5, 1, 2, 4 (with `sigma_tree` 0) | the same |
| migration | `migrate` 0, 0.25, 0.5 | SpOTr's arms, DestVI, Tangram, exact OT |
| spatial_model | clonal, brownian | the same |
| tree_source | true, reconstructed tree | SpOTr's arms |

Every world also runs the ablation arm `clones_true_spreads` (clone labels with the true spreads instead of learned ones).  Two seeds each.

## Environments

Each engine runs in its own conda environment (paths in `config.yaml: python`); `PYTHONNOUSERSITE=1` keeps `~/.local` packages out.

- `spotr`: `benchmark_env` (SpOTr's JAX model from `src/spotr/c2l_gw.py`, scvi-tools 1.3.3 for DestVI, Tangram, POT, umap-learn)
- `cassiopeia`: Cassiopeia 2.1.0 from GitHub commit `1ee5959` with the spatial extras
  (`pip install "cassiopeia-lineage[spatial] @ git+https://github.com/YosefLab/Cassiopeia@1ee5959eb9d3f8d4d26e2af5678234493bf54d6d"`; the
  PyPI release 1.0.4 does not build)
- `c2l`: cell2location 0.1.5 from PyPI with pyro-ppl 1.9.1 (1.9.2 breaks its import)

## Running

    snakemake --cores 8 --resources gpu=1                                            # everything
    snakemake --cores 8 --resources gpu=1 --configfile config.yaml config_smoke.yaml  # a small end-to-end check

`gpu` throttles the GPU rules (SpOTr, cell2location, DestVI, Tangram); cell2location's 30,000 epochs take about 15 minutes per world on one GPU.
Outputs: `results/worlds/<world>/seed<s>/` (simulation, problem, trees, maps, scores, readout), `results/summary.csv` (long table), and
`results/figures/` (`fig_benchmark`, `fig_ablation`, and `fig_example`, the standard example figure on one world).

## Expectations, fixed before running

- E206  heritable and positional studies, every world with `sigma_tree` or `sigma_space` above 0: SpOTr with any tree below SpOTr without the
  tree, and SpOTr without the tree below cell2location, DestVI, Tangram and exact OT (mean over seeds); the clone tree at or below clone labels in
  at least 80% of world-seeds.
- E207  heritable study, `sigma_tree` >= 0.5: the lineage tree below the clone tree at sub-clade level 3 in at least 80% of world-seeds.
- E208  migration: the tree's gain over no tree falls as `migrate` grows, and no tree arm is worse than no tree by more than 0.02.
- E209  tree_source: the reconstructed tree within 0.03 of the true tree (lineage arm).  (Its second half, on barcode saturation, was dropped with that study on 2026-10-31: SpOTr is not specific to lineage-tracing technologies.)
- E210  the lineage-resolution readout's precision at least 0.9, pooled over all worlds.
- read  how much fixing the spreads at the truth changes the error (the cost of learning them).
