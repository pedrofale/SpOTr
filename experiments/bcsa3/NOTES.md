# SpOTr on real BCSA3 data — first run

Status: **SpOTr does not clear the DestVI/scDOT bar on BCSA3/B1**, and a shuffled tree
does just as well as the real one, so the phylogeny is not currently contributing to the
mapping. The pipeline runs end to end; the blocking issues are in the premise and the
evaluation design, not only in the code.

## How to reproduce

```bash
# 1. export the SS3 phylogeny from Numbat's cell-level tree (needs R with ape+igraph)
SP=. /home/pedro/miniconda3/envs/cnv/bin/Rscript export_ss3_tree.R

# 2. build inputs (~50 s)
$BENCH/bin/python prep_bcsa3.py --section B1

# 3. run SpOTr, then score everything on identical footing
XLA_PYTHON_CLIENT_PREALLOCATE=false $BENCH/bin/python run_bcsa3.py --n-iters 300 --lr 0.05
$BENCH/bin/python compare.py B1
```

`$BENCH` = `/home/pedro/miniconda3/envs/benchmark_env` (jax 0.6.2 + GPU, squidpy, POT).
There is no `phylospatial` env on this machine any more; `benchmark_env` has everything.
`XLA_PYTHON_CLIENT_PREALLOCATE=false` is required whenever another job holds the GPU.

## Result (2298 spots, 12 clones, majority baseline 0.280)

| method | argmax acc | ARI | mean clone Pearson |
|---|---|---|---|
| predict majority clone | 0.280 | 0.000 | – |
| DestVI | 0.242 | 0.032 | 0.012 |
| scDOT | 0.262 | 0.008 | 0.017 |
| **NNLS on CNV profiles (control)** | 0.254 | 0.016 | **0.086** |
| SpOTr | 0.046 | 0.036 | 0.018 |

The DestVI/scDOT rows reproduce `notebooks/bcsa3_benchmark_real.ipynb` exactly, so the
scoring path is trustworthy.

## Three findings

### 1. `deconvolution_loss_ls` is inert (real bug, affects the simulations too)

`spot_clone_proportions = gamma.T @ clone_assignments`, but `gamma`'s column sums are
`b_j = 1/n_spots`. The "proportions" therefore come out `n_spots` times too small,
`spot_mean` collapses to ~0, and the loss degenerates to a constant `mean(Y**2)` with
near-zero gradients. Measured on B1: reported loss `1.17305`, `mean(Y**2)` = `1.17379`.

This only bites on the `use_loss_ls=True` path — which is the default in
`run_spotr_perclade_alpha`. With `use_loss_ls=False` the learned `spot_scales` silently
absorb the factor, which is probably why it went unnoticed. **This is a strong candidate
for why gradient-based alpha learning "never worked in general"** and why the work drifted
toward grid search.

Worked around at the call site in `run_bcsa3.py` by passing `clone_assignments * n_spots`.
The proper fix is to normalise the proportions inside the loss.

### 2. Balanced OT pins the output to the reference composition (formulation-level)

`prepare_ot_inputs` hardcodes a uniform source marginal, so each clone's total transported
mass is proportional to its SS3 cell count. On B1 the correlation between SpOTr's mean spot
proportion per clone and the SS3 clone frequency is **1.000** — the output *is* the
reference prior.

The two modalities sample the clones completely differently:

| | dominant clones |
|---|---|
| SS3 reference | clone 2 (171 of 444 cells, 38%) |
| Visium B1 (Numbat) | clones 10 (644), 9 (580), 5 (397), 4 (393); clone 2 has just 23 spots |

So SpOTr assigns clone 2 to 1316 of 2298 spots. Balanced OT *cannot* be right here: the
marginal constraint forces it to spend clone-2 mass that the tissue does not contain.
This is the same modality-composition gap already noted for C1 in
`bcsa3_numbat_results.ipynb`.

Equalising mass per clone (`--balance 40`) removes the prior but does not rescue the
result (acc 0.082, ARI 0.004) — it trades one wrong composition for another.

**The fix is unbalanced / partial OT**: replace the hard source marginal with a KL penalty
so clone abundances become free parameters fit to the spot data. That is what a
deconvolution actually needs, and it is a bounded change.

### 3. The signal is real but weak, and it is not an optimiser problem

- A forward-only sweep over alpha (0 → 1, no gradients) never beats the majority baseline:
  accuracy 0.09–0.12, ARI 0.006–0.016 throughout. So this is not the NaN/gradient issue.
- Sweeping `eps` from 0.2 to 0.002 moves the coupling from 54 to 240 effective cells per
  spot — it is not degenerate — and the metrics stay flat.
- Cross-modality clone profiles do carry signal: mean same-clone correlation **+0.143** vs
  **-0.010** for different clones, with 6/11 clones recovered by argmax. But nested
  subclones are highly collinear, which is also why the NNLS control dumps 94% of its mass
  onto clone 9 (scDOT's exact failure mode).

## Does the tree help? (tested directly)

Three tests, all on BCSA3/B1. Run with `tree_evidence.py` and `tree_ablation.py`.

**Test 1 - do spots whose clones are close in the tree sit close in space?**
Over 300k spot pairs: r = **+0.024**, against a label-permutation null of
-0.000 +/- 0.008 (z = +3.0, p = 0.008). Statistically detectable only because n is
huge; r^2 = 0.06%, so there is essentially nothing for GW to align to.

**Test 2 - do clone territories follow the phylogeny?**
Mantel test on the 9 clones with >=20 spots, tree distance vs centroid distance:
r = **+0.278, p = 0.111**. Not significant.

**Test 3 - shuffled-tree ablation.** Replace the tree with a permuted version (identical
distance distribution, wrong cells) and rerun:

| | argmax acc | ARI | clone r |
|---|---|---|---|
| real Numbat tree | 0.1198 | 0.0100 | 0.0050 |
| shuffled tree (3 seeds) | 0.1052 | 0.0146 | 0.0243 |
| no tree (alpha=0) | 0.100 | 0.024 | 0.037 |

**The real tree does not beat a randomly shuffled one.** It is better on accuracy by
+0.015 (inside the spread of the three shuffles, 0.049-0.168) and *worse* on both soft
metrics. Turning the tree off entirely is no worse than using it.

### Why this does not contradict the readiness analysis

The readiness score (9 concordant axes) asks: *is each subclonal branch spatially
structured in the Visium?* That is a per-branch question and the answer is genuinely yes.

Gromov-Wasserstein needs something strictly stronger: that the tree *metric* and the
spatial *metric* agree, i.e. phylogenetically adjacent clones occupy spatially adjacent
territory. Test 2 is the direct test of that and it fails (p = 0.11). Individual clones
can each form a tidy spatial domain while the arrangement of those domains has nothing to
do with the branching order -- which is what the data show here.

**This is the finding that determines whether the project is viable, and it was never
tested before.** The readiness analysis validated a different premise than the one the
method actually relies on.

## The evaluation target is circular

The benchmark scores against Numbat's per-spot clone calls -- which *are* CNVs called on
the Visium. So "just call CNVs on the Visium" scores 1.000 by construction, and SpOTr can
only ever approach it, never beat it. On this task Numbat alone already produces the
answer, so the mapping adds nothing.

For SpOTr to have a value proposition the task has to be one Visium CNV calling cannot do:
cell-level identity rather than spot-level clone; subclones too rare for Numbat to call
from spots; samples where Visium is too shallow for CNV calling but SS3 is not; or
transferring SS3-only information (SNVs, expression programmes, the finer phylogeny) onto
the tissue. Picking that task is a prerequisite for any further benchmarking.

## All five sections (Tests 1-3)

`tree_evidence_all.py` runs Tests 1-2 on every dataset/section that has a joint
SS3+Visium Numbat run. `dataset_requirements.py` adds the preconditions.

| dataset | clones | usable | coherence | comp TVD | unreferenced | r pairs | Mantel r | p |
|---|---|---|---|---|---|---|---|---|
| BCSA2/TumC2 | 4 | 2 | 0.239 | 0.777 | 0.608 | 0.115 | +0.376 | 0.44 |
| BCSA3/B1 | 12 | 8 | 0.051 | 0.680 | 0.261 | 0.025 | +0.278 | 0.12 |
| BCSA3/D1 | 12 | 8 | 0.096 | 0.549 | 0.194 | 0.015 | -0.054 | 0.77 |
| BCSA3/C1 | 8 | 6 | 0.082 | 0.655 | 0.000 | 0.039 | +0.505 | 0.24 |
| BCSA6/TumA1 | 19 | 10 | 0.029 | 0.762 | 0.526 | 0.031 | -0.010 | 0.94 |

**No section shows significant tree-territory agreement.** Test 1's r is 0.015-0.115 --
"significant" only because n = 300k pairs; r^2 is at most 1.3%.

**The sign is not even reproducible within one tumour.** B1 +0.278, D1 -0.054, C1 +0.505,
all the same sample and the same clone tree; C1 and D1 are the same region. A real
tree-space relationship cannot flip sign between adjacent sections. This is noise.

### Test 3 on D1: the structural term actively hurts

| | argmax acc | ARI | clone r |
|---|---|---|---|
| real Numbat tree | 0.063 | 0.016 | 0.012 |
| shuffled tree | 0.099 | 0.015 | -0.009 |
| **no tree (alpha=0)** | **0.140** | **0.114** | **0.038** |

On D1 switching the structural term off gives **7x the ARI** of using the real tree, and
beats every published baseline (DestVI 0.032, scDOT 0.008). B1 shows the same ordering more
weakly (no-tree 0.024 vs real 0.010). In both sections **no tree > shuffled tree > real
tree**. Adding the GW term degrades the mapping, because it forces alignment to a spatial
geometry the phylogeny does not predict.

Aside worth keeping: plain feature OT on CNV ratios (alpha=0) is the best method in the
whole comparison on D1. That is a real result, but it is not SpOTr's thesis -- no tree.

## Why none of these datasets can demonstrate SpOTr

Four preconditions, all measurable before any modelling (`dataset_requirements.py`):

**1. Clone labels must form spatial territories.** Measured as the excess over chance that
a spot's 12 nearest neighbours share its clone (0 = intermixed, 1 = clean territories).
Observed: **0.03-0.10** for BCSA3/BCSA6, i.e. essentially intermixed. BCSA2 is highest at
0.24 but has only 2 usable clones -- that is the tumour/normal boundary, not subclones.
If clones do not occupy territory, there is no spatial structure for GW to align to.

*This is where the readiness analysis and this result diverge.* Readiness measured Moran's I
on continuous per-event **posteriors**, which stays high for a smooth tumour-wide gradient.
Coherence measures segregation of the discrete **clone calls**. A dataset can pass the first
and fail the second, and all of these do.

**2. Tree distance must predict spatial distance.** Observed: no significant Mantel
correlation anywhere, and sign flips between sections of the same tumour.

**3. The two modalities must sample the same clones in similar proportions.** Composition
TVD is **0.55-0.78**, meaning 55-78% of the probability mass would have to move to reconcile
the SS3 plate with the tissue. Balanced OT cannot absorb that.

**4. Every spatially abundant clone needs a real single-cell reference.** Fraction of spots
whose clone has fewer than 10 SS3 cells: **BCSA2 61%, BCSA6 53%, B1 26%**. Half of some
sections has essentially nothing to map from.

### What a dataset would need instead

| requirement | target | observed here |
|---|---|---|
| spatially segregated subclones | coherence > 0.3 | 0.03-0.10 |
| tree-space metric agreement | Mantel r > 0.5, p < 0.05, reproducible across sections | n.s., sign flips |
| matched modality sampling | composition TVD < 0.2 | 0.55-0.78 |
| reference depth | >= 30-50 cells per clone holding >= 5% of spots | many clones at 9-19 |
| usable clones | >= 6 | 2-10 |
| ground truth | independent of Visium CNV calling | Numbat spot calls (circular) |

Concretely, that points away from "SS3 plate + Visium section from a breast tumour, clones
called by CNV" and towards:

- **A lineage-traced system** (Cas9/barcode recorder, e.g. mouse tumour or organoid) where
  the phylogeny is measured independently of expression and CNV. This fixes the circular
  ground truth *and* gives a real tree rather than a CNV-derived clone graph. The repo
  already has `cassiopeia` in `env.yaml` and `notebooks/spatial_lineage_tracing_data.ipynb`.
- **Spatially progressive expansion**: a lesion where sibling clones occupy adjacent
  territory, so tree distance and spatial distance actually co-vary. Worth checking Mantel r
  on pilot data *before* committing to a full experiment -- it costs one section.
- **Cells and spots from the same tissue block**, ideally dissociated from adjacent sections,
  so the compositions match instead of differing by 55-78%.
- **A task Visium CNV calling cannot do**, so there is something to win: cell-level identity,
  subclones too rare to call from spots, or transferring SS3-only information onto tissue.

## Does a coarser tree rescue it? (granularity.py)

Numbat on Visium resolves only a handful of clones, so the natural fallback is: keep a rich
phylogeny on the scRNA-seq side and only ask the Visium to anchor it coarsely. Tested by
cutting each clone tree into k = 2..K clades and re-measuring both preconditions.

**It does not rescue anything.**

| dataset | best coherence (at k) | best Mantel r (at k) | best p |
|---|---|---|---|
| BCSA2/TumC2 | **0.351** (k=2) | +0.376 (k=4) | 0.454 |
| BCSA3/B1 | 0.107 (k=5) | +0.428 (k=8) | 0.092 |
| BCSA3/D1 | 0.169 (k=4) | -0.407 (k=5) | 0.297 |
| BCSA3/C1 | 0.082 (k=8) | +0.627 (k=6) | 0.240 |
| BCSA6/TumA1 | 0.035 (k=7) | +0.142 (k=8) | 0.479 |

- Coherence never reaches the 0.3 target at any granularity, except BCSA2 at k=2 -- and that
  cut *is* the tumour/normal boundary, not subclonal structure.
- Mantel is **never significant at any k in any dataset** (best p = 0.092).
- B1 stays positive (+0.09 to +0.43) while D1 stays negative (-0.41 to -0.14) across all
  coarse cuts. Same tumour, same tree, opposite signs.

**The diagnostic value is in the BCSA2 row.** If a coarse split were spatially clean, cutting
coarser would reveal it -- which is exactly what happens for BCSA2's tumour/normal boundary
(0.351). It does not happen for BCSA3 or BCSA6. So the coherence measure is working; these
tumours are genuinely clonally **intermixed at every scale**. That is biology, not a
clone-calling resolution limit, and it is why a coarse anchor cannot save the mapping: the
spatial side has no clonal geometry to anchor to at any granularity.

Three breast samples all showing this is a consistent signal. Breast may simply be the wrong
tissue for this method.

## Screening a candidate dataset

The gate needs **Visium plus a clone caller only** -- no scRNA-seq. Run `granularity.py`
(via `tree_evidence_all.RUNS`) and require:

1. coherence > 0.3 at some granularity, driven by subclones rather than tumour/normal;
2. Mantel r significant, and **the same sign in at least two sections** of the same tumour.

Point 2 is the one BCSA fails hardest and the one most likely to be skipped -- a single
section can produce r = +0.5 by chance with 6 clades. Minutes to run per candidate.

Tumour types with documented *regionally segregated* (rather than intermixed) clones are the
ones worth screening: glioblastoma, ovarian, and renal cell carcinoma. HTAN's 48 matched
snRNA-seq + Visium samples span BRCA/CRC/PDAC/RCC/UCEC/CHOL and are the obvious first stop,
since they already supply the unmatched scRNA-seq side.

## What I would do next

1. Fix the proportion normalisation in `deconvolution_loss_ls`, then re-check whether
   gradient learning behaves on the *simulations* — finding 1 may explain the NaN/flat-loss
   history there.
2. **Decide what SpOTr is for**, given the circularity above. Unbalanced OT would fix the
   composition problem and make SpOTr a better deconvolver, but the ablation shows the
   tree is contributing nothing, so it would not make it a better *tree-informed* one.
   That ordering matters: fix the premise before the optimiser.
3. Use the precondition table to spec the next dataset, and screen any candidate with
   `dataset_requirements.py` + `tree_evidence_all.py` before investing in it. Both run off
   Numbat output and spot coordinates alone -- no modelling, minutes not weeks.

Not investigated: D1 and C1 (prep supports both via `--section`), and the per-clade
`use_decomp` path.
